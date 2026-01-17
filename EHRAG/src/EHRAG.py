from src.embedding_store import EmbeddingStore
from src.utils import min_max_normalize
import os
import json
from collections import defaultdict
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.ner import SpacyNER
import igraph as ig
import re
import logging
import time
from sklearn.cluster import Birch

logger = logging.getLogger(__name__)

def topk_numpy(arr, k, dim=0):
    idx = np.argpartition(-arr, kth=k, axis=dim)
    idx = idx.take(indices=range(k), axis=dim)
    val = np.take_along_axis(arr, indices=idx, axis=dim)
    sorted_idx = np.argsort(-val, axis=dim)
    idx = np.take_along_axis(idx, indices=sorted_idx, axis=dim)
    return idx

class EHRAG:
    def __init__(self, global_config):
        self.config = global_config
        logger.info(f"Initializing EHRAG with config: {self.config}")
        self.dataset_name = global_config.dataset_name
        self.non_seed = 0
        self.load_embedding_store()
        self.llm_model = self.config.llm_model
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.graph = ig.Graph(directed=False)

    def load_embedding_store(self):
        self.passage_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "passage_embedding.parquet"), batch_size=self.config.batch_size, namespace="passage")
        self.entity_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "entity_embedding.parquet"), batch_size=self.config.batch_size, namespace="entity")
        self.sentence_embedding_store = EmbeddingStore(self.config.embedding_model, db_filename=os.path.join(self.config.working_dir, self.dataset_name, "sentence_embedding.parquet"), batch_size=self.config.batch_size, namespace="sentence")

    def load_existing_data(self, passage_hash_ids):
        self.ner_results_path = os.path.join(self.config.working_dir, self.dataset_name, "ner_results.json")
        if os.path.exists(self.ner_results_path):
            existing_ner_reuslts = json.load(open(self.ner_results_path))
            existing_passage_hash_id_to_entities = existing_ner_reuslts["passage_hash_id_to_entities"]
            existing_sentence_to_entities = existing_ner_reuslts["sentence_to_entities"]
            existing_passage_hash_ids = set(existing_passage_hash_id_to_entities.keys())
            new_passage_hash_ids = set(passage_hash_ids) - existing_passage_hash_ids
            return existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids
        else:
            return {}, {}, passage_hash_ids

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)
        system_prompt = f"""As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations. Please do not give additional prefixes such as repeat the question beyond the clear answer, and try to be consistent with the language that the answer may need. Pay attention to the case of letters when answering. If the text paragraph does not provide enough information, try to answer using your own knowledge."""
        
        all_messages = []
        for retrieval_result in retrieval_results:
            question = retrieval_result["question"]
            sorted_passage = retrieval_result["sorted_passage"]
            prompt_user = """"""
            for passage in sorted_passage:
                prompt_user += f"{passage}\n"
            prompt_user += f"Question: {question}\n Thought: "
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_user}
            ]
            all_messages.append(messages)
            
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            all_qa_results = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA Reading (Parallel)"
            ))

        for qa_result, question_info in zip(all_qa_results, retrieval_results):
            try:
                pred_ans = qa_result.split('Answer:')[1].strip()
            except:
                pred_ans = qa_result
            question_info["pred_answer"] = pred_ans
        return retrieval_results
        
    def retrieve(self, questions):
        self.entity_hash_ids = list(self.entity_embedding_store.hash_id_to_text.keys())
        self.entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        self.passage_hash_ids = list(self.passage_embedding_store.hash_id_to_text.keys())
        self.passage_embeddings = np.array(self.passage_embedding_store.embeddings)
        self.sentence_hash_ids = list(self.sentence_embedding_store.hash_id_to_text.keys())
        self.sentence_embeddings = np.array(self.sentence_embedding_store.embeddings)
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}
        self.vertex_idx_to_node_name = {v.index: v["name"] for v in self.graph.vs if "name" in v.attributes()}

        embed_time = 0
        search_time = 0

        retrieval_results = []
        for question_info in tqdm(questions, desc="Retrieving"):
            t1 = time.time()
            question = question_info["question"]
            question_embedding = self.config.embedding_model.encode(question, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size)
            seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores = self.get_seed_entities(question)
            t2 = time.time()
            
            if len(seed_entities) != 0:
                sorted_passage_hash_ids, sorted_passage_scores = self.graph_search_with_seed_entities(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
                final_passage_hash_ids = sorted_passage_hash_ids[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.hash_id_to_text[passage_hash_id] for passage_hash_id in final_passage_hash_ids]
            else:
                sorted_passage_indices, sorted_passage_scores = self.dense_passage_retrieval(question_embedding)
                final_passage_indices = sorted_passage_indices[:self.config.retrieval_top_k]
                final_passage_scores = sorted_passage_scores[:self.config.retrieval_top_k]
                final_passages = [self.passage_embedding_store.texts[idx] for idx in final_passage_indices]
            
            t3 = time.time()
            embed_time += t2 - t1
            search_time += t3 - t2
            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_passage_scores,
                "gold_answer": question_info["answer"]
            }
            retrieval_results.append(result)
        
        print("Embedding time is", str(embed_time))
        print("Search time is", str(search_time))
        return retrieval_results

    def graph_search_with_seed_entities(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        entity_weights, actived_entities = self.calculate_entity_scores(question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores)
        passage_weights = self.calculate_passage_scores(question_embedding, actived_entities)
        node_weights = entity_weights + passage_weights
        ppr_sorted_passage_indices, ppr_sorted_passage_scores = self.run_ppr(node_weights)
        return ppr_sorted_passage_indices, ppr_sorted_passage_scores

    def run_ppr(self, node_weights):        
        reset_prob = np.where(np.isnan(node_weights) | (node_weights < 0), 0, node_weights)
        
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=self.config.damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
        
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_indices])
        sorted_indices_in_doc_scores = np.argsort(doc_scores)[::-1]
        sorted_passage_scores = doc_scores[sorted_indices_in_doc_scores]
        
        sorted_passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[i]] 
            for i in sorted_indices_in_doc_scores
        ]
        
        return sorted_passage_hash_ids, sorted_passage_scores.tolist()

    def calculate_entity_scores(self, question_embedding, seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
        actived_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        
        # [STEP 1] Dense Retrieval 
        for seed_entity_idx, seed_entity, seed_entity_hash_id, seed_entity_score in zip(seed_entity_indices, seed_entities, seed_entity_hash_ids, seed_entity_scores):
            actived_entities[seed_entity_hash_id] = (seed_entity_idx, seed_entity_score, 1)
            seed_entity_node_idx = self.node_name_to_vertex_idx[seed_entity_hash_id]
            entity_weights[seed_entity_node_idx] = seed_entity_score    

        # [STEP 2] (Semantic Expansion)
        if hasattr(self, 'entity_hash_id_to_cluster_id') and hasattr(self, 'cluster_id_to_entity_hash_ids'):
            
            
            SEMANTIC_DECAY = self.config.semantic_decay
            MAX_SEMANTIC_SEEDS = self.config.max_semantic_seeds
            
            SEMANTIC_PRUNING_THRESHOLD = self.config.iteration_threshold * 1.0 

            semantic_candidates = defaultdict(float)

            for seed_hash_id in seed_entity_hash_ids:
                if seed_hash_id not in actived_entities: continue
                
                cluster_id = self.entity_hash_id_to_cluster_id.get(seed_hash_id)
                if cluster_id is not None:
                    siblings = self.cluster_id_to_entity_hash_ids.get(cluster_id, [])
                    original_score = actived_entities[seed_hash_id][1] 

                    for sibling_id in siblings:
                        if sibling_id in actived_entities: continue
                        
                        potential_score = original_score * SEMANTIC_DECAY
                        if potential_score < SEMANTIC_PRUNING_THRESHOLD: continue
                        
                        if potential_score > semantic_candidates[sibling_id]:
                            semantic_candidates[sibling_id] = potential_score

            if semantic_candidates:
                sorted_semantics = sorted(
                    semantic_candidates.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:MAX_SEMANTIC_SEEDS]

                for sem_hash_id, sem_score in sorted_semantics:
                    node_idx = self.node_name_to_vertex_idx.get(sem_hash_id)
                    if node_idx is not None:
                        actived_entities[sem_hash_id] = (node_idx, sem_score, 1)
                        entity_weights[node_idx] = sem_score

        # [STEP 3] (Structural Propagation)
        used_sentence_hash_ids = set()
        current_entities = actived_entities.copy()
        
        iteration = 1
        while len(current_entities) > 0 and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (entity_id, entity_score, tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                
                sentence_hash_ids = [sid for sid in list(self.entity_hash_id_to_sentence_hash_ids.get(entity_hash_id, [])) if sid not in used_sentence_hash_ids]
                
                if not sentence_hash_ids:
                    continue
                
                sentence_indices = [self.sentence_embedding_store.hash_id_to_idx[sid] for sid in sentence_hash_ids]
                if not sentence_indices: continue 

                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_emb = question_embedding.reshape(-1, 1) if len(question_embedding.shape) == 1 else question_embedding
                
                sentence_similarities = np.dot(sentence_embeddings, question_emb).flatten()
                
                top_sentence_indices_local = np.argsort(sentence_similarities)[::-1][:self.config.top_k_sentence]
                
                for local_idx in top_sentence_indices_local:
                    top_sentence_index = local_idx 
                    top_sentence_hash_id = sentence_hash_ids[top_sentence_index]
                    top_sentence_score = sentence_similarities[top_sentence_index]
                    
                    used_sentence_hash_ids.add(top_sentence_hash_id)
                    
                    entity_hash_ids_in_sentence = self.sentence_hash_id_to_entity_hash_ids.get(top_sentence_hash_id, [])
                    for next_entity_hash_id in entity_hash_ids_in_sentence:
                        next_entity_score = entity_score * top_sentence_score
                        
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        
                        next_enitity_node_idx = self.node_name_to_vertex_idx.get(next_entity_hash_id)
                        if next_enitity_node_idx is None: continue

                        entity_weights[next_enitity_node_idx] += next_entity_score
                        
                        if next_entity_hash_id not in new_entities or next_entity_score > new_entities[next_entity_hash_id][1]:
                             new_entities[next_entity_hash_id] = (next_enitity_node_idx, next_entity_score, iteration+1)
            
            actived_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
            
        return entity_weights, actived_entities

    def calculate_passage_scores(self, question_embedding, actived_entities):
        passage_weights = np.zeros(len(self.graph.vs["name"]))
        
        dpr_passage_indices, dpr_passage_scores = self.dense_passage_retrieval(question_embedding)
        dpr_passage_scores = min_max_normalize(dpr_passage_scores)

        # [EHyperRAG Step 1] Calculate Topic Importance
        topic_importance = defaultdict(float)
        if hasattr(self, 'entity_hash_id_to_cluster_id'):
            for ent_hash, (_, entity_score, _) in actived_entities.items():
                cluster_id = self.entity_hash_id_to_cluster_id.get(ent_hash)
                if cluster_id is not None:
                    topic_importance[cluster_id] += entity_score
        
        # 
        TOPIC_RATIO = self.config.topic_ratio

        # [Step 2] 
        for i, dpr_passage_index in enumerate(dpr_passage_indices):
            total_entity_bonus = 0.0
            total_topic_bonus = 0.0
            
            passage_hash_id = self.passage_embedding_store.hash_ids[dpr_passage_index]
            dpr_passage_score = dpr_passage_scores[i]
            passage_text_lower = self.passage_embedding_store.hash_id_to_text[passage_hash_id].lower()
            
            seen_clusters_in_passage = set()

            for entity_hash_id, (entity_id, entity_score, tier) in actived_entities.items():
                entity_lower = self.entity_embedding_store.hash_id_to_text[entity_hash_id].lower()
                entity_occurrences = passage_text_lower.count(entity_lower)
                
                if entity_occurrences > 0:
                    # --- A. Entity Bonus ---
                    denom = tier if tier >= 1 else 1
                    entity_bonus = entity_score * math.log(1 + entity_occurrences) / denom
                    total_entity_bonus += entity_bonus

                    # --- B. Topic Bonus ---
                    if hasattr(self, 'entity_hash_id_to_cluster_id'):
                        c_id = self.entity_hash_id_to_cluster_id.get(entity_hash_id)
                        if c_id is not None and c_id in topic_importance:
                            if c_id not in seen_clusters_in_passage:
                                total_topic_bonus += topic_importance[c_id]
                                seen_clusters_in_passage.add(c_id)

            # [Step 3] 
            base_score = self.config.passage_ratio * dpr_passage_score + math.log(1 + total_entity_bonus)
            topic_score = TOPIC_RATIO * math.log(1 + total_topic_bonus)
            final_score = base_score + topic_score
            
            passage_node_idx = self.node_name_to_vertex_idx[passage_hash_id]
            passage_weights[passage_node_idx] = final_score * self.config.passage_node_weight

        return passage_weights

    def dense_passage_retrieval(self, question_embedding):
        question_emb = question_embedding.reshape(1, -1)
        question_passage_similarities = np.dot(self.passage_embeddings, question_emb.T).flatten()
        sorted_passage_indices = np.argsort(question_passage_similarities)[::-1]
        sorted_passage_scores = question_passage_similarities[sorted_passage_indices].tolist()
        return sorted_passage_indices, sorted_passage_scores
    
    def get_seed_entities(self, question):
        S = self.config.seed_entities_count

        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return [], [], [], []
        question_entity_embeddings = self.config.embedding_model.encode(question_entities, normalize_embeddings=True, show_progress_bar=False, batch_size=self.config.batch_size)

        similarities = np.dot(self.entity_embeddings, question_entity_embeddings.T)
        seed_entity_indices = []
        seed_entity_texts = []
        seed_entity_hash_ids = []
        seed_entity_scores = []       
        for query_entity_idx in range(len(question_entities)):
            entity_scores = similarities[:, query_entity_idx]
            best_entity_idxs = topk_numpy(entity_scores, S)
            for best_entity_idx in best_entity_idxs:
                best_entity_score = entity_scores[best_entity_idx]
                best_entity_hash_id = self.entity_hash_ids[best_entity_idx]
                best_entity_text = self.entity_embedding_store.hash_id_to_text[best_entity_hash_id]
                seed_entity_indices.append(best_entity_idx)
                seed_entity_texts.append(best_entity_text)
                seed_entity_hash_ids.append(best_entity_hash_id)
                seed_entity_scores.append(best_entity_score)

        return seed_entity_indices, seed_entity_texts, seed_entity_hash_ids, seed_entity_scores

    def index(self, passages):
        self.node_to_node_stats = defaultdict(dict)
        self.entity_to_sentence_stats = defaultdict(dict)
        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()
        existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_ids = self.load_existing_data(hash_id_to_passage.keys())
        if len(new_passage_hash_ids) > 0:
            new_hash_id_to_passage = {k : hash_id_to_passage[k] for k in new_passage_hash_ids}
            new_passage_hash_id_to_entities, new_sentence_to_entities = self.spacy_ner.batch_ner(new_hash_id_to_passage, self.config.max_workers)
            self.merge_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities)
        self.save_ner_results(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        entity_nodes, sentence_nodes, passage_hash_id_to_entities, self.entity_to_sentence, self.sentence_to_entity = self.extract_nodes_and_edges(existing_passage_hash_id_to_entities, existing_sentence_to_entities)
        self.sentence_embedding_store.insert_text(list(sentence_nodes))
        self.entity_embedding_store.insert_text(list(entity_nodes))

        self.add_entity_cluster_nodes_and_edges()

        self.entity_hash_id_to_sentence_hash_ids = {}
        for entity, sentence in self.entity_to_sentence.items():
            entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
            self.entity_hash_id_to_sentence_hash_ids[entity_hash_id] = [self.sentence_embedding_store.text_to_hash_id[s] for s in sentence]
        self.sentence_hash_id_to_entity_hash_ids = {}
        for sentence, entities in self.sentence_to_entity.items():
            sentence_hash_id = self.sentence_embedding_store.text_to_hash_id[sentence]
            self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id] = [self.entity_embedding_store.text_to_hash_id[e] for e in entities]
        self.add_entity_to_passage_edges(passage_hash_id_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()
        output_graphml_path = os.path.join(self.config.working_dir, self.dataset_name, "EHRAG.graphml")
        os.makedirs(os.path.dirname(output_graphml_path), exist_ok=True)   
        self.graph.write_graphml(output_graphml_path)

    def add_entity_cluster_nodes_and_edges(self):
        logger.info("Starting adaptive entity clustering using BIRCH...")
        
        entity_embeddings = np.array(self.entity_embedding_store.embeddings)
        entity_hash_ids = self.entity_embedding_store.hash_ids
        
        if len(entity_hash_ids) == 0:
            logger.warning("No entities to cluster.")
            return

        cluster_threshold = self.config.cluster_threshold
        
        try:
            birch_model = Birch(n_clusters=None, threshold=cluster_threshold, branching_factor=50)
            labels = birch_model.fit_predict(entity_embeddings)
            centroids = birch_model.subcluster_centers_
        except Exception as e:
            logger.error(f"BIRCH clustering failed: {e}")
            return

        unique_labels = np.unique(labels)
        logger.info(f"BIRCH adaptive clustering found {len(unique_labels)} concepts from {len(entity_hash_ids)} entities.")

        self.entity_hash_id_to_cluster_id = {}
        self.cluster_id_to_entity_hash_ids = defaultdict(list)
        self.cluster_nodes = set()
        
        concept_node_prefix = "CONCEPT_"
        
        top_d = self.config.top_k_entity_per_cluster
        
        cluster_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_indices[label].append(idx)

        for label in tqdm(cluster_indices.keys(), desc="Constructing Semantic Hyperedges"):
            concept_node_hash_id = f"{concept_node_prefix}{label}"
            self.cluster_nodes.add(concept_node_hash_id)
            
            indices = cluster_indices[label]
            centroid = centroids[label]
            cluster_embeddings = entity_embeddings[indices]
            
            dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            entity_dist_pairs = []
            for i, dist in enumerate(dists):
                original_idx = indices[i]
                entity_dist_pairs.append((dist, original_idx))
            
            entity_dist_pairs.sort(key=lambda x: x[0])
            top_candidates = entity_dist_pairs[:top_d]
            
            for dist, original_idx in top_candidates:
                entity_hash_id = entity_hash_ids[original_idx]
                
                self.entity_hash_id_to_cluster_id[entity_hash_id] = concept_node_hash_id
                self.cluster_id_to_entity_hash_ids[concept_node_hash_id].append(entity_hash_id)
                
                weight = 1.0 / (dist + 1e-6)
                
                self.node_to_node_stats[entity_hash_id][concept_node_hash_id] = weight
                self.node_to_node_stats[concept_node_hash_id][entity_hash_id] = weight

        logger.info(f"Added {len(self.cluster_nodes)} concept nodes via Adaptive BIRCH Clustering.")

    def add_adjacent_passage_edges(self):
        passage_id_to_text = self.passage_embedding_store.get_hash_id_to_text()
        index_pattern = re.compile(r'^(\d+):')
        indexed_items = [
            (int(match.group(1)), node_key)
            for node_key, text in passage_id_to_text.items()
            if (match := index_pattern.match(text.strip()))
        ]
        indexed_items.sort(key=lambda x: x[0])
        for i in range(len(indexed_items) - 1):
            current_node = indexed_items[i][1]
            next_node = indexed_items[i + 1][1]
            self.node_to_node_stats[current_node][next_node] = 1.0

    def augment_graph(self):
        self.add_nodes()
        self.add_edges()

    def add_nodes(self):
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()} 
        entity_hash_id_to_text = self.entity_embedding_store.get_hash_id_to_text()
        passage_hash_id_to_text = self.passage_embedding_store.get_hash_id_to_text()

        concept_node_hash_id_to_text = {}
        if hasattr(self, 'cluster_nodes'):
            for concept_hash_id in self.cluster_nodes:
                concept_node_hash_id_to_text[concept_hash_id] = concept_hash_id
        
        all_hash_id_to_text = {
            **entity_hash_id_to_text, 
            **passage_hash_id_to_text,
            **concept_node_hash_id_to_text
        }
        
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        
        for hash_id, text in all_hash_id_to_text.items():
            if hash_id not in existing_nodes:
                self.graph.add_vertex(name=hash_id, content=text)
        
        self.node_name_to_vertex_idx = {v["name"]: v.index for v in self.graph.vs if "name" in v.attributes()}   
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id] 
            for passage_id in passage_hash_ids 
            if passage_id in self.node_name_to_vertex_idx
        ]

    def add_edges(self):
        edges = []
        weights = []
        
        for node_hash_id, node_to_node_stats in self.node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue
                edges.append((node_hash_id, neighbor_hash_id))
                weights.append(weight)
        self.graph.add_edges(edges)
        self.graph.es['weight'] = weights

    def add_entity_to_passage_edges(self, passage_hash_id_to_entities):
        passage_to_entity_count ={} 
        passage_to_all_score = defaultdict(int)
        for passage_hash_id, entities in passage_hash_id_to_entities.items():
            passage = self.passage_embedding_store.hash_id_to_text[passage_hash_id]
            for entity in entities:
                entity_hash_id = self.entity_embedding_store.text_to_hash_id[entity]
                count = passage.count(entity)
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            score = count / passage_to_all_score[passage_hash_id]
            self.node_to_node_stats[passage_hash_id][entity_hash_id] = score

    def extract_nodes_and_edges(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        entity_nodes = set()
        sentence_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)
        entity_to_sentence= defaultdict(set)
        sentence_to_entity = defaultdict(set)
        for passage_hash_id, entities in existing_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)
        for sentence, entities in existing_sentence_to_entities.items():
            sentence_nodes.add(sentence)
            for entity in entities:
                entity_to_sentence[entity].add(sentence)
                sentence_to_entity[sentence].add(entity)
        return entity_nodes, sentence_nodes, passage_hash_id_to_entities, entity_to_sentence, sentence_to_entity

    def merge_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities, new_passage_hash_id_to_entities, new_sentence_to_entities):
        existing_passage_hash_id_to_entities.update(new_passage_hash_id_to_entities)
        existing_sentence_to_entities.update(new_sentence_to_entities)
        return existing_passage_hash_id_to_entities, existing_sentence_to_entities

    def save_ner_results(self, existing_passage_hash_id_to_entities, existing_sentence_to_entities):
        with open(self.ner_results_path, "w") as f:
            json.dump({"passage_hash_id_to_entities": existing_passage_hash_id_to_entities, "sentence_to_entities": existing_sentence_to_entities}, f)