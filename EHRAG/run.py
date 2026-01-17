import argparse
import json
from sentence_transformers import SentenceTransformer
from src.config import EHRAGConfig
from src.EHRAG import EHRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description="EHGRAG Runner")
    
    parser.add_argument("--dataset_name", type=str, default="hotpotqa", help="The dataset to use")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="model/all-mpnet-base-v2", help="The path of embedding model")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="The LLM model to use")
    parser.add_argument("--max_workers", type=int, default=16, help="Max number of parallel workers")

    parser.add_argument("--max_iterations", type=int, default=5, help="Max iterations for graph propagation")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="Threshold to continue propagation")
    parser.add_argument("--passage_ratio", type=float, default=2.0, help="Weight ratio for passage scores")
    parser.add_argument("--top_k_sentence", type=int, default=4, help="Top K sentences to expand from entity")
    parser.add_argument("--retrieval_top_k", type=int, default=5, help="Final number of passages to retrieve") 
    parser.add_argument("--damping", type=float, default=0.5, help="PPR damping factor") 
    parser.add_argument("--passage_node_weight", type=float, default=1.0, help="Weight multiplier for passage nodes") 

    parser.add_argument("--seed_entities_count", type=int, default=1, help="Number of seed entities (S) to select from question")
    parser.add_argument("--semantic_decay", type=float, default=0.1, help="Decay factor for semantic expansion")
    parser.add_argument("--max_semantic_seeds", type=int, default=2, help="Max number of semantic seeds to expand")
    parser.add_argument("--topic_ratio", type=float, default=0.5, help="Weight ratio for topic/cluster importance")
    
    parser.add_argument("--cluster_threshold", type=float, default=0.9, help="Threshold for Birch clustering (radius)")
    parser.add_argument("--top_k_entity_per_cluster", type=int, default=500, help="Top D entities to connect to a cluster center")

    return parser.parse_args()


def load_dataset(dataset_name): 
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    passages = [f'{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages

def load_embedding_model(embedding_model_path):
    embedding_model = SentenceTransformer(embedding_model_path, device="cuda")
    embedding_model.max_length = 512
    return embedding_model

def main():
    # Setup Time and Logging
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    args = parse_arguments()
    
    setup_logging(f"results/{args.dataset_name}/{time_str}/log.txt")
    
    # Load Models and Data
    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.dataset_name)
    llm_model = LLM_Model(args.llm_model)
    
    # Initialize Config with all arguments
    config = EHRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        max_workers=args.max_workers,
        llm_model=llm_model,
        
        # Retrieval Params
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        retrieval_top_k=args.retrieval_top_k,
        damping=args.damping,
        passage_node_weight=args.passage_node_weight,
        
        # EHGRAG Specific Params
        seed_entities_count=args.seed_entities_count,
        semantic_decay=args.semantic_decay,
        max_semantic_seeds=args.max_semantic_seeds,
        topic_ratio=args.topic_ratio,
        cluster_threshold=args.cluster_threshold,
        top_k_entity_per_cluster=args.top_k_entity_per_cluster
    )
    #print(config.top_k_entity_per_cluster)
    
    # Initialize and Index
    rag_model = EHRAG(global_config=config)
    rag_model.index(passages)
    
    # QA Process
    retrieval_results = rag_model.qa(questions)
    
    # Format Results
    formatted_results = []
    for question_info, solution in zip(questions, retrieval_results):
        formatted_results.append({
            "id": question_info["id"],
            "question": question_info["question"],
            "source": args.dataset_name,
            "context": solution.get("sorted_passage", ""),
            "evidence": question_info.get("evidence", ""),
            "question_type": question_info.get("question_type", ""),
            "generated_answer": solution.get("pred_answer", ""),
            "ground_truth": question_info.get("answer", "")
        })

    # Save Output
    os.makedirs(f"results/{args.dataset_name}/{time_str}", exist_ok=True)
    
    with open(f"results/{args.dataset_name}/{time_str}/benchmark.json", "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, ensure_ascii=False, indent=4)
        
    with open(f"results/{args.dataset_name}/{time_str}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=4)
    
    # Evaluate
    evaluator = Evaluator(llm_model=llm_model, predictions_path=f"results/{args.dataset_name}/{time_str}/predictions.json")
    evaluator.evaluate(max_workers=args.max_workers)

if __name__ == "__main__":
    main()