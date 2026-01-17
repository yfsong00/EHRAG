from dataclasses import dataclass
from src.utils import LLM_Model
from typing import Any, Optional

@dataclass
class EHRAGConfig:

    dataset_name: str
    embedding_model: Any = "all-mpnet-base-v2" 
    llm_model: Optional[LLM_Model] = None
    spacy_model: str = "en_core_web_trf"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    

    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100


    retrieval_top_k: int = 5
    max_iterations: int = 5
    top_k_sentence: int = 4
    passage_ratio: float = 2.0
    passage_node_weight: float = 0.1
    damping: float = 0.5
    iteration_threshold: float = 0.1


    seed_entities_count: int = 1      
    semantic_decay: float = 0.2    
    max_semantic_seeds: int = 2      
    topic_ratio: float = 0.1           
    

    cluster_threshold: float = 0.9     
    top_k_entity_per_cluster: int = 100 