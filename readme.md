# EHRAG

**EHRAG** is a lightweight GraphRAG-style question answering framework that augments dense retrieval with an entity-centered heterogeneous graph and semantic concept clustering.

According to the current codebase, the pipeline:

1. encodes passages, entities, and sentences with a sentence embedding model,
2. extracts named entities from passages and questions with spaCy,
3. builds a graph over passages, entities, and concept nodes induced by BIRCH clustering,
4. retrieves evidence through entity seeding, semantic expansion, structural propagation, passage rescoring, and Personalized PageRank,
5. generates answers with an LLM, and
6. evaluates predictions with both LLM-based correctness and normalized string containment.

The repository is built on top of ideas from **LinearRAG: Linear Graph Retrieval-Augmented Generation on Large-scale Corpora, ICLR2026**, thanks for their wonderful contribution to GraphRAG area.

---

## Repository Structure

```text
EHRAG-main/
├── readme.md
└── EHRAG/
    ├── run.py
    ├── run.sh
    └── src/
        ├── EHRAG.py
        ├── config.py
        ├── embedding_store.py
        ├── evaluate.py
        ├── ner.py
        └── utils.py
```

---

## Method Overview

Based on the implementation in `src/EHRAG.py`, EHRAG works as follows.

### 1. Offline indexing

During indexing, the system:

- stores passage embeddings in a parquet-backed embedding store,
- runs spaCy NER over passages,
- extracts:
  - **entity nodes**,
  - **sentence nodes**, and
  - **passage nodes**,
- stores entity and sentence embeddings,
- builds entity-to-sentence and sentence-to-entity mappings,
- connects entities to passages using occurrence-based edge weights,
- connects adjacent passages when their text begins with indexed prefixes such as `1:`, `2:`, etc.,
- clusters entity embeddings with **BIRCH** to create higher-level **concept nodes**, and
- exports the final graph as `EHRAG.graphml`.

### 2. Query-time retrieval

For each question, the retriever:

- extracts question entities with spaCy,
- finds top matching **seed entities** in the entity embedding space,
- optionally expands to semantically related entities through shared concept clusters,
- propagates over sentence-linked entities for several iterations,
- computes dense passage similarity,
- adds entity and topic-aware bonuses to passage scores,
- runs **Personalized PageRank (PPR)** over the graph, and
- returns the top passages as evidence.

### 3. Answer generation and evaluation

After retrieval, the system:

- concatenates retrieved passages into the QA prompt,
- queries an LLM for the final answer,
- extracts the text after `Answer:` when available,
- saves detailed predictions to disk, and
- evaluates results with:
  - **LLM Accuracy**: a second LLM judges whether the prediction matches the gold answer,
  - **Contain Accuracy**: normalized gold-answer containment in the predicted answer.

---

## Requirements

The code imports the following Python packages:

- `sentence-transformers`
- `spacy`
- `openai`
- `httpx`
- `numpy`
- `pandas`
- `tqdm`
- `scikit-learn`
- `python-igraph`
- `pyarrow` or another parquet backend for pandas

A minimal setup is therefore:

```bash
pip install sentence-transformers spacy openai httpx numpy pandas tqdm scikit-learn python-igraph pyarrow
```

You also need a compatible spaCy model. The code uses:

- `en_core_web_trf` for general-domain datasets,
- `en_core_sci_scibert` for medical-domain settings in `run.sh`.

Example:

```bash
python -m spacy download en_core_web_trf
```

If you use a scientific spaCy pipeline such as `en_core_sci_scibert`, install the corresponding model separately.

---

## Environment Variables

The LLM wrapper in `src/utils.py` reads:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Set them before running:

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_api_base_url
```

`OPENAI_BASE_URL` is optional if you use the default OpenAI endpoint, but the current code is written to support OpenAI-compatible backends as well.

---

## Data Format

The runner expects the following dataset layout:

```text
EHRAG/
└── dataset/
    └── <dataset_name>/
        ├── questions.json
        └── chunks.json
```

### `chunks.json`

A JSON list of passage strings:

```json
[
  "1: First passage text ...",
  "2: Second passage text ..."
]
```

### `questions.json`

A JSON list of question objects. From `run.py`, the following keys are used:

- `id`
- `question`
- `answer`
- optional: `evidence`
- optional: `question_type`

Example:

```json
[
  {
    "id": "q1",
    "question": "Who wrote the book?",
    "answer": "John Smith",
    "evidence": ["passage_12"],
    "question_type": "bridge"
  }
]
```

---

## How to Run

Change into the project directory that contains `run.py`:

```bash
cd EHRAG
```

### Run with command-line arguments

```bash
python run.py \
  --dataset_name hotpotqa \
  --spacy_model en_core_web_trf \
  --embedding_model sentence-transformers/all-mpnet-base-v2 \
  --llm_model gpt-4o-mini \
  --max_workers 16 \
  --max_iterations 5 \
  --iteration_threshold 0.4 \
  --passage_ratio 2.0 \
  --top_k_sentence 4 \
  --retrieval_top_k 5 \
  --damping 0.5 \
  --passage_node_weight 1.0 \
  --seed_entities_count 1 \
  --semantic_decay 0.1 \
  --max_semantic_seeds 2 \
  --topic_ratio 0.5 \
  --cluster_threshold 0.9 \
  --top_k_entity_per_cluster 500
```

### Run with the provided shell script

The included `run.sh` contains presets for several datasets such as:

- `medical`
- `musique`
- `2wikimultihop`
- `hotpotqa`

Use it as a template:

```bash
bash run.sh
```

---

## Main Arguments

The following arguments are exposed in `run.py`.

### Core settings

| Argument | Description | Default |
|---|---|---:|
| `--dataset_name` | Dataset name under `dataset/` | `hotpotqa` |
| `--spacy_model` | spaCy model for NER | `en_core_web_trf` |
| `--embedding_model` | Sentence embedding model path/name | `model/all-mpnet-base-v2` |
| `--llm_model` | LLM used for QA and evaluation | `gpt-4o-mini` |
| `--max_workers` | Parallel worker count | `16` |

### Retrieval and graph propagation

| Argument | Description | Default |
|---|---|---:|
| `--max_iterations` | Maximum propagation iterations | `5` |
| `--iteration_threshold` | Score threshold for continuing propagation | `0.4` |
| `--passage_ratio` | Weight applied to dense passage retrieval score | `2.0` |
| `--top_k_sentence` | Number of top sentences expanded per entity | `4` |
| `--retrieval_top_k` | Final number of retrieved passages | `5` |
| `--damping` | Damping factor for Personalized PageRank | `0.5` |
| `--passage_node_weight` | Multiplier applied to passage node weights | `1.0` |

### Specific semantic expansion

| Argument | Description | Default |
|---|---|---:|
| `--seed_entities_count` | Number of seed entities selected per query entity | `1` |
| `--semantic_decay` | Decay for semantically expanded entities | `0.1` |
| `--max_semantic_seeds` | Maximum number of semantic expansion seeds | `2` |
| `--topic_ratio` | Weight for concept/topic importance bonus | `0.5` |

### Concept clustering

| Argument | Description | Default |
|---|---|---:|
| `--cluster_threshold` | BIRCH threshold for adaptive clustering | `0.9` |
| `--top_k_entity_per_cluster` | Number of top entities linked to a concept node | `500` |

