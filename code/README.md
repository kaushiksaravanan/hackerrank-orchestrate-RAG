# Support Triage Agent

Python support triage agent for the HackerRank Orchestrate challenge.

## What It Does

- Reads support tickets from `support_tickets/support_tickets.csv`
- Retrieves evidence only from `data/`
- Classifies each ticket and drafts a grounded response
- Writes `support_tickets/output.csv` with the required schema

## Architecture

```mermaid
flowchart LR
    classDef io      fill:#e7f5ff,stroke:#1971c2,color:#0c4a73
    classDef cfg     fill:#f8f0fc,stroke:#9c36b5,color:#5c1d72
    classDef ret     fill:#edf2ff,stroke:#4263eb,color:#1e3a8a
    classDef llm     fill:#fff0f6,stroke:#c2255c,color:#7a1144
    classDef store   fill:#ebfbee,stroke:#2b8a3e,color:#1f6630
    classDef out     fill:#e6fcf5,stroke:#0ca678,color:#0a5f49
    classDef test    fill:#f1f3f5,stroke:#495057,color:#212529

    CSV([support_tickets.csv]):::io
    MAIN[main.py<br/>batch runner + CLI]:::io

    CFG[config.py<br/>paths · env · taxonomy]:::cfg
    PRE[preprocessor.py<br/>clean markdown corpus]:::cfg
    EMB[embeddings.py<br/>ONNX dense vectors]:::cfg

    IDX[indexer.py<br/>chunk + BM25 + cache]:::ret
    RETR[retriever.py<br/>BM25 + vector + RRF]:::ret
    AGENT[agent.py<br/>fast-path · LLM · grounding]:::ret

    PROMPTS[prompts.py<br/>system prompt + few-shot]:::llm
    PROXY[Anthropic proxy<br/>localhost:6655]:::llm

    DATA[(local corpus<br/>data/hackerrank · data/claude · data/visa)]:::store
    CACHE[(code/.cache<br/>model + pickled indices)]:::store

    OUT([output.csv]):::out
    TESTS[pytest<br/>67 passing tests]:::test

    CSV --> MAIN
    MAIN --> CFG
    CFG --> PRE --> EMB --> IDX
    DATA --> IDX --> CACHE
    IDX --> RETR --> AGENT
    PROMPTS --> AGENT
    AGENT <--> PROXY
    AGENT --> OUT
    TESTS -. covers .-> AGENT
    TESTS -. covers .-> RETR
    TESTS -. covers .-> IDX
```

- Pipeline: CSV intake → setup/config → corpus cleanup → embedding + indexing → hybrid retrieval → prompt + LLM response → grounding verification → output CSV
- `config.py`: paths, constants, taxonomy, env loading
- `preprocessor.py`: markdown cleanup for HackerRank, Claude, and Visa docs
- `embeddings.py`: HF Inference API embedder (preferred), ONNX local fallback, TF-IDF last resort
- `llm.py`: unified LLM client wrapping Anthropic + HuggingFace Inference API
- `indexer.py`: document loading, heading-based chunking, BM25/vector index build, model-aware cache
- `retriever.py`: hybrid retrieval with BM25, dense search, and RRF fusion
- `prompts.py`: system prompt, few-shot examples, prompt assembly
- `agent.py`: fast-path rules, LLM generation, grounding checks, ticket orchestration
- `main.py`: CLI entry point for batch CSV processing

## Requirements

- Python 3.14+
- Access to the local Anthropic-compatible proxy at `ANTHROPIC_BASE_URL`
- `ANTHROPIC_API_KEY` or `HAI_API_KEY` in `code/.env` or repo-root `.env`
- Optional `HG_TOKEN` for Hugging Face downloads

Install dependencies:

```bash
pip install -r code/requirements.txt
```

## Environment

Create `code/.env` with:

```env
ANTHROPIC_API_KEY=your_api_key_here
HAI_API_KEY=your_api_key_here
HG_TOKEN=your_huggingface_token_here
ANTHROPIC_BASE_URL=http://localhost:6655/anthropic/v1
```

The code normalizes the Anthropic base URL for the SDK automatically.

## Run

From the repository root:

```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```

Optional:

```bash
python code/main.py --rebuild-index
```

## Output Schema

The generated CSV contains:

- `issue`
- `subject`
- `company`
- `response`
- `product_area`
- `status`
- `request_type`
- `justification`

## Retrieval And Determinism

- Retrieval is limited to the local `data/` corpus
- Dense embeddings use `BAAI/bge-large-en-v1.5` (1024d) via HF Inference API, with ONNX and TF-IDF fallbacks
- Hybrid ranking combines BM25 and vector similarity with reciprocal rank fusion
- Cached indices are stored under `code/.cache/` (gitignored, auto-rebuilt on first run)
- Cache keys include a model hash so they auto-invalidate when the embedding model changes
- LLM calls use `temperature=0`
- Dependencies are pinned in `requirements.txt`

## Tests

Run from the repository root:

```bash
python -m pytest code/tests -q
```

The suite includes:

- unit tests for config, preprocessing, embeddings, indexing, retrieval, prompts, and agent logic
- integration tests that exercise the real end-to-end pipeline

## Submission Zip

The `code/.cache/` directory (~160 MB of pickle indices) is gitignored and must be excluded
from the zip. It is auto-rebuilt on first run. Use `git archive` to create a clean zip (~2.5 MB):

```bash
git archive --format=zip --output=submission.zip HEAD
```

This respects `.gitignore` and excludes `.cache/`, `__pycache__/`, `.env`, etc.
