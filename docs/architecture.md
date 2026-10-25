# Architecture

![End-to-end architecture](architecture.svg)

Uploads (Streamlit sidebar or `scripts/ingest.py`) go down the write path (left), which parses, chunks and embeds CVs into Chroma. Questions from the chat, and from `scripts/evaluate.py`, go down the read path (centre): the LangGraph pipeline in `src/cv_rag/graph.py`. If Grade keeps no chunks, the query is rewritten and retrieved again. The right-hand column shows the early exits: off-topic questions stop after Analyze, and if no relevant chunks survive the rewrites, the pipeline replies that it found no context. Retry limits are set by `MAX_QUERY_REWRITES` and `MAX_GENERATION_RETRIES` (both default to 1).

## System overview

```mermaid
flowchart TB
    subgraph IDX["1. Indexing: sidebar upload or scripts/ingest.py"]
        direction LR
        F[CV files<br/>PDF, DOCX, TXT, MD] --> L[Loaders<br/>extract text]
        L --> CH[Chunker<br/>by section]
        CH --> EM[Embeddings<br/>OpenAI]
        EM --> DB[(Chroma DB<br/>local, cosine)]
    end

    subgraph QA["2. Question answering: app/streamlit_app.py"]
        direction LR
        UI[Chat UI<br/>Streamlit] <--> G[LangGraph pipeline<br/>GPT-4o-mini, self-checking]
        G <--> RT[Retriever<br/>MMR, top 6, name filter]
    end

    subgraph EV["3. Evaluation: scripts/evaluate.py"]
        direction LR
        QS[Questions<br/>cv_eval.jsonl] --> RN[Runner<br/>retrieval metrics + GPT-4o judge]
        RN --> RP[Report<br/>results/latest.md]
    end

    DB <--> RT
    RN <--> G
```

## LangGraph pipeline (`src/cv_rag/graph.py`)

```mermaid
flowchart LR
    AN[Analyze<br/>intent, query] -->|CV question| RE[Retrieve<br/>Chroma MMR]
    AN -->|off topic| OT[Off topic<br/>canned reply]
    RE --> GR[Grade<br/>LLM filter]
    GR -->|relevant chunks| GE[Generate<br/>cited answer]
    GR -->|none, rewrites left| RW[Rewrite<br/>new query]
    RW --> RE
    GR -->|none, rewrites used| NC[No context<br/>give up]
    GE --> VF[Verify<br/>grounded?]
    VF -->|not grounded| GE
    VF -->|grounded or retries used| AS[Answer<br/>with sources]
```

Retry limits are set by `MAX_QUERY_REWRITES` and `MAX_GENERATION_RETRIES` (both default to 1).
