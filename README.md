# Trace

<div align="center">

Next-Generation Financial Assistant leveraging Hybrid RAG, Quantized SLMs, and Optical Character Recognition.

![Trace Screenshot](./image.png)

</div>

### 📖 Overview
Trace is not just an expense tracker; it is a production-grade implementation of a RAG (Retrieval-Augmented Generation) pipeline designed to solve the "unstructured data" problem in personal finance.

While traditional expense trackers rely on manual entry, Trace uses PaddleOCR and Structured LLM Extraction to convert raw receipt images into type-safe JSON. It enables natural language queries over financial data by utilizing a Hybrid Search Engine that fuses dense vector retrieval with sparse keyword matching, re-ranked by a cross-encoder for maximum accuracy.

## ✨ Key Engineering Features

### 🧠 The "Smart Stack" (RAG Architecture)
Hybrid Search Strategy: Solves the limitations of pure vector search by combining ChromaDB (Dense/Semantic) with BM25 (Sparse/Keyword). This allows the system to understand concepts like "dinner" while still finding exact matches for "$42.50".

Cross-Encoder Reranking: Implements a "Judge" model (ms-marco-MiniLM-L-6-v2) running on ONNX Runtime to re-score retrieval candidates, drastically reducing hallucinations.

Semantic Chunking: Deconstructs receipts into two distinct vector types:

Full Document: For high-level summaries (Merchant, Total, Date).

Item Granularity: Individual vector embeddings for every line item, allowing queries like "How much have I spent on milk?".

Zero-Loss Fallback: Intelligent retrieval logic that falls back to a full-context scan if semantic search confidence drops below threshold.

### ⚡ High-Performance Backend
Asynchronous Processing: Built on FastAPI with fully async endpoints for non-blocking I/O.

ONNX Optimization: Embedding models and Rerankers are quantized and run via ONNX, removing the need for heavy PyTorch dependencies in production.

Streaming Responses: Utilizes Server-Sent Events (SSE) to stream LLM tokens to the frontend in real-time, reducing perceived latency.

### 🎨 Modern Frontend
Tech: React 19, TypeScript, and Vite.

UX: "Matrix-style" real-time OCR visualization using coordinate mapping from PaddleOCR.

State: TanStack Query for optimistic updates and caching.

### 🏗️ System Architecture
The following diagram illustrates the data ingestion and retrieval pipeline:

```mermaid
flowchart TD
    %% ─── STYLING ───
    classDef user fill:#000000,stroke:#333,stroke-width:2px,color:white;
    classDef endpoint fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e;
    classDef service fill:#f0fdf4,stroke:#16a34a,stroke-width:2px,color:#14532d;
    classDef db fill:#fff7ed,stroke:#ea580c,stroke-width:2px,color:#7c2d12,shape:cyl;
    classDef process fill:#f3f4f6,stroke:#64748b,stroke-width:1px,color:#1e293b,stroke-dasharray: 5 5;
    classDef logic fill:#fef3c7,stroke:#d97706,stroke-width:2px,color:#92400e;

    %% ─── SHARED NODES ───
    User((USER)):::user
    
    subgraph DataLayer [Data Persistence Layer]
        ChromaDB[(ChromaDB)]:::db
    end

    %% ─── PATH 1: INGESTION ───
    subgraph Path1 [Path 1: Ingestion Pipeline]
        direction TB
        ScanEP["/ocr/scan"]:::endpoint
        ParseEP["/ocr/parse"]:::endpoint
        
        OCRSvc["OCR Service<br/>(PaddleOCR)"]:::service
        LLMParse["LLM Extraction<br/>(Instructor)"]:::service
        RAGStore["RAG Service<br/>(Ingest)"]:::service
        
        UI_Anim["Frontend Animation<br/>(Raw Polygons)"]:::process
        
        FullDoc["Full Receipt Doc"]:::process
        ItemDoc["Individual Item Chunks"]:::process

        %% Flow
        ScanEP --> OCRSvc
        OCRSvc -- "1. Raw Text + Polygons" --> UI_Anim
        
        ParseEP --> LLMParse
        LLMParse -- "2. Structured JSON" --> RAGStore
        
        RAGStore --> FullDoc
        RAGStore --> ItemDoc
    end

    %% ─── PATH 2: QUERY ───
    subgraph Path2 [Path 2: Hybrid RAG Logic]
        direction TB
        ChatEP["/ai/chat"]:::endpoint
        RAGQuery["RAG Service Wrapper"]:::service
        
        %% The Search Pillars
        Dense["1. Dense Search<br/>(ChromaDB / ONNX)"]:::logic
        Sparse["2. Sparse Search<br/>(BM25 / RAM)"]:::logic
        
        %% The "Secret Sauce" Logic Steps
        Merge["3. Merge & Deduplicate<br/>(Combine Candidates)"]:::logic
        Rerank["4. Cross-Encoder Rerank<br/>(Filter & Sort)"]:::logic
        Resolver["5. Context Resolver<br/>(Swap Item Chunk → Full Receipt)"]:::logic
        
        LLMResp["6. LLM Generation<br/>(Streaming)"]:::service

        %% Flow
        ChatEP --> RAGQuery
        RAGQuery -- "Query" --> Dense
        RAGQuery -- "Query" --> Sparse
        
        Dense --> Merge
        Sparse --> Merge
        
        Merge -- "Unique Candidates" --> Rerank
        Rerank -- "Top K Scored" --> Resolver
        Resolver -- "Full Context" --> LLMResp
    end

    %% ─── GLOBAL CONNECTIONS ───
    %% User Interactions
    User -- "1. Upload" --> ScanEP
    UI_Anim -.-> User
    User -- "2. Parse & Store" --> ParseEP
    User -- "3. Ask Question" --> ChatEP
    LLMResp -- "Stream Answer" --> User

    %% Database Interactions
    FullDoc --> ChromaDB
    ItemDoc --> ChromaDB
    
    %% Retrieval from DB
    ChromaDB <--> Dense
    ChromaDB -.-> |"Load BM25 Index"| Sparse
```

## 🛠️ Tech Stack

| Component         | Technology                | Description                                       |
| :---------------- | :------------------------ | :------------------------------------------------ |
| **Backend**       | ------------------------- | ------------------------------------------------- |
| Framework         | FastAPI                   | Async Python web server.                          |
| Vector Store      | ChromaDB                  | Persistent local vector storage.                  |
| OCR               | PaddleOCR                 | Lightweight, SOTA text detection.                 |
| LLM Orchestration | Instructor                | Structured output validation (Pydantic).          |
| Reranking         | Cross-Encoders (ONNX)     | Accelerated CPU inference for Cross-Encoders.     |
| **Frontend**      | ------------------------- | ------------------------------------------------- |
| Core              | React + Vite              | Fast component rendering.                         |
| Language          | TypeScript                | Strict type safety.                               |
| Styling           | TailwindCSS               | Utility-first styling.                            |
| UI Library        | Radix UI / Shadcn         | Accessible component primitives.                  |


## 🚀 Getting Started

### Prerequisites
Docker & Docker Compose
Ollama running Phi 3.5 (`ollama run phi3.5`)

### Installation

Clone the repository:

```bash
git clone https://github.com/createdbyadham/Trace.git
cd Trace
```

### Environment Configuration

Create a `.env` file in the `backend/` directory:

```
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=phi3.5
```

```bash
# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

# Run Server with Hot Reload
uvicorn main:app --reload --port 8000
```

📄 License
Distributed under the MIT License. See LICENSE for more information.
