# Trace - Intelligent Receipt Analysis Platform

A cutting-edge financial assistant that leverages Hybrid RAG, quantized LLMs and OCR to revolutionize receipt management. Trace moves beyond "Naive RAG" by implementing a production-grade architecture capable of understanding both semantic concepts ("dinner") and exact keywords ("$420.69").

![Trace Screenshot](./image.png)

## 🚀 Key Technical Features

### Advanced AI Architecture (The "Smart Stack")
- **Hybrid Search Engine:** Combines **Dense Vector Search** (ChromaDB) with **Sparse Keyword Search** (BM25) to ensure zero data loss.
- **Cross-Encoder Reranking:** Utilizes `ms-marco-MiniLM-L-6-v2` as a "Judge" model to re-score retrieval results for maximum relevance.
- **Smart Ingestion Pipeline:** - **PaddleOCR** for SOTA lightweight text detection.
  - **Quantized SLM Parsing:** Uses small language models to convert raw OCR spaghetti text into structured, type-safe JSON before storage.
- **Zero Double-LLM Calls:** Optimized retrieval pipeline runs entirely locally (CPU), sending only the perfect context to the LLM for the final answer.

### 💻 Modern Frontend
- **Performance:** React + TypeScript + Vite.
- **State:** TanStack Query for async state management.
- **UI/UX:** Framer Motion for animations, Radix UI for accessibility, and Tailwind CSS for styling.
- **Real-time:** Streaming responses using Server-Sent Events (SSE).

### ⚙️ Backend Infrastructure
- **FastAPI:** High-performance async Python backend.
- **ChromaDB:** Local vector store with persistent on-disk storage.
- **Optimization:** All models are ran using ONNX
- **Memory Management:** Custom semantic memory implementation (Chat History).
- **Containerization:** Multi-stage Docker builds with volume persistence for model weights.

## 🏗️ Architecture

The project follows a specialized Service-Oriented Architecture: