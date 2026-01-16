# 🔬 Advanced Research Paper Analyzer

An enterprise-grade, privacy-focused Retrieval-Augmented Generation (RAG) application designed for researchers and data scientists. This tool enables local, cost-free analysis of dense academic papers using a hybrid search architecture.



## 🌟 Key Features

* **Hybrid Search Engine:** Combines **Semantic (Dense)** retrieval via HuggingFace BGE embeddings and **Keyword (Sparse)** retrieval via BM25 to ensure technical terminology is never missed.
* **State-Aware Conversation:** Built with **LangChain Expression Language (LCEL)**, featuring a history-aware retriever that re-contextualizes follow-up questions for precise accuracy.
* **100% Local & Private:** Powered by **Ollama**, ensuring all data stays on your machine with zero API costs or data leakage.
* **Automated Context Management:** Features smart session handling that clears vector memory and chat history upon new document uploads.
* **Citation Transparency:** Provides source-backed responses with exact page-level citations and metadata extraction.

## 🏗️ System Architecture

The project follows a modular design pattern to separate concerns between ingestion, retrieval logic, and the user interface.



* **Ingestion Layer:** Utilizes `PyPDF` and `RecursiveCharacterTextSplitter` to maintain semantic coherence across chunks.
* **Vector Store:** Powered by **ChromaDB** for persistent, low-latency embedding storage.
* **LLM Orchestration:** Uses `llama3.1` (via Ollama) and `BAAI/bge-small-en-v1.5` embeddings for high-performance inference on local hardware.

## 📁 Project Structure

```text
research-paper-analyzer/
├── app.py                 # Streamlit UI & State Management
├── requirements.txt       # Version-pinned dependencies
├── engine/                # Core Logic Package
│   ├── __init__.py        
│   ├── ingestion.py       # PDF Parsing & Chunking
│   ├── hybrid_engine.py   # Hybrid Retrieval & LCEL Chains
│   └── prompts.py         # Versioned ChatPromptTemplates
└── data/                  # Persistent Vector Storage

```

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* [Ollama](https://ollama.com/) installed and running

### Installation

1. **Pull the LLM:**
```bash
ollama pull llama3.1

```


2. **Clone the Repository:**
```bash
git clone https://github.com/althaffazil/research-paper-analyzer.git
cd research-paper-analyzer

```


3. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


4. **Run the Application:**
```bash
streamlit run app.py

```



## 🛠️ Tech Stack

* **UI:** Streamlit
* **Orchestration:** LangChain (LCEL)
* **LLM:** Ollama (Llama 3.1)
* **Embeddings:** HuggingFace BGE Small
* **Vector DB:** ChromaDB
* **Search Algorithms:** BM25 (Rank-BM25), Cosine Similarity
