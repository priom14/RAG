# RAG Pipeline with Groq LLM and ChromaDB

A robust Retrieval-Augmented Generation (RAG) pipeline designed to process PDF documents, store their semantic representations in a vector database, and generate concise answers to queries using the Groq Llama-3.3-70B model.

## 🚀 Overview

This project implements a complete RAG workflow consisting of:
1.  **Data Ingestion**: Automated loading and processing of PDF files from a local directory (e.g., iTransformer.pdf, Nbeats.pdf).
2.  **Text Processing**: Intelligent document splitting using `RecursiveCharacterTextSplitter` into 1000-character chunks with a 200-character overlap.
3.  **Vector Store**: Semantic indexing using `SentenceTransformer` (all-MiniLM-L6-v2) and `ChromaDB` for persistent storage.
4.  **Retrieval**: Custom `RAGRetriever` logic with similarity score normalization: `1 / (1 + distance)`.
5.  **Generation**: Integration with Groq's high-speed inference API (`llama-3.3-70b-versatile`) to provide context-aware answers.

## 🛠️ Tech Stack

-   **LLM**: Groq (`llama-3.3-70b-versatile`)
-   **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
-   **Vector DB**: ChromaDB
-   **Framework**: LangChain Community (Document Loaders, Text Splitters)
-   **Environment**: Python, Jupyter Notebook, Dotenv

## 📂 Project Structure

-   `notebook/RAGpipline.ipynb`: Main implementation notebook.
-   `../data/`: Source directory for PDF documents.
-   `../data/vector_store/`: Persistent storage for document embeddings.
-   `.env`: Configuration for sensitive `GROQ_API_KEY`.

## ⚙️ Setup Instructions

### 1. Prerequisites
-   Python 3.10+
-   Groq API Key (obtain from console.groq.com)

### 2. Installation
```bash
pip install langchain-groq langchain-community sentence-transformers chromadb python-dotenv pypdf pymupdf scikit-learn