# PDF Q&A with RAG & LangChain

A Retrieval-Augmented Generation (RAG) application that lets you upload a PDF and ask natural-language questions about its contents. Built with LangChain, OpenAI and ChromaDB, served through a Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green?logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)

## How It Works

```
PDF Upload ──► Text Extraction ──► Chunking ──► Embeddings ──► Vector Store
                                                                    │
User Question ──────────────────► Semantic Search ◄─────────────────┘
                                        │
                                  Relevant Chunks + Question
                                        │
                                    GPT-4o ──► Answer
```

1. **Ingest** — Upload a PDF; text is extracted page-by-page with PyPDF2.
2. **Chunk** — The extracted text is split into 500-character chunks (with 100-character overlap) to preserve context across boundaries.
3. **Embed** — Each chunk is embedded using OpenAI's `text-embedding-3-small` model.
4. **Store** — Embeddings are indexed in an in-memory ChromaDB vector store.
5. **Retrieve** — When a question is asked, the most semantically similar chunks are retrieved.
6. **Generate** — Retrieved context and the question are passed to GPT-4o via a RAG prompt chain, producing a grounded answer.

## Project Structure

```
├── app.py          # Streamlit web app — PDF upload & interactive Q&A
├── main.py         # CLI demo — RAG pipeline with hardcoded sample documents
├── .env            # API keys (not committed)
└── README.md
```

| File | Purpose |
|------|---------|
| `app.py` | Production-ready Streamlit interface for uploading PDFs and querying them conversationally. Uses LangSmith's prompt registry for the RAG prompt. |
| `main.py` | Educational script that walks through every stage of the RAG pipeline step-by-step, using hardcoded documents about programming, AI, and Paris landmarks. |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Store | ChromaDB (in-memory) |
| Framework | LangChain (core, OpenAI, Chroma, text-splitters) |
| Prompt Management | LangSmith |
| Web UI | Streamlit |
| PDF Parsing | PyPDF2 |

## Getting Started

### Prerequisites

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A [LangSmith API key](https://smith.langchain.com/) (used by `app.py` for prompt registry)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/RAG-Langchain.git
cd RAG-Langchain

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install langchain langchain-openai langchain-chroma langchain-text-splitters \
            streamlit PyPDF2 python-dotenv langsmith chromadb
```

### Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
LANGSMITH_API_KEY=your-langsmith-api-key
```

### Usage

**Web App (Streamlit)**

```bash
streamlit run app.py
```

Then open your browser, upload a PDF, type a question, and hit **Submit**.

**CLI Demo**

```bash
python main.py
```

Runs the full RAG pipeline against built-in sample documents and prints the answer to the console.

## Architecture

The RAG chain is composed using LangChain's LCEL (LangChain Expression Language):

```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Each component is a **Runnable** — they snap together like building blocks, making the pipeline easy to extend with additional retrievers, re-rankers, or output handlers.

## License

This project is open-source and available under the [MIT License](LICENSE).
