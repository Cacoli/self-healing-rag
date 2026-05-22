# Self-Healing RAG Pipeline

A production-grade Retrieval-Augmented Generation system that critiques its own outputs and retries when hallucinations are detected.

# Architecture

User Query
↓
[Retriever] → Pinecone vector search
↓
[Generator] → Groq LLaMA 3.3 70B answers using ONLY retrieved context
↓
[Critic] → Detects hallucinations, verifies grounding
↓
PASS → Return answer
FAIL → [Reformulator] rewrites query → retry (max 2x)
FAIL after retries → Graceful fallback

##  Features

- **Self-healing loop** — critic agent detects hallucinations and triggers query reformulation
- **Zero hallucination policy** — model is instructed to use only retrieved context
- **Graceful fallback** — returns honest "I don't know" instead of making things up
- **Observable pipeline** — every step logged with verdict and reasoning
- **Streamlit UI** — clean web interface with pipeline details and retrieved chunks

##  Tech Stack

| Layer | Tool |
|---|---|
| Graph orchestration | LangGraph |
| LLM | Groq (LLaMA 3.3 70B) — free tier |
| Vector store | Pinecone — free tier |
| Embeddings | HuggingFace all-MiniLM-L6-v2 — local, free |
| Frontend | Streamlit |

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Cacoli/self-healing-rag.git
cd self-healing-rag
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install langgraph langchain langchain-groq langchain-pinecone pinecone sentence-transformers python-dotenv streamlit
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Fill in your keys in `.env`:
```env
GROQ_API_KEY=your_groq_key        # free at console.groq.com
PINECONE_API_KEY=your_pinecone_key # free at app.pinecone.io
PINECONE_INDEX_NAME=self-healing-rag
```

### 5. Ingest documents
```bash
python ingest.py
```

### 6. Run the app
```bash
python -m streamlit run app.py
```

#  Project Structure
self-healing-rag/
├── src/
│   ├── state.py          # Shared graph state definition
│   ├── retriever.py      # Pinecone vector retrieval node
│   ├── generator.py      # LLM answer generation node
│   ├── critic.py         # Hallucination detection node
│   ├── reformulator.py   # Query rewriting node
│   └── graph.py          # LangGraph assembly
├── app.py                # Streamlit web UI
├── ingest.py             # Document ingestion script
├── main.py               # CLI runner
└── .env.example          # Environment variable template

## How the Critic Works

The critic receives three inputs:
1. The retrieved context chunks
2. The original user question
3. The generated answer

It then checks every claim in the answer against the context. If any claim cannot be traced back to the retrieved chunks, it returns `FAIL` with a reasoning explanation. The reformulator uses this reasoning to rewrite a better search query.

##  Example Output
QUESTION: How does the attention mechanism work in transformers?
 [RETRIEVER] Found 4 chunks
 [GENERATOR] Answer generated
 [CRITIC] Verdict: PASS
Reasoning: All claims traceable to context
 FINAL ANSWER:
The attention mechanism computes a weighted sum of values
based on the similarity between queries and keys...
 Stats:
Retries: 0
Critic verdict: PASS
##  Getting Free API Keys

- **Groq** → [console.groq.com](https://console.groq.com)
- **Pinecone** → [app.pinecone.io](https://app.pinecone.io)