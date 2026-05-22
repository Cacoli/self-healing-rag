import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
DOCUMENTS = [
    # --- RAG ---
    "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a generative language model. Instead of relying solely on the model's parametric knowledge, RAG fetches relevant documents from an external knowledge base at inference time and passes them as context to the LLM, reducing hallucinations and improving factual accuracy.",
    "A RAG pipeline has three core components: an indexing phase where documents are chunked and embedded into a vector store, a retrieval phase where the user query is embedded and similar chunks are fetched, and a generation phase where the retrieved chunks are passed as context to an LLM to produce a grounded answer.",
    "Chunking strategy significantly affects RAG quality. Common approaches include fixed-size chunking (splitting by token count), sentence-based chunking (splitting on sentence boundaries), and semantic chunking (splitting based on topic shifts). Smaller chunks improve retrieval precision but may lose context; larger chunks preserve context but reduce precision.",
    "Hybrid search in RAG combines dense vector search (semantic similarity via embeddings) with sparse keyword search (BM25 or TF-IDF). This improves retrieval by catching both semantically similar documents and exact keyword matches, which is especially useful for technical queries with specific terminology.",
    "RAG evaluation metrics include: Faithfulness (is the answer grounded in retrieved context?), Answer Relevance (does the answer address the question?), Context Precision (are retrieved chunks relevant?), and Context Recall (were all relevant chunks retrieved?). Tools like RAGAS automate these evaluations.",

    # --- Embeddings ---
    "Word embeddings are dense vector representations of text where semantically similar words or sentences are close together in vector space. Models like Word2Vec and GloVe produce word-level embeddings, while sentence transformers like all-MiniLM-L6-v2 produce sentence-level embeddings that capture full semantic meaning.",
    "Sentence transformers use a siamese network architecture trained with contrastive learning to produce embeddings where similar sentences have high cosine similarity. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings and is a popular choice for RAG systems due to its speed and quality balance.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embeddings and outperforms older models on most retrieval benchmarks. Embedding dimensions affect both retrieval quality and storage cost — higher dimensions capture more nuance but require more memory and compute for similarity search.",
    "Cosine similarity measures the angle between two vectors, making it scale-invariant and ideal for comparing embeddings of different lengths. A cosine similarity of 1 means identical direction (most similar), 0 means orthogonal (unrelated), and -1 means opposite. Most vector databases use cosine or dot product similarity.",

    # --- Vector Databases ---
    "Vector databases are purpose-built systems for storing and searching high-dimensional embedding vectors. Unlike traditional databases that match exact values, vector databases use approximate nearest neighbor (ANN) algorithms like HNSW or IVF to find the most semantically similar vectors efficiently at scale.",
    "Pinecone is a managed vector database that handles indexing, scaling, and infrastructure automatically. It supports metadata filtering, allowing you to combine semantic search with structured filters like date ranges or categories. Pinecone's serverless tier is free and suitable for development and small production workloads.",
    "FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search. It offers multiple index types: Flat (exact, slow), IVF (inverted file, fast approximate), and HNSW (hierarchical navigable small world, best accuracy/speed tradeoff). FAISS runs locally with no cost but requires self-management.",
    "Chroma is an open-source vector database that runs locally or in the cloud. It is popular for prototyping RAG systems because it requires no external services, supports persistence to disk, and integrates natively with LangChain. For production at scale, managed solutions like Pinecone are preferred.",

    # --- LLM Concepts ---
    "Large Language Models (LLMs) are transformer-based neural networks trained on massive text corpora to predict the next token. At inference time, they generate text autoregressively — one token at a time — using the probability distribution learned during training. Popular LLMs include GPT-4, Claude, Gemini, and LLaMA.",
    "The transformer architecture introduced in 'Attention is All You Need' (2017) uses self-attention mechanisms to weigh the importance of each token relative to every other token in the input. This allows transformers to capture long-range dependencies in text far better than previous RNN and LSTM architectures.",
    "The attention mechanism computes a weighted sum of values based on the similarity between queries and keys. In multi-head attention, this process runs in parallel across multiple heads, each learning different relationship patterns in the data. The output is concatenated and projected to produce the final representation.",
    "Temperature controls the randomness of LLM outputs. A temperature of 0 makes the model deterministic — always picking the most likely token. Higher temperatures (0.7-1.0) increase diversity and creativity. For factual RAG systems, low temperatures (0.1-0.3) are preferred to reduce hallucinations.",
    "Context window refers to the maximum number of tokens an LLM can process in a single call, including both input and output. GPT-4 supports up to 128K tokens, Claude supports up to 200K tokens. Longer context windows allow more retrieved chunks to be passed to the model but increase cost and latency.",

    # --- Agents & LangGraph ---
    "An AI agent is a system where an LLM is given tools (web search, code execution, database queries) and autonomously decides which tools to call, in what order, to complete a task. Unlike a simple chain, agents can branch, loop, and adapt their behavior based on intermediate results.",
    "LangGraph is a framework for building stateful, multi-actor LLM applications as directed graphs. Each node in the graph is a processing step (a Python function), and edges define the flow between steps. Conditional edges allow dynamic routing based on the current state, enabling retry loops and branching workflows.",
    "The ReAct (Reasoning + Acting) pattern is a popular agent design where the LLM alternates between reasoning steps (thinking about what to do) and action steps (calling a tool). The results of actions are fed back into the context, allowing the model to reason about outcomes and plan next steps.",
    "Fine-tuning adapts a pretrained LLM to a specific domain or task by continuing training on a curated dataset. Full fine-tuning updates all model weights and is expensive. Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA update only a small number of additional parameters, making fine-tuning accessible on consumer hardware.",
    "Prompt engineering is the practice of crafting inputs to LLMs to elicit desired outputs. Key techniques include zero-shot prompting (no examples), few-shot prompting (2-5 examples in the prompt), chain-of-thought prompting (asking the model to reason step by step), and role prompting (assigning a persona to the model).",
    "RLHF (Reinforcement Learning from Human Feedback) is the technique used to align LLMs with human preferences. Human raters compare model outputs and rank them, training a reward model on these preferences. The LLM is then fine-tuned using PPO to maximize the reward model's score, improving helpfulness and safety.",
    "FAISS (Facebook AI Similarity Search) is an open-source library for efficient similarity search. It offers multiple index types: Flat (exact, slow), IVF (inverted file, fast approximate), and HNSW (hierarchical navigable small world, best accuracy/speed tradeoff). FAISS runs locally with no cost but requires self-management.",
    "The key difference between FAISS and Pinecone for production RAG systems is infrastructure management. FAISS requires you to handle scaling, persistence, and deployment yourself, making it suitable for small to medium datasets or research. Pinecone is a managed cloud service that scales automatically to billions of vectors, handles replication, and requires no infrastructure work.",
    "When choosing between FAISS and Pinecone for a production RAG system: use FAISS if you need free local deployment, have a small dataset under 1 million vectors, or need full control. Use Pinecone if you need automatic scaling, high availability, metadata filtering, or are building a production system where infrastructure management is a burden.",
]
def ingest():
    print("🚀 Starting ingestion...")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    pc         = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")

    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        print(f"📦 Creating index '{index_name}'...")
        pc.create_index(
            name      = index_name,
            dimension = 384,
            metric    = "cosine",
            spec      = ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("✅ Index created")
    else:
        print(f"✅ Index '{index_name}' already exists")

    index = pc.Index(index_name)

    print(f"\n📝 Embedding {len(DOCUMENTS)} documents...")
    vectors = []

    for i, doc in enumerate(DOCUMENTS):
        embedding = embedder.encode(doc).tolist()
        vectors.append({
            "id":       f"doc_{i}",
            "values":   embedding,
            "metadata": {"text": doc}
        })
        print(f"   [{i+1}/{len(DOCUMENTS)}] embedded")

    index.upsert(vectors=vectors)
    print(f"\n✅ Ingestion complete — {len(DOCUMENTS)} documents stored in Pinecone")
    print("You can now run main.py to query the system!")


if __name__ == "__main__":
    ingest()