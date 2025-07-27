import numpy as np
import time

from basic_rag import BasicEmbeddingsRAG
from bm25_retriever import BM25Retriever


def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    # Calculate RRF scores
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1 / (k + rank + 1)

    # Sort documents by their fused score in descending order
    sorted_docs = sorted(scores.keys(), key=lambda doc: scores[doc], reverse=True)
    sorted_docs = sorted_docs[:k]
    return sorted_docs


if __name__ == "__main__":
    query = "AI going rogue"
    run_id = "1"
    top_k = 10

    print(f"Loading data for run_id: {run_id}...")
    with open(f"data/jokes_{run_id}.txt", "r") as f:
        jokes = [line.strip() for line in f.readlines()]
    embeddings = np.load(f"data/embeddings_{run_id}.npy")
    print("Data loaded.")

    # 1. Initialize both retrievers
    print("\nInitializing retrievers...")
    vector_rag = BasicEmbeddingsRAG(jokes, embeddings)
    bm25_retriever = BM25Retriever(jokes)
    print("Retrievers initialized.")

    # 2. Get ranked lists from each retriever
    print(f"\nQuerying for: '{query}'")
    start_time = time.time()
    vector_results = vector_rag.get_nearest(query, k=top_k)
    vector_time = time.time() - start_time

    start_time = time.time()
    bm25_results = bm25_retriever.get_nearest(query, k=top_k)
    bm25_time = time.time() - start_time

    print(f"\n--- Vector Search Results (took {vector_time:.4f}s) ---")
    for i, res in enumerate(vector_results):
        print(f"{i+1}. {res}")

    print(f"\n--- BM25 Search Results (took {bm25_time:.4f}s) ---")
    for i, res in enumerate(bm25_results):
        print(f"{i+1}. {res}")

    # 3. Perform Rank Fusion
    fused_results = reciprocal_rank_fusion([vector_results, bm25_results])

    print(f"\n--- Fused and Re-ranked Results (Top {top_k}) ---")
    for i, res in enumerate(fused_results[:top_k]):
        print(f"{i+1}. {res}")
