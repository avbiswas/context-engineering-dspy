import numpy as np
from annoy import AnnoyIndex
import time

from vector_embedding import embed_texts


class AnnoyRAG:
    def __init__(self, texts, embeddings, num_trees=10):
        self.texts = texts
        self.embedding_dim = embeddings.shape[1]

        # Normalize embeddings for angular distance
        normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        # Create and build the Annoy index
        self.index = AnnoyIndex(self.embedding_dim, "angular")
        for i, vec in enumerate(normalized_embeddings):
            self.index.add_item(i, vec)
        self.index.build(num_trees)

    def get_nearest(self, query: str, k: int = 10):
        # Embed and normalize the query
        query_emb = embed_texts([query])
        normalized_query_emb = query_emb / np.linalg.norm(
            query_emb, axis=1, keepdims=True
        )

        # Get nearest neighbors
        nearest_indices = self.index.get_nns_by_vector(normalized_query_emb[0], k)

        return [self.texts[i] for i in nearest_indices]


class BasicEmbeddingsRAG:
    def __init__(self, texts, embeddings):
        self.texts = texts
        # Normalize embeddings for cosine similarity
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def get_nearest(self, query: str, k: int = 10):
        query_emb = embed_texts([query])
        # Normalize query embedding
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Calculate cosine similarity
        similarity = np.dot(query_emb, self.embeddings.T).flatten()

        # Get top k indices, sorted by similarity
        topk_indices_unsorted = np.argpartition(similarity, -k)[-k:]
        topk_indices_sorted = sorted(
            topk_indices_unsorted, key=lambda i: similarity[i], reverse=True
        )

        return [self.texts[i] for i in topk_indices_sorted]


if __name__ == "__main__":
    query = "AI is rogue"
    run_id = "1"

    print(f"Loading data for run_id: {run_id}...")
    with open(f"data/jokes_{run_id}.txt", "r") as f:
        jokes = [line.strip() for line in f.readlines()]
    embeddings = np.load(f"data/embeddings_{run_id}.npy")
    print("Data loaded.")

    # --- Annoy RAG ---
    print("\n--- Using AnnoyRAG ---")
    annoy_rag = AnnoyRAG(jokes, embeddings)

    start_time = time.time()
    nearest_annoy = annoy_rag.get_nearest(query, k=10)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print(nearest_annoy)
    print("-" * 20)

    # --- Basic RAG for comparison ---
    print("\n--- Using BasicEmbeddingsRAG (Exact Search) ---")
    basic_rag = BasicEmbeddingsRAG(jokes, embeddings)

    start_time = time.time()
    nearest_basic = basic_rag.get_nearest(query, k=10)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print(nearest_basic)
