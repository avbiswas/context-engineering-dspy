import numpy as np
from vector_embedding import embed_texts


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
    import time

    query = "Plants and trees"
    run_id = "1"
    with open(f"data/jokes_{run_id}.txt", "r") as f:
        jokes = [line.strip() for line in f.readlines()]
    embeddings = np.load(f"data/embeddings_{run_id}.npy")

    basic_rag = BasicEmbeddingsRAG(jokes, embeddings)

    start_time = time.time()
    nearest = basic_rag.get_nearest(query, k=10)

    print(f"Time taken: {time.time() - start_time}")
    print(nearest)
