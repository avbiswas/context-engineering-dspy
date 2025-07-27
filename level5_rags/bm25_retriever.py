import time
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, texts):
        self.texts = texts
        # Tokenize the texts (simple split is used for this example)
        tokenized_corpus = [doc.split(" ") for doc in texts]

        # Create the BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

    def get_nearest(self, query: str, k: int = 10):
        """
        Retrieves the top k most relevant documents for a given query
        using BM25 lexical search.
        """
        # Tokenize the query
        tokenized_query = query.split(" ")

        # Get the top n documents
        top_k_docs = self.bm25.get_top_n(tokenized_query, self.texts, n=k)

        return top_k_docs


if __name__ == "__main__":
    query = "Cell phones"
    run_id = "1"

    print(f"Loading data for run_id: {run_id}...")
    with open(f"data/jokes_{run_id}.txt", "r") as f:
        jokes = [line.strip() for line in f.readlines()]
    print("Data loaded.")

    # --- BM25 Retriever ---
    print("\n--- Using BM25Retriever (Lexical Search) ---")
    bm25_retriever = BM25Retriever(jokes)

    start_time = time.time()
    nearest_bm25 = bm25_retriever.get_nearest(query, k=10)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.6f} seconds")
    print(nearest_bm25)
