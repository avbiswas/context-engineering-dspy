import dspy
from typing import Optional

from bm25_retriever import BM25Retriever
from basic_rag import BasicEmbeddingsRAG
from rank_fusion import reciprocal_rank_fusion

from rich.console import Console

console = Console()


class HypotheticalDoc(dspy.Signature):
    """
    Given a query, generate hypothetical documents to search a database of one-liner jokes.
    """

    query: str = dspy.InputField(desc="User wants to fetch jokes related to this topic")
    retrieved_jokes: Optional[list[str]] = dspy.InputField(
        desc="Jokes previously retrieved from the db. Use these to further tune your search."
    )

    hypothetical_bm25_query: str = dspy.OutputField(
        desc="sentence to query to retrieve more jokes about the query from the database"
    )
    hypothetical_semantic_query: str = dspy.OutputField(
        desc="sentence to search with cosine similarity"
    )


class MultiHopHydeSearch(dspy.Module):
    def __init__(self, texts, embs, n_hops=3, k=10):
        self.predict = dspy.ChainOfThought(HypotheticalDoc)
        self.predict.set_lm(lm=dspy.LM("gemini/gemini-2.0-flash"))
        self.embedding_retriever = BasicEmbeddingsRAG(texts, embs)
        self.bm25_retriever = BM25Retriever(texts)

        self.n_hops = n_hops
        self.k = k

    def forward(self, query):
        retrieved_jokes = []
        all_jokes = []
        for _ in range(self.n_hops):

            new_query = self.predict(query=query, retrieved_jokes=retrieved_jokes)

            print(new_query)

            embedding_lists = self.embedding_retriever.get_nearest(
                new_query.hypothetical_semantic_query
            )
            bm25_lists = self.bm25_retriever.get_nearest(
                new_query.hypothetical_bm25_query
            )
            lists = [embedding_lists, bm25_lists]
            retrieved_jokes = reciprocal_rank_fusion(lists, k=self.k)
            all_jokes.extend(retrieved_jokes)

        return dspy.Prediction(jokes=all_jokes)


if __name__ == "__main__":
    import numpy as np

    query = "men"
    run_id = "1"
    k = 5
    n_hops = 3

    print(f"loading data for run_id: {run_id}...")
    with open(f"data/jokes_{run_id}.txt", "r") as f:
        jokes = [line.strip() for line in f.readlines()]
    embeddings = np.load(f"data/embeddings_{run_id}.npy")
    print("data loaded.")

    hyde = MultiHopHydeSearch(texts=jokes, embs=embeddings, n_hops=n_hops, k=k)

    retrieved_jokes = hyde(query=query).jokes

    console.print(retrieved_jokes)
