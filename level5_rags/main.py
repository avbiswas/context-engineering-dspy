import numpy as np
import dspy
import asyncio
from idea_gen import IdeaGenerator
from joke_gen import JokeGenerator
from hyde import MultiHopHydeSearch

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini"), temperature=1)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

idea_generator = IdeaGenerator(num_samples=3)
joke_generator = JokeGenerator()

run_id = "1"
with open(f"data/jokes_{run_id}.txt", "r") as f:
    jokes = [line.strip() for line in f.readlines()]
embeddings = np.load(f"data/embeddings_{run_id}.npy")

retriever = MultiHopHydeSearch(jokes, embeddings, n_hops=2, k=5)


async def main(query):
    idea = await idea_generator.acall(query=query)

    search_query = f"""
query={query}
setup={idea.setup}
punchline={idea.punchline}
        """
    punchlines = retriever(query=search_query).jokes
    joke = await joke_generator.acall(joke_idea=idea, punchlines=punchlines)
    return joke


if __name__ == "__main__":
    query = input("Query: \n")
    
    # query = "OpenAI Agents"
    output = asyncio.run(main(query))
    print(output)


