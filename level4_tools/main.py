import dspy
import asyncio
from idea_gen import IdeaGenerator
from joke_gen import JokeGenerator

# import mlflow
# mlflow.autolog()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("Tool calling")

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini"), temperature=1)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

idea_generator = IdeaGenerator(num_samples=5)
joke_generator = JokeGenerator(num_reflection_steps=2)

@mlflow.trace
async def main(query):
    idea = await idea_generator.acall(query=query)
    joke = await joke_generator.acall(joke_idea=idea)
    return joke


if __name__ == "__main__":
    query = input("Query: \n")
    output = asyncio.run(main(query))
    print(output)


