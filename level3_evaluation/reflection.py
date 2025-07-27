import time
import dspy
import asyncio
import random
import pandas as pd

from print_utils import print
from typing import List, Optional
from pydantic import BaseModel, Field

dspy.configure(track_usage=True)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)


class JokeIdea(BaseModel):
    setup: str
    contradiction: str
    punchline: str


class QueryToIdea(dspy.Signature):
    """
    You are a funny comedian and your goal is to generate a nice structure for a joke.

    """

    query: str = dspy.InputField()
    joke_idea: JokeIdea = dspy.OutputField()


class IdeaToJoke(dspy.Signature):
    """
    You are a funny comedian who likes to tell stories before delivering a punchline.
    You are always funny and act on the input joke idea.
    If you are provided a draft of a joke, your goal should to make it make it funnier and more punchy.
    """

    joke_idea: JokeIdea = dspy.InputField()
    joke_draft: Optional[str] = dspy.InputField(description="An existing joke that you need to either refine, or change")
    joke: str = dspy.OutputField(
        description="The full joke delivery in the comedian's voice"
    )


class JokeJudge(dspy.Signature):
    """Rank each joke idea between 1-N.
    Rank 1 is the most unique and funniest."""

    joke_idea: List[JokeIdea] = dspy.InputField()
    joke_ratings: List[int] = dspy.OutputField(description="Rank between 1, 2, 3 ... N")


def check_score_goodness(args, pred):
    num_samples = len(args["joke_idea"])
    same_length = len(pred.joke_ratings) == num_samples
    all_ranks_present = all([(i + 1) in pred.joke_ratings for i in range(num_samples)])
    return 1 if (same_length and all_ranks_present) else 0


class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, num_samples=2, num_reflection_steps=2, 
                 temperature=0.7,
                 idea_lm="openai/gpt-4.1-mini",
                 joke_lm="openai/gpt-4o"):
        self.query_to_idea = dspy.ChainOfThought(QueryToIdea)
        self.query_to_idea.set_lm(lm=dspy.LM(idea_lm, temperature=temperature))

        self.idea_to_joke = dspy.ChainOfThought(IdeaToJoke)
        self.idea_to_joke.set_lm(lm=dspy.LM(joke_lm, temperature=temperature))
        self.judge = dspy.Refine(
            module=dspy.ChainOfThought(JokeJudge),
            N=3, reward_fn=check_score_goodness, threshold=1,
        )
        self.judge.set_lm(dspy.LM("openai/gpt-4.1-mini"))
        self.num_samples = num_samples
        self.num_reflection_steps = num_reflection_steps
        
    async def aforward(self, query: str):

        joke_ideas = await asyncio.gather(
            *[self.query_to_idea.aforward(query=query) for _ in range(self.num_samples)]
        )

        print("Generated Joke Ideas: \n", joke_ideas)

        judge_score = self.judge(joke_idea=joke_ideas).joke_ratings
        print("Judge Score for each: ", judge_score)

        best_joke_idea_idx = judge_score.index(1)
        selected_joke_idea = joke_ideas[best_joke_idea_idx]
        print("Selected Joke Idea: \n", selected_joke_idea)
        
        joke = None
        for _ in range(self.num_reflection_steps):
            joke = self.idea_to_joke(joke_idea=selected_joke_idea,
                                     joke_draft=joke)
            print(joke)
        return joke


async def main():
    # Define hyperparameters
    joke_lms = ["openai/gpt-4.1", "gemini/gemini-1.5-pro"]
    idea_lms = ["openai/gpt-4.1-mini", "gemini/gemini-2.0-flash"]
    temperatures = [0.2, 0.7, 1.2]
    num_samples = [2, 3]
    num_reflection_steps = [1, 3]
    
    # Number of random combinations to test
    num_trials = 10

    # List to store results
    results = []

    for i in range(num_trials):
        # Randomly select hyperparameters
        selected_joke_lm = random.choice(joke_lms)
        selected_idea_lm = random.choice(idea_lms)
        selected_temperature = random.choice(temperatures)
        selected_num_samples = random.choice(num_samples)
        selected_num_reflection_steps = random.choice(num_reflection_steps)

        print(f"Trial {i+1}/{num_trials}: Running with: joke_lm={selected_joke_lm}, idea_lm={selected_idea_lm}, temperature={selected_temperature}, num_samples={selected_num_samples}, num_reflection_steps={selected_num_reflection_steps}")

        # Instantiate the generator with selected hyperparameters
        joke_generator = ConditionalJokeGenerator(
            joke_lm=selected_joke_lm,
            idea_lm=selected_idea_lm,
            temperature=selected_temperature,
            num_samples=selected_num_samples,
            num_reflection_steps=selected_num_reflection_steps
        )

        start_time = time.time()
        
        try:
            joke = await joke_generator.aforward(
                query="Write a joke about AI that has to do with them turning rogue."
            )
            latency = time.time() - start_time
            results.append({
                "joke_lm": selected_joke_lm,
                "idea_lm": selected_idea_lm,
                "temperature": selected_temperature,
                "num_samples": selected_num_samples,
                "num_reflection_steps": selected_num_reflection_steps,
                "joke": joke.joke,
                "latency": latency
            })
            print(f"Finished in {latency:.2f} seconds.")

        except Exception as e:
            print(f"An error occurred: {e}")
            latency = time.time() - start_time
            results.append({
                "joke_lm": selected_joke_lm,
                "idea_lm": selected_idea_lm,
                "temperature": selected_temperature,
                "num_samples": selected_num_samples,
                "num_reflection_steps": selected_num_reflection_steps,
                "joke": f"ERROR: {e}",
                "latency": latency
            })

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Print the DataFrame
    print(df)

    # Save the DataFrame to a CSV file
    df.to_csv("evaluation_results.csv", index=False)



if __name__ == "__main__":
    asyncio.run(main())
