import dspy
import asyncio
from print_utils import print
from typing import List
from pydantic import BaseModel, Field
dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini"))
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
    """
    joke_idea: JokeIdea = dspy.InputField()
    joke: str = dspy.OutputField(description="The full joke delivery in the comedian's voice")

class JokeJudge(dspy.Signature):
    """Rank each joke idea between 1-N. 
    Rank 1 is the most unique and funniest."""

    joke_idea: List[JokeIdea] = dspy.InputField()
    joke_rankings: List[int] = dspy.OutputField(description="Rank between 1, 2, 3 ... N")

class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, num_samples=5):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.judge = dspy.ChainOfThought(JokeJudge)
        self.num_samples = num_samples

    async def aforward(self, query: str):

        joke_ideas = await asyncio.gather(
            *[
                self.query_to_idea.acall(query=query) 
                for _ in range(self.num_samples)
            ]
        )

        print("Generated Joke Ideas: \n", joke_ideas)
        
            
        judge_score = self.judge(joke_idea=joke_ideas).joke_rankings
        print("Judge Score for each: ", judge_score)

        best_joke_idea_idx = judge_score.index(1)

        print("Selected Index: ", best_joke_idea_idx)
        selected_joke_idea = joke_ideas[best_joke_idea_idx]
        print("Selected Joke Idea: \n", selected_joke_idea)

        joke = self.idea_to_joke(joke_idea=selected_joke_idea)

        # Run with a different LLM
        # with dspy.context(lm=dspy.LM("gemini/gemini-1.5-pro")):
        #    joke = self.idea_to_joke(joke_idea=joke_idea)

        return joke

async def main():
    joke_generator = ConditionalJokeGenerator()
    joke = await joke_generator.acall(query="Write a joke about AI that has to do with them turning rogue.")

    print("---")
    print(joke)


if __name__ == "__main__":
    asyncio.run(main())
