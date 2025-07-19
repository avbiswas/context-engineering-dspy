
import dspy
import asyncio
from print_utils import print
from typing import List, Optional
from idea_gen import JokeIdea
from pydantic import BaseModel, Field

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini"), temperature=1)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)


class IdeaToJoke(dspy.Signature):
    """
    You are a funny comedian who likes to tell stories before delivering a punchline.
    You are always funny and act on the input joke idea.
    You are also provided some punch-lines from a joke database - this is just to help you get some thematic ideas. 
    """

    joke_idea: JokeIdea = dspy.InputField()
    punchlines: list[str] = dspy.InputField(desc="a list of punchlines from other jokes which you may want to take inspiration from")

    punch_line_ids: list[int] = dspy.OutputField(desc="which punchline idxs you used for inspiration")
    plan: str = dspy.OutputField(desc="how you will use the punchlines, and the joke idea together to form a joke") 
    joke: str = dspy.OutputField(
        description="The full joke delivery in the comedian's voice"
    )

class JokeGenerator(dspy.Module):
    def __init__(self):
        self.idea_to_joke = dspy.ChainOfThought(IdeaToJoke)
        self.idea_to_joke.set_lm(lm=dspy.LM("openai/gpt-4.1", temperature=0.7))
        
    async def acall(self, joke_idea: JokeIdea, punchlines: list[str]):

        joke = self.idea_to_joke(joke_idea=joke_idea,
                                punchlines=punchlines)
        return dspy.Prediction(
            inspiration=[punchlines[idx] for idx in joke.punch_line_ids],
            plan=joke.plan,
            joke=joke.joke
        )

