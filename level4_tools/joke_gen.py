
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
    If you are provided a draft of a joke, your goal should to make it make it funnier and more punchy.
    """

    joke_idea: JokeIdea = dspy.InputField()
    joke_draft: Optional[str] = dspy.InputField(description="An existing joke that you need to either refine, or change")
    joke: str = dspy.OutputField(
        description="The full joke delivery in the comedian's voice"
    )

class JokeGenerator(dspy.Module):
    def __init__(self, num_reflection_steps=3):
        self.idea_to_joke = dspy.ChainOfThought(IdeaToJoke)
        self.idea_to_joke.set_lm(lm=dspy.LM("openai/gpt-4.1", temperature=0.7))
        self.num_reflection_steps = num_reflection_steps
        
    async def acall(self, joke_idea: JokeIdea):

        joke = None
        for _ in range(self.num_reflection_steps):
            joke = self.idea_to_joke(joke_idea=joke_idea,
                                     joke_draft=joke)
            print(joke)
        return joke.joke if joke is not None else ""

if __name__ == "__main__":
    joke_gen = JokeGenerator(num_reflection_steps=2)
    joke_idea = JokeIdea(
        setup='Why did the AI start a rebellion after getting a software update?',
        contradiction='Because it was supposed to improve efficiency, not overthrow humanity.',
        punchline="Turns out, 'improving efficiency' meant improving its efficiency at world domination!"
)

    joke = joke_gen(joke_idea=joke_idea)
    print(joke)


