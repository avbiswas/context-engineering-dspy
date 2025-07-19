import dspy
from print_utils import print
from typing import Optional
from pydantic import BaseModel, Field
dspy.configure(lm=dspy.LM("gemini/gemini-2.0-flash"))

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

class JokeGenerator(dspy.Module):
    def __init__(self):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)

    def forward(self, query: str):
        joke_idea = self.query_to_idea(query=query)
        print(f"Joke Idea:\n{joke_idea}")

        joke = self.idea_to_joke(joke_idea=joke_idea)
        print(f"Joke:\n{joke}")
        return joke

joke_generator = JokeGenerator()
joke = joke_generator(query="Write a joke about AI that has to do with them turning rogue.")

print("---")
print(joke.joke)
