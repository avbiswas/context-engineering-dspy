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
    draft_joke: Optional[str] = dspy.InputField(description="a draft joke")
    feedback: Optional[str] = dspy.InputField(description="feedback on the draft joke")
    joke: str = dspy.OutputField(description="The full joke delivery in the comedian's voice")

class Refinement(dspy.Signature):
    """
    Given a joke, is it funny? If not, suggest a change.
    """
    joke_idea: JokeIdea = dspy.InputField()
    joke: str = dspy.InputField()
    feedback: str = dspy.OutputField()

class IterativeJokeGenerator(dspy.Module):
    def __init__(self, n_attempts: int = 3):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.refinement = dspy.ChainOfThought(Refinement)
        self.n_attempts = n_attempts

    def forward(self, query: str):
        joke_idea = self.query_to_idea(query=query)
        print(f"Joke Idea:\n{joke_idea}")
        
        draft_joke = None
        feedback = None

        for _ in range(self.n_attempts):
            print(f"--- Iteration {_ + 1} ---")

            joke = self.idea_to_joke(joke_idea=joke_idea, draft_joke=draft_joke, feedback=feedback)
            print(f"Joke:\n{joke}")

            feedback = self.refinement(joke_idea=joke_idea, joke=joke)
            print(f"Feedback:\n{feedback}")

            draft_joke = joke
            feedback = feedback.feedback


        return joke

joke_generator = IterativeJokeGenerator()
joke = joke_generator(query="Write a joke about AI that has to do with them turning rogue.")

print("---")
print(joke.joke)
