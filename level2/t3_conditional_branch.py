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

class JokeJudge(dspy.Signature):
    """Is this joke idea funny"""
    joke_idea: JokeIdea = dspy.InputField()
    joke_rating: int = dspy.OutputField(description="Rating between 1 to 5", le=5, ge=1)

class ConditionalJokeGenerator(dspy.Module):
    def __init__(self, max_attempts=3, good_idea_threshold=4):
        self.query_to_idea = dspy.Predict(QueryToIdea)
        self.idea_to_joke = dspy.Predict(IdeaToJoke)
        self.judge = dspy.ChainOfThought(JokeJudge)
        self.max_attempts = max_attempts
        self.good_idea_threshold = good_idea_threshold

    def forward(self, query: str):
        for _ in range(self.max_attempts):
            print(f"--- Iteration {_ + 1} ---")
            joke_idea = self.query_to_idea(query=query)
            print(f"Joke Idea:\n{joke_idea}")
            
            judge_score = self.judge(joke_idea=joke_idea).joke_rating

            print(f"\n\n---\nJudge score: ", judge_score)

            if judge_score >= self.good_idea_threshold:
                print("Judge said it was awesome, breaking the loop")
                break
        
        joke = self.idea_to_joke(joke_idea=joke_idea)

        # Run with a different LLM
        # with dspy.context(lm=dspy.LM("gemini/gemini-1.5-pro")):
        #    joke = self.idea_to_joke(joke_idea=joke_idea)

        return joke

joke_generator = ConditionalJokeGenerator()
joke = joke_generator(query="Write a joke about AI that has to do with them turning rogue.")

print("---")
print(joke)
