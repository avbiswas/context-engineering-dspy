
import dspy
import asyncio
from print_utils import print
from typing import List, Optional
from pydantic import BaseModel, Field
from tools import fetch_recent_news

class JokeIdea(BaseModel):
    setup: str
    contradiction: str
    punchline: str


class QueryToIdea(dspy.Signature):
    """
    You are a funny comedian and your goal is to generate a nice structure for a joke.
    You are given some sample punchlines from diverse topic ranges, you can use these punchlines to make your own jokes about the specific query.
    """

    query: str = dspy.InputField(desc="The theme of the joke")
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


class IdeaGenerator(dspy.Module):
    def __init__(self, num_samples=3):
        self.query_to_idea = dspy.ReAct(QueryToIdea,
                            tools=[fetch_recent_news],
                            max_iters=1)
        self.judge = dspy.Refine(
            module=dspy.ChainOfThought(JokeJudge),
            N=3, reward_fn=check_score_goodness, threshold=1,
        )
        
        self.query_to_idea.set_lm(
            lm=dspy.LM("openai/gpt-4.1-mini", temperature=1)
        )
        self.judge.set_lm(
            lm=dspy.LM("openai/gpt-4.1-mini", temperature=1)
        )

        self.num_samples = num_samples
        
    async def acall(self, query: str) -> JokeIdea:

        joke_ideas = await asyncio.gather(
            *[self.query_to_idea.acall(query=query) for _ in range(self.num_samples)]
        )

        print("Generated Joke Ideas: \n", joke_ideas)

        judge_score = self.judge(joke_idea=joke_ideas).joke_ratings
        print("Judge Score for each: ", judge_score)

        best_joke_idea_idx = judge_score.index(1)
        selected_joke_idea = joke_ideas[best_joke_idea_idx]
        print("Selected Joke Idea: \n", selected_joke_idea)
        
        return selected_joke_idea.joke_idea

async def main():
    joke_generator = QueryToIdea()
    joke = await joke_generator.acall(
        query="Write a joke about AI that has to do with them turning rogue."
    )

    print("---")
    print(joke)


if __name__ == "__main__":
    asyncio.run(main())
