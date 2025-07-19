import dspy
import asyncio
import random
import pandas as pd

dspy.configure(lm=dspy.LM("openai/gpt-4.1-mini"), track_usage=True)
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

class JokeComparer(dspy.Signature):
    """Compare between two jokes - which one is funnier?"""

    joke1: str = dspy.InputField(desc="Joke - 0")
    joke2: str = dspy.InputField(desc="Joke - 1")

    verdict: int = dspy.OutputField(le=1, ge=0)

comparer = dspy.ChainOfThought(JokeComparer)

async def comparisons(joke1, joke2):
    verdict = await comparer.acall(joke1=joke1, joke2=joke2)

    print(f"\nJoke 1: {joke1} \nJoke2: {joke2} \nVerdict:{verdict}")
    return verdict.verdict

async def elo_test(data) -> pd.DataFrame:
    idx_range = [_ for _ in range(len(data))]
    picked = [0 for _ in range(len(data))]
    won = [0 for _ in range(len(data))]

    num_contests = 25

    calls = []
    pairs = []
    
    for _ in range(num_contests):
        picked_idxs = random.sample(idx_range, k=2)

        pairs.append(picked_idxs)

        joke1 = data.iloc[picked_idxs[0]]["joke"]
        joke2 = data.iloc[picked_idxs[1]]["joke"]

        verdict_job = comparisons(joke1=joke1, joke2=joke2)
        calls.append(verdict_job)

    verdicts = await asyncio.gather(*calls)

    for p, v in zip(pairs, verdicts):
        picked[p[0]] += 1
        picked[p[1]] += 1
        won[p[v]] += 1 
    
    data["picked"] = picked
    data["won"] = won
    return data

if __name__ == "__main__":
    data = pd.read_csv("evaluation_results.csv")
    annotated_data = asyncio.run(elo_test(data))
    annotated_data.to_csv("evaluation_results_elo.csv")
    
