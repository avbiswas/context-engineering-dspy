import dspy
import asyncio
import os

from print_utils import time_it

dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
dspy.configure(lm=dspy.LM("openai/gpt-4.1-nano"))
predict = dspy.Predict("question->answer")

async def main():
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)

@time_it
async def test():
    await asyncio.gather(*[main() for _ in range(10)])
    # for _ in range(10):
    #     await main()

if __name__ == "__main__":
    asyncio.run(test())
