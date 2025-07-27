from print_utils import print
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# import mlflow
#
# mlflow.dspy.autolog()
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# mlflow.set_experiment("DSPy Expt")
#

class JokeSignature(dspy.Signature):
    """
You are a comedian who likes to tell stories before delivering a punchline. You are always funny.
    """
    query: str = dspy.InputField()
    setup: str = dspy.OutputField()
    punchline: str = dspy.OutputField()
    contradiction: str = dspy.OutputField()
    delivery: str = dspy.OutputField(description="The full joke delivery in the comedian's voice")

joke_generator = dspy.Predict(JokeSignature)

joke = joke_generator(query="Write a joke about AI that has to do with them turning rogue.")
print(joke)
