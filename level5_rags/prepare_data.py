import pandas as pd

def prepare_jokes():
    """
    Reads jokes from the original CSV and saves them to a text file,
    creating a single source of truth.
    """
    df = pd.read_csv("data/shortjokes.csv")
    jokes = df["Joke"].tolist()

    with open("data/jokes.txt", "w") as f:
        for joke in jokes:
            f.write(joke + "\n")

if __name__ == "__main__":
    prepare_jokes()
    print("Jokes have been extracted and saved to data/jokes.txt")

