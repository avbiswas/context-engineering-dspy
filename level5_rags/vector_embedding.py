import numpy as np
import pandas as pd
import uuid
import torch
from transformers import DistilBertModel, DistilBertTokenizer

device = torch.device("mps")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.to(device)


def embed_texts(texts):
    encoded_input = tokenizer(texts, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


if __name__ == "__main__":
    import time
    from tqdm import tqdm

    data = pd.read_csv("data/shortjokes.csv")
    jokes = data["Joke"].values
    jokes = jokes[:50000]

    # Define batch size
    batch_size = 512

    all_embeddings = []
    # Process texts in batches
    for i in tqdm(range(0, len(jokes), batch_size), desc="Generating embeddings"):
        batch_texts = jokes[i : i + batch_size].tolist()
        batch_embeddings = embed_texts(batch_texts)
        all_embeddings.append(batch_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)

    run_id = "1"

    print(f"Total embeddings generated: {len(embeddings)}")

    np.save(f"data/embeddings_{run_id}.npy", embeddings)

    with open(f"data/jokes_{run_id}.txt", "w") as f:
        for joke in jokes:
            f.write(joke + "\n")

    print(f"Embeddings and jokes saved with run ID: {run_id}")
