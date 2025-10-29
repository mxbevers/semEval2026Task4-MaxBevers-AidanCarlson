import sys
import pandas as pd
import torch
import numpy as np


def evaluate(labeled_data_path, embedding_lookup):
    df = pd.read_json(labeled_data_path, lines=True)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


baseline = "random" # or "sbert"
data = pd.read_json("src/data/dev_track_b.jsonl", lines=True)

if baseline == "random":
    embeddings = torch.rand((len(data), 512))
else:
    sys.exit("Invalid baseline")

embedding_lookup = dict(zip(data["text"], embeddings))
accuracy = evaluate("data/dev_track_a.jsonl", embedding_lookup)
print(f"Accuracy: {accuracy:.3f}")

np.save("src/output/track_b.npy", embeddings)