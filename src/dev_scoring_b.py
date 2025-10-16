import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

INPUT_PATH = Path("/app/input")
# INPUT_PATH = Path("/tmp/input")
reference_dir = INPUT_PATH / "ref"
prediction_dir = INPUT_PATH /  "res"
# score_dir = Path("/tmp/output/")
score_dir = Path("/app/output/")


def evaluate(labeled_data_path, embedding_lookup):
    df = pd.read_json(labeled_data_path, lines=True)

    # Map texts to embeddings
    df["anchor_embedding"] = df["anchor_text"].map(embedding_lookup)
    df["a_embedding"] = df["text_a"].map(embedding_lookup)
    df["b_embedding"] = df["text_b"].map(embedding_lookup)

    # Look up cosine similarities
    df["sim_a"] = df.apply(
        lambda row: row["anchor_embedding"].dot(row["a_embedding"]), axis=1
    )
    df["sim_b"] = df.apply(
        lambda row: row["anchor_embedding"].dot(row["b_embedding"]), axis=1
    )

    # Predict and calculate accuracy
    df["predicted_text_a_is_closer"] = df["sim_a"] > df["sim_b"]
    accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
    return accuracy


print("Reading prediction")
if os.path.exists(prediction_dir / "track_b.jsonl"):
    print("Opening", prediction_dir / "track_b.jsonl")
    pred = pd.read_json(prediction_dir / "track_b.jsonl", lines=True)
    embs = np.array([np.array(a) for a in pred["embedding"]])
elif os.path.exists(prediction_dir / "track_b.npy"):
    print("Opening", reference_dir / "track_b.jsonl and ", prediction_dir / "track_b.npy")
    pred = pd.read_json(reference_dir / "track_b.jsonl", lines=True)
    embs = np.load(prediction_dir / "track_b.npy", allow_pickle=False)
else:
    print("Invalid submission, you must either place `track_b.jsonl` or `track_b.npy` in the root of your zip.")
    raise ValueError("No submission file found for track B")
if 10 > embs.shape[-1] or 8192 < embs.shape[-1]:
    print(f"Disallowed embedding size: {embs.shape}. The rules allow for an embedding size of 10 to 8192.")
    raise ValueError("Illegal Embedding Size")
normed = embs / np.linalg.norm(embs, axis=1).reshape((-1, 1))
sim = np.matmul(normed, normed.T)
print("Dataframe", pred)
embedding_lookup = dict(zip(pred["text"], normed))

print("Checking Accuracy")
accuracy = evaluate(reference_dir / "track_a.jsonl", embedding_lookup)

scores = {
    "accuracy": accuracy,
}
print(scores)

with open(os.path.join(score_dir, "scores.json"), "w") as score_file:
    score_file.write(json.dumps(scores))
