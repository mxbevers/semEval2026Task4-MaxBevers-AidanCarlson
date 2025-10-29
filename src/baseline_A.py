import random
from enum import Enum

import pandas as pd
from pydantic import BaseModel

class ResponseEnum(str, Enum):
    A = "A"
    B = "B"

class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum

baseline = "random"  # or "openai"
df = pd.read_json("src/data/dev_track_a.jsonl", lines=True)

df["predicted_text_a_is_closer"] = df.apply(
    lambda row: random.choice([True, False]), axis=1
)
accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
print(f"Accuracy: {accuracy:.3f}")


df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

open("src/output/track_a.jsonl", "w").write(df.to_json(orient='records', lines=True))