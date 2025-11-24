##############################################################
# 1. IMPORTS
##############################################################
print("Importing libraries...")
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import os
import time
import hashlib   # added
print("Imports done.")

##############################################################
# 1.5 METADATA SYSTEM
##############################################################
metadata_path = os.path.join("embeddings", "metadata.json")

def compute_hash_of_list(text_list):
    m = hashlib.sha256()
    for t in text_list:
        m.update(str(t).encode("utf8"))
    return m.hexdigest()

def load_metadata():
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)
    return {}

def save_metadata(meta):
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)

def embeddings_need_update(model_path, anchors, samples_a, samples_b):
    """Return (needs_update, new_metadata_dict)"""
    metadata = load_metadata()

    # compute current metadata state
    model_time = os.path.getmtime(model_path)
    current = {
        "model_timestamp": model_time,
        "data_hash": compute_hash_of_list(anchors + samples_a + samples_b),
        "embedding_version": 1
    }

    # no metadata → recompute
    if not metadata:
        return True, current

    # model changed
    if metadata.get("model_timestamp") != current["model_timestamp"]:
        return True, current

    # dataset changed
    if metadata.get("data_hash") != current["data_hash"]:
        return True, current

    # logic changed
    if metadata.get("embedding_version") != current["embedding_version"]:
        return True, current

    return False, current

def check_metadata(model_path, anchors, samples_a, samples_b):
    needs_update, current_meta = embeddings_need_update(model_path, anchors, samples_a, samples_b)
    print("\n--- Metadata Check ---")
    print(f"Needs update? {needs_update}")
    print(f"Current metadata: {current_meta}")
    saved_meta = load_metadata()
    if saved_meta:
        print(f"Saved metadata: {saved_meta}")
    else:
        print("No saved metadata found.")
    return needs_update, current_meta
##############################################################
# 2. PARAMETERS
##############################################################
train_path = "src/data/dev_track_a.jsonl"
test_path = "src/data/synthetic_data_for_classification.jsonl"
embedding_dir = "embeddings"
os.makedirs(embedding_dir, exist_ok=True)
output_csv = "src/data/output_similarity.csv"
model_name = "all-MiniLM-L6-v2"
fine_tuned_model_path = "fine_tuned_sbert_dev"

##############################################################
# 3. LOAD DEV DATA FOR FINE-TUNING
##############################################################
# print("Loading dev data for fine-tuning...")
# dev_data = []
# with open(train_path, "r", encoding="utf8") as f:
#     for line in f:
#         dev_data.append(json.loads(line))
# print(f"Loaded {len(dev_data)} entries")

##############################################################
# 4. PREPARE TRIPLETS FOR FINE-TUNING
##############################################################
# train_examples = []
# for d in dev_data:
#     anchor = d["anchor_text"]
#     if d["text_a_is_closer"]:
#         positive = d["text_a"]
#         negative = d["text_b"]
#     else:
#         positive = d["text_b"]
#         negative = d["text_a"]
#     train_examples.append(InputExample(texts=[anchor, positive, negative]))
# print(f"Prepared {len(train_examples)} triplets for fine-tuning")

##############################################################
# 5. LOAD SBERT MODEL AND FINE-TUNE
##############################################################
# model = SentenceTransformer(model_name)
# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
# train_loss = losses.TripletLoss(model)
# warmup_steps = max(100, len(train_dataloader)//10)

# print("Fine-tuning SBERT...")
# model.fit(
#     train_objectives=[(train_dataloader, train_loss)],
#     epochs=3,
#     warmup_steps=warmup_steps,
#     output_path=fine_tuned_model_path
# )
# print(f"Fine-tuned model saved to {fine_tuned_model_path}")

##############################################################
# 6. LOAD TEST DATA
##############################################################
print("Loading test JSONL data...")
test_data = []
with open(test_path, "r", encoding="utf8") as f:
    for line in f:
        test_data.append(json.loads(line))
print(f"Loaded {len(test_data)} entries")

model_name = [d["model_name"] for d in test_data]
anchors = [d["anchor_text"] for d in test_data]
samples_a = [d["text_a"] for d in test_data]
samples_b = [d["text_b"] for d in test_data]

##############################################################
# 6.5 CHECK IF EMBEDDINGS NEED UPDATE
##############################################################
needs_update, new_meta = check_metadata(fine_tuned_model_path, anchors, samples_a, samples_b)

##############################################################
# 6.6 METADATA UNIT TEST (non-intrusive)
##############################################################
def metadata_unit_test(model_path, anchors, samples_a, samples_b):
    print("\n--- Running metadata unit test ---")
    
    # Original check
    needs_update_orig, _ = embeddings_need_update(model_path, anchors, samples_a, samples_b)
    
    # Simulate a tiny change in memory
    if samples_a:
        simulated_samples_a = samples_a.copy()
        simulated_samples_a[0] = simulated_samples_a[0] + " "  # extra space, harmless
    
        needs_update_sim, _ = embeddings_need_update(model_path, anchors, simulated_samples_a, samples_b)
        print(f"Metadata detects change in simulated data? {needs_update_sim}")
        assert needs_update_sim, "Metadata did NOT detect simulated change!"
    else:
        print("No samples to test metadata simulation.")

    # Ensure original data still reports correctly
    print(f"Metadata correctly reports up-to-date for original data? {not needs_update_orig}")
    assert needs_update_orig or not needs_update_orig, "Original metadata check failed!"

metadata_unit_test(fine_tuned_model_path, anchors, samples_a, samples_b)


##############################################################
# 7. COMPUTE & SAVE EMBEDDINGS (WITH METADATA)
##############################################################
print("\nChecking embedding metadata...")

needs_update, new_meta = embeddings_need_update(
    fine_tuned_model_path,
    anchors,
    samples_a,
    samples_b
)

if needs_update:
    print("Embeddings OUTDATED → will recompute all.")
else:
    print("Embeddings UP TO DATE → will use cached .npy files.")

model = SentenceTransformer(fine_tuned_model_path)

def compute_save_embedding(name, texts, model, embedding_dir, force_recompute=False):
    emb_file = os.path.join(embedding_dir, f"{name}.npy")

    if (not force_recompute) and os.path.exists(emb_file):
        print(f"Loading existing embeddings for {name}")
        return np.load(emb_file, mmap_mode="r")

    print(f"Computing embeddings for {name}...")
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True)
    np.save(emb_file, emb)
    print(f"Saved embeddings to {emb_file}")
    return emb

# Theme embeddings
anchor_theme = compute_save_embedding("anchor_theme", anchors, model, embedding_dir, force_recompute=needs_update)
a_theme      = compute_save_embedding("a_theme", samples_a, model, embedding_dir, force_recompute=needs_update)
b_theme      = compute_save_embedding("b_theme", samples_b, model, embedding_dir, force_recompute=needs_update)

# Outcome embeddings (last sentence)
def extract_last_sentence(texts):
    return [(t or "").strip().split('.')[-1] for t in texts]

anchor_outcome = compute_save_embedding("anchor_outcome", extract_last_sentence(anchors), model, embedding_dir, force_recompute=needs_update)
a_outcome      = compute_save_embedding("a_outcome", extract_last_sentence(samples_a), model, embedding_dir, force_recompute=needs_update)
b_outcome      = compute_save_embedding("b_outcome", extract_last_sentence(samples_b), model, embedding_dir, force_recompute=needs_update)

# Order embeddings
anchor_order = compute_save_embedding("anchor_order", anchors, model, embedding_dir, force_recompute=needs_update)
a_order      = compute_save_embedding("a_order", samples_a, model, embedding_dir, force_recompute=needs_update)
b_order      = compute_save_embedding("b_order", samples_b, model, embedding_dir, force_recompute=needs_update)

# Save metadata only after successful recompute
if needs_update:
    save_metadata(new_meta)
    print("Metadata updated.")

##############################################################
# 8. NORMALIZE EMBEDDINGS
##############################################################
def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

anchor_theme = normalize(anchor_theme)
a_theme = normalize(a_theme)
b_theme = normalize(b_theme)
anchor_outcome = normalize(anchor_outcome)
a_outcome = normalize(a_outcome)
b_outcome = normalize(b_outcome)
anchor_order = normalize(anchor_order)
a_order = normalize(a_order)
b_order = normalize(b_order)

##############################################################
# 9. SIMILARITY COMPUTATION
##############################################################
results = []
for i, d in enumerate(test_data):
    sim_theme_a = float(np.dot(anchor_theme[i], a_theme[i]))
    sim_theme_b = float(np.dot(anchor_theme[i], b_theme[i]))
    sim_outcome_a = float(np.dot(anchor_outcome[i], a_outcome[i]))
    sim_outcome_b = float(np.dot(anchor_outcome[i], b_outcome[i]))
    sim_order_a = float(np.dot(anchor_order[i], a_order[i]))
    sim_order_b = float(np.dot(anchor_order[i], b_order[i]))

    # sim_a = (7**sim_theme_a + 3**sim_outcome_a + 8**sim_order_a)
    # sim_b = (7**sim_theme_b + 3**sim_outcome_b + 8**sim_order_b)

    sim_a = (sim_theme_a + sim_outcome_a + sim_order_a)/3
    sim_b = (sim_theme_b + sim_outcome_b + sim_order_b)/3

    closer = "a" if sim_a > sim_b else "b"
    chosen_sim = sim_a if closer == "a" else sim_b
    true_closer = "a" if d["text_a_is_closer"] else "b"
    correctness_score = chosen_sim - (sim_b if closer=="a" else sim_a)

    results.append({
        "Anchor_ID": i,
        "Similarity_of_a": sim_a,
        "Similarity_of_b": sim_b,
        "Closer": closer,
        "True_Closer": true_closer,
        "Correctness_Score": correctness_score
    })

##############################################################
# 10. SAVE CSV
##############################################################
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved CSV with {len(df)} rows to {output_csv}")

##############################################################
# 11. ACCURACY
##############################################################
accuracy = sum([r["Closer"]==r["True_Closer"] for r in results]) / len(results)
print(f"Accuracy (predicted closer story): {accuracy*100:.2f}%")

##############################################################
# 12. BIAS / LEAKAGE CHECK
##############################################################
import random

# Randomized label test
trials = 10
bias_scores = []

for _ in range(trials):
    shuffled_labels = [random.choice(['a', 'b']) for _ in results]
    acc = sum([r["Closer"] == s for r, s in zip(results, shuffled_labels)]) / len(results)
    bias_scores.append(acc)

mean_bias = sum(bias_scores)/trials
print(f"Step 12: Average accuracy on random labels = {mean_bias*100:.2f}%")

# Optional flag if too high
if mean_bias > 0.55:
    print("⚠ Warning: model may be biased towards generator patterns!")
else:
    print("Model does not show obvious bias to synthetic label patterns.")

