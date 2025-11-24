import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import csv

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(text)
    filtered = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stop_words]
    return filtered

def map_to_hypernyms(tokens):
    hypernym_tokens = []
    for token in tokens:
        synsets = wordnet.synsets(token)
        if synsets:
            hypernyms = synsets[0].hypernyms()
            if hypernyms:
                hypernym_tokens.append(hypernyms[0].name().split('.')[0])
            else:
                hypernym_tokens.append(token)
        else:
            hypernym_tokens.append(token)
    return hypernym_tokens

def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    if union == 0:
        return 0.0
    return intersection / union

data = []
with open("src/data/sample_track_a.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open("src/sim_csv_data/jaccard_similarities.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["i", "n","a", "b", "true_a_closer", "estimate_a_closer"])

    for idx, item in enumerate(data):
        anchor_text = item.get("anchor_text", "")
        text_a = item.get("text_a", "")
        text_b = item.get("text_b", "")

        anchor_tokens = set(map_to_hypernyms(clean_text(anchor_text)))
        a_tokens = set(map_to_hypernyms(clean_text(text_a)))
        b_tokens = set(map_to_hypernyms(clean_text(text_b)))

        sim_n = jaccard_similarity(anchor_tokens, anchor_tokens)
        sim_a = jaccard_similarity(anchor_tokens, a_tokens)
        sim_b = jaccard_similarity(anchor_tokens, b_tokens)

        if(sim_a >= sim_b):
            a_closer = True
        else:
            a_closer = False

        writer.writerow([idx, sim_n,sim_a, sim_b, item.get("text_a_is_closer"),a_closer])

print("\n Stored")
