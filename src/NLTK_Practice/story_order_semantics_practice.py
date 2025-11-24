import json
import csv
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from difflib import SequenceMatcher

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def extract_events(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens, lang='eng')

    lemmatizer = nltk.WordNetLemmatizer()
    events = []

    for word, tag in tags:
        wn_pos = get_wordnet_pos(tag)
        if wn_pos == wn.VERB:
            lemma = lemmatizer.lemmatize(word.lower(), wn.VERB)
            events.append({"lemma": lemma, "pos": tag})

    return events


def semantic_similarity(word1, word2):
    syn1 = wn.synsets(word1, pos=wn.VERB)
    syn2 = wn.synsets(word2, pos=wn.VERB)
    if not syn1 or not syn2:
        return 0.0
    sim = syn1[0].path_similarity(syn2[0])
    return sim if sim else 0.0


def semantic_order_similarity(events1, events2, threshold=0.6):
    seq1 = []
    seq2 = []

    for e1 in events1:
        best_sim = 0
        best_verb = e1["lemma"]
        for e2 in events2:
            sim = semantic_similarity(e1["lemma"], e2["lemma"])
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_verb = e2["lemma"]
        seq1.append(best_verb)

    for e2 in events2:
        best_sim = 0
        best_verb = e2["lemma"]
        for e1 in events1:
            sim = semantic_similarity(e1["lemma"], e2["lemma"])
            if sim > best_sim:
                best_sim = sim
                if sim >= threshold:
                    best_verb = e1["lemma"]
        seq2.append(best_verb)

    return SequenceMatcher(None, seq1, seq2).ratio()


def semantic_content_similarity(events1, events2):
    if not events1 or not events2:
        return 0.0

    total_sim = 0
    for e1 in events1:
        best_sim = max(semantic_similarity(e1["lemma"], e2["lemma"]) for e2 in events2)
        total_sim += best_sim
    return total_sim / len(events1)

data = []
with open("src/data/sample_track_a.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open("src/sim_csv_data/semantic_order_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "i", "order_sim_a", "order_sim_b", "content_sim_a", "content_sim_b",
        "true_a_closer", "predicted_a_closer"
    ])

    for idx, item in enumerate(data):
        anchor_text = item.get("anchor_text", "")
        text_a = item.get("text_a", "")
        text_b = item.get("text_b", "")

        anchor_events = extract_events(anchor_text)
        a_events = extract_events(text_a)
        b_events = extract_events(text_b)

        order_sim_a = semantic_order_similarity(anchor_events, a_events)
        order_sim_b = semantic_order_similarity(anchor_events, b_events)
        content_sim_a = semantic_content_similarity(anchor_events, a_events)
        content_sim_b = semantic_content_similarity(anchor_events, b_events)

        total_sim_a = 0.3 * order_sim_a + 0.7 * content_sim_a
        total_sim_b = 0.3 * order_sim_b + 0.7 * content_sim_b

        predicted_a_closer = total_sim_a >= total_sim_b
        true_a_closer = item.get("text_a_is_closer", None)

        writer.writerow([
            idx, round(order_sim_a, 3), round(order_sim_b, 3),
            round(content_sim_a, 3), round(content_sim_b, 3),
            true_a_closer, predicted_a_closer
        ])