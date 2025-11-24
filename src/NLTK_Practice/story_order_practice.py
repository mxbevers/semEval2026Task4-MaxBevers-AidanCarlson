import json
import nltk
from nltk import word_tokenize, pos_tag
from difflib import SequenceMatcher
import csv

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

def extract_events(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens, lang='eng')
    
    events = []
    current_event = []

    for word, tag in tags:
        if tag.startswith('VB'):  # Verb
            if current_event:
                events.append(" ".join(current_event))
            current_event = [word]
        elif tag.startswith('NN') and current_event:
            current_event.append(word)
    if current_event:
        events.append(" ".join(current_event))
    
    return events


def event_order_similarity(events1, events2):
    sm = SequenceMatcher(None, events1, events2)
    return sm.ratio()


data = []
with open("src/data/sample_track_a.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))


with open("src/sim_csv_data/event_order_similarities.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["i", "sim_a", "sim_b", "true_a_closer", "estimate_a_closer"])

    for idx, item in enumerate(data):
        anchor_text = item.get("anchor_text", "")
        text_a = item.get("text_a", "")
        text_b = item.get("text_b", "")

        anchor_events = extract_events(anchor_text)
        a_events = extract_events(text_a)
        b_events = extract_events(text_b)

        sim_a = event_order_similarity(anchor_events, a_events)
        sim_b = event_order_similarity(anchor_events, b_events)

        a_closer = sim_a >= sim_b

        writer.writerow([idx, sim_a, sim_b, item.get("text_a_is_closer"), a_closer])