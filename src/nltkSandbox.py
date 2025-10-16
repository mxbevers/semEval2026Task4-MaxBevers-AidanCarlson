import pandas as pd    
import nltk
from nltk.corpus import treebank
import os
import sys
try:
	import tkinter as _tkinter
except Exception:
	_tkinter = None

# NLTK keeps library small by making you download these packages seperately:
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")
nltk.download('treebank')

jsonObj = pd.read_json(path_or_buf='src/data/sample_track_a.jsonl', lines=True)

df = pd.read_json("src/data/sample_track_a.jsonl", lines=True)
#print(df.head())

text = df.loc[0, 'anchor_text']
print(text[0:60])
short_text = text[0:60]

tokens = nltk.word_tokenize(short_text)
print("tokenization:")
print(tokens)
print("------------------------")

tagged = nltk.pos_tag(tokens)
print("tagged parts of speech:")
print(tagged)
print("------------------------")

entities = nltk.chunk.ne_chunk(tagged)
print("named entities:")
print(entities)
print("------------------------")

print("parse tree:")
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()


#cite NLTK if publishing ig:
# Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.