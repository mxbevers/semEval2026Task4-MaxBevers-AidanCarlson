import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Tokenize, lowercase, remove stopwords/punctuation."""
    tokens = word_tokenize(text)
    filtered = [w.lower() for w in tokens if w.isalpha() and w.lower() not in stop_words]
    return " ".join(filtered)

data = []
with open("src/data/sample_track_a.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

with open("cosine_similarities.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "ID", "N", "A", "B",
        "cos_sim_N_A", "cos_sim_N_B",
        "text_a_is_closer"
    ])
    
    for i, row in enumerate(data, start=1):
        ta = clean_text(row.get("text_a", ""))
        tb = clean_text(row.get("text_b", ""))
        anchor = clean_text(row.get("anchor_text", ""))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([anchor, ta, tb])
        
        cos_sim = cosine_similarity(tfidf_matrix)
        cos_n_a = cos_sim[0,1]  
        cos_n_b = cos_sim[0,2] 
        
        writer.writerow([
            f"{i}",   
            f"N{i}",  
            f"A{i}",  
            f"B{i}",  
            cos_n_a,
            cos_n_b,
            row.get("text_a_is_closer", "")
        ])
def main():
    return 0