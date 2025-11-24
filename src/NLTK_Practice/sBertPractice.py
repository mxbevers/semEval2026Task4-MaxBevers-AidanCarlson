from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "This is an example sentence",
    "Each sentence is converted to a vector",
    "The quick brown fox jumps over the lazy dog",
    "A fast, reddish-brown canine leaps over a sleepy hound"
]

embeddings = model.encode(sentences, convert_to_tensor=True)

# 4. Calculate cosine similarity between sentences
# Let's compare the first sentence with all others
cosine_scores = util.cos_sim(embeddings[0], embeddings)

# 5. Print the sentences and their similarity scores
print("Sentence:", sentences[0])
print("Similarity to other sentences:")
for i in range(len(sentences)):
    print(f"- '{sentences[i]}'\t\tScore: {cosine_scores[0][i].item():.4f}")

# You can also compare two specific sentences
sentence1 = "I love programming in Python."
sentence2 = "Python programming is my passion."
sentence3 = "The cat sat on the mat."

embedding1 = model.encode(sentence1, convert_to_tensor=True)
embedding2 = model.encode(sentence2, convert_to_tensor=True)
embedding3 = model.encode(sentence3, convert_to_tensor=True)

similarity_score = util.cos_sim(embedding1, embedding2)
print(f"\nSimilarity between '{sentence1}' and '{sentence2}': {similarity_score.item():.4f}")

similarity_score_diff = util.cos_sim(embedding1, embedding3)
print(f"Similarity between '{sentence1}' and '{sentence3}': {similarity_score_diff.item():.4f}")