import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download resources (only need to do this once)
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

# Example text
text = "NLTK makes text processing easy. Text processing with NLTK is powerful!"

# 1. Tokenize into words
words = word_tokenize(text)

# 2. Lowercase and remove stopwords/punctuation
stop_words = set(stopwords.words('english'))
filtered = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words]

# 3. Count word frequencies
freq = Counter(filtered)

print("Filtered words:", filtered)
print("Word frequencies:", freq)
