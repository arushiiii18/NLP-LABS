import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The quick brown foxes are jumping over the lazy dogs."

# Tokenization
tokens = word_tokenize(text)

print("Original Text:")
print(text)

print("\nTokens:")
print(tokens)

# Stopword Removal
stop_words = set(stopwords.words("english"))

filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

print("\nAfter Stopword Removal:")
print(filtered_tokens)

# Stemming
stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_tokens
]

print("\nAfter Stemming:")
print(stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word.lower())
    for word in filtered_tokens
]

print("\nAfter Lemmatization:")
print(lemmatized_words)