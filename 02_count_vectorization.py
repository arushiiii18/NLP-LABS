from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample corpus
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog barks at the quick brown fox",
    "the fox and the dog are friends"
]

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Convert text into count vectors
X = vectorizer.fit_transform(corpus)

# Get vocabulary
vocabulary = vectorizer.get_feature_names_out()

print("Vocabulary:")
print(vocabulary)

# Display Count Vectorization matrix
count_matrix = pd.DataFrame(
    X.toarray(),
    columns=vocabulary
)

print("\nCount Vectorization Matrix:")
print(count_matrix)