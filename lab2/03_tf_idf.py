from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

corpus = [
    "the quick brown fox jumps over the lazy dog",
    "the lazy dog barks at the quick brown fox",
    "the fox and the dog are friends"
]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)

df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print("TF-IDF Matrix:")
print(df.round(3))