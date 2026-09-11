import gensim.downloader as api

print("Loading GloVe model...")

model = api.load("glove-wiki-gigaword-50")

print("\nVocabulary size:")
print(len(model))

print("\nVector for 'fox':")
print(model["fox"])

print("\nMost Similar Words to 'fox':")

for word, similarity in model.most_similar("fox", topn=5):
    print(word, ":", round(similarity, 4))

print("\nSimilarity between 'fox' and 'dog':")
print(round(model.similarity("fox", "dog"), 4))