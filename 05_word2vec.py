from gensim.models import Word2Vec

sentences = [
    ["the", "quick", "brown", "fox"],
    ["the", "lazy", "dog"],
    ["the", "fox", "and", "the", "dog"],
    ["the", "quick", "dog"]
]

model = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=2,
    min_count=1,
    workers=4,
    epochs=100
)

print("Vocabulary:")
print(model.wv.index_to_key)

print("\nVector for 'fox':")
print(model.wv["fox"])

print("\nMost Similar Words to 'fox':")

for word, similarity in model.wv.most_similar("fox", topn=5):
    print(word, ":", round(similarity, 4))