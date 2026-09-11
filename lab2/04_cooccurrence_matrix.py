import numpy as np
import pandas as pd

sentence = "the quick brown fox jumps over the lazy dog"

words = sentence.split()

vocabulary = sorted(set(words))

word_to_index = {
    word: i for i, word in enumerate(vocabulary)
}

matrix = np.zeros(
    (len(vocabulary), len(vocabulary)),
    dtype=int
)

window_size = 2

for i, word in enumerate(words):

    start = max(0, i - window_size)
    end = min(len(words), i + window_size + 1)

    for j in range(start, end):

        if i != j:
            context_word = words[j]

            matrix[
                word_to_index[word],
                word_to_index[context_word]
            ] += 1

df = pd.DataFrame(
    matrix,
    index=vocabulary,
    columns=vocabulary
)

print("Vocabulary:")
print(vocabulary)

print("\nCo-occurrence Matrix:")
print(df)