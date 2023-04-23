import numpy as np

def char_to_index(char, alphabet="abcdefghijklmnopqrstuvwxyz"):
    return alphabet.index(char.lower())

def one_hot_vector(index, vector_length):
    vector = np.zeros(vector_length)
    vector[index] = 1
    return vector

def word_to_vectors(word, alphabet="abcdefghijklmnopqrstuvwxyz"):
    word_vectors = []
    for char in word:
        if char.lower() in alphabet:
            index = char_to_index(char, alphabet)
            vector = one_hot_vector(index, len(alphabet))
            word_vectors.append(vector)
    return np.array(word_vectors)

if __name__ == "__main__":
    word = "hello"
    word_vectors = word_to_vectors(word)
    print(f"Word: {word}")
    print("Vectors:")
    print(word_vectors)
