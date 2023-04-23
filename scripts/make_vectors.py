import numpy as np
from termcolor import colored
import argparse
import re

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

def print_colored_vectors(word_vectors):
    for row in word_vectors:
        colored_row = ""
        for val in row:
            if val == 1:
                colored_row += colored(str(int(val)), "green")
            else:
                colored_row += colored(str(int(val)), "red")
            colored_row += " "
        print(colored_row)

def main(input_file=None, word=None):
    if input_file:
        with open(input_file, 'r') as file:
            text = file.read()
            words = re.findall(r'\b\w+\b', text)
            for word in words:
                word_vectors = word_to_vectors(word)
                print(f"Word: {word}")
                print("Vectors:")
                print_colored_vectors(word_vectors)
                print()
    elif word:
        word_vectors = word_to_vectors(word)
        print(f"Word: {word}")
        print("Vectors:")
        print_colored_vectors(word_vectors)
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert words to one-hot encoded vectors.')
    parser.add_argument('-f', '--file', type=str, help='Path to the input file')
    parser.add_argument('-w', '--word', type=str, help='Convert a single word into vector data.')
    args = parser.parse_args()

    if args.file is None and args.word is None:
        parser.print_help()
        exit()

    main(input_file=args.file, word=args.word)
