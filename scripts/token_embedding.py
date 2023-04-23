
# Imports necessary packages such as numpy, os, and argparse.
# Defines a function load_glove_embeddings() that loads the pre-trained GloVe embeddings from a file and returns them as a KeyedVectors object.
# Defines a function get_embedding() that takes a token and a pre-trained embedding model as input and returns the embedding vector for the token if it exists in the model, or a zero vector if it does not.
# Defines a function create_embedding_matrix() that takes a tokenizer and a pre-trained embedding model as input and returns a matrix where each row corresponds to a token in the tokenizer and the values are their corresponding embedding vectors.
# Defines the main() function that uses argparse to parse command-line arguments for the paths to the pre-trained embedding file, tokenizer file, and output file.
# Loads the pre-trained embedding model using the load_glove_embeddings() function.
# Loads the tokenizer from a file using json.load().
# Creates an embedding matrix using create_embedding_matrix().
# Saves the embedding matrix to a numpy file using np.save().

import numpy as np
import os
import argparse
from gensim.models import KeyedVectors

def load_glove_embeddings(embedding_file):
    print("Loading GloVe embeddings...")
    embeddings = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
    print("Done!")
    return embeddings

def get_embedding(token, embeddings):
    if token in embeddings:
        return embeddings[token]
    else:
        return np.zeros(embeddings.vector_size)

def create_embedding_matrix(tokenizer, embeddings):
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embeddings.vector_size))
    for word, i in tokenizer.word_index.items():
        embedding_vector = get_embedding(word, embeddings)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def main():
    parser = argparse.ArgumentParser(description='Script to create and manage token embeddings.')
    parser.add_argument('--embedding_file', type=str, required=True, help='Path to GloVe embedding file')
    parser.add_argument('--tokenizer_file', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output file for embedding matrix')
    args = parser.parse_args()

    embeddings = load_glove_embeddings(args.embedding_file)

    with open(args.tokenizer_file, 'r') as f:
        tokenizer = json.load(f)

    embedding_matrix = create_embedding_matrix(tokenizer, embeddings)

    np.save(args.output_file, embedding_matrix)

if __name__ == '__main__':
    main()
