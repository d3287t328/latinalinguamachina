
# The script uses the argparse library to parse command line arguments. These include the path to the data file, the path to save the vocabulary file, the maximum vocabulary size, the type of tokenization to use (word, subword, or character), and the maximum subword vocabulary size (if using subword tokenization).
# The read_data_file function reads the data file and splits it into tokens (words, subwords, or characters) using the specified tokenization method.
# The create_vocab function takes the tokenized data and creates a vocabulary of unique tokens. This function uses Counter to count the occurrences of each token and then sorts them by frequency and alphabetically to break ties. For word and character tokenization, the function selects the top vocab_size-2 tokens to reserve spots for padding and unknown tokens. For subword tokenization, the function uses the


import argparse
from collections import Counter
import os
import pickle

parser = argparse.ArgumentParser(description='Create and save a vocabulary from a dataset')
parser.add_argument('--data_file', type=str, required=True,
                    help='Path to the data file')
parser.add_argument('--vocab_file', type=str, required=True,
                    help='Path to save the vocabulary file')
parser.add_argument('--vocab_size', type=int, default=50000,
                    help='Maximum size of the vocabulary')
parser.add_argument('--token_type', type=str, choices=['word', 'subword', 'char'], default='word',
                    help='Type of tokenization to use (word, subword, or character)')
parser.add_argument('--subword_vocab_size', type=int, default=10000,
                    help='Maximum size of the subword vocabulary (only applicable for subword tokenization)')

args = parser.parse_args()

def read_data_file(file_path):
    with open(file_path, 'r') as f:
        data = f.read().split()
    return data

def create_vocab(data, token_type, vocab_size=None, subword_vocab_size=None):
    if token_type == 'word':
        token_counts = Counter(data)
        sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
        tokens = [t[0] for t in sorted_tokens[:vocab_size-2]]  # -2 to reserve spots for padding and unknown tokens
    elif token_type == 'subword':
        from tokenizers import ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(data, vocab_size=subword_vocab_size)
        tokens = tokenizer.get_vocab().keys()
    elif token_type == 'char':
        token_counts = Counter(''.join(data))
        sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
        tokens = [t[0] for t in sorted_tokens[:vocab_size-2]]  # -2 to reserve spots for padding and unknown tokens
    tokens = ['<pad>', '<unk>'] + tokens
    return tokens

def save_vocab(vocab, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == '__main__':
    data = read_data_file(args.data_file)
    vocab = create_vocab(data, args.token_type, args.vocab_size, args.subword_vocab_size)
    save_vocab(vocab, args.vocab_file)
    print(f'Saved vocabulary of size {len(vocab)} to {args.vocab_file}.')
