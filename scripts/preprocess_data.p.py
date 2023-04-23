# python preprocess_data.py input.txt output.txt
# This will preprocess the text in input.txt and save the preprocessed text to output.txt.


import re
import argparse
import nltk
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    # Tokenize text into words
    words = word_tokenize(text)

    # Lowercase all words
    words = [word.lower() for word in words]

    # Remove special characters and numbers
    words = [re.sub(r'[^a-zA-Z]', '', word) for word in words]

    # Remove empty words
    words = [word for word in words if word]

    # Join words back into a single string
    text = ' '.join(words)

    return text

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Preprocess text data')
    parser.add_argument('input_file', type=str, help='Path to input file')
    parser.add_argument('output_file', type=str, help='Path to output file')
    args = parser.parse_args()

    # Read input file
    with open(args.input_file, 'r') as f:
        text = f.read()

    # Preprocess text
    text = preprocess_text(text)

    # Write preprocessed text to output file
    with open(args.output_file, 'w') as f:
        f.write(text)
