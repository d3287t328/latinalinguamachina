import argparse
import chardet
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
    with open(args.input_file, 'rb') as f:
        file_contents = f.read()
        file_encoding = chardet.detect(file_contents)['encoding']
        text = file_contents.decode(file_encoding)

    # Preprocess text
    text = preprocess_text(text)

    # Write preprocessed text to output file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write(text)

