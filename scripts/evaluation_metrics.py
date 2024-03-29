# python evaluation_metrics.py --test_file path/to/test_file.txt --predictions_file path/to/predictions_file.txt


import argparse
import math
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def calculate_perplexity(probabilities):
    # probabilities is a list of log probabilities
    # we first calculate the average log probability
    avg_log_prob = sum(probabilities) / len(probabilities)
    return math.exp(-avg_log_prob)

def calculate_bleu_score(candidate_sentences, reference_sentences):
    # candidate_sentences is a list of sentences generated by the model
    # reference_sentences is a list of reference sentences (i.e., ground truth)
    # we first tokenize the sentences
    candidate_sentences = [sentence.split() for sentence in candidate_sentences]
    reference_sentences = [[sentence.split()] for sentence in reference_sentences]
    return corpus_bleu(reference_sentences, candidate_sentences)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute evaluation metrics on test dataset')
    parser.add_argument('--test_file', type=str, required=True, help='Path to test file')
    parser.add_argument('--predictions_file', type=str, required=True, help='Path to predictions file')
    args = parser.parse_args()

    # load the test data and predictions
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_sentences = f.read().splitlines()
    with open(args.predictions_file, 'r', encoding='utf-8') as f:
        predicted_sentences = f.read().splitlines()

    # calculate perplexity
    test_probabilities = [float(sentence.split()[0]) for sentence in test_sentences]
    perplexity = calculate_perplexity(test_probabilities)
    print('Perplexity:', perplexity)

    # calculate BLEU score
    bleu_score = calculate_bleu_score(predicted_sentences, test_sentences)
    print('BLEU score:', bleu_score)
