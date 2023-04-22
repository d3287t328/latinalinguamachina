# Make a function gerund_infinitive that, given a string 
# returns the rest of the string with "to ". If the word
# doesn't end in "ing", return "That's not an English gerund!"

import argparse

def gerund_infinitive(string):
    if string.endswith('ing'):
        return f"to {string[:-3]}"
    else:
        return "That's not a gerund!"

parser = argparse.ArgumentParser(description='Help command for CLI flags')
parser.add_argument('word', type=str, nargs='?', default=None, help='Word to check if it is a gerund')
parser.add_argument('-v', '--version', action='store_true', help='Show version')

args = parser.parse_args()

if args.version:
    print('Version 0.0.1')
    exit()

if args.word is None:
    parser.print_help()
    exit()

result = gerund_infinitive(args.word)
print(result)
