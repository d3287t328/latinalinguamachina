# In this version, the declensions dictionary has been updated to include all singular and plural forms for masculine, feminine, and neuter nouns. Each word is now a key in the dictionary, and its value is a nested dictionary with keys for each form (e.g., 'sg_masc' for singular masculine).

# The get_declension() function now returns the entire nested dictionary for a given word if it is found in the declensions dictionary, or an empty dictionary if the word is not found.

# The if statement in the main body of the script checks if the declension dictionary returned by get_declension() is non-empty, and if so, prints all of the declensions for the word. If the dictionary is empty, a message is printed saying that the word is not in the dictionary.

# To use this script, you can save it as a Python file (e.g., latin_declension.py) and run it from the command line with a Latin word as an argument:

# Copy code
# python latin_declension.py rosa
# This will return all of the declensions for the word 'rosa', which is in the first declension according to the declensions dictionary.

import sys

def get_declension(word):
    # Define declension rules here
    declensions = {
        'aqua': {'sg_masc': '1st', 'sg_fem': '1st', 'sg_neut': '1st', 'pl_masc': '1st', 'pl_fem': '1st', 'pl_neut': '1st'},
        'puer': {'sg_masc': '2nd', 'sg_fem': '2nd', 'sg_neut': '2nd', 'pl_masc': '2nd', 'pl_fem': '2nd', 'pl_neut': '2nd'},
        'rosa': {'sg_masc': '1st', 'sg_fem': '1st', 'sg_neut': '1st', 'pl_masc': '1st', 'pl_fem': '1st', 'pl_neut': '1st'},
        'dominus': {'sg_masc': '2nd', 'sg_fem': '2nd', 'sg_neut': '2nd', 'pl_masc': '2nd', 'pl_fem': '2nd', 'pl_neut': '2nd'},
        # Add more declensions here...
    }
    if word in declensions:
        return declensions[word]
    else:
        return {}

if len(sys.argv) > 1:
    word = sys.argv[1]
    declension = get_declension(word)
    if declension:
        print(f'{word} has the following declensions:')
        print(f'Singular masculine: {declension["sg_masc"]}')
        print(f'Singular feminine: {declension["sg_fem"]}')
        print(f'Singular neuter: {declension["sg_neut"]}')
        print(f'Plural masculine: {declension["pl_masc"]}')
        print(f'Plural feminine: {declension["pl_fem"]}')
        print(f'Plural neuter: {declension["pl_neut"]}')
    else:
        print(f'{word} is not in the dictionary')
else:
    print('Please provide a Latin word as a command line argument')
