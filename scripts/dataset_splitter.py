# python dataset_splitter.py input_file.txt output_dir/
# You can also optionally specify the percentage of data 
# to use for training and validation using the --train_percent 
# and --val_percent arguments. The default values are 0.8 and 0.1, 
# respectively.

import argparse
import os
import random

def split_data(input_file, output_dir, train_percent, val_percent):
    # Read the preprocessed data from the input file
    with open(input_file, 'r') as f:
        data = f.readlines()

    # Shuffle the data
    random.shuffle(data)

    # Calculate the number of examples for each set
    num_examples = len(data)
    num_train = int(num_examples * train_percent)
    num_val = int(num_examples * val_percent)
    num_test = num_examples - num_train - num_val

    # Write the data to the output files
    train_file = os.path.join(output_dir, 'train.txt')
    val_file = os.path.join(output_dir, 'val.txt')
    test_file = os.path.join(output_dir, 'test.txt')

    with open(train_file, 'w') as f:
        f.writelines(data[:num_train])

    with open(val_file, 'w') as f:
        f.writelines(data[num_train:num_train+num_val])

    with open(test_file, 'w') as f:
        f.writelines(data[num_train+num_val:])

    print(f'Successfully split data into training, validation, and testing sets: {num_train} training examples, {num_val} validation examples, and {num_test} testing examples.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split preprocessed data into training, validation, and testing sets.')
    parser.add_argument('input_file', type=str, help='path to input file')
    parser.add_argument('output_dir', type=str, help='path to output directory')
    parser.add_argument('--train_percent', type=float, default=0.8, help='percentage of data to use for training')
    parser.add_argument('--val_percent', type=float, default=0.1, help='percentage of data to use for validation')
    args = parser.parse_args()

    split_data(args.input_file, args.output_dir, args.train_percent, args.val_percent)
