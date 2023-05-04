import os

def get_token_count(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        tokens = content.split()
        return len(tokens)

file1_name = input("Enter the file name for file 1: ")
file2_name = input("Enter the file name for file 2: ")

if os.path.exists(file1_name) and os.path.exists(file2_name):
    count1 = get_token_count(file1_name)
    count2 = get_token_count(file2_name)
    diff = abs(count1 - count2)
    print(f"Token count difference between {file1_name} and {file2_name} is {diff}")
else:
    print("One or both files do not exist.")

