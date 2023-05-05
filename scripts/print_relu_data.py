import torch

# Load the .pth file
state_dict = torch.load('trained_mnist_model.pth')

# Iterate through the items in the state dictionary
for key, value in state_dict.items():
    print(f"{key}:")
    print(value)
    print()

