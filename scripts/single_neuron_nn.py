import numpy as np

# Step 1: Define the input data
# Let's say our input data is a 1D array (vector) of length 3
input_data = np.array([0.5, 0.3, -0.1])

# Step 2: Initialize neuron's weights and bias
# The neuron has as many weights as the input_data length
weights = np.array([0.8, -0.6, 0.4])
bias = 0.2

# Step 3: Calculate the weighted sum
weighted_sum = np.dot(input_data, weights) + bias
# weighted_sum = (0.5 * 0.8) + (0.3 * (-0.6)) + (-0.1 * 0.4) + 0.2
# weighted_sum = 0.4 - 0.18 - 0.04 + 0.2
# weighted_sum = 0.38

# Step 4: Apply the activation function (ReLU in this case)
def relu(x):
    return max(0, x)

activated_output = relu(weighted_sum)
# activated_output = relu(0.38)
# activated_output = 0.38

print("Activated output of the neuron:", activated_output)

