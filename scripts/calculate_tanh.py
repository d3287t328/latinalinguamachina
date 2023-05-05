import math
import matplotlib.pyplot as plt
import numpy as np

def tanh_function(x):
    exp_x = math.exp(x)
    exp_neg_x = math.exp(-x)
    num = exp_x - exp_neg_x
    den = exp_x + exp_neg_x
    y = num / den
    print(f"Computing tanh({x}):")
    print(f"  Computing e^{x}...")
    print(f"    e^{x} = {exp_x:.2f}")
    print(f"  Computing e^(-{x})...")
    print(f"    e^(-{x}) = {exp_neg_x:.2f}")
    print(f"  Computing the numerator...")
    print(f"    e^{x} - e^(-{x}) = {num:.2f}")
    print(f"  Computing the denominator...")
    print(f"    e^{x} + e^(-{x}) = {den:.2f}")
    print(f"  Computing the final result...")
    print(f"    (e^{x} - e^(-{x})) / (e^{x} + e^(-{x})) = {y:.2f}")
    return y

# Get user input
x_str = input("Enter a floating point number for x: ")
try:
    x = float(x_str)
except ValueError:
    print("Invalid input. Please enter a floating point number.")
    exit()

# Compute tanh value
y = tanh_function(x)

# Create visualization
x_values = np.linspace(-10, 10, 200)
y_values = []
for x_val in x_values:
    y_val = tanh_function(x_val)
    y_values.append(y_val)
plt.plot(x_values, y_values)
plt.plot([x, x], [0, y], linestyle='--', color='red')
plt.plot([-10, 10], [y, y], linestyle='--', color='red')
plt.scatter(x, y, color='red')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title(f'tanh({x}) = {y:.2f}')
plt.show()

# Print output
print(f"The tanh value of {x} is {y:.2f}")

