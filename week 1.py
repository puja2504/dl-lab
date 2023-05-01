import numpy as np
input_size = 2
hidden_size = 2
output_size = 1
Weights1 = np.random.randn(input_size, hidden_size)
biases1 = np.random.randn(hidden_size)
Weights2 = np.random.randn(hidden_size, output_size)
biases2 = np.random.randn(output_size)
print("Weights between Input & Hidden layer = ", Weights1)
print("Biases in Hidden layer = ", biases1)
print("Weights between Hidden layer & Output = ", Weights2)
print("Biases in Output layer = ", biases2)
