import numpy as np
input_size = 2
hidden_size = 2
output_size = 1
Weights1 = np.random.randn(input_size, hidden_size)
biases1 = np.random.randn(hidden_size)
Weights2 = np.random.randn(hidden_size, output_size)
biases2 = np.random.randn(output_size)
print("Weights1 = ", Weights1)
print("biases1 = ", biases1)
print("Weights2 = ", Weights2)
print("biases2 = ", biases2)
