import numpy as np
x = np.array([2, 3])
input_size = 2
hidden_size = 2
output_size = 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.random.randn(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.random.randn(output_size)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
z1 = np.dot(x, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
output = sigmoid(z2)
print("Weighted sum: ", np.round(z2, 2))
print("Activation of the first node: ", np.round(a1[0], 2))
print("Activation of the second node: ", np.round(a1[1], 2))
print("Weighted sum of inputs to the output node: ", np.round(z2, 2))
print("Output of the network: ", np.round(output, 2))