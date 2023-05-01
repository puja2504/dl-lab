import numpy as np
def relu(x):
    return np.maximum(0, x)
input_size = 5
hidden_sizes = [3, 2, 3]
output_size = 1
weights = {}
biases = {}
for i, h in enumerate(hidden_sizes):
    if i == 0:
        weights[f"w{i+1}"] = np.random.randn(input_size, h)
        biases[f"b{i+1}"] = np.random.randn(h)
    else:
        weights[f"w{i+1}"] = np.random.randn(hidden_sizes[i-1], h)
        biases[f"b{i+1}"] = np.random.randn(h)
weights["w_out"] = np.random.randn(hidden_sizes[-1], output_size)
biases["b_out"] = np.random.randn(output_size)
def forward_propagation_with_relu(x):
    z1 = np.dot(x, weights["w1"]) + biases["b1"]
    a1 = relu(abs(z1))
    z2 = np.dot(a1, weights["w2"]) + biases["b2"]
    a2 = relu(abs(z2))
    z3 = np.dot(a2, weights["w3"]) + biases["b3"]
    a3 = relu(abs(z3))
    z_out = np.dot(a3, weights["w_out"]) + biases["b_out"]
    y_hat = relu(abs(z_out))
    return y_hat, [a1, a2, a3]
x = np.random.randn(input_size)
y_hat, activations = forward_propagation_with_relu(x)
for i, a in enumerate(activations):
    layer = f"layer {i+1}"
    print(f"Activations for {layer}: {a}")