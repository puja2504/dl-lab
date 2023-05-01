import numpy as np
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
network = {}
for i, (w, b) in enumerate(zip(weights.values(), biases.values())):
    layer = f"layer {i+1}"
    network[layer] = {"weights": w, "bias": b}
print(network)