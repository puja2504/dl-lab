import numpy as np

n = 2 # number of inputs
num_hidden_layers = 1 # number of hidden layers
m = 2 # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer

weights1 = np.around(np.random.uniform(size=n*m), decimals=2) # initialize the weights
biases1 = np.around(np.random.uniform(size=m), decimals=2) # initialize the biases

print("Weights between Input & Hidden layer",weights1)
print("Bias in Hidden layer",biases1)

weights2 = np.around(np.random.uniform(size=m*num_nodes_output), decimals=2) # initialize the weights
biases2 = np.around(np.random.uniform(size=num_nodes_output), decimals=2) # initialize the biases

print("Weights between Hidden layer & Output",weights2)
print("Bias in Output layer",biases2)