import numpy as np

n = 2 # number of inputs
num_hidden_layers = 1 # number of hidden layers
m = 2 # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=6), decimals=2) # initialize the biases
print(weights)
print(biases)
x_0 = 0.5 # input 1
x_1 = 0.85 # input 2

print('x0 is {} and x1 is {}'.format(x_0, x_1))
z_11 = x_0 * weights[0] + x_1 * weights[1] + biases[0]
z_12 = x_0 * weights[2] + x_1 * weights[3] + biases[1]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(np.around(z_11, decimals=2)))
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=2)))
a_11 = 1.0 / (1.0 + np.exp(-z_11))
a_12 = 1.0 / (1.0 + np.exp(-z_12))

print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=2)))
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=2)))

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=2)))

a_2 = 1.0 / (1.0 + np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=2)))
