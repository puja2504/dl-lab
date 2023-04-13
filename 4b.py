import numpy as np # import the Numpy library

num_inputs=5
num_hidden_layers=3
num_nodes_hidden=[3, 2, 3]
num_nodes_output=1
inputs = np.around(np.random.uniform(size=num_inputs), decimals=2)

small_network = initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output)

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def node_activation(weighted_sum):
    return (np.exp(weighted_sum)-np.exp(-1 * weighted_sum)) / (np.exp(weighted_sum)+np.exp(-1 * weighted_sum))


print('The inputs to the network are {}'.format(inputs))

node_weights = small_network['layer_1']['node_1']['weights']
node_bias = small_network['layer_1']['node_1']['bias']
weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
node_output  = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))

def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=2))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions

predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions, decimals=2)))