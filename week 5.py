import numpy as np
class Network:
    def __init__(self, layers):
        self.network = {}
        self.layers = layers
        
        for i in range(1, len(layers)):
            self.network[f"layer {i}"] =  {
                "weights" : np.random.rand(
                        layers[i - 1],     # per each row each output
                        layers[i]          # per each col each input
                    ),
                "bias" : np.random.rand(
                        layers[i]
                    )
            }
    
    def activation(self, values):
        """
            RELU activation function
        """
        relu = lambda x : max(0, x)
        
        return np.array([
            relu(value) for value in values
        ])
        
    def forward(self, input_layer):
        """
            outcome function which will give the output values output layer
            
            
            output = f(w.transpose * input + bias)
        """
        input_values = np.array(input_layer)
        
        for layer_name, layer in self.network.items():
            layer_output = (layer['weights'].transpose() @ input_values) + layer['bias']
          
            threshold_output = self.activation(layer_output)
        
            input_values = threshold_output
            
        return np.round(input_values, 2)
network = Network([5, 3, 2, 3, 1])
print(f"Output is {network.forward([2, 1, 1 , 1, 2])}")