import numpy as np
weights = { f"w{i}" : np.round(np.random.random(), 2) for i in range(1, 7)}
bias    = { f"b{i}" : np.round(np.random.random(), 2) for i in range(1, 4)}
print (weights)
print (bias)