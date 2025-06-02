import numpy as np

def get_xor_data(x_dim, n_data):
    """
    returns dataset x,y where:
    x = [n_data, x_dim] such that x_dim represents suze of a string of random bits (i.e. 0100)
    y = [n_data, 1] such that it represents the label of xor (i.e. either 0 or 1)
    """
    x = np.random.randint(0, 2, (n_data, x_dim))
    y = np.sum(x, axis=1, keepdims=True) % 2
    return x, y
