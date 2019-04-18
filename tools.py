import numpy as np

def normalize(*x):
    return np.arctan(x) * (2/np.pi)

def change_weights(method, *x):
    if method == 'random':
        return np.random.rand() * 2 - 1
        # np.random.normal(sum(weight)/len(weight), 0.05, 6)
    elif method == 'increment':
        coin = np.random.rand()
        increment_size = 0.1
        return [n + increment_size if np.random.rand() > 0.5 else n + increment_size for n in x]
