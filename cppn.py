import numpy as np
from func_neuron import FNUER
from tools import normalize
class CPPN:
    def __init__(self, **kwargs):
        self.functions = kwargs['funcs']
        self.input_wts = {}
        self.output_wts = np.random.rand(4)
        self.hidden_layer = {}
        self.coefficients = {}
        self.multipliers = {}
        for h in self.functions:
            self.coefficients[h] = np.random.rand()
            self.multipliers[h] = np.random.rand()
            self.input_wts[h] = np.random.rand(4)
            self.hidden_layer[h] = FNUER(self.coefficients[h], self.multipliers[h], h)
        
    def paint(self, z, x, y, j = None):
        out = []
        for nueron in self.hidden_layer:
            out.append(self.hidden_layer[nueron].calculate(nueron, x, y, z, self.input_wts[nueron]))
        final = 0
        for i in range(len(out)):
            final += out[i] * self.output_wts[i]
        return normalize(final)

        

    
        



