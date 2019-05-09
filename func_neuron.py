import numpy as np 
from scipy import signal
from CPPN.tools import normalize
class FNUER:
    def __init__(self, coefficient, multiplier, method):
        self.coefficient = coefficient
        self.multiplier = multiplier
        self.method = method
        
    # Returns the output of the signal nueron based on the input signal and 
    # the type of periodic function within the neuron. 
    def calculate(self, func, x, y, z, wts, j = None):
        in_put = x * wts[0] + y * wts[1] + x + wts[2]
        if j is not None:
            in_put += j * wts[3]
        in_put = normalize(in_put)
        value  = None
        if self.method == "sin":
            value = self.multiplier * \
                np.sin(in_put * self.coefficient)
        elif self.method == "cos":
            value = self.multiplier * \
                np.cos(in_put * self.coefficient)
        elif self.method == "sawtooth":
            value = self.multiplier * \
                signal.sawtooth(in_put * self.coefficient)
        # elif self.method == "":
        #     signal
        return value

        
