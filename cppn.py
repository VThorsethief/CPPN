import numpy as np
from cppn.CPPN.func_neuron import FNUER
from cppn.CPPN.tools import normalize, change_weights
class CPPN:
    def __init__(self, **kwargs):
        # self.functions = kwargs['funcs']
        self.functions = ['sin', 'cos', 'sawtooth']
        self.input_wts = {}
        self.output_wts = np.random.rand(4) * 2 - 1
        self.hidden_layer = {}
        self.coefficients = {}
        self.multipliers = {}
        for h in self.functions:
            # self.coefficients[h] = np.random.rand() * 2 - 1
            self.coefficients[h] = np.pi
            # self.multipliers[h] = np.random.rand() * 2 - 1
            self.multipliers[h] = 1
            self.input_wts[h] = np.random.rand(4) * 2 - 1
            self.hidden_layer[h] = FNUER(self.coefficients[h], self.multipliers[h], h)
        
    def paint(self, z, x, y, j = None):
        out = []
        for nueron in self.hidden_layer:
            out.append(self.hidden_layer[nueron].calculate(nueron, x, y, z, self.input_wts[nueron]))
        final = 0
        for i in range(len(out)):
            final += out[i] * self.output_wts[i]
        # return normalize(final)
        # # print(final/len(out))
        return final

    def paint_weights(self, in_vec):
        vec = []
        for term in range(len(in_vec)):
            vec.append(in_vec[term])
        for x in range(len(vec[0])):
            for y in range(len(vec[0][0])):
                vec[0][x][y] = self.paint(1, x, y)
        for x in range(len(vec[1])):
            for y in range(len(vec[1][0])-1):
                vec[1][x][y] = self.paint(2, x, y)
        for j in range(len(vec[2])):
            for x in range(len(vec[2][0])):
                for y in range(len(vec[2][0][0])):
                    vec[2][j][x][y] = self.paint(3, x, y, j=j)
        return vec

    def mutate(self):
        layer = np.random.randint(0, 2)
        # print(self.input_wts)
        # print("____")
        # print(self.coefficients)
        # print(self.multipliers)
        # print(self.output_wts)
        x = np.random.randint(0, len(self.functions))
        if (layer == 0):
            y = np.random.randint(4)
            self.input_wts[self.functions[x]][y] = \
                change_weights("random", self.input_wts[self.functions[x]][y])
        elif(layer == 5):
            y = np.random.randint(2)
            if y == 0:
                self.coefficients[self.functions[x]] = \
                    change_weights("random", self.coefficients[self.functions[x]])
            elif y == 1:
                self.multipliers[self.functions[x]] = \
                    change_weights("random", self.multipliers[self.functions[x]])
        elif(layer == 1):
            self.output_wts[x] = change_weights("random", self.output_wts[x])


        

    
        



