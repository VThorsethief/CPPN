import numpy as np
from CPPN.func_neuron import FNUER
from CPPN.tools import normalize, change_weights

# The class for the CPPN that each robot has. The variable names for input layer and output layer are 
# not related to the robots input output layer, the structure of the CPPN stays the same for all structures fo the robot.  
class CPPN:
    def __init__(self, imported_weights = None):
        # self.functions = kwargs['funcs']
        if imported_weights is not None:
            self.load(imported_weights)
        else:
            self.functions = ['sin', 'cos', 'sawtooth']
            self.input_wts = {}
            self.output_wts = np.random.rand(4) * 2 - 1
            self.coefficients = {}
            self.multipliers = {}
        self.hidden_layer = {}
        for h in self.functions:
            if imported_weights is None:
                # self.coefficients[h] = np.random.rand() * 2 - 1
                self.coefficients[h] = np.pi
                # self.multipliers[h] = np.random.rand() * 2 - 1
                self.multipliers[h] = 1
                self.input_wts[h] = np.random.rand(4) * 2 - 1
            self.hidden_layer[h] = FNUER(self.coefficients[h], self.multipliers[h], h)
        
    # Loading the cppn from a saved file in an Archive() class. Just a dictionary with the base values, 
    # theres no functional nueron class. 
    def load(self, imported):
        self.functions = imported['functions']
        self.input_wts = imported['input']
        self.output_wts = imported['output']
        self.coefficients = imported['cofficients']
        self.multipliers = imported['multipliers']

    # Changes the weights witihin the CPPN to change how it paints the weights. Changes internal values, 
    # returns nothing.  
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

    # Paints the synaptic weight for the spot indicated within the genome. The x,y,z and j 
    # layers are the location of the weight within the individuals whole genome. This calls 
    # the calculate function for the peridoic functions within the CPPN. Appends the output signals
    # and sets that as the new weight at that location in the genome.  
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

    # For painting the weights on the individuals entire genome. In_vec is the individuals genome,
    # this function iterates through each position and sends the coordinates to the paint() function. 
    # Returns the new weights for the genome.
    def paint_weights(self, in_vec):
        vec = []
        for term in range(len(in_vec)):
            vec.append(in_vec[term])
        
        # If there's no hidden layer, then we're done here. 
        if len(in_vec) < 3:
            for x in range(len(vec[0])):
                for y in range(len(vec[0][0])):
                    for z in range(len(vec[0][0][0])):
                        vec[0][x][y] = self.paint(1, x, y, z)    
            return vec
        
        # Painting all synapses from the input nuerons to the hidden nuerons
        for x in range(len(vec[0])):
            for y in range(len(vec[0][0])):
                vec[0][x][y] = self.paint(1, x, y)
        
        # Painting synapses from hidden neurons to other hidden nuerons
        for x in range(len(vec[1])):
            # for y in range(len(vec[1][0])-1):
            for y in range(len(vec[1][0])):
                vec[1][x][y] = self.paint(2, x, y)
        # Painting synapses from hidden nuerons to motors nuerons in the 
        # knees
        for j in range(len(vec[2])):
            for x in range(len(vec[2][0])):
                for y in range(len(vec[2][0][0])):
                    vec[2][j][x][y] = self.paint(3, x, y, j=j)
        return vec

    def save(self):
        temp = {
            'functions': [],
            'input' : {},
            # 'output': [],
            'hidden_layer':{},
            'coefficients':{},
            'multipliers':{}
        }

        for func in self.functions:
            temp['functions'].append(func)
            temp['input'][func] = self.input_wts[func]
            temp['output'] = self.output_wts
            # This will not be included because it will require us to pickle 
            # classes, instead this should have all the basic data to rebuild 
            # functional nuerons from scratch. 
            # temp['hidden_layer'][func] = self.hidden_layer[func]
            temp['coefficients'][func] = self.coefficients[func]
            temp['multipliers'][func] = self.multipliers[func]
        return temp






        

    
        



