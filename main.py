#%% Setup
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy import signal

#%%
# for x in range(25):
    # print(signal.gausspulse(x))
# print(signal.sawtooth(1))
num_sensors = 9
num_hidden = 4
num_of_legs = 6
knees = 2

test_vector = [np.zeros((num_sensors, num_hidden)), \
    np.zeros((num_hidden, num_hidden)),
    np.zeros((num_hidden, num_of_legs, knees))]
print(test_vector)
#%% Build a CPPN 
# Lets start with a CPPN in a 1 dimensional plane. 
x_axis = 9
y_axis = 4
output = 3
tr_names = ['sin', 'cos', 'sawtooth', "combined"]
transformations = {}
def versatile_function(name, x, y, z, multiplier, coefficient, weights):
    value = 0
    if name == "sin":
        value = np.sin(x * coefficient) * weights[0] + np.sin(y * coefficient) * weights[1] + np.sin(z * coefficient) * weights[2]
    elif name == "cos":
        value = np.cos(x * coefficient) * weights[0] + np.cos(y * coefficient) * weights[1] + np.cos(z * coefficient) * weights[2]
    elif name == "sawtooth":
        value = signal.sawtooth(x * coefficient) * weights[0] + \
            signal.sawtooth(y * coefficient) * weights[1] + \
                signal.sawtooth(z + coefficient) * weights[2]
    elif name == "combined":
        value = versatile_function("sin",x,y,z, multiplier, coefficient, weights) * weights[3] +\
            versatile_function("cos",x,y,z, multiplier, coefficient, weights) * weights[4] +\
                versatile_function("sawtooth",x,y,z, multiplier, coefficient, weights) * weights[5]
    return normalize(value * multiplier)

def normalize(*x):
    return np.arctan(x) * (np.pi/2)
    # return x

def change_weights(method, *x):
    if method == 'random':
        return np.random.normal(x, .25)
        # np.random.normal(sum(weight)/len(weight), 0.05, 6)
    elif method == 'increment':
        coin = np.random.rand()
        increment_size = 0.1
        return [n + increment_size if np.random.rand() > 0.5 else n + increment_size for n in x]

for name in tr_names:
    transformations[name] = np.zeros((x_axis, y_axis, output))
transformations['random']  = np.random.rand(x_axis,y_axis, output)
multiplier = 0
coefficient = 0
method = "random"
weights = np.random.rand(6)
multiplier = normalize(change_weights(method, multiplier))
coefficient = normalize(change_weights(method, coefficient))
for x in range(x_axis):
    for y in range(y_axis):
        index = np.random.randint(0, 8)
        if index < 6:
            weights[index] = normalize(*change_weights(method, weights[index]))
        elif index == 6:
            multiplier = normalize(change_weights(method, multiplier))
        else:
            coefficient = normalize(change_weights(method, coefficient))
        for z in range(output):
            for name in tr_names:
                transformations[name][x][y][z] = \
                    versatile_function(name, x, y, z, multiplier, coefficient, weights)
print('Complete')
#%% Plotting 
fig, ax = plt.subplots(ncols= 2, nrows=3, figsize = (10, 10))
axes = ax.flatten()
# ax1.imshow(random_sample, interpolation = 'nearest')

# ax2.imshow(transformations_sin, interpolation = 'nearest')
# ax3.imshow(transformations_cos, interpolation = 'nearest')
for i in range(len(tr_names)):
    axes[i].imshow(transformations[tr_names[i]], interpolation = 'nearest')
    axes[i].set_title(tr_names[i])
axes[len(tr_names)].imshow(transformations['random'], interpolation = 'nearest')
axes[len(tr_names)].set_title('random')
plt.legend()
plt.show()

# time.sleep(3)




#%% Implementation
fig2, ax2 = plt.subplots(ncols = 2, nrows= 2, figsize = (10, 10))
axes2 = ax2.flatten()
print(test_vector[0])
for n in range(3):
    axes[n].imshow(test_vector[n], interpolation = 'nearest')



#%% Network code 
from cppn import CPPN
tr_names = ['sin', 'cos', 'sawtooth']

kwargs = {}
kwargs['funcs'] = tr_names
sample = CPPN(**kwargs)
sample.paint(1, 2, 3)

for x in range(100):
    print(sample.paint(x, x * 2, x /3))



#%%
