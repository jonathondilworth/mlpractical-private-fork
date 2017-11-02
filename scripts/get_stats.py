import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import logging
from mlp.data_providers import MNISTDataProvider
from mlp.layers import AffineLayer, SoftmaxLayer, SigmoidLayer, ReluLayer, LeakyReluLayer, ELULayer, SELULayer
from mlp.errors import CrossEntropySoftmaxError
from mlp.models import MultipleLayerModel
from mlp.initialisers import ConstantInit, GlorotUniformInit, SELUInit
from mlp.learning_rules import GradientDescentLearningRule
from mlp.optimisers import Optimiser
import json


stats_interval = 1

keys = {'acc(train)': 1, 'acc(valid)': 3, 'error(train)': 0, 'error(valid)': 2, 'params_penalty': 4}

five_layers = np.load('5_layer.npy')
six_layers = np.load('6_layer.npy')
seven_layers = np.load('7_layer.npy')
eight_layers = np.load('8_layer.npy')

# Training Error, Learning Rate: 0.05

y_major_ticks = np.arange(0, 0.31, 0.1)
y_minor_ticks = np.arange(0, 0.31, 0.03)
x_major_ticks = np.arange(0, 101, 20)
x_minor_ticks = np.arange(0, 101, 5)

fig_1 = plt.figure(figsize=(11, 11))

ax_1 = fig_1.add_subplot(221)

for k in ['error(train)']:
    ax_1.plot(np.arange(1, five_layers.shape[0]) * stats_interval, 
              five_layers[1:, keys[k]], label='Training Error')

for k in ['error(valid)']:
    ax_1.plot(np.arange(1, five_layers.shape[0]) * stats_interval, 
              five_layers[1:, keys[k]], label='Validation Error')


ax_1.set_xticks(x_major_ticks)                                                       
ax_1.set_xticks(x_minor_ticks, minor=True)                                           
ax_1.set_yticks(y_major_ticks)                                                       
ax_1.set_yticks(y_minor_ticks, minor=True)
ax_1.grid(which='both')
ax_1.grid(which='minor', alpha=0.2)                                               
ax_1.grid(which='major', alpha=0.5)
ax_1.legend(loc=0)
ax_1.set_title('Five Layers')

# Training Error, Learing Rate: 0.1

ax_2 = fig_1.add_subplot(222)

for k in ['error(train)']:
    ax_2.plot(np.arange(1, six_layers.shape[0]) * stats_interval, 
              six_layers[1:, keys[k]], label='Training Error')

for k in ['error(valid)']:
    ax_2.plot(np.arange(1, six_layers.shape[0]) * stats_interval, 
              six_layers[1:, keys[k]], label='Validation Error')


ax_2.set_xticks(x_major_ticks)                                                       
ax_2.set_xticks(x_minor_ticks, minor=True)                                           
ax_2.set_yticks(y_major_ticks)                                                       
ax_2.set_yticks(y_minor_ticks, minor=True)
ax_2.grid(which='both')
ax_2.grid(which='minor', alpha=0.2)                                               
ax_2.grid(which='major', alpha=0.5)
ax_2.legend(loc=0)
ax_2.set_title('Six Layers')

# Training Error, Learning Rate: 0.2

ax_3 = fig_1.add_subplot(223)

for k in ['error(train)']:
    ax_3.plot(np.arange(1, seven_layers.shape[0]) * stats_interval, 
              seven_layers[1:, keys[k]], label='Training Error')

for k in ['error(valid)']:
    ax_3.plot(np.arange(1, seven_layers.shape[0]) * stats_interval, 
              seven_layers[1:, keys[k]], label='Validation Error')

ax_3.set_xticks(x_major_ticks)                                                       
ax_3.set_xticks(x_minor_ticks, minor=True)                                           
ax_3.set_yticks(y_major_ticks)                                                       
ax_3.set_yticks(y_minor_ticks, minor=True)
ax_3.grid(which='both')
ax_3.grid(which='minor', alpha=0.2)                                               
ax_3.grid(which='major', alpha=0.5)
ax_3.legend(loc=0)
ax_3.set_title('Seven Layers')

# Training Error, Learning Rate: 0.5

ax_4 = fig_1.add_subplot(224)

for k in ['error(train)']:
    ax_4.plot(np.arange(1, eight_layers.shape[0]) * stats_interval, 
              eight_layers[1:, keys[k]], label='ReLU')

for k in ['error(valid)']:
    ax_4.plot(np.arange(1, eight_layers.shape[0]) * stats_interval, 
              eight_layers[1:, keys[k]], label='LReLU')

ax_4.set_xticks(x_major_ticks)                                                       
ax_4.set_xticks(x_minor_ticks, minor=True)                                           
ax_4.set_yticks(y_major_ticks)                                                       
ax_4.set_yticks(y_minor_ticks, minor=True)
ax_4.grid(which='both')
ax_4.grid(which='minor', alpha=0.2)                                               
ax_4.grid(which='major', alpha=0.5)
ax_4.legend(loc=0)
ax_4.set_title('Eight Layers')

fig_1.text(0.5, 0.04, 'Epoch', ha='center', fontsize=16)
fig_1.text(0.04, 0.5, 'Step Size', va='center', rotation='vertical', fontsize=16)

plt.suptitle("How the number of hidden layers affect training error and accuracy")

plt.savefig('hidden_layers_affects_svg.svg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='svg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

plt.savefig('hidden_layers_affects_png.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
