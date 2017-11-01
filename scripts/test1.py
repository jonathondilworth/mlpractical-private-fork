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

all_stats_005 = np.load('all_stats_005_avg.npy')

fig_2 = plt.figure(figsize=(5, 5))

ax_2 = fig_2.add_subplot(111)

for k in ['acc(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[0].shape[0]) * stats_interval, 
              all_stats_005[0][1:, keys[k]], label='Sigmoid')
    
for k in ['acc(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[1].shape[0]) * stats_interval, 
              all_stats_005[1][1:, keys[k]], label='ReLU')
    
for k in ['acc(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[2].shape[0]) * stats_interval, 
              all_stats_005[2][1:, keys[k]], label='LReLU')
    
for k in ['acc(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[3].shape[0]) * stats_interval, 
              all_stats_005[3][1:, keys[k]], label='ELU')
    
for k in ['acc(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[4].shape[0]) * stats_interval, 
              all_stats_005[4][1:, keys[k]], label='SELU')
    
ax_2.legend(loc=0)
ax_2.set_xlabel('Epoch number')
ax_2.set_ylabel('Validation Accuracy')
ax_2.set_title('Test Title')

plt.savefig('test_1234.svg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='svg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

plt.savefig('test_1234.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)