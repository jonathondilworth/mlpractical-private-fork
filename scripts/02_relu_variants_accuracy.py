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

all_stats_02 = np.load('all_stats_02_avg.npy')

# Training Error, Learning Rate: 0.05

y_major_ticks = np.arange(0.92, 1.01, 0.02)
y_minor_ticks = np.arange(0.92, 1.01, 0.005)
x_major_ticks = np.arange(0, 101, 20)
x_minor_ticks = np.arange(0, 101, 5)

fig_1 = plt.figure(figsize=(10, 10))

ax_1 = fig_1.add_subplot(221)

for k in ['acc(train)']:
    ax_1.plot(np.arange(1, all_stats_02[1].shape[0]) * stats_interval, 
              all_stats_02[1][1:, keys[k]], label='Training Accuracy')

for k in ['acc(valid)']:
    ax_1.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[1][1:, keys[k]], label='Validation Accuracy')

ax_1.set_xticks(x_major_ticks)                                                       
ax_1.set_xticks(x_minor_ticks, minor=True)                                           
ax_1.set_yticks(y_major_ticks)                                                       
ax_1.set_yticks(y_minor_ticks, minor=True)
ax_1.grid(which='both')
ax_1.grid(which='minor', alpha=0.2)                                               
ax_1.grid(which='major', alpha=0.5)
ax_1.legend(loc=0)
ax_1.set_title('ReLU', fontsize=14)

#

ax_2 = fig_1.add_subplot(222)

for k in ['acc(train)']:
    ax_2.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[2][1:, keys[k]], label='Training Accuracy')

for k in ['acc(valid)']:
    ax_2.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[2][1:, keys[k]], label='Validation Accuracy')

ax_2.set_xticks(x_major_ticks)                                                       
ax_2.set_xticks(x_minor_ticks, minor=True)                                           
ax_2.set_yticks(y_major_ticks)                                                       
ax_2.set_yticks(y_minor_ticks, minor=True)
ax_2.grid(which='both')
ax_2.grid(which='minor', alpha=0.2)                                               
ax_2.grid(which='major', alpha=0.5)
ax_2.legend(loc=0)
ax_2.set_title('LReLU', fontsize=14)

#

ax_3 = fig_1.add_subplot(223)

for k in ['acc(train)']:
    ax_3.plot(np.arange(1, all_stats_02[3].shape[0]) * stats_interval, 
              all_stats_02[3][1:, keys[k]], label='Training Accuracy')

for k in ['acc(valid)']:
    ax_3.plot(np.arange(1, all_stats_02[3].shape[0]) * stats_interval, 
              all_stats_02[3][1:, keys[k]], label='Validation Accuracy')

ax_3.set_xticks(x_major_ticks)                                                       
ax_3.set_xticks(x_minor_ticks, minor=True)                                           
ax_3.set_yticks(y_major_ticks)                                                       
ax_3.set_yticks(y_minor_ticks, minor=True)
ax_3.grid(which='both')
ax_3.grid(which='minor', alpha=0.2)                                               
ax_3.grid(which='major', alpha=0.5)
ax_3.legend(loc=0)
ax_3.set_title('ELU', fontsize=14)

#

ax_4 = fig_1.add_subplot(224)

for k in ['acc(train)']:
    ax_4.plot(np.arange(1, all_stats_02[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]], label='Training Accuracy')

for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, all_stats_02[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]], label='Validation Accuracy')

ax_4.set_xticks(x_major_ticks)                                                       
ax_4.set_xticks(x_minor_ticks, minor=True)                                           
ax_4.set_yticks(y_major_ticks)                                                       
ax_4.set_yticks(y_minor_ticks, minor=True)
ax_4.grid(which='both')
ax_4.grid(which='minor', alpha=0.2)                                               
ax_4.grid(which='major', alpha=0.5)
ax_4.legend(loc=0)
ax_4.set_title('SELU', fontsize=14)

#

fig_1.text(0.5, 0.04, 'Epoch number', ha='center', fontsize=16)
fig_1.text(0.04, 0.5, 'Accuracy', va='center', rotation='vertical', fontsize=16)

plt.suptitle("ReLU Variant Activation Functions Training and Validation Accuracy, Step Size = 0.2", fontsize=16)

plt.savefig('02_relu_variants_accuracy_svg.svg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='svg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

plt.savefig('02_relu_variants_accuracy_png.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)
