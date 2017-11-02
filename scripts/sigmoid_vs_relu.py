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
all_stats_01 = np.load('all_stats_01_avg.npy')
all_stats_02 = np.load('all_stats_02_avg.npy')
all_stats_03 = np.load('all_stats_03_avg.npy')

# Step Size: 0.05

# PLOT ONE

fig_1 = plt.figure(figsize=(12, 12))

ax_1 = plt.subplot2grid((4,4), (0,0))

for k in ['error(train)']:
    ax_1.plot(np.arange(1, all_stats_005[0].shape[0]) * stats_interval, 
              all_stats_005[0][1:, keys[k]])

for k in ['error(train)']:
    ax_1.plot(np.arange(1, all_stats_005[1].shape[0]) * stats_interval, 
              all_stats_005[1][1:, keys[k]])

for k in ['error(train)']:
    ax_1.plot(np.arange(1, all_stats_005[2].shape[0]) * stats_interval, 
              all_stats_005[2][1:, keys[k]])

for k in ['error(train)']:
    ax_1.plot(np.arange(1, all_stats_005[3].shape[0]) * stats_interval, 
              all_stats_005[3][1:, keys[k]])

for k in ['error(train)']:
    ax_1.plot(np.arange(1, all_stats_005[4].shape[0]) * stats_interval, 
              all_stats_005[4][1:, keys[k]])

ax_1.legend(loc=0)
ax_1.set_title('Training Error')
ax_1.set_ylabel('0.05')

# PLOT TWO

ax_2 = plt.subplot2grid((4,4), (0,1))

for k in ['error(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[0].shape[0]) * stats_interval, 
              all_stats_005[0][1:, keys[k]])

for k in ['error(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[1].shape[0]) * stats_interval, 
              all_stats_005[1][1:, keys[k]])

for k in ['error(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[2].shape[0]) * stats_interval, 
              all_stats_005[2][1:, keys[k]])

for k in ['error(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[3].shape[0]) * stats_interval, 
              all_stats_005[3][1:, keys[k]])

for k in ['error(valid)']:
    ax_2.plot(np.arange(1, all_stats_005[4].shape[0]) * stats_interval, 
              all_stats_005[4][1:, keys[k]])

ax_2.legend(loc=0)
ax_2.set_title('Validation Error')
ax_2.legend(loc=0)

# PLOT THREE

ax_3 = plt.subplot2grid((4,4), (0,2))

for k in ['acc(train)']:
    ax_3.plot(np.arange(1, all_stats_005[0].shape[0]) * stats_interval, 
              all_stats_005[0][1:, keys[k]])

for k in ['acc(train)']:
    ax_3.plot(np.arange(1, all_stats_005[1].shape[0]) * stats_interval, 
              all_stats_005[1][1:, keys[k]])

for k in ['acc(train)']:
    ax_3.plot(np.arange(1, all_stats_005[2].shape[0]) * stats_interval, 
              all_stats_005[2][1:, keys[k]])

for k in ['acc(train)']:
    ax_3.plot(np.arange(1, all_stats_005[3].shape[0]) * stats_interval, 
              all_stats_005[3][1:, keys[k]])

for k in ['acc(train)']:
    ax_3.plot(np.arange(1, all_stats_005[4].shape[0]) * stats_interval, 
              all_stats_005[4][1:, keys[k]])

ax_3.legend(loc=0)
ax_3.set_title('Training Accuracy')

# PLOT FOUR

ax_4 = plt.subplot2grid((4,4), (0,3))

for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, all_stats_005[0].shape[0]) * stats_interval, 
              all_stats_005[0][1:, keys[k]], label='Sigmoid')

for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, all_stats_005[1].shape[0]) * stats_interval, 
              all_stats_005[1][1:, keys[k]], label='ReLU')

for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, all_stats_005[2].shape[0]) * stats_interval, 
              all_stats_005[2][1:, keys[k]], label='LReLU')

for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, all_stats_005[3].shape[0]) * stats_interval, 
              all_stats_005[3][1:, keys[k]], label='ELU')

for k in ['acc(valid)']:
    ax_4.plot(np.arange(1, all_stats_005[4].shape[0]) * stats_interval, 
              all_stats_005[4][1:, keys[k]], label='SELU')

ax_4.legend(loc=0)
ax_4.set_title('Validation Accuracy')

# PLOT FIVE

ax_5 = plt.subplot2grid((4,4), (1,0))

for k in ['error(train)']:
    ax_5.plot(np.arange(1, all_stats_01[0].shape[0]) * stats_interval, 
              all_stats_01[0][1:, keys[k]])

for k in ['error(train)']:
    ax_5.plot(np.arange(1, all_stats_01[1].shape[0]) * stats_interval, 
              all_stats_01[1][1:, keys[k]])

for k in ['error(train)']:
    ax_5.plot(np.arange(1, all_stats_01[2].shape[0]) * stats_interval, 
              all_stats_01[2][1:, keys[k]])

for k in ['error(train)']:
    ax_5.plot(np.arange(1, all_stats_01[3].shape[0]) * stats_interval, 
              all_stats_01[3][1:, keys[k]])

for k in ['error(train)']:
    ax_5.plot(np.arange(1, all_stats_01[4].shape[0]) * stats_interval, 
              all_stats_01[4][1:, keys[k]])

ax_5.legend(loc=0)
ax_5.set_ylabel('0.1')

# PLOT SIX

ax_6 = plt.subplot2grid((4,4), (1,1))

for k in ['error(valid)']:
    ax_6.plot(np.arange(1, all_stats_01[0].shape[0]) * stats_interval, 
              all_stats_01[0][1:, keys[k]])

for k in ['error(valid)']:
    ax_6.plot(np.arange(1, all_stats_01[1].shape[0]) * stats_interval, 
              all_stats_01[1][1:, keys[k]])

for k in ['error(valid)']:
    ax_6.plot(np.arange(1, all_stats_01[2].shape[0]) * stats_interval, 
              all_stats_01[2][1:, keys[k]])

for k in ['error(valid)']:
    ax_6.plot(np.arange(1, all_stats_01[3].shape[0]) * stats_interval, 
              all_stats_01[3][1:, keys[k]])

for k in ['error(valid)']:
    ax_6.plot(np.arange(1, all_stats_01[4].shape[0]) * stats_interval, 
              all_stats_01[4][1:, keys[k]])

ax_6.legend(loc=0)

# PLOT SEVEN

ax_7 = plt.subplot2grid((4,4), (1,2))

for k in ['acc(train)']:
    ax_7.plot(np.arange(1, all_stats_01[0].shape[0]) * stats_interval, 
              all_stats_01[0][1:, keys[k]])

for k in ['acc(train)']:
    ax_7.plot(np.arange(1, all_stats_01[1].shape[0]) * stats_interval, 
              all_stats_01[1][1:, keys[k]])

for k in ['acc(train)']:
    ax_7.plot(np.arange(1, all_stats_01[2].shape[0]) * stats_interval, 
              all_stats_01[2][1:, keys[k]])

for k in ['acc(train)']:
    ax_7.plot(np.arange(1, all_stats_01[3].shape[0]) * stats_interval, 
              all_stats_01[3][1:, keys[k]])

for k in ['acc(train)']:
    ax_7.plot(np.arange(1, all_stats_01[4].shape[0]) * stats_interval, 
              all_stats_01[4][1:, keys[k]])

ax_7.legend(loc=0)

# PLOT EIGHT

ax_8 = plt.subplot2grid((4,4), (1,3))

for k in ['acc(valid)']:
    ax_8.plot(np.arange(1, all_stats_01[0].shape[0]) * stats_interval, 
              all_stats_01[0][1:, keys[k]])

for k in ['acc(valid)']:
    ax_8.plot(np.arange(1, all_stats_01[1].shape[0]) * stats_interval, 
              all_stats_01[1][1:, keys[k]])

for k in ['acc(valid)']:
    ax_8.plot(np.arange(1, all_stats_01[2].shape[0]) * stats_interval, 
              all_stats_01[2][1:, keys[k]])

for k in ['acc(valid)']:
    ax_8.plot(np.arange(1, all_stats_01[3].shape[0]) * stats_interval, 
              all_stats_01[3][1:, keys[k]])

for k in ['acc(valid)']:
    ax_8.plot(np.arange(1, all_stats_01[4].shape[0]) * stats_interval, 
              all_stats_01[4][1:, keys[k]])

ax_8.legend(loc=0)

# PLOT NINE

ax_9 = plt.subplot2grid((4,4), (2,0))

for k in ['error(train)']:
    ax_9.plot(np.arange(1, all_stats_02[0].shape[0]) * stats_interval, 
              all_stats_02[0][1:, keys[k]])

for k in ['error(train)']:
    ax_9.plot(np.arange(1, all_stats_02[1].shape[0]) * stats_interval, 
              all_stats_02[1][1:, keys[k]])

for k in ['error(train)']:
    ax_9.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[2][1:, keys[k]])

for k in ['error(train)']:
    ax_9.plot(np.arange(1, all_stats_02[3].shape[0]) * stats_interval, 
              all_stats_02[3][1:, keys[k]])

for k in ['error(train)']:
    ax_9.plot(np.arange(1, all_stats_02[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]])

ax_9.legend(loc=0)
ax_9.set_ylabel('0.2')

# PLOT TEN

ax_10 = plt.subplot2grid((4,4), (2,1))

for k in ['error(valid)']:
    ax_10.plot(np.arange(1, all_stats_02[0].shape[0]) * stats_interval, 
              all_stats_02[0][1:, keys[k]])

for k in ['error(valid)']:
    ax_10.plot(np.arange(1, all_stats_02[1].shape[0]) * stats_interval, 
              all_stats_02[1][1:, keys[k]])

for k in ['error(valid)']:
    ax_10.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[2][1:, keys[k]])

for k in ['error(valid)']:
    ax_10.plot(np.arange(1, all_stats_02[3].shape[0]) * stats_interval, 
              all_stats_02[3][1:, keys[k]])

for k in ['error(valid)']:
    ax_10.plot(np.arange(1, all_stats_02[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]])

ax_10.legend(loc=0)

# PLOT ELEVEN

ax_11 = plt.subplot2grid((4,4), (2,2))

for k in ['acc(train)']:
    ax_11.plot(np.arange(1, all_stats_02[0].shape[0]) * stats_interval, 
              all_stats_02[0][1:, keys[k]])

for k in ['acc(train)']:
    ax_11.plot(np.arange(1, all_stats_02[1].shape[0]) * stats_interval, 
              all_stats_02[1][1:, keys[k]])

for k in ['acc(train)']:
    ax_11.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[2][1:, keys[k]])

for k in ['acc(train)']:
    ax_11.plot(np.arange(1, all_stats_02[3].shape[0]) * stats_interval, 
              all_stats_02[3][1:, keys[k]])

for k in ['acc(train)']:
    ax_11.plot(np.arange(1, all_stats_02[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]])

ax_11.legend(loc=0)

# PLOT TWELVE

ax_12 = plt.subplot2grid((4,4), (2,3))

for k in ['acc(valid)']:
    ax_12.plot(np.arange(1, all_stats_02[0].shape[0]) * stats_interval, 
              all_stats_02[0][1:, keys[k]])

for k in ['acc(valid)']:
    ax_12.plot(np.arange(1, all_stats_02[1].shape[0]) * stats_interval, 
              all_stats_02[1][1:, keys[k]])

for k in ['acc(valid)']:
    ax_12.plot(np.arange(1, all_stats_02[2].shape[0]) * stats_interval, 
              all_stats_02[2][1:, keys[k]])

for k in ['acc(valid)']:
    ax_12.plot(np.arange(1, all_stats_02[3].shape[0]) * stats_interval, 
              all_stats_02[3][1:, keys[k]])

for k in ['acc(valid)']:
    ax_12.plot(np.arange(1, all_stats_02[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]])

ax_12.legend(loc=0)

# PLOT THIRTEEN

ax_13 = plt.subplot2grid((4,4), (3,0))

for k in ['error(train)']:
    ax_13.plot(np.arange(1, all_stats_03[0].shape[0]) * stats_interval, 
              all_stats_03[0][1:, keys[k]])

for k in ['error(train)']:
    ax_13.plot(np.arange(1, all_stats_03[1].shape[0]) * stats_interval, 
              all_stats_03[1][1:, keys[k]])

for k in ['error(train)']:
    ax_13.plot(np.arange(1, all_stats_03[2].shape[0]) * stats_interval, 
              all_stats_03[2][1:, keys[k]])

for k in ['error(train)']:
    ax_13.plot(np.arange(1, all_stats_03[3].shape[0]) * stats_interval, 
              all_stats_03[3][1:, keys[k]])

for k in ['error(train)']:
    ax_13.plot(np.arange(1, all_stats_03[4].shape[0]) * stats_interval, 
              all_stats_03[4][1:, keys[k]])

ax_13.legend(loc=0)
ax_13.set_ylabel('0.3')

# PLOT FOURTEEN

ax_14 = plt.subplot2grid((4,4), (3,1))

for k in ['error(valid)']:
    ax_14.plot(np.arange(1, all_stats_03[0].shape[0]) * stats_interval, 
              all_stats_03[0][1:, keys[k]])

for k in ['error(valid)']:
    ax_14.plot(np.arange(1, all_stats_03[1].shape[0]) * stats_interval, 
              all_stats_03[1][1:, keys[k]])

for k in ['error(valid)']:
    ax_14.plot(np.arange(1, all_stats_03[2].shape[0]) * stats_interval, 
              all_stats_03[2][1:, keys[k]])

for k in ['error(valid)']:
    ax_14.plot(np.arange(1, all_stats_03[3].shape[0]) * stats_interval, 
              all_stats_03[3][1:, keys[k]])

for k in ['error(valid)']:
    ax_14.plot(np.arange(1, all_stats_03[4].shape[0]) * stats_interval, 
              all_stats_03[4][1:, keys[k]])

ax_14.legend(loc=0)

# PLOT FIVETEEN

ax_15 = plt.subplot2grid((4,4), (3,2))

for k in ['acc(train)']:
    ax_15.plot(np.arange(1, all_stats_03[0].shape[0]) * stats_interval, 
              all_stats_03[0][1:, keys[k]])

for k in ['acc(train)']:
    ax_15.plot(np.arange(1, all_stats_03[1].shape[0]) * stats_interval, 
              all_stats_03[1][1:, keys[k]])

for k in ['acc(train)']:
    ax_15.plot(np.arange(1, all_stats_03[2].shape[0]) * stats_interval, 
              all_stats_03[2][1:, keys[k]])

for k in ['acc(train)']:
    ax_15.plot(np.arange(1, all_stats_03[3].shape[0]) * stats_interval, 
              all_stats_03[3][1:, keys[k]])

for k in ['acc(train)']:
    ax_15.plot(np.arange(1, all_stats_03[4].shape[0]) * stats_interval, 
              all_stats_03[4][1:, keys[k]])

ax_15.legend(loc=0)

# PLOT SIXTEEN

ax_16 = plt.subplot2grid((4,4), (3,3))

for k in ['acc(valid)']:
    ax_16.plot(np.arange(1, all_stats_03[0].shape[0]) * stats_interval, 
              all_stats_03[0][1:, keys[k]])

for k in ['acc(valid)']:
    ax_16.plot(np.arange(1, all_stats_03[1].shape[0]) * stats_interval, 
              all_stats_03[1][1:, keys[k]])

for k in ['acc(valid)']:
    ax_16.plot(np.arange(1, all_stats_03[2].shape[0]) * stats_interval, 
              all_stats_03[2][1:, keys[k]])

for k in ['acc(valid)']:
    ax_16.plot(np.arange(1, all_stats_03[3].shape[0]) * stats_interval, 
              all_stats_03[3][1:, keys[k]])

for k in ['acc(valid)']:
    ax_16.plot(np.arange(1, all_stats_03[4].shape[0]) * stats_interval, 
              all_stats_02[4][1:, keys[k]])

ax_16.legend(loc=0)

plt.legend(loc='best')

fig_1.text(0.5, 0.04, 'Epoch', ha='center', fontsize=16)
fig_1.text(0.04, 0.5, 'Step Size', va='center', rotation='vertical', fontsize=16)

plt.suptitle("Comparison of Sigmoid with ReLU Variant Activation Fuctions", fontsize=16)

plt.savefig('sigmoid_vs_relu.svg', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='svg',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)

plt.savefig('sigmoid_vs_relu.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format='png',
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None)