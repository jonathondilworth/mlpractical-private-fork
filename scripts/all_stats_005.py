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

def train_model_and_plot_stats(
        model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True, output_p=True, name="test"):
    
    # As well as monitoring the error over training also monitor classification
    # accuracy i.e. proportion of most-probable predicted classes being equal to targets
    data_monitors={'acc': lambda y, t: (y.argmax(-1) == t.argmax(-1)).mean()}

    # Use the created objects to initialise a new Optimiser instance.
    optimiser = Optimiser(
        model, error, learning_rule, train_data, valid_data, data_monitors, notebook=notebook)

    # Run the optimiser for 5 epochs (full passes through the training set)
    # printing statistics every epoch.
    stats, keys, run_time = optimiser.train(num_epochs=num_epochs, stats_interval=stats_interval)

    if output_p:
        print(stats)
        #np.savetxt(name + ".csv", stats, delimiter=",")
    
    # Plot the change in the validation and training set error over training.
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['error(train)', 'error(valid)']:
        ax_1.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_1.legend(loc=0)
    ax_1.set_xlabel('Epoch number')

    # Plot the change in the validation and training set accuracy over training.
    fig_2 = plt.figure(figsize=(8, 4))
    ax_2 = fig_2.add_subplot(111)
    for k in ['acc(train)', 'acc(valid)']:
        ax_2.plot(np.arange(1, stats.shape[0]) * stats_interval, 
                  stats[1:, keys[k]], label=k)
    ax_2.legend(loc=0)
    ax_2.set_xlabel('Epoch number')
    
    return stats, keys, run_time, fig_1, ax_1, fig_2, ax_2

# Seed a random number generator
seed = 6102016 
rng = np.random.RandomState(seed)
batch_size = 50

# Set up a logger object to print info about the training run to stdout
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [logging.StreamHandler()]

# Create data provider objects for the MNIST data set
train_data = MNISTDataProvider('train', batch_size=batch_size, rng=rng)
valid_data = MNISTDataProvider('valid', batch_size=batch_size, rng=rng)

rng.seed(seed)
train_data.reset()
valid_data.reset()

def train_and_process(p_seed, p_learning_rate, p_num_epochs, p_stats_interval):

	r_stats = [None] * 5

	# SIGMOID BASELINE

	rng.seed(p_seed)
	train_data.reset()
	valid_data.reset()

	learning_rate = p_learning_rate
	num_epochs = p_num_epochs
	stats_interval = p_stats_interval
	input_dim, output_dim, hidden_dim = 784, 10, 100

	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)
	model = MultipleLayerModel([
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    SigmoidLayer(),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    SigmoidLayer(),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init),
	])

	error = CrossEntropySoftmaxError()
	# Use a basic gradient descent learning rule
	learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

	r_stats[0] = _[0]

	# RELU BASELINE

	rng.seed(p_seed)
	train_data.reset()
	valid_data.reset()

	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)
	model = MultipleLayerModel([
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    ReluLayer(),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    ReluLayer(),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init),
	])

	error = CrossEntropySoftmaxError()
	# Use a basic gradient descent learning rule
	learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

	r_stats[1] = _[0]

	# LRELU

	rng.seed(p_seed)
	train_data.reset()
	valid_data.reset()

	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)
	model = MultipleLayerModel([
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    LeakyReluLayer(),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    LeakyReluLayer(),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init),
	])

	error = CrossEntropySoftmaxError()
	# Use a basic gradient descent learning rule
	learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

	r_stats[2] = _[0]

	# ELU

	rng.seed(p_seed)
	train_data.reset()
	valid_data.reset()

	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)
	model = MultipleLayerModel([
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    ELULayer(),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    ELULayer(),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init),
	])

	error = CrossEntropySoftmaxError()
	# Use a basic gradient descent learning rule
	learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

	r_stats[3] = _[0]

	# SELU

	rng.seed(p_seed)
	train_data.reset()
	valid_data.reset()

	weights_init = GlorotUniformInit(rng=rng)
	biases_init = ConstantInit(0.)

	model = MultipleLayerModel([
	    AffineLayer(input_dim, hidden_dim, weights_init, biases_init), 
	    SELULayer(),
	    AffineLayer(hidden_dim, hidden_dim, weights_init, biases_init), 
	    SELULayer(),
	    AffineLayer(hidden_dim, output_dim, weights_init, biases_init),
	])

	error = CrossEntropySoftmaxError()
	# Use a basic gradient descent learning rule
	learning_rule = GradientDescentLearningRule(learning_rate=learning_rate)

	#Remember to use notebook=False when you write a script to be run in a terminal
	_ = train_model_and_plot_stats(
	    model, error, learning_rule, train_data, valid_data, num_epochs, stats_interval, notebook=True)

	r_stats[4] = _[0]

	return r_stats

all_stats_005_1 = train_and_process(6102016, 0.05, 100, 1)
all_stats_005_2 = train_and_process(1237878, 0.05, 100, 1)
all_stats_005_3 = train_and_process(2673676, 0.05, 100, 1)
all_stats_005_4 = train_and_process(8978283, 0.05, 100, 1)
all_stats_005_5 = train_and_process(5627351, 0.05, 100, 1)

all_stats_005_sigmoid = [all_stats_005_1[0], all_stats_005_2[0], all_stats_005_3[0], all_stats_005_4[0], all_stats_005_5[0]]
all_stats_005_relu = [all_stats_005_1[1], all_stats_005_2[1], all_stats_005_3[1], all_stats_005_4[1], all_stats_005_5[1]]
all_stats_005_lrelu = [all_stats_005_1[2], all_stats_005_2[2], all_stats_005_3[2], all_stats_005_4[2], all_stats_005_5[2]]
all_stats_005_elu = [all_stats_005_1[3], all_stats_005_2[3], all_stats_005_3[3], all_stats_005_4[3], all_stats_005_5[3]]
all_stats_005_selu = [all_stats_005_1[4], all_stats_005_2[4], all_stats_005_3[4], all_stats_005_4[4], all_stats_005_5[4]]

sigmoid_mean = np.mean(all_stats_005_sigmoid, axis=0)
relu_mean = np.mean(all_stats_005_relu, axis=0)
lrelu_mean = np.mean(all_stats_005_lrelu, axis=0)
elu_mean = np.mean(all_stats_005_elu, axis=0)
selu_mean = np.mean(all_stats_005_selu, axis=0)

#print(sigmoid_mean)
#print(relu_mean)
#print(lrelu_mean)
#print(elu_mean)
#print(selu_mean)

all_stats_005_avg = [sigmoid_mean, relu_mean, lrelu_mean, elu_mean, selu_mean]

np.save('all_stats_005_avg.npy', all_stats_005_avg)

with open('all_stats_005_avg.txt', 'w') as f: f.write(json.dumps(all_stats_005_avg, default=lambda x: list(x), indent=4))