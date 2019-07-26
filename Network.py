from math import exp
from random import seed
from random import random


# Initialize network

def initNetwork(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs
                    + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden
                    + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate activation for neuron

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation

def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Forward propagate input to a network output

def fProp(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Derivative of tanh

def dSigmoid(out):
    return out * (1.0 - out)


# Backpropagate error and store in neurons

def bPropError(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] \
                * dSigmoid(neuron['output'])


# Update network weights with error

def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] \
                    * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs

def trainNetwork(network, train, l_rate, n_epoch, n_outputs,):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = fProp(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in
                             range(len(expected))])
            bPropError(network, expected)
            update_weights(network, row, l_rate)

        print( '>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate,
                sum_error))



seed(1)
dataset = [
    [1, 1, 1, 1, 0],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 0, 1, 1, 0],
]
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
print(n_outputs)
print(n_inputs)
network = initNetwork(n_inputs, 3, n_outputs)
trainNetwork(network, dataset, 10, 1000, n_outputs)
test = [1,1,0,0]
out = fProp(network, test)
print(out.index(max(out)))
