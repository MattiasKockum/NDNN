#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an AI
    capable of selective memory
    capable of solving real time problems fast
    capable of simulating a Turing Machine
    It's supposed to be able to do convolution but not the best way
"""

from ActivationFunctions import *

import numpy as np
import matplotlib.pyplot as plt

from datetime import date


class NDNN(object):
    """
    A neural network, but won directional
    it has input, output, and hidden neurons
    and can process the input multiple time,
    sometime making it pass throug the same neurons more than once.
    I expect it to make them faster, smaller, and capable of making faster
    "life or death" decisions given the fact that the input neurons are in
    direct contact with the output neurons
    (and some other neurons make a short cut too),
    so I believe the Network will have both fast and slow thinking.
    """
    def __init__(
            self,
            nb_sensors = 1, # 1 neurone = 1 sensor and 0 actor
            nb_actors = 0,
            nb_add_neurons = 0,
            function = segments,
            weights = None,
            bias = None):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.function = function
        self.nb_neurons = nb_add_neurons + nb_actors + nb_sensors
        self.values = np.zeros((self.nb_neurons))

        if type(weights) == type(None):
            self.weights = self.generate_weights()
        else:
            self.weights = weights

        if type(bias) == type(None):
            self.bias = self.generate_bias()
        else:
            self.bias = bias

    def generate_weights(self, a=1):
        A = np.zeros((self.nb_neurons, self.nb_neurons))
        for i in range(self.nb_neurons):
            for j in range(self.nb_neurons):
                A[i][j] = np.random.pareto(a)*np.random.choice([1, -1])
        return(A)

    def generate_bias(self):
        return(np.random.rand(self.nb_neurons) - 0.5)

    def process(self, input_data):
        """
        What the network does
        """
        self.input(input_data)
        self.iterate()
        return(self.output())

    def input(self, values_inputs):
        self.values[:self.nb_sensors] += values_inputs

    def output(self):
        return(self.values[-self.nb_actors:])

    def iterate(self):
        """
        Iterate once and update network state
        """
        self.values = self.function(
            np.matmul(self.weights, self.values + self.bias))

    def reset(self):
        self.values = np.zeros(self.values.shape)


# Display

def displayNetwork(network):
    """
    Represents the network
    """
    fig, ax = plt.subplots()
    array = np.concatenate((
        np.array([network.values]),
        np.array([network.bias]),
        network.weights
    ))
    im = ax.imshow(array)
    ax.set_yticks(np.arange(network.nb_neurons + 2))
    ax.set_yticklabels(["values"] + ["bias"]
                       + ["weight"]*network.nb_neurons)
    ax.set_xticks(np.arange(network.nb_neurons))
    ax.set_xticklabels(
        ["sensor"]*network.nb_sensors
        + ["deep neuron"]*network.nb_add_neurons
        + ["actor"]*network.nb_actors
    )
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )
    ax.set_title("values of internal values, weights and bias")
    fig.tight_layout()
    plt.show()

def displayNetworkGrid(net, period=1, size=10, step=0.1):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        a = size
        b = -size
        X = np.arange(b, a, step)
        Y = np.arange(b, a, step)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        for x in range(X.shape[0]):
            for y in range(X.shape[1]):
                for i in range(period):
                    Z[x][y] = net.process([x*step+b, y*step+b])
                net.reset()

        # Plot the surface.
        surf = ax.plot_wireframe(X, Y, Z, color="green")

        plt.show()



# Save function

def save_network(network, file_name = None, mode = "a", add_date = True,
         extension=".svn"):
    """
    Saves the network into a file
    """
    if file_name == None:
        file_name = "NDNN"
    if add_date:
        file_name += str(date.today())
    file_name += extension
    f = open(file_name, mode)
    f.write(
        "Network\n"
        + str(network.nb_sensors) + "\n"
        + str(network.nb_actors) + "\n"
        + str(network.nb_add_neurons) + "\n"
    )
    for i in network.weights:
        for j in i:
            f.write(str(j) + "\n")
    for i in network.bias:
        f.write(str(i) + "\n")
    for i in network.values:
        f.write(str(i) + "\n")
    f.close()


def load_networks(file_name):
    """
    Extract all the Networks from a save file and put them in a list
    I lose data in the process, must investigate that
    seems that np.float64 doesn't do the job
    """
    f = open(file_name, "r")
    r = []
    exit = False
    while not exit:
        if f.readline() == "Network\n":
            nb_sensors = int(float(f.readline()))
            nb_actors = int(float(f.readline()))
            nb_add_neurons = int(float(f.readline()))
            size = nb_sensors + nb_add_neurons + nb_actors
            weights = np.zeros((size, size))
            bias = np.zeros((size))
            values = np.zeros((size))
            for i in range(size**2):
                weights[i//size][i%size] = np.float64(f.readline())
            for i in range(size):
                bias[i] = np.float64(f.readline())
            for i in range(size):
                values[i] = np.float64(f.readline())
            N = NDNN(nb_sensors, nb_actors, nb_add_neurons)
            N.weights = weights
            N.bias = bias
            r.append(N)
        else:
            exit = True
    return(r)
