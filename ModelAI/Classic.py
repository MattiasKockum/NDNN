#!/usr/bin/env python3

"""
This is here to prove the fact that given a certain structure
The NDNN is capable of the same things a classical NN is
"""

import sys
sys.path.append("..")

from AI import *

def under_diag(size):
    matrix = np.zeros((size, size))
    for i in range(size-1):
        matrix[i+1][i] = 1
    return(matrix)

class ClassicalNetwork(Network):
    """
    The NDNN reshaped
    """
    def __init__(self, layers, function=sigmoid):
        self.nb_sensors = layers[0]
        self.nb_actors = layers[-1]
        self.nb_add_neurons = sum(layers[1:-1])
        self.function = function
        self.nb_neurons = (self.nb_sensors + self.nb_actors +
                self.nb_add_neurons)
        self.values = np.zeros((self.nb_neurons))

        self.period = len(layers) -1

        self.directive = [[0]*len(layers)]*len(layers)
        layers_sum = [0]
        for i in layers:
            layers_sum.append(layers_sum[-1] + i)
        self.weights = np.zeros((self.nb_neurons, self.nb_neurons))
        l = len(layers)
        for indice_i, i in enumerate(range(l)):
            for indice_j, j in enumerate(range(l)):
                self.weights[
                    layers_sum[i]:layers_sum[i + 1],
                    layers_sum[j]:layers_sum[j + 1]
                ] = (
                    under_diag(len(layers))[i][j]
                    * (
                        np.random.rand(
                            layers[indice_i],
                            layers[indice_j]
                        )
                        - 0.5
                    )
                )
        self.bias = np.random.rand(self.nb_neurons) - 0.

    def process(self, input_data):
        """
        What the network does
        """
        self.input(input_data)
        for i in range(self.period):
            self.iterate()
        return(self.output())
