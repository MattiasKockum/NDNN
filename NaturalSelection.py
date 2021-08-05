#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
This program does the evolution part
"""

from NDNN import *

# Necessary
import numpy as np
import copy
import time




class Herd(object):
    """
    Herd of networks that evolve by reproducing
    """
    def __init__(
        self,
        nb_sensors = 1,
        nb_actors = 0,
        nb_add_neurons = 0,
        function = segments,
        size = 2,
        mutation_amplitude = 0.01,
        nb_tests = 1,
        **kwargs
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.function = function
        self.size = size
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.members = [
            NDNN(self.nb_sensors, self.nb_actors, self.nb_add_neurons,
                    self.function, **kwargs)
            for i in range(self.size)
        ]

    def evolve(self, problem, nb_generations=1):
        """
        The idea is to make the AI evolve by aproximating
        the gradient descent
        """
        for generation in range(nb_generations):
            # Evaluation of performances
            proba_reproduction = self.performances(problem)
            # Reproduction (with mutation) of NDNNs
            self.reproduce(proba_reproduction)

        if problem.do_end_display:
            problem.end_display()

    def performances(self, problem):
        """
        Evaluates performances of all the networks on the problem
        then normalises them for probability operations
        """
        score = np.zeros(self.size)

        # Evaluation #
        for index, member in enumerate(self.members):
            for i in range(self.nb_tests):
                score[index] += problem.experience(member)
                member.reset()

        # Normalisation #
        score = score - min(score)
        if list(score) == list(np.zeros(self.size)):
            score = np.ones(self.size)

        return(score/sum(score))

    def reproduce(self, proba_reproduction):
        """
        The copy of the successful networks with mutation
        """
        new_members = [copy.deepcopy(np.random.choice(
                        self.members, p=proba_reproduction))
                        for i in range(self.size)]
        for member in new_members:
            mutate(member, self.mutation_amplitude)
        self.members = new_members


# Mutation function

def mutate(network, mutation_amplitude):
    """
    Mutates the given NDNN
    """
    N = network.nb_neurons

    mut_weight = np.random.rand(N, N) - 0.5
    mut_biais = np.random.rand(N) - 0.5

    network.weights += mutation_amplitude * mut_weight
    network.bias += mutation_amplitude * mut_biais




# Save functions

def load_Herd(file_name, size = 100, mc = 0.1, ma = 0.001, nb_tests = 2):
    """
    Recreate a Herd based on the saved NDNN
    """
    N = load_network(file_name)[0]
    H = Herd(N.nb_sensors, N.nb_actors, N.nb_add_neurons, size,
             mc, ma, nb_tests, weights = N.weights, bias = N.bias)
    return(H)

# Misc functions

def date():
    t = time.localtime()
    return("_{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4]))

def maxindex(array):
    """
    Very useful for saying which of the output neurons is
    the most activated
    """
    return(list(array).index(max(array)))

