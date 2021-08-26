#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
This program does the evolution part
"""

from NDNN import *

# Necessary
import numpy as np
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
            **kwargs):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.function = function
        self.size = size
        self.mutation_amplitude = mutation_amplitude
        self.members = [
            NDNN(self.nb_sensors, self.nb_actors, self.nb_add_neurons,
                    self.function, **kwargs)
            for i in range(self.size)
        ]

    def evolve(self, problem, nb_generations=1, save_evo_name=None,
              final_herd_save_name = "Final_Herd"):
        """
        The idea is to make the AI evolve by aproximating
        the gradient descent
        """
        for generation in range(nb_generations):
            # Evaluation of performances
            proba_reproduction = self.performances(problem)
            # Reproduction (with mutation) of Networks
            self.reproduce(proba_reproduction)
            if save_evo_name != None:
                save_network(self.members[0], save_evo_name)

        if problem.do_end_display:
            problem.end_display()

        # Save the entire herd at the end
        for member in self.members:
            save_network(member, final_herd_save_name)

    def performances(self, problem):
        """
        Evaluates performances of all the networks on the problem
        then normalises them for probability operations
        """
        score = problem.experience(self.members)
        # Normalisation #
        #print(score)
        score = score - min(score)
        if list(score) == list(np.zeros(self.size)):
            score = np.ones(self.size)
        return(score/sum(score))

    def reproduce(self, proba_reproduction):
        """
        The copy of the successful networks with mutation
        """
        new_members = [deepcopy_network(np.random.choice(
                        self.members, p=proba_reproduction))
                        for i in range(self.size)]
        for member in new_members:
            mutate_network(member, self.mutation_amplitude)
        self.members = new_members


# Copying network function

def deepcopy_network(network):
    copy_network = NDNN(
        network.nb_sensors,
        network.nb_actors,
        network.nb_add_neurons,
        network.function,
        network.weights,
        network.bias
    )
    return(copy_network)


# Mutating network function

def mutate_network(network, mutation_amplitude):
    """
    Mutates the given NDNN
    """
    N = network.nb_neurons

    mut_weight = np.random.rand(N, N) - 0.5
    mut_biais = np.random.rand(N) - 0.5

    network.weights += mutation_amplitude * mut_weight
    network.bias += mutation_amplitude * mut_biais




# Save functions

def load_Herd(file_name, size = 100, mc = 0.1, ma = 0.001):
    """
    Recreate a Herd based on the saved NDNN
    """
    N = load_network(file_name)[0]
    H = Herd(N.nb_sensors, N.nb_actors, N.nb_add_neurons, size,
             mc, ma, weights = N.weights, bias = N.bias)
    return(H)


# Misc functions

def date():
    t = time.localtime()
    return("_{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4]))

