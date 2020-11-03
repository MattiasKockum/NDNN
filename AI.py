#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an AI
    capable of selective memory
    capable of solving real time problems fast
    capable of simulating a Turing Machine

The training is parallelized
"""

import numpy as np
import multiprocessing as mp
import copy
import turtle
import matplotlib.pyplot as plt


def sigmoid(x):
	return(2*((1/(1+np.e**(-x)))-0.5))

def extend(array, n=1):
    r = []
    for i in array:
        for j in range(n):
            r.append(copy.deepcopy(i))
    return(r)

def mean(array, n=1):
    r = []
    array += [0]*(-len(array)%n)
    for i in range(0, len(array), n):
        r.append(sum(array[i:i+n])/n)
    return(r)

def prob_reproduction(X):
    """
    A weird looking function for parallelization
    X[0] is a group of objects
    X[1] is their respective probability of beig copied
    X[2] = mutation_coefficent
    X[3] = mutation_amplitude
    returns the mutation of the chosen one
    """
    #np.random.seed() # without it every thread has the same RNG
    return(np.random.choice(X[0], p=X[1]).mutate(X[2], X[3]))

def evaluate(X):
    """
    Another weird looking function
    X[0] is a problem
    X[1] is a network
    returns the score of the network
    """
    return(X[0].experience(X[1]))


class Problem(object):
    """
    The frame of any "live" problem
    """
    def __init__(self):
        self.nb_sensors = 1
        self.nb_actors = 1
        self.period = 1
        print("Warning  : __init__ was not fully configured") 

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        while not self.end_condition():
            self.action(*Network.process(self.state(), self.period))
        score = self.score_real_time()
        self.reset()
        # score should always be > 0
        return(score*(score>0) + 0)

    def end_condition(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        print("Warning  : end_condition was not fully configured")
        return(True)

    def state(self):
        """
        Returns the state of the problem
        """
        print("Warning  : state was not fully configured")
        return(np.array([1]))

    # Other state related functions should be there

    def score_real_time(self):
        """
        Returns the score of the problem at the moment
        """
        print("Warning score_real_time was not fully configured")
        return(0)

    def action(self, input_data):
        """
        Computes the consequences of the input_data on the problem
        """
        print("Warning  : action was not fully configured")
        pass

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        print("Warning : experience was not fully configured")

    # Other display functions should be put here

    def reset(self):
        """
        Resets the problem
        """
        self.__init__()


class Herd(object):
    """
    Herd of networks that evolve by reproducing
    I should get the self.problem off
    """
    def __init__(
        self,
        nb_sensors = 1,
        nb_actors = 1,
        nb_add_neurons = 0,
        problem = None,
        size = 5,
        mutation_coefficent = 0.1,
        mutation_amplitude = 0.001,
        nb_tests = 2,
        **kwargs
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        if problem == None:
            self.Problem = Problem()
        else:
            self.Problem = problem
        self.size = size
        self.mutation_coefficent = mutation_coefficent
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.Problem_pool = extend([self.Problem], self.size*self.nb_tests)
        self.members = [
            Network(nb_sensors, nb_actors, nb_add_neurons, **kwargs)
            for i in range(size)
        ]
        self.array_scores = []

    def evolve(self, nb_generations=1):
        """
        The idea is to make the AI evolve by aproximating the gradient descent
        """
        for generation in range(nb_generations):
            # Evaluation of performances
            proba_reproduction = self.performances()
            # Reproduction (with mutation) of Networks
            self.reproduce(proba_reproduction)
            # Saves the scores
            self.array_scores.append(sum(self.score)/self.size)
        return(self.array_scores)

    def reproduce(self, proba_reproduction):
        """
        The copy of the successful networks with mutation
        parallelized
        """
        self.members = (
            mp.Pool().map(
                prob_reproduction,
                [(
                    self.members,
                    proba_reproduction,
                    self.mutation_coefficent,
                    self.mutation_amplitude
                )]*self.size
            )
        )

    def performances(self):
        """
        Evaluates performances then normalises them for probability operations
        Can be parallelized
        """
        self.members_pool = extend(self.members, self.nb_tests)
        # parallelize the evaluation of the networks
        # Still don't know why but paralleliztion here slows the hole thing
        member_s_points = mp.Pool().map(
            evaluate,
            [(P, M) for P,M in zip(self.Problem_pool, self.members_pool)]
        )
        self.score = mean(member_s_points, self.nb_tests)
        # if evey Network has a score of zero they reproduce with equal
        # proability
        if list(self.score) == list(np.zeros(self.size)):
            self.score = np.ones(self.size)
        score_modif = self.modif_score(self.score)
        return(score_modif)

    def modif_score(self, score):
        """
        Modifies the scores to make them useable in probability
        """
        return(score/sum(score))

    def scale(self, reproductive_members):
        """
        NOT USED YET
        scales up the members to make them able to reproduce
        """
        add_neurons = [0]*len(reproductive_members)
        nb_add_neurons = reproductive_members[0].nb_neurons
        for index, member in enumerate(reproductive_members[1:]):
            difference = member.nb_neurons - nb_add_neurons
            if difference > 0:
                nb_add_neurons = member.nb_neurons
                for i in range(index+1):
                    add_neurons[i] += difference
            else:
                add_neurons[index+1] -= difference
        for index, member in enumerate(reproductive_members):
            member.add_neurons(add_neurons[index])
        return(reproductive_members[0].nb_neurons)


class Network(object):
    """
    """
    def __init__(
        self,
        nb_sensors = 1,
        nb_actors = 0,
        nb_add_neurons = 0,
        **kwargs # "weight", "bias", "slices", "regions"
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.nb_neurons = nb_add_neurons + nb_actors + nb_sensors
        self.values = np.zeros((self.nb_neurons))
        if ("slices" in kwargs and "regions" in kwargs):
            # slices = list of size of groups of deep neurons
            # regions = array of 0 and 1 if the group shall exist
            self.slices = [nb_sensors] + kwargs["slices"] + [nb_actors]
            self.regions = kwargs["regions"]
            self.squared()
        elif "weight" not in kwargs and "bias" not in kwargs:
            self.random_set_up()
        elif (
            kwargs["weights"].shape == (nb_neurons, nb_neurons)
            and kwargs["bias"].shape == (nb_neurons,)
        ):
            self.weights = kwargs["weights"]
            self.bias = kwargs["bias"]
        else :
            raise(ValueError("Input matrices do not have the right format\
                    or both weights and bias or both slices and \
                    regions must be entered"))

    def random_set_up(self):
        self.weights = (
            np.random.rand(self.nb_neurons, self.nb_neurons)
            - 0.5
        )
        self.bias = np.random.rand(self.nb_neurons) - 0.5

    def squared(self):
        self.directive = [[0]*len(self.slices)]*len(self.slices)
        self.slices_sum = [0]
        for i in self.slices:
            self.slices_sum.append(self.slices_sum[-1] + i)
        self.weights = np.zeros((self.nb_neurons, self.nb_neurons))
        l = len(self.slices)
        for indice_i, i in enumerate(range(l)):
            for indice_j, j in enumerate(range(l)):
                self.weights[
                    self.slices_sum[i]:self.slices_sum[i + 1],
                    self.slices_sum[j]:self.slices_sum[j + 1]
                ] = (
                    self.regions[i][j]
                    * (
                        np.random.rand(
                            self.slices[indice_i],
                            self.slices[indice_j]
                        )
                        - 0.5
                    )
                )
        self.bias = np.random.rand(self.nb_neurons) - 0.5

    def __repr__(self):
        return("neurons : {}\n".format(self.nb_neurons)
               + "sensors : {}\n".format(self.nb_sensors)
               + "actors : {}\n".format(self.nb_actors)
               + "weights :\n{}\n".format(self.weights)
               + "bias :\n{}\n".format(self.bias)
               + "values :\n{}\n".format(self.values))

    def display(self):
        """
        Represents the network with
        """
        fig, ax = plt.subplots()
        array = np.concatenate((
            np.array([self.values]),
            np.array([self.bias]),
            self.weights
        ))
        im = ax.imshow(array)
        ax.set_yticks(np.arange(self.nb_neurons + 2))
        ax.set_yticklabels(["values"] + ["bias"] + ["weight"]*self.nb_neurons)
        ax.set_xticks(np.arange(self.nb_neurons))
        ax.set_xticklabels(
            ["sensor"]*self.nb_sensors
            + ["deep neuron"]*self.nb_add_neurons
            + ["actor"]*self.nb_actors
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

    def input(self, values_inputs):
        self.values[:self.nb_sensors] += values_inputs

    def output(self):
        return(self.values[-self.nb_actors:])

    def iteration(self):
        """
        We iterate once and update network state
        Hopefully I can parallelize the matrix multiplication
        """
        self.values = sigmoid(
            np.matmul(self.weights, self.values)
            + self.bias)

    def process(self, input_data, nb_iterations=1):
        """
        What the network does
        """
        self.input(input_data)
        for i in range(nb_iterations):
            self.iteration()
        return(self.output())

    def add_neurons(self, add_neurons=1):
        """
        NOT USED YET
        """
        # Creates bigger weights and bias arrays
        new_weights = np.zeros((self.weights.shape[0] + add_neurons,
                                   self.weights.shape[1] + add_neurons))
        new_bias = np.zeros((self.bias.shape[0] + add_neurons, ))
        nouvelles_values = np.zeros(self.nb_neurons + add_neurons)
        # Adds old values
        new_weights[:self.nb_sensors, :self.nb_sensors] = (
            self.weights[:self.nb_sensors, :self.nb_sensors])
        new_weights[-self.nb_sensors-1:, :self.nb_sensors] = (
            self.weights[-self.nb_sensors-1:, :self.nb_sensors])
        new_weights[-self.nb_sensors-1:, -self.nb_sensors-1:] = (
            self.weights[-self.nb_sensors-1:, -self.nb_sensors-1:])
        new_weights[:self.nb_sensors, -self.nb_sensors-1:] = (
            self.weights[:self.nb_sensors, -self.nb_sensors-1:])
        new_bias[:self.nb_sensors] = (
            self.bias[:self.nb_sensors])
        new_bias[-self.nb_sensors-1:] = (
            self.bias[-self.nb_sensors-1:])
        nouvelles_values[:self.nb_sensors] = (
            self.values[:self.nb_sensors])
        nouvelles_values[-self.nb_sensors-1:] = (
            self.values[-self.nb_sensors-1:])
        # Updates the network
        self.weights = new_weights
        self.bias = new_bias
        self.values = nouvelles_values
        self.nb_neurons += add_neurons
        # Returns the new number of nerons
        return(self.nb_neurons)

    def reset(self):
        self.values = np.zeros(self.values.shape)

    def mutate(self, mutation_coefficent, mutation_amplitude):
        """
        Return the mutated Network
        """
        np.random.seed() # For multiprocressing
        mn = copy.deepcopy(self) # mutated network
        for i in range(mn.nb_neurons*(mn.nb_neurons + 1)):
            # If there is a mutation
            if np.random.choice(
                [True, False],
                p = [mutation_coefficent, 1 - mutation_coefficent]
            ):
                # If the iterator corresponds to a weight, we modify it
                if i < mn.nb_neurons**2:
                    mn.weights[i//mn.nb_neurons][i%mn.nb_neurons] += (
                        np.random.normal(0, mutation_amplitude)
                    )
                # Elsif it corresponds to a bias we modify it
                elif i < mn.nb_neurons*(mn.nb_neurons + 1):
                    mn.bias[i - mn.nb_neurons**2] += (
                        np.random.normal(0, mutation_amplitude)
                    )
        return(mn)


class TestBench(object):
    """
    A test bench to verify everything works fine
    """
    def __init__(
        self,
        problem,
        nb_herds = 1,
        nb_generations = 20,
        nb_add_neurons = 9,
        size = 100,
        mutation_coefficent = 0.0001,
        mutation_amplitude = 0.01,
        nb_tests = 2,
        display_mode = None,
        **kwargs
    ):
        self.kwargs = kwargs
        if "slices" in kwargs:
            self.nb_add_neurons = sum(kwargs["slices"])
        else:
            self.nb_add_neurons = nb_add_neurons
        self.series = []
        self.problem = problem
        self.nb_sensors = problem.nb_sensors
        self.nb_actors = problem.nb_actors
        self.colors = ["r", "g", "b", "c", "m", "y", "k"]
        self.nb_herds = nb_herds
        self.nb_generations = nb_generations
        self.size = size
        self.mutation_coefficent = mutation_coefficent
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.display_mode = display_mode
        self.values_simple = self.nb_herds*[1]
        self.values_nb_add_neurons = [0, 1, 2, 3, 4, 5, 6]
        self.values_sizes = [5, 10, 50, 100, 500, 1000]
        self.values_mutation_coefficients = [0.0001, 0.000005, 0.00001]
        self.values_mutation_amplitude = [0.01, 0.005, 0.001]
        self.values_nb_tests = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.archives = []

    def test(self, mode = 0, nb_generations = None, values = None):
        if nb_generations == None:
            nb_generations = self.nb_generations
        base = [
            self.nb_sensors,
            self.nb_actors,
            self.nb_add_neurons,
            self.problem,
            self.size,
            self.mutation_coefficent,
            self.mutation_amplitude,
            self.nb_tests
        ]
        if mode in [0, "simple"]:
            if values == None:
                values = self.values_simple
            array_inputs = np.array([base for i in range(len(values))])
        if mode in [1, "nb_neurons"]:
            if values == None:
                values = self.values_nb_add_neurons
            array_inputs = np.array([base for i in range(len(values))])
            array_inputs[:,2] = values
        if mode in [2, "size"]:
            if values == None:
                values = self.values_sizes
            array_inputs = np.array([base for i in range(len(values))])
            array_inputs[:,4] = values
        if mode in [3, "coefficient_mutation"]:
            if values == None:
                values = self.values_mutation_coefficients
            array_inputs = np.array([base for i in range(len(values))])
            array_inputs[:,5] = values
        if mode in [4, "coefficient_amplitude"]:
            if values == None:
                values = self.values_mutation_amplitude
            array_inputs = np.array([base for i in range(len(values))])
            array_inputs[:,6] = values
        if mode in [5, "nb_tests"]:
            if values == None:
                values = self.values_nb_tests
            array_inputs = np.array([base for i in range(len(values))])
            array_inputs[:,6] = values
        if mode in [6, "multiple"]:
            if values == None:
                raise(ValueError("An array must be in input"))
            array_inputs = np.array([base for i in range(len(values))])
            array_inputs = values
        print(
            "\n",
            "(1) nb_captors\n",
            "(2) nb_actors\n",
            "(3) nb_add_neurons\n",
            "(4) Problem\n",
            "(5) herd's size\n",
            "(6) mutation_coefficent\n",
            "(7) mutation_amplitude\n",
            "(8) nb_tests\n",
            "(9) colors\n",
            "\n",
            "(1)(2)(3)(4)(5)(6)(7)(8)(9)"
        )
        for i in range(len(values)):
            print(*array_inputs[i], self.colors[i%len(self.colors)])
            H = Herd(*array_inputs[i], **self.kwargs)
            self.series.append(H.evolve(nb_generations))
            self.archives.append([H.members[0], self.series])
        if self.display_mode != None:
            if self.display_mode in [0, "console"]:
                self.display_console()
            if self.display_mode in [1, "plot"]:
                self.display_plot()
        self.series = []

    def display_console(self, archive = False):
        if archive:
            display_series = [i[1][0] for i in self.archives]
        else:
            display_series = self.series
        for indice, serie in enumerate(display_series):
            color = self.colors[indice%len(self.colors)]
            print("-- serie n°: {} -- color : {} --\n".format(indice, color))
            for i in serie:
                print("      {}".format(i))

    def display_plot(self, archive = False):
        if archive:
            display_series = [i[1][0] for i in self.archives]
        else:
            display_series = self.series
        for indice, serie in enumerate(display_series):
            color = self.colors[indice%len(self.colors)]
            plt.plot(
                [k for k in range(len(serie))],
                serie,
                color+"-*"
            )
        plt.show()


