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

# Necessary
import numpy as np
import multiprocessing as mp
import copy
import time
# Useful for compiling Network in machine code
import os
# Useful for easy data visualisation
import matplotlib.pyplot as plt


# activation functions

def sigmoid(x):
	return(-1 + 2/(1+np.e**(-x)))

def ramp(x):
    return(x*(x>0))

def threshold(x):
    return(1*(x>0) + 0 -1*(x<0))

# parallelization functions

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
    X[1] is their respective probability of being copied
    X[2] = mutation_coefficent
    X[3] = mutation_amplitude
    returns the mutation of the chosen one
    """
    return(np.random.choice(X[0], p=X[1]).mutate(X[2], X[3]))

def evaluate(X):
    """
    Another weird looking function
    X[0] is a problem
    X[1] is a network
    returns the score of the network
    """
    np.random.seed()
    X[0].reset()
    return_value = X[0].experience(X[1])
    X[1].reset()
    return(return_value)

def pooled_evolution(X):
    """
    Another one
    """
    r = X[0].evolve(X[1], X[2])
    return(r, (X[0].members[0], r))

# Save function

def load_network(file_name):
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
            period = int(float(f.readline()))
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
            N = Network(nb_sensors, nb_actors, nb_add_neurons, period,
                        weights=weights, bias=bias)
            r.append(N)
        else:
            exit = True
    return(r)

def load_Herd(file_name, size = 100, mc = 0.1, ma = 0.001, nb_tests = 2):
    """
    Recreate a Herd based on the saved Network
    """
    N = load_network(file_name)[0]
    H = Herd(N.nb_sensors, N.nb_actors, N.nb_add_neurons, N.period, size,
             mc, ma, nb_tests, weights = N.weights, bias = N.bias)
    return(H)

def load_score(file_name):
    """
    TO BE REWORKED
    """
    f = open(file_name, "r")
    r = []
    data = []
    for i in range(7):
        data.append(f.readline())
    nb_generations = int(f.readline()[35:])
    for i in range(nb_generations):
        line = f.readline()
        r.append(float(line[line.index(":") + 2:-2]))
    return(r)

# Misc functions

def date():
    t = time.localtime()
    return("_{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4]))

def under_diag(size):
    matrix = np.zeros((size, size))
    for i in range(size-1):
        matrix[i+1][i] = 1
    return(matrix)


class Problem(object):
    """
    The frame of any "live" problem
    """
    def __init__(self, do_display = False):
        self.nb_sensors = 1
        self.nb_actors = 1
        self.score = 0
        print("Warning  : __init__ was not fully configured")

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        while not self.end_condition():
            self.action(Network)
        self.score_update()
        score = self.score
        self.reset()
        return(score)

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

    def score_update(self):
        """
        Updates the score of the problem at the moment
        """
        print("Warning score_update was not fully configured")
        score = 0
        self.score = score*(score>0)
        # score should always be > 0

    def action(self, Network):
        """
        Computes the consequences of the input_data on the problem
        """
        print("Warning  : action was not fully configured")
        output = Network.process(self.state())
        pass

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        print("Warning : experience was not fully configured")

    # Other display functions should be put here

    def __name__(self):
        return("An_Unnamed_Problem")

    def reset(self):
        """
        Resets the problem
        """
        print("Maybe some HUGE PROBLEM is coming at you bro")
        self.__init__()


class Herd(object):
    """
    Herd of networks that evolve by reproducing
    """
    def __init__(
        self,
        nb_sensors = 1,
        nb_actors = 1,
        nb_add_neurons = 0,
        period = 1,
        size = 5,
        mutation_coefficent = 0.1,
        mutation_amplitude = 0.001,
        nb_tests = 1,
        do_display = False,
        **kwargs
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.period = period
        self.size = size
        self.mutation_coefficent = mutation_coefficent
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.do_display = do_display
        self.members = [
            Network(nb_sensors, nb_actors, nb_add_neurons, period, **kwargs)
            for i in range(size)
        ]
        self.array_scores = []
        self.date = date()
        self.max_score = 0
        self.max_score_index = 0

    def evolve(self, problem, nb_generations=1):
        """
        The idea is to make the AI evolve by aproximating the gradient descent
        Opens and closes the score output file multiple times so that it's
        possible to see what's going on in during the training
        """
        if problem == None:
            # The empty problem, just here for quick tests
            self.Problem = Problem()
        else:
            self.Problem = problem
        score_file = open(problem.__name__() + "_score" + self.date, "w")
        score_file.write(
            "score\n"
            + "number of added neurons : {}\n".format(self.nb_add_neurons)
            + "period : {}\n".format(self.period)
            + "size : {}\n".format(self.size)
            + "mutation coefficent : {}\n".format(self.mutation_coefficent)
            + "mutation_amplitude : {}\n".format(self.mutation_amplitude)
            + "number of tests : {}\n".format(self.nb_tests)
            + "number of generations to proceed : {}\n".format(nb_generations)
        )
        score_file.close()
        self.Problem_pool = extend([self.Problem], self.size*self.nb_tests)
        for pb in self.Problem_pool:
            pb.do_display = False
        if self.do_display:
            self.Problem_pool[0].do_display = True
        for generation in range(nb_generations):
            # Evaluation of performances
            proba_reproduction = self.performances()
            # Reproduction (with mutation) of Networks
            self.reproduce(proba_reproduction)
            # Saves the scores
            self.max_score = max(self.score)
            self.max_score_index = self.score.index(self.max_score)
            self.array_scores.append(self.max_score)
            # Saves one Network and the score evolution
            self.members[self.max_score_index].save(
                problem.__name__() + "_Network" + self.date, "w", False)
            score_file = open(problem.__name__() + "_score" + self.date, "a")
            score_file.write(
                "generation n° {} : {} \n".format(
                    generation, str(self.max_score)))
            score_file.close()
        score_file = open(problem.__name__() + "_score" + self.date, "a")
        score_file.write("End\n")
        score_file.close()
        return(self.array_scores)

    def performances(self):
        """
        Evaluates performances then normalises them for probability operations
        Can be parallelized
        """
        self.members_pool = extend(self.members, self.nb_tests)
        # parallelize the evaluation of the networks
        pool = mp.Pool()
        member_s_points = pool.map(
            evaluate,
            [(P, M) for P,M in zip(self.Problem_pool, self.members_pool)]
        )
        pool.close()
        # Put this code if you want to observe evolution, especially in the
        # Gradient Descent Problem because parallelization makes it not work
        #self.Problem.experience(self.members_pool[0])
        #self.members_pool[0].reset()
        self.score = mean(member_s_points, self.nb_tests)
        score_modif = self.modif_score(self.score)
        return(score_modif)

    def reproduce(self, proba_reproduction):
        """
        The copy of the successful networks with mutation
        parallelized
        """
        pool = mp.Pool()
        new_members = (
            pool.map(
                prob_reproduction,
                [(
                    self.members,
                    proba_reproduction,
                    self.mutation_coefficent,
                    self.mutation_amplitude
                )]*self.size
            )
        )
        self.members = new_members
        pool.close()

    def modif_score(self, score):
        """
        Modifies the scores to make them useable in probability
        """
        # I put the np.array in case the score isn't an array
        score = np.array(score)
        score = score - min(score)
        if list(score) == list(np.zeros(self.size)):
            # if evey Network has a score of zero they reproduce with equal
            # proability
            score = np.ones(self.size)
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
    A neural network, but won directional
    it has input, output, and hidden neurons
    and can process the input multiple time, sometime making it pass through
    the same neurons more than once.
    I expect it to make them faster, smaller, and capable of making faster
    "life or death" decisions given the fact that the input neurons are in
    direct contact with the output neurons (and some other neurons make a short
    cut too), so I believe the Network will have both fast and slow thinking.
    This is how the neurons are placed
    [
    input : [                 slowest thinking]
            [                                 ]
            [                                 ]
            [fastest thinking                 ]
                                          output
    ]
    """
    def __init__(
        self,
        nb_sensors = 1, # Note that if you want only one neuron just take
        nb_actors = 0, # one sensor and no actor
        nb_add_neurons = 0,
        period = 1,
        function = ramp,
        **kwargs # "weights", "bias", "slices", "regions"
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.period = period
        self.function = function
        self.nb_neurons = nb_add_neurons + nb_actors + nb_sensors
        self.values = np.zeros((self.nb_neurons))
        if ("slices" in kwargs and "regions" in kwargs):
            # slices = list of size of groups of deep neurons
            # regions = array of 0 and 1 if the group shall exist
            self.slices = kwargs["slices"]
            self.regions = kwargs["regions"]
            self.squared()
        elif "weights" not in kwargs and "bias" not in kwargs:
            self.random_set_up()
        elif (
            kwargs["weights"].shape == (self.nb_neurons, self.nb_neurons)
            and kwargs["bias"].shape == (self.nb_neurons,)
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

    def process(self, input_data, nb_iterations=1):
        """
        What the network does
        """
        self.input(input_data)
        for i in range(nb_iterations):
            self.iteration()
        return(self.output())

    def input(self, values_inputs):
        self.values[:self.nb_sensors] += values_inputs

    def output(self):
        return(self.values[-self.nb_actors:])

    def iteration(self):
        """
        We iterate once and update network state
        Hopefully I can parallelize the matrix multiplication
        """
        self.values = self.function(
            np.matmul(self.weights, self.values + self.bias))
        # Old way of doing, I'm testing this new one on top
        # I order to be able to make "dead connections"
        #self.values = self.function(
        #   np.matmul(self.weights, self.values) + self.bias)

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

    def save(self, file_name = None, mode = "a", add_date = True):
        """
        Saves the Network into a file
        """
        if file_name == None:
            file_name = "NDNN"
        if add_date:
            file_name += date()
        f = open(file_name, mode)
        f.write(
            "Network\n"
            + str(self.nb_sensors) + "\n"
            + str(self.nb_actors) + "\n"
            + str(self.nb_add_neurons) + "\n"
            + str(self.period) + "\n"
        )
        for i in self.weights:
            for j in i:
                f.write(str(j) + "\n")
        for i in self.bias:
            f.write(str(i) + "\n")
        for i in self.values:
            f.write(str(i) + "\n")
        f.close()

    def compile(self, c_code_name = None, add_date = False):
        """
        Saves a compiled and usable c version of the Network,
        this is intended to be the final thing to do before using the Network
        in its application
        """
        if c_code_name == None:
            c_code_name = "Exe"
        c_code_name += date() * add_date
        string_values = "{"
        for i in self.values:
            string_values += str(i) + ", "
        string_values = string_values[:-2]
        string_values += "}"
        string_bias = "{"
        for i in self.bias:
            string_bias += str(i) + ", "
        string_bias = string_bias[:-2]
        string_bias += "}"
        string_weights = "{"
        for i in self.weights:
            string_weights += "{"
            for j in i:
                    string_weights += str(j) + ", "
            string_weights = string_weights[:-2]
            string_weights+="}, "
        string_weights = string_weights[:-2]
        string_weights += "}"

        string_format_input = (self.nb_sensors*"%lf ")[:-1]
        string_input = ""
        for i in range(self.nb_sensors):
            string_input += "&values2[{}], ".format(i)
        string_input = string_input[:-2]
        string_format_output = (self.nb_actors*"%lf ")[:-1]
        string_output = ""
        for i in range(self.nb_actors):
            string_output += "N.values[NB_SENSORS + NB_ACTORS + {}], ".format(i)
        string_output = string_output[:-2]

        c_code = (
            """#include <stdio.h> //\n"""
            + """#include <stdlib.h> //\n"""
            + """#include <math.h>\n"""
            + """\n"""
            + """#define NB_SENSORS {}\n""".format(self.nb_sensors)
            + """#define NB_ACTORS {}\n""".format(self.nb_actors)
            + """#define NB_ADD_NEURONS {}\n""".format(self.nb_add_neurons)
            + """#define PERIOD {}\n""".format(self.period)
            + """#define NB_TOTAL_NEURONS NB_SENSORS + NB_ADD_NEURONS"""
                + """ + NB_ACTORS\n"""
            + """\n"""
            + """typedef struct Network\n"""
            + """{\n"""
            + """    double values[NB_TOTAL_NEURONS];\n"""
            + """    double bias[NB_TOTAL_NEURONS];\n"""
            + """    double weights[NB_TOTAL_NEURONS][NB_TOTAL_NEURONS];\n"""
            + """}\n"""
            + """Network;\n"""
            + """\n"""
            + """double sigmoid(double x)\n"""
            + """{\n"""
            + """    double r;\n"""
            + """    r = -1 + (2/(1+exp(-x)));\n"""
            + """    return r;\n"""
            + """}\n"""
            + """\n"""
            + """double ramp(double x)\n"""
            + """{\n"""
            + """    if (x>0)\n"""
            + """    {\n"""
            + """        return x;\n"""
            + """    }\n"""
            + """    else\n"""
            + """    {\n"""
            + """        return 0;\n"""
            + """    }\n"""
            + """}\n"""
            + """\n"""
            + """void iteration(Network *N)\n"""
            + """{\n"""
            + """    double values2[NB_TOTAL_NEURONS];\n"""
            + """    int i;\n"""
            + """    int j;\n"""
            + """    for (i=0; i<NB_TOTAL_NEURONS; i++)\n"""
            + """    {\n"""
            + """        values2[i] = 0;\n"""
            + """    }\n"""
            + """    for (i=0; i<NB_TOTAL_NEURONS; i++)\n"""
            + """    {\n"""
            + """        for (j=0; j<NB_TOTAL_NEURONS; j++)\n"""
            + """        {\n"""
            + """            values2[j]+=N->weights[j][i]*(N->bias[i]+"""
                + """N->values[i]);\n"""
            + """        }\n"""
            + """    }\n"""
            + """    for (i=0; i<NB_TOTAL_NEURONS; i++)\n"""
            + """    {\n"""
            + """        N->values[i] = {}(values2[i]);\n""".format(
                self.function.__name__)
            + """    }\n"""
            + """}\n"""
            + """\n"""
            + """int main(int argc, char * argv[])\n"""
            + """{\n"""
            + """    if (argc != 3)\n"""
            + """    {\n"""
            + """        printf("Use format : input_file output_file\\n");\n"""
            + """        return 1;\n"""
            + """    }\n"""
            + """    Network N = {\n"""
            + """    {},\n""".format(string_values)
            + """    {},\n""".format(string_bias)
            + """    {},\n""".format(string_weights)
            + """    };\n"""
            + """    double values2[NB_TOTAL_NEURONS];\n"""
            + """    FILE *input_file;\n"""
            + """    input_file = fopen(argv[1], "r");\n"""
            + """    if (input_file == NULL)\n"""
            + """    {\n"""
            + """        perror("input_file opening");\n"""
            + """        return 1;\n"""
            + """    }\n"""
            + """    FILE *output_file;\n"""
            + """    output_file = fopen(argv[2], "w");\n"""
            + """\n"""
            + """    if (output_file == NULL)\n"""
            + """    {\n"""
            + """        perror("output_file opening");\n"""
            + """        fclose(input_file);\n"""
            + """        return 1;\n"""
            + """    }\n"""
            + """    int i;\n"""
            + """    while (fscanf(input_file, "{}\\n", {}))\n""".format(
                string_format_input, string_input)
            + """    {\n"""
            + """        // input\n"""
            + """        for (i=0; i<NB_SENSORS; i++)\n"""
            + """        {\n"""
            + """            N.values[i] += values2[i];\n"""
            + """        }\n"""
            + """        // process\n"""
            + """        for (i=0; i<PERIOD; i++)\n"""
            + """        {\n"""
            + """            iteration(&N);\n"""
            + """        }\n"""
            + """        // output\n"""
            + """        fprintf(output_file, "{}\\n", {});\n""".format(
                string_format_output, string_output)
            + """    }\n"""
            + """    fclose(output_file);\n"""
            + """    fclose(input_file);\n"""
            + """    return 0;\n"""
            + """}\n\n"""
            )
        f = open(c_code_name + ".c", "w")
        f.write(c_code)
        f.close()
        os.system("gcc -o {} {} -lm".format(c_code_name, c_code_name + ".c"))
        os.system("rm {}".format(c_code_name + ".c"))

    def reset(self):
        self.values = np.zeros(self.values.shape)

    def display_console(self):
        print("neurons : {}\n".format(self.nb_neurons)
               + "sensors : {}\n".format(self.nb_sensors)
               + "actors : {}\n".format(self.nb_actors)
               + "added neurons : {}\n".format(self.nb_add_neurons)
               + "period : {}\n".format(self.period)
               + "values :\n{}\n".format(self.values)
               + "bias :\n{}\n".format(self.bias)
               + "weights :\n{}\n".format(self.weights)
             )

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
        period = 1,
        size = 100,
        mutation_coefficent = 0.0001,
        mutation_amplitude = 0.01,
        nb_tests = 2,
        do_display_execution = False,
        display_results_mode = None,
        **kwargs
    ):
        self.kwargs = kwargs
        if "slices" in kwargs:
            self.nb_add_neurons = sum(kwargs["slices"][1:-1])
        else:
            self.nb_add_neurons = nb_add_neurons
        self.series = []
        self.problem = problem
        self.nb_sensors = problem.nb_sensors
        self.nb_actors = problem.nb_actors
        self.period = period
        self.colors = ["r", "g", "b", "c", "m", "y", "k"]
        self.nb_herds = nb_herds
        self.nb_generations = nb_generations
        self.size = size
        self.mutation_coefficent = mutation_coefficent
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.do_display_execution = do_display_execution
        self.display_results_mode = display_results_mode
        self.values_simple = self.nb_herds*[1]
        self.values_nb_add_neurons = [0, 1, 2, 3, 4, 5, 6]
        self.values_period = [1, 2, 3, 4, 5, 6, 7]
        self.values_sizes = [5, 10, 50, 100, 500, 1000]
        self.values_mutation_coefficients = [0.0001, 0.000005, 0.00001]
        self.values_mutation_amplitude = [0.01, 0.005, 0.001]
        self.values_nb_tests = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.archives = []

    def test(self, mode = "simple", nb_generations = None, values = None):
        if nb_generations == None:
            nb_generations = self.nb_generations
        base = [
            self.nb_sensors,
            self.nb_actors,
            self.nb_add_neurons,
            self.period,
            self.size,
            self.mutation_coefficent,
            self.mutation_amplitude,
            self.nb_tests,
            self.do_display_execution
        ]
        if mode in [0, "simple"]:
            if values == None:
                values = self.values_simple
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
        elif mode in [1, "nb_add_neurons"]:
            if values == None:
                values = self.values_nb_add_neurons
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,2] = values
        elif mode in [2, "period"]:
            if values == None:
                values = self.period
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,3] = values
        elif mode in [3, "size"]:
            if values == None:
                values = self.values_sizes
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,4] = values
        elif mode in [4, "coefficient_mutation"]:
            if values == None:
                values = self.values_mutation_coefficients
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,5] = values
        elif mode in [5, "coefficient_amplitude"]:
            if values == None:
                values = self.values_mutation_amplitude
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,6] = values
        elif mode in [6, "nb_tests"]:
            if values == None:
                values = self.values_nb_tests
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,7] = values
        elif mode in [7, "multiple"]:
            if values == None:
                raise(ValueError("An array must be in input"))
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs = values
        # Pre-display
        test_colors = np.array([[self.colors[i%len(self.colors)]]
                       for i in range(len(values))])
        test_values = np.concatenate((array_inputs[:,2:-1], test_colors),
                                     axis = 1)
        print(test_values)
        self.display_table(test_values)
        # Starts learning !
        for i in range(len(values)):
            H = Herd(*array_inputs[i], **self.kwargs)
            self.series.append(H.evolve(self.problem, nb_generations))
            self.archives.append([H.members[0], self.series])
        # display
        if self.display_results_mode != None:
            if self.display_results_mode in [0, "console"]:
                self.display_console()
            if self.display_results_mode in [1, "plot"]:
                self.display_plot()
        # reset the self.series for if it is needed after
        self.series = []

    def display_table(self, Variables_values):
        print(self.mutation_coefficent)
        Variables_name_1 = np.array(["nb of added", "",
                          "herd's", "mutation",
                          "mutation", "nb of", ""])
        Variables_name_2 = np.array(["neurons", "period",
                                     "size", "coefficent", "amplitude",
                                     "tests", "color"])
        form ="{:<12} {:<7} {:<7} {:<11} {:<11} {:<7} {:<0}"
        for i in [Variables_name_1, Variables_name_2, *Variables_values]:
            print(form.format(*i))

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
