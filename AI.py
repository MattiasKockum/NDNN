#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an AI
    capable of selective memory
    capable of solving real time problems fast
    capable of simulating a Turing Machine
    It's supposed to be able de to convolution but not the best way
"""

# Necessary
import numpy as np
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

def segments(x):
    return((-1-x)*(x<-1) + x + (1-x)*(x>1))

def threshold(x):
    return(1*(x>0) + 0 -1*(x<0))

def convolution(entry, kernel):
    """
    entry and kernel must be numpy arrays
    kernel must be able to fit into entry
    """
    dim_diff = len(entry.shape) - len(kernel.shape) 
    if dim_diff < 0:
        raise(ValueError("kernel's dimension greater than entry's"))
    kernel = np.expand_dims(kernel, axis=[i for i in range(dim_diff)])
    dim = len(kernel.shape)
    shape_diff = [entry.shape[i] - kernel.shape[i] + 1 for i in range(dim)]
    if True in [i < 0 for i in shape_diff]:
        raise(ValueError("kernel bigger than entry in one dimension"))
    output = np.zeros(shape_diff)
    iterator = np.zeros((dim))
    while True in (iterator != shape_diff):
        pass
        # incrementation
        i = -1
        iterator[i] += 1
        while iterator[i] > shape_diff[i]:
            iterator[i] = 0
            i -= 1
            iterator[i] += 1
    return(output)

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

def maxindex(array):
    """
    Very useful for saying which of the output neurons is the most activated
    """
    return(list(array).index(max(array)))

class Problem(object):
    """
    The frame of any "live" problem
    """
    def __init__(self, do_run_display = False, do_end_display = False,):
        print("Warning  : __init__ was not fully configured")
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.nb_sensors = 1
        self.nb_actors = 1
        self.score = 0

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        total_score = 0
        while not self.end_condition():
            output = Network.process(self.state())
            self.action(output)
            if self.do_run_display:
                self.run_display()
        self.score_update()
        total_score += self.score
        self.reset()
        return(score/self.number_tests)

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
        # score should always be > 0
        self.score = self.score*(self.score>0)

    def action(self, output):
        """
        Computes the consequences of the input_data on the problem
        """
        print("Warning  : action was not fully configured")
        pass

    # Other action related functions should be put here

    def run_display(self):
        """
        Shows how things are doing
        """
        print("Warning : run_display was not fully configured")

    def end_display(self):
        """
        Shows what happened
        """
        print("Warning : end_display was not fully configured")

    # Other display functions should be put here

    def __name__(self):
        return("An_Unnamed_Problem")

    def reset(self):
        """
        Resets the problem
        """
        print("Maybe some HUGE PROBLEM is coming at you bro")
        self.__init__(self.do_run_display, self.do_end_display)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        pass


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
        function = segments,
        reset_after_process = True,
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
        self.function = function
        self.reset_after_process = reset_after_process
        self.size = size
        self.mutation_coefficent = mutation_coefficent
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.do_display = do_display
        self.make_members(kwargs)
        self.array_scores = []
        self.date = date()
        self.max_score = 0
        self.max_score_index = 0

    def make_members(self, kwargs):
        self.members = [
            Network(self.nb_sensors, self.nb_actors, self.nb_add_neurons,
                    self.period, self.function, self.reset_after_process,
                    **kwargs)
            for i in range(self.size)
        ]

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
        score_file = open(self.Problem.__name__() + "_score" + self.date, "w")
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
        for generation in range(nb_generations):
            # Evaluation of performances
            proba_reproduction = self.performances()
            # Reproduction (with mutation) of Networks
            self.reproduce(proba_reproduction)
            # Saves the scores
            self.max_score = max(self.score)
            self.max_score_index = list(self.score).index(self.max_score)
            self.array_scores.append(self.max_score)
            # Saves one Network and the score evolution
            self.members[self.max_score_index].save(
                self.Problem.__name__() + "_Network" + self.date, "w", False)
            score_file = open(
                self.Problem.__name__() + "_score" + self.date, "a"
            )
            score_file.write(
                "generation n° {} : {} \n".format(
                    generation, str(self.max_score)))
            score_file.close()
        score_file = open(self.Problem.__name__() + "_score" + self.date, "a")
        score_file.write("End\n")
        if self.Problem.do_end_display:
            self.Problem.end_display()
        score_file.close()
        return(self.array_scores)

    def performances(self):
        """
        Evaluates performances then normalises them for probability operations
        """
        self.score = np.zeros(self.size)
        for index, member in enumerate(self.members):
            member_score = 0
            for i in range(self.nb_tests):
                member_score += self.Problem.experience(member)
                member.reset()
            member_score /= self.nb_tests
            self.score[index] = member_score
        score_modif = self.modif_score(self.score)
        return(score_modif)

    def reproduce(self, proba_reproduction):
        """
        The copy of the successful networks with mutation
        """
        new_members = [copy.deepcopy(np.random.choice(
                        self.members, p=proba_reproduction))
                        for i in range(self.size)]
        for member in new_members:
            member.mutate(self.mutation_coefficent, self.mutation_amplitude)
        self.members = new_members

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
        function = segments,
        reset_after_process = True,
        **kwargs # "weights", "bias", "slices", "regions"
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.period = period
        self.function = function
        self.reset_after_process = reset_after_process
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
        output = self.output()
        if self.reset_after_process:
            self.reset()
        return(output)

    def input(self, values_inputs):
        self.values[:self.nb_sensors] += values_inputs

    def output(self):
        return(self.values[-self.nb_actors:])

    def iteration(self):
        """
        We iterate once and update network state
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
        for i in range(self.nb_neurons*(self.nb_neurons + 1)):
            # If there is a mutation
            if np.random.choice(
                [True, False],
                p = [mutation_coefficent, 1 - mutation_coefficent]
            ):
                # If the iterator corresponds to a weight, we modify it
                if i < self.nb_neurons**2:
                    self.weights[i//self.nb_neurons][i%self.nb_neurons] += (
                        np.random.normal(0, mutation_amplitude)
                    )
                # Elsif it corresponds to a bias we modify it
                elif i < self.nb_neurons*(self.nb_neurons + 1):
                    self.bias[i - self.nb_neurons**2] += (
                        np.random.normal(0, mutation_amplitude)
                    )

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

    def compile(self, c_code_name = None, add_date = False, save_exe = False):
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
            + """#define FUNCTION {}\n""".format(self.function.__name__)
            + """#define VALUES {}\n""".format(string_values)
            + """#define BIAS {}\n""".format(string_bias)
            + """#define WEIGHTS {}\n""".format(string_weights)
            + """#define NB_TOTAL_NEURONS NB_SENSORS + NB_ADD_NEURONS"""
                + """ + NB_ACTORS\n"""
            + """\n"""
            + """typedef struct Network\n"""
            + """{\n"""
            + """    float values[NB_TOTAL_NEURONS];\n"""
            + """    float bias[NB_TOTAL_NEURONS];\n"""
            + """    float weights[NB_TOTAL_NEURONS][NB_TOTAL_NEURONS];\n"""
            + """}\n"""
            + """Network;\n"""
            + """\n"""
            + """float sigmoid(float x)\n"""
            + """{\n"""
            + """    float r;\n"""
            + """    r = -1 + (2/(1+exp(-x)));\n"""
            + """    return r;\n"""
            + """}\n"""
            + """\n"""
            + """float ramp(float x)\n"""
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
            + """float segments(float x)\n"""
            + """{\n"""
            + """   if (x>1)\n"""
            + """    {\n"""
            + """        return 1;\n"""
            + """    }\n"""
            + """   if (x<-1)\n"""
            + """    {\n"""
            + """        return -1;\n"""
            + """    }\n"""
            + """ return x;\n"""
            + """}\n"""
            + """\n"""
            + """void iteration(Network *N)\n"""
            + """{\n"""
            + """    float values2[NB_TOTAL_NEURONS];\n"""
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
            + """        N->values[i] = FUNCTION(values2[i]);\n"""
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
            + """    VALUES,\n"""
            + """    BIAS,\n"""
            + """    WEIGHTS,\n"""
            + """    };\n"""
            + """    float values2[NB_TOTAL_NEURONS];\n"""
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
            + """}\n"""
            )
        f = open(c_code_name + ".c", "w")
        f.write(c_code)
        f.close()
        os.system("gcc -o {} {} -lm".format(c_code_name, c_code_name + ".c"))
        if not save_exe:
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
        nb_add_neurons = 0,
        period = 1,
        function = segments,
        reset_after_process = True,
        size = 100,
        mutation_coefficent = 1,
        mutation_amplitude = 0.1,
        nb_tests = 1,
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
        self.nb_neurons = self.nb_sensors + self.nb_actors + self.nb_add_neurons
        self.period = period
        self.function = function
        self.reset_after_process = reset_after_process
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
            self.function,
            self.reset_after_process,
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
            array_inputs[:,6] = values
        elif mode in [4, "coefficient_mutation"]:
            if values == None:
                values = self.values_mutation_coefficients
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,7] = values
        elif mode in [5, "coefficient_amplitude"]:
            if values == None:
                values = self.values_mutation_amplitude
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,8] = values
        elif mode in [6, "nb_tests"]:
            if values == None:
                values = self.values_nb_tests
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,9] = values
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

    def set_estimated(self):
        self.mutation_amplitude = self.estimated_mutation_amplitude()
        self.mutation_coefficent = self.estimated_mutation_coefficient()
        self.size = self.estimated_size()
        self.nb_tests = self.estimated_nb_tests()
        self.nb_generations = self.estimated_nb_generations()

    def estimation(self):
        X = []
        X.append(self.estimated_mutation_amplitude())
        X.append(self.estimated_mutation_coefficient())
        X.append(self.estimated_size())
        X.append(self.estimated_nb_tests())
        X.append(self.estimated_nb_generations())
        print(("mutation_amplitude : {}\nmutation_coefficient : {}"
              + "\nsize : {}\nnb_tests: {}\nnb_generations : {}").format(
              X[0], X[1], X[2], X[3], X[4]))
        return(X)

    def estimated_distance(self):
        """
        An approximation of the distance between a random Network and
        the perfectly fit Network (if ever it exists)
        """
        return(np.sqrt(self.nb_neurons*(self.nb_neurons + 1)))

    def estimated_mutation_amplitude(self):
        """
        An idea of the good mutation_amplitude to progress
        I want it to be pretty sure that at least one Network will not move
        from the best place I've found so that I don't lose it
        Also I need to be sure I don't search to far from where the goal is
        """
        # This value of 100 is a test
        return(self.estimated_distance()/5)

    def estimated_mutation_coefficient(self):
        """
        An idea of the good mutation_coefficent to progress
        I am now less sure this will be useful to keep differnt from 0
        """
        return(1)

    def estimated_size(self):
        """
        An idea of the good size to progress
        I want to be pretty sure that at least one of the Networks will be on
        the right path
        """
        return(2*self.nb_neurons*(self.nb_neurons + 1))

    def estimated_nb_generations(self):
        """
        An idea of the good nb_generations to attain the perfect Network
        """
        return(100)

    def estimated_nb_tests(self):
        """
        An idea of the good number of tests to do to fit the problem
        """
        l = 100
        tolerance = 0.6
        N = Network(self.nb_sensors, self.nb_actors, self.nb_add_neurons,
                    self.period, self.function, self.reset_after_process)
        tests = [self.problem.experience(N) for i in range(l)]
        mean = [np.mean(tests[:i+1]) for i in range(l)]
        last = mean[-1]
        i = 1
        while mean[i] > last*(1+tolerance) or mean[i] < last*(1-tolerance):
            i += 1
        return(i)

    def display_table(self, Variables_values):
        Variables_name_1 = ['nb added', 'nb', "function's", "herd's",
                                     'mutation', 'mutation', 'nb of', '']
        Variables_name_2 = ['neurons', 'period', 'name', 'size',
                                'coefficent', 'amplitude', 'tests', 'color']
        form ="{:<9} {:<7} {:<11} {:<7} {:<11} {:<10} {:<0}"
        Printable_values = []
        for i in Variables_values:
            j = list(i)
            Printable_values.append(j[0:2] + [j[2].__name__] + j[4:5]
                                    + [str(j[5])[0:9]] + [str(j[6])[0:9]]
                                    + [j[-2]])
        for i in [Variables_name_1, Variables_name_2, *Printable_values]:
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
