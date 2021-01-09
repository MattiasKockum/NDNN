#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an AI
The training is parallelized
"""

# Necessary
from AI import *
import pyopencl as cl
# Useful for compiling Network in machine code
import os
# Useful for easy data visualisation
import matplotlib.pyplot as plt

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

# Members are put in form of a list [biais...weights...]
# (size = nb_neurons*(nb_neurons + 1))

def prob_reproduction(X):
    """
    A weird looking function for parallelization
    X[0] is a group of members
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
    X[1].reset()
    return_value = X[0].experience(X[1])
    return(return_value)

def pooled_evolution(X):
    """
    Another one
    """
    r = X[0].evolve(X[1], X[2])
    return(r, (X[0].members[0], r))


class GPU_Problem(Problem):
    """
    The frame of any "live" problem, but in GPU
    """
    def Kernel_code(self):
        return("This should be some kernel code")


class GPU_Herd(Herd):
    """
    Herd of networks that evolve by reproducing
    """
    def evolve(self, problem, nb_generations=1):
        """
        The idea is to make the AI evolve by aproximating the gradient descent
        Opens and closes the score output file multiple times so that it's
        possible to see what's going on in during the training
        """
        # Saving problem
        if problem == None:
            # The empty problem, just here for quick tests
            self.Problem = GPU_Problem()
        else:
            self.Problem = problem
        # Paralellization values
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)
        mf = cl.mem_flags
        kernel_program = cl.Program(context, self.Problem.Kernel_code()).build()
        score = [0]*(self.size*self.nb_tests)
        # Opening score file
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
        # Display instruction
        for pb in self.Problem_pool:
            pb.do_display = False
        if self.do_display:
            self.Problem_pool[0].do_display = True
        # Evolution
        for generation in range(nb_generations):
            # Evaluation of performances
            kernel_program.experience(
                queue,
                self.Problem.Kernel_init(self.size*self.nb_tests),
                biasAndWeights,
                score
            )
            # Reproduction (with mutation) of Networks
            self.reproduce(proba_reproduction)
            # Saves the scores
            self.max_score = max(self.score)
            self.max_score_index = self.score.index(self.max_score)
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
        score_file.close()
        return(self.array_scores)

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

class GPU_Network(Network):
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
    def flatten(self):
        return(self.bias + self.weights.reshape((self.nb_neurons**2, )))


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
        function = segments,
        reset_after_process = True,
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
            function,
            reset_after_process,
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

    def display_table(self, Variables_values):
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
