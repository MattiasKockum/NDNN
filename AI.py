#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an AI
This file is about importing classes together
"""

from ActivationFunctions import *
from NDNN import *
from NaturalSelection import *
from Problem import *

# Necessary
import numpy as np
import copy
import time
# Useful for easy data visualisation
import matplotlib.pyplot as plt



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
        function = segments,
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
        self.function = function
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
            self.function,
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
        elif mode in [2, "size"]:
            if values == None:
                values = self.values_sizes
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,6] = values
        elif mode in [3, "coefficient_mutation"]:
            if values == None:
                values = self.values_mutation_coefficients
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,7] = values
        elif mode in [4, "coefficient_amplitude"]:
            if values == None:
                values = self.values_mutation_amplitude
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,8] = values
        elif mode in [5, "nb_tests"]:
            if values == None:
                values = self.values_nb_tests
            array_inputs = np.array(
                [base for i in range(len(values))],
                dtype = object
            )
            array_inputs[:,9] = values
        elif mode in [6, "multiple"]:
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
        An approximation of the distance between a random NDNN and
        the perfectly fit NDNN (if ever it exists)
        """
        return(np.sqrt(self.nb_neurons*(self.nb_neurons + 1)))

    def estimated_mutation_amplitude(self):
        """
        An idea of the good mutation_amplitude to progress
        I want it to be pretty sure that at least one NDNN will not move
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
        I want to be pretty sure that at least one of the NDNNs will be on
        the right path
        """
        return(2*self.nb_neurons*(self.nb_neurons + 1))

    def estimated_nb_generations(self):
        """
        An idea of the good nb_generations to attain the perfect NDNN
        """
        return(100)

    def estimated_nb_tests(self):
        """
        An idea of the good number of tests to do to fit the problem
        """
        l = 100
        tolerance = 0.6
        N = NDNN(self.nb_sensors, self.nb_actors, self.nb_add_neurons,
                    self.function)
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
        Variables_name_2 = ['neurons', 'name', 'size',
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
