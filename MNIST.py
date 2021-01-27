#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks on the well known
mnist
"""

from AI import *

import numpy as np
import mnist as mn
# Only here for GPU processing (feel free to remove at will)
from GPU_code_maker import *
# Only here for fancy looking graphs (feel free to remove at will)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


TEST_IMAGES, TEST_LABELS = mn.test_images(), mn.test_labels()
LEN = len(TEST_IMAGES)
SIZE = len(TEST_IMAGES[0])
SQUISHED_IMAGES = np.zeros((LEN), dtype='object')
for i in range(LEN):
    SQUISHED_IMAGES[i] = np.resize(TEST_IMAGES[i], (SIZE**2, ))
    SQUISHED_IMAGES[i] = SQUISHED_IMAGES[i]/max(SQUISHED_IMAGES[i])

class MNIST(Problem):
    """
    This is a Classifier to improve how I do my Networks
    """
    def __init__(self, do_display = False):
        self.nb_sensors = 28*28 # size of the image
        self.nb_actors = 10 # number of possible anwsers (maybe try in binary)
        self.score = 0
        self.do_display = do_display
        self.classified = False
        self.output = np.zeros((10))
        index = np.random.randint(0, LEN)
        self.image = TEST_IMAGES[index]
        self.squished_image = SQUISHED_IMAGES[index]
        self.number = TEST_LABELS[index]

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        while not self.end_condition():
            self.action(Network)
        self.score_update()
        score = self.score
        if self.do_display:
            self.display()
        self.reset()
        return(score)

    def end_condition(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        return(self.classified)

    def state(self):
        """
        Returns the state of the problem
        """
        return(self.squished_image)

    # Other state related functions should be there

    def score_update(self):
        """
        Updates the score of the problem at the moment
        """
        score = 0
        if maxindex(self.output) == self.number:
            score = 1
        # score should always be > 0
        self.score = (score>0)*score

    def action(self, Network):
        """
        Computes the consequences of the input_data on the problem
        """
        self.output = Network.process(self.state())
        self.classified = True

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self.image)
        plt.show()

    # Other display functions should be put here

    def __name__(self):
        return("MNIST")

    def Kernel_code(self):
        """
        Only here for GPU processing
        """
        return(C_to_string("Kernel_MNIST.c"))
        #return(C_to_string("Kernel_MNIST_test.c"))

    def Kernel_inputs(self, length):
        """
        Returns the inputs values that will init the problem
        They are here for randomisation reasons
        """
        inputs = []
        for i in range(length):
            i_input = []
            i_input.append(self.number)
            for j in self.squished_image:
                i_input.append(j)
            self.reset()
            inputs.append(i_input)
        return(inputs)

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.do_display)


def main():
    nb_sensors = 28*28
    nb_actors = 10
    nb_add_neurons = 16 + 16
    period = 3
    function = segments
    reset_after_process = True
    size = 100
    mutation_coefficient = 1
    mutation_amplitude = 0.01
    nb_tests = 100
    do_display = False
    nb_herds = 1
    nb_generations = 100
    do_display_execution = False,
    display_results_mode = "console"
    slices = [28*28, 16, 16, 10]
    regions = under_diag(4)
    P = MNIST(False)
    H = Herd(nb_sensors, nb_actors, nb_add_neurons, period, function,
             reset_after_process, size, mutation_coefficient,
             mutation_amplitude, nb_tests, do_display)
    H.evolve(P, nb_generations)

if __name__ == "__main__":
    main()
