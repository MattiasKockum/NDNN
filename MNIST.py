#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks on the well known
mnist
"""

from AI import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import mnist as mn

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
        if list(self.output).index(max(self.output)) == self.number:
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

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.do_display)


def main():
    nb_sensors = 28*28
    nb_actors = 10
    nb_add_neurons = 30
    period = 3
    size = 50
    mutatation_coefficient = 0.1
    mutation_amplitude = 0.001
    nb_tests = 10
    do_display = False
    H = Herd(nb_sensors, nb_actors, nb_add_neurons, period, size,
             mutatation_coefficient, mutation_amplitude, nb_tests, do_display,
             slices = [28*28, 15, 15, 10], regions=under_diag(4))
    P = MNIST(False)
    H.evolve(P, 20)
    P.do_display = True
    N = H.members[0]
    for i in range(10):
        print(P.experience(N))

if __name__ == "__main__":
    main()