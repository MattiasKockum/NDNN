#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks
"""

from AI import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


class Classifier(Problem):
    """
    This is a Classifier to improve how I do my Networks
    This one will sort a 2D plane populated with Red and Blue dots
    A negative output means Blue, a positive means Red
    White means we don't care or no value was given
    """
    def __init__(self, do_run_display = False, do_end_display = False,
                plane = None):
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.nb_sensors = 2
        self.nb_actors = 1
        self.score = 0
        if type(plane) == type(None):
            self.plane = np.array([
                [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0], # o----->x
                [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0], # |
                [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0], # |
                [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0], # |
                [ 0,  0,  0,  1,  1,  1,  0,  0,  0,  0], # V
                [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0], # y
                [ 0,  0,  0,  0,  0,  1,  0,  0,  0,  0], #
                [ 0, -1, -1,  0,  0,  1,  0, -1, -1, -1], #
                [ 0, -1, -1, -1,  0,  1,  0, -1, -1, -1], #
                [ 0,  0,  0,  0,  0,  1,  0, -1, -1, -1], #
            ])
        else:
            self.plane = plane
        self.size = len(self.plane)
        self.score_max = 0
        for row in self.plane:
            for value in row:
                self.score_max += abs(value)
        self.colors = {-1: "b", 0: "w", 1: "r"}
        self.classification = np.zeros((self.size, self.size))
        self.classified = False

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        To be consistent the Network should be of the form:
            [[0, 0, 0], [0, 0, 0], [X, Y, 0]]
        """
        self.Network = Network
        while not self.end_condition():
            self.action()
        self.score_update()
        score = self.score
        if self.do_run_display:
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
        print("Warning  : state was not fully configured")
        return(np.array([1]))

    # Other state related functions should be there

    def score_update(self):
        """
        Updates the score of the problem at the moment
        1 good prediction = +1pt
        1 bad prediction = -1pt
        """
        score = 0
        for y, row in enumerate(self.plane):
            for x, value in enumerate(row):
                score += value*self.classification[y][x]
        # score should always be > 0
        score /= self.score_max
        self.score = (score>0)*score

    def action(self):
        """
        Computes the consequences of the input_data on the problem
        """
        preclassification = [[
            self.Network.process(np.array((x/self.size, y/self.size)))
            for x in range(self.size)]
            for y in range(self.size)]
        self.classification = [[
            threshold(value[0])
            for value in row]
            for row in preclassification]
        self.classified = True

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        for y, row in enumerate(self.plane):
            for x, value in enumerate(row):
                plt.scatter(x, y, color=self.colors[value], marker="x")
        for y, row in enumerate(self.classification):
            for x, value in enumerate(row):
                plt.scatter(x, y, color=self.colors[value], marker="$O$")
        plt.show()

    # Other display functions should be put here

    def __name__(self):
        return("Classifier")

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.do_run_display, self.do_end_display, self.plane)



def main():
    C = Classifier(None, False)
    T = TestBench(C)
    T.test()
    #H = Herd(2, 1, 15, 3, 100, 0.1, 0.001, 1, True,
    #         slices = [2, 5, 5, 5, 1], regions=under_diag(18))
    #H.evolve(C, 20)
    #C.do_run_display = True
    #N = H.members[0]
    #print(C.experience(N))

if __name__ == "__main__":
    main()
