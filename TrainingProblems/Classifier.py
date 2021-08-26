#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks
"""

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../ModelAI")


from AI import *
from Classic import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


plane0 = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
])

plane1 = 1.1*np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
])

plane2 = 1.1*np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  0,  0,  0,  0,  0,  0, -1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0, -1,  0,  0,  0,  0,  0,  0,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
])


class Classifier(Problem):
    """
    This is a Classifier to improve how I do my Networks
    This one will sort a 2D plane populated with Red and Blue dots
    A negative output means Blue, a positive means Red
    White means we don't care or no value was given
    """
    def __init__(self, plane = plane1,
                 do_run_display = False,
                 do_end_display = True):
        # Common
        self.nb_sensors = 2
        self.nb_actors = 1
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        # Class specific
        self.plane = plane
        self.size = len(self.plane)

    def experience(self, networks):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        self.experience_preparation(networks)
        while not self.experience_ended():
            self.problem_preparation()
            value = self.plane[self.x][self.y]
            output = network.process(self.state())
            network.reset()
            self.action(output)
        # Here do_run_display is moved out of the loop for a reason
        if self.do_run_display:
            self.run_display(network)
        self.score_update()
        score = self.score
        self.reset()
        print(score)
        return(score)

    def experience_preparation(self, networks):
        # Common
        self.score = np.zeros((len(networks)))
        self.networks = networks
        # Class specific
        self.score = 0
        self.classified = False
        self.x = 0
        self.y = 0

    def problem_preparation(self):
        # Common
        self.score = np.zeros((len(networks)))
        playing_index = self.organisation()
        # Class specific

    def experience_ended(self):
        """
        True if every network has been evaluated
        False otherwise
        """
        print("Warning  : experience_ended was not fully configured")
        return(True)

    def organisation(self):
        """
        Return the indexes of the network(s) that must play the next game
        Can be a tree for 1v1
        Can be a line for solo evaluation
        Can be everyone at the same time etc
        """
        print("Warning  : organisation was not fully configured")
        return(0)

    def problem_ended(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        return(self.classified)

    def state(self):
        """
        Returns the state of the problem
        """
        return([self.x, self.y])

    # Other state related functions should be there

    def score_update(self):
        """
        Updates the score of the problem at the moment
        """
        self.score = (self.score>0)*self.score

    def action(self, output):
        """
        Computes the consequences of the input_data on the problem
        """
        guess = output[0]
        value = self.plane[self.x][self.y]
        if value != 0:
            # Diff can't be 0
            self.score += abs(1/(value - guess))
        # Update the coords
        self.x += 1
        self.x %= self.size
        if self.x == 0:
            self.y += 1
            self.y %= self.size
        if (self.x, self.y) == (0, 0):
            self.classified = True


    # Other action related functions should be put here

    def run_display(self, network):
       """
       Shows how things are doing
       """
       fig = plt.figure()
       ax = fig.gca(projection='3d')

       # Make data.
       X = np.arange(0, self.size, 1)
       Y = np.arange(0, self.size, 1)
       X, Y = np.meshgrid(X, Y)
       Z = np.zeros(X.shape)
       for x in range(X.shape[0]):
           for y in range(X.shape[1]):
               Z[y][x] = network.process([x, y])
               network.reset()

       # Plot the surface.
       surf = ax.plot_wireframe(X, Y, Z, color="#0F0F0F0F")

       for x in range(self.size):
           for y in range(self.size):
               value = self.plane[x][y]
               if value > 1:
                   ax.scatter([x], [y], [1], color="red")
               elif value < 0:
                   ax.scatter([x], [y], [-1], color="blue")

       plt.show()

    def end_display(self):
       """
       Shows what happened
       """
       pass

    # Other display functions should be put here

    def __name__(self):
        return("Classifier")

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.plane,
                      self.do_run_display, self.do_end_display)



def main():
    C = Classifier(plane2, False, False)
    H = Herd(2, 1, 4, sigmoid, 2, 0.01)
    N1 = ClassicalNetwork([2, 2, 2, 1])
    N2 = ClassicalNetwork([2, 2, 2, 1])
    H.members = [N1, N2]
    H.evolve(C, 500)
    N = H.members[0]
    print(type(N))
    displayNetwork(N)
    C.do_run_display = True
    print(C.experience(N))
    return(0)

if __name__ == "__main__":
    main()
