#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to test my networks
"""


# Necessary
import numpy as np

# Class specific

# For Main
import sys
sys.path.append(".")
sys.path.append("..")
from AI import *
sys.path.append("../ModelAI")
from Classic import *



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

plane1 = np.array([
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

plane2 = np.array([
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

plane3 = np.array([
    [-1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
    [-1, -1, -1, -1, -1,  1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
    [ 1,  1,  1,  1,  1, -1, -1, -1, -1, -1]
])




class Classifier(Problem):
    """
    The frame of any "live" problem
    The problem has to be coded in a copy of it
    just so that the function stay the same with every problem
    """
    def __init__(self, plane = plane1, nb_process=1,
                 do_run_display = False, do_end_display = False):
        # Common
        self.nb_sensors = 2
        self.nb_actors = 1
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        # Class specific
        self.plane = plane
        self.nb_process = nb_process
        self.size = len(plane)
        self.nb_points = sum(sum(abs(plane)))

    def experience(self, networks):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        self.experience_preparation(networks)
        while not self.experience_ended():
            self.problem_preparation()
            while not self.problem_ended():
                self.action()
            if self.do_run_display:
                self.run_display(self.networks[self.playing_index[0]])
            self.reset()
            print("Max Score : {}".format(max(self.score)))
        return(self.score)

    def experience_preparation(self, networks):
        # Common
        self.score = np.zeros((len(networks)))
        self.networks = networks
        # Class specific
        self.done_all_networks = False
        self.last_index = len(networks) - 1
        self.playing_index = [-1]

    def problem_preparation(self):
        # Common
        self.playing_index = self.organisation()
        # Class specific
        self.done_network = False
        self.x = 0
        self.y = 0

    def experience_ended(self):
        """
        True if every network has been evaluated
        False otherwise
        """
        return(self.done_all_networks)

    def problem_ended(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        return(self.done_network)

    def organisation(self):
        """
        Return the indexes of the network(s) that must play the next game
        Can be a tree for 1v1
        Can be a line for solo evaluation
        Can be everyone at the same time etc
        """
        return([self.playing_index[0] + 1])

    def state(self):
        """
        Returns the state of the problem
        """
        return([self.x, self.y])

    # Other state related functions should be there

    def action(self):
        """
        Computes what the networks do and puts the score accordingly
        """
        network = self.networks[self.playing_index[0]]
        for i in range(self.nb_process):
            output = network.process(self.state())[0]
            print(output)
        print("------")
        value = self.plane[self.x][self.y]
        if value != 0:
            d = abs(value - output)
            self.score[self.playing_index[0]] += np.exp(-d)/self.nb_points
        # Upgrade position
        self.x += 1
        if self.x == self.size:
            self.x = 0
            self.y += 1
        if self.y == self.size:
            self.done_network = True
            if self.playing_index[0] == self.last_index:
                self.done_all_networks = True

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
                for i in range(self.nb_process):
                    Z[y][x] = network.process([x, y])
                network.reset()

        # Plot the surface.
        surf = ax.plot_wireframe(X, Y, Z, color="#0F0F0F0F")

        for x in range(self.size):
            for y in range(self.size):
                value = self.plane[x][y]
                if value == 1:
                    ax.scatter([x], [y], [1], color="red")
                elif value == -1:
                    ax.scatter([x], [y], [-1], color="blue")

        plt.show()

    def end_display(self):
        """
        Shows what happened
        """
        print("Warning : end_display was not fully configured")

    # Other display functions should be put here

    def __name__(self):
        return("Classifier")

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.plane, self.nb_process,
                      self.do_run_display, self.do_end_display)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        self.__init__(self.plane, self.nb_process,
                      self.do_run_display, self.do_end_display)


# External functions of the problem should be put here


def main(parameters):
    # General evolution
    P = Classifier(plane3, 3, do_run_display=False)
    H = Herd(P.nb_sensors, P.nb_actors,
             8, sigmoid, size=10, mutation_amplitude=4)
    H.evolve(P, 50, "sauvenet")
    # Exploitation
    N = H.members[0]
    P.run_display(N)


if __name__ == "__main__":
    main(sys.argv)
