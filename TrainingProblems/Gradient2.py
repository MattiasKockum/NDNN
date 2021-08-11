#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to test the basic problem
And to prove Gradient descent work well
"""


# Necessary
import numpy as np

# Class specific
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# For Main
import sys
sys.path.append(".")
sys.path.append("..")
from AI import *


class Gradient(Problem):
    """
    The frame of any "live" problem
    The problem has to be coded in a copy of it
    just so that the function stay the same with every problem
    """
    def __init__(self, do_run_display = False, do_end_display = False,
                clean_after_shown=True, points=[]):
        # Common
        self.nb_sensors = 1
        self.nb_actors = 0
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        # Class specific
        self.clean_after_shown = clean_after_shown
        self.points = points

    def experience(self, networks):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        self.score = np.zeros((len(networks)))
        self.networks = networks
        self.operation_done = False
        while not self.experience_ended():
            playing_index = self.organisation()
            while not self.problem_ended():
                self.action(playing_index)
                if self.do_run_display:
                    self.run_display()
            self.reset()
        return(self.score)

    def experience_ended(self):
        """
        True if every network has been evaluated
        False otherwise
        """
        return(self.operation_done)

    def organisation(self):
        """
        Return the indexes of the network(s) that must play the next game
        Can be a tree for 1v1
        Can be a line for solo evaluation
        Can be everyone at the same time etc
        """
        return([i for i in range(len(self.networks))])

    def problem_ended(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        return(self.operation_done)

    def state(self):
        """
        Returns the state of the problem
        """
        #print("Warning  : state was not fully configured")
        #We don't ware here
        return(np.array([1]))

    # Other state related functions should be there

    def action(self, playing_index):
        """
        Computes what the networks do and puts the score accordingly
        """
        for index in playing_index:
            network = self.networks[index]
            w = network.weights[0][0]
            b = network.bias[0]
            self.score[index] = gradient(w, b)
            self.points.append((w, b))
        self.operation_done = True

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
        step = 1
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(-0.5, 0.5, 0.01)
        Y = np.arange(-0.5, 0.5, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = gradient(X, Y)

        # Plot the surface.
        surf = ax.plot_wireframe(X, Y, Z, color="#0F0F0F0F")

        xs = [i[0] for i in self.points]
        ys = [i[1] for i in self.points]
        zs = [gradient(x, y) for x, y in zip(xs, ys)]
        for gen in range(0, len(zs), step):
            ax.scatter(
                xs[gen:gen+step],
                ys[gen:gen+step],
                zs[gen:gen+step],
                color=(
                    "#"
                    + hex(0)[2:]
                    + hex(0)[2:]
                    + hex(int(256*gen/len(zs)))[2:]
                )
            )

        plt.show()

        if self.clean_after_shown:
            self.clean()
    # Other display functions should be put here

    def __name__(self):
        return("Gradient")

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.do_run_display, self.do_end_display,
                      self.clean_after_shown, self.points)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        self.__init__(self.do_run_display, self.do_end_display,
                      self.clean_after_shown, [])

def gradient(X, Y):
    return(
        + 1*np.exp(-(((X-0)/0.2)**2+((Y+0)/0.2)**2))
        + 1.2*np.exp(-(((X-0.2)/0.1)**2+((Y+0.2)/0.1)**2))
        + 1.4*np.exp(-(((X+0.2)/0.05)**2+((Y+0.2)/0.05)**2))
    )


def main(parameters):
    P = Gradient(False, True)
    H = Herd(P.nb_sensors, P.nb_actors,
             0, size=2, mutation_amplitude=0.01)
    H.evolve(P, 200)

if __name__ == "__main__":
    main(sys.argv)
