#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
"""


# Necessary
import numpy as np
# Class specifique
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

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
    def __init__(self, do_run_display = False, do_end_display = True,
                score=None, clean_after_shown=True, points=[]):
        self.nb_sensors = 1
        self.nb_actors = 0
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.score = score
        self.clean_after_shown = clean_after_shown
        self.points = points
        self.operation_done = False

    def experience(self, networks):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        self.score = np.zeros((len(networks)))
        # Exemple for a solo problem
        for index, network in enumerate(networks):
            while not self.end_condition():
                self.action(network, index)
                if self.do_run_display:
                    self.run_display()
            self.reset()
        return(self.score)

    def end_condition(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        return(self.operation_done)

    def state(self):
        """
        Returns the state of the problem
        """
        print("Warning  : state was not fully configured")
        return(np.array([1]))

    # Other state related functions should be there

    def action(self, network, index):
        """
        Computes the consequences of the input_data on the problem
        """
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

    def end_display(self, step=1):
        """
        Shows what happened
        """
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
                      self.score, self.clean_after_shown, self.points)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        self.__init__(self.do_run_display, self.do_end_display,
                      None, self.clean_after_shown)


def gradient(X, Y):
    return(
        + 1*np.exp(-(((X-0)/0.2)**2+((Y+0)/0.2)**2))
        + 1.2*np.exp(-(((X-0.2)/0.1)**2+((Y+0.2)/0.1)**2))
        + 1.4*np.exp(-(((X+0.2)/0.05)**2+((Y+0.2)/0.05)**2))
    )


def main(parameters):
    H = Herd(1, 0, 0, size=2, mutation_amplitude=0.01)
    P = Gradient()
    H.evolve(P, 200)

if __name__ == "__main__":
    main(sys.argv)
