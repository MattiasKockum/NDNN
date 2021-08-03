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


class Gradient_Descent_Test():
    """
    """
    def __init__(self, do_run_display = False, do_end_display = True,
                 clean_after_shown = True, points = []):
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.clean_after_shown = clean_after_shown
        self.points = points
        self.nb_sensors = 1
        self.nb_actors = 0
        self.bias = 0
        self.weight = 0

    def action(self, inputs):
        pass

    def state(self):
        return(np.array([1]))

    def experience(self, Network):
        self.bias = Network.bias[0]
        self.weight = Network.weights[0][0]
        # below is a useful test to see if parallelization works
        #print(
            #"Problem : {}\n     Network : {}\n     Network hash : {}".format(
                #self.__hash__(),
                #self.weight + self.bias,
                #Network.__hash__()
            #)
        #)
        self.points.append((self.weight, self.bias))
        score = self.score_real_time()
        self.reset()
        return(score)

    def score_real_time(self):
        score = gradient(self.weight, self.bias)
        return(score*(score>0) + 0)

    def end_display(self, step=1):
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

    def __name__(self):
        return("Gradient_Descent_Test")

    def reset(self):
        self.__init__(self.do_run_display, self.do_end_display,
                      self.clean_after_shown, self.points)

    def clean(self):
        self.__init__(self.do_run_display, self.do_end_display,
                      self.clean_after_shown, [])


def gradient(X, Y):
    return(
        + 1*np.exp(-(((X-0)/0.2)**2+((Y+0)/0.2)**2))
        + 1.2*np.exp(-(((X-0.2)/0.1)**2+((Y+0.2)/0.1)**2))
        + 1.4*np.exp(-(((X+0.2)/0.05)**2+((Y+0.2)/0.05)**2))
    )


def main(
    nb_herds = 1,
    nb_generations = 10,
    nb_add_neurons = 0,
    period = 1,
    function = segments,
    reset_after_process = True,
    size = 20, # The higher the better the exploration is
    mutation_coefficient = 1,
    mutation_amplitude = 0.01,
    nb_tests = 1, # put higher to speedtest your CPU/GPU
    do_display_execution = False,
    display_results_mode = "console"
    ):
    # Replace nb_generations by 1 to see evolution frame by frame
    P = Gradient_Descent_Test()
    TB = TestBench(
        P, # Problem
        nb_herds,
        nb_generations,
        nb_add_neurons,
        period,
        function,
        reset_after_process,
        size,
        mutation_coefficient,
        mutation_amplitude,
        nb_tests,
        do_display_execution,
        display_results_mode
    )
    TB.set_estimated()
    TB.test("simple")

if __name__ == "__main__":
    pass
    main()

