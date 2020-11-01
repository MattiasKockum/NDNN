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

class Centre_Game_1(Problem):
    """
    A problem where a mouse must find a piece of cheese which is always in the
    center of the field, the mouse spawns on a random location and knows where
    it is at any moment
    """
    def __init__(self, displayed = False):
        self.nb_sensors = 2
        self.nb_actors = 2
        self.x = np.random.rand() - 0.5
        self.y = np.random.rand() - 0.5
        self.displayed = displayed
        self.nb_remaining_actions = 10
        self.period = 1
        if displayed:
            self.t = turtle.Turtle()
            self.t.speed(speed = 0)
            self.t.penup()
            self.t.setpos(250*self.x, 250*self.y)
            self.t.pendown()

    def action(self, dx, dy):
        """
        Takes in input the action of the network and applies it to the problem
        """
        self.x += dx
        self.y += dy
        if self.displayed:
            self.t.setpos(250*self.x, 250*self.y)

    def state(self):
        return(self.x, self.y)

    def experience(self, Network):
        """
        Tests the network
        """
        while self.nb_remaining_actions > 0:
            self.nb_remaining_actions -= 1
            self.action(*Network.process(self.state(), self.period))
        score = self.score_real_time()
        self.reset()
        return(score)

    def score_real_time(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def reset(self):
        self.__init__(self.displayed)


class Centre_Game_2(Problem):
    """
    A problem where a mouse must find a piece of cheese which is always in the
    center of the field, the mouse spawns on a random location and knows how
    far it is from it
    """
    def __init__(self, displayed = False):
        self.nb_sensors = 1
        self.nb_actors = 2
        self.x = np.random.rand() - 0.5
        self.y = np.random.rand() - 0.5
        self.displayed = displayed
        self.nb_remaining_actions = 10
        self.period = 1
        if displayed:
            self.display()

    def display(self):
        self.t = turtle.Turtle()
        self.t.speed(speed = 0)
        self.t.penup()
        self.t.setpos(250*self.x, 250*self.y)
        self.t.pendown()

    def action(self, dx, dy):
        """
        Takes in input the action of the network and applies it to the problem
        """
        self.x += dx
        self.y += dy
        if self.displayed:
            self.t.setpos(250*self.x, 250*self.y)

    def state(self):
        return((self.x**2 + self.y**2)**(1/2))

    def experience(self, Network):
        """
        Tests the network
        """
        while self.nb_remaining_actions > 0:
            self.nb_remaining_actions -= 1
            self.action(*Network.process(self.state(), self.period))
        score = self.score_real_time()
        self.reset()
        return(score)

    def score_real_time(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def reset(self):
        self.__init__(self.displayed)


class Gradient_Descent_Test():
    """
    """
    def __init__(self, timer = 1, results = [], do_display = False):
        self.timer = timer
        self.results = results
        self.do_display = do_display
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
        # below is a useful test to see if parallelization works
        #print(
        #    "Problem : {}\n     Network : {}\n     Network hash : {}".format(
        #        self.__hash__(),
        #        self.weight + self.bias,
        #        Network.__hash__()
        #    )
        #)
        self.weight = Network.weights[0][0]
        self.results.append((self.weight, self.bias))
        score = self.score_real_time()
        self.reset()
        if self.do_display and len(self.results) >= self.timer:
            self.display()
        return(score)

    def score_real_time(self):
        return(gradient(self.weight, self.bias))

    def reset(self):
        self.__init__(self.timer, self.results, self.do_display)

    def display(self, step=1):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        X = np.arange(-0.5, 0.5, 0.01)
        Y = np.arange(-0.5, 0.5, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = gradient(X, Y)

        # Plot the surface.
        surf = ax.plot_wireframe(X, Y, Z, color="#0F0F0F0F")

        xs = [i[0] for i in self.results]
        ys = [i[1] for i in self.results]
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


def gradient(X, Y):
    return(
        + 1*np.exp(-(((X-0)/0.2)**2+((Y+0)/0.2)**2))
        + 1.2*np.exp(-(((X-0.2)/0.1)**2+((Y+0.2)/0.1)**2))
        + 1.4*np.exp(-(((X+0.2)/0.05)**2+((Y+0.2)/0.05)**2))
    )

def main_test_gradient(
    nb_herds = 1,
    nb_generations = 100,
    nb_add_neurons = 0,
    size = 200,
    mutation_coefficient = 0.1,
    mutation_amplitude = 0.001,
    nb_tests = 1,
    do_display = False
    ):
    # Replace nb_generations by 1 to see evolution frame by frame
    P = Gradient_Descent_Test(nb_generations, [], False)
    TB = TestBench(
        P, # Problem
        nb_herds,
        nb_generations,
        nb_add_neurons,
        size,
        mutation_coefficient,
        mutation_amplitude,
        nb_tests,
        do_display
    )
    TB.test(0)
    return(P.results)

def main_test_game2():
    P = Centre_Game_2(False)
    TB = TestBench(
        P,
        nb_herds = 1,
        nb_generations = 5,
        nb_add_neurons = 9,
        size = 5,
        mutation_coefficent = 0.1,
        mutation_amplitude = 0.001,
        nb_tests = 5,
        slices=[3, 3],
        regions=[
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False]
        ]
    )
    TB.test(5)

def main():
    #main_test_game2()
    #main_test_gradient(1, 9, 0, 100, 1, 0.01, 1)
    main_test_gradient()


if __name__ == "__main__":
    pass
    main()

