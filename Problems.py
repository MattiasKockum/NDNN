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
            self.action(*Network.action(self.state()))
        score = self.points_real_time()
        self.reset()
        return(score)

    def points_real_time(self):
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
            self.action(*Network.action(self.state()))
        score = self.points_real_time()
        self.reset()
        return(score)

    def points_real_time(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def reset(self):
        self.__init__(self.displayed)


class Gradient_Descent_Test():
    """
    """
    def __init__(self, results = []):
        self.results = results
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
        self.results.append((self.weight, self.bias))
        score = self.score_real_time()
        self.reset()
        return(score)

    def score_real_time(self):
        return(gradient(self.weight, self.bias))

    def reset(self):
        self.__init__(self.results)

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
        #+ 1*np.exp(-(((X-3)/2)**2+((Y+3)/3)**2))
        #+ 1*np.exp(-(((X+2)/1)**2+((Y-2)/2)**2))
        #+ 1*np.exp(-(((X-2)/3)**2+((Y+5)/1)**2))
        #+ 1*np.exp(-(((X+4)/2)**2+((Y+1)/5)**2))
        #+ 1*np.exp(-(((X-4)/4)**2+((Y-2)/4)**2))
        #+ 1*np.exp(-(((X+3)/1)**2+((Y-3)/3)**2))
        #+ 1*np.exp(-(((X-2)/2)**2+((Y+3)/3)**2))
    )

def main():
    #main_test_game2()
    main_test_gradient()

def main_test_gradient(
    nb_herds = 1,
    nb_generations = 900,
    nb_add_neurons = 0,
    size = 2,
    mutation_coefficient = 1,
    mutation_amplitude = 0.01,
    nb_tests = 1
    ):
    P = Gradient_Descent_Test()
    TB = TestBench(
        P, # Problem
        nb_herds,
        nb_generations,
        nb_add_neurons,
        size,
        mutation_coefficient,
        mutation_amplitude,
        nb_tests
    )
    TB.test(0)
    P.display(size)
    return(P.results)

def main_test_game2():
    P = Centre_Game_2(False)
    TB = TestBench(
        P,
        nb_herds = 1,
        nb_generations = 50,
        nb_add_neurons = 9,
        size = 50,
        mutation_coefficent = 0.0001,
        mutation_amplitude = 0.01,
        nb_tests = 16,
        slices=[3, 3],
        regions=[
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False]
        ]
    )
    TB.test(5)


if __name__ == "__main__":
    pass
    main()

