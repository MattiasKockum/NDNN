#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks
"""

from AI import *

import turtle
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
            self.action(*Network.process(self.state()))
        score = self.score_real_time()
        self.reset()
        return(score*(score>0) + 0)

    def score_real_time(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def __name__(self):
        return("Centre_Game_1")

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
            self.action(*Network.process(self.state()))
        score = self.score_real_time()
        self.reset()
        return(score*(score>0) + 0)

    def score_real_time(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def __name__(self):
        return("Centre_Game_2")

    def reset(self):
        self.__init__(self.displayed)


def main_test_game2():
    P = Centre_Game_2(False)
    TB = TestBench(
        P,
        nb_herds = 1,
        nb_generations = 5,
        nb_add_neurons = 9,
        period = 1,
        size = 5,
        mutation_coefficent = 0.1,
        mutation_amplitude = 0.001,
        nb_tests = 5,
        slices=[3, 3],
        regions=[
            [False, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False]
        ]
    )
    TB.test(5)

def main():
    #main_test_game2()
    main_test_gradient(1, 9, 0, 100, 1, 0.01, 1)


if __name__ == "__main__":
    pass
    main()

