#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks
"""

from AI import *

class Centre_Game(Problem):
    """
    A problem where a mouse must find a piece of cheese which is always in the
    center of the field, the mouse spawns on a random location and knows where
    it is at any moment
    """
    def __init__(self, printing = False):
        self.nb_sensors = 2
        self.nb_actors = 2
        self.x = np.random.rand() - 0.5
        self.y = np.random.rand() - 0.5
        self.printing = printing
        self.nb_remaining_actions = 10
        if printing:
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
        if self.printing:
            self.t.setpos(250*self.x, 250*self.y)

    def state(self):
        return(self.x, self.y)

    def experience(self, Chaîne):
        """
        Tests the network
        """
        while self.nb_remaining_actions > 0:
            self.nb_remaining_actions -= 1
            self.action(*Chaîne.action(self.state()))
        score = self.points_real_time()
        self.reset()
        return(score)

    def points_real_time(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def reset(self):
        self.__init__(self.printing)

