#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train and test my networks
"""

from AI import *

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
        self.__init__(self.displayed)


def main():
    P = Centre_Game_2(False)
    TB = TestBench(
        P,
        slices=[3, 3],
        regions=[
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False]
        ])
    TB.test(4)

if __name__ == "__main__":
    main()

