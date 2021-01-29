#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 25/01/2021
The aim of this program is to train and test my networks
One specific aim of this problem is to ensure the fact that my Network
is capable of creating memories
"""

from AI import *

import numpy as np


class Catch():
    """
    Catch is a game that takes place on a grid
    The player starts in the middle if the bottom layer and can only move
    on it.
    Signals will randomly apear on the top layer few turns after what an
    object will fall from where the signal was.
    The goal here is to test if the AI is capable of memorising the signal
    and move toward the place where the object will fall while the signal
    isn't present anymore.

    player = 0.1
    object = 0.2
    spot = 0.3
    """
    def __init__(self, do_run_display = False, do_end_display = False,
                 size = 4, max_score = 100):
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.nb_sensors = size*(2*size + 1)
        self.nb_actors = 3 # go left, stay, go right
        self.score = 0
        self.size = size
        self.max_score = max_score
        self.lost = False
        self.spotted = False
        self.falling_object = False
        self.spot = -1
        self.wait = 0
        self.object = [-1, -1] # [y, x]
        self.player = [size - 1, size]
        self.grid = np.zeros((size, 2*size + 1))
        self.grid[tuple(self.player)] = 0.1

    def experience(self, Network):
        while not self.end_condition():
            output = Network.process(self.state())
            self.action(output)
            if self.do_run_display:
                self.run_display()
        self.score_update()
        score = self.score
        self.reset()
        return(score)

    def end_condition(self):
        if self.lost or self.score == self.max_score:
            return(True)
        return(False)

    def state(self):
        return(self.grid.reshape((self.size*(2*self.size + 1), )))

    def score_update(self):
        self.score = self.score*(self.score>0)

    def action(self, output):
        direction = maxindex(output)
        # Moving the player
        if direction == 0 and self.player[1] != 0:
            # Go left
            self.grid[tuple(self.player)] = 0
            self.player[1] -= 1
            self.grid[tuple(self.player)] = 0.1
        if direction == 2 and self.player[1] != 2*self.size:
            # Go right
            self.grid[tuple(self.player)] = 0
            self.player[1] += 1
            self.grid[tuple(self.player)] = 0.1
        # Handeling the object
        if self.falling_object:
            # Moving the object
            self.grid[tuple(self.object)] = 0
            self.object[0] += 1
            if self.object == self.player:
                # CATCH !
                self.falling_object = False
                self.object = [-1, -1]
                self.score += 1
            else:
                # Continuing to fall
                self.grid[tuple(self.object)] = 0.2
            if self.object[0] == self.size - 1:
                # game over
                self.lost = True
        else:
            # If there is no object
            if not self.spotted:
                # Spotting the object
                self.spot = np.random.randint(2*self.size + 1)
                self.spotted = True
                self.grid[0, self.spot] = 0.3
                self.wait = self.size
            else:
                # Creating the object
                self.grid[0, self.spot] = 0
                if self.wait == 0:
                    self.grid[0, self.spot] = 0.2
                    self.object = [0, self.spot]
                    self.falling_object = True
                    self.spotted = False
                self.wait -= 1

    def run_display(self):
        for i in self.grid:
            for j in i:
                print(int(10*j), end = "")
            print("")
        print("")

    def end_display(self):
        print("score : {}".format(self.score))

    def __name__(self):
        return("Catch")

    def reset(self):
        self.__init__(self.do_run_display, self.do_end_display, self.size,
                      self.max_score)

    def clean(self):
        pass

def main():
    P = Catch(False, False, 4, 100)
    TB = TestBench(P, 1, 50, 9, 3, nb_tests=10)
    TB.test(0)
    TB.display_console(True)

if __name__ == "__main__":
    main()
