#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an universal real time problem
"""


# Necessary
import numpy as np

# Class specific

# For Main
import sys
sys.path.append(".")
sys.path.append("..")
from AI import *


#class YourProblem(Problem):
class Problem(object):
    """
    The frame of any "live" problem
    The problem has to be coded in a copy of it
    just so that the function stay the same with every problem
    """
    def __init__(self, do_run_display = False, do_end_display = False):
        print("Warning  : __init__ was not fully configured")
        # Common
        self.nb_sensors = 1
        self.nb_actors = 1
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        # Class specific

    def experience(self, networks):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        self.experience_preparation(networks)
        while not self.experience_ended():
            self.problem_preparation()
            while not self.problem_ended():
                self.action(playing_index)
                if self.do_run_display:
                    self.run_display()
            self.reset()
        return(self.score)

    def experience_preparation(self, networks):
        # Values set here are not affected by reset of the problem
        # Common
        self.score = np.zeros((len(networks)))
        self.networks = networks
        # Class specific

    def problem_preparation(self):
        # Common
        self.score = np.zeros((len(networks)))
        playing_index = self.organisation()
        # Class specific

    def experience_ended(self):
        """
        True if every network has been evaluated
        False otherwise
        """
        print("Warning  : experience_ended was not fully configured")
        return(True)

    def organisation(self):
        """
        Return the indexes of the network(s) that must play the next game
        Can be a tree for 1v1
        Can be a line for solo evaluation
        Can be everyone at the same time etc
        """
        print("Warning  : organisation was not fully configured")
        return(0)

    def problem_ended(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        print("Warning  : problem_ended was not fully configured")
        return(True)

    def state(self):
        """
        Returns the state of the problem
        """
        print("Warning  : state was not fully configured")
        return(np.array([1]))

    # Other state related functions should be there

    def action(self, playing_index):
        """
        Computes what the networks do and puts the score accordingly
        """
        print("Warning  : action was not fully configured")
        #network = self.network[playing_index[0]]
        #output = network.process(self.state())
        self.score[playing_index] = 0

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
        print("Warning : end_display was not fully configured")

    # Other display functions should be put here

    def __name__(self):
        return("Problem")

    def reset(self):
        """
        Resets the problem
        """
        print("Maybe some HUGE PROBLEM if not configured")
        self.__init__(self.do_run_display, self.do_end_display)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        print("Maybe some HUGE PROBLEM if not configured")
        self.__init__(self.do_run_display, self.do_end_display)


# External functions of the problem should be put here


def main(parameters):
    P = YourProblem()
    H = Herd(P.nb_sensors, P.nb_actors,
             0, size=2, mutation_amplitude=0.01)
    H.evolve(P, 200)

if __name__ == "__main__":
    main(sys.argv)
