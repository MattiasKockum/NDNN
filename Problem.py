#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an universal real time problem
"""


from ActivationFunctions import *

# Necessary
import numpy as np



class Problem(object):
    """
    The frame of any "live" problem
    The problem has to be coded in a copy of it
    just so that the function stay the same with every problem
    """
    def __init__(self, do_run_display = False, do_end_display = False):
        print("Warning  : __init__ was not fully configured")
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.nb_sensors = 1
        self.nb_actors = 1
        self.score = 0

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        while not self.end_condition():
            output = Network.process(self.state())
            self.action(output)
            if self.do_run_display:
                self.run_display()
        self.score_update()
        self.reset()
        return(self.score)

    def end_condition(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        print("Warning  : end_condition was not fully configured")
        return(True)

    def state(self):
        """
        Returns the state of the problem
        """
        print("Warning  : state was not fully configured")
        return(np.array([1]))

    # Other state related functions should be there

    def score_update(self):
        """
        Updates the score of the problem at the moment
        """
        print("Warning score_update was not fully configured")
        # score should always be > 0
        self.score = self.score*(self.score>0)

    def action(self, output):
        """
        Computes the consequences of the input_data on the problem
        """
        print("Warning  : action was not fully configured")
        pass

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
        return("An_Unnamed_Problem")

    def reset(self):
        """
        Resets the problem
        """
        print("Maybe some HUGE PROBLEM is coming at you")
        self.__init__(self.do_run_display, self.do_end_display)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        pass

