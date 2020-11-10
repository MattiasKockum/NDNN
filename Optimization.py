#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 9/11/2020
The aim of this program is to create an problem
    capable of optimization in the init values of an actual AI
"""


from Ai import *


class Init_Values_Opt(Problem):
    """
    This problem is entended to try and find the best suited init values
    for a problem (size, mutation_amplitude/coefficent period etc...)
    It is mainly made for automation of the TestBench class' work
    """
    def __init__(self, problem, time_span = 3600):
        self.nb_sensors = 2 # The init values (2*(2+1)=6)
        self.nb_actors = 0
        self.score = 0
        self.problem = problem
        self.time_span = time_span

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        self.end_time = time.time() + self.time_span
        self.Herd = Herd(
            nb_add_neurons = int(100*(Network.weights[0][0] + 0.5)),
            period = int(100*(Network.weights[0][1] + 0.5)),
            size = int(100*(Network.weights[1][0] + 0.5)),
            mutation_coefficent = Network.weights[1][1] + 0.5,
            mutation_amplitude = Network.bias[0] + 0.5,
            nb_tests = int(100*(Network.bias[1] + 0.5))
        )
        while not self.end_condition():
            self.action()
        self.score_update()
        score = self.score
        self.reset()
        return(score*(score>0))

    def end_condition(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        if time.time() > self.end_time:
            return(True)
        return(True)

    def state(self):
        """
        Returns the state of the problem
        """
        return(np.array([1]))

    # Other state related functions should be there

    def score_update(self):
        """
        Updates the score of the problem at the moment
        """
        # score should always be > 0
        # Maybe there is a better way of analysing the curve
        self.score = self.Herd.array_scores[-1]

    def action(self):
        """
        Computes the consequences of the input_data on the problem
        """
        self.Herd.evolve(self.problem, 1, False)

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        pass

    # Other display functions should be put here

    def __name__(self):
        return("Init_Values_Opt")

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.problem)

