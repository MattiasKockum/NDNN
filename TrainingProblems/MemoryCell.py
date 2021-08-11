#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 16/02/2021
The aim of this program is to create an AI
    purposed at testing the cration of memory cells
"""

class MemoryCell(Problem):
    """
    A MemoryCell problem
    """
    def __init__(self, do_run_display = False, do_end_display = False,
                 total_turns = 3):
        self.do_run_display = do_run_display
        self.do_end_display = do_end_display
        self.nb_sensors = 1
        self.nb_actors = 1
        self.score = 0
        self.output = -1
        self.total_turns = total_turns
        self.remaining_turns = self.total_turns
        self.number = np.random.random()

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
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
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        """
        if self.remaining_turns == 0 :
            return(True)
        return(False)

    def state(self):
        """
        Returns the state of the problem
        """
        if self.remaining_turns == self.total_turns:
            return(self.number)
        return(-1)

    # Other state related functions should be there

    def score_update(self):
        """
        Updates the score of the problem at the moment
        """
        self.score = 0
        self.score = self.score*(self.score>0)
        # score should always be > 0

    def action(self, output):
        """
        Computes the consequences of the input_data on the problem
        """
        self.output = output
        self.remaining_turns -= 1

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
        return("MemoryCell")

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(self.do_run_display, self.do_end_display)

    def clean(self):
        """
        Removes eventual values that stay upon reset
        """
        pass

