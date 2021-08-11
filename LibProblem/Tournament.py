#!/usr/bin/env python3

import numpy as np

class TournamentTree(object):
    """
    A tournament tree
    """
    def __init__(self, indexes, super_l = None):
        self.l = len(indexes)
        if super_l == None:
            super_l = self.l
        self.score = np.zeros((super_l))
        self.winner = None
        if self.l == 0:
            self.l_child = None
            self.r_child = None
        elif self.l == 1:
            self.winner = indexes[0]
            self.l_child = None
            self.r_child = None
        else:
            mid = int(self.l/2)
            self.l_child = TournamentTree(indexes[0:mid], super_l)
            self.r_child = TournamentTree(indexes[mid:self.l+1], super_l)

    def next_opposition(self):
        if self.winner != None:
            return([self.winner])
        elif self.r_child.winner == None:
            return(self.r_child.next_opposition())
        elif self.l_child.winner == None:
            return(self.l_child.next_opposition())
        else:
            return([self.l_child.winner, self.r_child.winner])

    def win(self, winner):
        self.score[winner] += 1
        if winner in [self.l_child.winner, self.r_child.winner]:
            self.winner = winner
        if self.l_child.winner == None:
            self.l_child.win(winner)
        if self.r_child.winner == None:
            self.r_child.win(winner)

    def str_recursive(self, depth):
        s = ""
        if type(self.l_child) == type(self):
            s += self.l_child.str_recursive(depth+1)
        elif type(self.l_child) == int:
            s += (1+depth)*"\t" + str(self.l_child) + "\n"
        s += depth*"\t"
        if self.winner == None:
            s += "."
        else:
            s += str(self.winner)
        s += "\n"
        if type(self.r_child) == type(self):
            s += self.r_child.str_recursive(depth+1)
        elif type(self.r_child) == int:
            s += (1+depth)*"\t" + str(self.r_child) + "\n"
        return(s)

    def __str__(self):
        return(self.str_recursive(0))

    def __repr__(self):
        return(self.__str__())

