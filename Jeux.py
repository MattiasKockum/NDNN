#!/usr/bin/env python3

# Programme écrit par Mattias Kockum
# Le 15 juillet 2020
# Le but de ce programme est de créer des jeux pour entraîner mes Chaînes

from Markov import *

class Jeu_Centre(Problème):
    """
    Un Problème où un point sur un tableau en 2D doit se déplacer vers le 
    centre, la chaîne connaît sa position
    """
    def __init__(self, affichage = False):
        self.x = np.random.rand() - 0.5
        self.y = np.random.rand() - 0.5
        self.affichage = affichage
        self.nb_actions_restantes = 10
        if affichage:
            self.t = turtle.Turtle()
            self.t.speed(speed = 0)
            self.t.penup()
            self.t.setpos(250*self.x, 250*self.y)
            self.t.pendown()

    def action(self, dx, dy):
        self.x += dx
        self.y += dy
        if self.affichage:
            self.t.setpos(250*self.x, 250*self.y)

    def état(self):
        return(self.x, self.y)

    def experience(self, Chaîne):
        while self.nb_actions_restantes > 0:
            self.nb_actions_restantes -= 1
            self.action(*Chaîne.action(self.état()))
        score = self.points_temps_réel()
        self.ràz()
        print(score)
        return(score)

    def points_temps_réel(self):
        return(1/(1 + (self.x**2 + self.y**2)))

    def ràz(self):
        self.__init__(self.affichage)
