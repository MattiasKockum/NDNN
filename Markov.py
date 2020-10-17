#!/usr/bin/env python3
  2 
  3 # Programme écrit par Mattias Kockum
  4 # Le 15 juillet 2020
  5 # Le but de ce programme est de créer une chaîne de Markov qui serait une IA
  6 # évolutive fonctionnant en temps réel avec mémoire
  7 
  8 import numpy as np
  9 import copy
 10 import turtle
 11 import matplotlib.pyplot as plt
 12 
 13 
 14 def sigmoid(x):
 15     return(2*((1/(1+2.7**-(x)))-0.5))
 16 
 17 
 18 class Problème():
 19     """
 20     Le cadre d'un problème adapté à une chaîne de Markov
 21     """
 22     def __init__(self, avertissement = True):
 23         self.avertissement = avertissement
 24         self.nb_capteurs = 0
 25         self.nb_acteurs = 0
 26         if self.avertissement:
 27             print("Attention __init__ du problème n'a pas bien été configurée")
 28 
 29     def action(self, entrées):
 30         if self.avertissement:
 31             print("Attention action du problème n'a pas bien été configurée")
 32         pass
 33 
 34     def état(self):
 35         if self.avertissement:
 36             print("Attention état du problème n'a pas bien été configurée")
 37         return(None)
 38 
 39     def experience(self, Chaîne):
 40         if self.avertissement:
 41             print(
 42                 "Attention experience du problème n'a pas bien été configurée"
 43             )
 44         score = self.points_temps_réel()
411 lignes copiées
