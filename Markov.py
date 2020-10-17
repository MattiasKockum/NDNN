#!/usr/bin/env python3

# Programme écrit par Mattias Kockum
# Le 15 juillet 2020
# Le but de ce programme est de créer une chaîne de Markov qui serait une IA
# évolutive fonctionnant en temps réel avec mémoire

import numpy as np
import copy
import turtle
import matplotlib.pyplot as plt


def sigmoid(x):
	return(2*((1/(1+2.7**-(x)))-0.5))


class Problème():
    """
    Le cadre d'un problème adapté à une chaîne de Markov
    """
    def __init__(self, avertissement = True):
        self.avertissement = avertissement
        self.nb_capteurs = 0
        self.nb_acteurs = 0
        if self.avertissement:
            print("Attention __init__ du problème n'a pas bien été configurée")

    def action(self, entrées):
        if self.avertissement:
            print("Attention action du problème n'a pas bien été configurée")
        pass

    def état(self):
        if self.avertissement:
            print("Attention état du problème n'a pas bien été configurée")
        return(None)

    def experience(self, Chaîne):
        if self.avertissement:
            print(
                "Attention experience du problème n'a pas bien été configurée"
            )
        score = self.points_temps_réel()
        self.ràz()
        return(score)

    def points_temps_réel(self):
        if self.avertissement:
            print("Attention points_temps_réel du problème n'a pas bien"
                  + "été configurée")
        return(0)

    def ràz(self):
        self.__init__(self.avertissement)

class Troupeau():
    """
    Troupeau de chaînes de Markov pouvant évoluer en se reproduisant
    """
    def __init__(
            self,
            nb_capteurs,
            nb_acteurs,
            nb_neurones_sup,
            Problème,
            taille = 30,
            coefficient_mutation = 0, # si 0 alors init spéciale
            nb_tests = 10):
        """
        Initialisation
        """
        self.taille = taille
        self.nb_capteurs = nb_capteurs
        self.nb_acteurs = nb_acteurs
        self.nb_neurones_sup = nb_neurones_sup
        self.coefficient_mutation = coefficient_mutation
        self.Problème = Problème
        self.membres = [
            Chaine(nb_capteurs, nb_acteurs, nb_neurones_sup)
            for i in range(taille)
        ]
        if self.coefficient_mutation == 0:
            # Ça je l'ajuste au feeling hein
            self.coefficient_mutation = (
                1/(self.nb_neurones_sup*(self.nb_neurones_sup+1))
            )
        self.nb_tests = nb_tests
        self.tableau_scores = []

    def évoluer(self, nb_itérations):
        """
        Fait évoluer le troupeau
        le principe est de multiplier ne nombre de ceux qui réussissent
        (en fonction de leur score)
        puis de tous les faire muter
        Cela doit être équivalent à une descente de gradient "à l'arrache"
        """
        for itération in range(nb_itérations):
            #print("Évolution !")
            # On calcule le score normalisé des chaînes
            proba_reproduction = self.performances()
            # On met à jour la liste des membres (il y a des doublons là)
            nouveaux_membres = [
                copy.deepcopy(
                    np.random.choice(
                    self.membres,
                    p=proba_reproduction
                    )
                )
            for i in range(self.taille)
            ]
            self.membres = nouveaux_membres
            # Hop on fait muter tout ce beau monde
            for chaîne in self.membres:
                chaîne.muter(self.coefficient_mutation)
            self.tableau_scores.append(sum(self.score)/self.taille)
            #print("Moyenne : " + str(sum(self.score)/self.taille))
            #print("Min : " + str(min(self.score)))
            #print("Max : " + str(max(self.score)))
            #print("\nTableau des moyennes : {}\n".format(self.tableau_scores))
        return(self.tableau_scores)

    def performances(self):
        """
        On évalue la performance de la chaîne
        Puis on normalise les scores pour effectuer des calculs de proba
        """
        self.score = np.zeros(self.taille)
        for i, j in enumerate(self.membres):
            points = (
                sum([self.Problème.experience(j) for k in range(self.nb_tests)])
                /self.nb_tests
            )
            #print(points)
            if points > 0:
                self.score[i] = points
            else:
                # Je retire les points négatifs pour les probas, de toute
                # façon un score nul équivaut à auqune reproduction
                self.score[i] = 0
        if list(self.score) == list(np.zeros(self.taille)):
            # Si toutes les chaînes ont fait un score nul
            # alors je les fait toutes se reproduire une fois et muter
            # une fois
            self.score = np.ones(self.taille)
        score_modifié = self.modif_score(self.score)
        return(score_modifié)

    def modif_score(self, score):
        """
        On tripatouille les scores pour jouer sur la reproduction
        Et faire en sorte que leur somme fasse 1 (proba)
        """
        return(score/sum(score))

    def mise_à_l_échelle(self, membres_reproducteurs):
        """
        On agrandit la matrice si besoin est pour la faire se reproduire/muter
        avec une matrice plus grande
        """
        neurones_en_plus = [0]*len(membres_reproducteurs)
        nb_neurones_sup = membres_reproducteurs[0].nb_neurones
        for index, membre in enumerate(membres_reproducteurs[1:]):
            différence = membre.nb_neurones - nb_neurones_sup
            if différence > 0:
                nb_neurones_sup = membre.nb_neurones
                for i in range(index+1):
                    neurones_en_plus[i] += différence
            else:
                neurones_en_plus[index+1] -= différence
        for index, membre in enumerate(membres_reproducteurs):
            membre.ajouter_neurones(neurones_en_plus[index])
        return(membres_reproducteurs[0].nb_neurones)


class Chaine():
    """
    Chaîne de markov
    Chaque neurone prend en entrée tout les autres neurones * le poids de
    la conection + son biais (d'où les deux couches de matrice)
    Le principe est que l'environnement influe sur certain neurones
    prédéfinis et à chaque ittération de l'environnement les neurones dans
    les couches mettent à jour leur valeur
    Exemple :
        temps 0:
            -0-0-0-0-
        temps 1:
            -o-0-0-0-
        temps 2:
            -0-o-0-0-
    Le réseau est donc vide au temps 0 et ne réagit pas avant un petit
    instant, mais à partir de là il possède une "mémoire de travail"
    Les nb_capteurs premiers neurones sont les capteurs, les nb_acteurs
    derniers neurones sont les acteurs
    """
    def __init__(
            self,
            nb_capteurs,
            nb_acteurs,
            nb_neurones_sup,
            poids = None,
            biais = None):
        self.nb_capteurs = nb_capteurs
        self.nb_acteurs = nb_acteurs
        self.nb_neurones = nb_neurones_sup + nb_acteurs + nb_capteurs
        self.valeurs = np.zeros((self.nb_neurones))
        if type(poids) == type(None) and type(biais) == type(None):
            self.poids = (
                np.random.rand(self.nb_neurones, self.nb_neurones)
                - 0.5
            )
            self.biais = np.random.rand(self.nb_neurones) - 0.5
        elif (
            poids.shape == (nb_neurones, nb_neurones)
            and biais.shape == (nb_neurones,)
        ):
            self.poids = poids
            self.biais = biais
        else :
            raise(ValueError("Les matrices en entrée n'ont pas le bon format"))

    def __repr__(self):
        return("Nombre de neurones : {}\n".format(self.nb_neurones)
               + "Nombre de capteurs : {}\n".format(self.nb_capteurs)
               + "Nombre d'acteurs : {}\n".format(self.nb_acteurs)
               + "poids :\n{}\n".format(self.poids)
               + "biais :\n{}\n".format(self.biais)
               + "valeurs :\n{}\n".format(self.valeurs))

    def entrée(self, valeurs_entrées):
        self.valeurs[:self.nb_capteurs] += valeurs_entrées

    def sortie(self):
        return(self.valeurs[-self.nb_acteurs:])

    def itération(self):
        """
        On fait avancer le temps de une étape en mettant la liste en entrée et
        en mettant à jour l'état de réseau
        """
        self.valeurs = sigmoid(
            np.matmul(self.poids, self.valeurs)
            + self.biais)

    def action(self, entrée):
        """
        Ce que fait la chaîne à un moment donné en fonction de son entrée et
        de ces valeurs en mémoire
        """
        self.entrée(entrée)
        self.itération()
        return(self.sortie())

    def ajouter_neurones(self, neurones_en_plus=1):
        # On crée des nouvelles matrices poids et biais plus grandes
        nouveaux_poids = np.zeros((self.poids.shape[0] + neurones_en_plus,
                                   self.poids.shape[1] + neurones_en_plus))
        nouveaux_biais = np.zeros((self.biais.shape[0] + neurones_en_plus, ))
        nouvelles_valeurs = np.zeros(self.nb_neurones + neurones_en_plus)
        # On affecte les anciennes valeurs dans les nouvelles matrices
        nouveaux_poids[:self.nb_capteurs, :self.nb_capteurs] = (
            self.poids[:self.nb_capteurs, :self.nb_capteurs])
        nouveaux_poids[-self.nb_capteurs-1:, :self.nb_capteurs] = (
            self.poids[-self.nb_capteurs-1:, :self.nb_capteurs])
        nouveaux_poids[-self.nb_capteurs-1:, -self.nb_capteurs-1:] = (
            self.poids[-self.nb_capteurs-1:, -self.nb_capteurs-1:])
        nouveaux_poids[:self.nb_capteurs, -self.nb_capteurs-1:] = (
            self.poids[:self.nb_capteurs, -self.nb_capteurs-1:])
        nouveaux_biais[:self.nb_capteurs] = (
            self.biais[:self.nb_capteurs])
        nouveaux_biais[-self.nb_capteurs-1:] = (
            self.biais[-self.nb_capteurs-1:])
        nouvelles_valeurs[:self.nb_capteurs] = (
            self.valeurs[:self.nb_capteurs])
        nouvelles_valeurs[-self.nb_capteurs-1:] = (
            self.valeurs[-self.nb_capteurs-1:])
        # On met à jour la chaîne
        self.poids = nouveaux_poids
        self.biais = nouveaux_biais
        self.valeurs = nouvelles_valeurs
        self.nb_neurones += neurones_en_plus
        # On renvoie le nouveau nombre de neurones
        return(self.nb_neurones)

    def muter(self, coefficient_mutation):
        """
        On fait muter la chaîne
        Attention c'est une mutatione en place
        La chaîne est modifiée
        """
        ajouts_neurones = 0
        for i in range(self.nb_neurones*(self.nb_neurones + 1) + 1):
            # Si il y a une mutation
            if np.random.choice(
                [True, False],
                p = [coefficient_mutation, 1 - coefficient_mutation]
            ):
                # Si le compteur est dans les poids on en modifie un
                if i < self.nb_neurones**2:
                    self.poids[i//self.nb_neurones][i%self.nb_neurones] = (
                        np.random.rand() - 0.5
                    )
                # Sinon si on est dans les baisi on en modifie un
                elif i < self.nb_neurones*(self.nb_neurones + 1):
                    self.biais[i - self.nb_neurones**2] = (
                        np.random.rand() - 0.5
                    )
                # Sinon on ajoute un neurone (pas implémenté encore...)
                else:
                    ajouts_neurones += 1
            pass

class BancDeTest():
    """
    Un banc de test pour vérifier que tout se passe bien avec mes chaînes
    """

    def __init__(
        self,
        problème,
        nb_troupeaux = 7,
        nb_générations = 300,
        taille = 10,
        coefficient_mutation = 0.01,
        nb_tests = 5
    ):
        self.suites = []
        self.problème = problème
        self.nb_capteurs = problème.nb_capteurs
        self.nb_acteurs = problème.nb_acteurs
        self.couleurs = ["r", "g", "b", "c", "m", "y", "k"]
        self.nb_troupeaux = nb_troupeaux
        self.nb_générations = nb_générations
        self.nb_neurones_sup = 0
        self.taille = taille
        self.coefficient_mutation = coefficient_mutation
        self.nb_tests = nb_tests
        self.plage_simple = self.nb_troupeaux*[1]
        self.plage_nb_neurones_sup = [0, 1, 2, 3, 4, 5, 6]
        self.plage_tailles = [2, 5, 10, 15, 20, 30]
        self.plage_coefficients_mutation = [0.01, 0.005, 0.002, 0.001]
        self.plage_nb_tests = [1, 2, 3, 4, 5, 6]
        self.suites_archives = []

    def représenter(self):
        for indice, suite in enumerate(self.suites):
            plt.plot(
                [k for k in range(len(suite))],
                suite,
                self.couleurs[indice%len(self.couleurs)]+"-*"
            )
        plt.show()

    def test(self, mode = 0, nb_générations = None, plage = None):
        if nb_générations == None:
            nb_générations = self.nb_générations
        base = [
            self.nb_capteurs,
            self.nb_acteurs,
            self.nb_neurones_sup,
            self.problème,
            self.taille,
            self.coefficient_mutation,
            self.nb_tests
        ]
        if mode in [0, "simple"]:
            if plage == None:
                plage = self.plage_simple
            tableau_entrées = np.array([base for i in range(len(plage))])
        if mode in [1, "nb_neurones"]:
            if plage == None:
                plage = self.plage_nb_neurones_sup
            tableau_entrées = np.array([base for i in range(len(plage))])
            tableau_entrées[:,2] = plage
        if mode in [2, "tailles"]:
            if plage == None:
                plage = self.plage_tailles
            tableau_entrées = np.array([base for i in range(len(plage))])
            tableau_entrées[:,4] = plage
        if mode in [3, "coefficients_mutation"]:
            if plage == None:
                plage = self.plage_coefficients_mutation
            tableau_entrées = np.array([base for i in range(len(plage))])
            tableau_entrées[:,5] = plage
        if mode in [4, "nb_tests"]:
            if plage == None:
                plage = self.plage_nb_tests
            tableau_entrées = np.array([base for i in range(len(plage))])
            tableau_entrées[:,6] = plage
        if mode in [5, "multiple"]:
            if plage == None:
                raise(ValueError("Il faut rentrer une plage de valeurs"))
            tableau_entrées = np.array([base for i in range(len(plage))])
            tableau_entrées = plage
        for i in range(len(plage)):
            print(*tableau_entrées[i], self.couleurs[i%len(self.couleurs)])
            T = Troupeau(*tableau_entrées[i])
            self.suites.append(T.évoluer(nb_générations))
        self.représenter()
        self.suites_archives.append(self.suites)
        self.suites = []


print(
"""
P = Jeu_Centre(False)
BdT = BancDeTest(P)
BdT.test(3)
"""
)
