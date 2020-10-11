#!/usr/bin/env python3

import numpy as np
import turtle
from Markov import *

class Circuit():
    """
    Un circuit de voitures
    repère :

         ---------> +y
        |
        |
        |
        |
       \/
       +x
    """
    def __init__(self, taille=8):
        self.taille = taille
        self.pos0 = np.array([0.5, 0.5])
        self.dir0 = np.array([1.0, 0.0])
        self.parcours, self.chemin = self.génération_parcours()

    def __repr__(self):
        s = ""
        for x in range(self.taille):
            for y in range(self.taille):
                s += (
                    2*chr(24)*bool(self.parcours[x, y] == -1)
                    + ((str(int(self.parcours[x, y]))
                        + " "*bool(self.parcours[x, y] < 10))
                       *bool(self.parcours[x, y] > -1))
                    + "\n"*bool(y == self.taille - 1)
                )
        return(s)

    def génération_parcours(self):
        parcours = -np.ones((self.taille, self.taille))
        chemin = []
        i = 0
        pb = 0 # sert à ne pas rester bloqué
        x = 0
        y = 0
        while self.condition_fin(i, x, y):
            direction = np.random.randint(5)
            if direction == 0:
                if y + 1 < self.taille:
                    if parcours[x, y + 1] == -1:
                        parcours[x, y] = i
                        chemin.append((x, y))
                        i += 1
                        y += 1
                        pb = 0
            elif direction == 1:
                if x - 1 > 0:
                    if parcours[x - 1, y] == -1:
                        parcours[x, y] = i
                        chemin.append((x, y))
                        i += 1
                        x -= 1
                        pb = 0
            elif direction == 2:
                if y - 1 > 0:
                    if parcours[x, y - 1] == -1:
                        parcours[x, y] = i
                        chemin.append((x, y))
                        i += 1
                        y -= 1
                        pb = 0
            elif direction == 3:
                if x + 1 < self.taille:
                    if parcours[x + 1, y] == -1:
                        parcours[x, y] = i
                        chemin.append((x, y))
                        i += 1
                        x += 1
                        pb = 0
            if pb == 5:
                return(self.génération_parcours())
            pb += 1
        self.posfin = np.array([x, y])
        parcours[x, y] = i
        return(parcours, chemin)

    def condition_fin(self, i, x, y):
        """
        """
        return(
            i < self.taille
            or not (
                x == 0
                or x == self.taille - 1
                or y == 0
                or y == self.taille - 1)
              )


class Voiture(Problème):
    """
    Une voiture
    Un peu de physique :
        m*a = sum(F)
        m*a = moteur + friction
        ~ a = ~ moteur
        => v(t) = moteur*t + V0
        => pos(t) = (1/2)*moteur*t**2 + V0*t + pos0

        => moteur(t) = input(t)
        => v(t + Δt) = moteur(t)*Δt + v(t)
        => pos(t + Δt) = (1/2)*moteur(t)*Δt**2 + v(t)*Δt + pos(t)
    """
    def __init__(
            self,
            Δd,
            Δt,
            dmax,
            afficher = False,
            taille = 8,
            rayon_braquage = 1,
            qualité_moteur = 20):
        self.nb_capteur = 12 # dépend du nb de rayons que j'envoie...
        self.nb_acteurs = 2
        self.Circuit = Circuit(taille)
        # Je met cette ligne pour garder un circuit fixe pour voir
        """
        self.Circuit.parcours = np.array(
            [
                [ 0.,  1.,  2., -1., -1., -1., -1., -1.],
                [-1., -1.,  3., -1., -1., -1., -1., -1.],
                [-1., -1.,  4., -1., -1., -1., -1., -1.],
                [-1., -1.,  5., -1., -1., -1., -1., -1.],
                [-1., -1.,  6.,  7.,  8., -1., -1., -1.],
                [-1., -1., -1., -1.,  9., -1., -1., -1.],
                [-1., -1., -1., -1., 10., -1., -1., -1.],
                [-1., -1., -1., -1., 11., -1., -1., -1.]
            ]
        )
        """
        self.pos = self.Circuit.pos0
        self.dir = self.Circuit.dir0
        self.vitesse = np.array([0.0, 0.0])
        self.accélération = np.array([0.0, 0.0]) 
        self.taille = taille
        self.Δd = Δd
        self.Δt = Δt
        self.dmax = dmax
        self.rayon_braquage = rayon_braquage
        self.qualité_moteur = qualité_moteur
        self.afficher = afficher
        self.score_max = 0
        self.màj_état()
        # je met une position impossible pour rentrer dans la boucle continuer
        self.pos_précédente = np.array([-1, -1])

    def experience(self, Chaîne):
        if self.afficher:
            self.représenter()
        while self.continuer():
            action_Chaîne = Chaîne.action(self.état)
            self.action(*action_Chaîne)
        score = self.points_temps_réel()
        self.ràz()
        return(score)

    def action(self, pédale, volant):
        self.accélération += self.qualité_moteur*pédale*self.dir
        self.dir = rotation(self.dir, self.rayon_braquage*volant)
        projection_pos = (
            (1/2)*self.accélération*self.Δt**2
            + self.vitesse*self.Δt
            + self.pos
        )
        projection_direction = self.pos - projection_pos
        projection_distance = np.linalg.norm(projection_direction)
        distance_parcourue = self.rayon(projection_direction,
                                        projection_distance)
        self.pos_précédente = self.pos
        if distance_parcourue != projection_distance:
            # On se prend un obstacle
            self.vitesse = np.array([0.0, 0.0])
            norme_dir = (
                projection_direction
                /np.linalg.norm(projection_direction)
            )
            self.pos += norme_dir*distance_parcourue
        else :
            self.vitesse += self.accélération*self.Δt
            self.pos = projection_pos
        if self.afficher:
            self.poser_tortue(self.pos[0], self.pos[1])
        self.màj_état()

    def continuer(self):
        return(not np.array_equal(self.pos_précédente, self.pos))

    def capteurs(self):
        return(
            np.array([
                self.rayon(rotation(self.dir, np.pi/3), self.dmax),
                self.rayon(rotation(self.dir, -np.pi/3), self.dmax),
                self.rayon(self.dir, self.dmax)
            ])
        )

    def màj_état(self):
        """
        Les premiières valeurs correspondent à la positions, puis la direction,
        l'accélération, les capteurs et enfin les points
        """
        self.état = (
            np.array([
                *self.pos,
                *self.dir,
                *self.accélération,
                *self.capteurs(),
                *self.prochaine_pos(),
                self.points_temps_réel()
            ]
        ))

    def prochaine_pos(self):
        if self.points_temps_réel() + 1 < len(self.Circuit.chemin):
            return(self.Circuit.chemin[int(self.points_temps_réel()) + 1])
        return((-1, -1))

    def état_pos(self, pos):
        """
        Sert à obtenir l'état d'un endroit du circuit (mur ou route)
        """
        if (pos[0] < 0
            or self.taille < pos[0] + 1
            or pos[1] < 0
            or self.taille < pos[1] + 1
           ):
            return(-1)
        return(self.Circuit.parcours[
            int(np.floor(pos[0])),
            int(np.floor(pos[1]))
        ])

    def rayon(self, direction, distance):
        """
        le rayon part de pos dans la direction direction, avance avec un
        pas de Δd, il renvoie la distance à laquelle il s'arrête :
        soit celle à un mur, soit quand il a atteint distance
        """
        itérateur = 0
        while itérateur*self.Δd < distance:
            itérateur += 1
            if self.état_pos(
                self.pos + direction*itérateur*self.Δd/np.linalg.norm(direction)
            ) == -1:
                return((itérateur - 1)*self.Δd)
        return(distance)

    def points_temps_réel(self):
        score_endroit = self.état_pos(self.pos)
        if score_endroit > self.score_max:
            self.score_max = score_endroit
        return(self.score_max)

    def représenter(self):
        self.t = turtle.Turtle()
        self.t.speed(speed = 0)
        self.t.color("brown")
        for x in range(self.taille):
            for y in range(self.taille):
                if self.Circuit.parcours[x, y] == -1:
                    # On fait un carré
                    self.t.penup()
                    self.poser_tortue(x, y)
                    self.t.pendown()
                    self.poser_tortue(x, y + 1)
                    self.poser_tortue(x + 1, y + 1)
                    self.poser_tortue(x + 1, y)
                    self.poser_tortue(x, y)
        self.t.penup()
        self.poser_tortue(0, 0)
        self.t.pendown()
        self.poser_tortue(self.taille, 0)
        self.poser_tortue(self.taille, self.taille)
        self.poser_tortue(0, self.taille)
        self.poser_tortue(0, 0)
        self.t.penup()
        self.t.color("black")
        self.poser_tortue(self.Circuit.pos0[0], self.Circuit.pos0[1])
        self.t.pendown()

    def poser_tortue(self, x, y):
        self.t.setpos((500/self.taille)*x - 250, (500/self.taille)*y - 250)

    def ràz(self):
        if self.afficher:
            self.t.clear()
        self.__init__(
            self.Δd,
            self.Δt,
            self.dmax,
            self.afficher,
            self.taille,
            self.rayon_braquage,
            self.qualité_moteur
        )


def rotation(vecteur2D, angle): # angle en radians
    matrice_rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    return(np.matmul(matrice_rotation, vecteur2D))


print("""
taille_circuit = 8
affichage = False
V = Voiture(0.01, 0.01, taille_circuit, affichage)

nb_capteurs = 12
nb_acteurs = 2
nb_neurones = 18
Problème = V
taille_troupeau = 20
coef_mutation = 0.03
nb_tests = 10
T = Troupeau(
      nb_capteurs,
      nb_acteurs,
      nb_neurones,
      Problème,
      taille_troupeau,
      coef_mutation,
      nb_tests
     )

for i in range(5000):
    T.évoluer()
""")
def main():
    V = Voiture(0.01, 0.01, 4, True)
    T = Troupeau(12, 2, 17, V, 60,)
    C = T.membres[0]
    print(V.experience(C))
    V.t.mainloop()
    V.t.bye()

if __name__ == "__main__":
    pass
    #main()
