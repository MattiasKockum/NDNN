#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train a car to drive itself
"""

import numpy as np
import turtle
from AI import *

class Circuit():
    """
    A circuit for cars
    """
    def __init__(self, size=8):
        self.size = size
        self.pos0 = np.array([0.5, 0.5])
        self.dir0 = np.array([1.0, 0.0])
        self.road, self.path = self.road_generation()
        self.path_len = len(self.path)

    def __repr__(self):
        s = ""
        for x in range(self.size):
            for y in range(self.size):
                s += (
                    2*chr(24)*bool(self.road[x, y] == -1)
                    + ((str(int(self.road[x, y]))
                        + " "*bool(self.road[x, y] < 10))
                       *bool(self.road[x, y] > -1))
                    + "\n"*bool(y == self.size - 1)
                )
        return(s)

    def road_generation(self):
        road = -np.ones((self.size, self.size))
        path = []
        i = 0
        pb = 0 # for not getting stuck
        x = 0
        y = 0
        while self.end_condition(i, x, y):
            direction = np.random.randint(5)
            if direction == 0:
                if y + 1 < self.size:
                    if road[x, y + 1] == -1:
                        road[x, y] = i
                        path.append((x, y))
                        i += 1
                        y += 1
                        pb = 0
            elif direction == 1:
                if x - 1 > 0:
                    if road[x - 1, y] == -1:
                        road[x, y] = i
                        path.append((x, y))
                        i += 1
                        x -= 1
                        pb = 0
            elif direction == 2:
                if y - 1 > 0:
                    if road[x, y - 1] == -1:
                        road[x, y] = i
                        path.append((x, y))
                        i += 1
                        y -= 1
                        pb = 0
            elif direction == 3:
                if x + 1 < self.size:
                    if road[x + 1, y] == -1:
                        road[x, y] = i
                        path.append((x, y))
                        i += 1
                        x += 1
                        pb = 0
            if pb == 5:
                return(self.road_generation())
            pb += 1
        self.posfin = np.array([x, y])
        road[x, y] = i
        return(road, path)

    def end_condition(self, i, x, y):
        """
        """
        return(
            i < self.size
            or not (
                x == 0
                or x == self.size - 1
                or y == 0
                or y == self.size - 1)
              )


class Car(Problem):
    """
    A car
    A little bit of physics:
        m*a = sum(F)
        m*a = engine + friction
        ~ a = ~ engine
        => speed(t) = engine*t + speed(0)
        => pos(t) = (1/2)*engine*t**2 + speed(0)*t + pos(0)

        => engine(t) = input(t)
        => speed(t + Δt) = engine(t)*Δt + speed(t)
        => pos(t + Δt) = (1/2)*engine(t)*Δt**2 + speed(t)*Δt + pos(t)
    """
    def __init__(
        self,
        displayed = False,
        Δd = 0.01,
        Δt = 0.01,
        dmax = 4,
        size = 8,
        turning_circle = 1,
        engine_quality = 20
    ):
        self.Circuit = Circuit(size)
        # A fixed circuit if needed
        """
        self.Circuit.road = np.array(
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
        self.speed = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0]) 
        self.size = size
        self.Δd = Δd
        self.Δt = Δt
        self.dmax = dmax
        self.turning_circle = turning_circle
        self.engine_quality = engine_quality
        self.displayed = displayed
        self.score_max = 0
        self.state_update()
        # An impossible position in order to get into the loop
        self.pos_precedente = np.array([-1, -1])
        self.nb_sensors = 9 + len(self.captors())
        self.nb_actors = 2

    def experience(self, Chaîne):
        if self.displayed:
            self.display()
        while self.continue_condition():
            action_Chaîne = Chaîne.action(self.state)
            self.action(*action_Chaîne)
        score = self.points_real_time()
        self.reset()
        return(score)

    def action(self, pedale, volant):
        self.acceleration += self.engine_quality*pedale*self.dir
        self.dir = rotation(self.dir, self.turning_circle*volant)
        pos_projection = (
            (1/2)*self.acceleration*self.Δt**2
            + self.speed*self.Δt
            + self.pos
        )
        direction_projection = self.pos - pos_projection
        distance_projection = np.linalg.norm(direction_projection)
        drived_distance = self.ray(
            direction_projection,
            distance_projection
        )
        self.pos_precedente = self.pos
        if drived_distance != distance_projection:
            # On se prend un obstacle
            self.speed = np.array([0.0, 0.0])
            norm_dir = (
                direction_projection
                /np.linalg.norm(direction_projection)
            )
            self.pos += norm_dir*drived_distance
        else :
            self.speed += self.acceleration*self.Δt
            self.pos = pos_projection
        if self.displayed:
            self.put_down_turtle(self.pos[0], self.pos[1])
        self.state_update()

    def continue_condition(self):
        return(not np.array_equal(self.pos_precedente, self.pos))

    def captors(self):
        return(
            np.array([
                self.ray(rotation(self.dir, np.pi/3), self.dmax),
                self.ray(rotation(self.dir, -np.pi/3), self.dmax),
                self.ray(self.dir, self.dmax)
            ])
        )

    def state_update(self):
        self.state = (
            np.array([
                *self.pos,
                *self.dir,
                *self.acceleration,
                *self.captors(),
                *self.next_pos(),
                self.points_real_time()
            ]
        ))

    def next_pos(self):
        if self.points_real_time() + 1 < len(self.Circuit.path):
            return(self.Circuit.path[int(self.points_real_time()) + 1])
        return((-1, -1))

    def state_pos(self, pos):
        """
        Returns the state of a postion (wall or road)
        """
        if (pos[0] < 0
            or self.size < pos[0] + 1
            or pos[1] < 0
            or self.size < pos[1] + 1
           ):
            return(-1)
        return(self.Circuit.road[
            int(np.floor(pos[0])),
            int(np.floor(pos[1]))
        ])

    def ray(self, direction, distance):
        """
        The ray starts from pos in the direction direction, go forward
        with a step Δd, returns the distance at which it stoped : wether
        it stoped from collision or from max range
        """
        iterator = 0
        while iterator*self.Δd < distance:
            iterator += 1
            if self.state_pos(
                self.pos + direction*iterator*self.Δd/np.linalg.norm(direction)
            ) == -1:
                return((iterator - 1)*self.Δd)
        return(distance)

    def points_real_time(self):
        score_endroit = self.state_pos(self.pos)
        if score_endroit > self.score_max:
            self.score_max = score_endroit
        return(self.score_max)

    def display(self):
        self.t = turtle.Turtle()
        self.t.speed(speed = 0)
        self.t.color("brown")
        for x in range(self.size):
            for y in range(self.size):
                if self.Circuit.road[x, y] == -1:
                    # Making a square
                    self.t.penup()
                    self.put_down_turtle(x, y)
                    self.t.pendown()
                    self.put_down_turtle(x, y + 1)
                    self.put_down_turtle(x + 1, y + 1)
                    self.put_down_turtle(x + 1, y)
                    self.put_down_turtle(x, y)
        self.t.penup()
        self.put_down_turtle(0, 0)
        self.t.pendown()
        self.put_down_turtle(self.size, 0)
        self.put_down_turtle(self.size, self.size)
        self.put_down_turtle(0, self.size)
        self.put_down_turtle(0, 0)
        self.t.penup()
        self.t.color("black")
        self.put_down_turtle(self.Circuit.pos0[0], self.Circuit.pos0[1])
        self.t.pendown()

    def put_down_turtle(self, x, y):
        self.t.setpos((500/self.size)*x - 250, (500/self.size)*y - 250)

    def reset(self):
        if self.displayed:
            self.t.clear()
        self.__init__(
            self.displayed,
            self.Δd,
            self.Δt,
            self.dmax,
            self.size,
            self.turning_circle,
            self.engine_quality
        )


def rotation(vecteur2D, angle): # angle in radians
    matrice_rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    return(np.matmul(matrice_rotation, vecteur2D))


def main():
    P = Car(False)
    TB = TestBench(
        P,
        slices=[5, 4],
        regions=[
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False,False]
        ])
    TB.test(0)

if __name__ == "__main__":
    main()

