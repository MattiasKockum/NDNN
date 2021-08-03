#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On 15/07/2020
The aim of this program is to train a car to drive itself
"""

import numpy as np
import turtle
from AI import *


def rotation(Vector2D, angle): # angle in radians
    matrice_rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
    return(np.matmul(matrice_rotation, Vector2D))


class Circuit():
    """
    A circuit for cars
    """
    def __init__(self, size=8):
        self.size = size
        self.pos0 = np.array([0.5, 0.5])
        self.dir0 = np.array([1.0, 0.0])
        self.road = -np.ones((self.size, self.size))
        self.path = []
        self.road_generation()
        self.path_len = self.road[self.pos_final]

    def road_generation(self):
        self.make_circuit()
        self.make_points()

    def make_circuit(self):
        """
        Makes an empty circuit
        """
        dead_end = []
        possibilites = [0, 3] # To be updated if the starting point changes
        i = 0
        x = 0
        y = 0
        while not self.end_construction_condition(i, x, y):
            direction = np.random.choice(possibilites)
            possibilites.remove(direction)
            if (
                direction == 0
                and y + 1 < self.size
                and (x, y + 1) not in self.path
                and (x, y + 1) not in dead_end
            ):
                self.path.append((x, y))
                possibilites = [1, 2, 3]
                i += 1
                y += 1
            elif (
                direction == 1
                and x - 1 > 0
                and (x - 1, y) not in self.path
                and (x - 1, y) not in dead_end
            ):
                self.path.append((x, y))
                possibilites = [0, 2, 3]
                i += 1
                x -= 1
            elif (
                direction == 2
                and y - 1 > 0
                and (x, y - 1) not in self.path
                and (x, y - 1) not in dead_end
            ):
                self.path.append((x, y))
                possibilites = [0, 1, 3]
                i += 1
                y -= 1
            elif (
                direction == 3
                and x + 1 < self.size
                and (x + 1, y) not in self.path
                and (x + 1, y) not in dead_end
            ):
                self.path.append((x, y))
                possibilites = [0, 1, 2]
                i += 1
                x += 1
            if possibilites == []:
                # Have self.path all the neighbours
                dead_end.append((x, y))
                self.path = self.path[:-1]
                x, y = self.path[-1]
                possibilites = [0, 1, 2, 3]
        self.pos_final = (x, y)
        self.path.append((x, y))

    def end_construction_condition(self, i, x, y):
        return(
            i >= self.size
            and (
                x == 0
                or x == self.size - 1
                or y == 0
                or y == self.size - 1)
              )

    def make_points(self):
        """
        Puts the points on the circuit
        """
        still_to_explore = [(0, 0)]
        self.road[0][0] = 0
        while still_to_explore != []:
            x, y = still_to_explore[-1]
            still_to_explore = still_to_explore[:-1]
            actual_points = self.road[x][y]
            if (
                x + 1 < self.size
                and (x + 1, y) in self.path
                and not (-1 < self.road[x + 1][y] < actual_points)
            ):
                still_to_explore.append((x + 1, y))
                self.road[x + 1][y] = actual_points + 1
            if (
                y + 1 < self.size
                and (x, y + 1) in self.path
                and not (-1 < self.road[x][y + 1] < actual_points)
            ):
                still_to_explore.append((x, y + 1))
                self.road[x][y + 1] = actual_points + 1
            if (
                x - 1 < self.size
                and (x - 1, y) in self.path
                and not (-1 < self.road[x - 1][y] < actual_points)
            ):
                still_to_explore.append((x - 1, y))
                self.road[x - 1][y] = actual_points + 1
            if (
                y - 1 < self.size
                and (x, y - 1) in self.path
                and not (-1 < self.road[x][y - 1] < actual_points)
            ):
                still_to_explore.append((x, y - 1))
                self.road[x][y - 1] = actual_points + 1

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
        size = 8,
        Δd = 0.01, # The length precision of the physics engine
        Δt = 0.01, # The time precision of the physics engine
        dmax = 4, # The distance at which the car can see
        turning_circle = 20, # quality of the car
        engine_quality = 20,
        circuit = None,
    ):
        # Circuit code
        if circuit == None:
            self.Circuit = Circuit(size)
            self.circuit_pre_defined = False
        else:
            self.Circuit = circuit
            self.circuit_pre_defined = True
        self.size = self.Circuit.size
        self.path_len = self.Circuit.path_len
        self.pos = self.Circuit.pos0
        self.dir = self.Circuit.dir0
        # Physics
        self.speed = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.Δd = Δd
        self.Δt = Δt
        # Car
        self.dmax = dmax
        self.turning_circle = turning_circle
        self.engine_quality = engine_quality
        # Misc
        self.displayed = displayed
        self.score = 0
        # An impossible position in order to get into the loop
        self.previous_pos = np.array([-1, -1])
        # Network info
        self.nb_sensors = 9 + len(self.captors())
        self.nb_actors = 2

    def experience(self, Network):
        if self.displayed:
            self.display()
        while not self.end_condition():
            self.action(*Network.process(self.state()))
        score = self.score
        self.reset()
        return(score)

    def end_condition(self):
        return(
            self.state_pos(self.pos) == self.path_len
            or np.array_equal(self.previous_pos, self.pos)
        )

    def state(self):
        return(
            np.array([
                *self.pos,
                *self.dir,
                *self.acceleration,
                *self.captors(),
                *self.next_pos(),
                self.score
            ]
        ))

    def captors(self):
        return(
            np.array([
                self.ray(rotation(self.dir, np.pi/3), self.dmax),
                self.ray(rotation(self.dir, -np.pi/3), self.dmax),
                self.ray(self.dir, self.dmax)
            ])
        )

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

    def next_pos(self):
        if self.score + 1 < len(self.Circuit.path):
            return(self.Circuit.path[int(self.score) + 1])
        return((-1, -1))

    def score_update(self):
        """
        Returns how far the car has gone, even if it turned back
        """
        score_here = self.state_pos(self.pos)/self.path_len
        if score_here > self.score:
            self.score = score_here

    def action(self, pedale, volant):
        self.acceleration += self.engine_quality*pedale*self.dir
        self.dir = rotation(self.dir, self.turning_circle*volant)
        # The position at which the car should arive without any obstacle
        pos_projection = (
            (1/2)*self.acceleration*self.Δt**2
            + self.speed*self.Δt
            + self.pos
        )
        # The vector the car is following at the moment
        direction_projection = self.pos - pos_projection
        distance_projection = np.linalg.norm(direction_projection)
        drived_distance = self.ray(
            direction_projection,
            distance_projection
        )
        self.previous_pos = self.pos
        if drived_distance != distance_projection:
            # Hit an obstacle !
            self.speed = np.array([0.0, 0.0])
            norm_dir = (
                direction_projection
                /np.linalg.norm(direction_projection)
            )
            self.pos += norm_dir*drived_distance
        else :
            # No obstacle encountered
            self.speed += self.acceleration*self.Δt
            self.pos = pos_projection
        if self.displayed:
            self.put_down_turtle(self.pos[0], self.pos[1])
        self.score_update()

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

    def __name__(self):
        return("Car")

    def reset(self):
        if self.displayed:
            self.t.clear()
        if self.circuit_pre_defined:
            circuit = self.Circuit
        else:
            circuit = None
        self.__init__(
            self.displayed,
            self.size,
            self.Δd,
            self.Δt,
            self.dmax,
            self.turning_circle,
            self.engine_quality,
            circuit = circuit
        )


def main():
    P = Car(False, 4)
    TB = TestBench(
        P,
        1, # nb_herds
        30, # nb_generations
        9, # nb_add_neurons
        3, # period
        500, # size
        0.02, # mutation_coefficient
        0.001, # mutation_amplitude
        10, # nb_tests
        slices=[P.nb_sensors, 5, 4, P.nb_actors],
        regions=[
            [False, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False]
        ]
    )
    TB.test(5)

if __name__ == "__main__":
    main()

