#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 06/11/2020
The aim of this program is to create a real time GAN
"""

from AI import *


class Gan(Problem):
    """
    A GAN which simulates simultanious evolution
    useful for generating cute pics of cats
    but the fact that the Networks play continuously in time could be useful
    to learn how to walk : One network makes a terrain hard to walk on (or even
    throws bullets on the walker), the other tries to walk on and so on
    """
    def __init__(
        self,
        nb_sensors_generator,
        nb_actors_generator,
        nb_sensors_discriminator,
        nb_actors_discriminator,
        discriminator, # is a Problem (written d some times)
        d_data,
        d_add_neurons = 5,
        d_period = 5,
        d_herd_size = 100,
        d_mutation_coefficent = 0.1,
        d_mutation_amplitude = 0.001,
        d_nb_tests = 1,
        d_herd = None
        ):
        print("Warning  : __init__ was not fully configured")
        self.nb_sensors_generator = nb_sensors_generator
        self.nb_actors_generator= nb_actors_generator
        self.nb_sensors_discriminator = nb_sensors_discriminator
        self.nb_actors_discriminator = nb_actors_discriminator
        self.d_data = d_data
        self.discriminator = Discriminator(d_data)
        if d_herd == None:
            self.d_Herd = Herd(
                nb_sensors = discriminator.nb_sensors,
                nb_actors = discriminator.nb_actors,
                nb_add_neurons = d_add_neuronsd,
                period = d_period,
                size = d_herd_size,
                mutation_coefficent = d_mutation_coefficent,
                mutation_amplitude = d_mutation_amplitude,
                nb_tests = d_nb_tests
            )
        else:
            self.d_Herd = d_herd

    def experience(self, Generator):
        """
        Computes the actions of some networks on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        while not self.end_condition():
            generation = Generator.process(self.state())
            self.action(*generation)
            self.discriminator.generated.append(generation)
            self.d_Herd.evolve(self.discriminator, 1)
        score = self.score_real_time()
        self.reset()
        # score should always be > 0
        return(score*(score>0))

    def end_condition(self):
        """
        True if the Problem is finished for whatever reason
        False if it goes on
        Typically we want the discriminator to hit at random at this point
        """
        print("Warning  : end_condition was not fully configured")
        return(True)

    def state(self):
        """
        returns noise
        """
        print("Warning  : state was not fully configured")
        return(np.random.rand((1)))

    # Other state related functions should be there

    def score_real_time(self):
        """
        Returns the score of the problem at the moment
        """
        print("Warning score_real_time was not fully configured")
        return(0)

    def action(self, Generator_data):
        """
        Computes the consequences of the input_data on the problem
        Here we just record what was generated
        """
        self.generation = Generator_data

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        print("Warning : experience was not fully configured")

    # Other display functions should be put here

    def reset(self):
        """
        Resets the problem
        """
        self.__init__(
            self.nb_sensors_generator,
            self.nb_actors_generator,
            self.nb_sensors_discriminator,
            self.nb_actors_discriminator,
            self.discriminator,
            self.d_data,
            d_herd = self.d_Herd
        )

class Discriminator(Problem):
    """
    The frame of any discriminator
    """
    def __init__(self, data):
        self.nb_sensors = 1
        self.nb_actors = 1
        print("Warning  : __init__ was not fully configured")

    def experience(self, Network):
        """
        Computes the actions of a network on the problem
        This is the main function of a problem
        """
        print("Warning  : experience was not fully configured")
        while not self.end_condition():
            self.action(*Network.process(self.state()))
        score = self.score_real_time()
        self.reset()
        # score should always be > 0
        return(score*(score>0))

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

    def score_real_time(self):
        """
        Returns the score of the problem at the moment
        """
        print("Warning score_real_time was not fully configured")
        return(0)

    def action(self, input_data):
        """
        Computes the consequences of the input_data on the problem
        """
        print("Warning  : action was not fully configured")
        pass

    # Other action related functions should be put here

    def display(self):
        """
        Shows how things are doing
        """
        print("Warning : experience was not fully configured")

    # Other display functions should be put here

    def reset(self):
        """
        Resets the problem
        """
        print("Maybe some HUGE PROBLEM is coming at you bro")
        self.__init__()

