#!/usr/bin/env python3

"""
Program written by Mattias Kockum
On the 15/7/2020
The aim of this program is to create an AI
    capable of selective memory
    capable of solving real time problems fast
    capable of simulating a Turing Machine
The training is parallelized
"""

# Necessary
from AI import *
import multiprocessing as mp


# parallelization functions

def extend(array, n=1):
    r = []
    for i in array:
        for j in range(n):
            r.append(copy.deepcopy(i))
    return(r)

def mean(array, n=1):
    r = []
    array += [0]*(-len(array)%n)
    for i in range(0, len(array), n):
        r.append(sum(array[i:i+n])/n)
    return(r)

def prob_reproduction(X):
    """
    A weird looking function for parallelization
    X[0] is a group of objects
    X[1] is their respective probability of being copied
    X[2] = mutation_coefficent
    X[3] = mutation_amplitude
    returns the mutation of the chosen one
    """
    return(np.random.choice(X[0], p=X[1]).mutate(X[2], X[3]))

def evaluate(X):
    """
    Another weird looking function
    X[0] is a problem
    X[1] is a network
    returns the score of the network
    """
    np.random.seed()
    X[0].reset()
    X[1].reset()
    return_value = X[0].experience(X[1])
    return(return_value)

def pooled_evolution(X):
    """
    Another one
    """
    r = X[0].evolve(X[1], X[2])
    return(r, (X[0].members[0], r))


class CPU_Herd(Herd):
    """
    Herd of networks that evolve by reproducing
    """
    def __init__(
        self,
        nb_sensors = 1,
        nb_actors = 1,
        nb_add_neurons = 0,
        period = 1,
        function = segments,
        reset_after_process = True,
        size = 5,
        mutation_coefficent = 0.1,
        mutation_amplitude = 0.001,
        nb_tests = 1,
        do_display = False,
        **kwargs
    ):
        self.nb_sensors = nb_sensors
        self.nb_actors = nb_actors
        self.nb_add_neurons = nb_add_neurons
        self.period = period
        self.size = size
        self.mutation_coefficent = mutation_coefficent
        self.mutation_amplitude = mutation_amplitude
        self.nb_tests = nb_tests
        self.do_display = do_display
        self.members = [
            Network(nb_sensors, nb_actors, nb_add_neurons, period, function,
                    reset_after_process, **kwargs)
            for i in range(size)
        ]
        self.array_scores = []
        self.date = date()
        self.max_score = 0
        self.max_score_index = 0

    def evolve(self, problem, nb_generations=1):
        """
        The idea is to make the AI evolve by aproximating the gradient descent
        Opens and closes the score output file multiple times so that it's
        possible to see what's going on in during the training
        """
        if problem == None:
            # The empty problem, just here for quick tests
            self.Problem = Problem()
        else:
            self.Problem = problem
        score_file = open(problem.__name__() + "_score" + self.date, "w")
        score_file.write(
            "score\n"
            + "number of added neurons : {}\n".format(self.nb_add_neurons)
            + "period : {}\n".format(self.period)
            + "size : {}\n".format(self.size)
            + "mutation coefficent : {}\n".format(self.mutation_coefficent)
            + "mutation_amplitude : {}\n".format(self.mutation_amplitude)
            + "number of tests : {}\n".format(self.nb_tests)
            + "number of generations to proceed : {}\n".format(nb_generations)
        )
        score_file.close()
        self.Problem_pool = extend([self.Problem], self.size*self.nb_tests)
        for pb in self.Problem_pool:
            pb.do_display = False
        if self.do_display:
            self.Problem_pool[0].do_display = True
        for generation in range(nb_generations):
            # Evaluation of performances
            proba_reproduction = self.performances()
            # Reproduction (with mutation) of Networks
            self.reproduce(proba_reproduction)
            # Saves the scores
            self.max_score = max(self.score)
            self.max_score_index = self.score.index(self.max_score)
            self.array_scores.append(self.max_score)
            # Saves one Network and the score evolution
            self.members[self.max_score_index].save(
                problem.__name__() + "_Network" + self.date, "w", False)
            score_file = open(problem.__name__() + "_score" + self.date, "a")
            score_file.write(
                "generation n° {} : {} \n".format(
                    generation, str(self.max_score)))
            score_file.close()
        score_file = open(problem.__name__() + "_score" + self.date, "a")
        score_file.write("End\n")
        score_file.close()
        return(self.array_scores)

    def performances(self):
        """
        Evaluates performances then normalises them for probability operations
        Can be parallelized
        """
        self.members_pool = extend(self.members, self.nb_tests)
        # parallelize the evaluation of the networks
        pool = mp.Pool()
        member_s_points = pool.map(
            evaluate,
            [(P, M) for P,M in zip(self.Problem_pool, self.members_pool)]
        )
        pool.close()
        # Put this code if you want to observe evolution, especially in the
        # Gradient Descent Problem because parallelization makes it not work
        #self.Problem.experience(self.members_pool[0])
        #self.members_pool[0].reset()
        self.score = mean(member_s_points, self.nb_tests)
        score_modif = self.modif_score(self.score)
        return(score_modif)

    def reproduce(self, proba_reproduction):
        """
        The copy of the successful networks with mutation
        parallelized
        """
        pool = mp.Pool()
        new_members = (
            pool.map(
                prob_reproduction,
                [(
                    self.members,
                    proba_reproduction,
                    self.mutation_coefficent,
                    self.mutation_amplitude
                )]*self.size
            )
        )
        self.members = new_members
        pool.close()

