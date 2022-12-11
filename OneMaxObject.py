"""
06, Nov, 2022
Code by Tianyu Gong
"""

import random


"""
This script implements a useful BitString object, it defines a way to represent bitstring, and provides some methods
to manipulate bitstrings, especially some methods related to Evolutionary Algorithm, like mutate and crossover.

A test function is added at the end, you can read and run it to know how to operate on BitString instances.
"""

class BitString:
    # This class variable is used to count how many fitness evaluations has been done
    # increase this value using self.__class__.n_fitness_evals += 1 when the bitstring is initialized or changed
    # If the solver is executed multiple times, you need to zero this value manually
    n_fitness_evals = 0
    def __init__(self, length):
        """
        This class represent a bitstring, you can find many useful methods here
        :param length: the length of the bitstring
        """
        self.__length = length
        self.bits = []
        for i in range(length):
            self.bits.append(random.randint(0, 1))
        self.__class__.n_fitness_evals += 1

    @property
    def fitness(self):
        """ Count how many 1s in the bitstring """
        return self.bits.count(1)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "".join([str(bit) for bit in self.bits])

    def isAllOnes(self):
        return self.fitness == self.__length

    def copy(self):
        new_bitstring = BitString(self.__length)
        new_bitstring.bits = self.bits[:]
        return new_bitstring

    def revertBit(self, idx):
        self.bits[idx] = 1 - self.bits[idx]

    def mutate(self, n_bits):
        """
        Mutate the bitstring, by randomly revert some bits (1 to 0 or 0 to 1)
        :param n_bits: how many bits to mutate
        mutated twice or more. Note that mutating a bit even times with make no change to it. default: True.
        """
        assert n_bits <= self.__length, \
            "Cannot have number of mutation bits > total length and with replacement at the same time."
        indices = random.sample(list(range(self.__length)), k=n_bits)
        for i in indices:
            self.revertBit(i)
        # accumulate number of fitness evaluations
        self.__class__.n_fitness_evals += 1

    def probabilisticMutation(self, rate):
        """
        Mutate the bitstring, revert each bit with certain probability
        :param rate: the probability of revert a bit
        """
        for i in range(self.__length):
            if random.random() < rate:
                self.revertBit(i)
        self.__class__.n_fitness_evals += 1

    @staticmethod
    def singlePointCrossover(bs1, bs2, point):
        """
        Do single point crossover
        :param bs1: first bitstring
        :param bs2: second bitstring
        :param point: the index of crossover, the resulting element at the point will come from bs2
        :return: a new bitstring, the result of crossover
        """
        new_bitstring = bs1.copy()
        new_bitstring.bits[point:] = bs2.bits[point:]
        return new_bitstring

    @staticmethod
    def randomMaskCrossover(bs1, bs2):
        """
        Do random mask crossover
        :param bs1: bitstring 1
        :param bs2: bitstring 2
        :return: a new bitstring, the result of crossover
        """
        new_bitstring = bs1.copy()
        for bi in range(len(bs1.bits)):
            if random.random() < 0.5:
                new_bitstring.bits[bi] = bs2.bits[bi]
        return new_bitstring

    @staticmethod
    def zeroString(length):
        """
        Return a bitstring containing all 0
        :param length: bitstring length
        :return: a bitstring containing all 0
        """
        bit_string = BitString(length)
        bit_string.bits = [0] * length
        return bit_string

    @staticmethod
    def oneString(length):
        """
        Return a bitstring containing all 1
        :param length: bitstring length
        :return: a bitstring containing all 1
        """
        bit_string = BitString(length)
        bit_string.bits = [1] * length
        return bit_string


class Population(list):
    def __init__(self, size, bitstring_length):
        """
        Population class, it is actually a list object, but with some additional methods like getBest and
        tournamentSelect, you can use this object like how you use a list.
        :param size: population size
        :param bitstring_length: the length of all bitstrings
        """
        super(Population, self).__init__()
        # Popularize myself, so when you initialize an instance of this object, what you will get is a
        # normal python list, but filled with values, and have some additional methods
        self.size = size
        for _ in range(self.size):
            self.append(BitString(bitstring_length))

    def getBest(self):
        """ Get the best bitstring in the population """
        # def func(x):
        #       return 2 * x
        # func = lambda x : 2*x
        # a = func(3)
        return max(self, key=lambda bs: bs.fitness)

    def getWorst(self):
        """ Get the worst bitstring in the population """
        return min(self, key=lambda bs: bs.fitness)

    def getArgWorst(self):
        # [0, 1, 2, 3]
        # [8, 7, 10, 15]
        return min(range(self.size), key=lambda i: self[i].fitness)

    def getAvgFitness(self):
        """ Get the average fitness of the entire population """
        fitnesses = [bitstring.fitness for bitstring in self]
        return sum(fitnesses) / self.size

    def tournamentSelect(self, tournament_size, n_select):
        """
        Randomly sample tournament_size elements in population and choose the best among them, repeat n_select times
        :param tournament_size: how many to sample from the population
        :param n_select: how many elements selected finally
        :return: a list of elements selected from tournament
        """
        if tournament_size > len(self):
            tournament_size = len(self)
        selected = []
        for _ in range(n_select):
            group = random.sample(self, tournament_size)
            best_in_group = max(group, key=lambda bs: bs.fitness)
            selected.append(best_in_group)
        return selected

    def __str__(self):
        """ rewrite __str__ for better printing behavior """
        return " ".join([str(bitstring) for bitstring in self])


def testBitString():
    print()
    # initialize
    s = BitString(8)
    print("BitString(8) ->", s)
    # initialize all 0 bitstring
    s_0 = BitString.zeroString(8)
    print("s_0 =", s_0)  # 00000000
    # initialize all 1 bitstring
    s_1 = BitString.oneString(8)
    print("s_1 =", s_1)  # 11111111
    # other methods
    s_0.mutate(2)
    print("s_0.mutate(2), s_0 =", s_0)
    s_1.revertBit(0)
    print("s_1.revertBit(0), s_1 =", s_1)
    s_2 = BitString.singlePointCrossover(s_0, s_1, 3)
    print("s_2 =", s_2)
    s_3 = BitString.randomMaskCrossover(s_0, s_1)
    print("s_3 =", s_3)
    print("s_3.fitness =", s_3.fitness)
    print("s_3.copy() =", s_3.copy())
    print("s_3.isAllOnes()", s_3.isAllOnes())


def testPopulation():
    print()
    p = Population(3, 6)
    print(p)
    print(p.getBest())
    print(p.getWorst())
    print(p.getArgWorst())
    print(p.getAvgFitness())
    print(p.tournamentSelect(2, 2))

