from __future__ import annotations  # Requires Python 3.7+
import random
from typing import *


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
    def __init__(self, length: int):
        """
        This class represent a bitstring, you can find many useful methods here
        :param length: the length of the bitstring
        """
        self.__length = length
        self.bits = [random.randint(0, 1) for _ in range(length)]
        self.__class__.n_fitness_evals += 1

    @property
    def fitness(self) -> int:
        """ Count how many 1s in the bitstring """
        return self.bits.count(1)

    def __len__(self):
        """
        s = BitString(15)
        len(s) -> returns 15
        """
        return self.__length

    def __getitem__(self, idx):
        """
        s = BitString(15)
        s[0] -> the first bit of this bitstring
        """
        if type(idx) == slice:
            sublist = self.bits[idx]
            new_string = BitString(len(sublist))
            new_string.bits = sublist
            return new_string
        return self.bits[idx]

    def __setitem__(self, key, value) -> None:
        """
        s = BitString(10)
        s[1] = True -> sets the 2nd element to 1
        s[-1] = 0 -> sets the last element to 0
        """
        if type(key) == slice:
            start = 0 if key.start is None else key.start
            step = 1 if key.step is None else key.step
            stop = len(self.bits) if key.stop is None else key.stop
            for i, k in enumerate(range(start, stop, step)):
                self.bits[k] = 1 if value[i] in [1, "1", True] else 0
        else:
            self.bits[key] = 1 if value in [1, "1", True] else 0

    def __and__(self, other: BitString):
        """ bitwise and """
        new_string = BitString(self.__length)
        new_string.bits = list(map(lambda b1, b2: b1 & b2, self, other))
        return new_string

    def __or__(self, other):
        """ bitwise or """
        new_string = BitString(self.__length)
        new_string.bits = list(map(lambda b1, b2: b1 | b2, self, other))
        return new_string

    def __str__(self):
        return "".join([str(bit) for bit in self.bits])

    def __repr__(self):
        return self.__str__()

    def isAllOnes(self) -> bool:
        """ to check whether the bit string contains only bit 1 """
        return self.fitness == self.__length

    def copy(self) -> BitString:
        """ This is implemented to prevent multiple variables point to the same memory address """
        new_bitstring = BitString(self.__length)
        new_bitstring.bits = self.bits[:]
        return new_bitstring

    def revertBit(self, idx: int) -> None:
        """ make a bit at idx revert, if it is 1 then make it 0, if it is 0 then make it 1 """
        self.bits[idx] = 1 - self.bits[idx]

    def mutate(self, n_bits: int, with_replacement: bool = True) -> None:
        """
        Mutate the bitstring, by randomly revert some bits (1 to 0 or 0 to 1)
        :param n_bits: how many bits to mutate
        :param with_replacement: if set to True, the mutated bits won't be mutated again, otherwise, one bit can be
        mutated twice or more. Note that mutating a bit even times with make no change to it. default: True.
        """

        if with_replacement:
            assert n_bits <= self.__length, \
                "Cannot have number of mutation bits > total length and with replacement at the same time."
            indices = random.sample(list(range(self.__length)), k=n_bits)
        else:
            indices = random.choices(list(range(self.__length)), k=n_bits)

        for i in indices:
            self.revertBit(i)
        self.__class__.n_fitness_evals += 1

    def probabilisticMutation(self, rate: float) -> None:
        """
        Mutate the bitstring, revert each bit with certain probability
        :param rate: the probability of revert a bit
        """
        for i in range(self.__len__()):
            if random.random() < rate:
                self.revertBit(i)
        self.__class__.n_fitness_evals += 1

    @staticmethod
    def singlePointCrossover(bs1: BitString, bs2: BitString, point: int) -> BitString:
        """
        Do single point crossover
        :param bs1: first bitstring
        :param bs2: second bitstring
        :param point: the index of crossover, the resulting element at the point will come from bs2
        :return: a new bitstring, the result of crossover
        """
        assert len(bs1) == len(bs2), "two bitstrings must have equal length"
        new_bitstring = BitString(len(bs1))
        new_bitstring.bits = bs1.bits[:point] + bs2.bits[point:]
        return new_bitstring

    @staticmethod
    def randomMaskCrossover(bs1: BitString, bs2: BitString) -> BitString:
        """
        Do random mask crossover
        :param bs1: bitstring 1
        :param bs2: bitstring 2
        :return: a new bitstring, the result of crossover
        """
        assert len(bs1) == len(bs2), "two bitstrings must have equal length"
        new_bitstring = BitString(len(bs1))
        for bi in range(len(bs1)):
            new_bitstring.bits[bi] = bs1.bits[bi] if random.random() < 0.5 else bs2.bits[bi]
        return new_bitstring

    @staticmethod
    def zeroString(length: int) -> BitString:
        """
        Return a bitstring containing all 0
        :param length: bitstring length
        :return: a bitstring containing all 0
        """
        bit_string = BitString(length)
        bit_string.bits = [0] * length
        return bit_string

    @staticmethod
    def oneString(length: int) -> BitString:
        """
        Return a bitstring containing all 1
        :param length: bitstring length
        :return: a bitstring containing all 1
        """
        bit_string = BitString(length)
        bit_string.bits = [1] * length
        return bit_string


class Population(list):
    def __init__(self, size: int, bitstring_length: int):
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

    def getBest(self) -> BitString:
        """ Get the best bitstring in the population """
        return max(self, key=lambda bs: bs.fitness)

    def getAvgFitness(self) -> float:
        """ Get the average fitness of the entire population """
        return sum([bitstring.fitness for bitstring in self]) / self.size

    def tournamentSelect(self, tournament_size: int, n_select: int) -> List[BitString]:
        """
        Randomly sample tournament_size elements in population and choose the best among them, repeat n_sleecct times
        :param tournament_size: how many to sample from the population
        :param n_select: how many elements selected finally
        :return: a list of elements selected from tournament
        """
        selected = []
        for _ in range(n_select):
            group = random.sample(self, tournament_size)
            selected.append(max(group, key=lambda bs: bs.fitness))
        return selected

    def __repr__(self):
        """ rewrite __repr__ for better debugging experience """
        return f"Population of size {self.size} and bitstring length {len(self[0])}"

    def __str__(self):
        """ rewrite __str__ for better printing behavior """
        return " ".join([str(bitstring) for bitstring in self])

def testBitString():
    # initialize
    s = BitString(8)
    print("BitString(8) ->", s)

    # get bitstring length
    print("len(s) =", len(s))   # 8

    # initialize all 0 bitstring
    s_0 = BitString.zeroString(8)
    print("s_0 =", s_0)  # 00000000

    # initialize all 1 bitstring
    s_1 = BitString.oneString(8)
    print("s_1 =", s_1)  # 11111111

    # set a bit
    s[:] = '01100001'
    print("s[:] = '01100001', s= ", s)    # 01100001
    s[0] = 1
    print("s[0] = 1, s=", s)    # 11100001
    s[3:] = '10000'
    print("s[3:] = '10000', s=", s)    # 11110000
    s[0:7:2] = '0011'
    print("s[0:7:2] = '0011', s=", s)    # 01011010

    # get a bit
    print("s[2] =", s[2])     # (0)
    print("s[-1] =", s[-1])    # 0
    print("s[:-4] =", s[:-4])   # 1010

    # other methods
    s_0.mutate(2)
    print("s_0.mutate(2), s_0 =", s_0)
    s_1.mutate(4, with_replacement=False)
    print("s_1.mutate(4, with_replacement=False), s_1 =", s_1)
    s_1.revertBit(0)
    print("s_1.revertBit(0), s_1 =", s_1)
    s_2 = BitString.singlePointCrossover(s_0, s_1, 3)
    print("s_2 =", s_2)
    s_3 = BitString.randomMaskCrossover(s_0, s_1)
    print("s_3 =", s_3)
    print("s_3.fitness =", s_3.fitness)
    print("s_3.copy() =", s_3.copy())
    print("s_3.isAllOnes()", s_3.isAllOnes())

if __name__ == '__main__':
    # testBitString()
    p = Population(10, 5)
    print(p)
    p.pop()
    print(p)
