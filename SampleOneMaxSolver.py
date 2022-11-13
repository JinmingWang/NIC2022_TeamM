from OneMaxObject import BitString
from typing import *

"""
This is just a sample solver, it is not good for Worst OneMax Problem
Because it finds the solution too quickly
This just shows a very simple EA to solve the problem
"""
class GreedyMutationSolver:
    def __init__(self, bitstring_len: int, population_size: int, n_mutation: int, n_iter: int):
        """
        A very stupid, simple solver, does poorly on Worst OneMax Problem
        :param bitstring_len: how is the length for bitstring
        :param population_size: population size
        :param n_mutation: how many to mutate each bitstring
        :param n_iter: number of total iterations
        """
        self.population = [BitString(bitstring_len) for _ in range(population_size)]
        self.n_mutation = n_mutation
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len

    def run(self):
        """ Run the EA """
        best_answer_found_at = -1
        best_solution = BitString.zeroString(self.bitstring_len)

        for i in range(self.n_iter):
            # Find the best bitstring at this time
            best_bitstring = max(self.population, key=lambda bs: bs.fitness)
            print(f"Best fitness = {best_bitstring.fitness}")
            if best_bitstring.fitness > best_solution.fitness:
                best_solution = best_bitstring.copy()
                best_answer_found_at = i

            # Just take the last 10 bitstring in population, and use the best bitstring to replace them
            for i in range(10):
                self.population.pop()
                self.population.insert(0, best_bitstring.copy())

            # If a
            for bs in self.population:
                if not bs.isAllOnes():
                    bs.mutate(self.n_mutation)

        best_bitstring = max(self.population, key=lambda bs: bs.fitness)
        print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
              f"found at iteration = {best_answer_found_at}")




if __name__ == '__main__':
    solver = GreedyMutationSolver(20, 20, 2, 1000)
    solver.run()



