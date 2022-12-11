import random

from OneMaxObject import BitString, Population
from typing import *
import matplotlib.pyplot as plt

SIZE = 15  # size of the problem, the length of bitstring
N_GENERATIONS = 1100  # number of generations

"""
This solver can take any function as the mutation decay function, the function takes generation number and returns
mutation rate
"""


class PowerfulSolver:
    def __init__(self, bitstring_len: int,
                 population_size: int,
                 mutation_rate_func: Callable[[int], float],
                 t_size: int,
                 t_n_select: int,
                 crossover_rate_func: Callable[[int], float],
                 n_iter: int):
        """
        A very stupid, simple solver, does poorly on Worst OneMax Problem
        :param bitstring_len: how is the length for bitstring
        :param population_size: population size
        :param mutation_rate_func: a function that given generation number and returns mutation rate
        :param n_iter: number of total iterations
        """
        self.population = Population(population_size, bitstring_len)
        self.mutation_func = mutation_rate_func
        self.t_size = t_size
        self.t_n_select = t_n_select
        self.crossover_rate_func = crossover_rate_func
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len
        self.best_fitness_list = []

    def run(self, verbose: bool = False) -> Tuple[int, int]:
        """ Run the EA """
        best_answer_found_at = -1
        best_fitness_so_far = 0

        for i in range(self.n_iter):
            # Find the best bitstring at this time
            parents = self.population.tournamentSelect(self.t_size, self.t_n_select)
            children = []
            # update crossover rate
            crossover_rate = self.crossover_rate_func(i)
            # do crossover and produce children
            for pi in range(len(parents)):
                for pj in range(pi + 1, len(parents)):
                    if random.random() < crossover_rate:
                        children.append(BitString.randomMaskCrossover(parents[pi], parents[pj]))
                    else:
                        children.append(parents[pi].copy())

            # Add best bitstring in population to children
            best_bitstring = self.population.getBest()
            children.append(best_bitstring.copy())
            self.best_fitness_list.append(best_bitstring.fitness)

            # mutate each child then add all children to population
            n_children = len(children)
            mutation_rate = self.mutation_func(i)
            for child in children:
                child.probabilisticMutation(mutation_rate)
            self.population.extend(children)

            # Remove the worst n_children items
            for _ in range(n_children):
                self.population.pop(self.population.getArgWorst())

            best_bitstring = self.population.getBest()
            if best_bitstring.fitness > best_fitness_so_far:
                best_fitness_so_far = best_bitstring.fitness
                best_answer_found_at = i

        if verbose:
            best_bitstring = self.population.getBest()
            print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
                  f"found at iteration = {best_answer_found_at}")

        return best_fitness_so_far, best_answer_found_at


def experiment(bitstring_len: int,
               population_size: int,
               mutation_rate_func: Callable[[int], float],
               t_size: int,
               t_n_select: int,
               crossover_rate_func: Callable[[int], float],
               n_iter: int):
    n_solves = 0
    average_solved_at = 0
    avg_fitness_evals = 0
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        print(f"{i + 1}/{n_tests}")
        solver = PowerfulSolver(bitstring_len, population_size, mutation_rate_func, t_size, t_n_select, crossover_rate_func, n_iter)
        best_fitness, best_found_at = solver.run()
        EA_records = [solver.best_fitness_list[i] + EA_records[i] for i in range(N_GENERATIONS)]
        if best_fitness == SIZE:
            n_solves += 1
            average_solved_at += best_found_at
        avg_fitness_evals += BitString.n_fitness_evals

    avg_fitness_evals /= n_tests
    average_solved_at /= n_solves
    EA_records = [rec / n_tests for rec in EA_records]
    print(f"OneMax problem solved {n_solves} out of {n_tests} runs, \n"
          f"Problem is solved with {average_solved_at} iterations in average, \n"
          f"Average number of fitness evaluations is {avg_fitness_evals}.\n")
    plt.plot(EA_records)
    plt.title("The performance curve of PowerfulSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Best fitness of {n_tests} runs")
    plt.show()


if __name__ == '__main__':
    pop_size = 10
    init_m = 1.8127543060530658
    slope_m = -0.0016558461699318148
    t_size = 5
    t_n_select = 2
    init_c = 2.6194913922128302
    slope_c = -0.00039274961188433814
    min_mutation = 0.049647885507263775


    def crossover_rate_update_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)


    def mutation_rate_update_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)


    experiment(SIZE,
               pop_size,
               mutation_rate_update_func,
               t_size, t_n_select,
               crossover_rate_update_func,
               N_GENERATIONS)
