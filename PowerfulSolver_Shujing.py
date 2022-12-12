import random

from OneMaxObject import BitString, Population
from typing import *
import matplotlib.pyplot as plt

SIZE = 15  # size of the problem, the length of bitstring
N_GENERATIONS = 1000  # number of generations

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


def experiment(bitstring_len, population_size, mutation_rate_func, t_size, t_n_select, crossover_rate_func, n_iter, plot_and_print=True):
    n_solves = 0
    average_solved_at = 0
    avg_fitness_evals = 0
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        if plot_and_print:
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
    if plot_and_print:
        print(f"OneMax problem solved {n_solves} out of {n_tests} runs, \n"
              f"Problem is solved with {average_solved_at} iterations in average, \n"
              f"Average number of fitness evaluations is {avg_fitness_evals}.\n")
        plt.plot(EA_records)
        plt.title("The performance curve of PowerfulSolver")
        plt.xlabel("Iteration")
        plt.ylabel(f"Average Best fitness of {n_tests} runs")
        plt.show()
    return EA_records, average_solved_at, avg_fitness_evals, n_solves


def parameterFind2():
    """
    It is possible that we can use slope_m for slope_c and also init_m for init_c
    crossover_rate = slope_c * gen_n + init_c
    mutation_rate = slope_m * gen_c + init_m

    The key of finding applicable parameters for PowerfulSolver is to make reasonable assumptions, so that the number of
    tunable parameters is reduced, so the number of loops is reduced, so the experiment code does not take very long
    time to run. Other than the previous assumption, I designed another experiment to find parameters that based on the
    assumption that parameter slope_c can be equal to slope_m, and the parameter init_c can be the equal to init_m.
    Because both crossover and mutation have similar target
    """
    pop_size = 5
    init_m = 0.9
    slope_m = -0.001
    init_c = 0.9
    slope_c = -0.001
    min_mutation = 0.0285714285714
    def m_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)

    def c_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)

    ea_records_list = []
    labels = []
    solve_gens = []
    iteration = 0

    for ts in [1, 2, 3, 4, 5]:
        for tn in [1, 2, 3, 4]:
            ea_records, average_solved_at, avg_fitness_evals, n_solves = \
                experiment(SIZE, pop_size, m_func, ts, tn, c_func, N_GENERATIONS, plot_and_print=False)

            if n_solves >= 95:
                ea_records_list.append(ea_records)
                labels.append(f"init_c={init_c}, slope_c={slope_c}, ts={ts}, tn={tn}")
                solve_gens.append(average_solved_at)
            iteration += 1
            print(f"{iteration}/20")

        indices_rank = sorted(range(len(labels)), key=lambda i: solve_gens[i])
        ea_records_list = [ea_records_list[i] for i in indices_rank[-10:]]
        labels = [labels[i] for i in indices_rank[-10:]]
        solve_gens = [solve_gens[i] for i in indices_rank[-10:]]

    plots = [plt.plot(ea_records)[0] for ea_records in ea_records_list]
    plt.legend(plots, labels)
    plt.title("The performance curve of PowerfulSolver")
    plt.xlabel("Generation")
    plt.ylabel(f"Average fitness")
    plt.show()


if __name__ == '__main__':
    pop_size = 15
    init_m = 1.8127543060530658
    slope_m = -0.0016558461699318148
    min_mutation = 0.049647885507263775

    t_size = 5
    t_n_select = 2
    init_c = 2.6194913922128302
    slope_c = -0.00039274961188433814

    # def crossover_rate_update_func(generation_num):
    #     return max(slope_c * generation_num + init_c, min_mutation)
    #
    #
    # def mutation_rate_update_func(generation_num):
    #     return max(slope_m * generation_num + init_m, min_mutation)
    #
    #
    # experiment(SIZE,
    #            pop_size,
    #            mutation_rate_update_func,
    #            t_size, t_n_select,
    #            crossover_rate_update_func,
    #            N_GENERATIONS)

    parameterFind()
