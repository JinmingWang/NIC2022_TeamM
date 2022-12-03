from OneMaxObject import BitString, Population
from typing import *
import matplotlib.pyplot as plt

SIZE = 15   # size of the problem, the length of bitstring
N_GENERATIONS = 1000    # number of generations

"""
This solver takes more iterations to solve OneMax problem, and also solves it with greater probability
After several tests, LinearMutationSolver(15, 4, 9/15, 8/750/15, 1/30, 1000)
solves the problem in around 98 out of 100 runs, 
with an average of 730 - 770 iterations
"""
class FunctionalMutationSolver:
    def __init__(self, bitstring_len: int, population_size: int, mutation_func: Callable[[int], float], n_iter: int):
        """
        A very stupid, simple solver, does poorly on Worst OneMax Problem
        :param bitstring_len: how is the length for bitstring
        :param population_size: population size
        :param mutation_func: a function that given generation number and returns mutation rate
        :param n_iter: number of total iterations
        """
        self.population = Population(population_size, bitstring_len)
        self.mutation_func = mutation_func
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len
        self.best_fitness_list = []

    def run(self, verbose: bool=False) -> Tuple[int, int]:
        """ Run the EA """
        best_answer_found_at = -1
        best_fitness_so_far = 0

        for i in range(self.n_iter):
            # Find the best bitstring at this time
            best_bitstring = self.population.getBest()
            self.best_fitness_list.append(best_bitstring.fitness)
            # print(f"Best fitness = {best_bitstring.fitness}")
            if best_bitstring.fitness > best_fitness_so_far:
                best_fitness_so_far = best_bitstring.fitness
                best_answer_found_at = i

            # Just take a bitstring in population, and use the best bitstring to replace them
            self.population.pop()
            self.population.insert(0, best_bitstring.copy())

            mutation_rate = self.mutation_func(i)
            for bs in self.population:
                if not bs.isAllOnes():
                    bs.probabilisticMutation(mutation_rate)

        if verbose:
            best_bitstring = self.population.getBest()
            print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
                  f"found at iteration = {best_answer_found_at}")

        return best_fitness_so_far, best_answer_found_at


def experiment(solver_class: Callable, *args, **kwargs):
    n_solves = 0
    average_solved_at = 0
    avg_fitness_evals = 0
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        print(f"{i+1}/{n_tests}")
        solver = solver_class(*args, **kwargs)
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
    plt.title("The performance curve of QuadraticDecayMutationFunc")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Best fitness of {n_tests} runs")
    plt.show()


if __name__ == '__main__':

    def linearDecayMutationFunc(generation_num: int) -> float:
        # solving takes 700 - 730 iterations, 99 out of 100 runs solved
        return max((-1/1375) * generation_num + 0.6, 1/30)

    def quadraticDecayMutationFunc(generation_num: int) -> float:
        # solving takes 770 - 830 iterations, 98 out of 100 runs solved
        return max(-6.54545454e-7 * generation_num**2 - 0.000447273 * generation_num + 0.8, 1/30)

    solver = FunctionalMutationSolver(SIZE, population_size=4, mutation_func=linearDecayMutationFunc,
                                      n_iter=N_GENERATIONS)
    solver.run(verbose=True)

    # experiment(FunctionalMutationSolver, SIZE, 4, quadraticDecayMutationFunc, N_GENERATIONS)

    pop_size = 5
    c0 = 0.6191933448286507
    c1 = -0.0006379984949527008
    c2 = -1.2635768176710518e-08
    min_mutation = 0.03245960272199035
    experiment(FunctionalMutationSolver, SIZE, pop_size,
               lambda gen_idx: max(c2 * gen_idx ** 2 + c1 * gen_idx + c0, min_mutation),
               N_GENERATIONS)




