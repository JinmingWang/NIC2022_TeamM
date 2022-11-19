from OneMaxObject import BitString
from typing import *

SIZE = 15   # size of the problem, the length of bitstring
N_GENERATIONS = 1000    # number of generations

"""
This solver takes more iterations to solve OneMax problem, and also solves it with greater probability
After several tests, LinearMutationSolver(15, 4, 6/15, 4/900/15, 1/15, 1000)
solves the problem in around 99 out of 100 runs, 
with an average of 400 - 450 iterations
"""
class LinearDecayMutationSolver:
    def __init__(self, bitstring_len: int, population_size: int, init_mutation: float, mutation_decay: float,
                 min_mutation: float, n_iter: int):
        """
        A very stupid, simple solver, does poorly on Worst OneMax Problem
        :param bitstring_len: how is the length for bitstring
        :param population_size: population size
        :param init_mutation: the initial mutation number applied to each bitstring
        :param mutation_decay: how much to decay mutation on each generation (this decay is subtracted from the mutation)
        :param min_mutation: the minimum mutation number
        :param n_iter: number of total iterations
        """
        self.population = [BitString(bitstring_len) for _ in range(population_size)]
        self.mutation_rate = init_mutation
        self.mutation_decay = mutation_decay
        self.min_mutation = min_mutation
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len

    def run(self, verbose: bool=False) -> Tuple[int, int]:
        """ Run the EA """
        best_answer_found_at = -1
        best_solution = BitString.zeroString(self.bitstring_len)

        for i in range(self.n_iter):
            # Find the best bitstring at this time
            best_bitstring = max(self.population, key=lambda bs: bs.fitness)
            # print(f"Best fitness = {best_bitstring.fitness}")
            if best_bitstring.fitness > best_solution.fitness:
                best_solution = best_bitstring.copy()
                best_answer_found_at = i

            # Just take a bitstring in population, and use the best bitstring to replace them
            self.population.pop()
            self.population.insert(0, best_bitstring.copy())

            for bs in self.population:
                if not bs.isAllOnes():
                    bs.probabilisticMutation(self.mutation_rate)
                    if self.mutation_rate - self.mutation_decay > self.min_mutation:
                        self.mutation_rate = self.mutation_rate - self.mutation_decay
                    else:
                        self.mutation_rate = self.min_mutation

        best_bitstring = max(self.population, key=lambda bs: bs.fitness)
        if verbose:
            print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
                  f"found at iteration = {best_answer_found_at}")

        return best_bitstring.fitness, best_answer_found_at


def experiment(solver_class: Callable, *args, **kwargs):
    n_solves = 0
    average_found_at = 0

    for i in range(100):
        print(f"{i+1}/100")
        solver = solver_class(*args, **kwargs)
        best_fitness, best_found_at = solver.run()
        if best_fitness == SIZE:
            n_solves += 1
        average_found_at += best_found_at

    average_found_at /= 100
    print(f"OneMax problem solved {n_solves} out of 100 runs, \n"
          f"Problem is solved with {average_found_at} iterations in average.")


if __name__ == '__main__':
    solver = LinearDecayMutationSolver(SIZE, population_size=4, init_mutation=3/15, mutation_decay=2/900/15,
                                       min_mutation=1/15, n_iter=N_GENERATIONS)
    solver.run(verbose=True)

    experiment(LinearDecayMutationSolver, SIZE, 4, 6/15, 4/900/15, 1/15, N_GENERATIONS)




