from OneMaxObject import BitString, Population
from typing import *
import matplotlib.pyplot as plt

SIZE = 15   # size of the problem, the length of bitstring
N_GENERATIONS = 1000    # number of generations

"""
This is just a sample solver, it is not good for Worst OneMax Problem
Because it finds the solution too quickly
This just shows a very simple EA to solve the problem
After several tests, GreedyMutationSolver(15, 3, 1, 1000) solves the problem in around 97 out of 100 runs, 
with an average of 240 - 300 iterations
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
        self.population = Population(population_size, bitstring_len)    # our population, see Population class
        self.n_mutation = n_mutation
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len
        # record is used to keep fitness information of all generations
        # it is kind of a history log of the EA
        self.record = {
            "best fitness": [],
            "avg fitness": []
        }

    def run(self, verbose: bool = False) -> Tuple[int, int]:
        """
        Run the EA
        :param verbose: whether to print the result of this run
        :return: the best fitness found, and the generation index when the best fitness if found
        """
        best_answer_found_at = -1
        best_fitness_so_far = 0

        for i in range(self.n_iter):
            # Find the best bitstring in the present population
            best_bitstring = self.population.getBest()
            # do recording
            self.record["best fitness"].append(best_bitstring.fitness)
            self.record["avg fitness"].append(self.population.getAvgFitness())
            # if the best bitstring currently found is better than the best so far, update the best one so far
            if best_bitstring.fitness > best_fitness_so_far:
                best_fitness_so_far = best_bitstring.fitness
                best_answer_found_at = i

            # Just take a bitstring in population, and use the best bitstring to replace them
            self.population.pop()
            self.population.insert(0, best_bitstring.copy())

            # mutate every single item in the population except the optimal bitstring
            for bs in self.population:
                if not bs.isAllOnes():
                    bs.mutate(self.n_mutation)

        if verbose:
            best_bitstring = self.population.getBest()
            print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
                  f"found at iteration = {best_answer_found_at}")

        return best_fitness_so_far, best_answer_found_at


def experiment(solver_class: Callable, *args, **kwargs):
    n_solves = 0    # number of runs that the OneMax problem is solved
    average_solved_at = 0    # average number of generations the EA takes to solve EA
    avg_fitness_evals = 0   # average number of fitness evaluations the EA takes in 1000 generations
    # This is average best fitness of EA
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        print(f"{i+1}/{n_tests}")
        solver = solver_class(*args, **kwargs)
        best_fitness, best_found_at = solver.run()
        EA_records = [solver.record["best fitness"][i] + EA_records[i] for i in range(N_GENERATIONS)]
        # accumulate n_solves, and accumulate average_solved_at only the problem is solved
        # because if the problem is not solved, best_found_at != problem_solved_at
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
    plt.title("The performance curve of SampleOneMaxSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Best fitness of {n_tests} runs")
    plt.show()


if __name__ == '__main__':
    solver = GreedyMutationSolver(bitstring_len=SIZE, population_size=4, n_mutation=1, n_iter=N_GENERATIONS)
    solver.run()

    experiment(GreedyMutationSolver, SIZE, 3, 1, N_GENERATIONS)




