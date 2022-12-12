from OneMaxObject import BitString, Population
from typing import *
import matplotlib.pyplot as plt
from sys import argv
from argparse import ArgumentParser

SIZE = 15   # size of the problem, the length of bitstring
N_GENERATIONS = 1000    # number of generations

"""
This is just a sample solver, it is not good for Worst OneMax Problem
Because it finds the solution too quickly
This just shows a very simple EA to solve the problem
After several tests, GreedyMutationSolver(15, 3, 1, 1000) solves the problem in around 97 out of 100 runs, 
with an average of 240 - 300 iterations
"""
class SimpleOneMaxSolver:
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


def experiment(*args, plot_and_print=True, **kwargs):
    n_solves = 0    # number of runs that the OneMax problem is solved
    average_solved_at = 0    # average number of generations the EA takes to solve EA
    avg_fitness_evals = 0   # average number of fitness evaluations the EA takes in 1000 generations
    # This is average best fitness of EA
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        if plot_and_print:
            print(f"{i+1}/{n_tests}")
        solver = SimpleOneMaxSolver(*args, **kwargs)
        best_fitness, best_found_at = solver.run()
        EA_records = [solver.record["best fitness"][i] + EA_records[i] for i in range(N_GENERATIONS)]
        # accumulate n_solves, and accumulate average_solved_at only the problem is solved
        # because if the problem is not solved, best_found_at != problem_solved_at
        if best_fitness == SIZE:
            n_solves += 1
            average_solved_at += best_found_at
        avg_fitness_evals += BitString.n_fitness_evals

    avg_fitness_evals /= n_tests
    if n_solves == 0:
        average_solved_at = 1000
    else:
        average_solved_at /= n_solves
    EA_records = [rec / n_tests for rec in EA_records]
    print(f"OneMax problem solved {n_solves} out of {n_tests} runs, \n"
          f"Problem is solved with {average_solved_at} iterations in average, \n"
          f"Average number of fitness evaluations is {avg_fitness_evals}.\n")

    if plot_and_print:
        plt.plot(EA_records)
        plt.title("The performance curve of SampleOneMaxSolver")
        plt.xlabel("Iteration")
        plt.ylabel(f"Average Best fitness of {n_tests} runs")
        plt.show()

    return EA_records, average_solved_at, avg_fitness_evals, n_solves


def parameterSearch1():
    """ Run parameter search, stage 1 """

    # Only the data of top 10 best parameter combination is stored
    convergence_records = []
    labels = []
    solve_generations = []

    # Run iteration for population and num mutations in valid range
    for pop_size in range(1, 20):
        for n_mutation in range(1, 15):
            print(f"Running {pop_size=}, {n_mutation=}")
            convergence_record, average_solved_at, avg_fitness_evals, n_solves = \
                experiment(SimpleOneMaxSolver, SIZE, pop_size, n_mutation, N_GENERATIONS, plot_and_print=False)

            # Do not even consider the once that are too bad
            if n_solves >= 95 and avg_fitness_evals < 100000:
                convergence_records.append(convergence_record)
                labels.append(f"p={pop_size}, m={n_mutation}")
                solve_generations.append(average_solved_at)

        # sort plots according to solve generation, get the best 10
        indices_ranking = sorted(range(len(labels)), key=lambda i: solve_generations[i], reverse=True)
        convergence_records = [convergence_records[i] for i in indices_ranking[:10]]
        labels = [labels[i] for i in indices_ranking[:10]]
        solve_generations = [solve_generations[i] for i in indices_ranking[:10]]

    # plot the top-10
    plots = [plt.plot(convergence_record)[0] for convergence_record in convergence_records]
    plt.legend(plots, labels)
    plt.title("The performance curve of SampleOneMaxSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average fitness of 100 runs")
    plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-p", "--pop_size", default=4, type=int, help="The population size when run the algorithm")
    parser.add_argument("-m", "--mutation", default=1, type=int, help="The mutation size when run the algorithm")
    parser.add_argument("-r", "--run_once", help="Whether to run algorithm just once", action="store_true")
    parser.add_argument("-e", "--evaluate", help="Whether to run the algorithm many time and evaluate", action="store_true")
    parser.add_argument("-s", "--search_param", help="Whether to search for parameters", action="store_true")

    args = parser.parse_args()

    if args.run_once:
        SimpleOneMaxSolver(bitstring_len=SIZE,
                           population_size=args.pop_size,
                           n_mutation=args.mutation,
                           n_iter=N_GENERATIONS).run(verbose=True)
    elif args.evaluate:
        experiment(SIZE, args.pop_size, args.mutation, N_GENERATIONS)
    elif args.search_param:
        parameterSearch1()
    else:
        print("Wrong parameters, you must run with one of [-r, -e, -s], now, run -e for default")
        experiment(SIZE, args.pop_size, args.mutation, N_GENERATIONS)




