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
class LinearDecayMutationSolver:
    def __init__(self, bitstring_len, population_size, init_mutation, mutation_decay, min_mutation, n_iter):
        # initialize population according to population size and bitstring length
        self.population = Population(population_size, bitstring_len)
        # set mutation rate to the initial mutation
        self.mutation_rate = init_mutation
        # y = kx + b
        # mutation_rate = mutation_decay * generation_number + init_mutation
        self.mutation_decay = mutation_decay
        # when mutation_rate <= min_mutation, does not decrease mutation anymore
        self.min_mutation = min_mutation
        # number of total generations
        self.n_iter = n_iter
        # length of bitstring, should be 15
        self.bitstring_len = bitstring_len
        # used to store running statistics
        self.record = {
            "best fitness": [],
            "avg fitness": []
        }

    def run(self, print_result=False):
        best_answer_found_at = -1
        best_fitness_so_far = 0

        for i in range(self.n_iter):
            # Find the best bitstring at this time
            best_bitstring = self.population.getBest()
            # store the best fitness in the population
            self.record["best fitness"].append(best_bitstring.fitness)
            # store the average fitness of whole population
            self.record["avg fitness"].append(self.population.getAvgFitness())

            # keep track the best bitstring ever found
            if best_bitstring.fitness > best_fitness_so_far:
                best_fitness_so_far = best_bitstring.fitness
                best_answer_found_at = i

            # Just remove the last bitstring in population, and insert a copy of the best bitstring at beginning
            self.population.pop()
            self.population.insert(0, best_bitstring.copy())

            # mutate every bitstring that is non-optimal
            for bs in self.population:
                if not bs.isAllOnes():
                    bs.probabilisticMutation(self.mutation_rate)

            # update mutation rate
            if self.mutation_rate - self.mutation_decay > self.min_mutation:
                self.mutation_rate = self.mutation_rate - self.mutation_decay
            else:
                self.mutation_rate = self.min_mutation

        if print_result:
            best_bitstring = self.population.getBest()
            print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
                  f"found at iteration = {best_answer_found_at}")

        return best_fitness_so_far, best_answer_found_at


def experiment(bitstring_len, population_size, init_mutation, mutation_decay, min_mutation, n_iter):
    # how many times do we solve the problem
    n_solves = 0
    # on average, in what generation is the problem solved, larger is better, our target it to make it larger
    average_solved_at = 0
    # on average, how many times id the fitness evaluated
    avg_fitness_evals = 0
    # records of the algorithm, the i-th element is the average best fitness over all runs in i-th generation
    EA_records = [0] * N_GENERATIONS

    # run algorithm 100 times
    n_tests = 100
    for i in range(n_tests):
        # reset number of evaluations
        BitString.n_fitness_evals = 0
        # print(i+1, "/", n_tests) # process bar
        # initialize algorithm and run it once
        solver = LinearDecayMutationSolver(bitstring_len, population_size, init_mutation, mutation_decay, min_mutation, n_iter)
        best_fitness, best_found_at = solver.run()

        # accumulate fitness for every generation
        for i in range(N_GENERATIONS):
            EA_records[i] += solver.record["best fitness"][i]

        # if the optimal bitstring is found
        if best_fitness == SIZE:
            n_solves += 1
            # accumulate average solve generation
            average_solved_at += best_found_at
        # accumulate average fitness evaluations
        avg_fitness_evals += BitString.n_fitness_evals

    # get averages
    avg_fitness_evals /= n_tests
    average_solved_at /= n_solves
    for rec in EA_records:
        rec /= n_tests

    print(f"OneMax problem solved {n_solves} out of {n_tests} runs, \n"
          f"Problem is solved with {average_solved_at} iterations in average, \n"
          f"Average number of fitness evaluations is {avg_fitness_evals}.\n")
    # plt.plot(EA_records)
    # plt.title("The performance curve of LinearDecayMutationSolver")
    # plt.xlabel("Iteration")
    # plt.ylabel(f"Average Best fitness of {n_tests} runs")
    # plt.show()


def findBestInRecord():
    file = open("LinearDecayResults2.txt", "r")

    experiment_data = []

    lines = file.readlines()
    n_lines = len(lines)
    for line_i in range(0, n_lines, 5):
        words = lines[line_i].strip().split()   # remove \n and split into words
        init_mutation = float(words[3])
        mutation_decay = float(words[4])
        min_mutation = float(words[5])

        words = lines[line_i + 1].strip().split()
        n_solves = int(words[3])

        words = lines[line_i + 2].strip().split()
        solve_generation = float(words[4])

        words = lines[line_i + 3].strip().split()
        last_word = words[-1]
        fitness_evals = float(last_word[:-1])

        if n_solves > 95:
            experiment_data.append([init_mutation, mutation_decay, min_mutation, n_solves, solve_generation, fitness_evals])

    # sort experiment data from the largest solve_generation to smallest
    sorted_data = sorted(experiment_data, key = lambda single_data: single_data[4], reverse=True)

    print(len(sorted_data))

    for i in range(20):
        print("Top", i + 1, ":", sorted_data[i])

    file.close()


def paramSearch1():
    for init_mutation in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        for mutation_decay in [i/1000 for i in range(1, 11)]:
            for min_mutation in [1/10, 1/15, 1/20, 1/25, 1/30, 1/35, 1/40]:
                print("Running with parameter", init_mutation, mutation_decay, min_mutation)
                experiment(SIZE, 4, init_mutation, mutation_decay, min_mutation, N_GENERATIONS)

def paramSearch2():
    for init_mutation in [0.7, 0.8, 0.9, 1.0]:
        for mutation_decay in [0.0008, 0.0009, 0.001, 0.0011, 0.0012]:
            for min_mutation in [1/10, 1/15, 1/20, 1/25, 1/30, 1/35, 1/40]:
                print("Running with parameter", init_mutation, mutation_decay, min_mutation)
                experiment(SIZE, 5, init_mutation, mutation_decay, min_mutation, N_GENERATIONS)


if __name__ == '__main__':
    solver = LinearDecayMutationSolver(SIZE, population_size=4, init_mutation=3/15, mutation_decay=2/950/15,
                                       min_mutation=1/15, n_iter=N_GENERATIONS)
    solver.run(print_result=True)


    # paramSearch2()

    findBestInRecord()

    # init_mutation = 9/15
    # mutation_decay = 8/750/15
    # min_mutation = 1/30
    # experiment(SIZE, 4, init_mutation, mutation_decay, min_mutation, N_GENERATIONS)




