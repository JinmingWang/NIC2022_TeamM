from OneMaxObject import BitString, Population
import matplotlib.pyplot as plt
from argparse import ArgumentParser

SIZE = 15   # size of the problem, the length of bitstring
N_GENERATIONS = 1000    # number of generations

class LinearDecayMutationSolver:
    def __init__(self, bitstring_len, population_size, init_mutation, mutation_decay, min_mutation, n_gens):
        # initialize population according to population size and bitstring length
        self.pop = Population(population_size, bitstring_len)
        # set mutation rate to the initial mutation
        self.mut_rate = init_mutation
        # y = kx + b
        # mutation_rate = mutation_decay * generation_number + init_mutation
        self.mut_slope = mutation_decay
        # when mutation_rate <= min_mutation, does not decrease mutation anymore
        self.min_mutation = min_mutation
        # number of total generations
        self.n_gens = n_gens
        # length of bitstring, should be 15
        self.bs_len = bitstring_len
        # used to store running statistics
        self.record = {
            "best fitness": [],
            "avg fitness": []
        }

    def run(self, print_result=False):
        best_found_gen = -1
        best_fitness = 0

        for i in range(self.n_gens):
            # Find the best bitstring at this time
            best_bs = self.pop.getBest()
            # store the best fitness in the population
            self.record["best fitness"].append(best_bs.fitness)
            # store the average fitness of whole population
            self.record["avg fitness"].append(self.pop.getAvgFitness())

            # keep track the best bitstring ever found
            if best_bs.fitness > best_fitness:
                best_fitness = best_bs.fitness
                best_found_gen = i

            # Just remove the last bitstring in population, and insert a copy of the best bitstring at beginning
            self.pop.pop()
            self.pop.insert(0, best_bs.copy())

            # mutate every bitstring that is non-optimal
            for bs in self.pop:
                if not bs.isAllOnes():
                    bs.probabilisticMutation(self.mut_rate)

            # update mutation rate
            if self.mut_rate - self.mut_slope > self.min_mutation:
                self.mut_rate = self.mut_rate - self.mut_slope
            else:
                self.mut_rate = self.min_mutation

        if print_result:
            best_bs = self.pop.getBest()
            print(f"Final best bitstring = {best_bs}, fitness = {best_bs.fitness}, "
                  f"found at iteration = {best_found_gen}")

        return best_fitness, best_found_gen


def experiment(bitstring_len, population_size, init_mutation, mutation_decay, min_mutation, n_iter, plot=False):
    # how many times do we solve the problem
    n_solves = 0
    # on average, in what generation is the problem solved, larger is better, our target it to make it larger
    average_solved_gen = 0
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
        best_fitness, best_found_gen = solver.run()

        # accumulate fitness for every generation
        for i in range(N_GENERATIONS):
            EA_records[i] += solver.record["best fitness"][i]

        # if the optimal bitstring is found
        if best_fitness == SIZE:
            n_solves += 1
            # accumulate average solve generation
            average_solved_gen += best_found_gen
        # accumulate average fitness evaluations
        avg_fitness_evals += BitString.n_fitness_evals

    # get averages
    avg_fitness_evals /= n_tests
    average_solved_gen /= n_solves
    for rec in EA_records:
        rec /= n_tests

    print(f"OneMax problem solved {n_solves} out of {n_tests} runs, \n"
          f"Problem is solved with {average_solved_gen} iterations in average, \n"
          f"Average number of fitness evaluations is {avg_fitness_evals}.\n")

    if plot:
        plt.plot(EA_records)
        plt.title("The performance curve of LinearDecayMutationSolver")
        plt.xlabel("Iteration")
        plt.ylabel(f"Average Best fitness of {n_tests} runs")
        plt.show()


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
    parser = ArgumentParser()
    parser.add_argument("--pop_size", default=15, type=int, help="The population size when run the algorithm")
    parser.add_argument("--init_mutation", default=1.8127543060530658, type=float, help="The initial mutation at the first generation")
    parser.add_argument("--mutation_decay", default=-0.0016558461699318148, type=float, help="decay of mutation rate in each generation")
    parser.add_argument("--min_mutation", default=0.049647885507263775, type=float, help="the minimum value for mutation")

    parser.add_argument("-r", "--run_once", help="Whether to run algorithm just once", action="store_true")
    parser.add_argument("-e", "--evaluate", help="Whether to run the algorithm many time and evaluate",
                        action="store_true")
    parser.add_argument("-s", "--search_param", help="Whether to search for parameters", action="store_true")
    parser.add_argument("-a", "--analyze_searching", help="whether to analyze the searching results", action="store_true")

    args = parser.parse_args()

    if args.run_once:
        solver = LinearDecayMutationSolver(SIZE,
                                           population_size=args.pop_size,
                                           init_mutation=args.init_mutation,
                                           mutation_decay=args.mutation_decay,
                                           min_mutation=args.min_mutation,
                                           n_gens=N_GENERATIONS)
        solver.run(print_result=True)
    elif args.evaluate:
        experiment(SIZE, args.pop_size, args.init_mutation, args.mutation_decay, args.min_mutation, N_GENERATIONS)
    elif args.search_param:
        paramSearch2()
    elif args.analyze_searching:
        findBestInRecord()
    else:
        print("Wrong parameters, you must run with one of [-r, -e, -s, -a], now, run -e for default")
        experiment(SIZE, args.pop_size, args.init_mutation, args.mutation_decay, args.min_mutation, N_GENERATIONS)




