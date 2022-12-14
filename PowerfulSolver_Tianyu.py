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
    def __init__(self, bitstring_len,population_size,mutation_rate_func,t_size,t_n_select,crossover_rate_func,n_iter):
        '''
        The idea behind this algorithm was to create a method to use linear degradation across
        mutation and crossover
        '''
        #initialize the paramaters
        self.population = Population(population_size, bitstring_len)
        self.t_size = t_size
        self.t_n_select = t_n_select
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len
        self.best_fitness_list = []
        self.mutation_func = mutation_rate_func
        self.crossover_rate_func = crossover_rate_func

    def run(self):
        """ Run the EA """
        bestAnswerFoundAt = 0 #where is the best answer found
        bestFitnesSoFar = 0 #keep tab of the best fitness so far
        avgOfPopulation=[]
        doCrossOver=True #should it do crossover
        doMutatation=True #should it do mutation


        for i in range(self.n_iter): #run the code 100 times
            # Find the best bitstring at this time
            parents = self.population.tournamentSelect(self.t_size, self.t_n_select) #run tournament selection
            children = [] #array is used to store children
            # update crossover rate
            if(doCrossOver): #either do crossover
                crossover_rate = self.crossover_rate_func(i) #
                # do crossover and produce children
                for indexOfParent1 in range(len(parents)):
                    for indexOfParent2 in range(indexOfParent1 + 1, len(parents)): #this is done to ensure the same parent is not selected again
                        if random.random() < crossover_rate: #if the number is less than crossover, basically satisfies the condition
                            children.append(BitString.randomMaskCrossover(parents[indexOfParent1], parents[indexOfParent2]))  #then do the crossover
                        else:
                            children.append(parents[indexOfParent1].copy()) #otherwise just take parent1's bits

                # Add best bitstring in population to children
                bestBitString = self.population.getBest() #get the current best bitstring in the population
                # children.append(bestBitString.copy()) #add this to the children
            else: #do not do crossover just copy parents to children
                bestBitString = self.population.getBest()
                children=parents.copy()

            if(doMutatation):
                # mutate each child then add all children to population
                mutation_rate = self.mutation_func(i)
                for child in children:
                    child.probabilisticMutation(mutation_rate)


            self.population.extend(children) #add children to the population

            numberOfChildren = len(children) #we need to replace equal to the number of children
            # Remove the worst numberOfChildren items
            for j in range(numberOfChildren):
                self.population.pop(self.population.getArgWorst())  #remove the worst elements from population

            bestBitString = self.population.getBest()
            if bestBitString.fitness > bestFitnesSoFar: #this is for printing the best value
                bestFitnesSoFar = bestBitString.fitness #note down the best fitness, this is used later to count how many times we satisfied the count criteria
                bestAnswerFoundAt = i
            self.best_fitness_list.append(bestBitString.fitness)  # list of best fitness, this is used for plotting, nothing else!
            avgOfPopulation.append(self.population.getAvgFitness())  # average is stored, this is also for graphs only!

        return bestFitnesSoFar, bestAnswerFoundAt, avgOfPopulation


def experiment(bitstring_len, population_size, mutation_rate_func, t_size, t_n_select, crossover_rate_func, n_iter):
    n_solves = 0
    average_solved_at = 0
    avg_fitness_evals = 0
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        solver = PowerfulSolver(bitstring_len, population_size, mutation_rate_func, t_size, t_n_select, crossover_rate_func, n_iter)
        best_fitness, best_found_at, average_of_pop = solver.run()
        EA_records = [solver.best_fitness_list[i] + EA_records[i] for i in range(N_GENERATIONS)]
        if best_fitness == SIZE:
            n_solves += 1
            average_solved_at += best_found_at
        avg_fitness_evals += BitString.n_fitness_evals

    avg_fitness_evals /= n_tests
    average_solved_at /= n_solves
    EA_records = [rec / n_tests for rec in EA_records]
    return EA_records, average_solved_at, avg_fitness_evals, n_solves


def parameterFind1():
    """
    Find Parameters, but 8 parameters are too much, so, for the parameters pop_size, init_m, slope_m and min_mutation,
    the best values for LinearDecayMutationSolver is used here and fixed. I search for the other 4 parameters.
    """
    pop_size = 5
    init_m = 0.9
    slope_m = -0.001
    min_mutation = 0.0285714285714
    def m_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)

    ea_records_list = []
    labels = []
    solve_gens = []
    iteration = 0
    for init_c in [-2, 1, 0, 1, 2]:
        for slope_c in [-0.002, -0.001, 0.001, 0.002]:
            def c_func(generation_num):
                return slope_c * generation_num + init_c
            for ts in [1, 3, 5]:
                for tn in [2, 3]:
                    ea_records, average_solved_at, avg_fitness_evals, n_solves = experiment(SIZE, pop_size, m_func, ts, tn, c_func, N_GENERATIONS, plot_and_print=False)

                    if n_solves >= 95:
                        ea_records_list.append(ea_records)
                        labels.append(f"init_c={init_c}, slope_c={slope_c}, ts={ts}, tn={tn}")
                        solve_gens.append(average_solved_at)
                    iteration += 1
                    print(f"{iteration}/120")

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


def parameterFind2():
    """
    Find Parameters, but 8 parameters are too much, so, for the parameters pop_size, init_m, slope_m and min_mutation,
    the best values for LinearDecayMutationSolver is used here and fixed. I search for the other 4 parameters.
    """
    pop_size = 5
    init_m = 0.9
    slope_m = -0.001
    min_mutation = 0.0285714285714
    tn = 2
    ts = 1
    def m_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)

    ea_records_list = []
    labels = []
    solve_gens = []
    iteration = 0
    for init_c in [-2, -1.5, 1, 0.5, 0, 0.5, 1, 1.5, 2]:    # 9
        for slope_c in [-0.0025, -0.002, -0.0015, -0.001, -0.0005, 0.0005, 0.001, 0.0015, 0.002, 0.0025]:
            def c_func(generation_num):
                return slope_c * generation_num + init_c
            ea_records, average_solved_at, avg_fitness_evals, n_solves = \
                experiment(SIZE, pop_size, m_func, ts, tn, c_func, N_GENERATIONS)

            if n_solves >= 95:
                ea_records_list.append(ea_records)
                labels.append(f"init_c={init_c}, slope_c={slope_c}, ts={ts}, tn={tn}")
                solve_gens.append(average_solved_at)
            iteration += 1
            print(f"{iteration}/90")

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
    parameterFind1()
    parameterFind2()

