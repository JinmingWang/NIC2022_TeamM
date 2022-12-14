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
    def __init__(self, bitstring_len,population_size,t_size,t_n_select,n_iter,doMutatation,doCrossOver):
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
        self.doMutatation=doMutatation
        self.doCrossOver=doCrossOver


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
                crossover_rate = getCrossOverRate(i) #
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
                mutation_rate = getMutationRate(i)
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


def experiment(bitstring_len, population_size, t_size, t_n_select, n_iter):
    n_solves = 0
    average_solved_at = 0
    avg_fitness_evals = 0
    EA_records = [0 for _ in range(N_GENERATIONS)]

    n_tests = 100
    for i in range(n_tests):
        BitString.n_fitness_evals = 0
        solver = PowerfulSolver(bitstring_len, population_size, t_size, t_n_select, n_iter, True, True)
        best_fitness, best_found_at, pop_avg = solver.run()
        EA_records = [solver.best_fitness_list[i] + EA_records[i] for i in range(N_GENERATIONS)]
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
    return EA_records, average_solved_at, avg_fitness_evals, n_solves


if __name__ == '__main__':

    """
    It is possible that we can use slope_m for slope_c and also init_m for init_c
    """
    pop_size = 5
    init_m = 0.9
    slope_m = -0.001
    init_c = 0.9
    slope_c = -0.001
    min_value = 0.0285714285714
    def getMutationRate(generation_num):
        return max(slope_m * generation_num + init_m, min_value)

    def getCrossOverRate(generation_num):
        return slope_c * generation_num + init_c

    ea_data_list = []
    iteration = 0

    for ts in range(1, 6):
        for tn in range(1, 5):
            ea_records, average_solved_at, avg_fitness_evals, n_solves = \
                experiment(SIZE, pop_size, ts, tn, N_GENERATIONS, plot_and_print=False)

            if n_solves > 94:
                ea_data_list.append(
                    (ea_records, f"init_c={init_c}, slope_c={slope_c}, ts={ts}, tn={tn}", average_solved_at)
                )
            iteration += 1
            print(f"{iteration}/20")

    ea_data_list.sort(key=lambda ea_data: ea_data[2])

    plots = [plt.plot(ea_data[0])[0] for ea_data in ea_data_list]
    plt.legend(plots, [ea_data[1] for ea_data in ea_data_list])
    plt.title("Convergence Plots")
    plt.xlabel("Generations")
    plt.ylabel(f"Fitness")
    plt.show()
