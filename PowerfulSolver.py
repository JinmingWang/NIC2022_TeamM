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
        self.mutation_rate_func = mutation_rate_func
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
                mutation_rate = self.mutation_rate_func(i)
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

        return bestFitnesSoFar, bestAnswerFoundAt


def experiment(bitstring_len,population_size,getMutationRate,getCrossOverRate,t_size,t_n_select,n_iter):
    n_solves = 0
    average_solved_at = 0
    avg_fitness_evals = 0

    EA_records = [0 for _ in range(N_GENERATIONS)] #Create an array of size 1000 filled with 0's

    n_tests = 100
    avg_of_population_list=[]


    for i in range(n_tests): #this is run 100 times
        BitString.n_fitness_evals = 0
        print(f"{i + 1}/{n_tests}")
        solver = PowerfulSolver(bitstring_len, population_size, getMutationRate, t_size, t_n_select, getCrossOverRate, n_iter) #this is run 1100 times
        best_fitness, best_found_at = solver.run()
        EA_records = [solver.best_fitness_list[i] + EA_records[i] for i in range(N_GENERATIONS)]
        #if we have reached the maximum number of bits
        if best_fitness == SIZE:
            n_solves += 1 #this is out of 100 times how many times the program was successful
            average_solved_at += best_found_at
        avg_fitness_evals += BitString.n_fitness_evals

    #this is for the experiement to find the variance
    my_avg=[] #contains the average of 1000 generations of each of the 100 runs

    for avg_of_populations in avg_of_population_list: #this is the 100 list
        avg_of_population_thousand=0
        for avg_of_population in avg_of_populations: #this should be 1000 run
            avg_of_population_thousand+=avg_of_population
        avg_of_population_thousand=avg_of_population_thousand/n_iter #mean of the 1000
        my_avg.append(avg_of_population_thousand) #append the mean of 1000

    # print(average_fitness_list)
    avg_fitness_evals /= n_tests
    average_solved_at /= n_solves
    EA_records = [rec / n_tests for rec in EA_records]

    print(f"OneMax problem solved {n_solves} out of {n_tests} runs, \n"
          f"Problem is solved with {average_solved_at} iterations in average, \n"
          f"Average number of fitness evaluations is {avg_fitness_evals}.\n")

    #plot for EA records
    plt.plot(EA_records)
    plt.title("The performance curve of PowerfulSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Best fitness of {n_tests} runs")
    plt.show()


if __name__ == '__main__':
    pop_size = 11
    slope_m = -0.0017042824069558224
    init_m = 1.8144359063754794
    t_size = 5
    t_n_select = 3
    init_c = 2.6497520200282216
    slope_c = -0.0006327768640716376
    min_mutation = 0.04312303793228932

    def getMutationRate(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)


    def getCrossOverRate(generation_num):
        return slope_c * generation_num + init_c

    experiment(SIZE,
               pop_size,
               getMutationRate,
               getCrossOverRate,
               t_size,
               t_n_select,
               N_GENERATIONS)