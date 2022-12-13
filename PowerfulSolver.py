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
        self.mutation_func = mutation_rate_func
        self.t_size = t_size
        self.t_n_select = t_n_select
        self.crossover_rate_func = crossover_rate_func
        self.n_iter = n_iter
        self.bitstring_len = bitstring_len
        self.best_fitness_list = []
        self.doMutatation=doMutatation
        self.doCrossOver=doCrossOver


    def run(self):
        """ Run the EA """
        bestAnswerFoundAt = 0 #where is the best answer found
        bestFitnessSoFar = 0 #keep tab of the best fitness so far
        avgOfPopulation=[]
        doCrossOver=True #should it do crossover
        doMutatation=True #should it do mutation

        for i in range(self.n_iter):
            # Find the best bitstring at this time
            parents = self.population.tournamentSelect(self.t_size, self.t_n_select)
            children = []
            if(doCrossOver):

                crossover_rate = self.crossover_rate_func(i) # update crossover rate
                # do crossover and produce children
                for indexOfParent1 in range(len(parents)):
                    for indexOfParent2 in range(indexOfParent1 + 1, len(parents)): #this is done to ensure the same parent is not selected again
                        if random.random() < crossover_rate: #if the number is less than crossover, basically satisfies the condition
                            children.append(BitString.randomMaskCrossover(parents[indexOfParent1], parents[indexOfParent2]))  #then do the crossover
                        else:
                            children.append(parents[indexOfParent1].copy()) #otherwise just take parent1's bits

                # Add best bitstring in population to children
                best_bitstring = self.population.getBest() #get the current best bitstring in the population
                children.append(best_bitstring.copy()) #add this to the children
            else:
                children=[parent.copy() for parent in parents] #copy the same parent as a child
                bestBitString = self.population.getBest()

            # mutate each child then add all children to population

            if(doMutatation):
                mutation_rate = self.mutation_func(i)
            else:
                mutation_rate = 0

            for child in children:
                child.probabilisticMutation(mutation_rate)

            self.population.extend(children)

            numberOfChildren = len(children) #we need to replace equal to the number of children
            # Remove the worst numberOfChildren items
            for i in range(numberOfChildren):
                self.population.pop(self.population.getArgWorst())  #remove the worst elements from population

            best_bitstring = self.population.getBest()
            if best_bitstring.fitness > bestFitnessSoFar:
                bestFitnessSoFar = best_bitstring.fitness
                bestAnswerFoundAt = i
            self.best_fitness_list.append(
                best_bitstring.fitness)  # list of best fitness, this is used for plotting, nothing else!
            avgOfPopulation.append(
                self.population.getAvgFitness())  # average is stored, this is also for graphs only!


        best_bitstring = self.population.getBest()
        print(f"Final best bitstring = {best_bitstring}, fitness = {best_bitstring.fitness}, "
              f"found at iteration = {bestAnswerFoundAt}")

        return bestFitnessSoFar, bestAnswerFoundAt, avgOfPopulation


def experiment(bitstring_len,population_size,mutation_rate_func,t_size,t_n_select,crossover_rate_func,n_iter):
    numberOfSuccessfulSolves = 0 #number of successful solves
    averageSolvedAt = 0
    avgFitnessEvals = 0

    EA_records = [0 for _ in range(N_GENERATIONS)] #Create an array of size 1000 filled with 0's
    avgOfPopulationRecords=[0 for _ in range(N_GENERATIONS)] #Similarly create for recording average population


    numberOfTests = 100
    avgOfPopulationList=[]


    for i in range(numberOfTests): #this is run 100 times
        BitString.n_fitness_evals = 0
        print(f"{i + 1}/{numberOfTests}")
        solver = PowerfulSolver(bitstring_len, population_size, mutation_rate_func, t_size, t_n_select, crossover_rate_func, n_iter) #this is run 1100 times
        bestFitness, best_found_at,avg_of_population = solver.run()
        EA_records = [solver.best_fitness_list[i] + EA_records[i] for i in range(N_GENERATIONS)]
        avgOfPopulationRecords=[avg_of_population[i] + avgOfPopulationRecords[i] for i in range(N_GENERATIONS) ]
        #if we have fulfilled the condition of reaching the maximum number of bits
        if bestFitness == SIZE:
            numberOfSuccessfulSolves += 1 #this is out of 100 times how many times the program was successful
            averageSolvedAt += best_found_at
        avgFitnessEvals += BitString.n_fitness_evals
        avgOfPopulationList.append(avg_of_population)

    #this is for the experiement to find the variance
    my_avg=[] #contains the average of 1000 generations of each of the 100 runs

    for avgOfPopulation in avgOfPopulationList: #this is the 100 list
        avg_of_population_thousand=0
        for avg_of_population in avgOfPopulation: #this should be 1000 run
            avg_of_population_thousand+=avg_of_population
        avg_of_population_thousand=avg_of_population_thousand/n_iter #mean of the 1000
        my_avg.append(avg_of_population_thousand) #append the mean of 1000
    average_fitness_list.append(my_avg)
    # print(average_fitness_list)
    avgFitnessEvals /= numberOfTests
    if numberOfSuccessfulSolves == 0: #if it was not able to converge
        averageSolvedAt = 0
    else:
        averageSolvedAt /= numberOfSuccessfulSolves
    EA_records = [rec / numberOfTests for rec in EA_records] #get the average of the 100 iterations for each of the 1000 generations
    avgOfPopulationRecords=[rec / numberOfTests for rec in avgOfPopulationRecords] #Similarly create for recording average population

    graph_plotting_points.append(EA_records) #this is used for plotting graphs only!
    graph_plotting_points_for_average.append(avgOfPopulationRecords) #this is used for plotting graphs only!
    print(f"OneMax problem solved {numberOfSuccessfulSolves} out of {numberOfTests} runs, \n"
          f"Problem is solved with {averageSolvedAt} iterations in average, \n"
          f"Average number of fitness evaluations is {avgFitnessEvals}.\n")

    #plot for EA records
    plt.plot(EA_records)
    plt.title("The performance curve of PowerfulSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Best fitness of {numberOfTests} runs")
    plt.show()

    #plot for Variance
    plt.plot(my_avg)
    plt.title("The Variance of PowerfulSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Avg Fitness Over {n_iter} runs")
    plt.show()

    # plot for Mean
    plt.plot(avgOfPopulationRecords)
    plt.title("The Mean Fitness of PowerfulSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Avg Fitness Over {n_iter} runs")
    plt.show()


if __name__ == '__main__':
    doMutatation=True
    doCrossOver=True
    random.seed(9001)
    graph_plotting_points=[]
    average_fitness_list=[]
    graph_plotting_points_for_average=[]
    #best parameters only
    # pop_size = 10
    # init_m = 1.8127543060530658
    # slope_m = -0.0016558461699318148
    # t_size = 5
    # t_n_select = 2
    # init_c = 2.6194913922128302
    # slope_c = -0.00039274961188433814
    # min_mutation = 0.049647885507263775


    def crossover_rate_update_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)


    def mutation_rate_update_func(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)


    #params contain the best parameters to use
    params=[{'pop_size': 11, 'slope_m': -0.0017042824069558224, 'init_m': 1.8144359063754794, 'slope_c': -0.0006327768640716376, 'init_c': 2.6497520200282216, 't_size': 5, 't_n_select': 3, 'min_mutation': 0.04312303793228932},
            {'pop_size': 11, 'slope_m': -0.0017205531432836565, 'init_m': 1.859049176254803,
             'slope_c': -0.0006521540432490244, 'init_c': 2.6451775614584956, 't_size': 5, 't_n_select': 4,
             'min_mutation': 0.04483079376396767},
            {'pop_size': 10, 'slope_m': -0.0016699342558476283, 'init_m': 1.8588093846386473,
             'slope_c': -0.0006048634490720213, 'init_c': 2.8129018459331836, 't_size': 5, 't_n_select': 6,
             'min_mutation': 0.046804546648857616},
            {'pop_size': 11, 'slope_m': -0.0016926280983670906, 'init_m': 1.8006879964071576,
             'slope_c': -0.0006321599465726696, 'init_c': 2.664898290386694, 't_size': 5, 't_n_select': 3,
             'min_mutation': 0.05326772127033047},
            {'pop_size': 12, 'slope_m': -0.001680714245082619, 'init_m': 1.8452428553333604,
             'slope_c': -0.0006266274216993737, 'init_c': 3.3988098163408704, 't_size': 5, 't_n_select': 5,
             'min_mutation': 0.06670510633311266}
            ]
    for i in params:
        pop_size = i['pop_size']
        init_m = i['init_m']
        slope_m = i['slope_m']
        t_size = i['t_size']
        t_n_select = i['t_n_select']
        init_c = i['init_c']
        slope_c = i['slope_c']
        min_mutation = i['min_mutation']

        experiment(SIZE,pop_size,mutation_rate_update_func,t_size, t_n_select,crossover_rate_update_func,N_GENERATIONS)
    #this is used to plot the graph using the top 5 best fitnesses
    # for idx,i in enumerate(graph_plotting_points):
    #     plt.plot(range(len(i)), i, label=f'Max Fitness for Parameter {idx}')
    # plt.legend()
    # plt.title("The performance curve of PowerfulSolver")
    # plt.xlabel("Iteration")
    # plt.ylabel(f"Average Best fitness of 100 runs")
    # plt.show()
    # this is used to plot average
    for idx,i in enumerate(graph_plotting_points_for_average):
        plt.plot(range(len(i)), i, label=f'Average Fitness for Parameter {idx+1}')
    plt.legend()
    plt.title("The Mean Fitness of Population")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Fitness of 100 runs")
    plt.show()




