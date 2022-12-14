import random

from OneMaxObject import BitString, Population
from typing import *
import matplotlib.pyplot as plt

SIZE = 15  # size of the problem, the length of bitstring
N_GENERATIONS = 1100  # number of generations

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


def experiment(bitstring_len,population_size,t_size,t_n_select,n_iter):
    numberOfSuccessfulSolves = 0
    avgSolvedAt = 0
    avgFitnessEvals = 0

    EA_records = [0 for _ in range(N_GENERATIONS)] #Create an array of size 1000 filled with 0's
    avgOfPopulationRecords=[0 for _ in range(N_GENERATIONS)]

    numberOfTests = 100
    avgOfPopulationList=[]

    for i in range(numberOfTests): #this is run 100 times
        BitString.n_fitness_evals = 0
        print(f"{i + 1}/{numberOfTests}")
        solver = PowerfulSolver(bitstring_len, population_size, t_size, t_n_select, n_iter,doMutatation,doCrossOver)  #initialize the v
        bestFitness, best_found_at,avg_of_population = solver.run() #this is run 1100 times
        EA_records = [solver.best_fitness_list[i] + EA_records[i] for i in range(N_GENERATIONS)]
        avgOfPopulationRecords=[avg_of_population[i] + avgOfPopulationRecords[i] for i in range(N_GENERATIONS) ]
        # print(bestFitness)
        if bestFitness == SIZE: #if the max 1's = len of string that is we reached the maximum condition
            numberOfSuccessfulSolves += 1 #this is out of 100 times how many times the program was successful
            avgSolvedAt += best_found_at #this is an addition of the generations it took to solve the problem. The number is later divided by the number of iterations to get the mean
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
    averageFitnessList.append(my_avg)

    avgFitnessEvals /= numberOfTests #to get the average fitness evaluations divide it by the total number of tests
    if numberOfSuccessfulSolves==0: #if it was not able to converge
        avgSolvedAt=0
    else:
        avgSolvedAt /= numberOfSuccessfulSolves
    EA_records = [rec / numberOfTests for rec in EA_records] #get the average of the 100 iterations for each of the 1000 generations
    avgOfPopulationRecords=[rec / numberOfTests for rec in avgOfPopulationRecords] #Similarly create for recording average population

    graphPlottingPoints.append(EA_records) #this is used for plotting graphs only!
    graphPlottingPointsForAverage.append(avgOfPopulationRecords) #this is used for plotting graphs only!
    print(f"OneMax problem solved {numberOfSuccessfulSolves} out of {numberOfTests} runs, \n"
          f"Problem is solved with {avgSolvedAt} iterations in average, \n"
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
    graphPlottingPoints=[]
    averageFitnessList=[]
    graphPlottingPointsForAverage=[]
    graphForCheckingCrossoverAndMutataion=[]
    doMutatation=True
    doCrossOver=False
    #best parameters only
    # pop_size = 10
    # init_m = 1.8127543060530658
    # slope_m = -0.0016558461699318148
    # t_size = 5
    # t_n_select = 2
    # init_c = 2.6194913922128302
    # slope_c = -0.00039274961188433814
    # min_mutation = 0.049647885507263775
    # random.seed(9001)  #use seed to test against same value

    #params contain the best parameters to use
    params = [{'pop_size': 2, 'slope_m': -0.00190035547769348, 'init_m': 2.0696841644030086, 'slope_c': -0.0013222773451903853, 'init_c': 0.15289537516658042, 't_size': 5, 't_n_select': 6, 'min_mutation': 0.07067868493754675},
              {'pop_size': 4, 'slope_m': -0.001957701034362916, 'init_m': 2.1202870215539527, 'slope_c': -0.0013596318060912116, 'init_c': -0.04210412920200261, 't_size': 5, 't_n_select': 6, 'min_mutation': 0},
              {'pop_size': 4, 'slope_m': -0.001864088670680492, 'init_m': 1.9907319381493194, 'slope_c': -0.0012258770440246344, 'init_c': 0.17111789659469562, 't_size': 2, 't_n_select': 5, 'min_mutation': 0.0255478955483956},
              {'pop_size': 2, 'slope_m': -0.001909124358725183, 'init_m': 2.050354491204239, 'slope_c': -0.0013035474594771438, 'init_c': -0.012345997363558903, 't_size': 3, 't_n_select': 6, 'min_mutation': 0.07017756023655906},
              {'pop_size': 4, 'slope_m': -0.0019034109601921356, 'init_m': 2.0167117971704744, 'slope_c': -0.000528620854297548, 'init_c': -0.15894540543435776, 't_size': 2, 't_n_select': 4, 'min_mutation': 0.049206159137482476}
              ]
    paramsForComparison=[
        [{"bestValues":{'pop_size': 9, 'slope_m': -0.0013920363120503486, 'init_m': 1.581559434774981, 'slope_c': -0.0005515963259128969, 'init_c': 0.9835054558088345, 't_size': 4, 't_n_select': 5, 'min_mutation': 0.05484982521868023},
          "doCrossOver":True,
          "doMutation":True
          },
         {
            "bestValues":{'pop_size': 5, 'slope_m': -0.0015825355573312217, 'init_m': 1.6874908882421278, 'slope_c': -0.0010039723832776116, 'init_c': 1.8017024424766042, 't_size': 2, 't_n_select': 4, 'min_mutation': 0.08984000988570444},
             "doCrossOver": False,
             "doMutation": True
         },
         {
             "bestValues":{'pop_size': 6, 'slope_m': -0.0014973464526205587, 'init_m': 1.4889355415694925, 'slope_c': -0.0008667713521799179, 'init_c': 1.581189649004805, 't_size': 2, 't_n_select': 2, 'min_mutation': 0.01613491833920627},
             "doCrossOver": True,
             "doMutation": False
         }
         ],

    ]

    def getCrossOverRate(generation_num):
        return slope_c * generation_num + init_c


    def getMutationRate(generation_num):
        return max(slope_m * generation_num + init_m, min_mutation)


    for param in params: #iterate through the top 5 EA predicted values provided in the list
        pop_size = param['pop_size']
        init_m = param['init_m']
        slope_m = param['slope_m']
        t_size = param['t_size']
        t_n_select = param['t_n_select']
        init_c = param['init_c']
        slope_c = param['slope_c']
        min_mutation = param['min_mutation']

        experiment(SIZE,pop_size,t_size, t_n_select,N_GENERATIONS)


    #this is used to plot the graph using the top 5 best fitnesses
    for idx,i in enumerate(graphPlottingPoints):
        plt.plot(range(len(i)), i, label=f'Max Fitness for Parameter {idx}')
    plt.legend()
    plt.title("The performance curve of PowerfulSolver")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Best fitness of 100 runs")
    plt.show()

    #this is used to plot average
    for idx,i in enumerate(graphPlottingPointsForAverage):
        plt.plot(range(len(i)), i, label=f'Average Fitness for Parameter {idx+1}')
    plt.legend()
    plt.title("The Mean Fitness of Population")
    plt.xlabel("Iteration")
    plt.ylabel(f"Average Fitness of 100 runs")
    plt.show()




