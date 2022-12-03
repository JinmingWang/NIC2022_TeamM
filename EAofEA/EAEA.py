from __future__ import annotations  # Requires Python 3.7+
from typing import *
import matplotlib.pyplot as plt
from OneMaxObject import BitString, Population
from FunctionalMutationSolver import FunctionalMutationSolver
import random
import pickle
from multiprocessing import Pool

MAX_POP_SIZE = 20
SIZE = 15   # size of the problem, the length of bitstring
N_GENERATIONS = 1000    # number of generations

def getRandomBetween(lower: float, upper: float) -> float:
    """
    Generate a random number between lower and upper, using uniform distribution
    :param lower: lower bound for the random number
    :param upper: upper bound for the random number
    :return: the generated random number
    """
    diff = upper - lower
    return random.random() * diff + lower


def clip(num: float, lower: float, upper: float) -> float:
    """
    Restrict a number in range between lower and upper, if the number of smaller than lower, then set it to lower,
    if the number is larger than upper, then set it to upper, if the number is in between, then remain unchanged
    :param num: the number to be clipped
    :param lower: lower bound
    :param upper: upper bound
    :return: the clipped number
    """
    return max(min(num, upper), lower)


def evaluationSubprocess(param_comb: ParamCombination, child_id: int) -> Tuple[ParamCombination, int]:
    param_comb.evaluate()
    return param_comb, child_id


def onSubprocessError(err):
    print(f"Subprocess Error: {err}")


class ParamCombination:
    def __init__(self):
        self.pop_size = random.randint(1, 10)
        self.c2 = getRandomBetween(-1e-6, -1e-7)
        self.c1 = getRandomBetween(-0.001, -0.0001)
        self.c0 = getRandomBetween(0.1, 1.0)
        self.min_mutation = getRandomBetween(1/40, 1/10)
        self.fitness = 9999
        self.avg_solved_at = 0
        self.n_solves = 0

    def __str__(self):
        string = f"{self.pop_size=} {self.c0=} {self.c1=} {self.c2=} {self.min_mutation=} {self.fitness=}" \
                 f" {self.avg_solved_at=} {self.n_solves=}"
        return string.replace("self.", "")

    def mutate(self, mutation_factor: float) -> None:

        # It is not good to change population size a lot, because it has big influence on algorithm performance,
        # so change population size with low probability
        popsize_mutate_num = random.random()
        if popsize_mutate_num < 0.05:
            self.pop_size = min(self.pop_size + 1, MAX_POP_SIZE)
        elif popsize_mutate_num < 0.1:
            self.pop_size = max(1, self.pop_size - 1)

        # 50% percent chance to mutate c2
        c2_mutation_num = random.random()
        if c2_mutation_num < 0.7:
            self.c2 = clip(getRandomBetween(-2e-8 * mutation_factor, 2e-8 * mutation_factor), -3e-5, 3e-5)

        # 50% percent chance to mutate c1
        c1_mutation_num = random.random()
        if c1_mutation_num < 0.7:
            self.c1 = clip(self.c1 + getRandomBetween(-1e-5 * mutation_factor, 1e-5 * mutation_factor), -0.003, 0.003)

        c0_mutate_num = random.random()
        if c0_mutate_num < 0.7:
            self.c0 = clip(self.c0 + getRandomBetween(-0.01 * mutation_factor, 0.01 * mutation_factor), 0.1, 1.0)

        min_mutate_num = random.random()
        if min_mutate_num < 0.7:
            self.min_mutation = clip(self.min_mutation +
                                     getRandomBetween(-0.005 * mutation_factor, 0.005 * mutation_factor), 0, 1)

    def evaluate(self) -> None:
        """
        Compute the fitness of this set of parameters, the fitness is computed by actually run the algorithm many times,

        We have fitness_solve, it rewards high solving rate, because we should have a solve rate of > 95%
        We do not want it to increase even more, because increasing fitness_solve means decreasing fitness_worst
        So the maximum value for fitness_solve is 0.96
        fitness_solve = min(0.96, n_solves / n_tests)

        We have fitness_worst, it rewards high average solution found generation
        fitness_worst = avg_solved_at / N_GENERATIONS

        And we have fitness_overall, which is a weighted combine of both fitness functions
        fitness_solve has a factor of 2, because solving the problem is more important than having good performance
        fitness_overall = 2 * fitness_solve + fitness_worst
        """
        self.n_solves = 0
        self.avg_solved_at = 0

        mutation_func = lambda gen_idx: max(self.c2 * gen_idx ** 2 + self.c1 * gen_idx + self.c0, self.min_mutation)

        n_tests = 100
        for i in range(n_tests):
            BitString.n_fitness_evals = 0
            solver = FunctionalMutationSolver(SIZE, self.pop_size, mutation_func, N_GENERATIONS)
            best_fitness, best_found_at = solver.run()
            if best_fitness == SIZE:
                self.n_solves += 1
                self.avg_solved_at += best_found_at

        if self.n_solves == 0:
            self.avg_solved_at = 1000
        else:
            self.avg_solved_at /= self.n_solves

        fitness_solve = self.n_solves / n_tests
        fitness_worst = self.avg_solved_at / N_GENERATIONS
        # 1% solve success rate == solves in 20 more generations
        self.fitness = 2 * fitness_solve + fitness_worst
        if self.n_solves < 95:
            self.fitness /= 2


    def copy(self):
        new_param = ParamCombination()
        new_param.pop_size = self.pop_size
        new_param.c0 = self.c0
        new_param.c1 = self.c1
        new_param.c2 = self.c2
        new_param.min_mutation = self.min_mutation
        new_param.fitness = self.fitness
        new_param.n_solves = self.n_solves
        new_param.avg_solved_at = self.avg_solved_at
        return new_param

    def toList(self):
        return [self.pop_size, self.c0, self.c1, self.c2, self.min_mutation, self.fitness, self.n_solves,
                self.avg_solved_at]

    @staticmethod
    def fromList(data_list: List):
        new_param = ParamCombination()
        new_param.pop_size, new_param.c0, new_param.c1, new_param.c2, \
        new_param.min_mutation, new_param.fitness, new_param.n_solves, new_param.avg_solved_at = data_list
        return new_param

    @staticmethod
    def crossover(param1: ParamCombination, param2: ParamCombination) -> ParamCombination:
        """
        Do crossover between 2 ParamCombinations
        For each attribute (population_size, c0, c1, c2, min_mutation), there are 3 choices,
        the first is to inherit from parent1,
        the second is to take the average between parents,
        the third choice is to inherit from parent2
        :param param1: parent 1
        :param param2: parent 2
        :return: child, result of crossover
        """
        new_param = ParamCombination()

        choice = random.choice([0, 1, 2])
        if choice == 0:
            new_param.pop_size = param1.pop_size
        elif choice == 1:
            new_param.pop_size = (param1.pop_size + param2.pop_size) // 2
        else:
            new_param.pop_size = param2.pop_size

        choice = random.choice([0, 1, 2])
        if choice == 0:
            new_param.c0 = param1.c0
        elif choice == 1:
            new_param.c0 = (param1.c0 + param2.c0) / 2
        else:
            new_param.c0 = param2.c0

        choice = random.choice([0, 1, 2])
        if choice == 0:
            new_param.c1 = param1.c1
        elif choice == 1:
            new_param.c1 = (param1.c1 + param2.c1) / 2
        else:
            new_param.c1 = param2.c1

        choice = random.choice([0, 1, 2])
        if choice == 0:
            new_param.c2 = param1.c2
        elif choice == 1:
            new_param.c2 = (param1.c2 + param2.c2) / 2
        else:
            new_param.c2 = param2.c2

        choice = random.choice([0, 1, 2])
        if choice == 0:
            new_param.min_mutation = param1.min_mutation
        elif choice == 1:
            new_param.min_mutation = (param1.min_mutation + param2.min_mutation) / 2
        else:
            new_param.min_mutation = param2.min_mutation

        return new_param


class CombinationPopulation(list):
    def __init__(self, population_size: int):
        super(CombinationPopulation, self).__init__()
        p = Pool(5)
        for i in range(population_size):
            self.append(ParamCombination())

        def onSubprocessFinished(results):
            evaluated_child, child_id = results
            self[child_id] = evaluated_child.copy()
            print(f"Population initializing {child_id + 1}/{population_size}")

        for i in range(population_size):
            p.apply_async(evaluationSubprocess, [self[i], i], callback=onSubprocessFinished,
                          error_callback=onSubprocessError)

        p.close()
        p.join()
        print("Population Initialization Done")

    def getArgBest(self) -> int:
        """ Get the index of the best ParamCombination in the population """
        return max(range(len(self)), key=lambda i: self[i].fitness)

    def getArgWorst(self) -> int:
        """ Get the index of the worst ParamCombination in the population """
        return min(range(len(self)), key=lambda i: self[i].fitness)

    def tournamentSelect(self, tournament_size: int, n_select: int) -> List[ParamCombination]:
        """
        Randomly sample tournament_size elements in population and choose the best among them, repeat n_select times
        :param tournament_size: how many to sample from the population
        :param n_select: how many elements selected finally
        :return: a list of elements selected from tournament
        """
        selected = []
        for _ in range(n_select):
            group = random.sample(self, tournament_size)
            selected.append(max(group, key=lambda bs: bs.fitness))
        return selected


class OneMaxSolverParamFinder:
    def __init__(self, population_size: int, tournament_size: int, tournament_selections: int, mutation_factor: float,
                 n_generations: int):
        self.population: CombinationPopulation[ParamCombination] = CombinationPopulation(population_size)
        self.t_size = tournament_size
        self.t_selections = tournament_selections
        self.mutation_factor = mutation_factor
        self.n_generations = n_generations

    def run(self, n_processes: int = 4):
        """
        Run EA algorithm with multiprocessing
        :param n_processes: number of processes
        """

        children = []
        def onSubprocessFinished(results):
            evaluated_child, child_id = results
            children[child_id] = evaluated_child.copy()

        for gi in range(self.n_generations):
            print(f"Running generation {gi}")
            # --- STEP 1: Select n groups of candidates with group size of t_size,
            # choose the best one from each group as parent
            parents = self.population.tournamentSelect(self.t_size, self.t_selections)

            # --- STEP 2: Create one child for all possible pairs of parents
            children = []
            for pi in range(self.t_selections):
                for pj in range(pi + 1, self.t_selections):
                    children.append(ParamCombination.crossover(parents[pi], parents[pj]))

            # --- STEP 3: mutate on all children and evaluate them
            # this step requires the most time, so apply multiprocessing to it
            # start evaluation for many children at the same time
            pool = Pool(n_processes)
            for ci, child in enumerate(children):
                child.mutate(self.mutation_factor)
                pool.apply_async(evaluationSubprocess, [child.copy(), ci],
                                 callback=onSubprocessFinished, error_callback=onSubprocessError)

            # Code after these 2 lines must wait until all subprocesses are done, all children are evaluated
            # The fitness for every child will be put into children_fitness
            pool.close()
            pool.join()

            # STEP 4: Re-evaluate the best one, because the evaluation result may vary due to randomness
            # An evaluation may report extremely good result once,
            # but report not very good results if evaluated many times
            best_idx = self.population.getArgBest()
            self.population[best_idx].evaluate()

            # STEP 5: put all m children into population, and remove m worst instance
            # the population size should remain the same
            n_children = len(children)
            self.population.extend(children)
            for _ in range(n_children):
                self.population.pop(self.population.getArgWorst())

            self.reportBest()

    def reportBest(self):
        best_param_comb = self.population[self.population.getArgBest()]
        print("Best =", best_param_comb)

    def save(self, path: str):
        state_dict = {
            "args": {
                "population_size": 0,
                "tournament_size": self.t_size,
                "tournament_selections": self.t_selections,
                "mutation_factor": self.mutation_factor,
                "n_generations": self.n_generations
            },
            "population": [param_comb.toList() for param_comb in self.population]
        }

        with open(path, "wb") as out_file:
            pickle.dump(state_dict, out_file)
        print(f"ParameterFinder is saved to {path}")

    @staticmethod
    def load(path: str) -> OneMaxSolverParamFinder:
        with open(path, "rb") as in_file:
            state_dict = pickle.load(in_file)
        param_finder = OneMaxSolverParamFinder(**state_dict["args"])
        param_finder.population.extend(
            [ParamCombination.fromList(param_list) for param_list in state_dict["population"]]
        )
        return param_finder


    def evaluatePopulation(self, n_processes: int = 5, n_evals: int = 10):
        param_indices = []
        param_fitness = []
        for i in range(len(self.population)):
            param_indices.extend([i] * n_evals)
            param_fitness.extend([0] * n_evals)

        idx = 0
        def onSubprocessFinish(results):
            param_comb, idx = results
            param_fitness[idx] = param_comb.fitness

        for pi, param_comb in enumerate(self.population):
            print(f"Evaluating param_comb {pi+1}/{len(self.population)}")
            p = Pool(n_processes)
            for ei in range(n_evals):
                p.apply_async(evaluationSubprocess, [param_comb, idx], callback=onSubprocessFinish,
                              error_callback=onSubprocessError)
                idx += 1
            p.close()
            p.join()

        plt.figure(1, figsize=(16, 9))
        for i in range(0, n_evals * len(self.population), n_evals):
            avg = sum(param_fitness[i: i+n_evals]) / n_evals
            plt.scatter(param_indices[i: i+n_evals], param_fitness[i: i+n_evals], marker=f"${i//n_evals}$")
            plt.text(i/n_evals, avg, f"{avg:.5f}")

        plt.title("Multiple Fitness Evaluations on Population")
        plt.xlabel("ParamComb Index")
        plt.ylabel("ParamComb Fitness")
        plt.show()



if __name__ == '__main__':
    # param_finder = OneMaxSolverParamFinder(population_size=30,
    #                                        tournament_size=3,
    #                                        tournament_selections=3,
    #                                        mutation_factor=1.5,
    #                                        n_generations=100)
    # param_finder.run()
    # print("EA done")
    # param_finder.reportBest()
    # param_finder.save("EA.pkl")

    param_finder = OneMaxSolverParamFinder.load("EA_2022Dec13.pkl")
    # param_finder.evaluatePopulation(5, 10)

    # For now, EA_2022Dec13.pkl report an average evaluation iteration of 807, solve rate of 99%
    for i in range(10):
        param_finder.population[19].evaluate()
        print(param_finder.population[19])
