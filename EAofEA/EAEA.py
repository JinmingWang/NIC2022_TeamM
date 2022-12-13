from __future__ import annotations  # Requires Python 3.7+
from typing import *
import matplotlib.pyplot as plt
from OneMaxObject import BitString, Population
from PowerfulSolver import PowerfulSolver
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
        self.params = {
            "pop_size": random.randint(3, 15),
            "slope_m": getRandomBetween(-0.002, 0),
            "init_m": getRandomBetween(0.0, 3.0),
            "slope_c": getRandomBetween(-0.002, 0),
            "init_c": getRandomBetween(0.0, 3.0),
            "t_size": random.randint(1, 5),
            "t_n_select": 2,
            "min_mutation": getRandomBetween(0, 0.1)
        }

        self.fitness = 9999
        self.avg_solved_at = 0
        self.n_solves = 0

    def __getitem__(self, item):
        return self.params[item]

    def __setitem__(self, key, value):
        self.params[key] = value

    def __str__(self):
        return str(self.params) + f"\n{self.fitness=} {self.avg_solved_at=} {self.n_solves=}".replace("self.", "")

    def mutate(self, mutation_factor: float) -> None:

        # It is not good to change population size a lot, because it has big influence on algorithm performance,
        # so change population size with low probability
        if random.random() < 0.4:
            self["pop_size"] += 1 if random.random() >= 0.5 else -1

        self["slope_m"] = self["slope_m"] + getRandomBetween(-1e-5 * mutation_factor, 1e-5 * mutation_factor)

        self["init_m"] = self["init_m"] + getRandomBetween(-0.01 * mutation_factor, 0.01 * mutation_factor)

        self["slope_c"] = self["slope_c"] + getRandomBetween(-1e-5 * mutation_factor, 1e-5 * mutation_factor)

        self["init_c"] = self["init_c"] + getRandomBetween(-0.01 * mutation_factor, 0.01 * mutation_factor)

        if random.random() < 0.2:
            if random.random() >= 0.5:
                self["t_size"] += 1
            else:
                self["t_size"] -= 1
            self["t_size"] = clip(self["t_size"], 1, self["pop_size"]//2)

        if random.random() < 0.2:
            self["t_n_select"] += int(clip(1 if random.random() >= 0.5 else -1, 2, 4))

        self["min_mutation"] = clip(self["min_mutation"] +
                                     getRandomBetween(-0.005 * mutation_factor, 0.005 * mutation_factor), 0, 1)

    def evaluate(self) -> None:
        """
        Compute the fitness of this set of parameters, the fitness is computed by actually run the algorithm many times,

        We have fitness_solve, it rewards high solving rate, because we should have a solve rate of > 95%
        We do not want it to increase even more, because increasing fitness_solve means decreasing fitness_worst
        So the maximum value for fitness_solve is 0.96
        fitness_solve = 2 *  (n_solves / n_tests)

        We have fitness_worst, it rewards high average solution found generation
        fitness_worst = avg_solved_at / N_GENERATIONS

        And we have fitness_overall, which is a weighted combine of both fitness functions
        fitness_solve has a factor of 2, because solving the problem is more important than having good performance
        fitness_overall = 2 * fitness_solve + fitness_worst

        If n_solves < 95
        Then the fitness if halfed
        """

        self.n_solves = 0
        self.avg_solved_at = 0

        mutation_rate_func = lambda gen_idx: max(self["slope_m"] * gen_idx + self["init_m"], self["min_mutation"])
        crossover_rate_func = lambda gen_idx: max(self["slope_c"] * gen_idx + self["init_c"], self["min_mutation"])

        n_tests = 100
        for i in range(n_tests):
            BitString.n_fitness_evals = 0
            doMutatation=False
            doCrossOver=True
            solver = PowerfulSolver(SIZE, self["pop_size"], mutation_rate_func, self["t_size"], self["t_n_select"],
                                    crossover_rate_func, N_GENERATIONS,doMutatation,doCrossOver)
            best_fitness, best_found_at,average_fitness = solver.run()
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
        if fitness_solve >= 0.95:
            weight = 2 - (1 - fitness_solve) * 20
            self.fitness = 2 * fitness_solve + weight * fitness_worst
        else:
            self.fitness = 2 * fitness_solve + fitness_worst

    def copy(self):
        new_param = ParamCombination()
        new_param.params = self.params
        new_param.fitness = self.fitness
        new_param.n_solves = self.n_solves
        new_param.avg_solved_at = self.avg_solved_at
        return new_param

    def toList(self):
        return [self.params, self.fitness, self.n_solves, self.avg_solved_at]

    @staticmethod
    def fromList(data_list: List):
        new_param = ParamCombination()
        new_param.params, new_param.fitness, new_param.n_solves, new_param.avg_solved_at = data_list
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

        def selectOrMerge(p1, p2, key):
            choice = random.choice([0, 1, 2])
            if choice == 0:
                return p1[key]
            elif choice == 1:
                return (p1[key] + p2[key]) / 2
            else:
                return p2[key]

        new_param["pop_size"] = int(selectOrMerge(param1, param2, "pop_size"))
        new_param["init_m"] = selectOrMerge(param1, param2, "init_m")
        new_param["slope_m"] = selectOrMerge(param1, param2, "slope_m")
        new_param["init_c"] = selectOrMerge(param1, param2, "init_c")
        new_param["slope_c"] = selectOrMerge(param1, param2, "slope_c")
        new_param["t_size"] = int(selectOrMerge(param1, param2, "t_size"))
        new_param["t_n_select"] = int(selectOrMerge(param1, param2, "t_n_select"))
        new_param["min_mutation"] = selectOrMerge(param1, param2, "min_mutation")

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
        self.history = []

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
            self.save(f"EAEA_g{gi}.pkl")

    def reportBest(self):
        best_param_comb = self.population[self.population.getArgBest()]
        self.history.append(best_param_comb.fitness)
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
        """

        :param n_processes: number of processes to use
        :param n_evals: number of evaluations
        :return:
        """
        param_indices = []
        param_fitness = []
        for i in range(len(self.population)):
            param_indices.extend([i] * n_evals)
            param_fitness.extend([0] * n_evals)

        idx = 0
        def onSubprocessFinish(results):
            param_comb, i = results
            # print(f"Fitness of population[{i}] = {param_comb.fitness}")
            # print(f"Population[{i}] = {param_comb}")
            param_fitness[i] = param_comb.fitness

        for pi, param_comb in enumerate(self.population):
            print(f"Evaluating param_comb {pi+1}/{len(self.population)}")
            p = Pool(n_processes)
            for ei in range(n_evals):
                p.apply_async(evaluationSubprocess, [param_comb, idx], callback=onSubprocessFinish,
                              error_callback=onSubprocessError)
                idx += 1
            p.close()
            p.join()

        population_evals = []
        for i in range(0, n_evals * len(self.population), n_evals):
            param_comb_i = i//n_evals
            # average fitness of this set of parameters
            avg_fitness = sum(param_fitness[i: i+n_evals]) / n_evals
            min_fitness = min(param_fitness[i: i+n_evals])
            max_fitness = max(param_fitness[i: i+n_evals])

            population_evals.append((param_comb_i, avg_fitness, min_fitness, max_fitness, self.population[param_comb_i]))

        # sort population according to fitness values, and print from the highest fitness one to the lowest fitness one
        population_evals = sorted(population_evals, key=lambda item: item[1], reverse=True)
        for item_eval in population_evals:
            print(item_eval[0:4], end=": ")
            print(item_eval[4])

        #     plt.scatter(param_indices[i: i+n_evals], param_fitness[i: i+n_evals], marker=f"${param_comb_i}$")
        #     plt.text(i/n_evals, avg_fitness, f"{avg_fitness:.5f}")
        #
        # plt.title("Multiple Fitness Evaluations on Population")
        # plt.xlabel("ParamComb Index")
        # plt.ylabel("ParamComb Fitness")
        # plt.show()



if __name__ == '__main__':
    # param_finder = OneMaxSolverParamFinder(population_size=20,
    #                                        tournament_size=2,
    #                                        tournament_selections=3,
    #                                        mutation_factor=4.0,
    #                                        n_generations=30)
    param_finder = OneMaxSolverParamFinder.load('EAEA_g0.pkl')
    param_finder.run(n_processes=8)
    plt.plot(range(30), param_finder.history)
    plt.title("Convergence Plot of EAofEA")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show()

    # Load and continue the algorithm
    # param_finder = OneMaxSolverParamFinder.load("EAEA_g30.pkl")
    # param_finder.run(8)


    # param_finder = OneMaxSolverParamFinder.load("EAEA_g30.pkl")
    # param_finder.evaluatePopulation(8, 8)

