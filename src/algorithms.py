from functools import wraps
from itertools import combinations
import os
from time import time
import random
from typing import List, Tuple
import numpy as np

def measure_time(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time()
        result = func(self, *args, **kwargs)
        end = time()
        self.execution_time = round(end - start, 6)
        return result

    return wrapper


class KnapsackData:
    def __init__(self, capacity: int, weights: List[int], values: List[int], optimal_weights: List[int]):
        self.capacity = capacity
        self.weights = weights
        self.values = values
        self.optimal_weights = optimal_weights


class Algorithm:
    def __init__(self, data: KnapsackData):
        self.capacity = data.capacity
        self.weights = data.weights
        self.values = data.values
        self.execution_time = None
        self.inter_solutions = 0

    def get_total_value(self, result: List[int]) -> int:
        return sum(res * value for value, res in zip(self.values, result))

    def get_total_weight(self, result: List[int]) -> int:
        return sum(res * weight for weight, res in zip(self.weights, result))

    def solve(self) -> List[int]:
        pass

    @measure_time
    def __call__(self) -> List[int]:
        self.execution_time = 0
        self.inter_solutions = 0
        return self.solve()

    def get_execution_time(self) -> float:
        return self.execution_time


class GeneticAlgorithm(Algorithm):
    def __init__(self, data: KnapsackData,
                 population_size: int = 100,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism: bool = True):
        super().__init__(data)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.num_items = len(self.weights)
        self.best_individual = None
        self.best_fitness = -1

    def initialize_population(self) -> List[List[int]]:
        """Инициализация популяции"""
        population = []
        for _ in range(self.population_size):
            while True:
                individual = [random.randint(0, 1) for _ in range(self.num_items)]
                if self.get_total_weight(individual) <= self.capacity:
                    population.append(individual)
                    break
        return population

    def fitness(self, individual: List[int]) -> int:
        """Функция приспособленности"""
        total_weight = self.get_total_weight(individual)
        if total_weight > self.capacity:
            return 0

        total_value = self.get_total_value(individual)

        if total_value > self.best_fitness:
            self.best_fitness = total_value
            self.best_individual = individual.copy()

        return total_value

    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Турнирная селекция"""
        tournament = random.choices(population, k=self.tournament_size)
        return max(tournament, key=lambda ind: self.fitness(ind)).copy()

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Одноточечный кроссовер"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, self.num_items - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutate(self, individual: List[int]) -> List[int]:
        """Мутация"""
        for i in random.sample(range(self.num_items), k=int(self.mutation_rate * self.num_items)):
            individual[i] ^= 1

        if self.get_total_weight(individual) > self.capacity:
            ones = [i for i, val in enumerate(individual) if val == 1]
            while ones and self.get_total_weight(individual) > self.capacity:
                individual[ones.pop()] = 0

        return individual

    def evolve(self, population: List[List[int]]) -> List[List[int]]:
        """Эволюция"""
        new_population = []

        if self.elitism and self.best_individual is not None:
            new_population.append(self.best_individual.copy())

        target_size = self.population_size - (1 if self.elitism else 0)
        while len(new_population) < target_size:
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))

            if len(new_population) < target_size:
                new_population.append(self.mutate(child2))

        return new_population

    def solve(self) -> List[int]:
        """Метод решения"""
        population = self.initialize_population()

        self.best_individual = max(population, key=lambda ind: self.fitness(ind))

        for _ in range(self.generations):
            population = self.evolve(population)
            self.inter_solutions += self.population_size

        return self.best_individual


class GeneticAlgorithmTwoApprox(GeneticAlgorithm):
    def __init__(self, data: KnapsackData,
                 population_size: int = 100,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism: bool = True):
        super().__init__(data, population_size, generations, crossover_rate, mutation_rate, tournament_size, elitism)
        self.alg = TwoApproxAlgorithm(data)

    def initialize_population(self) -> List[List[int]]:
        """Инициализация популяции"""
        item = self.alg.solve()
        population = [item]*self.population_size
        return population


class GeneticAlgorithmGreed(GeneticAlgorithm):
    def __init__(self, data: KnapsackData,
                 population_size: int = 100,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism: bool = True):
        super().__init__(data, population_size, generations, crossover_rate, mutation_rate, tournament_size, elitism)
        self.alg = TwoApproxAlgorithm(data)

    def initialize_population(self) -> List[List[int]]:
        """Полностью жадная инициализация популяции с вариациями"""
        population = []
        sorted_items = sorted(
            [(i, self.values[i] / self.weights[i]) for i in range(self.num_items)],
            key=lambda x: -x[1]
        )

        for _ in range(self.population_size):
            individual = [0] * self.num_items
            remaining_capacity = self.capacity

            for i, _ in sorted_items:
                if self.weights[i] <= remaining_capacity:
                    individual[i] = 1
                    remaining_capacity -= self.weights[i]

            if random.random() < 0.3:
                ones = [i for i, val in enumerate(individual) if val == 1]
                if ones:
                    individual[random.choice(ones)] = 0

            population.append(individual)

        return population


class TwoApproxAlgorithm(Algorithm):
    def greed_search(self):
        qualities = {i: value / weight for i, weight in enumerate(self.weights) for value in self.values}
        qualities = dict(sorted(qualities.items(), key=lambda quality: quality[1], reverse=True))

        sum_weights = 0
        result = [0] * len(self.weights)
        for item in qualities.keys():
            if sum_weights + self.weights[item] <= self.capacity:
                sum_weights += self.weights[item]
                result[item] = 1
                self.inter_solutions += 1
            if sum_weights == self.capacity:
                break

        return result

    def max_greed_search(self):
        sorted_values = sorted(self.values, reverse=True)

        for i, value in enumerate(sorted_values):
            self.inter_solutions += 1
            if self.weights[i] <= self.capacity:
                return i, self.values[i]

    def solve(self) -> List[int]:
        self.inter_solutions = 0
        greed_result = self.greed_search()
        max_greed_result, max_greed_value = self.max_greed_search()
        greed_value = self.get_total_value(greed_result)
        return greed_result if greed_value > max_greed_value else max_greed_result


class DPWeights(Algorithm):
    def dp_table(self) -> np.ndarray:
        n = len(self.weights)
        W = self.capacity
        table = np.zeros((n + 1, W + 1), dtype=object)

        for i in range(n + 1):
            for w in range(W + 1):
                if i == 0 or w == 0:
                    table[i][w] = (0, 0)
                elif self.weights[i - 1] > w:
                    table[i][w] = (table[i - 1][w][0], w)
                else:
                    w_ind = w - self.weights[i - 1]
                    take = self.values[i - 1] + table[i - 1][w_ind][0]
                    no_take = table[i - 1][w][0]
                    if take > no_take:
                        table[i][w] = (take, w_ind)
                    else:
                        table[i][w] = (no_take, w)
                    self.inter_solutions += 1
        return table

    def get_weight_by_table(self, table: np.ndarray) -> List[int]:
        result = [0] * len(self.weights)

        w = self.capacity
        for i in range(len(self.weights), 0, -1):
            _, pre_w = table[i][w]
            if pre_w != w:
                result[i - 1] = 1
            w = pre_w
        return result

    def solve(self) -> List[int]:
        self.inter_solutions = 0
        table = self.dp_table()
        result = self.get_weight_by_table(table)
        return result


class BranchAndBound(Algorithm):
    def solve(self) -> List[int]:
        self.inter_solutions = 0
        n = len(self.weights)

        items = sorted([(self.values[i], self.weights[i], i)
                        for i in range(n)],
                       key=lambda x: x[0] / x[1], reverse=True)

        best_value = 0
        best_solution = [0] * n
        current_solution = [0] * n

        def bound(i, current_weight, current_value):
            remaining_weight = self.capacity - current_weight
            bound_value = current_value
            j = i

            while j < n and remaining_weight > 0:
                if items[j][1] <= remaining_weight:
                    bound_value += items[j][0]
                    remaining_weight -= items[j][1]
                else:
                    bound_value += items[j][0] * (remaining_weight / items[j][1])
                    remaining_weight = 0
                j += 1
            return bound_value

        def backtrack(i, current_weight, current_value):
            nonlocal best_value, best_solution
            self.inter_solutions += 1

            if current_weight <= self.capacity and current_value > best_value:
                best_value = current_value
                best_solution = current_solution.copy()

            if i == n:
                return

            if bound(i, current_weight, current_value) > best_value:
                if current_weight + items[i][1] <= self.capacity:
                    original_idx = items[i][2]
                    current_solution[original_idx] = 1
                    backtrack(i + 1, current_weight + items[i][1], current_value + items[i][0])
                    current_solution[original_idx] = 0

                backtrack(i + 1, current_weight, current_value)

        backtrack(0, 0, 0)

        result = [0] * n
        for i in range(n):
            original_idx = items[i][2]
            result[original_idx] = best_solution[original_idx]

        return result


class PTAS(Algorithm):
    def solve(self) -> List[int]:
        self.inter_solutions = 0
        epsilon = 0.5  # Approximation parameter (0 < ε ≤ 1)

        n = len(self.values)
        if n == 0:
            return []

        items = sorted([(self.values[i], self.weights[i], i)
                        for i in range(n)],
                       key=lambda x: x[0] / x[1], reverse=True)

        m = min(int(np.ceil(1 / epsilon)), n)

        best_value = 0
        best_solution = [0] * n

        for k in range(0, m + 1):
            for subset in combinations(items, k):
                self.inter_solutions += 1

                subset_weight = sum(item[1] for item in subset)
                subset_value = sum(item[0] for item in subset)

                if subset_weight > self.capacity:
                    continue

                current_solution = [0] * n
                for item in subset:
                    current_solution[item[2]] = 1

                remaining_capacity = self.capacity - subset_weight
                current_value = subset_value

                for item in items:
                    if item not in subset and item[1] <= remaining_capacity:
                        current_solution[item[2]] = 1
                        current_value += item[0]
                        remaining_capacity -= item[1]

                if current_value > best_value:
                    best_value = current_value
                    best_solution = current_solution.copy()

        return best_solution


class FilesKnapsack:
    def __init__(self, capacity_file: str, weights_file: str, values_file: str, optimal_weights_file: str):
        self.capacity_file = capacity_file
        self.weights_file = weights_file
        self.values_file = values_file
        self.optimal_weights_file = optimal_weights_file


def read_knapsack_data(files: FilesKnapsack) -> KnapsackData:
    with open(files.capacity_file, "r") as f:
        capacity = int(f.readline())
    with open(files.weights_file, "r") as f:
        weights = list(map(int, f.readlines()))
    with open(files.values_file, "r") as f:
        values = list(map(int, f.readlines()))
    with open(files.optimal_weights_file, "r") as f:
        optimal_weights = list(map(int, f.readlines()))

    return KnapsackData(capacity, weights, values, optimal_weights)


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    bench_path = os.path.join(project_root, 'benchmarks', 'p02')

    capacity_file = os.path.join(bench_path, 'p02_c.txt')
    weights_file = os.path.join(bench_path, 'p02_w.txt')
    values_file = os.path.join(bench_path, 'p02_p.txt')
    optimal_weights_file = os.path.join(bench_path, 'p02_s.txt')
    files = FilesKnapsack(capacity_file, weights_file, values_file, optimal_weights_file)

    data = read_knapsack_data(files)
    print(GeneticAlgorithm(data)())
    print(data.optimal_weights)
