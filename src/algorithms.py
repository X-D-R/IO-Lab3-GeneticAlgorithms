from functools import wraps
import os
from time import time
import random
from typing import List, Tuple


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
                 population_size: int = 50,
                 generations: int = 50,
                 crossover_rate: float = 0.9,
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

    def initialize_population(self) -> List[List[int]]:
        """Создает начальную популяцию случайных решений"""
        population = []
        for _ in range(self.population_size):
            individual = [random.randint(0, 1) for _ in range(self.num_items)]
            while self.get_total_weight(individual) > self.capacity:
                individual = [random.randint(0, 1) for _ in range(self.num_items)]
            population.append(individual)
        return population

    def fitness(self, individual: List[int]) -> float:
        """Функция приспособленности"""
        total_weight = self.get_total_weight(individual)
        total_value = self.get_total_value(individual)

        if total_weight <= self.capacity:
            return total_value
        else:
            penalty = (total_weight - self.capacity) / self.capacity
            return max(0.0, total_value * (1 - penalty))

    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        """Турнирная селекция"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda ind: self.fitness(ind))
        return winner.copy()

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Двухточечный кроссовер"""
        child1, child2 = [], []
        for p1, p2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child1.append(p1)
                child2.append(p2)
            else:
                child1.append(p2)
                child2.append(p1)
        return child1, child2

    def mutate(self, individual: List[int]) -> List[int]:
        """Мутация - инвертирование бита"""
        for i in range(self.num_items):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]

        # Исправляем решение, если оно стало недопустимым
        while self.get_total_weight(individual) > self.capacity:
            # Отключаем случайный предмет
            ones = [i for i, val in enumerate(individual) if val == 1]
            if not ones:
                break
            individual[random.choice(ones)] = 0

        return individual

    def evolve(self, population: List[List[int]]) -> List[List[int]]:
        """Создает новое поколение"""
        new_population = []
        if self.elitism:
            best_individual = max(population, key=lambda ind: self.fitness(ind))
            new_population.append(best_individual.copy())
            start_index = 1
        else:
            start_index = 0
        for i in range(start_index, self.population_size, 2):
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return new_population

    def solve(self) -> List[int]:
        """Основной метод решения задачи"""
        population = self.initialize_population()
        best_fitness = -1
        no_improvement = 0
        max_no_improvement = 15

        for generation in range(self.generations):
            population = self.evolve(population)
            current_best = max(population, key=lambda x: self.fitness(x))
            current_fitness = self.fitness(current_best)

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= max_no_improvement:
                    break
        best_solution = max(population, key=lambda ind: self.fitness(ind))
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
    print(GeneticAlgorithm(data).solve())
    print(data.optimal_weights)
