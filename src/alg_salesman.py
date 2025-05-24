import math
import random
import time
from typing import List


class SalesmanData:
    def __init__(self, path: str):
        self.dimension = 0
        self.weight_type = None
        self.weight_format = None
        self.coords = []
        self.matrix = []
        self.load_file(path)

    def load_file(self, path: str):
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]

        is_coords = False
        is_matrix = False
        flat_matrix = []

        for line in lines:
            if 'DIMENSION' in line:
                self.dimension = int(line.split(":")[-1])
            elif 'EDGE_WEIGHT_TYPE' in line:
                self.weight_type = line.split(":")[-1].strip()
            elif 'EDGE_WEIGHT_FORMAT' in line:
                self.weight_format = line.split(":")[-1].strip()
            elif 'NODE_COORD_SECTION' in line:
                is_coords = True
                continue
            elif 'EDGE_WEIGHT_SECTION' in line:
                is_coords = False
                is_matrix = True
                continue
            elif 'EOF' in line:
                break

            if is_coords:
                parts = line.split()
                if len(parts) >= 3:
                    self.coords.append((float(parts[1]), float(parts[2])))
            elif is_matrix:
                flat_matrix.extend(map(int, line.split()))

        if self.weight_type in ('EUC_2D', 'ATT') and self.coords:
            self.make_matrix_from_coords()
        elif self.weight_type == 'EXPLICIT':
            if self.weight_format == 'FULL_MATRIX':
                self.make_matrix_from_full(flat_matrix)
            elif self.weight_format == 'LOWER_DIAG_ROW':
                self.make_matrix_from_lower(flat_matrix)
        else:
            raise ValueError(f'Ошибка, нет такого типа {self.weight_type}')

    def dist_euc(self, a, b):
        return round(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def dist_att(self, a, b):
        dist = math.sqrt(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / 10.0)
        dist_r = round(dist)
        return dist_r + 1 if dist_r < dist else dist_r

    def make_matrix_from_coords(self):
        self.matrix = [[0] * self.dimension for _ in range(self.dimension)]
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    self.matrix[i][j] = 0
                    continue
                dist = 0
                if self.weight_type == 'EUC_2D':
                    dist = self.dist_euc(self.coords[i], self.coords[j])
                elif self.weight_type == 'ATT':
                    dist = self.dist_att(self.coords[i], self.coords[j])
                self.matrix[i][j] = dist

    def make_matrix_from_full(self, flat_matrix):
        self.matrix = []
        idx = 0
        for _ in range(self.dimension):
            row = flat_matrix[idx:idx + self.dimension]
            self.matrix.append(row)
            idx += self.dimension

    def make_matrix_from_lower(self, flat_matrix):
        self.matrix = [[0] * self.dimension for _ in range(self.dimension)]
        idx = 0
        for i in range(self.dimension):
            for j in range(i + 1):
                dist = flat_matrix[idx]
                self.matrix[i][j], self.matrix[j][i] = dist, dist
                idx += 1

    def get_matrix(self):
        return self.matrix


class SalesmanAlgorithm:
    def __init__(self, matrix: List[List[int]], population_size: int = 100, generations: int = 1000,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.2, tournament_size: int = 5,
                 elitism: bool = True, time_limit_sec: int = 300):
        self.matrix = matrix
        self.N = len(matrix)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.start_time = None
        self.time_limit = time_limit_sec

    def initialize_population(self) -> List[List[int]]:
        base = list(range(self.N))
        return [random.sample(base, self.N) for _ in range(self.population_size)]

    def total_distance(self, tour: List[int]) -> float:
        return sum(self.matrix[tour[i]][tour[(i + 1) % self.N]] for i in range(self.N))

    def fitness(self, tour: List[int]) -> float:
        return -self.total_distance(tour)

    def tournament_selection(self, population: List[List[int]]) -> List[int]:
        tournament = random.choices(population, k=self.tournament_size)
        return max(tournament, key=lambda ind: self.fitness(ind)).copy()

    def crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        start, end = sorted(random.sample(range(self.N), 2))
        child = [None] * self.N
        child[start:end] = parent1[start:end]

        mapping = dict(zip(parent1[start:end], parent2[start:end]))
        for i in range(self.N):
            if i >= start and i < end:
                continue
            gene = parent2[i]
            while gene in mapping and gene in child:
                gene = mapping[gene]
            child[i] = gene
        return child

    def mutate(self, tour: List[int]) -> None:
        for i in range(self.N):
            if random.random() < self.mutation_rate:
                j = random.randint(0, self.N - 1)
                tour[i], tour[j] = tour[j], tour[i]
                break

    def evolve(self, population: List[List[int]]) -> List[List[int]]:
        new_population = []
        if self.elitism:
            best = max(population, key=self.fitness)
            new_population.append(best[:])
        while len(new_population) < self.population_size:
            p1 = self.tournament_selection(population)
            p2 = self.tournament_selection(population)
            child = self.crossover(p1, p2) if random.random() < self.crossover_rate else p1[:]
            self.mutate(child)
            new_population.append(child)
        return new_population

    def solve(self):
        self.start_time = time.time()
        population = self.initialize_population()
        best_solution = min(population, key=self.total_distance)
        best_distance = self.total_distance(best_solution)

        for _ in range(self.generations):
            if time.time() - self.start_time > self.time_limit:
                break
            population = self.evolve(population)
            candidate = min(population, key=self.total_distance)
            dist = self.total_distance(candidate)
            if dist < best_distance:
                best_solution, best_distance = candidate, dist

        exec_time = time.time() - self.start_time
        return best_solution, best_distance, exec_time


if __name__ == '__main__':
    data = SalesmanData('benchmarks/salesman/att48.tsp')
    mtx = data.matrix
    alg = SalesmanAlgorithm(data.get_matrix(), generations=1000, time_limit_sec=300)

    best_tour, distance, duration = alg.solve()
    print('Best distance:', distance)
    print('Best tour:', best_tour)
    print('Execution time:', round(duration, 2), 'sec')
