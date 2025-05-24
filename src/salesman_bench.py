import os
import time
import pandas as pd
from src.alg_salesman import SalesmanData, SalesmanAlgorithm
from typing import Type, List, Tuple


class TSPBenchmark:
    def __init__(self, algorithm, bench_dir: str, runs: int = 3):
        self.algorithm = algorithm
        self.runs = runs
        self.bench_dir = bench_dir
        self.results = []

    def run_all(self):
        files = [f for f in os.listdir(self.bench_dir) if f.endswith('.tsp')]
        for file in sorted(files):
            print(f'Benchmarking {file} ...')
            full_path = os.path.join(self.bench_dir, file)
            self.run_one(full_path, file)

        df = pd.DataFrame(self.results)
        df.to_csv('salesman_results.csv', index=False)
        print(df)

    def run_one(self, path: str, benchmark: str):
        data = SalesmanData(path)
        matrix = data.get_matrix()

        best_distance = float('inf')
        total_time = 0.0
        best_tour = None

        for _ in range(self.runs):
            alg = self.algorithm(matrix)
            tour, distance, duration = alg.solve()
            total_time += duration
            if distance < best_distance:
                best_distance = distance
                best_tour = tour

        avg_time = total_time / self.runs
        self.results.append({
            'benchmark': benchmark,
            'best_distance': round(best_distance, 3),
            'avg_time_sec': round(avg_time, 3),
            'tour': best_tour
        })


if __name__ == '__main__':
    benchmark = TSPBenchmark(SalesmanAlgorithm, 'benchmarks/salesman',3)
    benchmark.run_all()
