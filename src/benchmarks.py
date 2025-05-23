import os
from typing import Type, Tuple
import pandas as pd
from src.algorithms import GeneticAlgorithm, read_knapsack_data, Algorithm, FilesKnapsack


class Benchmark:
    def __init__(self, algorithm_classes: Tuple[Type[Algorithm]], runs: int = 1):
        self.algorithm_classes = algorithm_classes
        self.runs = runs  # Количество запусков для усреднения результатов

    def run_all_benchmarks(self):
        results = []
        results_diff = []
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        bench_base = os.path.join(project_root, 'benchmarks')

        for bench_id in range(1, 8):
            print('-' * 20)
            print(f'Benchmark #{bench_id}')
            bench_path = os.path.join(bench_base, f'p0{bench_id}')
            self.run_one_benchmark(bench_path, results, results_diff, bench_id)

        self.save_results(results, results_diff, project_root)

    def run_one_benchmark(self, bench_path: str, results: list, results_diff: list, bench_id: int):
        files = FilesKnapsack(
            os.path.join(bench_path, f'p0{bench_id}_c.txt'),
            os.path.join(bench_path, f'p0{bench_id}_w.txt'),
            os.path.join(bench_path, f'p0{bench_id}_p.txt'),
            os.path.join(bench_path, f'p0{bench_id}_s.txt')
        )
        data = read_knapsack_data(files)

        for algorithm_class in self.algorithm_classes:
            stats = self.run_algorithm(algorithm_class, data)

            print(f"{algorithm_class.__name__}: {stats['percentage_difference']}%")

            results.append({
                'bench id': bench_id,
                'algorithm': algorithm_class.__name__,
                'time': stats['avg_time'],
                'number of inter solutions': stats['inter_solutions'],
                'alg weights': stats['actual_weights'],
                'alg total weight': stats['actual_total_weight'],
                'alg profit': stats['actual_value']
            })

            results_diff.append({
                'bench id': bench_id,
                'algorithm': algorithm_class.__name__,
                'time': stats['avg_time'],
                'number of inter solutions': stats['inter_solutions'],
                'alg weights': stats['actual_weights'],
                'expected weights': data.optimal_weights,
                'capacity': data.capacity,
                'alg total weight': stats['actual_total_weight'],
                'expected total weight': stats['expected_total_weight'],
                'alg profit': stats['actual_value'],
                'expected profit': stats['expected_value'],
                'profit difference': stats['actual_difference'],
                'percentage profit difference': stats['percentage_difference']
            })

    def run_algorithm(self, algorithm_class: Type[Algorithm], data: FilesKnapsack) -> dict:
        total_time = 0
        best_result = None
        best_value = -1

        for _ in range(self.runs):
            algorithm = algorithm_class(data)
            result = algorithm()
            total_time += algorithm.execution_time

            current_value = algorithm.get_total_value(result)
            if current_value > best_value:
                best_value = current_value
                best_result = result
                inter_solutions = algorithm.inter_solutions

        avg_time = total_time / self.runs

        expected_value = algorithm.get_total_value(data.optimal_weights)
        actual_difference = expected_value - best_value
        percentage_difference = round((actual_difference / expected_value) * 100, 4) if expected_value != 0 else 0

        return {
            'avg_time': avg_time,
            'inter_solutions': inter_solutions,
            'actual_weights': best_result,
            'actual_total_weight': algorithm.get_total_weight(best_result),
            'actual_value': best_value,
            'expected_total_weight': algorithm.get_total_weight(data.optimal_weights),
            'expected_value': expected_value,
            'actual_difference': actual_difference,
            'percentage_difference': percentage_difference
        }

    def save_results(self, results: list, results_diff: list, output_dir: str):
        data = pd.DataFrame(results).sort_values(by=['bench id', 'algorithm'])
        data_diff = pd.DataFrame(results_diff).sort_values(by=['bench id', 'algorithm'])

        print("\nSummary Results:")
        print(data)

        data.to_csv(os.path.join(output_dir, 'report_single_run.csv'), index=False)
        data_diff.to_csv(os.path.join(output_dir, 'report_diff_single_run.csv'), index=False)


if __name__ == '__main__':
    custom_ga_params = {
        'population_size': 100,
        'generations': 100,
        'crossover_rate': 0.85,
        'mutation_rate': 0.05,
        'tournament_size': 3,
        'elitism': True
    }

    class CustomGA(GeneticAlgorithm):
        def __init__(self, data):
            super().__init__(data, **custom_ga_params)


    benchmark = Benchmark(algorithm_classes=(CustomGA,), runs=1)
    benchmark.run_all_benchmarks()