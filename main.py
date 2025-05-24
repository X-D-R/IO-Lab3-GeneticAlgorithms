import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
bench_path = os.path.join(project_root, 'plots')

data = pd.read_csv('report_diff.csv')
filtered_data = data[data['algorithm'] != 'DPWeights']

sns.set_style("whitegrid")
plt.figure(figsize=(12, 8))

# Plot 2: Execution time by benchmark and algorithm
plt.figure(figsize=(14, 8))
sns.barplot(x='bench id', y='time', hue='algorithm', data=filtered_data)
plt.title('Execution Time by Benchmark and Algorithm (seconds)')
plt.xlabel('Benchmark ID')
plt.ylabel('Time (seconds)')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(bench_path, 'execution_time.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Algorithm profit by benchmark and algorithm
plt.figure(figsize=(14, 8))
sns.barplot(x='bench id', y='alg profit', hue='algorithm', data=data)
plt.title('Algorithm Profit by Benchmark and Algorithm')
plt.xlabel('Benchmark ID')
plt.ylabel('Profit')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(bench_path, 'algorithm_profit.png'), dpi=300, bbox_inches='tight')
plt.close()