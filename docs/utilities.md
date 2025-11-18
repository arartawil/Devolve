# DEvolve Utilities Module

Comprehensive utilities for optimization experiments, statistical analysis, and result management.

## Overview

The utilities module provides:
- **Performance Metrics** - Calculate success rate, ERT, convergence speed
- **Statistical Tests** - Wilcoxon, Friedman, Nemenyi, effect sizes
- **Parallel Execution** - Speed up evaluations and multiple runs
- **Seed Management** - Ensure reproducibility
- **I/O Functions** - Save, load, and export results

---

## 1. Performance Metrics (`metrics.py`)

### Calculate Error

```python
from devolve.utils import calculate_error

error = calculate_error(best_fitness=0.001, optimal_value=0.0)
print(f"Error: {error:.6f}")  # 0.001000
```

### Success Rate

Fraction of runs achieving target error:

```python
from devolve.utils import calculate_success_rate

runs = [0.001, 0.005, 0.1, 0.002]  # Best fitness from each run
sr = calculate_success_rate(runs, target_error=0.01, optimal_value=0.0)
print(f"Success rate: {sr:.2%}")  # 75.00%
```

### Expected Running Time (ERT)

Average function evaluations accounting for failures:

```python
from devolve.utils import calculate_ert

runs = [
    (0.001, 1000),  # (fitness, function_evaluations)
    (0.005, 1500),
    (0.1, 2000),
    (0.002, 1200)
]
ert = calculate_ert(runs, target_error=0.01, optimal_value=0.0)
print(f"ERT: {ert:.0f} FEs")
```

**Formula**: ERT = mean(FEs for successful runs) / success_rate

### Convergence Speed

Analyze optimization progress:

```python
from devolve.utils import calculate_convergence_speed

history = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
metrics = calculate_convergence_speed(history, target_error=0.01)

print(f"Iterations to target: {metrics['iterations_to_target']}")
print(f"Average improvement: {metrics['avg_improvement_rate']:.2f}")
print(f"Linear rate: {metrics['linear_rate']:.4f}")
print(f"Final error: {metrics['final_error']:.6e}")
```

### Stability Analysis

```python
from devolve.utils import calculate_stability

runs = [0.001, 0.002, 0.0015, 0.0018, 0.0012]
stability = calculate_stability(runs)

print(f"Mean: {stability['mean']:.6f}")
print(f"Std: {stability['std']:.6f}")
print(f"CV: {stability['cv']:.4f}")  # Coefficient of variation
print(f"Median: {stability['median']:.6f}")
print(f"IQR: {stability['iqr']:.6f}")
```

### PerformanceMetrics Class

Convenient interface:

```python
from devolve.utils import PerformanceMetrics

metrics = PerformanceMetrics()
sr = metrics.success_rate(runs, target_error=0.01)
ert = metrics.ert(runs_with_fes, target_error=0.01)
stability = metrics.stability(runs)
```

---

## 2. Statistical Tests (`stats.py`)

### Wilcoxon Signed-Rank Test

Compare two paired algorithms (non-parametric paired t-test):

```python
from devolve.utils import wilcoxon_test

alg1_results = [0.01, 0.02, 0.015, 0.018, 0.012]
alg2_results = [0.05, 0.06, 0.055, 0.058, 0.052]

result = wilcoxon_test(alg1_results, alg2_results)
print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")  # True if p < 0.05
print(result['interpretation'])
```

### Friedman Test

Compare multiple algorithms (non-parametric ANOVA):

```python
from devolve.utils import friedman_test

results = {
    'JADE': [0.01, 0.05, 0.02, 0.03],
    'SHADE': [0.02, 0.06, 0.03, 0.04],
    'L-SHADE': [0.015, 0.055, 0.025, 0.035]
}

result = friedman_test(results)
print(f"χ²={result['statistic']:.4f}, p={result['p_value']:.4f}")
print(f"Mean ranks: {result['mean_ranks']}")
print(result['interpretation'])
```

### Nemenyi Post-Hoc Test

Pairwise comparisons after Friedman test:

```python
from devolve.utils import nemenyi_posthoc_test

posthoc = nemenyi_posthoc_test(results, alpha=0.05)
print(f"Critical difference: {posthoc['critical_difference']:.3f}")
print(f"Significant pairs: {posthoc['significant_pairs']}")
print(f"Mean ranks: {posthoc['mean_ranks']}")
```

### Effect Size

Quantify magnitude of difference:

```python
from devolve.utils import calculate_effect_size

# Cohen's d
es = calculate_effect_size(alg1_results, alg2_results, method='cohen')
print(f"Cohen's d: {es['effect_size']:.3f}")
print(f"Magnitude: {es['magnitude']}")  # negligible/small/medium/large

# Hedges' g (bias-corrected)
es = calculate_effect_size(alg1_results, alg2_results, method='hedges')

# Vargha-Delaney A
es = calculate_effect_size(alg1_results, alg2_results, method='vargha')
```

### StatisticalTests Class

```python
from devolve.utils import StatisticalTests

tests = StatisticalTests()
wilcoxon_result = tests.wilcoxon(alg1_results, alg2_results)
friedman_result = tests.friedman(results)
effect = tests.effect_size(alg1_results, alg2_results)
```

---

## 3. Parallel Execution (`parallel.py`)

### Parallel Function Evaluation

Speed up population evaluation:

```python
from devolve.utils import parallel_evaluate
import numpy as np

def expensive_function(x):
    return np.sum(x**2)

population = np.random.randn(100, 10)
fitness = parallel_evaluate(
    population, 
    expensive_function, 
    n_jobs=4,  # Use 4 cores
    use_tqdm=True  # Show progress bar
)
```

### Parallel Optimization Runs

Run multiple independent optimizations:

```python
from devolve.utils import parallel_optimize
from devolve import JADE, Sphere

problem = Sphere(dimensions=10)
results = parallel_optimize(
    optimizer_class=JADE,
    problem=problem,
    n_runs=30,
    n_jobs=8,
    population_size=50,
    max_iterations=100
)

fitness_values = [f for _, f in results]
print(f"Mean: {np.mean(fitness_values):.6e}")
print(f"Std: {np.std(fitness_values):.6e}")
```

### Optimal Number of Jobs

Get recommendation based on problem characteristics:

```python
from devolve.utils import get_optimal_n_jobs

# For cheap functions (< 1ms)
n_jobs = get_optimal_n_jobs(population_size=100, function_cost='cheap')

# For expensive functions (> 100ms, simulations, etc.)
n_jobs = get_optimal_n_jobs(population_size=100, function_cost='expensive')
```

---

## 4. Seed Management (`seed.py`)

### Set Seed for Reproducibility

```python
from devolve.utils import set_seed

set_seed(42)
# All random operations now reproducible
```

### Generate Seed Sequence

Create seeds for multiple independent runs:

```python
from devolve.utils import get_seed_sequence

seeds = get_seed_sequence(n_runs=10, base_seed=42)
# [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
```

### Ensure Full Reproducibility

```python
from devolve.utils import ensure_reproducibility

info = ensure_reproducibility(seed=42)
print(f"Seed: {info['seed']}")
print(f"NumPy: {info['numpy_version']}")
print(f"Python: {info['python_version']}")
```

### Advanced: Split Seeds

Create independent seeds using hashing:

```python
from devolve.utils import split_seed

# For parallel runs with different but reproducible seeds
seeds = split_seed(seed=42, n_splits=10)
```

### Save and Restore State

```python
from devolve.utils import get_reproducible_state, restore_reproducible_state

# Save current state
state = get_reproducible_state()

# ... do some random operations ...

# Restore to saved state
restore_reproducible_state(state)
```

---

## 5. Input/Output (`io.py`)

### Save and Load Results

**JSON format** (human-readable):

```python
from devolve.utils import save_results, load_results

results = {
    'algorithm': 'JADE',
    'problem': 'Sphere',
    'best_fitness': 0.001,
    'best_position': np.array([0.1, 0.2, 0.3]),
    'history': {'iteration': [0, 1, 2], 'fitness': [10.0, 1.0, 0.1]}
}

save_results(results, 'results.json', format='json')
loaded = load_results('results.json')
```

**Pickle format** (exact precision):

```python
save_results(results, 'results.pkl', format='pickle')
loaded = load_results('results.pkl')
```

### Export to CSV

```python
from devolve.utils import export_to_csv

# Dictionary format
data = {
    'Algorithm': ['JADE', 'SHADE', 'L-SHADE'],
    'Mean': [0.001, 0.002, 0.0015],
    'Std': [0.0001, 0.0002, 0.00015]
}
export_to_csv(data, 'comparison.csv')

# List of dicts format
history = [
    {'iteration': 0, 'fitness': 10.0},
    {'iteration': 1, 'fitness': 1.0},
    {'iteration': 2, 'fitness': 0.1}
]
export_to_csv(history, 'history.csv')
```

### Export to LaTeX Table

Publication-ready tables:

```python
from devolve.utils import export_to_latex_table

data = {
    'Algorithm': ['JADE', 'SHADE', 'L-SHADE'],
    'Mean': [0.00123, 0.00245, 0.00156],
    'Std': [0.00012, 0.00023, 0.00015]
}

export_to_latex_table(
    data,
    'comparison_table.tex',
    caption='Algorithm Comparison on Sphere Function',
    label='tab:sphere_comparison',
    format_spec='.6f',
    bold_best=True  # Bold the best value in each numeric column
)
```

Output:
```latex
\begin{table}[htbp]
\centering
\caption{Algorithm Comparison on Sphere Function}
\label{tab:sphere_comparison}
\begin{tabular}{lrr}
\toprule
Algorithm & Mean & Std \\
\midrule
JADE & \textbf{0.001230} & \textbf{0.000120} \\
SHADE & 0.002450 & 0.000230 \\
L-SHADE & 0.001560 & 0.000150 \\
\bottomrule
\end{tabular}
\end{table}
```

### Export Comparison Table

Automatic statistics calculation:

```python
from devolve.utils import export_comparison_table

algorithm_results = {
    'JADE': [0.001, 0.002, 0.0015, 0.0018],
    'SHADE': [0.002, 0.003, 0.0025, 0.0028],
    'L-SHADE': [0.0015, 0.0025, 0.002, 0.0023]
}

# CSV format with statistics
export_comparison_table(
    algorithm_results,
    'comparison.csv',
    format='csv',
    include_statistics=True  # Adds Mean, Std, Min, Max, Median columns
)

# LaTeX format
export_comparison_table(
    algorithm_results,
    'comparison.tex',
    format='latex'
)
```

---

## Complete Workflow Example

```python
from devolve import JADE, SHADE, LSHADE, Sphere
from devolve.utils import *

# 1. Ensure reproducibility
ensure_reproducibility(seed=42)

# 2. Setup
problem = Sphere(dimensions=10)
algorithms = {'JADE': JADE, 'SHADE': SHADE, 'L-SHADE': LSHADE}
seeds = get_seed_sequence(30, base_seed=42)

# 3. Run experiments in parallel
results = {}
for alg_name, AlgClass in algorithms.items():
    print(f"Running {alg_name}...")
    runs = parallel_optimize(
        AlgClass, problem, n_runs=30, n_jobs=8,
        population_size=50, max_iterations=100, seeds=seeds
    )
    results[alg_name] = [f for _, f in runs]

# 4. Calculate metrics
for alg_name, fitness_values in results.items():
    stability = calculate_stability(fitness_values)
    sr = calculate_success_rate(fitness_values, target_error=0.01)
    print(f"\n{alg_name}:")
    print(f"  Mean: {stability['mean']:.6e} ± {stability['std']:.6e}")
    print(f"  Success rate: {sr:.2%}")

# 5. Statistical tests
tests = StatisticalTests()
friedman = tests.friedman(results)
print(f"\nFriedman test: χ²={friedman['statistic']:.4f}, p={friedman['p_value']:.4f}")

nemenyi = tests.nemenyi(results)
print(f"Significant pairs: {nemenyi['significant_pairs']}")

# 6. Export results
export_comparison_table(results, 'results.csv', format='csv')
export_comparison_table(results, 'results.tex', format='latex')

# 7. Save complete results
all_results = {
    'algorithm_results': results,
    'friedman_test': friedman,
    'nemenyi_test': nemenyi,
    'experiment_config': {
        'problem': 'Sphere',
        'dimensions': 10,
        'n_runs': 30,
        'seed': 42
    }
}
save_results(all_results, 'complete_results.json')

print("\n✅ Analysis complete! Results saved.")
```

---

## Best Practices

### Reproducibility
1. **Always call `ensure_reproducibility()` at the start**
2. Use `get_seed_sequence()` for multiple runs
3. Save seed information with results

### Statistical Testing
1. Use **Wilcoxon** for pairwise comparisons (2 algorithms)
2. Use **Friedman + Nemenyi** for multiple algorithms (3+)
3. Always report **effect sizes** alongside p-values
4. Run at least **30 independent runs** for robust statistics

### Parallel Execution
1. Use `parallel_optimize()` for multiple runs
2. Use `parallel_evaluate()` only for expensive functions
3. Check `get_optimal_n_jobs()` for guidance

### Result Management
1. **JSON** for sharing and visualization
2. **Pickle** for exact numerical precision
3. **CSV** for spreadsheet analysis
4. **LaTeX** for publication tables

---

## Performance Notes

### Metrics
- All metrics: O(n) where n = number of runs
- ERT calculation: O(n)
- Convergence speed: O(m) where m = iterations

### Statistical Tests
- Wilcoxon: O(n log n)
- Friedman: O(kn) where k = algorithms, n = problems
- Nemenyi: O(k²) pairwise comparisons

### Parallel Execution
- Overhead: ~50-100ms per parallel batch
- Beneficial when: function_time > 10ms
- Recommended: n_jobs = min(n_runs, cpu_count)

---

## See Also

- **examples/utilities_comprehensive.py** - Complete workflow examples
- **test_utilities.py** - Unit tests for all functions
- **docs/algorithms.md** - Algorithm selection guide
- **docs/benchmarks.md** - Benchmark problems reference
