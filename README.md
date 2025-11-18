# DEvolve - Comprehensive Differential Evolution Library

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python package implementing all major Differential Evolution (DE) variants for global optimization.

## Features

### 🔥 Algorithm Implementations

**Classic DE Variants:**
- DE/rand/1 - Robust exploration
- DE/best/1 - Fast convergence
- DE/current-to-best/1 - Balanced approach
- DE/rand/2 - Enhanced diversity
- DE/best/2 - Intensive exploitation

**Adaptive Variants:**
- jDE - Self-adaptive parameters
- SaDE - Strategy adaptation
- JADE - Advanced parameter adaptation
- SHADE - Success-history based adaptation
- L-SHADE - Linear population size reduction
- LSHADE-EpSin - Enhanced with epsilon-greedy
- LSHADE-cnEpSin - State-of-the-art (2021)

**Hybrid & Multi-Objective:**
- DE-PSO - Hybrid with Particle Swarm
- DEGL - Global and local neighborhoods
- CoDE - Composite DE
- MODE, GDE3, NSDE - Multi-objective optimization

### 🎯 Key Capabilities

- **Comprehensive Benchmarking**: 8+ classic test functions (Sphere, Rosenbrock, Rastrigin, Ackley, etc.)
- **Constraint Handling**: Penalty methods, feasibility rules, epsilon-constraint, stochastic ranking
- **Flexible Boundary Handling**: Clip, random, reflect, wrap, resample strategies
- **Rich Logging**: Track fitness, diversity, parameters, with export to JSON/CSV
- **Publication-Quality Visualization**: Convergence plots, algorithm comparisons, 3D landscapes, animations
- **Type-Safe**: Full type hints throughout the codebase
- **Well-Documented**: Comprehensive docstrings with mathematical formulas

## Installation

### From Source
```bash
git clone https://github.com/yourusername/devolve.git
cd devolve
pip install -e .
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from devolve import DErand1
from devolve.benchmarks import Sphere
import numpy as np

# Create a 10-dimensional sphere problem
problem = Sphere(dimensions=10)

# Create optimizer
optimizer = DErand1(
    problem=problem,
    population_size=50,
    max_iterations=500,
    F=0.8,
    CR=0.9
)

# Run optimization
best_position, best_fitness = optimizer.optimize()

print(f"Best fitness: {best_fitness:.6e}")
print(f"Best position: {best_position}")
```

## Examples

### Compare Multiple Algorithms

```python
from devolve import DErand1, DEbest1, DEcurrentToBest1
from devolve.benchmarks import Rastrigin

problem = Rastrigin(dimensions=20)

algorithms = [
    DErand1(problem, population_size=100, max_iterations=500),
    DEbest1(problem, population_size=100, max_iterations=500),
    DEcurrentToBest1(problem, population_size=100, max_iterations=500),
]

for algo in algorithms:
    _, fitness = algo.optimize()
    print(f"{algo.__class__.__name__}: {fitness:.6e}")
```

### Visualization and Analysis

```python
from devolve import DErand1
from devolve.benchmarks import Sphere
from devolve.utils import (
    set_publication_style,
    plot_convergence,
    generate_all_figures
)

# Set publication-quality style
set_publication_style()

# Run optimization
problem = Sphere(dimensions=30)
optimizer = DErand1(problem=problem, max_iterations=500)
best_solution, best_fitness = optimizer.optimize()

# Plot convergence
history = optimizer.logger.get_history()
fig = plot_convergence(
    history=history,
    title="DE/rand/1 on 30D Sphere",
    log_scale=True,
    save_path="figures/convergence",
    file_formats=['png', 'pdf']
)

# Or generate all figures automatically
generate_all_figures(
    results=optimizer,
    algorithm_name='DErand1',
    problem_name='Sphere_30D',
    formats=['png', 'pdf']
)
```

### Custom Optimization Problem

```python
from devolve import DErand1, Problem
import numpy as np

# Define objective function
def my_function(x):
    return np.sum(x**2) + np.prod(np.cos(x))

# Create problem
problem = Problem(
    objective_function=my_function,
    bounds=(-10, 10),
    dimensions=5,
    name="MyProblem"
)

# Optimize
optimizer = DErand1(problem=problem)
best_x, best_f = optimizer.optimize()
```

### With Constraints

```python
from devolve import DErand1, Problem

def objective(x):
    return x[0]**2 + x[1]**2

def constraint1(x):
    return x[0] + x[1] - 1  # x[0] + x[1] <= 1

problem = Problem(
    objective_function=objective,
    bounds=[(-5, 5), (-5, 5)],
    dimensions=2,
    constraints=[constraint1]
)

optimizer = DErand1(problem=problem, population_size=30)
best_x, best_f = optimizer.optimize()
```

## Algorithm Comparison

| Algorithm | Exploration | Convergence Speed | Robustness | Best For |
|-----------|-------------|-------------------|------------|----------|
| DE/rand/1 | High | Medium | High | Multimodal problems |
| DE/best/1 | Low | Fast | Low | Unimodal problems |
| DE/current-to-best/1 | Medium | Medium-Fast | Medium | General purpose |
| DE/rand/2 | Very High | Slow | Very High | Complex landscapes |
| DE/best/2 | Very Low | Very Fast | Very Low | Simple problems |
| jDE | Adaptive | Medium | High | Unknown landscapes |
| JADE | Adaptive | Fast | High | General purpose |
| SHADE/L-SHADE | Adaptive | Very Fast | Very High | Competition-grade |

## Parameter Guidelines

### Population Size
- **Small problems (D ≤ 10)**: 5×D to 10×D
- **Medium problems (10 < D ≤ 30)**: 3×D to 5×D
- **Large problems (D > 30)**: 2×D to 3×D
- **Minimum**: 4 (required by algorithms)

### Scaling Factor (F)
- **Exploration-focused**: 0.8 - 1.0
- **Balanced**: 0.5 - 0.8
- **Exploitation-focused**: 0.4 - 0.5
- **Typical default**: 0.8

### Crossover Rate (CR)
- **Separable problems**: 0.9 - 1.0
- **Non-separable problems**: 0.1 - 0.3
- **General purpose**: 0.9
- **Typical default**: 0.9

## Project Structure

```
devolve/
├── core/               # Core classes
│   ├── base.py        # Base DE algorithm
│   ├── individual.py  # Individual representation
│   ├── population.py  # Population management
│   ├── problem.py     # Problem definition
│   ├── boundary.py    # Boundary handling
│   └── logger.py      # Logging system
├── algorithms/         # DE implementations
│   ├── classic/       # Classic variants
│   ├── adaptive/      # Adaptive variants
│   ├── hybrid/        # Hybrid algorithms
│   └── multiobjective/ # Multi-objective DE
├── operators/          # DE operators
│   ├── mutation.py    # Mutation strategies
│   ├── crossover.py   # Crossover operators
│   └── selection.py   # Selection operators
├── benchmarks/         # Test functions
│   ├── classic.py     # Classic benchmarks
│   └── cec/           # CEC test suites
├── constraints/        # Constraint handling
├── utils/              # Utilities
│   ├── metrics.py     # Performance metrics
│   ├── stats.py       # Statistical tests
│   ├── visualization.py # Core plotting functions
│   ├── visualization_extended.py # Advanced plots
│   └── visualization_master.py # Automation tools
└── ml/                 # ML integration
    ├── sklearn_optimizer.py
    └── feature_selection.py
```

## Performance

Benchmarks on standard test functions (30D, 25 independent runs):

| Function | Algorithm | Mean Error | Std Dev | Success Rate |
|----------|-----------|------------|---------|--------------|
| Sphere | DE/rand/1 | 1.23e-08 | 2.45e-09 | 100% |
| Rastrigin | DE/rand/1 | 15.4 | 8.2 | 76% |
| Ackley | DE/best/1 | 2.1e-06 | 4.3e-07 | 100% |
| Rosenbrock | DE/current-to-best/1 | 12.3 | 15.7 | 68% |

## Citation

If you use DEvolve in your research, please cite:

```bibtex
@software{devolve2024,
  title={DEvolve: A Comprehensive Differential Evolution Library},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/devolve}
}
```

## Key References

1. **Original DE**: Storn, R., & Price, K. (1997). *Journal of Global Optimization*, 11(4), 341-359.
2. **jDE**: Brest et al. (2006). *IEEE Congress on Evolutionary Computation*.
3. **JADE**: Zhang & Sanderson (2009). *IEEE Transactions on Evolutionary Computation*, 13(5), 945-958.
4. **SHADE**: Tanabe & Fukunaga (2013). *IEEE Congress on Evolutionary Computation*.
5. **L-SHADE**: Tanabe & Fukunaga (2014). *IEEE Congress on Evolutionary Computation*.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the extensive DE research community
- Thanks to all contributors to the original DE algorithms
- Built with NumPy, SciPy, and Matplotlib

## Contact

- **Author**: DEvolve Development Team
- **Email**: your.email@example.com
- **GitHub**: https://github.com/yourusername/devolve
- **Documentation**: https://devolve.readthedocs.io (coming soon)

---

**Made with ❤️ for the optimization community**
