# Visualization Module Documentation

## Overview

The DEvolve visualization module provides publication-quality plotting functions for analyzing and comparing Differential Evolution algorithms. All figures are automatically saved in organized folder structures with multiple format support (PNG, PDF, SVG, EPS).

## Installation

The visualization module requires additional dependencies:

```bash
pip install matplotlib>=3.5.0
pip install seaborn>=0.11.0
pip install tqdm>=4.62.0
```

Or install all dependencies at once:

```bash
pip install -e ".[visualization]"
```

## Quick Start

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
problem = Sphere(dimensions=10)
optimizer = DErand1(problem=problem, max_iterations=200)
best_solution, best_fitness = optimizer.optimize()

# Plot convergence
history = optimizer.logger.get_history()
fig = plot_convergence(history, title="DE/rand/1 on Sphere")
fig.savefig("convergence.png", dpi=300)
```

## Module Structure

The visualization module is organized into three files:

1. **`visualization.py`** - Core plotting functions (convergence, comparison, statistical)
2. **`visualization_extended.py`** - Advanced plots (2D population, 3D landscapes, animations)
3. **`visualization_master.py`** - Automation tools (comprehensive reports, table generation)

## Available Functions

### Publication Styling

#### `set_publication_style()`
Configure matplotlib for publication-quality figures.

**Settings:**
- Font: Times New Roman (12pt text, 14pt labels, 16pt titles)
- Line width: 2.0
- DPI: 300 for saving
- Grid: Light gray, dashed
- Colors: Colorblind-friendly Okabe-Ito palette

**Example:**
```python
from devolve.utils import set_publication_style
set_publication_style()
```

### Folder Organization

#### `setup_figure_folders(base_path="figures")`
Create organized folder structure for saving figures.

**Returns:** Dictionary mapping category names to paths

**Folder Structure:**
```
figures/
├── convergence/      # Convergence curves
├── population/       # Population scatter plots
├── comparison/       # Algorithm comparisons
├── parameters/       # Parameter evolution (F, CR)
├── diversity/        # Diversity metrics
├── statistical/      # Box plots, statistical tests
├── animations/       # GIF/MP4 animations
├── 3d_landscapes/    # 3D surface plots
├── combined/         # Multi-subplot reports
└── tables/           # LaTeX tables
```

**Example:**
```python
from devolve.utils import setup_figure_folders
folders = setup_figure_folders("my_results/figures")
print(folders['convergence'])  # 'my_results/figures/convergence'
```

### Convergence Plots

#### `plot_convergence(history, title, log_scale, show_mean, show_std, ...)`
Plot convergence curve showing best fitness over iterations.

**Parameters:**
- `history`: dict or list - Fitness history
- `title`: str - Plot title
- `log_scale`: bool - Use log scale for y-axis
- `show_mean`: bool - Plot mean fitness line
- `show_std`: bool - Plot std deviation as shaded area
- `save_path`: str - Path to save (without extension)
- `figsize`: tuple - (width, height) in inches
- `dpi`: int - Resolution
- `file_formats`: list - ['png', 'pdf', 'svg', 'eps']

**Example:**
```python
from devolve.utils import plot_convergence

history = optimizer.logger.get_history()
fig = plot_convergence(
    history=history,
    title="DE/rand/1 on Sphere",
    log_scale=True,
    show_mean=True,
    show_std=True,
    save_path="figures/convergence/sphere",
    file_formats=['png', 'pdf']
)
```

#### `plot_convergence_with_ci(runs_data, confidence_level, ...)`
Convergence plot with confidence intervals from multiple runs.

**Parameters:**
- `runs_data`: List[List[float]] - Multiple fitness histories
- `confidence_level`: float - e.g., 0.95 for 95% CI
- `show_median`: bool - Show median line
- `show_mean`: bool - Show mean line
- `show_best`: bool - Show best run
- `show_worst`: bool - Show worst run

**Example:**
```python
from devolve.utils import plot_convergence_with_ci

# Run algorithm 30 times
runs_data = []
for i in range(30):
    optimizer = DErand1(problem=problem, seed=i)
    optimizer.optimize()
    runs_data.append(optimizer.logger.get_history()['best_fitness'])

# Plot with confidence intervals
fig = plot_convergence_with_ci(
    runs_data=runs_data,
    confidence_level=0.95,
    title="DE/rand/1 (30 runs)",
    log_scale=True,
    show_best=True,
    save_path="figures/convergence/ci_plot"
)
```

### Algorithm Comparison

#### `plot_algorithm_comparison(results_dict, title, log_scale, ...)`
Compare convergence of multiple algorithms on the same plot.

**Parameters:**
- `results_dict`: Dict[str, List/Dict] - Algorithm histories
- `title`: str - Plot title
- `log_scale`: bool - Use log scale

**Example:**
```python
from devolve import DErand1, DEbest1, JADE
from devolve.utils import plot_algorithm_comparison

# Run multiple algorithms
algorithms = {'DE/rand/1': DErand1, 'DE/best/1': DEbest1}
results = {}

for name, AlgoClass in algorithms.items():
    optimizer = AlgoClass(problem=problem)
    optimizer.optimize()
    results[name] = optimizer.logger.get_history()

# Plot comparison
fig = plot_algorithm_comparison(
    results_dict=results,
    title="Algorithm Comparison on Rastrigin",
    log_scale=False,
    save_path="figures/comparison/algorithms"
)
```

### Statistical Analysis

#### `plot_statistical_comparison(results, metric, plot_type, show_significance, ...)`
Statistical comparison using box plots or violin plots.

**Parameters:**
- `results`: Dict[str, List[float]] - Final fitness values from multiple runs
- `metric`: str - Metric name
- `plot_type`: str - 'box', 'violin', or 'boxen'
- `show_significance`: bool - Show statistical significance markers

**Significance markers:**
- `***`: p < 0.001
- `**`: p < 0.01
- `*`: p < 0.05
- `ns`: not significant

**Example:**
```python
from devolve.utils import plot_statistical_comparison

# Run algorithms multiple times
results = {}
for algo_name in ['DE/rand/1', 'JADE', 'L-SHADE']:
    results[algo_name] = []
    for run in range(30):
        optimizer = AlgoClass(problem=problem, seed=run)
        _, fitness = optimizer.optimize()
        results[algo_name].append(fitness)

# Plot statistical comparison
fig = plot_statistical_comparison(
    results=results,
    metric='best_fitness',
    title="Statistical Comparison (30 runs)",
    plot_type='box',
    show_significance=True,
    save_path="figures/statistical/boxplot"
)
```

### Population Visualization (2D)

#### `plot_population_2d(population, fitness_values, best_solution, bounds, contour_function, ...)`
Scatter plot of population in 2D search space.

**Parameters:**
- `population`: np.ndarray - (N, 2) array
- `fitness_values`: np.ndarray - (N,) array
- `best_solution`: np.ndarray - (2,) array
- `bounds`: List[Tuple] - [(x_min, x_max), (y_min, y_max)]
- `contour_function`: Callable - Function for contour background
- `show_contour`: bool - Display contour plot

**Example:**
```python
from devolve.utils import plot_population_2d

# Run on 2D problem
problem = Rastrigin(dimensions=2)
optimizer = DErand1(problem=problem)
best_solution, _ = optimizer.optimize()

# Get population
pop = np.array([ind.position for ind in optimizer.population.individuals])
fitness = np.array([ind.fitness for ind in optimizer.population.individuals])

# Plot
fig = plot_population_2d(
    population=pop,
    fitness_values=fitness,
    best_solution=best_solution,
    iteration=optimizer.current_iteration,
    bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    contour_function=lambda x: problem.objective_function(x),
    save_path="figures/population/final"
)
```

### Animations

#### `animate_population_2d(population_history, fitness_history, best_history, bounds, ...)`
Create animated GIF or MP4 showing population evolution.

**Parameters:**
- `population_history`: List[np.ndarray] - Populations at each iteration
- `fitness_history`: List[np.ndarray] - Fitness at each iteration
- `best_history`: List[np.ndarray] - Best solutions at each iteration
- `fps`: int - Frames per second
- `interval`: int - Milliseconds between frames

**Example:**
```python
from devolve.utils import animate_population_2d

# Note: You need to store population history during optimization
# This requires modifying the optimizer to save intermediate populations

animate_population_2d(
    population_history=optimizer.population_history,
    fitness_history=optimizer.fitness_history,
    best_history=optimizer.best_history,
    bounds=[(-5, 5), (-5, 5)],
    contour_function=lambda x: problem.objective_function(x),
    save_path="figures/animations/evolution.gif",
    fps=10
)
```

### 3D Landscapes

#### `plot_3d_landscape(function, bounds, population, best_solution, resolution, ...)`
3D surface plot of fitness landscape.

**Parameters:**
- `function`: Callable - Benchmark function
- `bounds`: List[Tuple] - [(x_min, x_max), (y_min, y_max)]
- `population`: np.ndarray - Current population (optional)
- `best_solution`: np.ndarray - Best solution (optional)
- `resolution`: int - Grid resolution
- `elevation`: float - View angle elevation
- `azimuth`: float - View angle azimuth

**Example:**
```python
from devolve.utils import plot_3d_landscape
from devolve.benchmarks import Rastrigin

problem = Rastrigin(dimensions=2)

fig = plot_3d_landscape(
    function=lambda x: problem.objective_function(x),
    bounds=[(-5.12, 5.12), (-5.12, 5.12)],
    population=pop,
    best_solution=best_solution,
    resolution=100,
    elevation=30,
    azimuth=45,
    save_path="figures/3d_landscapes/rastrigin"
)
```

### Parameter Evolution (Adaptive Algorithms)

#### `plot_parameter_evolution(f_history, cr_history, ...)`
Plot evolution of F and CR parameters over iterations.

**Parameters:**
- `f_history`: Dict - {'mean': [...], 'std': [...], 'min': [...], 'max': [...]}
- `cr_history`: Dict - Same format as f_history

**Example:**
```python
from devolve.utils import plot_parameter_evolution

# For adaptive algorithms that track parameter evolution
f_hist = {'mean': [...], 'std': [...]}
cr_hist = {'mean': [...], 'std': [...]}

fig = plot_parameter_evolution(
    f_history=f_hist,
    cr_history=cr_hist,
    save_path="figures/parameters/evolution"
)
```

### Diversity Metrics

#### `plot_diversity(diversity_history, title, ...)`
Plot population diversity metrics over time.

**Parameters:**
- `diversity_history`: List or Dict - Diversity values over time

**Example:**
```python
from devolve.utils import plot_diversity, calculate_diversity

# Calculate diversity during optimization
diversity_hist = []
for iteration in range(max_iterations):
    # ... optimization step
    diversity = calculate_diversity(population_array)
    diversity_hist.append(diversity)

# Plot
fig = plot_diversity(
    diversity_history=diversity_hist,
    title="Population Diversity Over Time",
    save_path="figures/diversity/metrics"
)
```

### LaTeX Tables

#### `generate_comparison_table(results_dict, metrics, bold_best, ...)`
Generate LaTeX table for paper.

**Parameters:**
- `results_dict`: Dict[str, Dict[str, float]] - Algorithm results
- `metrics`: List[str] - Metrics to include
- `bold_best`: bool - Bold best values
- `format_scientific`: bool - Use scientific notation

**Example:**
```python
from devolve.utils import generate_comparison_table

results = {
    'DE/rand/1': {'Mean': 1.23e-05, 'Std': 4.56e-06, 'Best': 8.90e-06},
    'JADE': {'Mean': 9.87e-06, 'Std': 3.21e-06, 'Best': 7.65e-06}
}

latex_str = generate_comparison_table(
    results_dict=results,
    metrics=['Mean', 'Std', 'Best'],
    save_path="figures/tables/comparison.tex",
    caption="Algorithm Comparison",
    label="tab:comparison"
)

print(latex_str)
```

### Comprehensive Reports

#### `create_comprehensive_report(results, algorithm_name, problem_name, ...)`
Create comprehensive figure with multiple subplots (2×3 grid).

**Subplots:**
- (a) Convergence curve
- (b) Population scatter (2D)
- (c) F parameter evolution
- (d) Diversity metrics
- (e) Fitness distribution
- (f) CR parameter evolution

**Example:**
```python
from devolve.utils import create_comprehensive_report

fig = create_comprehensive_report(
    results=optimizer_results,
    algorithm_name='DE/rand/1',
    problem_name='Sphere_30D',
    save_path="figures/combined/report",
    file_formats=['png', 'pdf']
)
```

### Automatic Generation

#### `generate_all_figures(results, algorithm_name, problem_name, ...)`
Master function that generates all figures automatically.

**Generates:**
1. Convergence curve
2. Population scatter (if 2D)
3. Parameter evolution (if adaptive)
4. Diversity plot
5. 3D landscape (if 2D)
6. Comprehensive report

**Example:**
```python
from devolve.utils import generate_all_figures

# After optimization
folders = generate_all_figures(
    results=optimizer_results,
    algorithm_name='LSHADE',
    problem_name='Rastrigin_30D',
    base_save_path='figures',
    formats=['png', 'pdf', 'svg'],
    dpi=300,
    generate_animation=False
)

print(f"Figures saved in: {folders}")
```

## Color Palettes

The module uses the **Okabe-Ito colorblind-friendly palette**:

```python
OKABE_ITO = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'yellow': '#ECE133',
    'sky': '#56B4E9',
    'vermillion': '#CC3311',
    'purple': '#CC78BC',
    'gray': '#949494'
}
```

## File Naming Convention

All saved figures follow this naming pattern:

```
{algorithm}_{problem}_{dimension}D_{figtype}_{timestamp}.{ext}
```

**Examples:**
- `LSHADE_Sphere_30D_convergence_20240101_120000.png`
- `DE_Rastrigin_10D_population_20240101_120000.pdf`
- `JADE_comparison_CEC2017_statistical_20240101_120000.svg`

## Complete Example

```python
from devolve import DErand1, DEbest1
from devolve.benchmarks import Sphere, Rastrigin
from devolve.utils import (
    set_publication_style,
    setup_figure_folders,
    plot_convergence,
    plot_algorithm_comparison,
    generate_all_figures
)

# 1. Set publication style
set_publication_style()

# 2. Setup folders
folders = setup_figure_folders("my_results")

# 3. Run optimization
problem = Sphere(dimensions=30)
optimizer = DErand1(problem=problem, max_iterations=500)
best_solution, best_fitness = optimizer.optimize()

# 4. Generate all figures
generate_all_figures(
    results=optimizer_results,
    algorithm_name='DErand1',
    problem_name='Sphere_30D',
    base_save_path='my_results',
    formats=['png', 'pdf'],
    dpi=300
)

print(f"All figures saved! Best fitness: {best_fitness:.6e}")
```

## Tips for Research Papers

### High-Quality Figures
```python
# Use these settings for journal submissions
set_publication_style()
fig = plot_convergence(
    history=history,
    figsize=(10, 6),  # Standard column width
    dpi=300,          # Minimum for print
    file_formats=['pdf', 'eps']  # Vector formats
)
```

### Statistical Comparison
```python
# Run 30 independent trials (common in EC research)
n_runs = 30
results = {}

for algo_name in algorithm_list:
    results[algo_name] = []
    for run in range(n_runs):
        optimizer = AlgoClass(problem=problem, seed=run)
        _, fitness = optimizer.optimize()
        results[algo_name].append(fitness)

# Generate box plot with significance tests
plot_statistical_comparison(
    results=results,
    show_significance=True,  # Requires scipy
    save_path="statistical_comparison"
)
```

### LaTeX Integration
```python
# Generate table
latex_str = generate_comparison_table(
    results_dict=results,
    save_path="tables/results.tex"
)

# In your LaTeX document:
# \input{figures/tables/results.tex}
```

## Troubleshooting

### Missing Dependencies
```bash
pip install matplotlib seaborn tqdm scipy
```

### Font Issues
If Times New Roman is not available:
```python
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Serif'
```

### Memory Issues with Animations
Reduce resolution and frame rate:
```python
animate_population_2d(
    ...,
    dpi=72,      # Lower DPI
    fps=5,       # Fewer frames per second
    resolution=50  # Lower grid resolution
)
```

## References

For more details, see:
- Example script: `examples/03_visualization_demo.py`
- Source code: `devolve/utils/visualization*.py`
- Main README: `README.md`

## Citation

If you use DEvolve visualization in your research, please cite:

```bibtex
@software{devolve2024,
  title={DEvolve: A Comprehensive Differential Evolution Library},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/devolve}
}
```
