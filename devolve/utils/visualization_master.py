"""
Master Figure Generation and Utility Functions
Part 3: Comprehensive reports, table generation, and automation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

from .visualization import (
    OKABE_ITO, OKABE_ITO_COLORS, _save_figure, setup_figure_folders,
    plot_convergence, plot_algorithm_comparison, plot_statistical_comparison
)

try:
    from .visualization_extended import (
        plot_population_2d, plot_3d_landscape, plot_parameter_evolution,
        plot_diversity, calculate_diversity
    )
    EXTENDED_AVAILABLE = True
except ImportError:
    EXTENDED_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable) if self.iterable else []
        def update(self, n=1):
            pass


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_comparison_table(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = ['Mean', 'Std', 'Best', 'Worst', 'Success Rate'],
    save_path: str = "comparison_table.tex",
    bold_best: bool = True,
    format_scientific: bool = True,
    caption: str = "Algorithm Comparison on Benchmark Functions",
    label: str = "tab:comparison"
) -> str:
    """
    Generate LaTeX table for paper.
    
    Parameters:
    -----------
    results_dict : Dict[str, Dict[str, float]]
        Dictionary mapping algorithm names to their metrics
        Format: {
            'Algorithm1': {'Mean': 1.23e-05, 'Std': 4.56e-06, ...},
            'Algorithm2': {'Mean': 9.87e-06, ...},
            ...
        }
    metrics : List[str]
        List of metric names to include in table
    save_path : str
        Path to save LaTeX table file
    bold_best : bool
        Bold the best value in each column
    format_scientific : bool
        Use scientific notation
    caption : str
        Table caption
    label : str
        LaTeX label for reference
    
    Returns:
    --------
    str
        LaTeX table string
    
    Example:
        >>> results = {
        ...     'DE/rand/1': {'Mean': 1.23e-05, 'Std': 4.56e-06, 'Best': 8.90e-06},
        ...     'JADE': {'Mean': 9.87e-06, 'Std': 3.21e-06, 'Best': 7.65e-06}
        ... }
        >>> latex_str = generate_comparison_table(results)
    """
    # Find best values for each metric
    best_values = {}
    for metric in metrics:
        values = []
        for algo_results in results_dict.values():
            if metric in algo_results:
                values.append(algo_results[metric])
        
        if values:
            # For most metrics, lower is better (except Success Rate)
            if 'success' in metric.lower() or 'rate' in metric.lower():
                best_values[metric] = max(values)
            else:
                best_values[metric] = min(values)
    
    # Helper function to format values
    def format_value(value, is_best=False):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return '--'
        
        if format_scientific and isinstance(value, float) and abs(value) < 0.01:
            formatted = f"{value:.2e}"
        elif isinstance(value, float):
            formatted = f"{value:.4f}"
        else:
            formatted = str(value)
        
        if is_best and bold_best:
            formatted = f"\\textbf{{{formatted}}}"
        
        return formatted
    
    # Build LaTeX table
    latex_lines = []
    
    # Table header
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append(f"\\caption{{{caption}}}")
    latex_lines.append(f"\\label{{{label}}}")
    
    # Column specification
    n_cols = len(metrics) + 1
    col_spec = "l|" + "c" * len(metrics)
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\hline")
    
    # Header row
    header = "Algorithm & " + " & ".join(metrics) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")
    
    # Data rows
    for algo_name, algo_results in results_dict.items():
        row_values = [algo_name.replace('_', '\\_')]  # Escape underscores
        
        for metric in metrics:
            value = algo_results.get(metric, None)
            is_best = (value == best_values.get(metric, None))
            row_values.append(format_value(value, is_best))
        
        row = " & ".join(row_values) + " \\\\"
        latex_lines.append(row)
    
    # Table footer
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Join lines
    latex_str = "\n".join(latex_lines)
    
    # Save to file
    with open(save_path, 'w') as f:
        f.write(latex_str)
    
    print(f"LaTeX table saved to: {save_path}")
    return latex_str


# ============================================================================
# COMPREHENSIVE REPORT FIGURE
# ============================================================================

def create_comprehensive_report(
    results: Any,
    algorithm_name: str = "Algorithm",
    problem_name: str = "Problem",
    save_path: str = "comprehensive_report.png",
    figsize: Tuple[float, float] = (16, 12),
    dpi: int = 300,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Create a comprehensive figure with multiple subplots.
    
    Parameters:
    -----------
    results : Any
        Results object from optimization (should have attributes like:
        history, population_history, best_history, etc.)
    algorithm_name : str
        Name of the algorithm
    problem_name : str
        Name of the problem
    save_path : str
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    file_formats : list
        File formats to save
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Layout (2x3 grid):
    +-------------------+-------------------+-------------------+
    | Convergence       | Population 2D     | Parameter F       |
    +-------------------+-------------------+-------------------+
    | Diversity         | Fitness Dist.     | Parameter CR      |
    +-------------------+-------------------+-------------------+
    
    Example:
        >>> fig = create_comprehensive_report(
        ...     results=optimizer_results,
        ...     algorithm_name='L-SHADE',
        ...     problem_name='Rastrigin_30D'
        ... )
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f'{algorithm_name} on {problem_name}',
                fontsize=20, fontweight='bold')
    
    # Subplot (a): Convergence
    ax1 = fig.add_subplot(gs[0, 0])
    if hasattr(results, 'history') or hasattr(results, 'best_fitness_history'):
        history = getattr(results, 'history', getattr(results, 'best_fitness_history', []))
        if isinstance(history, dict):
            iterations = history.get('iteration', list(range(len(history.get('best_fitness', [])))))
            best_fitness = history.get('best_fitness', [])
        else:
            iterations = list(range(len(history)))
            best_fitness = history
        
        ax1.plot(iterations, best_fitness, linewidth=2.5, color=OKABE_ITO['blue'],
                marker='o', markevery=max(1, len(iterations)//10))
        ax1.set_xlabel('Iterations', fontsize=12)
        ax1.set_ylabel('Best Fitness', fontsize=12)
        ax1.set_title('(a) Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_yscale('log')
    
    # Subplot (b): Population 2D (if 2D problem)
    ax2 = fig.add_subplot(gs[0, 1])
    if hasattr(results, 'final_population') and hasattr(results, 'dimension'):
        if results.dimension == 2:
            pop = results.final_population
            fitness = getattr(results, 'final_fitness', np.sum(pop**2, axis=1))
            best = getattr(results, 'best_solution', pop[np.argmin(fitness)])
            
            scatter = ax2.scatter(pop[:, 0], pop[:, 1], c=fitness,
                                cmap='viridis', s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
            ax2.scatter(best[0], best[1], marker='*', s=300, color='red',
                       edgecolors='black', linewidth=2, zorder=10)
            ax2.set_xlabel('$x_1$', fontsize=12)
            ax2.set_ylabel('$x_2$', fontsize=12)
            ax2.set_title('(b) Final Population', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
        else:
            ax2.text(0.5, 0.5, f'{results.dimension}D Problem\n(2D plot not applicable)',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('(b) Population', fontsize=14, fontweight='bold')
            ax2.axis('off')
    else:
        ax2.text(0.5, 0.5, 'Population data\nnot available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('(b) Population', fontsize=14, fontweight='bold')
        ax2.axis('off')
    
    # Subplot (c): Parameter F (if adaptive algorithm)
    ax3 = fig.add_subplot(gs[0, 2])
    if hasattr(results, 'f_history') and results.f_history:
        if isinstance(results.f_history, dict) and 'mean' in results.f_history:
            f_mean = results.f_history['mean']
            iterations_f = list(range(len(f_mean)))
            ax3.plot(iterations_f, f_mean, linewidth=2.5, color=OKABE_ITO['blue'])
            
            if 'std' in results.f_history:
                f_std = results.f_history['std']
                f_mean_arr = np.array(f_mean)
                f_std_arr = np.array(f_std)
                ax3.fill_between(iterations_f, f_mean_arr - f_std_arr,
                                f_mean_arr + f_std_arr, alpha=0.3, color=OKABE_ITO['blue'])
        else:
            values = results.f_history.get('values', results.f_history)
            iterations_f = list(range(len(values)))
            ax3.plot(iterations_f, values, linewidth=2.5, color=OKABE_ITO['blue'])
        
        ax3.set_xlabel('Iterations', fontsize=12)
        ax3.set_ylabel('F Parameter', fontsize=12)
        ax3.set_title('(c) F Parameter Evolution', fontsize=14, fontweight='bold')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'F parameter data\nnot available\n(not adaptive algorithm)',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('(c) F Parameter', fontsize=14, fontweight='bold')
        ax3.axis('off')
    
    # Subplot (d): Diversity
    ax4 = fig.add_subplot(gs[1, 0])
    if hasattr(results, 'diversity_history') and results.diversity_history:
        diversity = results.diversity_history
        if isinstance(diversity, list):
            iterations_div = list(range(len(diversity)))
            ax4.plot(iterations_div, diversity, linewidth=2.5, color=OKABE_ITO['green'],
                    marker='o', markevery=max(1, len(iterations_div)//10))
        ax4.set_xlabel('Iterations', fontsize=12)
        ax4.set_ylabel('Diversity', fontsize=12)
        ax4.set_title('(d) Population Diversity', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
    else:
        ax4.text(0.5, 0.5, 'Diversity data\nnot available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('(d) Diversity', fontsize=14, fontweight='bold')
        ax4.axis('off')
    
    # Subplot (e): Fitness Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    if hasattr(results, 'final_fitness'):
        fitness_vals = results.final_fitness
        ax5.hist(fitness_vals, bins=20, color=OKABE_ITO['orange'], alpha=0.7,
                edgecolor='black', linewidth=0.5)
        ax5.axvline(x=np.min(fitness_vals), color='red', linestyle='--',
                   linewidth=2, label=f'Best: {np.min(fitness_vals):.2e}')
        ax5.set_xlabel('Fitness Value', fontsize=12)
        ax5.set_ylabel('Frequency', fontsize=12)
        ax5.set_title('(e) Final Fitness Distribution', fontsize=14, fontweight='bold')
        ax5.legend(loc='best', frameon=False)
        ax5.grid(True, alpha=0.3, linestyle='--', axis='y')
    else:
        ax5.text(0.5, 0.5, 'Fitness distribution\ndata not available',
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('(e) Fitness Distribution', fontsize=14, fontweight='bold')
        ax5.axis('off')
    
    # Subplot (f): Parameter CR (if adaptive algorithm)
    ax6 = fig.add_subplot(gs[1, 2])
    if hasattr(results, 'cr_history') and results.cr_history:
        if isinstance(results.cr_history, dict) and 'mean' in results.cr_history:
            cr_mean = results.cr_history['mean']
            iterations_cr = list(range(len(cr_mean)))
            ax6.plot(iterations_cr, cr_mean, linewidth=2.5, color=OKABE_ITO['orange'])
            
            if 'std' in results.cr_history:
                cr_std = results.cr_history['std']
                cr_mean_arr = np.array(cr_mean)
                cr_std_arr = np.array(cr_std)
                ax6.fill_between(iterations_cr, cr_mean_arr - cr_std_arr,
                                cr_mean_arr + cr_std_arr, alpha=0.3, color=OKABE_ITO['orange'])
        else:
            values = results.cr_history.get('values', results.cr_history)
            iterations_cr = list(range(len(values)))
            ax6.plot(iterations_cr, values, linewidth=2.5, color=OKABE_ITO['orange'])
        
        ax6.set_xlabel('Iterations', fontsize=12)
        ax6.set_ylabel('CR Parameter', fontsize=12)
        ax6.set_title('(f) CR Parameter Evolution', fontsize=14, fontweight='bold')
        ax6.set_ylim(-0.1, 1.1)
        ax6.grid(True, alpha=0.3, linestyle='--')
    else:
        ax6.text(0.5, 0.5, 'CR parameter data\nnot available\n(not adaptive algorithm)',
                ha='center', va='center', transform=ax6.transAxes, fontsize=12)
        ax6.set_title('(f) CR Parameter', fontsize=14, fontweight='bold')
        ax6.axis('off')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


# ============================================================================
# MASTER FIGURE GENERATION
# ============================================================================

def generate_all_figures(
    results: Any,
    algorithm_name: str,
    problem_name: str,
    base_save_path: str = "figures",
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
    generate_animation: bool = False
) -> Dict[str, str]:
    """
    Generate all figures automatically after optimization.
    
    Parameters:
    -----------
    results : Any
        Results object from optimization
    algorithm_name : str
        Name of the algorithm
    problem_name : str
        Name of the problem
    base_save_path : str
        Base directory for saving figures
    formats : list
        File formats to save (e.g., ['png', 'pdf', 'svg'])
    dpi : int
        Resolution
    generate_animation : bool
        Whether to generate animation (can be slow)
    
    Returns:
    --------
    Dict[str, str]
        Dictionary of folder paths
    
    Generated figures:
    1. Convergence curve
    2. Population scatter (if 2D)
    3. Parameter evolution (if adaptive)
    4. Diversity plot
    5. 3D landscape (if 2D)
    6. Comprehensive report
    
    Naming convention:
    {algorithm_name}_{problem_name}_{figure_type}_{timestamp}.{format}
    
    Example:
        >>> folders = generate_all_figures(
        ...     results=optimizer.results,
        ...     algorithm_name='LSHADE',
        ...     problem_name='Rastrigin_30D',
        ...     base_save_path='my_figures',
        ...     formats=['png', 'pdf']
        ... )
    """
    # Setup folders
    print(f"\n{'='*60}")
    print(f"GENERATING FIGURES: {algorithm_name} on {problem_name}")
    print(f"{'='*60}")
    
    folders = setup_figure_folders(base_save_path)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    figures_generated = []
    
    with tqdm(total=6, desc="Generating figures") as pbar:
        # 1. Convergence plot
        try:
            history = getattr(results, 'history', getattr(results, 'best_fitness_history', None))
            if history:
                save_path = f"{folders['convergence']}/{algorithm_name}_{problem_name}_convergence_{timestamp}"
                plot_convergence(history, title=f"{algorithm_name} Convergence on {problem_name}",
                               save_path=save_path, file_formats=formats, dpi=dpi)
                figures_generated.append(f"Convergence: {save_path}")
            pbar.update(1)
        except Exception as e:
            print(f"  ⚠ Convergence plot failed: {e}")
            pbar.update(1)
        
        # 2. Population (if 2D)
        try:
            if EXTENDED_AVAILABLE and hasattr(results, 'final_population'):
                dimension = getattr(results, 'dimension', results.final_population.shape[1])
                if dimension == 2:
                    save_path = f"{folders['population']}/{algorithm_name}_{problem_name}_population_{timestamp}"
                    from .visualization_extended import plot_population_2d
                    plot_population_2d(
                        results.final_population,
                        getattr(results, 'final_fitness', np.sum(results.final_population**2, axis=1)),
                        getattr(results, 'best_solution', results.final_population[0]),
                        getattr(results, 'iteration', 0),
                        getattr(results, 'bounds', [(-5, 5), (-5, 5)]),
                        save_path=save_path,
                        file_formats=formats,
                        dpi=dpi
                    )
                    figures_generated.append(f"Population: {save_path}")
            pbar.update(1)
        except Exception as e:
            print(f"  ⚠ Population plot failed: {e}")
            pbar.update(1)
        
        # 3. Parameter evolution (if available)
        try:
            if EXTENDED_AVAILABLE and hasattr(results, 'f_history') and hasattr(results, 'cr_history'):
                if results.f_history and results.cr_history:
                    save_path = f"{folders['parameters']}/{algorithm_name}_{problem_name}_parameters_{timestamp}"
                    from .visualization_extended import plot_parameter_evolution
                    plot_parameter_evolution(results.f_history, results.cr_history,
                                            save_path=save_path, file_formats=formats, dpi=dpi)
                    figures_generated.append(f"Parameters: {save_path}")
            pbar.update(1)
        except Exception as e:
            print(f"  ⚠ Parameter plot failed: {e}")
            pbar.update(1)
        
        # 4. Diversity
        try:
            if EXTENDED_AVAILABLE and hasattr(results, 'diversity_history'):
                if results.diversity_history:
                    save_path = f"{folders['diversity']}/{algorithm_name}_{problem_name}_diversity_{timestamp}"
                    from .visualization_extended import plot_diversity
                    plot_diversity(results.diversity_history,
                                  title=f"Diversity: {algorithm_name} on {problem_name}",
                                  save_path=save_path, file_formats=formats, dpi=dpi)
                    figures_generated.append(f"Diversity: {save_path}")
            pbar.update(1)
        except Exception as e:
            print(f"  ⚠ Diversity plot failed: {e}")
            pbar.update(1)
        
        # 5. Animation (optional)
        if generate_animation:
            try:
                if EXTENDED_AVAILABLE and hasattr(results, 'population_history'):
                    dimension = getattr(results, 'dimension', 2)
                    if dimension == 2:
                        save_path = f"{folders['animations']}/{algorithm_name}_{problem_name}_animation_{timestamp}.gif"
                        from .visualization_extended import animate_population_2d
                        animate_population_2d(
                            results.population_history,
                            results.fitness_history,
                            results.best_history,
                            getattr(results, 'bounds', [(-5, 5), (-5, 5)]),
                            save_path=save_path,
                            fps=10,
                            dpi=100
                        )
                        figures_generated.append(f"Animation: {save_path}")
            except Exception as e:
                print(f"  ⚠ Animation failed: {e}")
        pbar.update(1)
        
        # 6. Comprehensive report
        try:
            save_path = f"{folders['combined']}/{algorithm_name}_{problem_name}_report_{timestamp}"
            create_comprehensive_report(
                results,
                algorithm_name=algorithm_name,
                problem_name=problem_name,
                save_path=save_path,
                file_formats=formats,
                dpi=dpi
            )
            figures_generated.append(f"Report: {save_path}")
            pbar.update(1)
        except Exception as e:
            print(f"  ⚠ Comprehensive report failed: {e}")
            pbar.update(1)
    
    print(f"\n✓ Generated {len(figures_generated)} figure types")
    print(f"  Saved to: {base_save_path}/")
    print(f"{'='*60}\n")
    
    return folders
