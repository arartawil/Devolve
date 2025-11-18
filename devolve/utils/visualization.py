"""
Comprehensive Visualization Module for Differential Evolution Algorithms

This module provides publication-quality plotting functions for:
- Convergence analysis
- Algorithm comparison
- Population dynamics
- Parameter evolution
- Statistical analysis
- 3D landscapes
- Animations

All figures are automatically saved in organized folder structures with
multiple format support (PNG, PDF, SVG, EPS).

Author: DEvolve Package
License: MIT
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback dummy tqdm
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0
        def __iter__(self):
            return iter(self.iterable) if self.iterable else []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def update(self, n=1):
            self.n += n


# ============================================================================
# COLOR PALETTES (Colorblind-Friendly)
# ============================================================================

# Okabe-Ito palette (colorblind safe)
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

OKABE_ITO_COLORS = list(OKABE_ITO.values())

# Line styles for multiple algorithms
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]


# ============================================================================
# PUBLICATION-READY STYLING
# ============================================================================

def set_publication_style() -> None:
    """
    Configure matplotlib for publication-quality figures.
    
    Settings:
    - Font: Times New Roman or Arial (serif)
    - Font size: 12pt for text, 14pt for labels, 16pt for titles
    - Line width: 2.0
    - DPI: 300
    - Figure size: width=10, height=6 (inches)
    - Grid: Light gray, dashed
    - Colors: Colorblind-friendly Okabe-Ito palette
    
    This should be called once at the beginning of your script.
    
    Example:
        >>> from devolve.utils.visualization import set_publication_style
        >>> set_publication_style()
    """
    plt.rcParams.update({
        # Figure
        'figure.figsize': (10, 6),
        'figure.dpi': 100,  # Display DPI (save DPI set separately)
        'figure.autolayout': True,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        
        # Lines
        'lines.linewidth': 2.0,
        'lines.markersize': 8,
        'lines.markeredgewidth': 1.0,
        
        # Grid
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.color': 'gray',
        
        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.framealpha': 0.8,
        
        # Axes
        'axes.linewidth': 1.0,
        'axes.edgecolor': 'black',
        'axes.axisbelow': True,  # Grid below data
        
        # Colors (colorblind-friendly Okabe-Ito)
        'axes.prop_cycle': cycler('color', OKABE_ITO_COLORS)
    })


# ============================================================================
# FOLDER ORGANIZATION
# ============================================================================

def setup_figure_folders(base_path: str = "figures") -> Dict[str, str]:
    """
    Create organized folder structure for saving figures.
    
    Parameters:
    -----------
    base_path : str
        Base directory path for all figures (default: "figures")
    
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping category names to their folder paths
    
    Folder Structure:
    -----------------
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
    
    Example:
        >>> folders = setup_figure_folders("my_results/figures")
        >>> print(folders['convergence'])
        'my_results/figures/convergence'
    """
    folders = {
        'convergence': os.path.join(base_path, 'convergence'),
        'population': os.path.join(base_path, 'population'),
        'comparison': os.path.join(base_path, 'comparison'),
        'parameters': os.path.join(base_path, 'parameters'),
        'diversity': os.path.join(base_path, 'diversity'),
        'statistical': os.path.join(base_path, 'statistical'),
        'animations': os.path.join(base_path, 'animations'),
        '3d_landscapes': os.path.join(base_path, '3d_landscapes'),
        'combined': os.path.join(base_path, 'combined'),
        'tables': os.path.join(base_path, 'tables')
    }
    
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def _save_figure(fig: plt.Figure, save_path: str, 
                 file_formats: List[str] = ['png'], dpi: int = 300) -> None:
    """
    Save figure in multiple formats.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    save_path : str
        Base path without extension
    file_formats : List[str]
        List of file formats (e.g., ['png', 'pdf', 'svg', 'eps'])
    dpi : int
        Resolution for raster formats (default: 300)
    """
    for fmt in file_formats:
        full_path = f"{save_path}.{fmt}"
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight', format=fmt)
        print(f"Saved: {full_path}")


# ============================================================================
# PART 1: CONVERGENCE PLOTS
# ============================================================================

def plot_convergence(
    history: Union[Dict[str, List], List[float]],
    title: str = "Convergence Plot",
    xlabel: str = "Iterations",
    ylabel: str = "Fitness Value",
    log_scale: bool = False,
    show_grid: bool = True,
    show_mean: bool = False,
    show_std: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Plot convergence curve showing best fitness over iterations.
    
    Parameters:
    -----------
    history : dict or list
        If dict: {'iteration': [...], 'best_fitness': [...], 
                  'mean_fitness': [...], 'std_fitness': [...]}
        If list: Just best fitness values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    log_scale : bool
        Use logarithmic scale for y-axis (useful for large fitness ranges)
    show_grid : bool
        Display grid lines
    show_mean : bool
        Plot mean fitness line (if available in history)
    show_std : bool
        Plot standard deviation as shaded area (if available)
    save_path : str, optional
        Path to save figure (without extension)
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution for saved figure
    file_formats : list
        List of file formats to save (e.g., ['png', 'pdf', 'svg'])
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Example:
        >>> history = {'best_fitness': [100, 50, 25, 10, 5, 2]}
        >>> fig = plot_convergence(history, title="DE/rand/1 on Sphere")
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Parse history
    if isinstance(history, dict):
        iterations = history.get('iteration', list(range(len(history.get('best_fitness', [])))))
        best_fitness = history.get('best_fitness', [])
        mean_fitness = history.get('mean_fitness', None)
        std_fitness = history.get('std_fitness', None)
    else:
        iterations = list(range(len(history)))
        best_fitness = history
        mean_fitness = None
        std_fitness = None
    
    # Plot best fitness (main line)
    ax.plot(iterations, best_fitness, linewidth=2.5, label='Best Fitness', 
            color=OKABE_ITO['blue'], marker='o', markevery=max(1, len(iterations)//10))
    
    # Plot mean fitness (if requested and available)
    if show_mean and mean_fitness is not None:
        ax.plot(iterations, mean_fitness, linewidth=2.0, linestyle='--', 
                label='Mean Fitness', color=OKABE_ITO['orange'], alpha=0.8)
    
    # Plot standard deviation (if requested and available)
    if show_std and mean_fitness is not None and std_fitness is not None:
        mean_arr = np.array(mean_fitness)
        std_arr = np.array(std_fitness)
        ax.fill_between(iterations, mean_arr - std_arr, mean_arr + std_arr,
                        alpha=0.2, color=OKABE_ITO['orange'], label='±1 Std Dev')
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.grid(show_grid, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


def plot_convergence_with_ci(
    runs_data: List[List[float]],
    confidence_level: float = 0.95,
    title: str = "Convergence with Confidence Intervals",
    xlabel: str = "Iterations",
    ylabel: str = "Fitness Value",
    log_scale: bool = False,
    show_median: bool = True,
    show_mean: bool = True,
    show_best: bool = True,
    show_worst: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Convergence plot with confidence intervals from multiple runs.
    
    Parameters:
    -----------
    runs_data : List[List[float]]
        List of fitness histories from multiple runs
    confidence_level : float
        Confidence level for interval (default: 0.95 for 95% CI)
    show_median : bool
        Show median line
    show_mean : bool
        Show mean line
    show_best : bool
        Show best run (thin line)
    show_worst : bool
        Show worst run (thin line)
    ... (other parameters same as plot_convergence)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Example:
        >>> runs = [[100, 50, 25], [110, 55, 28], [95, 48, 24]]
        >>> fig = plot_convergence_with_ci(runs, title="10 Runs of DE/rand/1")
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy array (pad if needed)
    max_len = max(len(run) for run in runs_data)
    padded_runs = []
    for run in runs_data:
        padded = np.full(max_len, np.nan)
        padded[:len(run)] = run
        padded_runs.append(padded)
    
    data_array = np.array(padded_runs)  # Shape: (n_runs, n_iterations)
    iterations = np.arange(max_len)
    
    # Calculate statistics (ignoring NaN values)
    with np.errstate(invalid='ignore'):
        median = np.nanmedian(data_array, axis=0)
        mean = np.nanmean(data_array, axis=0)
        
        # Confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.nanpercentile(data_array, lower_percentile, axis=0)
        ci_upper = np.nanpercentile(data_array, upper_percentile, axis=0)
        
        best_run = np.nanmin(data_array, axis=0)
        worst_run = np.nanmax(data_array, axis=0)
    
    # Plot confidence interval
    ax.fill_between(iterations, ci_lower, ci_upper, alpha=0.3, 
                    color=OKABE_ITO['blue'], 
                    label=f'{int(confidence_level*100)}% Confidence Interval')
    
    # Plot median
    if show_median:
        ax.plot(iterations, median, linewidth=2.5, color=OKABE_ITO['blue'], 
                label='Median', marker='o', markevery=max(1, max_len//10))
    
    # Plot mean
    if show_mean:
        ax.plot(iterations, mean, linewidth=2.0, linestyle='--', 
                color=OKABE_ITO['orange'], label='Mean', alpha=0.8)
    
    # Plot best run
    if show_best:
        ax.plot(iterations, best_run, linewidth=1.0, linestyle=':', 
                color=OKABE_ITO['green'], label='Best Run', alpha=0.6)
    
    # Plot worst run
    if show_worst:
        ax.plot(iterations, worst_run, linewidth=1.0, linestyle=':', 
                color=OKABE_ITO['vermillion'], label='Worst Run', alpha=0.6)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


# ============================================================================
# PART 2: ALGORITHM COMPARISON
# ============================================================================

def plot_algorithm_comparison(
    results_dict: Dict[str, Union[List[float], Dict]],
    title: str = "Algorithm Comparison",
    xlabel: str = "Iterations",
    ylabel: str = "Best Fitness",
    log_scale: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 7),
    dpi: int = 300,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Compare convergence of multiple algorithms on the same plot.
    
    Parameters:
    -----------
    results_dict : Dict[str, Union[List, Dict]]
        Dictionary mapping algorithm names to their history
        Format: {'Algorithm1': [fitness_values], 'Algorithm2': {...}, ...}
        Or: {'Algorithm1': {'best_fitness': [...]}, ...}
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    log_scale : bool
        Use logarithmic scale for y-axis
    save_path : str, optional
        Path to save figure (without extension)
    figsize : tuple
        Figure size (width, height) in inches
    dpi : int
        Resolution for saved figure
    file_formats : list
        List of file formats to save
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Example:
        >>> results = {
        ...     'DE/rand/1': [100, 50, 25, 10],
        ...     'DE/best/1': [100, 45, 20, 8],
        ...     'JADE': [100, 40, 15, 5]
        ... }
        >>> fig = plot_algorithm_comparison(results)
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = OKABE_ITO_COLORS
    line_styles = LINE_STYLES
    
    for idx, (algo_name, history) in enumerate(results_dict.items()):
        color = colors[idx % len(colors)]
        linestyle = line_styles[idx % len(line_styles)]
        
        # Parse history
        if isinstance(history, dict):
            fitness_values = history.get('best_fitness', [])
        else:
            fitness_values = history
        
        iterations = list(range(len(fitness_values)))
        
        # Plot line
        ax.plot(iterations, fitness_values, linewidth=2.0, linestyle=linestyle,
                color=color, label=algo_name, marker='o', 
                markevery=max(1, len(iterations)//10), markersize=6)
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=False, fontsize=11)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


def plot_statistical_comparison(
    results: Dict[str, List[float]],
    metric: str = 'best_fitness',
    title: str = "Statistical Comparison",
    ylabel: str = "Fitness Value",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    plot_type: str = 'box',
    show_significance: bool = False,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Statistical comparison of algorithms over multiple runs using box plots or violin plots.
    
    Parameters:
    -----------
    results : Dict[str, List[float]]
        Dictionary mapping algorithm names to list of final fitness values
        Format: {'Algo1': [run1, run2, ...], 'Algo2': [...], ...}
    metric : str
        Name of the metric being compared
    title : str
        Plot title
    ylabel : str
        Y-axis label
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    plot_type : str
        'box', 'violin', or 'boxen'
    show_significance : bool
        Show statistical significance markers (requires scipy)
    file_formats : list
        File formats to save
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Example:
        >>> results = {
        ...     'DE/rand/1': [10.5, 11.2, 9.8, 10.1, 10.7],
        ...     'JADE': [8.5, 9.1, 8.2, 8.8, 8.9],
        ...     'L-SHADE': [5.2, 5.5, 5.1, 5.3, 5.4]
        ... }
        >>> fig = plot_statistical_comparison(results, plot_type='violin')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    if SEABORN_AVAILABLE:
        import pandas as pd
        data_list = []
        for algo_name, values in results.items():
            for value in values:
                data_list.append({'Algorithm': algo_name, metric: value})
        df = pd.DataFrame(data_list)
        
        # Create plot with seaborn
        import seaborn as sns
        if plot_type == 'box':
            sns.boxplot(data=df, x='Algorithm', y=metric, ax=ax, palette=OKABE_ITO_COLORS)
            sns.stripplot(data=df, x='Algorithm', y=metric, ax=ax, color='black', 
                         alpha=0.3, size=3, jitter=True)
        elif plot_type == 'violin':
            sns.violinplot(data=df, x='Algorithm', y=metric, ax=ax, palette=OKABE_ITO_COLORS)
            sns.stripplot(data=df, x='Algorithm', y=metric, ax=ax, color='black',
                         alpha=0.3, size=3, jitter=True)
        elif plot_type == 'boxen':
            sns.boxenplot(data=df, x='Algorithm', y=metric, ax=ax, palette=OKABE_ITO_COLORS)
        
        # Mark means
        means = df.groupby('Algorithm')[metric].mean()
        for idx, (algo, mean_val) in enumerate(means.items()):
            ax.plot(idx, mean_val, marker='D', markersize=10, color='red', 
                    markeredgecolor='black', markeredgewidth=1.5, label='Mean' if idx == 0 else '')
    else:
        # Fallback to matplotlib boxplot
        data_values = list(results.values())
        labels = list(results.keys())
        ax.boxplot(data_values, labels=labels)
        
        # Add mean markers
        for idx, values in enumerate(data_values, 1):
            ax.plot(idx, np.mean(values), marker='D', markersize=10, color='red',
                    markeredgecolor='black', markeredgewidth=1.5)
    
    # Statistical significance (if requested)
    if show_significance:
        try:
            from scipy import stats
            algo_names = list(results.keys())
            n_algos = len(algo_names)
            
            # Perform pairwise t-tests
            for i in range(n_algos - 1):
                data1 = results[algo_names[i]]
                data2 = results[algo_names[i + 1]]
                _, p_value = stats.ttest_ind(data1, data2)
                
                # Determine significance
                if p_value < 0.001:
                    sig_marker = '***'
                elif p_value < 0.01:
                    sig_marker = '**'
                elif p_value < 0.05:
                    sig_marker = '*'
                else:
                    sig_marker = 'ns'
                
                # Add marker above bars
                y_max = max(max(data1), max(data2))
                ax.text((i + i + 1) / 2, y_max * 1.05, sig_marker, 
                       ha='center', fontsize=12, fontweight='bold')
        except ImportError:
            pass  # scipy not available
    
    # Formatting
    ax.set_xlabel('Algorithm', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    if SEABORN_AVAILABLE:
        ax.legend(loc='best', frameon=False)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig
