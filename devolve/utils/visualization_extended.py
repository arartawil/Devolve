"""
Extended Visualization Functions for DEvolve
Part 2: Population dynamics, 3D landscapes, animations, and advanced plots

This is a continuation of visualization.py with additional plotting functions.
Import from this module or use the combined interface from visualization.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Union, Tuple, Callable, Any

from .visualization import (
    OKABE_ITO, OKABE_ITO_COLORS, _save_figure, setup_figure_folders
)

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
# PART 3: POPULATION VISUALIZATION (2D)
# ============================================================================

def plot_population_2d(
    population: np.ndarray,
    fitness_values: np.ndarray,
    best_solution: np.ndarray,
    iteration: int,
    bounds: List[Tuple[float, float]],
    contour_function: Optional[Callable] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300,
    show_contour: bool = True,
    colormap: str = 'viridis',
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Scatter plot of population in 2D search space.
    
    Parameters:
    -----------
    population : np.ndarray
        Array of shape (N, 2) for 2D problems
    fitness_values : np.ndarray
        Array of shape (N,) containing fitness for each individual
    best_solution : np.ndarray
        Best individual position (shape: (2,))
    iteration : int
        Current iteration number
    bounds : List[Tuple[float, float]]
        Search space bounds [(x_min, x_max), (y_min, y_max)]
    contour_function : Callable, optional
        Function to plot contour (benchmark function)
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    show_contour : bool
        Whether to show contour plot as background
    colormap : str
        Colormap for fitness values
    file_formats : list
        File formats to save
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Example:
        >>> pop = np.random.uniform(-5, 5, (50, 2))
        >>> fitness = np.sum(pop**2, axis=1)
        >>> best = pop[np.argmin(fitness)]
        >>> fig = plot_population_2d(pop, fitness, best, iteration=10,
        ...                           bounds=[(-5, 5), (-5, 5)])
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot contour if function provided
    if show_contour and contour_function is not None:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        
        x = np.linspace(x_min, x_max, 200)
        y = np.linspace(y_min, y_max, 200)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function on grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = contour_function(np.array([X[i, j], Y[i, j]]))
        
        # Plot contour
        contour = ax.contour(X, Y, Z, levels=30, cmap='gray', alpha=0.3, linewidths=0.5)
        contourf = ax.contourf(X, Y, Z, levels=30, cmap='gray', alpha=0.1)
    
    # Scatter plot of population colored by fitness
    scatter = ax.scatter(population[:, 0], population[:, 1], 
                        c=fitness_values, cmap=colormap, s=100, 
                        edgecolors='black', linewidth=0.5, alpha=0.8,
                        label='Population')
    
    # Highlight best solution
    ax.scatter(best_solution[0], best_solution[1], 
              marker='*', s=500, color='red', edgecolors='black', 
              linewidth=2, zorder=10, label='Best Solution')
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Fitness Value', fontsize=12)
    
    # Formatting
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title(f'Population at Iteration {iteration}', fontsize=16, fontweight='bold')
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.legend(loc='best', frameon=False)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


def animate_population_2d(
    population_history: List[np.ndarray],
    fitness_history: List[np.ndarray],
    best_history: List[np.ndarray],
    bounds: List[Tuple[float, float]],
    contour_function: Optional[Callable] = None,
    save_path: str = "animation.gif",
    fps: int = 10,
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 100,
    interval: int = 100,
    colormap: str = 'viridis'
) -> None:
    """
    Create animated GIF or MP4 showing population evolution.
    
    Parameters:
    -----------
    population_history : List[np.ndarray]
        List of populations at each iteration
    fitness_history : List[np.ndarray]
        List of fitness values at each iteration
    best_history : List[np.ndarray]
        List of best solutions at each iteration
    bounds : List[Tuple[float, float]]
        Search space bounds
    contour_function : Callable, optional
        Function to plot contour background
    save_path : str
        Path to save animation (e.g., "animation.gif" or "animation.mp4")
    fps : int
        Frames per second
    figsize : tuple
        Figure size
    dpi : int
        Resolution (lower for faster generation)
    interval : int
        Milliseconds between frames
    colormap : str
        Colormap for fitness values
    
    Example:
        >>> # After optimization
        >>> animate_population_2d(
        ...     population_history=optimizer.population_history,
        ...     fitness_history=optimizer.fitness_history,
        ...     best_history=optimizer.best_history,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     save_path="figures/animations/evolution.gif"
        ... )
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare contour background (computed once)
    if contour_function is not None:
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        
        x = np.linspace(x_min, x_max, 150)
        y = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x, y)
        
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = contour_function(np.array([X[i, j], Y[i, j]]))
    
    # Initialize plot elements
    scatter = None
    best_scatter = None
    trail_line = None
    
    def update_frame(frame_num):
        nonlocal scatter, best_scatter, trail_line
        
        ax.clear()
        
        # Plot contour background
        if contour_function is not None:
            ax.contourf(X, Y, Z, levels=30, cmap='gray', alpha=0.1)
            ax.contour(X, Y, Z, levels=20, cmap='gray', alpha=0.3, linewidths=0.5)
        
        # Get current data
        population = population_history[frame_num]
        fitness = fitness_history[frame_num]
        best = best_history[frame_num]
        
        # Plot population
        scatter = ax.scatter(population[:, 0], population[:, 1],
                           c=fitness, cmap=colormap, s=100,
                           edgecolors='black', linewidth=0.5, alpha=0.8)
        
        # Plot best solution
        best_scatter = ax.scatter(best[0], best[1],
                                 marker='*', s=500, color='red',
                                 edgecolors='black', linewidth=2, zorder=10)
        
        # Plot trail of best solutions (last 10 iterations)
        if frame_num > 0:
            trail_length = min(10, frame_num)
            trail = np.array(best_history[max(0, frame_num-trail_length):frame_num+1])
            trail_line, = ax.plot(trail[:, 0], trail[:, 1], 'r--', 
                                 linewidth=2, alpha=0.5, label='Best Trail')
        
        # Formatting
        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        ax.set_title(f'Iteration {frame_num}', fontsize=16, fontweight='bold')
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add iteration counter text
        ax.text(0.02, 0.98, f'Iteration: {frame_num}',
               transform=ax.transAxes, fontsize=14, fontweight='bold',
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    # Create animation
    print(f"Generating animation with {len(population_history)} frames...")
    anim = FuncAnimation(fig, update_frame, frames=len(population_history),
                        interval=interval, repeat=True, blit=False)
    
    # Save animation
    if save_path.endswith('.gif'):
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    elif save_path.endswith('.mp4'):
        anim.save(save_path, writer='ffmpeg', fps=fps, dpi=dpi)
    else:
        anim.save(save_path + '.gif', writer=PillowWriter(fps=fps), dpi=dpi)
    
    plt.close()
    print(f"Animation saved to: {save_path}")


# ============================================================================
# PART 4: 3D SURFACE PLOTS
# ============================================================================

def plot_3d_landscape(
    function: Callable,
    bounds: List[Tuple[float, float]],
    population: Optional[np.ndarray] = None,
    best_solution: Optional[np.ndarray] = None,
    resolution: int = 100,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 9),
    dpi: int = 300,
    elevation: float = 30,
    azimuth: float = 45,
    colormap: str = 'viridis',
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    3D surface plot of fitness landscape.
    
    Parameters:
    -----------
    function : Callable
        Benchmark function to plot
    bounds : List[Tuple[float, float]]
        Search space bounds [(x_min, x_max), (y_min, y_max)]
    population : np.ndarray, optional
        Current population (N, 2)
    best_solution : np.ndarray, optional
        Best solution (2,)
    resolution : int
        Grid resolution
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    dpi : int
        Resolution
    elevation : float
        View angle elevation (degrees)
    azimuth : float
        View angle azimuth (degrees)
    colormap : str
        Colormap for surface
    file_formats : list
        File formats to save
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    
    Example:
        >>> def sphere(x):
        ...     return np.sum(x**2)
        >>> fig = plot_3d_landscape(sphere, bounds=[(-5, 5), (-5, 5)])
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh grid
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function(np.array([X[i, j], Y[i, j]]))
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=0.8,
                          linewidth=0, antialiased=True, shade=True)
    
    # Plot population if provided
    if population is not None:
        pop_z = np.array([function(ind) for ind in population])
        ax.scatter(population[:, 0], population[:, 1], pop_z,
                  c='black', marker='o', s=50, alpha=0.6, label='Population')
    
    # Plot best solution if provided
    if best_solution is not None:
        best_z = function(best_solution)
        ax.scatter(best_solution[0], best_solution[1], best_z,
                  c='red', marker='*', s=300, edgecolors='black',
                  linewidths=2, label='Best Solution')
    
    # Colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Formatting
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_zlabel('Fitness', fontsize=12)
    ax.set_title('3D Fitness Landscape', fontsize=16, fontweight='bold')
    ax.view_init(elev=elevation, azim=azimuth)
    
    if population is not None or best_solution is not None:
        ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


# ============================================================================
# PART 5: PARAMETER EVOLUTION (for Adaptive DE)
# ============================================================================

def plot_parameter_evolution(
    f_history: Dict[str, List[float]],
    cr_history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    dpi: int = 300,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Plot evolution of F and CR parameters over iterations.
    
    Parameters:
    -----------
    f_history : Dict[str, List[float]]
        History of F values {'mean': [...], 'std': [...], 'min': [...], 'max': [...]}
        Or just {'values': [...]} for single values per iteration
    cr_history : Dict[str, List[float]]
        History of CR values (same format as f_history)
    save_path : str, optional
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
    
    Example:
        >>> f_hist = {'mean': [0.5, 0.6, 0.7], 'std': [0.1, 0.1, 0.05]}
        >>> cr_hist = {'mean': [0.9, 0.85, 0.8], 'std': [0.05, 0.06, 0.04]}
        >>> fig = plot_parameter_evolution(f_hist, cr_hist)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # Parse F history
    if 'mean' in f_history:
        iterations = list(range(len(f_history['mean'])))
        f_mean = f_history['mean']
        f_std = f_history.get('std', None)
        f_min = f_history.get('min', None)
        f_max = f_history.get('max', None)
    else:
        iterations = list(range(len(f_history.get('values', []))))
        f_mean = f_history.get('values', [])
        f_std = None
        f_min = None
        f_max = None
    
    # Parse CR history
    if 'mean' in cr_history:
        cr_mean = cr_history['mean']
        cr_std = cr_history.get('std', None)
        cr_min = cr_history.get('min', None)
        cr_max = cr_history.get('max', None)
    else:
        cr_mean = cr_history.get('values', [])
        cr_std = None
        cr_min = None
        cr_max = None
    
    # Plot F parameter
    ax1.plot(iterations, f_mean, linewidth=2.5, color=OKABE_ITO['blue'], label='Mean F')
    
    if f_std is not None:
        f_mean_arr = np.array(f_mean)
        f_std_arr = np.array(f_std)
        ax1.fill_between(iterations, f_mean_arr - f_std_arr, f_mean_arr + f_std_arr,
                        alpha=0.3, color=OKABE_ITO['blue'], label='±1 Std Dev')
    
    if f_min is not None and f_max is not None:
        ax1.plot(iterations, f_min, linewidth=1.0, linestyle=':', 
                color=OKABE_ITO['blue'], alpha=0.6, label='Min/Max')
        ax1.plot(iterations, f_max, linewidth=1.0, linestyle=':',
                color=OKABE_ITO['blue'], alpha=0.6)
    
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_ylabel('F Parameter', fontsize=14)
    ax1.set_title('Parameter Evolution', fontsize=16, fontweight='bold')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='best', frameon=False)
    
    # Plot CR parameter
    ax2.plot(iterations, cr_mean, linewidth=2.5, color=OKABE_ITO['orange'], label='Mean CR')
    
    if cr_std is not None:
        cr_mean_arr = np.array(cr_mean)
        cr_std_arr = np.array(cr_std)
        ax2.fill_between(iterations, cr_mean_arr - cr_std_arr, cr_mean_arr + cr_std_arr,
                        alpha=0.3, color=OKABE_ITO['orange'], label='±1 Std Dev')
    
    if cr_min is not None and cr_max is not None:
        ax2.plot(iterations, cr_min, linewidth=1.0, linestyle=':',
                color=OKABE_ITO['orange'], alpha=0.6, label='Min/Max')
        ax2.plot(iterations, cr_max, linewidth=1.0, linestyle=':',
                color=OKABE_ITO['orange'], alpha=0.6)
    
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Iterations', fontsize=14)
    ax2.set_ylabel('CR Parameter', fontsize=14)
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


def plot_diversity(
    diversity_history: Union[List[float], Dict[str, List[float]]],
    title: str = "Population Diversity Over Time",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    file_formats: List[str] = ['png']
) -> plt.Figure:
    """
    Plot population diversity metrics over time.
    
    Parameters:
    -----------
    diversity_history : List[float] or Dict[str, List[float]]
        If list: Single diversity metric over time
        If dict: Multiple metrics {'metric1': [...], 'metric2': [...], ...}
    title : str
        Plot title
    save_path : str, optional
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
    
    Example:
        >>> div_hist = {
        ...     'std_dev': [10, 8, 6, 4, 2],
        ...     'avg_distance': [15, 12, 9, 6, 3]
        ... }
        >>> fig = plot_diversity(div_hist)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(diversity_history, list):
        # Single metric
        iterations = list(range(len(diversity_history)))
        ax.plot(iterations, diversity_history, linewidth=2.5,
               color=OKABE_ITO['blue'], marker='o',
               markevery=max(1, len(iterations)//10), label='Diversity')
    else:
        # Multiple metrics
        colors = OKABE_ITO_COLORS
        line_styles = LINE_STYLES
        
        for idx, (metric_name, values) in enumerate(diversity_history.items()):
            iterations = list(range(len(values)))
            color = colors[idx % len(colors)]
            linestyle = line_styles[idx % len(line_styles)]
            
            # Normalize to [0, 1] for comparison
            values_norm = np.array(values)
            if values_norm.max() > 0:
                values_norm = (values_norm - values_norm.min()) / (values_norm.max() - values_norm.min())
            
            ax.plot(iterations, values_norm, linewidth=2.0, linestyle=linestyle,
                   color=color, label=metric_name, marker='o',
                   markevery=max(1, len(iterations)//10), markersize=6)
    
    # Formatting
    ax.set_xlabel('Iterations', fontsize=14)
    ax.set_ylabel('Diversity (Normalized)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=False)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        _save_figure(fig, save_path, file_formats, dpi)
    
    return fig


def calculate_diversity(population: np.ndarray) -> float:
    """
    Calculate population diversity as average distance to centroid.
    
    Parameters:
    -----------
    population : np.ndarray
        Population array of shape (N, D)
    
    Returns:
    --------
    float
        Diversity measure
    
    Example:
        >>> pop = np.random.randn(50, 10)
        >>> diversity = calculate_diversity(pop)
    """
    centroid = np.mean(population, axis=0)
    distances = np.linalg.norm(population - centroid, axis=1)
    diversity = np.mean(distances)
    return diversity
