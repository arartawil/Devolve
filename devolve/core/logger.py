"""
Logging system for tracking optimization progress.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import numpy as np


class OptimizationLogger:
    """
    Tracks and logs optimization progress during Differential Evolution runs.
    
    The logger records fitness values, parameters, diversity metrics, and timing
    information throughout the optimization process. Supports logging to console,
    file, and structured data export.
    
    Parameters
    ----------
    log_to_console : bool, optional
        If True, print progress to console. Default is True.
    log_to_file : bool, optional
        If True, write logs to file. Default is False.
    log_file : str, optional
        Path to log file. Default is "optimization.log".
    verbose : int, optional
        Verbosity level (0=minimal, 1=normal, 2=detailed). Default is 1.
    log_interval : int, optional
        Log every N iterations. Default is 1 (log every iteration).
    
    Attributes
    ----------
    history : Dict[str, List]
        Dictionary storing logged data across iterations.
    start_time : float
        Timestamp when optimization started.
    
    Examples
    --------
    >>> logger = OptimizationLogger(verbose=1, log_interval=10)
    >>> logger.start()
    >>> for i in range(100):
    ...     logger.log_iteration(
    ...         iteration=i,
    ...         best_fitness=10.5 - i*0.1,
    ...         mean_fitness=15.0 - i*0.1,
    ...         diversity=1.0 - i*0.01
    ...     )
    >>> logger.end()
    >>> print(logger.get_summary())
    """
    
    def __init__(
        self,
        log_to_console: bool = True,
        log_to_file: bool = False,
        log_file: str = "optimization.log",
        verbose: int = 1,
        log_interval: int = 1
    ):
        """Initialize the logger."""
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.log_file = Path(log_file)
        self.verbose = verbose
        self.log_interval = log_interval
        
        # History storage
        self.history: Dict[str, List[Any]] = {
            'iteration': [],
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'std_fitness': [],
            'diversity': [],
            'feasibility_ratio': [],
            'function_evaluations': [],
            'elapsed_time': [],
        }
        
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        # Open log file if needed
        if self.log_to_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                f.write("=== Differential Evolution Optimization Log ===\n")
                f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def start(self) -> None:
        """Mark the start of optimization."""
        self.start_time = time.time()
        if self.log_to_console and self.verbose > 0:
            print("="*60)
            print("Starting Differential Evolution Optimization")
            print("="*60)
    
    def end(self) -> None:
        """Mark the end of optimization."""
        self.end_time = time.time()
        duration = self.get_elapsed_time()
        
        if self.log_to_console and self.verbose > 0:
            print("="*60)
            print(f"Optimization completed in {duration:.2f} seconds")
            print("="*60)
        
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\n=== Optimization Completed ===\n")
                f.write(f"Total time: {duration:.2f} seconds\n")
    
    def log_iteration(
        self,
        iteration: int,
        best_fitness: float,
        mean_fitness: Optional[float] = None,
        worst_fitness: Optional[float] = None,
        std_fitness: Optional[float] = None,
        diversity: Optional[float] = None,
        feasibility_ratio: Optional[float] = None,
        function_evaluations: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Log information for a single iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        best_fitness : float
            Best fitness value in current population.
        mean_fitness : float, optional
            Mean fitness of population.
        worst_fitness : float, optional
            Worst fitness in population.
        std_fitness : float, optional
            Standard deviation of fitness.
        diversity : float, optional
            Population diversity metric.
        feasibility_ratio : float, optional
            Ratio of feasible individuals.
        function_evaluations : int, optional
            Total function evaluations so far.
        **kwargs
            Additional custom metrics to log.
        """
        # Only log at specified intervals
        if iteration % self.log_interval != 0 and iteration != 0:
            return
        
        # Store in history
        self.history['iteration'].append(iteration)
        self.history['best_fitness'].append(best_fitness)
        self.history['mean_fitness'].append(mean_fitness)
        self.history['worst_fitness'].append(worst_fitness)
        self.history['std_fitness'].append(std_fitness)
        self.history['diversity'].append(diversity)
        self.history['feasibility_ratio'].append(feasibility_ratio)
        self.history['function_evaluations'].append(function_evaluations)
        self.history['elapsed_time'].append(self.get_elapsed_time())
        
        # Store custom metrics
        for key, value in kwargs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # Console output
        if self.log_to_console and self.verbose > 0:
            self._print_iteration(iteration, best_fitness, mean_fitness, 
                                 diversity, feasibility_ratio)
        
        # File output
        if self.log_to_file:
            self._write_to_file(iteration, best_fitness, mean_fitness,
                               diversity, feasibility_ratio)
    
    def _print_iteration(
        self,
        iteration: int,
        best_fitness: float,
        mean_fitness: Optional[float],
        diversity: Optional[float],
        feasibility_ratio: Optional[float]
    ) -> None:
        """Print iteration info to console."""
        if self.verbose == 1:
            # Compact format
            msg = f"Iter {iteration:5d}: Best={best_fitness:.6e}"
            if mean_fitness is not None:
                msg += f", Mean={mean_fitness:.6e}"
            print(msg)
        elif self.verbose >= 2:
            # Detailed format
            print(f"\n--- Iteration {iteration} ---")
            print(f"  Best Fitness: {best_fitness:.6e}")
            if mean_fitness is not None:
                print(f"  Mean Fitness: {mean_fitness:.6e}")
            if diversity is not None:
                print(f"  Diversity: {diversity:.6f}")
            if feasibility_ratio is not None:
                print(f"  Feasibility: {feasibility_ratio:.2%}")
            print(f"  Elapsed Time: {self.get_elapsed_time():.2f}s")
    
    def _write_to_file(
        self,
        iteration: int,
        best_fitness: float,
        mean_fitness: Optional[float],
        diversity: Optional[float],
        feasibility_ratio: Optional[float]
    ) -> None:
        """Write iteration info to file."""
        with open(self.log_file, 'a') as f:
            f.write(f"Iteration {iteration}: ")
            f.write(f"Best={best_fitness:.6e}")
            if mean_fitness is not None:
                f.write(f", Mean={mean_fitness:.6e}")
            if diversity is not None:
                f.write(f", Diversity={diversity:.6f}")
            if feasibility_ratio is not None:
                f.write(f", Feasibility={feasibility_ratio:.2%}")
            f.write("\n")
    
    def get_elapsed_time(self) -> float:
        """
        Get elapsed time since optimization started.
        
        Returns
        -------
        float
            Elapsed time in seconds, or 0.0 if not started.
        """
        if self.start_time is None:
            return 0.0
        current = self.end_time if self.end_time else time.time()
        return current - self.start_time
    
    def get_history(self, key: str) -> List[Any]:
        """
        Get history for a specific metric.
        
        Parameters
        ----------
        key : str
            Name of the metric.
        
        Returns
        -------
        List
            List of values for that metric.
        """
        return self.history.get(key, [])
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the optimization run.
        
        Returns
        -------
        Dict
            Dictionary with summary statistics.
        """
        if not self.history['best_fitness']:
            return {'status': 'No data logged'}
        
        best_fitness_values = [f for f in self.history['best_fitness'] if f is not None]
        
        summary = {
            'total_iterations': len(self.history['iteration']),
            'final_best_fitness': best_fitness_values[-1] if best_fitness_values else None,
            'best_fitness_overall': min(best_fitness_values) if best_fitness_values else None,
            'total_time': self.get_elapsed_time(),
            'convergence_achieved': self._check_convergence(),
        }
        
        return summary
    
    def _check_convergence(self, window: int = 10, tolerance: float = 1e-6) -> bool:
        """
        Check if optimization has converged.
        
        Parameters
        ----------
        window : int
            Number of recent iterations to check.
        tolerance : float
            Maximum change considered as converged.
        
        Returns
        -------
        bool
            True if converged, False otherwise.
        """
        if len(self.history['best_fitness']) < window:
            return False
        
        recent = self.history['best_fitness'][-window:]
        recent = [f for f in recent if f is not None]
        
        if not recent:
            return False
        
        fitness_range = max(recent) - min(recent)
        return fitness_range < tolerance
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export history to JSON file.
        
        Parameters
        ----------
        filepath : str
            Path to JSON file.
        """
        # Convert numpy types to Python types for JSON serialization
        export_data = {}
        for key, values in self.history.items():
            export_data[key] = [
                float(v) if isinstance(v, (np.floating, np.integer)) else v
                for v in values
            ]
        
        export_data['summary'] = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export history to CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file.
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = [key for key in self.history.keys() if self.history[key]]
            writer.writerow(headers)
            
            # Write data
            n_rows = len(self.history['iteration'])
            for i in range(n_rows):
                row = [self.history[key][i] if i < len(self.history[key]) else None
                      for key in headers]
                writer.writerow(row)
    
    def clear(self) -> None:
        """Clear all logged data."""
        for key in self.history:
            self.history[key].clear()
        self.start_time = None
        self.end_time = None
    
    def __repr__(self) -> str:
        """String representation."""
        n_iterations = len(self.history['iteration'])
        return f"OptimizationLogger(iterations={n_iterations}, verbose={self.verbose})"
