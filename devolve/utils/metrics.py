"""
Performance Metrics for Optimization

Provides functions for calculating common performance metrics used in
evolutionary algorithm benchmarking and comparison.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union


def calculate_error(best_fitness: float, optimal_value: float) -> float:
    """
    Calculate the error between obtained fitness and known optimum.
    
    Parameters
    ----------
    best_fitness : float
        The best fitness value obtained by the algorithm.
    optimal_value : float
        The known optimal value of the problem.
    
    Returns
    -------
    float
        Absolute error |best_fitness - optimal_value|.
    
    Examples
    --------
    >>> error = calculate_error(0.001, 0.0)
    >>> print(f"Error: {error:.6f}")
    Error: 0.001000
    """
    return abs(best_fitness - optimal_value)


def calculate_success_rate(
    runs: List[float],
    target_error: float,
    optimal_value: float = 0.0
) -> float:
    """
    Calculate the success rate of achieving target error.
    
    Success is defined as achieving a fitness within target_error of the
    optimal value.
    
    Parameters
    ----------
    runs : List[float]
        List of best fitness values from multiple runs.
    target_error : float
        Target error threshold for success.
    optimal_value : float, optional
        The known optimal value. Default is 0.0.
    
    Returns
    -------
    float
        Success rate as a fraction in [0, 1].
    
    Examples
    --------
    >>> runs = [0.0001, 0.001, 0.1, 0.005]
    >>> sr = calculate_success_rate(runs, target_error=0.01, optimal_value=0.0)
    >>> print(f"Success rate: {sr:.2%}")
    Success rate: 75.00%
    """
    if not runs:
        return 0.0
    
    successes = sum(1 for fitness in runs 
                   if abs(fitness - optimal_value) <= target_error)
    return successes / len(runs)


def calculate_ert(
    runs: List[Tuple[float, int]],
    target_error: float,
    optimal_value: float = 0.0
) -> Optional[float]:
    """
    Calculate Expected Running Time (ERT).
    
    ERT is the expected number of function evaluations to reach the target
    error, accounting for unsuccessful runs.
    
    ERT = (sum of FEs for successful runs) / (number of successful runs)
          if success_rate > 0, else infinity.
    
    More precisely: ERT = mean_FEs / success_rate
    
    Parameters
    ----------
    runs : List[Tuple[float, int]]
        List of tuples (best_fitness, function_evaluations) from multiple runs.
    target_error : float
        Target error threshold.
    optimal_value : float, optional
        The known optimal value. Default is 0.0.
    
    Returns
    -------
    float or None
        Expected running time (average FEs accounting for failures).
        Returns None if no runs were successful.
    
    Examples
    --------
    >>> runs = [(0.001, 1000), (0.005, 1500), (0.1, 2000), (0.002, 1200)]
    >>> ert = calculate_ert(runs, target_error=0.01, optimal_value=0.0)
    >>> print(f"ERT: {ert:.0f} function evaluations")
    ERT: 1567 function evaluations
    
    References
    ----------
    Hansen, N., et al. (2016). COCO: A platform for comparing continuous
    optimizers in a black-box setting. arXiv:1603.08785.
    """
    if not runs:
        return None
    
    successful_runs = [(fitness, fes) for fitness, fes in runs
                      if abs(fitness - optimal_value) <= target_error]
    
    if not successful_runs:
        return None  # No successful runs
    
    # Success rate
    success_rate = len(successful_runs) / len(runs)
    
    # Average FEs for successful runs
    mean_fes = np.mean([fes for _, fes in successful_runs])
    
    # ERT accounts for failures
    ert = mean_fes / success_rate
    
    return float(ert)


def calculate_convergence_speed(
    history: Union[List[float], Dict[str, List[float]]],
    target_error: float,
    optimal_value: float = 0.0
) -> Dict[str, float]:
    """
    Calculate convergence speed metrics from optimization history.
    
    Metrics include:
    - Iterations to target (first time reaching target error)
    - Average improvement rate
    - Linear convergence rate (if applicable)
    - Final convergence
    
    Parameters
    ----------
    history : List[float] or Dict[str, List[float]]
        Optimization history. Either a list of best fitness values per iteration,
        or a dict with 'best_fitness' key containing the list.
    target_error : float
        Target error threshold.
    optimal_value : float, optional
        Known optimal value. Default is 0.0.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with convergence metrics:
        - 'iterations_to_target': First iteration reaching target (-1 if never)
        - 'avg_improvement_rate': Average fitness improvement per iteration
        - 'linear_rate': Linear convergence rate (slope in log space)
        - 'final_error': Final error from optimum
        - 'convergence_achieved': Whether target was reached
    
    Examples
    --------
    >>> history = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    >>> metrics = calculate_convergence_speed(history, target_error=0.01)
    >>> print(f"Iterations to target: {metrics['iterations_to_target']}")
    Iterations to target: 4
    """
    # Extract fitness history
    if isinstance(history, dict):
        fitness_history = history.get('best_fitness', [])
    else:
        fitness_history = history
    
    if not fitness_history:
        return {
            'iterations_to_target': -1,
            'avg_improvement_rate': 0.0,
            'linear_rate': 0.0,
            'final_error': float('inf'),
            'convergence_achieved': False
        }
    
    # Calculate errors
    errors = [abs(f - optimal_value) for f in fitness_history]
    
    # Iterations to target
    iterations_to_target = -1
    for i, error in enumerate(errors):
        if error <= target_error:
            iterations_to_target = i
            break
    
    # Average improvement rate
    if len(fitness_history) > 1:
        improvements = [fitness_history[i] - fitness_history[i+1] 
                       for i in range(len(fitness_history)-1)]
        avg_improvement_rate = np.mean(improvements)
    else:
        avg_improvement_rate = 0.0
    
    # Linear convergence rate (in log space)
    linear_rate = 0.0
    if len(errors) > 2:
        # Filter out zeros/negatives for log
        log_errors = []
        log_iters = []
        for i, err in enumerate(errors):
            if err > 0:
                log_errors.append(np.log10(err))
                log_iters.append(i)
        
        if len(log_errors) > 1:
            # Linear regression in log space
            coeffs = np.polyfit(log_iters, log_errors, 1)
            linear_rate = coeffs[0]  # Slope
    
    return {
        'iterations_to_target': iterations_to_target,
        'avg_improvement_rate': float(avg_improvement_rate),
        'linear_rate': float(linear_rate),
        'final_error': float(errors[-1]),
        'convergence_achieved': iterations_to_target >= 0
    }


def calculate_auc(history: List[float], normalize: bool = True) -> float:
    """
    Calculate Area Under the Curve for convergence history.
    
    Lower AUC indicates faster convergence to good solutions.
    
    Parameters
    ----------
    history : List[float]
        Best fitness values over iterations.
    normalize : bool, optional
        If True, normalize by number of iterations. Default is True.
    
    Returns
    -------
    float
        Area under the convergence curve.
    
    Examples
    --------
    >>> history = [100, 50, 25, 12, 6, 3, 1.5]
    >>> auc = calculate_auc(history)
    >>> print(f"AUC: {auc:.2f}")
    """
    if not history or len(history) < 2:
        return 0.0
    
    # Trapezoidal integration
    auc = np.trapz(history)
    
    if normalize:
        auc = auc / len(history)
    
    return float(auc)


def calculate_stability(runs: List[float]) -> Dict[str, float]:
    """
    Calculate stability metrics across multiple runs.
    
    Parameters
    ----------
    runs : List[float]
        Best fitness values from multiple independent runs.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with stability metrics:
        - 'mean': Mean of best fitness values
        - 'std': Standard deviation
        - 'cv': Coefficient of variation (std/mean)
        - 'min': Best result
        - 'max': Worst result
        - 'median': Median result
        - 'iqr': Interquartile range
    
    Examples
    --------
    >>> runs = [0.001, 0.002, 0.0015, 0.0018, 0.0012]
    >>> stability = calculate_stability(runs)
    >>> print(f"CV: {stability['cv']:.4f}")
    """
    if not runs:
        return {
            'mean': 0.0,
            'std': 0.0,
            'cv': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'iqr': 0.0
        }
    
    runs_array = np.array(runs)
    mean = np.mean(runs_array)
    std = np.std(runs_array)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'cv': float(std / mean) if mean != 0 else float('inf'),
        'min': float(np.min(runs_array)),
        'max': float(np.max(runs_array)),
        'median': float(np.median(runs_array)),
        'iqr': float(np.percentile(runs_array, 75) - np.percentile(runs_array, 25))
    }


class PerformanceMetrics:
    """
    Calculate performance metrics for optimization runs.
    
    This class provides a convenient interface for computing various
    performance metrics from optimization results.
    
    Examples
    --------
    >>> metrics = PerformanceMetrics()
    >>> runs = [0.001, 0.005, 0.1, 0.002]
    >>> sr = metrics.success_rate(runs, target_error=0.01)
    >>> print(f"Success rate: {sr:.2%}")
    Success rate: 75.00%
    """
    
    @staticmethod
    def error(best_fitness: float, optimal_value: float) -> float:
        """Calculate error from optimum."""
        return calculate_error(best_fitness, optimal_value)
    
    @staticmethod
    def success_rate(
        runs: List[float],
        target_error: float,
        optimal_value: float = 0.0
    ) -> float:
        """Calculate success rate."""
        return calculate_success_rate(runs, target_error, optimal_value)
    
    @staticmethod
    def ert(
        runs: List[Tuple[float, int]],
        target_error: float,
        optimal_value: float = 0.0
    ) -> Optional[float]:
        """Calculate Expected Running Time."""
        return calculate_ert(runs, target_error, optimal_value)
    
    @staticmethod
    def convergence_speed(
        history: Union[List[float], Dict[str, List[float]]],
        target_error: float,
        optimal_value: float = 0.0
    ) -> Dict[str, float]:
        """Calculate convergence speed metrics."""
        return calculate_convergence_speed(history, target_error, optimal_value)
    
    @staticmethod
    def auc(history: List[float], normalize: bool = True) -> float:
        """Calculate area under convergence curve."""
        return calculate_auc(history, normalize)
    
    @staticmethod
    def stability(runs: List[float]) -> Dict[str, float]:
        """Calculate stability metrics."""
        return calculate_stability(runs)
