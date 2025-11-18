"""
Parallel Evaluation Utilities

Provides parallel function evaluation using multiprocessing for speeding up
population-based optimization algorithms.
"""

import numpy as np
from typing import Callable, List, Optional, Any
from multiprocessing import Pool, cpu_count
from functools import partial


def parallel_evaluate(
    population: np.ndarray,
    function: Callable,
    n_jobs: int = -1,
    use_tqdm: bool = True,
    chunk_size: Optional[int] = None
) -> np.ndarray:
    """
    Evaluate a population in parallel using multiprocessing.
    
    Parameters
    ----------
    population : np.ndarray
        Population matrix of shape (population_size, dimensions).
        Each row is an individual to evaluate.
    function : Callable
        Objective function that takes a 1D array and returns a scalar.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all available CPUs. Default is -1.
    use_tqdm : bool, optional
        Show progress bar with tqdm. Default is True.
    chunk_size : int, optional
        Size of chunks for parallel processing. If None, automatically determined.
        Default is None.
    
    Returns
    -------
    np.ndarray
        Array of fitness values, shape (population_size,).
    
    Examples
    --------
    >>> import numpy as np
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>> 
    >>> population = np.random.randn(100, 10)
    >>> fitness = parallel_evaluate(population, sphere, n_jobs=4)
    >>> print(f"Best fitness: {np.min(fitness):.6f}")
    
    Notes
    -----
    - Parallel evaluation is beneficial for expensive objective functions.
    - For cheap functions, overhead may make sequential evaluation faster.
    - The function must be picklable (defined at module level, not lambda).
    """
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    n_jobs = min(n_jobs, len(population))
    
    # For very small populations, use sequential evaluation
    if len(population) < n_jobs or n_jobs == 1:
        if use_tqdm:
            try:
                from tqdm import tqdm
                return np.array([function(ind) for ind in tqdm(population, desc="Evaluating")])
            except ImportError:
                pass
        return np.array([function(ind) for ind in population])
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = max(1, len(population) // (n_jobs * 4))
    
    # Parallel evaluation
    try:
        with Pool(processes=n_jobs) as pool:
            if use_tqdm:
                try:
                    from tqdm import tqdm
                    fitness_values = list(tqdm(
                        pool.imap(function, population, chunksize=chunk_size),
                        total=len(population),
                        desc=f"Evaluating (n_jobs={n_jobs})"
                    ))
                except ImportError:
                    fitness_values = pool.map(function, population, chunksize=chunk_size)
            else:
                fitness_values = pool.map(function, population, chunksize=chunk_size)
        
        return np.array(fitness_values)
    
    except Exception as e:
        # Fallback to sequential evaluation on error
        print(f"Warning: Parallel evaluation failed ({e}), falling back to sequential.")
        return np.array([function(ind) for ind in population])


def parallel_optimize(
    optimizer_class: Any,
    problem: Any,
    n_runs: int,
    n_jobs: int = -1,
    seeds: Optional[List[int]] = None,
    **optimizer_kwargs
) -> List[tuple]:
    """
    Run multiple optimization runs in parallel.
    
    Useful for statistical testing and benchmarking.
    
    Parameters
    ----------
    optimizer_class : class
        The optimizer class to use (e.g., JADE, LSHADE).
    problem : Problem
        The optimization problem.
    n_runs : int
        Number of independent runs.
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all CPUs. Default is -1.
    seeds : List[int], optional
        Random seeds for each run. If None, uses sequential seeds.
    **optimizer_kwargs
        Additional arguments passed to optimizer constructor.
    
    Returns
    -------
    List[tuple]
        List of (best_x, best_f) tuples from each run.
    
    Examples
    --------
    >>> from devolve import JADE, Sphere
    >>> problem = Sphere(dimensions=10)
    >>> results = parallel_optimize(
    ...     JADE, problem, n_runs=10, n_jobs=4,
    ...     population_size=50, max_iterations=100
    ... )
    >>> fitness_values = [f for _, f in results]
    >>> print(f"Mean: {np.mean(fitness_values):.6e}")
    >>> print(f"Std: {np.std(fitness_values):.6e}")
    """
    # Generate seeds if not provided
    if seeds is None:
        seeds = list(range(n_runs))
    elif len(seeds) != n_runs:
        raise ValueError(f"Number of seeds ({len(seeds)}) must match n_runs ({n_runs})")
    
    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1
    
    n_jobs = min(n_jobs, n_runs)
    
    # Define single run function
    def single_run(seed: int) -> tuple:
        """Run a single optimization."""
        optimizer = optimizer_class(
            problem=problem,
            random_seed=seed,
            **optimizer_kwargs
        )
        return optimizer.optimize()
    
    # Run in parallel
    if n_jobs == 1:
        # Sequential execution
        try:
            from tqdm import tqdm
            results = [single_run(seed) for seed in tqdm(seeds, desc="Optimization runs")]
        except ImportError:
            results = [single_run(seed) for seed in seeds]
    else:
        # Parallel execution
        try:
            with Pool(processes=n_jobs) as pool:
                try:
                    from tqdm import tqdm
                    results = list(tqdm(
                        pool.imap(single_run, seeds),
                        total=n_runs,
                        desc=f"Optimization runs (n_jobs={n_jobs})"
                    ))
                except ImportError:
                    results = pool.map(single_run, seeds)
        except Exception as e:
            print(f"Warning: Parallel execution failed ({e}), using sequential.")
            results = [single_run(seed) for seed in seeds]
    
    return results


def get_optimal_n_jobs(
    population_size: int,
    function_cost: str = 'medium'
) -> int:
    """
    Get recommended number of parallel jobs based on problem characteristics.
    
    Parameters
    ----------
    population_size : int
        Size of the population to evaluate.
    function_cost : str, optional
        Cost of function evaluation: 'cheap', 'medium', or 'expensive'.
        Default is 'medium'.
    
    Returns
    -------
    int
        Recommended number of parallel jobs.
    
    Examples
    --------
    >>> n_jobs = get_optimal_n_jobs(population_size=100, function_cost='expensive')
    >>> print(f"Recommended n_jobs: {n_jobs}")
    
    Notes
    -----
    - 'cheap': Functions taking < 1ms (e.g., Sphere, Rosenbrock)
    - 'medium': Functions taking 1-100ms
    - 'expensive': Functions taking > 100ms (e.g., simulations, neural networks)
    """
    max_cpus = cpu_count()
    
    if function_cost == 'cheap':
        # For cheap functions, overhead dominates
        # Use fewer cores or sequential
        if population_size < 50:
            return 1
        elif population_size < 200:
            return min(2, max_cpus)
        else:
            return min(4, max_cpus)
    
    elif function_cost == 'medium':
        # Moderate parallelization
        if population_size < 20:
            return min(2, max_cpus)
        elif population_size < 100:
            return min(max_cpus // 2, max_cpus)
        else:
            return max_cpus
    
    elif function_cost == 'expensive':
        # Maximize parallelization
        return min(population_size, max_cpus)
    
    else:
        raise ValueError(f"Unknown function_cost: {function_cost}")


def benchmark_parallel_speedup(
    function: Callable,
    population: np.ndarray,
    n_jobs_list: Optional[List[int]] = None
) -> dict:
    """
    Benchmark speedup from parallel evaluation.
    
    Tests different numbers of parallel jobs to find optimal configuration.
    
    Parameters
    ----------
    function : Callable
        Objective function to benchmark.
    population : np.ndarray
        Population to evaluate.
    n_jobs_list : List[int], optional
        List of n_jobs values to test. If None, tests [1, 2, 4, ..., max_cpus].
    
    Returns
    -------
    dict
        Dictionary with n_jobs as keys and (time, speedup) as values.
    
    Examples
    --------
    >>> import numpy as np
    >>> def expensive_function(x):
    ...     import time
    ...     time.sleep(0.01)  # Simulate expensive computation
    ...     return np.sum(x**2)
    >>> 
    >>> population = np.random.randn(100, 10)
    >>> results = benchmark_parallel_speedup(expensive_function, population)
    >>> for n_jobs, (time_taken, speedup) in results.items():
    ...     print(f"n_jobs={n_jobs}: {time_taken:.2f}s (speedup={speedup:.2f}x)")
    """
    import time
    
    if n_jobs_list is None:
        max_cpus = cpu_count()
        n_jobs_list = [1]
        n = 2
        while n <= max_cpus:
            n_jobs_list.append(n)
            n *= 2
        if max_cpus not in n_jobs_list:
            n_jobs_list.append(max_cpus)
    
    results = {}
    baseline_time = None
    
    for n_jobs in n_jobs_list:
        start = time.time()
        _ = parallel_evaluate(population, function, n_jobs=n_jobs, use_tqdm=False)
        elapsed = time.time() - start
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        results[n_jobs] = (elapsed, speedup)
    
    return results
