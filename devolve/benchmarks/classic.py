"""Classic benchmark functions for optimization."""

import numpy as np
from ..core.problem import Problem


class Sphere(Problem):
    """
    Sphere function: f(x) = Σ x_i^2
    
    Global minimum: f(0,...,0) = 0
    Bounds: typically [-5.12, 5.12]
    Unimodal, convex, separable
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-5.12, 5.12)):
        def sphere_func(x):
            return np.sum(x**2)
        
        super().__init__(
            objective_function=sphere_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=np.zeros(dimensions),
            name="Sphere"
        )


class Rosenbrock(Problem):
    """
    Rosenbrock function: f(x) = Σ[100(x_{i+1} - x_i^2)^2 + (1-x_i)^2]
    
    Global minimum: f(1,...,1) = 0
    Bounds: typically [-2.048, 2.048] or [-5, 10]
    Unimodal, non-convex, non-separable, valley-shaped
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-2.048, 2.048)):
        def rosenbrock_func(x):
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        super().__init__(
            objective_function=rosenbrock_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=np.ones(dimensions),
            name="Rosenbrock"
        )


class Rastrigin(Problem):
    """
    Rastrigin function: f(x) = 10n + Σ[x_i^2 - 10cos(2πx_i)]
    
    Global minimum: f(0,...,0) = 0
    Bounds: typically [-5.12, 5.12]
    Highly multimodal, separable
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-5.12, 5.12)):
        def rastrigin_func(x):
            return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        
        super().__init__(
            objective_function=rastrigin_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=np.zeros(dimensions),
            name="Rastrigin"
        )


class Ackley(Problem):
    """
    Ackley function
    
    Global minimum: f(0,...,0) = 0
    Bounds: typically [-32.768, 32.768]
    Multimodal, non-separable
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-32.768, 32.768)):
        def ackley_func(x):
            n = len(x)
            sum_sq = np.sum(x**2)
            sum_cos = np.sum(np.cos(2 * np.pi * x))
            return (-20 * np.exp(-0.2 * np.sqrt(sum_sq / n)) -
                    np.exp(sum_cos / n) + 20 + np.e)
        
        super().__init__(
            objective_function=ackley_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=np.zeros(dimensions),
            name="Ackley"
        )


class Griewank(Problem):
    """
    Griewank function
    
    Global minimum: f(0,...,0) = 0
    Bounds: typically [-600, 600]
    Multimodal
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-600, 600)):
        def griewank_func(x):
            sum_sq = np.sum(x**2 / 4000)
            prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
            return sum_sq - prod_cos + 1
        
        super().__init__(
            objective_function=griewank_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=np.zeros(dimensions),
            name="Griewank"
        )


class Schwefel(Problem):
    """
    Schwefel function
    
    Global minimum: f(420.9687,...,420.9687) = 0
    Bounds: typically [-500, 500]
    Multimodal, deceptive
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-500, 500)):
        def schwefel_func(x):
            return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))
        
        optimum_pos = np.full(dimensions, 420.9687)
        super().__init__(
            objective_function=schwefel_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=optimum_pos,
            name="Schwefel"
        )


class Michalewicz(Problem):
    """
    Michalewicz function
    
    Global minimum varies with dimension
    Bounds: [0, π]
    Multimodal, steep ridges
    """
    
    def __init__(self, dimensions: int = 10, m: int = 10):
        def michalewicz_func(x):
            i = np.arange(1, len(x) + 1)
            return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi)**(2*m))
        
        # Known optima for specific dimensions
        known_optima = {
            2: -1.8011,
            5: -4.687658,
            10: -9.66015
        }
        
        super().__init__(
            objective_function=michalewicz_func,
            bounds=(0, np.pi),
            dimensions=dimensions,
            optimum=known_optima.get(dimensions),
            name="Michalewicz"
        )


class Zakharov(Problem):
    """
    Zakharov function
    
    Global minimum: f(0,...,0) = 0
    Bounds: typically [-5, 10] or [-10, 10]
    Unimodal
    """
    
    def __init__(self, dimensions: int = 10, bounds: tuple = (-5, 10)):
        def zakharov_func(x):
            i = np.arange(1, len(x) + 1)
            sum1 = np.sum(x**2)
            sum2 = np.sum(0.5 * i * x)
            return sum1 + sum2**2 + sum2**4
        
        super().__init__(
            objective_function=zakharov_func,
            bounds=bounds,
            dimensions=dimensions,
            optimum=0.0,
            optimum_position=np.zeros(dimensions),
            name="Zakharov"
        )


# Registry of all benchmark functions
BENCHMARK_FUNCTIONS = {
    'sphere': Sphere,
    'rosenbrock': Rosenbrock,
    'rastrigin': Rastrigin,
    'ackley': Ackley,
    'griewank': Griewank,
    'schwefel': Schwefel,
    'michalewicz': Michalewicz,
    'zakharov': Zakharov,
}


def get_benchmark(name: str, dimensions: int = 10, **kwargs) -> Problem:
    """
    Get a benchmark problem by name.
    
    Parameters
    ----------
    name : str
        Name of the benchmark function.
    dimensions : int
        Number of dimensions.
    **kwargs
        Additional arguments for the benchmark.
    
    Returns
    -------
    Problem
        The benchmark problem instance.
    """
    name_lower = name.lower()
    if name_lower not in BENCHMARK_FUNCTIONS:
        available = list(BENCHMARK_FUNCTIONS.keys())
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    
    return BENCHMARK_FUNCTIONS[name_lower](dimensions=dimensions, **kwargs)
