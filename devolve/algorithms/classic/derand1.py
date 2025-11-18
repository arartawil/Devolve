"""
DE/rand/1 - Classic Differential Evolution with rand/1 mutation.

This is the original and most widely used DE variant, providing a good
balance between exploration and exploitation.
"""

from typing import Optional, Callable
import numpy as np
from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...core.logger import OptimizationLogger
from ...operators.mutation import rand_1
from ...operators.crossover import binomial_crossover, exponential_crossover


class DErand1(BaseDifferentialEvolution):
    """
    DE/rand/1 algorithm with binomial or exponential crossover.
    
    This is the classic DE variant that uses random base vector with one
    difference vector for mutation.
    
    Mutation strategy:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_{r_1} + F \\cdot (\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})
    
    where :math:`r_1, r_2, r_3` are mutually exclusive random indices
    different from the target index :math:`i`.
    
    Parameters
    ----------
    problem : Problem
        The optimization problem to solve.
    population_size : int, optional
        Number of individuals in the population. Default is 50.
        Recommended: 5*D to 10*D where D is dimensionality.
    max_iterations : int, optional
        Maximum number of generations. Default is 1000.
    F : float, optional
        Scaling factor for mutation (typically 0.4-1.0). Default is 0.8.
        Larger values increase exploration, smaller values increase exploitation.
    CR : float, optional
        Crossover probability (0.0-1.0). Default is 0.9.
        Higher values make the algorithm more aggressive.
    crossover_strategy : str, optional
        Crossover strategy: 'binomial' or 'exponential'. Default is 'binomial'.
    boundary_strategy : str, optional
        Strategy for handling boundary violations. Default is "clip".
    early_stopping : bool, optional
        Enable early stopping when converged. Default is False.
    early_stopping_tolerance : float, optional
        Tolerance for early stopping. Default is 1e-6.
    early_stopping_generations : int, optional
        Number of generations with no improvement to trigger early stop. Default is 50.
    random_seed : Optional[int], optional
        Random seed for reproducibility. Default is None.
    callback : Optional[Callable], optional
        Callback function called after each iteration. Default is None.
    logger : Optional[OptimizationLogger], optional
        Logger instance for tracking progress. Default is None.
    
    Attributes
    ----------
    crossover_strategy : str
        The crossover strategy being used.
    
    Examples
    --------
    >>> from devolve import DErand1, Problem
    >>> import numpy as np
    >>> 
    >>> # Define a simple sphere function
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>> 
    >>> # Create problem
    >>> problem = Problem(
    ...     objective_function=sphere,
    ...     bounds=(-5.0, 5.0),
    ...     dimensions=10,
    ...     optimum=0.0,
    ...     name="Sphere"
    ... )
    >>> 
    >>> # Create and run optimizer
    >>> optimizer = DErand1(
    ...     problem=problem,
    ...     population_size=50,
    ...     max_iterations=500,
    ...     F=0.8,
    ...     CR=0.9
    ... )
    >>> best_position, best_fitness = optimizer.optimize()
    >>> print(f"Best fitness: {best_fitness:.6f}")
    
    Notes
    -----
    DE/rand/1 characteristics:
    - **Exploration**: High - random base vector promotes diversity
    - **Convergence speed**: Moderate
    - **Risk of premature convergence**: Low
    - **Best for**: Multimodal problems, problems requiring robust search
    - **Population size**: Typically 5*D to 10*D
    - **Parameter tuning**: 
        - F ∈ [0.5, 0.9]: balanced
        - F ∈ [0.4, 0.5]: exploitation
        - F ∈ [0.9, 1.0]: exploration
        - CR ∈ [0.8, 1.0]: recommended for most problems
    
    References
    ----------
    Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient
    heuristic for global optimization over continuous spaces. Journal of global
    optimization, 11(4), 341-359. DOI: 10.1023/A:1008202821328
    
    Price, K., Storn, R. M., & Lampinen, J. A. (2006). Differential evolution:
    a practical approach to global optimization. Springer Science & Business Media.
    
    See Also
    --------
    DEbest1 : Faster convergence but higher risk of premature convergence
    DErand2 : More diverse search with two difference vectors
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: int = 50,
        max_iterations: int = 1000,
        F: float = 0.8,
        CR: float = 0.9,
        crossover_strategy: str = 'binomial',
        boundary_strategy: str = "clip",
        early_stopping: bool = False,
        early_stopping_tolerance: float = 1e-6,
        early_stopping_generations: int = 50,
        random_seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        logger: Optional[OptimizationLogger] = None
    ):
        """Initialize DE/rand/1 algorithm."""
        super().__init__(
            problem=problem,
            population_size=population_size,
            max_iterations=max_iterations,
            F=F,
            CR=CR,
            boundary_strategy=boundary_strategy,
            early_stopping=early_stopping,
            early_stopping_tolerance=early_stopping_tolerance,
            early_stopping_generations=early_stopping_generations,
            random_seed=random_seed,
            callback=callback,
            logger=logger
        )
        
        self.crossover_strategy = crossover_strategy.lower()
        if self.crossover_strategy not in ['binomial', 'exponential']:
            raise ValueError("crossover_strategy must be 'binomial' or 'exponential'")
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """
        Create mutant vector using DE/rand/1 mutation.
        
        Selects three random individuals (r1, r2, r3) distinct from target
        and creates mutant: v = x_r1 + F * (x_r2 - x_r3)
        
        Parameters
        ----------
        target_idx : int
            Index of the target individual.
        
        Returns
        -------
        np.ndarray
            Mutant vector.
        """
        return rand_1(
            population=self.population,
            target_idx=target_idx,
            F=self.F,
            rng=self.rng
        )
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Create trial vector using binomial or exponential crossover.
        
        Parameters
        ----------
        target : np.ndarray
            Target individual's position vector.
        mutant : np.ndarray
            Mutant vector from mutation operation.
        
        Returns
        -------
        np.ndarray
            Trial vector.
        """
        if self.crossover_strategy == 'binomial':
            return binomial_crossover(target, mutant, self.CR, self.rng)
        else:  # exponential
            return exponential_crossover(target, mutant, self.CR, self.rng)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"DE/rand/1(population_size={self.population_size}, "
                f"F={self.F}, CR={self.CR}, "
                f"crossover={self.crossover_strategy})")
