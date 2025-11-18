"""
DE/best/1 - Differential Evolution with best/1 mutation strategy.

Fast convergence but higher risk of premature convergence.
"""

from typing import Optional, Callable
import numpy as np
from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...core.logger import OptimizationLogger
from ...operators.mutation import best_1
from ...operators.crossover import binomial_crossover, exponential_crossover


class DEbest1(BaseDifferentialEvolution):
    """
    DE/best/1 algorithm - uses best individual as base vector.
    
    Mutation strategy:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_{best} + F \\cdot (\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2})
    
    This variant converges faster than DE/rand/1 but has higher risk of
    getting stuck in local optima. Best for unimodal or simple multimodal problems.
    
    Parameters
    ----------
    problem : Problem
        The optimization problem to solve.
    population_size : int, optional
        Number of individuals. Default is 50.
    max_iterations : int, optional
        Maximum generations. Default is 1000.
    F : float, optional
        Scaling factor. Default is 0.8.
    CR : float, optional
        Crossover probability. Default is 0.9.
    crossover_strategy : str, optional
        'binomial' or 'exponential'. Default is 'binomial'.
    
    References
    ----------
    Storn, R., & Price, K. (1997). DOI: 10.1023/A:1008202821328
    """
    
    def __init__(self, problem: Problem, population_size: int = 50,
                 max_iterations: int = 1000, F: float = 0.8, CR: float = 0.9,
                 crossover_strategy: str = 'binomial', **kwargs):
        super().__init__(problem, population_size, max_iterations, F, CR, **kwargs)
        self.crossover_strategy = crossover_strategy.lower()
    
    def mutate(self, target_idx: int) -> np.ndarray:
        return best_1(self.population, target_idx, self.F, self.rng)
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        if self.crossover_strategy == 'binomial':
            return binomial_crossover(target, mutant, self.CR, self.rng)
        return exponential_crossover(target, mutant, self.CR, self.rng)
