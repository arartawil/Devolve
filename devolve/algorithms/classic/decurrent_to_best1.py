"""DE/current-to-best/1 mutation strategy."""

from typing import Optional, Callable
import numpy as np
from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...operators.mutation import current_to_best_1
from ...operators.crossover import binomial_crossover, exponential_crossover


class DEcurrentToBest1(BaseDifferentialEvolution):
    """
    DE/current-to-best/1 algorithm.
    
    Mutation:  v_i = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    
    Balances exploration and exploitation. Good general-purpose variant.
    """
    
    def __init__(self, problem: Problem, population_size: int = 50,
                 max_iterations: int = 1000, F: float = 0.8, CR: float = 0.9,
                 crossover_strategy: str = 'binomial', **kwargs):
        super().__init__(problem, population_size, max_iterations, F, CR, **kwargs)
        self.crossover_strategy = crossover_strategy.lower()
    
    def mutate(self, target_idx: int) -> np.ndarray:
        return current_to_best_1(self.population, target_idx, self.F, self.rng)
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        if self.crossover_strategy == 'binomial':
            return binomial_crossover(target, mutant, self.CR, self.rng)
        return exponential_crossover(target, mutant, self.CR, self.rng)
