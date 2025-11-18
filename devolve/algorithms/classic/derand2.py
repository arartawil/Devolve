"""DE/rand/2 with two difference vectors."""

from typing import Optional, Callable
import numpy as np
from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...operators.mutation import rand_2
from ...operators.crossover import binomial_crossover, exponential_crossover


class DErand2(BaseDifferentialEvolution):
    """
    DE/rand/2 algorithm.
    
    Mutation: v_i = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    
    More diverse search, slower convergence. Requires population_size >= 6.
    """
    
    def __init__(self, problem: Problem, population_size: int = 50,
                 max_iterations: int = 1000, F: float = 0.8, CR: float = 0.9,
                 crossover_strategy: str = 'binomial', **kwargs):
        if population_size < 6:
            raise ValueError("DE/rand/2 requires population_size >= 6")
        super().__init__(problem, population_size, max_iterations, F, CR, **kwargs)
        self.crossover_strategy = crossover_strategy.lower()
    
    def mutate(self, target_idx: int) -> np.ndarray:
        return rand_2(self.population, target_idx, self.F, self.rng)
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        if self.crossover_strategy == 'binomial':
            return binomial_crossover(target, mutant, self.CR, self.rng)
        return exponential_crossover(target, mutant, self.CR, self.rng)
