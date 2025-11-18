"""
L-SHADE: Linear Population Size Reduction SHADE

This module implements the L-SHADE algorithm proposed by Tanabe & Fukunaga (2014),
which extends SHADE with linear population size reduction for improved performance.

Reference:
    Tanabe, R., & Fukunaga, A. S. (2014).
    Improving the search performance of SHADE using linear population size reduction.
    In 2014 IEEE Congress on Evolutionary Computation (pp. 1658-1665).

Key Features:
- All SHADE features (historical memory, weighted means)
- Linear population size reduction from NP_init to NP_min
- NP_init typically set to 18*D
- NP_min typically set to 4
- Removal of worst individuals during reduction

Author: DEvolve Package
License: MIT
"""

from typing import Optional, List, Tuple
import numpy as np

from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...core.individual import Individual
from ...core.population import Population
from ...operators.crossover import binomial_crossover


class LSHADE(BaseDifferentialEvolution):
    """
    L-SHADE: Linear Population Size Reduction SHADE
    
    L-SHADE enhances SHADE by gradually reducing the population size during
    optimization. This allows for aggressive exploration early on and focused
    exploitation later.
    
    Mathematical Formulation:
    -------------------------
    1. Population Size Schedule:
       NP(G) = round(NP_min + (NP_init - NP_min) * (max_FEs - FEs) / max_FEs)
       
       where:
       - NP_init = 18 * D (initial population size)
       - NP_min = 4 (minimum population size)
       - FEs = function evaluations
       - max_FEs = maximum function evaluations
    
    2. Population Reduction:
       When NP(G) < current_NP:
           Remove (current_NP - NP(G)) worst individuals
    
    3. Memory and Archive (inherited from SHADE):
       Same as SHADE with H=5-10, weighted means, etc.
    
    4. Mutation and Crossover:
       Same as SHADE (DE/current-to-pbest/1 with archive)
    
    Parameters:
    -----------
    problem : Problem
        The optimization problem to solve
    population_size : int, optional
        Initial population size (default: 18 * dimensions)
    max_iterations : int, default=1000
        Maximum number of iterations
    H : int, default=5
        Size of historical memory
    p : float, default=0.11
        Proportion of top individuals for pbest selection
    NP_min : int, default=4
        Minimum population size
    archive_rate : float, default=2.6
        Archive size multiplier (archive_size = population_size * archive_rate)
    seed : int, optional
        Random seed for reproducibility
    
    Attributes:
    -----------
    NP_init : int
        Initial population size
    NP_min : int
        Minimum population size
    current_NP : int
        Current population size
    M_F : np.ndarray
        Historical memory for F parameters
    M_CR : np.ndarray
        Historical memory for CR parameters
    archive : List[np.ndarray]
        External archive
    
    Example:
    --------
    >>> from devolve import LSHADE
    >>> from devolve.benchmarks import Schwefel
    >>> 
    >>> # Create problem
    >>> problem = Schwefel(dimensions=30)
    >>> 
    >>> # Create optimizer (NP_init will be 18*30 = 540)
    >>> optimizer = LSHADE(
    ...     problem=problem,
    ...     max_iterations=1000,
    ...     H=5,
    ...     NP_min=4
    ... )
    >>> 
    >>> # Run optimization
    >>> best_solution, best_fitness = optimizer.optimize()
    >>> print(f"Best fitness: {best_fitness:.6e}")
    
    Notes:
    ------
    - Population reduction balances exploration and exploitation
    - NP_init = 18*D is recommended for most problems
    - NP_min = 4 works well, but can be adjusted (4-10)
    - Archive size is typically 2-3 times current population size
    - Best performing algorithm in CEC2014 competition
    
    References:
    -----------
    [1] Tanabe, R., & Fukunaga, A. S. (2014).
        Improving the search performance of SHADE using linear population
        size reduction. IEEE Congress on Evolutionary Computation, 1658-1665.
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: Optional[int] = None,
        max_iterations: int = 1000,
        H: int = 5,
        p: float = 0.11,
        NP_min: int = 4,
        archive_rate: float = 2.6,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize L-SHADE optimizer."""
        # Set initial population size (18*D if not specified)
        if population_size is None:
            population_size = 18 * problem.dimensions
        
        super().__init__(
            problem=problem,
            population_size=population_size,
            max_iterations=max_iterations,
            F=0.5,
            CR=0.5,
            random_seed=seed,
            **kwargs
        )
        
        # L-SHADE specific parameters
        self.NP_init = population_size
        self.NP_min = NP_min
        self.current_NP = population_size
        self.H = H
        self.p = p
        self.archive_rate = archive_rate
        
        # Calculate max function evaluations
        self.max_FEs = max_iterations * population_size
        
        # Initialize historical memory
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.k = 0
        
        # External archive (larger than SHADE)
        self.archive: List[np.ndarray] = []
        self.max_archive_size = int(population_size * archive_rate)
        
        # Storage for successful parameters
        self._S_F: List[float] = []
        self._S_CR: List[float] = []
        self._delta_f: List[float] = []
        
        # Storage for current generation parameters
        self._current_F: np.ndarray = np.zeros(population_size)
        self._current_CR: np.ndarray = np.zeros(population_size)
        self._current_memory_idx: np.ndarray = np.zeros(population_size, dtype=int)
        
        # History tracking
        self.F_history: List[float] = []
        self.CR_history: List[float] = []
        self.NP_history: List[int] = []
    
    def initialize_population(self) -> None:
        """Initialize population and archive."""
        super().initialize_population()
        
        self.archive = []
        self.current_NP = self.NP_init
        
        if self.logger.verbose > 0:
            print(f"L-SHADE initialized with NP_init={self.NP_init}, "
                  f"NP_min={self.NP_min}, H={self.H}, p={self.p}")
    
    def _calculate_target_NP(self) -> int:
        """
        Calculate target population size based on function evaluations.
        
        Returns:
        --------
        int
            Target population size for current iteration
        
        Formula:
        --------
        NP = round(NP_min + (NP_init - NP_min) * (max_FEs - FEs) / max_FEs)
        """
        FEs = self.function_evaluations
        ratio = (self.max_FEs - FEs) / self.max_FEs
        target_NP = round(self.NP_min + (self.NP_init - self.NP_min) * ratio)
        
        # Ensure at least NP_min
        return max(self.NP_min, target_NP)
    
    def _reduce_population(self, target_NP: int) -> None:
        """
        Reduce population size by removing worst individuals.
        
        Parameters:
        -----------
        target_NP : int
            Target population size
        """
        if target_NP >= self.current_NP:
            return
        
        num_to_remove = self.current_NP - target_NP
        
        # Sort individuals by fitness (worst last)
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        sorted_indices = np.argsort(fitness_values)
        
        # Keep best target_NP individuals
        new_individuals = [self.population.individuals[i] for i in sorted_indices[:target_NP]]
        
        # Update population
        self.population.individuals = new_individuals
        self.current_NP = target_NP
        
        # Resize parameter arrays
        self._current_F = np.zeros(target_NP)
        self._current_CR = np.zeros(target_NP)
        self._current_memory_idx = np.zeros(target_NP, dtype=int)
    
    def _generate_F(self, memory_idx: int) -> float:
        """Generate F parameter using Cauchy distribution."""
        while True:
            F = self.rng.standard_cauchy() * 0.1 + self.M_F[memory_idx]
            if F > 0:
                break
        return min(F, 1.0)
    
    def _generate_CR(self, memory_idx: int) -> float:
        """Generate CR parameter using Normal distribution."""
        CR = self.rng.normal(self.M_CR[memory_idx], 0.1)
        return np.clip(CR, 0.0, 1.0)
    
    def _select_pbest(self) -> Individual:
        """Select a random individual from the top p% of the population."""
        p_best_size = max(2, int(self.p * self.current_NP))
        
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        sorted_indices = np.argsort(fitness_values)
        
        pbest_idx = self.rng.choice(sorted_indices[:p_best_size])
        return self.population.individuals[pbest_idx]
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """Perform DE/current-to-pbest/1 mutation with archive."""
        self._current_target_idx = target_idx
        
        # Select random memory index
        r_i = self.rng.integers(self.H)
        self._current_memory_idx[target_idx] = r_i
        
        # Generate parameters
        F_i = self._generate_F(r_i)
        CR_i = self._generate_CR(r_i)
        
        self._current_F[target_idx] = F_i
        self._current_CR[target_idx] = CR_i
        
        # Get target
        x_i = self.population.individuals[target_idx].position
        
        # Select pbest
        x_pbest = self._select_pbest().position
        
        # Select r1
        candidates = list(range(self.current_NP))
        candidates.remove(target_idx)
        r1 = self.rng.choice(candidates)
        x_r1 = self.population.individuals[r1].position
        
        # Select r2 from population ∪ archive
        if len(self.archive) > 0 and self.rng.random() < 0.5:
            x_r2 = self.archive[self.rng.integers(len(self.archive))]
        else:
            candidates.remove(r1)
            r2 = self.rng.choice(candidates)
            x_r2 = self.population.individuals[r2].position
        
        # Mutation
        mutant = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Perform binomial crossover."""
        idx = getattr(self, '_current_target_idx', 0)
        CR_i = self._current_CR[idx]
        
        trial = binomial_crossover(
            target=target,
            mutant=mutant,
            CR=CR_i,
            rng=self.rng
        )
        
        return trial
    
    def select(self, target: Individual, trial: Individual) -> Individual:
        """Select between target and trial."""
        idx = getattr(self, '_current_target_idx', 0)
        
        if trial.is_better_than(target):
            # Store successful parameters
            self._S_F.append(self._current_F[idx])
            self._S_CR.append(self._current_CR[idx])
            
            delta_f = abs(target.fitness - trial.fitness)
            self._delta_f.append(delta_f)
            
            # Add to archive
            self._add_to_archive(target.position.copy())
            
            selected = trial
        else:
            selected = target
        
        return selected
    
    def _add_to_archive(self, position: np.ndarray) -> None:
        """Add a solution to the external archive."""
        if len(self.archive) < self.max_archive_size:
            self.archive.append(position)
        else:
            idx = self.rng.integers(len(self.archive))
            self.archive[idx] = position
    
    def _update_memory(self) -> None:
        """Update historical memory using weighted means."""
        if len(self._S_F) == 0:
            return
        
        S_F = np.array(self._S_F)
        S_CR = np.array(self._S_CR)
        weights = np.array(self._delta_f)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Weighted Lehmer mean for F
        mean_L_F = np.sum(weights * S_F**2) / np.sum(weights * S_F)
        
        # Weighted arithmetic mean for CR
        mean_A_CR = np.sum(weights * S_CR)
        
        # Update memory
        self.M_F[self.k] = mean_L_F
        self.M_CR[self.k] = mean_A_CR
        
        self.k = (self.k + 1) % self.H
        
        # Clear success sets
        self._S_F = []
        self._S_CR = []
        self._delta_f = []
    
    def _log_iteration(self, iteration: int) -> None:
        """Override to update memory, reduce population, and track statistics."""
        # Update memory
        self._update_memory()
        
        # Calculate target population size
        target_NP = self._calculate_target_NP()
        
        # Reduce population if needed
        if target_NP < self.current_NP:
            self._reduce_population(target_NP)
        
        # Track history
        self.F_history.append(np.mean(self.M_F))
        self.CR_history.append(np.mean(self.M_CR))
        self.NP_history.append(self.current_NP)
        
        # Call base logging
        super()._log_iteration(iteration)
        
        # Log info
        if iteration % 100 == 0 and self.logger.verbose > 1:
            print(f"Iteration {iteration}: NP={self.current_NP}, "
                  f"M_F={np.mean(self.M_F):.3f}, M_CR={np.mean(self.M_CR):.3f}, "
                  f"Archive={len(self.archive)}")
    
    def get_parameter_statistics(self) -> dict:
        """Get statistics about parameter evolution."""
        return {
            'M_F': self.M_F.copy(),
            'M_CR': self.M_CR.copy(),
            'mean_F': np.mean(self.M_F),
            'mean_CR': np.mean(self.M_CR),
            'F_history': self.F_history,
            'CR_history': self.CR_history,
            'NP_history': self.NP_history,
            'current_NP': self.current_NP,
            'NP_init': self.NP_init,
            'NP_min': self.NP_min,
            'archive_size': len(self.archive)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LSHADE(NP_init={self.NP_init}, NP_min={self.NP_min}, "
            f"current_NP={self.current_NP}, H={self.H}, "
            f"mean_F={np.mean(self.M_F):.3f})"
        )
