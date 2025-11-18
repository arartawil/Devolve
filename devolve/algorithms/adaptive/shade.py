"""
SHADE: Success-History based Adaptive Differential Evolution

This module implements the SHADE algorithm proposed by Tanabe & Fukunaga (2013),
which maintains a historical memory of successful parameters and uses weighted
means for parameter adaptation.

Reference:
    Tanabe, R., & Fukunaga, A. (2013).
    Success-history based parameter adaptation for differential evolution.
    In 2013 IEEE Congress on Evolutionary Computation (pp. 71-78).

Key Features:
- Historical memory H of size 5-10 storing past successful parameters
- Weighted Lehmer mean for F parameter update
- Weighted arithmetic mean for CR parameter update
- Round-robin memory cell update strategy
- DE/current-to-pbest/1 mutation with archive

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


class SHADE(BaseDifferentialEvolution):
    """
    SHADE: Success-History based Adaptive Differential Evolution
    
    SHADE maintains a historical memory of successful parameter values and
    uses weighted means to adapt F and CR parameters. The memory is updated
    in a round-robin fashion.
    
    Mathematical Formulation:
    -------------------------
    1. Memory Initialization:
       M_F = [0.5, 0.5, ..., 0.5]  (H cells)
       M_CR = [0.5, 0.5, ..., 0.5]  (H cells)
    
    2. Parameter Generation (for each individual):
       Select random memory index r_i
       F_i ~ Cauchy(M_F[r_i], 0.1), truncated to [0, 1]
       CR_i ~ Normal(M_CR[r_i], 0.1), truncated to [0, 1]
    
    3. Mutation (DE/current-to-pbest/1):
       v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
    
    4. Memory Update (after generation):
       If S_F is not empty:
           M_F[k] = weighted_lehmer_mean(S_F, weights=Δf)
           M_CR[k] = weighted_arithmetic_mean(S_CR, weights=Δf)
           k = (k + 1) mod H
    
    Parameters:
    -----------
    problem : Problem
        The optimization problem to solve
    population_size : int, default=100
        Number of individuals in the population
    max_iterations : int, default=1000
        Maximum number of iterations
    H : int, default=5
        Size of historical memory (typically 5-10)
    p : float, default=0.11
        Proportion of top individuals for pbest selection
    archive_size : int, optional
        Size of external archive (default: population_size)
    seed : int, optional
        Random seed for reproducibility
    
    Attributes:
    -----------
    M_F : np.ndarray
        Historical memory for F parameters (size H)
    M_CR : np.ndarray
        Historical memory for CR parameters (size H)
    k : int
        Current memory index (for round-robin update)
    archive : List[np.ndarray]
        External archive of replaced solutions
    
    Example:
    --------
    >>> from devolve import SHADE
    >>> from devolve.benchmarks import Rastrigin
    >>> 
    >>> # Create problem
    >>> problem = Rastrigin(dimensions=30)
    >>> 
    >>> # Create optimizer
    >>> optimizer = SHADE(
    ...     problem=problem,
    ...     population_size=100,
    ...     max_iterations=1000,
    ...     H=5,
    ...     p=0.11
    ... )
    >>> 
    >>> # Run optimization
    >>> best_solution, best_fitness = optimizer.optimize()
    >>> print(f"Best fitness: {best_fitness:.6e}")
    
    Notes:
    ------
    - Historical memory H=5 works well for most problems
    - Larger H (up to 10) may help for noisy problems
    - Weighted Lehmer mean emphasizes larger F values
    - Round-robin update prevents memory staleness
    - Archive size equal to NP is recommended
    
    References:
    -----------
    [1] Tanabe, R., & Fukunaga, A. (2013).
        Success-history based parameter adaptation for differential evolution.
        IEEE Congress on Evolutionary Computation, 71-78.
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: int = 100,
        max_iterations: int = 1000,
        H: int = 5,
        p: float = 0.11,
        archive_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize SHADE optimizer."""
        super().__init__(
            problem=problem,
            population_size=population_size,
            max_iterations=max_iterations,
            F=0.5,  # Initial (not used directly)
            CR=0.5,  # Initial (not used directly)
            random_seed=seed,
            **kwargs
        )
        
        # SHADE specific parameters
        self.H = H
        self.p = p
        self.archive_size = archive_size if archive_size is not None else population_size
        
        # Initialize historical memory
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.k = 0  # Current memory index
        
        # External archive
        self.archive: List[np.ndarray] = []
        
        # Storage for successful parameters (per generation)
        self._S_F: List[float] = []
        self._S_CR: List[float] = []
        self._delta_f: List[float] = []
        self._memory_indices: List[int] = []  # Track which memory cells were used
        
        # Storage for current generation parameters
        self._current_F: np.ndarray = np.zeros(population_size)
        self._current_CR: np.ndarray = np.zeros(population_size)
        self._current_memory_idx: np.ndarray = np.zeros(population_size, dtype=int)
        
        # History tracking
        self.F_history: List[float] = []
        self.CR_history: List[float] = []
    
    def initialize_population(self) -> None:
        """Initialize population and archive."""
        super().initialize_population()
        
        # Initialize archive as empty
        self.archive = []
        
        if self.logger.verbose > 0:
            print(f"SHADE initialized with H={self.H}, p={self.p}")
    
    def _generate_F(self, memory_idx: int) -> float:
        """
        Generate F parameter using Cauchy distribution.
        
        Parameters:
        -----------
        memory_idx : int
            Index of memory cell to use
        
        Returns:
        --------
        float
            F value sampled from Cauchy(M_F[memory_idx], 0.1)
        """
        while True:
            F = self.rng.standard_cauchy() * 0.1 + self.M_F[memory_idx]
            if F > 0:
                break
        
        # Truncate to [0, 1]
        if F > 1.0:
            F = 1.0
        
        return F
    
    def _generate_CR(self, memory_idx: int) -> float:
        """
        Generate CR parameter using Normal distribution.
        
        Parameters:
        -----------
        memory_idx : int
            Index of memory cell to use
        
        Returns:
        --------
        float
            CR value sampled from Normal(M_CR[memory_idx], 0.1)
        """
        CR = self.rng.normal(self.M_CR[memory_idx], 0.1)
        return np.clip(CR, 0.0, 1.0)
    
    def _select_pbest(self) -> Individual:
        """Select a random individual from the top p% of the population."""
        p_best_size = max(2, int(self.p * self.population_size))
        
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        sorted_indices = np.argsort(fitness_values)
        
        pbest_idx = self.rng.choice(sorted_indices[:p_best_size])
        return self.population.individuals[pbest_idx]
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """
        Perform DE/current-to-pbest/1 mutation with archive.
        
        Parameters:
        -----------
        target_idx : int
            Index of the target individual
        
        Returns:
        --------
        np.ndarray
            Mutant vector
        """
        self._current_target_idx = target_idx
        
        # Select random memory index
        r_i = self.rng.integers(self.H)
        self._current_memory_idx[target_idx] = r_i
        
        # Generate F and CR from selected memory cell
        F_i = self._generate_F(r_i)
        CR_i = self._generate_CR(r_i)
        
        # Store parameters
        self._current_F[target_idx] = F_i
        self._current_CR[target_idx] = CR_i
        
        # Get target individual
        x_i = self.population.individuals[target_idx].position
        
        # Select pbest from top p%
        x_pbest = self._select_pbest().position
        
        # Select random individual r1
        candidates = list(range(self.population_size))
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
        
        # DE/current-to-pbest/1 mutation
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
        """Select between target and trial, tracking successful parameters."""
        idx = getattr(self, '_current_target_idx', 0)
        
        if trial.is_better_than(target):
            # Store successful parameters
            self._S_F.append(self._current_F[idx])
            self._S_CR.append(self._current_CR[idx])
            
            # Store fitness improvement
            delta_f = abs(target.fitness - trial.fitness)
            self._delta_f.append(delta_f)
            
            # Track which memory cell was used
            self._memory_indices.append(self._current_memory_idx[idx])
            
            # Add to archive
            self._add_to_archive(target.position.copy())
            
            selected = trial
        else:
            selected = target
        
        return selected
    
    def _add_to_archive(self, position: np.ndarray) -> None:
        """Add a solution to the external archive."""
        if len(self.archive) < self.archive_size:
            self.archive.append(position)
        else:
            idx = self.rng.integers(len(self.archive))
            self.archive[idx] = position
    
    def _update_memory(self) -> None:
        """
        Update historical memory using weighted means.
        
        Uses weighted Lehmer mean for F and weighted arithmetic mean for CR.
        Updates memory cell k in round-robin fashion.
        """
        if len(self._S_F) == 0:
            return
        
        # Convert to arrays
        S_F = np.array(self._S_F)
        S_CR = np.array(self._S_CR)
        weights = np.array(self._delta_f)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        # Weighted Lehmer mean for F
        mean_L_F = np.sum(weights * S_F**2) / np.sum(weights * S_F)
        
        # Weighted arithmetic mean for CR
        mean_A_CR = np.sum(weights * S_CR)
        
        # Update memory at position k
        self.M_F[self.k] = mean_L_F
        self.M_CR[self.k] = mean_A_CR
        
        # Advance memory index (round-robin)
        self.k = (self.k + 1) % self.H
        
        # Clear success sets
        self._S_F = []
        self._S_CR = []
        self._delta_f = []
        self._memory_indices = []
    
    def _log_iteration(self, iteration: int) -> None:
        """Override to update memory and track statistics."""
        # Update memory first
        self._update_memory()
        
        # Track parameter history (mean of memory)
        self.F_history.append(np.mean(self.M_F))
        self.CR_history.append(np.mean(self.M_CR))
        
        # Call base logging
        super()._log_iteration(iteration)
        
        # Log parameter info
        if iteration % 100 == 0 and self.logger.verbose > 1:
            print(f"Iteration {iteration}: "
                  f"M_F={np.mean(self.M_F):.3f}±{np.std(self.M_F):.3f}, "
                  f"M_CR={np.mean(self.M_CR):.3f}±{np.std(self.M_CR):.3f}, "
                  f"Archive={len(self.archive)}")
    
    def get_parameter_statistics(self) -> dict:
        """
        Get statistics about parameter evolution.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - 'M_F': Current F memory array
            - 'M_CR': Current CR memory array
            - 'mean_F': Mean of F memory
            - 'mean_CR': Mean of CR memory
            - 'F_history': History of mean F values
            - 'CR_history': History of mean CR values
            - 'archive_size': Current archive size
        """
        return {
            'M_F': self.M_F.copy(),
            'M_CR': self.M_CR.copy(),
            'mean_F': np.mean(self.M_F),
            'mean_CR': np.mean(self.M_CR),
            'F_history': self.F_history,
            'CR_history': self.CR_history,
            'archive_size': len(self.archive)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SHADE(population_size={self.population_size}, "
            f"H={self.H}, p={self.p}, "
            f"mean_F={np.mean(self.M_F):.3f}, mean_CR={np.mean(self.M_CR):.3f})"
        )
