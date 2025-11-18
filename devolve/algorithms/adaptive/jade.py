"""
JADE: Adaptive Differential Evolution with Optional External Archive

This module implements the JADE algorithm proposed by Zhang & Sanderson (2009),
which uses historical success information to adapt parameters and maintains
an external archive of recently replaced solutions.

Reference:
    Zhang, J., & Sanderson, A. C. (2009).
    JADE: Adaptive differential evolution with optional external archive.
    IEEE Transactions on Evolutionary Computation, 13(5), 945-958.

Key Features:
- Parameter adaptation using Cauchy and Normal distributions
- μ_F and μ_CR updated based on successful parameters
- External archive of size NP storing recently replaced solutions
- DE/current-to-pbest/1 mutation strategy
- Optional greedy archive usage

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


class JADE(BaseDifferentialEvolution):
    """
    JADE: Adaptive Differential Evolution with Optional External Archive
    
    JADE adapts F and CR parameters using historical information about successful
    parameter values. It uses Cauchy distribution for F and Normal distribution
    for CR, and maintains an external archive of replaced solutions.
    
    Mathematical Formulation:
    -------------------------
    For each individual i at generation t:
    
    1. Parameter Generation:
       F_i ~ Cauchy(μ_F, 0.1), truncated to [0, 1]
       CR_i ~ Normal(μ_CR, 0.1), truncated to [0, 1]
    
    2. Mutation (DE/current-to-pbest/1):
       v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
       
       where x_pbest is randomly chosen from the top p% (p=0.05-0.20)
       and x_r2 may come from population ∪ archive
    
    3. Crossover (binomial):
       u_ij = v_ij if rand() < CR_i or j = j_rand else x_ij
    
    4. Parameter Adaptation (after each generation):
       μ_F = (1-c) * μ_F + c * mean_L(S_F)    (Lehmer mean)
       μ_CR = (1-c) * μ_CR + c * mean_A(S_CR)  (arithmetic mean)
       
       where S_F and S_CR are sets of successful parameters
       c is the learning rate (typically 0.1)
    
    Parameters:
    -----------
    problem : Problem
        The optimization problem to solve
    population_size : int, default=50
        Number of individuals in the population
    max_iterations : int, default=1000
        Maximum number of iterations
    c : float, default=0.1
        Learning rate for parameter adaptation
    p : float, default=0.05
        Proportion of top individuals for pbest selection (0.05-0.20)
    initial_mu_F : float, default=0.5
        Initial mean value for F parameter
    initial_mu_CR : float, default=0.5
        Initial mean value for CR parameter
    archive_size : int, optional
        Size of external archive (default: population_size)
    seed : int, optional
        Random seed for reproducibility
    
    Attributes:
    -----------
    mu_F : float
        Current mean value for F generation
    mu_CR : float
        Current mean value for CR generation
    archive : List[np.ndarray]
        External archive of replaced solutions
    F_history : List[float]
        History of μ_F values over iterations
    CR_history : List[float]
        History of μ_CR values over iterations
    
    Example:
    --------
    >>> from devolve import JADE
    >>> from devolve.benchmarks import Rosenbrock
    >>> 
    >>> # Create problem
    >>> problem = Rosenbrock(dimensions=30)
    >>> 
    >>> # Create optimizer
    >>> optimizer = JADE(
    ...     problem=problem,
    ...     population_size=100,
    ...     max_iterations=1000,
    ...     c=0.1,
    ...     p=0.05
    ... )
    >>> 
    >>> # Run optimization
    >>> best_solution, best_fitness = optimizer.optimize()
    >>> print(f"Best fitness: {best_fitness:.6e}")
    
    Notes:
    ------
    - Cauchy distribution has heavier tails than Normal, allowing more exploration
    - The pbest selection (top p%) balances exploration and exploitation
    - Archive prevents premature convergence by maintaining diversity
    - μ_F typically converges to higher values (0.6-0.9)
    - μ_CR typically converges to high values (0.8-0.99)
    
    References:
    -----------
    [1] Zhang, J., & Sanderson, A. C. (2009).
        JADE: Adaptive differential evolution with optional external archive.
        IEEE Transactions on Evolutionary Computation, 13(5), 945-958.
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: int = 50,
        max_iterations: int = 1000,
        c: float = 0.1,
        p: float = 0.05,
        initial_mu_F: float = 0.5,
        initial_mu_CR: float = 0.5,
        archive_size: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize JADE optimizer."""
        super().__init__(
            problem=problem,
            population_size=population_size,
            max_iterations=max_iterations,
            F=initial_mu_F,  # Initial mean F
            CR=initial_mu_CR,  # Initial mean CR
            random_seed=seed,
            **kwargs
        )
        
        # JADE specific parameters
        self.c = c  # Learning rate
        self.p = p  # Proportion for pbest selection
        self.initial_mu_F = initial_mu_F
        self.initial_mu_CR = initial_mu_CR
        self.archive_size = archive_size if archive_size is not None else population_size
        
        # Parameter means
        self.mu_F = initial_mu_F
        self.mu_CR = initial_mu_CR
        
        # External archive
        self.archive: List[np.ndarray] = []
        
        # History tracking
        self.F_history: List[float] = []
        self.CR_history: List[float] = []
        
        # Storage for successful parameters (per generation)
        self._S_F: List[float] = []
        self._S_CR: List[float] = []
        self._delta_f: List[float] = []  # Fitness improvements
        
        # Storage for current generation parameters
        self._current_F: np.ndarray = np.zeros(population_size)
        self._current_CR: np.ndarray = np.zeros(population_size)
    
    def initialize_population(self) -> None:
        """Initialize population and archive."""
        super().initialize_population()
        
        # Initialize archive as empty
        self.archive = []
        
        if self.logger.verbose > 0:
            print(f"JADE initialized with c={self.c}, p={self.p}, "
                  f"μ_F={self.mu_F:.3f}, μ_CR={self.mu_CR:.3f}")
    
    def _generate_F(self) -> float:
        """
        Generate F parameter using Cauchy distribution.
        
        Returns:
        --------
        float
            F value sampled from Cauchy(μ_F, 0.1), truncated to [0, 1]
        
        Notes:
        ------
        Cauchy distribution has heavier tails than Normal, allowing
        occasional large mutations for better exploration.
        """
        while True:
            F = self.rng.standard_cauchy() * 0.1 + self.mu_F
            if F > 0:  # Reject negative values
                break
        
        # Truncate to [0, 1]
        if F > 1.0:
            F = 1.0
        
        return F
    
    def _generate_CR(self) -> float:
        """
        Generate CR parameter using Normal distribution.
        
        Returns:
        --------
        float
            CR value sampled from Normal(μ_CR, 0.1), truncated to [0, 1]
        """
        CR = self.rng.normal(self.mu_CR, 0.1)
        
        # Truncate to [0, 1]
        return np.clip(CR, 0.0, 1.0)
    
    def _select_pbest(self) -> Individual:
        """
        Select a random individual from the top p% of the population.
        
        Returns:
        --------
        Individual
            Randomly selected individual from top p%
        """
        # Calculate number of top individuals
        p_best_size = max(2, int(self.p * self.population_size))
        
        # Get indices sorted by fitness (best first)
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        sorted_indices = np.argsort(fitness_values)
        
        # Select random index from top p%
        pbest_idx = self.rng.choice(sorted_indices[:p_best_size])
        
        return self.population.individuals[pbest_idx]
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """
        Perform DE/current-to-pbest/1 mutation with archive.
        
        Uses the mutation strategy:
        v_i = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
        
        where x_r2 can be from population ∪ archive.
        
        Parameters:
        -----------
        target_idx : int
            Index of the target individual
        
        Returns:
        --------
        np.ndarray
            Mutant vector
        """
        # Store current index for crossover
        self._current_target_idx = target_idx
        
        # Generate F and CR for this individual
        F_i = self._generate_F()
        CR_i = self._generate_CR()
        
        # Store parameters
        self._current_F[target_idx] = F_i
        self._current_CR[target_idx] = CR_i
        
        # Get target individual
        x_i = self.population.individuals[target_idx].position
        
        # Select pbest from top p%
        x_pbest = self._select_pbest().position
        
        # Select random individual r1 (different from target)
        candidates = list(range(self.population_size))
        candidates.remove(target_idx)
        r1 = self.rng.choice(candidates)
        x_r1 = self.population.individuals[r1].position
        
        # Select r2 from population ∪ archive
        if len(self.archive) > 0 and self.rng.random() < 0.5:
            # Select from archive
            x_r2 = self.archive[self.rng.integers(len(self.archive))]
        else:
            # Select from population (different from target and r1)
            candidates.remove(r1)
            r2 = self.rng.choice(candidates)
            x_r2 = self.population.individuals[r2].position
        
        # DE/current-to-pbest/1 mutation
        mutant = x_i + F_i * (x_pbest - x_i) + F_i * (x_r1 - x_r2)
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Perform binomial crossover.
        
        Parameters:
        -----------
        target : np.ndarray
            Target vector
        mutant : np.ndarray
            Mutant vector
        
        Returns:
        --------
        np.ndarray
            Trial vector
        """
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
        """
        Select between target and trial, tracking successful parameters.
        
        If trial is better:
        - Store F and CR in success sets
        - Add target to archive
        - Record fitness improvement
        """
        idx = getattr(self, '_current_target_idx', 0)
        
        if trial.is_better_than(target):
            # Trial wins - record successful parameters
            F_i = self._current_F[idx]
            CR_i = self._current_CR[idx]
            
            # Store successful parameters
            self._S_F.append(F_i)
            self._S_CR.append(CR_i)
            
            # Store fitness improvement (for weighted mean)
            delta_f = abs(target.fitness - trial.fitness)
            self._delta_f.append(delta_f)
            
            # Add replaced solution to archive
            self._add_to_archive(target.position.copy())
            
            selected = trial
        else:
            selected = target
        
        return selected
    
    def _add_to_archive(self, position: np.ndarray) -> None:
        """
        Add a solution to the external archive.
        
        If archive is full, randomly replace an existing solution.
        
        Parameters:
        -----------
        position : np.ndarray
            Position vector to add
        """
        if len(self.archive) < self.archive_size:
            self.archive.append(position)
        else:
            # Replace random archive member
            idx = self.rng.integers(len(self.archive))
            self.archive[idx] = position
    
    def _update_parameters(self) -> None:
        """
        Update μ_F and μ_CR based on successful parameters.
        
        Uses weighted Lehmer mean for F and weighted arithmetic mean for CR.
        Weights are based on fitness improvements.
        """
        if len(self._S_F) == 0:
            # No successful parameters this generation
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
        if len(S_F) > 0:
            mean_L_F = np.sum(weights * S_F**2) / np.sum(weights * S_F)
            self.mu_F = (1 - self.c) * self.mu_F + self.c * mean_L_F
        
        # Weighted arithmetic mean for CR
        if len(S_CR) > 0:
            mean_A_CR = np.sum(weights * S_CR)
            self.mu_CR = (1 - self.c) * self.mu_CR + self.c * mean_A_CR
        
        # Clear success sets for next generation
        self._S_F = []
        self._S_CR = []
        self._delta_f = []
    
    def _log_iteration(self, iteration: int) -> None:
        """Override to add parameter statistics and update parameters."""
        # Update parameters first (at end of generation)
        self._update_parameters()
        
        # Track parameter history
        self.F_history.append(self.mu_F)
        self.CR_history.append(self.mu_CR)
        
        # Call base logging
        super()._log_iteration(iteration)
        
        # Log parameter info (if verbose)
        if iteration % 100 == 0 and self.logger.verbose > 1:
            print(f"Iteration {iteration}: μ_F={self.mu_F:.3f}, μ_CR={self.mu_CR:.3f}, "
                  f"Archive size={len(self.archive)}")
    
    def get_parameter_statistics(self) -> dict:
        """
        Get statistics about parameter evolution.
        
        Returns:
        --------
        dict
            Dictionary containing parameter statistics:
            - 'mu_F': Current mean F value
            - 'mu_CR': Current mean CR value
            - 'F_history': List of μ_F values over iterations
            - 'CR_history': List of μ_CR values over iterations
            - 'archive_size': Current size of external archive
            - 'archive_capacity': Maximum archive size
        
        Example:
        --------
        >>> optimizer = JADE(problem, population_size=50)
        >>> optimizer.optimize()
        >>> stats = optimizer.get_parameter_statistics()
        >>> print(f"Final μ_F: {stats['mu_F']:.3f}")
        """
        return {
            'mu_F': self.mu_F,
            'mu_CR': self.mu_CR,
            'F_history': self.F_history,
            'CR_history': self.CR_history,
            'archive_size': len(self.archive),
            'archive_capacity': self.archive_size
        }
    
    def __repr__(self) -> str:
        """String representation of JADE optimizer."""
        return (
            f"JADE(population_size={self.population_size}, "
            f"c={self.c}, p={self.p}, "
            f"μ_F={self.mu_F:.3f}, μ_CR={self.mu_CR:.3f})"
        )
