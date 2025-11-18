"""
jDE: Self-Adaptive Differential Evolution

This module implements the jDE algorithm proposed by Brest et al. (2006),
which adapts F and CR parameters during the optimization process.

Reference:
    Brest, J., Greiner, S., Boskovic, B., Mernik, M., & Zumer, V. (2006).
    Self-adapting control parameters in differential evolution: A comparative
    study on numerical benchmark problems.
    IEEE Transactions on Evolutionary Computation, 10(6), 646-657.

Key Features:
- Self-adaptive F and CR parameters for each individual
- Parameters updated before mutation with probability τ
- F sampled from U[F_l, F_u] = U[0.1, 1.0]
- CR sampled from U[0, 1]
- Uses DE/rand/1/bin mutation strategy

Author: DEvolve Package
License: MIT
"""

from typing import Optional, Tuple, List
import numpy as np

from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...core.individual import Individual
from ...core.population import Population
from ...operators.mutation import rand_1
from ...operators.crossover import binomial_crossover
from ...operators.selection import greedy_selection


class JDE(BaseDifferentialEvolution):
    """
    jDE: Self-Adaptive Differential Evolution
    
    jDE adapts the F and CR parameters for each individual during the
    optimization process. Before creating a trial vector, each individual's
    F and CR values are adapted with probability τ₁ and τ₂ respectively.
    
    Mathematical Formulation:
    -------------------------
    For each individual i at generation t:
    
    1. Parameter Adaptation (with probability τ):
       if rand() < τ₁:
           F_i,t+1 = F_l + rand() * (F_u - F_l)
       else:
           F_i,t+1 = F_i,t
           
       if rand() < τ₂:
           CR_i,t+1 = rand()
       else:
           CR_i,t+1 = CR_i,t
    
    2. Mutation (DE/rand/1):
       v_i = x_r1 + F_i * (x_r2 - x_r3)
    
    3. Crossover (binomial):
       u_ij = v_ij if rand() < CR_i or j = j_rand else x_ij
    
    4. Selection (greedy):
       x_i,t+1 = u_i if f(u_i) < f(x_i) else x_i
    
    Parameters:
    -----------
    problem : Problem
        The optimization problem to solve
    population_size : int, default=50
        Number of individuals in the population
    max_iterations : int, default=1000
        Maximum number of iterations
    tau1 : float, default=0.1
        Probability of adapting F parameter (τ₁)
    tau2 : float, default=0.1
        Probability of adapting CR parameter (τ₂)
    F_lower : float, default=0.1
        Lower bound for F parameter (F_l)
    F_upper : float, default=1.0
        Upper bound for F parameter (F_u)
    initial_F : float, default=0.5
        Initial F value for all individuals
    initial_CR : float, default=0.9
        Initial CR value for all individuals
    seed : int, optional
        Random seed for reproducibility
    
    Attributes:
    -----------
    F_values : np.ndarray
        Current F value for each individual
    CR_values : np.ndarray
        Current CR value for each individual
    F_history : List[float]
        History of mean F values over iterations
    CR_history : List[float]
        History of mean CR values over iterations
    
    Example:
    --------
    >>> from devolve import JDE
    >>> from devolve.benchmarks import Sphere
    >>> 
    >>> # Create problem
    >>> problem = Sphere(dimensions=30)
    >>> 
    >>> # Create optimizer
    >>> optimizer = JDE(
    ...     problem=problem,
    ...     population_size=50,
    ...     max_iterations=1000,
    ...     tau1=0.1,
    ...     tau2=0.1
    ... )
    >>> 
    >>> # Run optimization
    >>> best_solution, best_fitness = optimizer.optimize()
    >>> print(f"Best fitness: {best_fitness:.6e}")
    
    Notes:
    ------
    - Default parameters (τ₁=0.1, τ₂=0.1) work well for most problems
    - F is sampled from [0.1, 1.0] to ensure sufficient diversity
    - CR is sampled from [0, 1] with no restrictions
    - Successful parameter values propagate to offspring
    - More explorative than fixed-parameter DE
    
    References:
    -----------
    [1] Brest, J., Greiner, S., Boskovic, B., Mernik, M., & Zumer, V. (2006).
        Self-adapting control parameters in differential evolution: A comparative
        study on numerical benchmark problems.
        IEEE Transactions on Evolutionary Computation, 10(6), 646-657.
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: int = 50,
        max_iterations: int = 1000,
        tau1: float = 0.1,
        tau2: float = 0.1,
        F_lower: float = 0.1,
        F_upper: float = 1.0,
        initial_F: float = 0.5,
        initial_CR: float = 0.9,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize jDE optimizer."""
        super().__init__(
            problem=problem,
            population_size=population_size,
            max_iterations=max_iterations,
            F=initial_F,  # Initial F (will be adapted)
            CR=initial_CR,  # Initial CR (will be adapted)
            random_seed=seed,
            **kwargs
        )
        
        # jDE specific parameters
        self.tau1 = tau1
        self.tau2 = tau2
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.initial_F = initial_F
        self.initial_CR = initial_CR
        
        # Initialize parameter arrays (one F and CR per individual)
        self.F_values: np.ndarray = np.full(population_size, initial_F)
        self.CR_values: np.ndarray = np.full(population_size, initial_CR)
        
        # History tracking
        self.F_history: List[float] = []
        self.CR_history: List[float] = []
    
    def initialize_population(self) -> None:
        """
        Initialize population with random solutions and parameter values.
        
        Extends base initialization to set up individual F and CR values.
        """
        super().initialize_population()
        
        # Initialize F and CR values for all individuals
        self.F_values = np.full(self.population_size, self.initial_F)
        self.CR_values = np.full(self.population_size, self.initial_CR)
        
        # Log initial parameters (if verbose)
        if self.logger.verbose > 0:
            print(f"jDE initialized with τ₁={self.tau1}, τ₂={self.tau2}, "
                  f"F∈[{self.F_lower}, {self.F_upper}]")
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """
        Perform mutation using DE/rand/1 strategy with adapted F.
        
        Adapts F parameter before mutation with probability τ₁.
        Uses individual-specific F value.
        
        Parameters:
        -----------
        target_idx : int
            Index of the target individual
        
        Returns:
        --------
        np.ndarray
            Mutant vector
        
        Formula:
        --------
        v_i = x_r1 + F_i * (x_r2 - x_r3)
        
        where r1, r2, r3 are randomly selected distinct indices ≠ target_idx
        """
        # Store current index for crossover and selection
        self._current_target_idx = target_idx
        
        # Adapt F with probability tau1 before mutation
        if self.rng.random() < self.tau1:
            self.F_values[target_idx] = (
                self.F_lower + 
                self.rng.random() * (self.F_upper - self.F_lower)
            )
        
        # Get current F for this individual
        F_i = self.F_values[target_idx]
        
        # Perform DE/rand/1 mutation
        mutant = rand_1(
            population=self.population,
            target_idx=target_idx,
            F=F_i,
            rng=self.rng
        )
        
        return mutant
    
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Perform binomial crossover with adapted CR.
        
        This is called by the base class during optimization.
        Adapts CR before crossover with probability τ₂.
        
        Parameters:
        -----------
        target : np.ndarray
            Target vector (parent)
        mutant : np.ndarray
            Mutant vector
        
        Returns:
        --------
        np.ndarray
            Trial vector
        
        Formula:
        --------
        u_ij = v_ij  if rand() < CR_i or j = j_rand
               x_ij  otherwise
        """
        # Get the current target index (set in mutate)
        idx = getattr(self, '_current_target_idx', 0)
        
        # Adapt CR with probability tau2 before crossover
        if self.rng.random() < self.tau2:
            self.CR_values[idx] = self.rng.random()
        
        # Get current CR for this individual
        CR_i = self.CR_values[idx]
        
        # Perform binomial crossover
        trial = binomial_crossover(
            target=target,
            mutant=mutant,
            CR=CR_i,
            rng=self.rng
        )
        
        return trial
    
    def _log_iteration(self, iteration: int) -> None:
        """Override to add parameter statistics."""
        # Call base logging
        super()._log_iteration(iteration)
        
        # Track parameter history
        mean_F = np.mean(self.F_values)
        mean_CR = np.mean(self.CR_values)
        self.F_history.append(mean_F)
        self.CR_history.append(mean_CR)
        
        # Log parameter info (if verbose)
        if iteration % 100 == 0 and self.logger.verbose > 1:
            print(f"Iteration {iteration}: F={mean_F:.3f}±{np.std(self.F_values):.3f}, "
                  f"CR={mean_CR:.3f}±{np.std(self.CR_values):.3f}")
    
    def get_parameter_statistics(self) -> dict:
        """
        Get statistics about F and CR parameter evolution.
        
        Returns:
        --------
        dict
            Dictionary containing parameter statistics:
            - 'F_mean': Mean F value across population
            - 'F_std': Standard deviation of F values
            - 'F_min': Minimum F value
            - 'F_max': Maximum F value
            - 'CR_mean': Mean CR value across population
            - 'CR_std': Standard deviation of CR values
            - 'CR_min': Minimum CR value
            - 'CR_max': Maximum CR value
            - 'F_history': List of mean F values over iterations
            - 'CR_history': List of mean CR values over iterations
        
        Example:
        --------
        >>> optimizer = JDE(problem, population_size=50)
        >>> optimizer.optimize()
        >>> stats = optimizer.get_parameter_statistics()
        >>> print(f"Final F: {stats['F_mean']:.3f} ± {stats['F_std']:.3f}")
        """
        return {
            'F_mean': np.mean(self.F_values),
            'F_std': np.std(self.F_values),
            'F_min': np.min(self.F_values),
            'F_max': np.max(self.F_values),
            'CR_mean': np.mean(self.CR_values),
            'CR_std': np.std(self.CR_values),
            'CR_min': np.min(self.CR_values),
            'CR_max': np.max(self.CR_values),
            'F_history': self.F_history,
            'CR_history': self.CR_history
        }
    
    def __repr__(self) -> str:
        """String representation of jDE optimizer."""
        return (
            f"JDE(population_size={self.population_size}, "
            f"τ₁={self.tau1}, τ₂={self.tau2}, "
            f"F∈[{self.F_lower}, {self.F_upper}])"
        )
