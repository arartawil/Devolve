"""
LSHADE-EpSin: Enhanced L-SHADE with Epsilon-greedy and Sinusoidal Adaptation

This module implements the LSHADE-EpSin algorithm proposed by Awad et al. (2018),
which enhances L-SHADE with epsilon-greedy strategy selection and sinusoidal
parameter modulation for improved exploration.

Reference:
    Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2018).
    Problem definitions and evaluation criteria for the CEC 2017 special session
    and competition on single objective real-parameter numerical optimization.
    Technical Report, Nanyang Technological University, Singapore.

Key Features:
- All L-SHADE features (memory, population reduction)
- Epsilon-greedy strategy selection (ε decreases linearly)
- Sinusoidal modulation of p parameter
- Enhanced exploration-exploitation balance

Author: DEvolve Package
License: MIT
"""

from typing import Optional, List
import numpy as np

from ...core.base import BaseDifferentialEvolution
from ...core.problem import Problem
from ...core.individual import Individual
from ...core.population import Population
from ...operators.crossover import binomial_crossover


class LSHADEEpSin(BaseDifferentialEvolution):
    """
    LSHADE-EpSin: L-SHADE with Epsilon-greedy and Sinusoidal Adaptation
    
    LSHADE-EpSin extends L-SHADE with:
    1. Epsilon-greedy strategy selection for balanced exploration/exploitation
    2. Sinusoidal modulation of p parameter for dynamic adaptation
    
    Mathematical Formulation:
    -------------------------
    1. Epsilon-greedy Strategy Selection:
       ε(t) = ε_0 * (1 - t/max_t)
       
       With probability ε(t):
           Use exploratory strategy (larger p, more random)
       With probability 1-ε(t):
           Use exploitative strategy (smaller p, best-focused)
    
    2. Sinusoidal p Modulation:
       p(t) = p_min + (p_max - p_min) * (1 + sin(2π * t/T)) / 2
       
       where T is the period (e.g., max_iterations/4)
    
    3. Population Size Reduction (from L-SHADE):
       NP(t) = round(NP_min + (NP_init - NP_min) * (max_FEs - FEs) / max_FEs)
    
    4. Memory Update (from SHADE):
       Weighted Lehmer mean for F, weighted arithmetic mean for CR
    
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
    p_min : float, default=0.05
        Minimum p value for pbest selection
    p_max : float, default=0.20
        Maximum p value for pbest selection
    epsilon_0 : float, default=0.25
        Initial epsilon value for epsilon-greedy
    NP_min : int, default=4
        Minimum population size
    archive_rate : float, default=2.6
        Archive size multiplier
    seed : int, optional
        Random seed for reproducibility
    
    Attributes:
    -----------
    NP_init : int
        Initial population size
    current_NP : int
        Current population size
    M_F : np.ndarray
        Historical memory for F
    M_CR : np.ndarray
        Historical memory for CR
    epsilon : float
        Current epsilon value
    p_current : float
        Current p value (modulated sinusoidally)
    
    Example:
    --------
    >>> from devolve import LSHADEEpSin
    >>> from devolve.benchmarks import Ackley
    >>> 
    >>> # Create problem
    >>> problem = Ackley(dimensions=30)
    >>> 
    >>> # Create optimizer
    >>> optimizer = LSHADEEpSin(
    ...     problem=problem,
    ...     max_iterations=1000,
    ...     epsilon_0=0.25
    ... )
    >>> 
    >>> # Run optimization
    >>> best_solution, best_fitness = optimizer.optimize()
    >>> print(f"Best fitness: {best_fitness:.6e}")
    
    Notes:
    ------
    - Epsilon-greedy helps avoid premature convergence
    - Sinusoidal modulation provides periodic exploration
    - Winner of CEC2017 competition
    - Works well on complex multimodal problems
    - p oscillates between exploration and exploitation
    
    References:
    -----------
    [1] Awad, N. H., et al. (2018). CEC 2017 special session and competition
        on single objective real-parameter numerical optimization.
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: Optional[int] = None,
        max_iterations: int = 1000,
        H: int = 5,
        p_min: float = 0.05,
        p_max: float = 0.20,
        epsilon_0: float = 0.25,
        NP_min: int = 4,
        archive_rate: float = 2.6,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize LSHADE-EpSin optimizer."""
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
        
        # LSHADE-EpSin specific parameters
        self.NP_init = population_size
        self.NP_min = NP_min
        self.current_NP = population_size
        self.H = H
        self.p_min = p_min
        self.p_max = p_max
        self.epsilon_0 = epsilon_0
        self.archive_rate = archive_rate
        
        # Sinusoidal period (1/4 of total iterations)
        self.T = max_iterations / 4.0
        
        # Current adaptive parameters
        self.epsilon = epsilon_0
        self.p_current = (p_min + p_max) / 2.0
        
        # Max function evaluations
        self.max_FEs = max_iterations * population_size
        
        # Initialize memory
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.k = 0
        
        # Archive
        self.archive: List[np.ndarray] = []
        self.max_archive_size = int(population_size * archive_rate)
        
        # Success storage
        self._S_F: List[float] = []
        self._S_CR: List[float] = []
        self._delta_f: List[float] = []
        
        # Current parameters
        self._current_F: np.ndarray = np.zeros(population_size)
        self._current_CR: np.ndarray = np.zeros(population_size)
        self._current_memory_idx: np.ndarray = np.zeros(population_size, dtype=int)
        
        # History
        self.F_history: List[float] = []
        self.CR_history: List[float] = []
        self.NP_history: List[int] = []
        self.epsilon_history: List[float] = []
        self.p_history: List[float] = []
        
        # Current iteration (for modulation)
        self._current_iter = 0
    
    def initialize_population(self) -> None:
        """Initialize population and archive."""
        super().initialize_population()
        
        self.archive = []
        self.current_NP = self.NP_init
        self._current_iter = 0
        
        if self.logger.verbose > 0:
            print(f"LSHADE-EpSin initialized with NP_init={self.NP_init}, "
                  f"ε₀={self.epsilon_0}, p∈[{self.p_min}, {self.p_max}]")
    
    def _update_epsilon(self, iteration: int) -> None:
        """
        Update epsilon value linearly.
        
        ε(t) = ε₀ * (1 - t/max_t)
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        """
        progress = iteration / self.max_iterations
        self.epsilon = self.epsilon_0 * (1.0 - progress)
    
    def _update_p_sinusoidal(self, iteration: int) -> None:
        """
        Update p value using sinusoidal modulation.
        
        p(t) = p_min + (p_max - p_min) * (1 + sin(2π * t/T)) / 2
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        """
        phase = 2.0 * np.pi * iteration / self.T
        modulation = (1.0 + np.sin(phase)) / 2.0
        self.p_current = self.p_min + (self.p_max - self.p_min) * modulation
    
    def _use_exploratory_strategy(self) -> bool:
        """
        Decide whether to use exploratory strategy based on epsilon-greedy.
        
        Returns:
        --------
        bool
            True if exploratory strategy should be used
        """
        return self.rng.random() < self.epsilon
    
    def _calculate_target_NP(self) -> int:
        """Calculate target population size."""
        FEs = self.function_evaluations
        ratio = (self.max_FEs - FEs) / self.max_FEs
        target_NP = round(self.NP_min + (self.NP_init - self.NP_min) * ratio)
        return max(self.NP_min, target_NP)
    
    def _reduce_population(self, target_NP: int) -> None:
        """Reduce population by removing worst individuals."""
        if target_NP >= self.current_NP:
            return
        
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        sorted_indices = np.argsort(fitness_values)
        
        new_individuals = [self.population.individuals[i] for i in sorted_indices[:target_NP]]
        self.population.individuals = new_individuals
        self.current_NP = target_NP
        
        self._current_F = np.zeros(target_NP)
        self._current_CR = np.zeros(target_NP)
        self._current_memory_idx = np.zeros(target_NP, dtype=int)
    
    def _generate_F(self, memory_idx: int, exploratory: bool = False) -> float:
        """
        Generate F parameter.
        
        Parameters:
        -----------
        memory_idx : int
            Memory cell index
        exploratory : bool
            Whether to use exploratory (wider) distribution
        
        Returns:
        --------
        float
            F value
        """
        scale = 0.15 if exploratory else 0.1
        
        while True:
            F = self.rng.standard_cauchy() * scale + self.M_F[memory_idx]
            if F > 0:
                break
        
        return min(F, 1.0)
    
    def _generate_CR(self, memory_idx: int, exploratory: bool = False) -> float:
        """
        Generate CR parameter.
        
        Parameters:
        -----------
        memory_idx : int
            Memory cell index
        exploratory : bool
            Whether to use exploratory (wider) distribution
        
        Returns:
        --------
        float
            CR value
        """
        scale = 0.15 if exploratory else 0.1
        CR = self.rng.normal(self.M_CR[memory_idx], scale)
        return np.clip(CR, 0.0, 1.0)
    
    def _select_pbest(self, exploratory: bool = False) -> Individual:
        """
        Select pbest individual.
        
        Parameters:
        -----------
        exploratory : bool
            If True, use larger p (more exploration)
        
        Returns:
        --------
        Individual
            Selected pbest individual
        """
        # Use current p (modulated) or larger p for exploration
        p = self.p_current * 1.5 if exploratory else self.p_current
        p = min(p, 0.5)  # Cap at 0.5
        
        p_best_size = max(2, int(p * self.current_NP))
        
        fitness_values = np.array([ind.fitness for ind in self.population.individuals])
        sorted_indices = np.argsort(fitness_values)
        
        pbest_idx = self.rng.choice(sorted_indices[:p_best_size])
        return self.population.individuals[pbest_idx]
    
    def mutate(self, target_idx: int) -> np.ndarray:
        """Perform DE/current-to-pbest/1 mutation with epsilon-greedy."""
        self._current_target_idx = target_idx
        
        # Decide strategy
        exploratory = self._use_exploratory_strategy()
        
        # Select memory
        r_i = self.rng.integers(self.H)
        self._current_memory_idx[target_idx] = r_i
        
        # Generate parameters
        F_i = self._generate_F(r_i, exploratory)
        CR_i = self._generate_CR(r_i, exploratory)
        
        self._current_F[target_idx] = F_i
        self._current_CR[target_idx] = CR_i
        
        # Get target
        x_i = self.population.individuals[target_idx].position
        
        # Select pbest
        x_pbest = self._select_pbest(exploratory).position
        
        # Select r1
        candidates = list(range(self.current_NP))
        candidates.remove(target_idx)
        r1 = self.rng.choice(candidates)
        x_r1 = self.population.individuals[r1].position
        
        # Select r2
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
        """Select and track successful parameters."""
        idx = getattr(self, '_current_target_idx', 0)
        
        if trial.is_better_than(target):
            self._S_F.append(self._current_F[idx])
            self._S_CR.append(self._current_CR[idx])
            
            delta_f = abs(target.fitness - trial.fitness)
            self._delta_f.append(delta_f)
            
            self._add_to_archive(target.position.copy())
            
            selected = trial
        else:
            selected = target
        
        return selected
    
    def _add_to_archive(self, position: np.ndarray) -> None:
        """Add to archive."""
        if len(self.archive) < self.max_archive_size:
            self.archive.append(position)
        else:
            idx = self.rng.integers(len(self.archive))
            self.archive[idx] = position
    
    def _update_memory(self) -> None:
        """Update memory using weighted means."""
        if len(self._S_F) == 0:
            return
        
        S_F = np.array(self._S_F)
        S_CR = np.array(self._S_CR)
        weights = np.array(self._delta_f)
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        mean_L_F = np.sum(weights * S_F**2) / np.sum(weights * S_F)
        mean_A_CR = np.sum(weights * S_CR)
        
        self.M_F[self.k] = mean_L_F
        self.M_CR[self.k] = mean_A_CR
        
        self.k = (self.k + 1) % self.H
        
        self._S_F = []
        self._S_CR = []
        self._delta_f = []
    
    def _log_iteration(self, iteration: int) -> None:
        """Override to update adaptive parameters."""
        self._current_iter = iteration
        
        # Update epsilon and p
        self._update_epsilon(iteration)
        self._update_p_sinusoidal(iteration)
        
        # Update memory
        self._update_memory()
        
        # Reduce population
        target_NP = self._calculate_target_NP()
        if target_NP < self.current_NP:
            self._reduce_population(target_NP)
        
        # Track history
        self.F_history.append(np.mean(self.M_F))
        self.CR_history.append(np.mean(self.M_CR))
        self.NP_history.append(self.current_NP)
        self.epsilon_history.append(self.epsilon)
        self.p_history.append(self.p_current)
        
        # Base logging
        super()._log_iteration(iteration)
        
        # Custom logging
        if iteration % 100 == 0 and self.logger.verbose > 1:
            print(f"Iteration {iteration}: NP={self.current_NP}, "
                  f"ε={self.epsilon:.3f}, p={self.p_current:.3f}, "
                  f"M_F={np.mean(self.M_F):.3f}")
    
    def get_parameter_statistics(self) -> dict:
        """Get parameter statistics."""
        return {
            'M_F': self.M_F.copy(),
            'M_CR': self.M_CR.copy(),
            'mean_F': np.mean(self.M_F),
            'mean_CR': np.mean(self.M_CR),
            'F_history': self.F_history,
            'CR_history': self.CR_history,
            'NP_history': self.NP_history,
            'epsilon_history': self.epsilon_history,
            'p_history': self.p_history,
            'current_NP': self.current_NP,
            'current_epsilon': self.epsilon,
            'current_p': self.p_current,
            'archive_size': len(self.archive)
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LSHADEEpSin(NP={self.current_NP}, ε={self.epsilon:.3f}, "
            f"p={self.p_current:.3f}, mean_F={np.mean(self.M_F):.3f})"
        )
