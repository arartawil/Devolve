"""
Abstract base class for all Differential Evolution algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, List, Tuple, Any
import numpy as np
from .individual import Individual
from .population import Population
from .problem import Problem
from .boundary import BoundaryHandler
from .logger import OptimizationLogger


class BaseDifferentialEvolution(ABC):
    """
    Abstract base class for all Differential Evolution variants.
    
    This class provides the common framework and infrastructure that all DE
    variants inherit from. Subclasses must implement the abstract methods
    for mutation and crossover strategies.
    
    Parameters
    ----------
    problem : Problem
        The optimization problem to solve.
    population_size : int, optional
        Number of individuals in the population. Default is 50.
    max_iterations : int, optional
        Maximum number of generations. Default is 1000.
    F : float, optional
        Scaling factor for mutation (typically 0.4-1.0). Default is 0.8.
    CR : float, optional
        Crossover probability (0.0-1.0). Default is 0.9.
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
        Logger instance for tracking progress. Default is None (creates new).
    
    Attributes
    ----------
    problem : Problem
        The optimization problem.
    population : Population
        Current population of individuals.
    best_individual : Individual
        Best individual found so far.
    best_fitness_history : List[float]
        History of best fitness values per iteration.
    convergence_data : dict
        Additional convergence metrics.
    rng : np.random.Generator
        Random number generator.
    boundary_handler : BoundaryHandler
        Handler for boundary violations.
    logger : OptimizationLogger
        Logger for tracking progress.
    
    Examples
    --------
    Subclasses should implement mutation and crossover:
    
    >>> class MyDE(BaseDifferentialEvolution):
    ...     def mutate(self, target_idx: int) -> np.ndarray:
    ...         # Implement mutation strategy
    ...         pass
    ...     
    ...     def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
    ...         # Implement crossover strategy
    ...         pass
    
    Notes
    -----
    The basic DE algorithm follows this structure:
    
    1. Initialize population randomly within bounds
    2. Evaluate fitness of all individuals
    3. For each generation:
       a. For each target individual:
          i. Create mutant vector (mutation)
          ii. Create trial vector (crossover)
          iii. Evaluate trial vector
          iv. Select better between target and trial
       b. Update best solution
       c. Check stopping criteria
    4. Return best solution found
    
    References
    ----------
    Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient
    heuristic for global optimization over continuous spaces. Journal of global
    optimization, 11(4), 341-359. DOI: 10.1023/A:1008202821328
    """
    
    def __init__(
        self,
        problem: Problem,
        population_size: int = 50,
        max_iterations: int = 1000,
        F: float = 0.8,
        CR: float = 0.9,
        boundary_strategy: str = "clip",
        early_stopping: bool = False,
        early_stopping_tolerance: float = 1e-6,
        early_stopping_generations: int = 50,
        random_seed: Optional[int] = None,
        callback: Optional[Callable] = None,
        logger: Optional[OptimizationLogger] = None
    ):
        """Initialize the Differential Evolution algorithm."""
        self.problem = problem
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.F = F
        self.CR = CR
        self.early_stopping = early_stopping
        self.early_stopping_tolerance = early_stopping_tolerance
        self.early_stopping_generations = early_stopping_generations
        self.callback = callback
        
        # Initialize random number generator
        self.rng = np.random.default_rng(random_seed)
        
        # Initialize boundary handler
        self.boundary_handler = BoundaryHandler(
            strategy=boundary_strategy,
            bounds=problem.bounds,
            rng=self.rng
        )
        
        # Initialize logger
        self.logger = logger if logger is not None else OptimizationLogger()
        
        # Initialize population and tracking variables
        self.population: Optional[Population] = None
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
        self.convergence_data = {
            'mean_fitness': [],
            'diversity': [],
            'feasibility_ratio': [],
            'stagnation_counter': 0
        }
        
        # Function evaluation counter
        self.function_evaluations = 0
    
    def initialize_population(self) -> None:
        """
        Initialize population with random individuals within bounds.
        
        Creates population_size individuals with positions uniformly
        distributed within the problem bounds.
        """
        self.population = Population()
        
        for _ in range(self.population_size):
            position = self.problem.random_solution(self.rng)
            individual = Individual(position=position)
            self.population.add(individual)
        
        # Evaluate initial population
        self.evaluate_population()
    
    def evaluate_population(self) -> None:
        """
        Evaluate fitness for all individuals in the population.
        
        This method evaluates the objective function and constraints
        for each individual that hasn't been evaluated yet.
        """
        for individual in self.population:
            if not individual.is_evaluated:
                self.evaluate_individual(individual)
    
    def evaluate_individual(self, individual: Individual) -> None:
        """
        Evaluate a single individual.
        
        Parameters
        ----------
        individual : Individual
            The individual to evaluate.
        """
        # Evaluate objective function
        individual.fitness = self.problem.evaluate(individual.position)
        self.function_evaluations += 1
        
        # Evaluate constraints
        _, violation = self.problem.evaluate_constraints(individual.position)
        individual.constraint_violation = violation
    
    @abstractmethod
    def mutate(self, target_idx: int) -> np.ndarray:
        """
        Create a mutant vector for the target individual.
        
        This method must be implemented by subclasses to define
        the specific mutation strategy.
        
        Parameters
        ----------
        target_idx : int
            Index of the target individual in the population.
        
        Returns
        -------
        np.ndarray
            Mutant vector.
        
        Notes
        -----
        Common mutation strategies include:
        - DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
        - DE/best/1: v = x_best + F * (x_r1 - x_r2)
        - DE/current-to-best/1: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
        """
        pass
    
    @abstractmethod
    def crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """
        Create a trial vector through crossover.
        
        This method must be implemented by subclasses to define
        the crossover strategy.
        
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
        
        Notes
        -----
        Common crossover strategies:
        - Binomial: Each dimension with probability CR
        - Exponential: Consecutive L dimensions starting from random position
        """
        pass
    
    def select(self, target: Individual, trial: Individual) -> Individual:
        """
        Select the better individual between target and trial.
        
        Uses Deb's feasibility rules for constrained problems.
        
        Parameters
        ----------
        target : Individual
            Current target individual.
        trial : Individual
            New trial individual.
        
        Returns
        -------
        Individual
            The selected (better) individual.
        """
        if trial.is_better_than(target):
            trial.age = 0  # Reset age for new individual
            return trial
        else:
            target.age += 1
            return target
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute the Differential Evolution optimization.
        
        Returns
        -------
        best_position : np.ndarray
            Position of the best solution found.
        best_fitness : float
            Fitness of the best solution found.
        
        Notes
        -----
        This is the main optimization loop that:
        1. Initializes the population
        2. Iterates through generations
        3. Applies mutation, crossover, and selection
        4. Tracks convergence and best solution
        5. Checks stopping criteria
        """
        # Initialize
        self.logger.start()
        self.initialize_population()
        self.best_individual = self.population.get_best().copy()
        self.best_fitness_history.append(self.best_individual.fitness)
        
        # Log initial state
        self._log_iteration(0)
        
        # Main optimization loop
        for iteration in range(1, self.max_iterations + 1):
            # Create new population
            new_population = Population()
            
            # Process each individual
            for target_idx in range(self.population.size):
                # Get target individual
                target = self.population[target_idx]
                
                # Mutation
                mutant_vector = self.mutate(target_idx)
                
                # Boundary handling
                mutant_vector = self.boundary_handler.handle(mutant_vector)
                
                # Crossover
                trial_vector = self.crossover(target.position, mutant_vector)
                
                # Boundary handling for trial
                trial_vector = self.boundary_handler.handle(trial_vector)
                
                # Create trial individual
                trial = Individual(position=trial_vector)
                
                # Evaluate trial
                self.evaluate_individual(trial)
                
                # Selection
                selected = self.select(target, trial)
                new_population.add(selected.copy())
            
            # Replace population
            self.population = new_population
            
            # Update best solution
            current_best = self.population.get_best()
            if current_best.is_better_than(self.best_individual):
                self.best_individual = current_best.copy()
                self.convergence_data['stagnation_counter'] = 0
            else:
                self.convergence_data['stagnation_counter'] += 1
            
            # Track convergence
            self.best_fitness_history.append(self.best_individual.fitness)
            self._update_convergence_data()
            
            # Log iteration
            self._log_iteration(iteration)
            
            # Callback
            if self.callback is not None:
                self.callback(self, iteration)
            
            # Check early stopping
            if self.early_stopping and self._check_early_stopping():
                if self.logger.verbose > 0:
                    print(f"\nEarly stopping at iteration {iteration}")
                break
        
        self.logger.end()
        
        return self.best_individual.position, self.best_individual.fitness
    
    def _update_convergence_data(self) -> None:
        """Update convergence metrics."""
        fitness_values = self.population.get_fitness_values()
        self.convergence_data['mean_fitness'].append(np.mean(fitness_values))
        self.convergence_data['diversity'].append(self.population.get_diversity())
        self.convergence_data['feasibility_ratio'].append(
            self.population.get_feasibility_ratio()
        )
    
    def _log_iteration(self, iteration: int) -> None:
        """Log iteration information."""
        fitness_values = self.population.get_fitness_values()
        
        self.logger.log_iteration(
            iteration=iteration,
            best_fitness=self.best_individual.fitness,
            mean_fitness=np.mean(fitness_values),
            worst_fitness=np.max(fitness_values),
            std_fitness=np.std(fitness_values),
            diversity=self.population.get_diversity(),
            feasibility_ratio=self.population.get_feasibility_ratio(),
            function_evaluations=self.function_evaluations
        )
    
    def _check_early_stopping(self) -> bool:
        """
        Check if early stopping criteria are met.
        
        Returns
        -------
        bool
            True if should stop early, False otherwise.
        """
        # Check stagnation
        if self.convergence_data['stagnation_counter'] >= self.early_stopping_generations:
            return True
        
        # Check fitness improvement
        if len(self.best_fitness_history) >= self.early_stopping_generations:
            recent = self.best_fitness_history[-self.early_stopping_generations:]
            improvement = abs(max(recent) - min(recent))
            if improvement < self.early_stopping_tolerance:
                return True
        
        return False
    
    @property
    def population_diversity(self) -> float:
        """
        Get current population diversity.
        
        Returns
        -------
        float
            Population diversity metric.
        """
        if self.population is None:
            return 0.0
        return self.population.get_diversity()
    
    def get_best_solution(self) -> Tuple[np.ndarray, float]:
        """
        Get the best solution found.
        
        Returns
        -------
        position : np.ndarray
            Position of the best solution.
        fitness : float
            Fitness of the best solution.
        """
        if self.best_individual is None:
            raise ValueError("No optimization has been run yet")
        return self.best_individual.position, self.best_individual.fitness
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"population_size={self.population_size}, "
                f"F={self.F}, "
                f"CR={self.CR})")
