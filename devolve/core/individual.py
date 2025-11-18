"""
Individual class representing a solution in the population.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Individual:
    """
    Represents a single solution (individual) in the population.
    
    An individual consists of a position vector in the search space,
    its corresponding fitness value, and information about constraint violations.
    
    Attributes
    ----------
    position : np.ndarray
        The position vector in the search space (decision variables).
    fitness : float
        The fitness (objective function) value. Default is infinity (uninitialized).
    constraint_violation : float
        The total constraint violation. Default is 0.0 (feasible).
    F : Optional[float]
        Individual's scaling factor (for adaptive DE variants). Default is None.
    CR : Optional[float]
        Individual's crossover rate (for adaptive DE variants). Default is None.
    age : int
        Number of generations this individual has survived. Default is 0.
    
    Examples
    --------
    >>> import numpy as np
    >>> ind = Individual(position=np.array([1.0, 2.0, 3.0]))
    >>> ind.fitness = 10.5
    >>> ind.is_better_than(Individual(position=np.zeros(3), fitness=15.0))
    True
    
    Notes
    -----
    For minimization problems, lower fitness values are better.
    The `constraint_violation` should be 0.0 for feasible solutions.
    """
    
    position: np.ndarray
    fitness: float = np.inf
    constraint_violation: float = 0.0
    F: Optional[float] = None
    CR: Optional[float] = None
    age: int = 0
    
    def __post_init__(self):
        """Validate the individual after initialization."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        
        if self.position.ndim != 1:
            raise ValueError("Position must be a 1D array")
    
    @property
    def dimensions(self) -> int:
        """
        Get the dimensionality of the solution.
        
        Returns
        -------
        int
            Number of dimensions (decision variables).
        """
        return len(self.position)
    
    @property
    def is_feasible(self) -> bool:
        """
        Check if the individual is feasible (satisfies all constraints).
        
        Returns
        -------
        bool
            True if feasible (constraint_violation == 0), False otherwise.
        """
        return self.constraint_violation == 0.0
    
    @property
    def is_evaluated(self) -> bool:
        """
        Check if the individual has been evaluated.
        
        Returns
        -------
        bool
            True if fitness is not infinity, False otherwise.
        """
        return not np.isinf(self.fitness)
    
    def copy(self) -> 'Individual':
        """
        Create a deep copy of the individual.
        
        Returns
        -------
        Individual
            A new Individual instance with copied attributes.
        """
        return Individual(
            position=self.position.copy(),
            fitness=self.fitness,
            constraint_violation=self.constraint_violation,
            F=self.F,
            CR=self.CR,
            age=self.age
        )
    
    def is_better_than(self, other: 'Individual', 
                       use_feasibility_rules: bool = True) -> bool:
        """
        Compare this individual with another using Deb's feasibility rules.
        
        Parameters
        ----------
        other : Individual
            The individual to compare with.
        use_feasibility_rules : bool, optional
            If True, use Deb's feasibility rules. If False, use only fitness.
            Default is True.
        
        Returns
        -------
        bool
            True if this individual is better than the other, False otherwise.
        
        Notes
        -----
        Deb's Feasibility Rules:
        1. If both are feasible, compare fitness values
        2. If one is feasible and one is not, feasible wins
        3. If both are infeasible, smaller constraint violation wins
        
        References
        ----------
        Deb, K. (2000). An efficient constraint handling method for genetic
        algorithms. Computer Methods in Applied Mechanics and Engineering,
        186(2-4), 311-338.
        """
        if not use_feasibility_rules:
            return self.fitness < other.fitness
        
        # Rule 1: Both feasible - compare fitness
        if self.is_feasible and other.is_feasible:
            return self.fitness < other.fitness
        
        # Rule 2: One feasible - feasible wins
        if self.is_feasible and not other.is_feasible:
            return True
        if not self.is_feasible and other.is_feasible:
            return False
        
        # Rule 3: Both infeasible - smaller violation wins
        return self.constraint_violation < other.constraint_violation
    
    def distance_to(self, other: 'Individual') -> float:
        """
        Calculate Euclidean distance to another individual.
        
        Parameters
        ----------
        other : Individual
            The individual to calculate distance to.
        
        Returns
        -------
        float
            Euclidean distance between position vectors.
        """
        return np.linalg.norm(self.position - other.position)
    
    def __lt__(self, other: 'Individual') -> bool:
        """Less than operator for sorting (uses is_better_than)."""
        return self.is_better_than(other)
    
    def __le__(self, other: 'Individual') -> bool:
        """Less than or equal operator."""
        return self.is_better_than(other) or self.fitness == other.fitness
    
    def __eq__(self, other: 'Individual') -> bool:
        """Equality operator (based on position)."""
        if not isinstance(other, Individual):
            return False
        return np.allclose(self.position, other.position)
    
    def __repr__(self) -> str:
        """String representation of the individual."""
        status = "feasible" if self.is_feasible else "infeasible"
        return (f"Individual(fitness={self.fitness:.6f}, "
                f"dimensions={self.dimensions}, "
                f"status={status})")
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()
