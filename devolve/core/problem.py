"""
Problem class for defining optimization problems.
"""

from typing import Callable, List, Optional, Tuple, Union
import numpy as np


class Problem:
    """
    Defines an optimization problem with objective function, bounds, and constraints.
    
    This class encapsulates all information about an optimization problem,
    including the objective function, search space bounds, constraints,
    and known optimal values (if available).
    
    Parameters
    ----------
    objective_function : Callable[[np.ndarray], float]
        The objective function to minimize. Takes a 1D numpy array and returns
        a scalar fitness value.
    bounds : Union[Tuple[float, float], List[Tuple[float, float]]]
        Search space bounds. Can be:
        - A tuple (lower, upper) for same bounds on all dimensions
        - A list of tuples [(lower_1, upper_1), ..., (lower_d, upper_d)]
    dimensions : int
        Number of dimensions (decision variables).
    constraints : Optional[List[Callable[[np.ndarray], float]]], optional
        List of constraint functions. Each function should return <= 0 for
        feasible solutions. Default is None.
    optimum : Optional[float], optional
        Known global optimum value. Default is None.
    optimum_position : Optional[np.ndarray], optional
        Known global optimum position. Default is None.
    name : Optional[str], optional
        Name of the problem. Default is None.
    
    Attributes
    ----------
    objective_function : Callable
        The objective function.
    bounds : np.ndarray
        Array of shape (dimensions, 2) with [lower, upper] bounds per dimension.
    dimensions : int
        Problem dimensionality.
    constraints : List[Callable]
        List of constraint functions.
    optimum : Optional[float]
        Known optimal value.
    optimum_position : Optional[np.ndarray]
        Known optimal position.
    name : str
        Problem name.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Simple sphere function
    >>> def sphere(x):
    ...     return np.sum(x**2)
    >>> problem = Problem(
    ...     objective_function=sphere,
    ...     bounds=(-5.0, 5.0),
    ...     dimensions=10,
    ...     optimum=0.0,
    ...     name="Sphere"
    ... )
    >>> x = np.zeros(10)
    >>> print(problem.evaluate(x))
    0.0
    
    >>> # Problem with constraints
    >>> def constrained_func(x):
    ...     return x[0]**2 + x[1]**2
    >>> def constraint1(x):
    ...     return x[0] + x[1] - 1  # x[0] + x[1] <= 1
    >>> problem = Problem(
    ...     objective_function=constrained_func,
    ...     bounds=[(-2.0, 2.0), (-2.0, 2.0)],
    ...     dimensions=2,
    ...     constraints=[constraint1]
    ... )
    """
    
    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
        dimensions: int,
        constraints: Optional[List[Callable[[np.ndarray], float]]] = None,
        optimum: Optional[float] = None,
        optimum_position: Optional[np.ndarray] = None,
        name: Optional[str] = None
    ):
        """Initialize the optimization problem."""
        self.objective_function = objective_function
        self.dimensions = dimensions
        self.constraints = constraints if constraints else []
        self.optimum = optimum
        self.optimum_position = optimum_position
        self.name = name if name else "UnnamedProblem"
        
        # Process bounds
        if isinstance(bounds, tuple) and len(bounds) == 2:
            # Same bounds for all dimensions
            self.bounds = np.array([[bounds[0], bounds[1]] for _ in range(dimensions)])
        elif isinstance(bounds, list):
            # Individual bounds per dimension
            if len(bounds) != dimensions:
                raise ValueError(f"Length of bounds ({len(bounds)}) must match dimensions ({dimensions})")
            self.bounds = np.array(bounds)
        else:
            raise ValueError("Bounds must be a tuple (lower, upper) or list of tuples")
        
        # Validate bounds
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Lower bounds must be less than upper bounds")
    
    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate the objective function at position x.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector to evaluate.
        
        Returns
        -------
        float
            Objective function value.
        """
        return self.objective_function(x)
    
    def evaluate_constraints(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluate all constraints at position x.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector to evaluate.
        
        Returns
        -------
        is_feasible : bool
            True if all constraints are satisfied (g(x) <= 0).
        total_violation : float
            Sum of constraint violations. Zero if feasible.
        
        Notes
        -----
        Constraint violation is calculated as:
        violation = Σ max(0, g_i(x))
        where g_i are constraint functions.
        """
        if not self.constraints:
            return True, 0.0
        
        total_violation = 0.0
        for constraint in self.constraints:
            value = constraint(x)
            if value > 0:
                total_violation += value
        
        is_feasible = total_violation == 0.0
        return is_feasible, total_violation
    
    def is_within_bounds(self, x: np.ndarray) -> bool:
        """
        Check if position x is within bounds.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector to check.
        
        Returns
        -------
        bool
            True if within bounds, False otherwise.
        """
        return np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1])
    
    def clip_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """
        Clip position x to bounds.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector to clip.
        
        Returns
        -------
        np.ndarray
            Clipped position vector.
        """
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    
    def random_solution(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Generate a random solution within bounds.
        
        Parameters
        ----------
        rng : np.random.Generator, optional
            Random number generator. If None, uses default RNG.
        
        Returns
        -------
        np.ndarray
            Random position vector within bounds.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        return lower + rng.random(self.dimensions) * (upper - lower)
    
    def get_bounds_range(self) -> np.ndarray:
        """
        Get the range (width) of bounds for each dimension.
        
        Returns
        -------
        np.ndarray
            Array of ranges (upper - lower) per dimension.
        """
        return self.bounds[:, 1] - self.bounds[:, 0]
    
    def get_bounds_center(self) -> np.ndarray:
        """
        Get the center point of the search space.
        
        Returns
        -------
        np.ndarray
            Center position vector.
        """
        return (self.bounds[:, 0] + self.bounds[:, 1]) / 2.0
    
    def distance_to_optimum(self, x: np.ndarray) -> Optional[float]:
        """
        Calculate Euclidean distance to known optimum position.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector.
        
        Returns
        -------
        float or None
            Distance to optimum, or None if optimum position is unknown.
        """
        if self.optimum_position is None:
            return None
        return np.linalg.norm(x - self.optimum_position)
    
    def error_to_optimum(self, fitness: float) -> Optional[float]:
        """
        Calculate error relative to known optimum value.
        
        Parameters
        ----------
        fitness : float
            Fitness value to compare.
        
        Returns
        -------
        float or None
            Absolute error |fitness - optimum|, or None if optimum is unknown.
        """
        if self.optimum is None:
            return None
        return abs(fitness - self.optimum)
    
    def __repr__(self) -> str:
        """String representation of the problem."""
        opt_str = f", optimum={self.optimum:.6f}" if self.optimum is not None else ""
        constr_str = f", constraints={len(self.constraints)}" if self.constraints else ""
        return (f"Problem(name='{self.name}', "
                f"dimensions={self.dimensions}"
                f"{opt_str}"
                f"{constr_str})")
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()
