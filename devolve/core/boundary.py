"""
Boundary handling strategies for constraint violations.
"""

from enum import Enum
from typing import Optional
import numpy as np


class BoundaryStrategy(Enum):
    """
    Enumeration of available boundary handling strategies.
    
    Attributes
    ----------
    CLIP : str
        Clip (saturate) to boundary values.
    RANDOM : str
        Replace with random values within bounds.
    REFLECT : str
        Reflect across the boundary.
    WRAP : str
        Wrap around using periodic boundaries.
    RESAMPLE : str
        Resample until within bounds (with max attempts).
    """
    CLIP = "clip"
    RANDOM = "random"
    REFLECT = "reflect"
    WRAP = "wrap"
    RESAMPLE = "resample"


class BoundaryHandler:
    """
    Handles boundary constraint violations in Differential Evolution.
    
    When offspring solutions violate the search space boundaries,
    this class provides various strategies to handle them.
    
    Parameters
    ----------
    strategy : BoundaryStrategy or str, optional
        The boundary handling strategy to use. Default is CLIP.
    bounds : np.ndarray, optional
        Array of shape (dimensions, 2) with [lower, upper] bounds.
        Must be set before calling handle(). Default is None.
    rng : np.random.Generator, optional
        Random number generator for strategies that need randomness.
        Default is None (creates new generator).
    
    Attributes
    ----------
    strategy : BoundaryStrategy
        The active boundary handling strategy.
    bounds : np.ndarray
        The search space bounds.
    rng : np.random.Generator
        Random number generator.
    
    Examples
    --------
    >>> import numpy as np
    >>> bounds = np.array([[-5, 5], [-5, 5]])
    >>> handler = BoundaryHandler(strategy="clip", bounds=bounds)
    >>> x = np.array([6.0, -7.0])  # Violates bounds
    >>> x_fixed = handler.handle(x)
    >>> print(x_fixed)
    [5. -5.]
    
    References
    ----------
    Lampinen, J. (2002). A constraint handling approach for the differential
    evolution algorithm. Proceedings of the 2002 Congress on Evolutionary
    Computation, 2, 1468-1473.
    """
    
    def __init__(
        self,
        strategy: str = "clip",
        bounds: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None
    ):
        """Initialize the boundary handler."""
        if isinstance(strategy, str):
            try:
                self.strategy = BoundaryStrategy(strategy.lower())
            except ValueError:
                valid = [s.value for s in BoundaryStrategy]
                raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid}")
        else:
            self.strategy = strategy
        
        self.bounds = bounds
        self.rng = rng if rng is not None else np.random.default_rng()
    
    def set_bounds(self, bounds: np.ndarray) -> None:
        """
        Set the search space bounds.
        
        Parameters
        ----------
        bounds : np.ndarray
            Array of shape (dimensions, 2) with [lower, upper] bounds.
        """
        self.bounds = bounds
    
    def handle(self, x: np.ndarray) -> np.ndarray:
        """
        Apply boundary handling to vector x.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector that may violate bounds.
        
        Returns
        -------
        np.ndarray
            Position vector with boundary violations corrected.
        
        Raises
        ------
        ValueError
            If bounds have not been set.
        """
        if self.bounds is None:
            raise ValueError("Bounds must be set before handling")
        
        if self.strategy == BoundaryStrategy.CLIP:
            return self._clip(x)
        elif self.strategy == BoundaryStrategy.RANDOM:
            return self._random(x)
        elif self.strategy == BoundaryStrategy.REFLECT:
            return self._reflect(x)
        elif self.strategy == BoundaryStrategy.WRAP:
            return self._wrap(x)
        elif self.strategy == BoundaryStrategy.RESAMPLE:
            return self._resample(x)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _clip(self, x: np.ndarray) -> np.ndarray:
        """
        Clip (saturate) to boundary values.
        
        If x_i < lower_i, set x_i = lower_i
        If x_i > upper_i, set x_i = upper_i
        
        This is the most common and simplest strategy.
        """
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])
    
    def _random(self, x: np.ndarray) -> np.ndarray:
        """
        Replace violating dimensions with random values within bounds.
        
        For each dimension i:
        If x_i violates bounds, x_i = lower_i + rand() * (upper_i - lower_i)
        
        This increases diversity but may be too disruptive.
        """
        x_new = x.copy()
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        
        # Find violating dimensions
        violates_lower = x < lower
        violates_upper = x > upper
        violates = violates_lower | violates_upper
        
        # Replace with random values
        if np.any(violates):
            n_violates = np.sum(violates)
            x_new[violates] = (lower[violates] + 
                              self.rng.random(n_violates) * 
                              (upper[violates] - lower[violates]))
        
        return x_new
    
    def _reflect(self, x: np.ndarray) -> np.ndarray:
        """
        Reflect across the boundary.
        
        If x_i < lower_i, set x_i = lower_i + (lower_i - x_i)
        If x_i > upper_i, set x_i = upper_i - (x_i - upper_i)
        
        If still violates after reflection, clip to bounds.
        This preserves distance from boundary.
        """
        x_new = x.copy()
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        
        # Reflect lower violations
        violates_lower = x < lower
        if np.any(violates_lower):
            x_new[violates_lower] = lower[violates_lower] + (lower[violates_lower] - x[violates_lower])
        
        # Reflect upper violations
        violates_upper = x > upper
        if np.any(violates_upper):
            x_new[violates_upper] = upper[violates_upper] - (x[violates_upper] - upper[violates_upper])
        
        # Clip if still violates (can happen with large violations)
        x_new = np.clip(x_new, lower, upper)
        
        return x_new
    
    def _wrap(self, x: np.ndarray) -> np.ndarray:
        """
        Wrap around using periodic boundaries (toroidal topology).
        
        x_i = lower_i + ((x_i - lower_i) mod (upper_i - lower_i))
        
        This is useful for problems with periodic nature.
        """
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        range_width = upper - lower
        
        # Normalize to [0, range_width]
        x_normalized = x - lower
        
        # Apply modulo
        x_wrapped = np.mod(x_normalized, range_width)
        
        # Shift back to [lower, upper]
        return lower + x_wrapped
    
    def _resample(self, x: np.ndarray, max_attempts: int = 100) -> np.ndarray:
        """
        Resample violating dimensions until within bounds.
        
        This strategy keeps trying random values for violating dimensions
        until they fall within bounds. Falls back to clipping after max_attempts.
        
        Parameters
        ----------
        max_attempts : int
            Maximum number of resampling attempts before falling back to clip.
        """
        x_new = x.copy()
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        
        for attempt in range(max_attempts):
            violates_lower = x_new < lower
            violates_upper = x_new > upper
            violates = violates_lower | violates_upper
            
            if not np.any(violates):
                break
            
            # Resample violating dimensions
            n_violates = np.sum(violates)
            x_new[violates] = (lower[violates] + 
                              self.rng.random(n_violates) * 
                              (upper[violates] - lower[violates]))
        
        # Final clip if still violates
        return np.clip(x_new, lower, upper)
    
    def is_within_bounds(self, x: np.ndarray) -> bool:
        """
        Check if vector x is within bounds.
        
        Parameters
        ----------
        x : np.ndarray
            Position vector to check.
        
        Returns
        -------
        bool
            True if all dimensions are within bounds, False otherwise.
        """
        if self.bounds is None:
            raise ValueError("Bounds must be set before checking")
        
        return np.all(x >= self.bounds[:, 0]) and np.all(x <= self.bounds[:, 1])
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BoundaryHandler(strategy={self.strategy.value})"
