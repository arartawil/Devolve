"""
Crossover operators for Differential Evolution.

This module implements binomial, exponential, and arithmetic crossover strategies.
"""

import numpy as np


def binomial_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    CR: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Binomial (uniform) crossover operator.
    
    Each dimension of the trial vector is inherited from either the mutant
    or target vector based on the crossover probability CR.
    
    .. math::
        u_{i,j} = \\begin{cases}
        v_{i,j} & \\text{if } rand_j \\leq CR \\text{ or } j = j_{rand} \\\\
        x_{i,j} & \\text{otherwise}
        \\end{cases}
    
    where :math:`j_{rand}` is a randomly chosen dimension to ensure at least
    one parameter is inherited from the mutant.
    
    Parameters
    ----------
    target : np.ndarray
        Target individual's position vector.
    mutant : np.ndarray
        Mutant vector from mutation operation.
    CR : float
        Crossover probability (0.0-1.0).
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Trial vector after crossover.
    
    Notes
    -----
    This is the most commonly used crossover in DE. It treats each dimension
    independently, making it suitable for separable problems.
    
    References
    ----------
    Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient
    heuristic for global optimization over continuous spaces. Journal of global
    optimization, 11(4), 341-359.
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> target = np.array([1.0, 2.0, 3.0, 4.0])
    >>> mutant = np.array([5.0, 6.0, 7.0, 8.0])
    >>> trial = binomial_crossover(target, mutant, CR=0.9, rng=rng)
    """
    dimensions = len(target)
    
    # Generate random values for each dimension
    rand_values = rng.random(dimensions)
    
    # Ensure at least one dimension from mutant (j_rand)
    j_rand = rng.integers(0, dimensions)
    
    # Create trial vector
    trial = np.where(
        (rand_values <= CR) | (np.arange(dimensions) == j_rand),
        mutant,
        target
    )
    
    return trial


def exponential_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    CR: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Exponential (two-point) crossover operator.
    
    Copies L consecutive dimensions from the mutant vector, where L is
    geometrically distributed with parameter CR.
    
    Algorithm:
    1. Start at random dimension j_rand
    2. Copy dimensions from mutant while rand() < CR
    3. Stop and use target for remaining dimensions
    
    Parameters
    ----------
    target : np.ndarray
        Target individual's position vector.
    mutant : np.ndarray
        Mutant vector from mutation operation.
    CR : float
        Crossover probability (0.0-1.0).
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Trial vector after crossover.
    
    Notes
    -----
    This crossover maintains consecutive sequences of parameters, making it
    more suitable for problems with parameter dependencies (non-separable problems).
    
    The number of inherited dimensions L follows a geometric distribution:
    P(L = l) = CR^l * (1 - CR) for l < D, or CR^l for l = D
    
    Examples
    --------
    >>> trial = exponential_crossover(target, mutant, CR=0.9, rng=rng)
    """
    dimensions = len(target)
    trial = target.copy()
    
    # Random starting dimension
    j = rng.integers(0, dimensions)
    L = 0  # Number of parameters to copy
    
    # Copy consecutive dimensions while rand() < CR
    while True:
        trial[j] = mutant[j]
        L += 1
        j = (j + 1) % dimensions  # Wrap around
        
        # Stop if we've copied all dimensions or probability test fails
        if L >= dimensions or rng.random() >= CR:
            break
    
    return trial


def arithmetic_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Arithmetic (blend) crossover operator.
    
    Creates trial vector as weighted average of target and mutant.
    
    .. math::
        \\mathbf{u}_i = \\alpha \\cdot \\mathbf{v}_i + (1 - \\alpha) \\cdot \\mathbf{x}_i
    
    Parameters
    ----------
    target : np.ndarray
        Target individual's position vector.
    mutant : np.ndarray
        Mutant vector from mutation operation.
    alpha : float, optional
        Blending factor (0.0-1.0). Default is 0.5 (equal blend).
    
    Returns
    -------
    np.ndarray
        Trial vector after crossover.
    
    Notes
    -----
    This is less commonly used in DE but can be useful for smooth optimization
    landscapes. When alpha=0.5, creates midpoint between target and mutant.
    
    Examples
    --------
    >>> trial = arithmetic_crossover(target, mutant, alpha=0.7)
    """
    return alpha * mutant + (1 - alpha) * target


def no_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    **kwargs
) -> np.ndarray:
    """
    No crossover - returns mutant vector unchanged.
    
    This can be useful for testing pure mutation strategies or
    for specific algorithm variants that don't use crossover.
    
    Parameters
    ----------
    target : np.ndarray
        Target individual's position vector (unused).
    mutant : np.ndarray
        Mutant vector from mutation operation.
    **kwargs
        Additional arguments (ignored).
    
    Returns
    -------
    np.ndarray
        Mutant vector unchanged.
    
    Examples
    --------
    >>> trial = no_crossover(target, mutant)
    >>> np.array_equal(trial, mutant)
    True
    """
    return mutant.copy()


# Crossover strategy registry
CROSSOVER_STRATEGIES = {
    'binomial': binomial_crossover,
    'bin': binomial_crossover,
    'exponential': exponential_crossover,
    'exp': exponential_crossover,
    'arithmetic': arithmetic_crossover,
    'none': no_crossover,
}


def get_crossover_strategy(name: str):
    """
    Get crossover strategy function by name.
    
    Parameters
    ----------
    name : str
        Name of the crossover strategy. Options:
        - 'binomial' or 'bin': Binomial crossover
        - 'exponential' or 'exp': Exponential crossover
        - 'arithmetic': Arithmetic crossover
        - 'none': No crossover
    
    Returns
    -------
    callable
        Crossover strategy function.
    
    Raises
    ------
    ValueError
        If strategy name is not recognized.
    
    Examples
    --------
    >>> crossover_func = get_crossover_strategy('binomial')
    >>> trial = crossover_func(target, mutant, CR=0.9, rng=rng)
    """
    name_lower = name.lower()
    if name_lower not in CROSSOVER_STRATEGIES:
        available = list(set(CROSSOVER_STRATEGIES.keys()))
        raise ValueError(f"Unknown crossover strategy '{name}'. Available: {available}")
    return CROSSOVER_STRATEGIES[name_lower]


def adaptive_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    CR: float,
    rng: np.random.Generator,
    strategy: str = 'binomial'
) -> np.ndarray:
    """
    Adaptive crossover that selects strategy based on name.
    
    Convenience function that routes to the appropriate crossover strategy.
    
    Parameters
    ----------
    target : np.ndarray
        Target individual's position vector.
    mutant : np.ndarray
        Mutant vector from mutation operation.
    CR : float
        Crossover probability (0.0-1.0).
    rng : np.random.Generator
        Random number generator.
    strategy : str, optional
        Name of crossover strategy. Default is 'binomial'.
    
    Returns
    -------
    np.ndarray
        Trial vector after crossover.
    
    Examples
    --------
    >>> trial = adaptive_crossover(target, mutant, CR=0.9, rng=rng, strategy='exponential')
    """
    crossover_func = get_crossover_strategy(strategy)
    
    # Different crossover strategies have different signatures
    if strategy in ['binomial', 'bin', 'exponential', 'exp']:
        return crossover_func(target, mutant, CR, rng)
    elif strategy == 'arithmetic':
        # Use CR as alpha for arithmetic crossover
        return crossover_func(target, mutant, alpha=CR)
    else:  # 'none'
        return crossover_func(target, mutant)
