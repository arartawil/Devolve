"""
Mutation operators for Differential Evolution.

This module implements all standard DE mutation strategies with vectorized
NumPy operations for efficiency.
"""

from typing import Optional, List
import numpy as np
from ..core.population import Population


def rand_1(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    DE/rand/1 mutation strategy.
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_{r_1} + F \\cdot (\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})
    
    where :math:`r_1, r_2, r_3` are random distinct indices different from :math:`i`.
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor (typically 0.4-1.0).
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    This is a classic mutation strategy that uses three random individuals.
    It provides good exploration but may converge slowly.
    
    References
    ----------
    Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient
    heuristic for global optimization over continuous spaces. Journal of global
    optimization, 11(4), 341-359.
    
    Examples
    --------
    >>> from devolve.core import Population, Individual
    >>> import numpy as np
    >>> pop = Population([Individual(np.array([i, i])) for i in range(5)])
    >>> rng = np.random.default_rng(42)
    >>> mutant = rand_1(pop, target_idx=0, F=0.8, rng=rng)
    """
    # Select three random distinct individuals
    indices = [i for i in range(population.size) if i != target_idx]
    r1, r2, r3 = rng.choice(indices, size=3, replace=False)
    
    # Get positions
    x_r1 = population[r1].position
    x_r2 = population[r2].position
    x_r3 = population[r3].position
    
    # Apply mutation: v = x_r1 + F * (x_r2 - x_r3)
    mutant = x_r1 + F * (x_r2 - x_r3)
    
    return mutant


def best_1(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    DE/best/1 mutation strategy.
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_{best} + F \\cdot (\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2})
    
    where :math:`\\mathbf{x}_{best}` is the best individual in the population.
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    This strategy uses the best individual as the base vector, leading to
    fast convergence but increased risk of premature convergence.
    Recommended for unimodal problems or when fast convergence is desired.
    
    Examples
    --------
    >>> mutant = best_1(pop, target_idx=0, F=0.8, rng=rng)
    """
    # Get best individual
    x_best = population.get_best().position
    
    # Select two random distinct individuals
    indices = [i for i in range(population.size) if i != target_idx]
    r1, r2 = rng.choice(indices, size=2, replace=False)
    
    x_r1 = population[r1].position
    x_r2 = population[r2].position
    
    # Apply mutation: v = x_best + F * (x_r1 - x_r2)
    mutant = x_best + F * (x_r1 - x_r2)
    
    return mutant


def current_to_best_1(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    DE/current-to-best/1 mutation strategy.
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_i + F \\cdot (\\mathbf{x}_{best} - \\mathbf{x}_i) + 
        F \\cdot (\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2})
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    This strategy balances exploration and exploitation by combining the
    current individual with the best individual and a random difference.
    Good for problems requiring a balance between convergence speed and diversity.
    
    Examples
    --------
    >>> mutant = current_to_best_1(pop, target_idx=0, F=0.8, rng=rng)
    """
    # Get target and best
    x_target = population[target_idx].position
    x_best = population.get_best().position
    
    # Select two random distinct individuals
    indices = [i for i in range(population.size) if i != target_idx]
    r1, r2 = rng.choice(indices, size=2, replace=False)
    
    x_r1 = population[r1].position
    x_r2 = population[r2].position
    
    # Apply mutation: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    mutant = x_target + F * (x_best - x_target) + F * (x_r1 - x_r2)
    
    return mutant


def current_to_rand_1(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator,
    K: Optional[float] = None
) -> np.ndarray:
    """
    DE/current-to-rand/1 mutation strategy.
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_i + K \\cdot (\\mathbf{x}_{r_1} - \\mathbf{x}_i) + 
        F \\cdot (\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3})
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor for difference vector.
    rng : np.random.Generator
        Random number generator.
    K : float, optional
        Scaling factor for current-to-rand component. Default is F.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    This strategy is useful for multimodal problems and is used in SaDE.
    
    Examples
    --------
    >>> mutant = current_to_rand_1(pop, target_idx=0, F=0.8, rng=rng)
    """
    if K is None:
        K = F
    
    # Get target
    x_target = population[target_idx].position
    
    # Select three random distinct individuals
    indices = [i for i in range(population.size) if i != target_idx]
    r1, r2, r3 = rng.choice(indices, size=3, replace=False)
    
    x_r1 = population[r1].position
    x_r2 = population[r2].position
    x_r3 = population[r3].position
    
    # Apply mutation
    mutant = x_target + K * (x_r1 - x_target) + F * (x_r2 - x_r3)
    
    return mutant


def rand_2(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    DE/rand/2 mutation strategy.
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_{r_1} + F \\cdot (\\mathbf{x}_{r_2} - \\mathbf{x}_{r_3}) + 
        F \\cdot (\\mathbf{x}_{r_4} - \\mathbf{x}_{r_5})
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    Uses two difference vectors for more diverse search.
    Slower convergence but better exploration of the search space.
    Requires population size >= 6.
    
    Examples
    --------
    >>> mutant = rand_2(pop, target_idx=0, F=0.8, rng=rng)
    """
    if population.size < 6:
        raise ValueError("rand_2 requires population size >= 6")
    
    # Select five random distinct individuals
    indices = [i for i in range(population.size) if i != target_idx]
    r1, r2, r3, r4, r5 = rng.choice(indices, size=5, replace=False)
    
    # Get positions
    x_r1 = population[r1].position
    x_r2 = population[r2].position
    x_r3 = population[r3].position
    x_r4 = population[r4].position
    x_r5 = population[r5].position
    
    # Apply mutation: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    
    return mutant


def best_2(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    DE/best/2 mutation strategy.
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_{best} + F \\cdot (\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2}) + 
        F \\cdot (\\mathbf{x}_{r_3} - \\mathbf{x}_{r_4})
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor.
    rng : np.random.Generator
        Random number generator.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    Combines best individual base with two difference vectors.
    Very fast convergence but highest risk of premature convergence.
    Requires population size >= 5.
    
    Examples
    --------
    >>> mutant = best_2(pop, target_idx=0, F=0.8, rng=rng)
    """
    if population.size < 5:
        raise ValueError("best_2 requires population size >= 5")
    
    # Get best individual
    x_best = population.get_best().position
    
    # Select four random distinct individuals
    indices = [i for i in range(population.size) if i != target_idx]
    r1, r2, r3, r4 = rng.choice(indices, size=4, replace=False)
    
    x_r1 = population[r1].position
    x_r2 = population[r2].position
    x_r3 = population[r3].position
    x_r4 = population[r4].position
    
    # Apply mutation: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
    mutant = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
    
    return mutant


def current_to_pbest_1(
    population: Population,
    target_idx: int,
    F: float,
    rng: np.random.Generator,
    p: float = 0.1,
    archive: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    """
    DE/current-to-pbest/1 mutation strategy (used in JADE and SHADE).
    
    Mutation formula:
    
    .. math::
        \\mathbf{v}_i = \\mathbf{x}_i + F \\cdot (\\mathbf{x}_{pbest} - \\mathbf{x}_i) + 
        F \\cdot (\\mathbf{x}_{r_1} - \\mathbf{x}_{r_2})
    
    where :math:`\\mathbf{x}_{pbest}` is randomly selected from the top p% individuals.
    
    Parameters
    ----------
    population : Population
        Current population.
    target_idx : int
        Index of the target individual.
    F : float
        Scaling factor.
    rng : np.random.Generator
        Random number generator.
    p : float, optional
        Percentage of top individuals to select from (0 < p <= 1). Default is 0.1.
    archive : List[np.ndarray], optional
        External archive of previous solutions. x_r2 can be from archive. Default is None.
    
    Returns
    -------
    np.ndarray
        Mutant vector.
    
    Notes
    -----
    This is a more sophisticated strategy that biases towards good solutions
    without focusing solely on the single best individual. Used in JADE and SHADE.
    
    References
    ----------
    Zhang, J., & Sanderson, A. C. (2009). JADE: Adaptive differential evolution
    with optional external archive. IEEE Transactions on evolutionary computation,
    13(5), 945-958.
    
    Examples
    --------
    >>> mutant = current_to_pbest_1(pop, target_idx=0, F=0.8, rng=rng, p=0.1)
    """
    # Get target
    x_target = population[target_idx].position
    
    # Get top p% individuals
    top_individuals = population.get_top_p_percent(p)
    x_pbest = rng.choice(top_individuals).position
    
    # Select r1 from population (different from target)
    indices = [i for i in range(population.size) if i != target_idx]
    r1 = rng.choice(indices)
    x_r1 = population[r1].position
    
    # Select r2 from population or archive
    if archive is not None and len(archive) > 0 and rng.random() < 0.5:
        # Select from archive
        x_r2 = archive[rng.integers(len(archive))]
    else:
        # Select from population
        r2 = rng.choice(indices)
        x_r2 = population[r2].position
    
    # Apply mutation
    mutant = x_target + F * (x_pbest - x_target) + F * (x_r1 - x_r2)
    
    return mutant


# Mutation strategy registry
MUTATION_STRATEGIES = {
    'rand/1': rand_1,
    'best/1': best_1,
    'current-to-best/1': current_to_best_1,
    'current-to-rand/1': current_to_rand_1,
    'rand/2': rand_2,
    'best/2': best_2,
    'current-to-pbest/1': current_to_pbest_1,
}


def get_mutation_strategy(name: str):
    """
    Get mutation strategy function by name.
    
    Parameters
    ----------
    name : str
        Name of the mutation strategy.
    
    Returns
    -------
    callable
        Mutation strategy function.
    
    Raises
    ------
    ValueError
        If strategy name is not recognized.
    """
    if name not in MUTATION_STRATEGIES:
        available = list(MUTATION_STRATEGIES.keys())
        raise ValueError(f"Unknown mutation strategy '{name}'. Available: {available}")
    return MUTATION_STRATEGIES[name]
