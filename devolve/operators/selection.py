"""
Selection operators for Differential Evolution.

This module implements various selection strategies for choosing between
parent and offspring individuals.
"""

from typing import List
import numpy as np
from ..core.individual import Individual


def greedy_selection(
    target: Individual,
    trial: Individual,
    use_feasibility_rules: bool = True
) -> Individual:
    """
    Greedy (elitist) selection operator.
    
    Selects the better individual between target and trial.
    This is the standard selection method in Differential Evolution.
    
    Selection rule:
    
    .. math::
        \\mathbf{x}_i(t+1) = \\begin{cases}
        \\mathbf{u}_i(t) & \\text{if } f(\\mathbf{u}_i) < f(\\mathbf{x}_i) \\\\
        \\mathbf{x}_i(t) & \\text{otherwise}
        \\end{cases}
    
    Parameters
    ----------
    target : Individual
        Current target individual (parent).
    trial : Individual
        New trial individual (offspring).
    use_feasibility_rules : bool, optional
        If True, use Deb's feasibility rules for constrained problems.
        Default is True.
    
    Returns
    -------
    Individual
        The selected (better) individual.
    
    Notes
    -----
    When use_feasibility_rules=True, applies Deb's rules:
    1. If both feasible: compare fitness
    2. If one feasible: feasible wins
    3. If both infeasible: smaller violation wins
    
    References
    ----------
    Deb, K. (2000). An efficient constraint handling method for genetic
    algorithms. Computer Methods in Applied Mechanics and Engineering,
    186(2-4), 311-338.
    
    Examples
    --------
    >>> from devolve.core import Individual
    >>> import numpy as np
    >>> target = Individual(position=np.array([1.0, 2.0]), fitness=10.0)
    >>> trial = Individual(position=np.array([1.5, 2.5]), fitness=8.0)
    >>> selected = greedy_selection(target, trial)
    >>> selected.fitness
    8.0
    """
    if trial.is_better_than(target, use_feasibility_rules):
        return trial.copy()
    else:
        return target.copy()


def tournament_selection(
    individuals: List[Individual],
    tournament_size: int = 2,
    rng: np.random.Generator = None,
    use_feasibility_rules: bool = True
) -> Individual:
    """
    Tournament selection operator.
    
    Randomly selects tournament_size individuals and returns the best one.
    
    Parameters
    ----------
    individuals : List[Individual]
        List of individuals to select from.
    tournament_size : int, optional
        Number of individuals in tournament. Default is 2.
    rng : np.random.Generator, optional
        Random number generator. If None, creates new one.
    use_feasibility_rules : bool, optional
        If True, use Deb's feasibility rules. Default is True.
    
    Returns
    -------
    Individual
        Winner of the tournament.
    
    Notes
    -----
    Tournament selection introduces selection pressure while maintaining
    diversity. Larger tournament sizes increase selection pressure.
    
    Examples
    --------
    >>> individuals = [Individual(np.random.randn(5)) for _ in range(10)]
    >>> winner = tournament_selection(individuals, tournament_size=3, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if tournament_size > len(individuals):
        tournament_size = len(individuals)
    
    # Randomly select tournament participants
    tournament_indices = rng.choice(len(individuals), size=tournament_size, replace=False)
    tournament = [individuals[i] for i in tournament_indices]
    
    # Find best in tournament
    best = tournament[0]
    for ind in tournament[1:]:
        if ind.is_better_than(best, use_feasibility_rules):
            best = ind
    
    return best.copy()


def probabilistic_selection(
    target: Individual,
    trial: Individual,
    temperature: float = 1.0,
    rng: np.random.Generator = None
) -> Individual:
    """
    Probabilistic (simulated annealing-style) selection.
    
    Accepts worse solutions with probability based on fitness difference
    and temperature parameter.
    
    Acceptance probability:
    
    .. math::
        P(accept) = \\exp\\left(-\\frac{\\Delta f}{T}\\right)
    
    where :math:`\\Delta f = f(trial) - f(target)` and :math:`T` is temperature.
    
    Parameters
    ----------
    target : Individual
        Current target individual.
    trial : Individual
        New trial individual.
    temperature : float, optional
        Temperature parameter controlling acceptance of worse solutions.
        Higher values = more exploration. Default is 1.0.
    rng : np.random.Generator, optional
        Random number generator. If None, creates new one.
    
    Returns
    -------
    Individual
        Selected individual.
    
    Notes
    -----
    This operator allows occasional acceptance of worse solutions to escape
    local optima. The temperature parameter should typically decrease over time.
    
    Examples
    --------
    >>> selected = probabilistic_selection(target, trial, temperature=0.5, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Always accept if trial is better
    if trial.is_better_than(target):
        return trial.copy()
    
    # Accept worse solution probabilistically
    delta_f = trial.fitness - target.fitness
    
    # Avoid overflow in exp
    if delta_f / temperature > 100:
        acceptance_prob = 0.0
    else:
        acceptance_prob = np.exp(-delta_f / temperature)
    
    if rng.random() < acceptance_prob:
        return trial.copy()
    else:
        return target.copy()


def rank_based_selection(
    individuals: List[Individual],
    rng: np.random.Generator = None,
    use_feasibility_rules: bool = True
) -> Individual:
    """
    Rank-based selection operator.
    
    Selection probability is proportional to rank rather than absolute fitness,
    reducing selection pressure and maintaining diversity.
    
    Parameters
    ----------
    individuals : List[Individual]
        List of individuals to select from.
    rng : np.random.Generator, optional
        Random number generator. If None, creates new one.
    use_feasibility_rules : bool, optional
        If True, use Deb's feasibility rules for ranking. Default is True.
    
    Returns
    -------
    Individual
        Selected individual.
    
    Notes
    -----
    Ranks are assigned after sorting (best = rank 1).
    Selection probability is inversely proportional to rank.
    
    Examples
    --------
    >>> selected = rank_based_selection(individuals, rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sort individuals (best first)
    if use_feasibility_rules:
        sorted_inds = sorted(individuals, key=lambda x: (
            not x.is_feasible,
            x.constraint_violation if not x.is_feasible else x.fitness
        ))
    else:
        sorted_inds = sorted(individuals, key=lambda x: x.fitness)
    
    # Assign selection probabilities based on rank
    n = len(sorted_inds)
    ranks = np.arange(n, 0, -1)  # Best = highest rank
    probabilities = ranks / ranks.sum()
    
    # Select based on probabilities
    selected_idx = rng.choice(n, p=probabilities)
    return sorted_inds[selected_idx].copy()


def adaptive_selection(
    target: Individual,
    trial: Individual,
    generation: int,
    max_generations: int,
    use_feasibility_rules: bool = True
) -> Individual:
    """
    Adaptive selection that transitions from exploration to exploitation.
    
    Early in optimization: more probabilistic (allows worse solutions)
    Late in optimization: more greedy (stricter selection)
    
    Parameters
    ----------
    target : Individual
        Current target individual.
    trial : Individual
        New trial individual.
    generation : int
        Current generation number.
    max_generations : int
        Maximum number of generations.
    use_feasibility_rules : bool, optional
        If True, use Deb's feasibility rules. Default is True.
    
    Returns
    -------
    Individual
        Selected individual.
    
    Notes
    -----
    Uses greedy selection, but the implementation can be extended to
    incorporate probabilistic acceptance early in the run.
    
    Examples
    --------
    >>> selected = adaptive_selection(target, trial, generation=50, max_generations=1000)
    """
    # Simple adaptive strategy: use greedy selection
    # Can be extended with temperature schedule for probabilistic selection
    return greedy_selection(target, trial, use_feasibility_rules)


# Selection strategy registry
SELECTION_STRATEGIES = {
    'greedy': greedy_selection,
    'tournament': tournament_selection,
    'probabilistic': probabilistic_selection,
    'rank_based': rank_based_selection,
    'adaptive': adaptive_selection,
}


def get_selection_strategy(name: str):
    """
    Get selection strategy function by name.
    
    Parameters
    ----------
    name : str
        Name of the selection strategy. Options:
        - 'greedy': Standard greedy selection
        - 'tournament': Tournament selection
        - 'probabilistic': Probabilistic selection
        - 'rank_based': Rank-based selection
        - 'adaptive': Adaptive selection
    
    Returns
    -------
    callable
        Selection strategy function.
    
    Raises
    ------
    ValueError
        If strategy name is not recognized.
    
    Examples
    --------
    >>> selection_func = get_selection_strategy('greedy')
    >>> selected = selection_func(target, trial)
    """
    name_lower = name.lower()
    if name_lower not in SELECTION_STRATEGIES:
        available = list(SELECTION_STRATEGIES.keys())
        raise ValueError(f"Unknown selection strategy '{name}'. Available: {available}")
    return SELECTION_STRATEGIES[name_lower]
