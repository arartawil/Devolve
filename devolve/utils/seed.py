"""
Random Seed Management

Utilities for ensuring reproducibility in stochastic optimization algorithms.
"""

import numpy as np
import random
from typing import List, Optional


# Global seed state
_global_seed: Optional[int] = None


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Sets seeds for:
    - NumPy's random number generator
    - Python's built-in random module
    - Stores globally for DEvolve
    
    Parameters
    ----------
    seed : int
        Random seed value.
    
    Examples
    --------
    >>> set_seed(42)
    >>> x = np.random.randn(5)
    >>> set_seed(42)
    >>> y = np.random.randn(5)
    >>> assert np.allclose(x, y)  # Same values
    
    Notes
    -----
    Call this at the beginning of your script to ensure reproducible results.
    """
    global _global_seed
    _global_seed = seed
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set Python's random seed
    random.seed(seed)
    
    # For compatibility with newer NumPy versions
    try:
        np.random.default_rng(seed)
    except AttributeError:
        pass


def get_seed() -> Optional[int]:
    """
    Get the currently set global seed.
    
    Returns
    -------
    int or None
        The current global seed, or None if not set.
    
    Examples
    --------
    >>> set_seed(42)
    >>> print(get_seed())
    42
    """
    return _global_seed


def get_seed_sequence(
    n_runs: int,
    base_seed: Optional[int] = None
) -> List[int]:
    """
    Generate a sequence of seeds for multiple independent runs.
    
    Creates deterministic but different seeds for each run to ensure
    independence while maintaining reproducibility.
    
    Parameters
    ----------
    n_runs : int
        Number of seeds to generate.
    base_seed : int, optional
        Base seed for generation. If None, uses global seed or random.
    
    Returns
    -------
    List[int]
        List of n_runs seeds.
    
    Examples
    --------
    >>> seeds = get_seed_sequence(5, base_seed=42)
    >>> print(seeds)
    [42, 43, 44, 45, 46]
    
    >>> # Or use random generation
    >>> seeds = get_seed_sequence(5, base_seed=100)
    >>> print(len(seeds))
    5
    
    Notes
    -----
    Seeds are generated as: base_seed + i for i in range(n_runs)
    This ensures they are different but reproducible.
    """
    if base_seed is None:
        base_seed = _global_seed if _global_seed is not None else np.random.randint(0, 10000)
    
    return [base_seed + i for i in range(n_runs)]


def ensure_reproducibility(seed: int = 42) -> dict:
    """
    Ensure full reproducibility for experiments.
    
    Sets all random seeds and returns environment information for documentation.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value. Default is 42.
    
    Returns
    -------
    dict
        Dictionary with reproducibility information:
        - 'seed': The seed used
        - 'numpy_version': NumPy version
        - 'python_version': Python version
    
    Examples
    --------
    >>> info = ensure_reproducibility(seed=42)
    >>> print(f"Using seed: {info['seed']}")
    >>> print(f"NumPy version: {info['numpy_version']}")
    
    Notes
    -----
    For full reproducibility:
    1. Call this function at the start of your script
    2. Use the same versions of all libraries
    3. Run on the same hardware (if using parallel execution)
    4. Document the returned information
    """
    import sys
    
    # Set all seeds
    set_seed(seed)
    
    # Collect environment info
    info = {
        'seed': seed,
        'numpy_version': np.__version__,
        'python_version': sys.version,
    }
    
    return info


def create_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a NumPy random number generator with optional seed.
    
    Preferred method for modern NumPy (>= 1.17) applications.
    
    Parameters
    ----------
    seed : int, optional
        Random seed. If None, uses global seed or creates unseeded generator.
    
    Returns
    -------
    np.random.Generator
        NumPy random number generator.
    
    Examples
    --------
    >>> rng = create_rng(seed=42)
    >>> x = rng.standard_normal(5)
    >>> print(x.shape)
    (5,)
    
    Notes
    -----
    Use this for algorithm implementations that need independent RNG instances.
    """
    if seed is None:
        seed = _global_seed
    
    if seed is not None:
        return np.random.default_rng(seed)
    else:
        return np.random.default_rng()


def split_seed(seed: int, n_splits: int) -> List[int]:
    """
    Split a seed into multiple independent seeds using hashing.
    
    Useful for parallel runs where you want different but reproducible seeds.
    
    Parameters
    ----------
    seed : int
        Base seed to split.
    n_splits : int
        Number of seeds to generate.
    
    Returns
    -------
    List[int]
        List of n_splits independent seeds.
    
    Examples
    --------
    >>> seeds = split_seed(42, 3)
    >>> print(len(seeds))
    3
    >>> # Each seed is different but reproducible
    >>> seeds2 = split_seed(42, 3)
    >>> assert seeds == seeds2
    
    Notes
    -----
    Uses NumPy's SeedSequence for proper seed splitting.
    """
    try:
        # Modern NumPy approach
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_splits)
        return [int(s.entropy) for s in child_seeds]
    except AttributeError:
        # Fallback for older NumPy
        rng = np.random.RandomState(seed)
        return [int(rng.randint(0, 2**31 - 1)) for _ in range(n_splits)]


def get_reproducible_state() -> dict:
    """
    Get current random state for later restoration.
    
    Returns
    -------
    dict
        Dictionary containing random states from NumPy and Python's random.
    
    Examples
    --------
    >>> state = get_reproducible_state()
    >>> # ... do some random operations ...
    >>> restore_reproducible_state(state)
    >>> # ... random sequence continues from saved point ...
    
    See Also
    --------
    restore_reproducible_state : Restore saved random state
    """
    return {
        'numpy_state': np.random.get_state(),
        'random_state': random.getstate(),
        'global_seed': _global_seed
    }


def restore_reproducible_state(state: dict) -> None:
    """
    Restore random state from saved state.
    
    Parameters
    ----------
    state : dict
        State dictionary from get_reproducible_state().
    
    Examples
    --------
    >>> set_seed(42)
    >>> x1 = np.random.randn(3)
    >>> state = get_reproducible_state()
    >>> x2 = np.random.randn(3)
    >>> restore_reproducible_state(state)
    >>> x3 = np.random.randn(3)
    >>> assert np.allclose(x2, x3)  # Same values after restoration
    
    See Also
    --------
    get_reproducible_state : Save current random state
    """
    global _global_seed
    
    if 'numpy_state' in state:
        np.random.set_state(state['numpy_state'])
    
    if 'random_state' in state:
        random.setstate(state['random_state'])
    
    if 'global_seed' in state:
        _global_seed = state['global_seed']
