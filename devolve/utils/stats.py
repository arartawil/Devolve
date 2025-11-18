"""
Statistical Tests for Algorithm Comparison

Provides statistical hypothesis tests and effect size calculations for
comparing optimization algorithms on benchmark problems.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
import warnings


def wilcoxon_test(
    results1: List[float],
    results2: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, Union[float, str]]:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Tests whether two related paired samples come from the same distribution.
    Non-parametric alternative to paired t-test.
    
    Parameters
    ----------
    results1 : List[float]
        Results from first algorithm (paired with results2).
    results2 : List[float]
        Results from second algorithm (paired with results1).
    alternative : str, optional
        'two-sided', 'less', or 'greater'. Default is 'two-sided'.
    
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary with keys:
        - 'statistic': Test statistic
        - 'p_value': P-value
        - 'significant': Boolean, True if p < 0.05
        - 'interpretation': Text interpretation
    
    Examples
    --------
    >>> alg1_results = [0.01, 0.02, 0.015, 0.018, 0.012]
    >>> alg2_results = [0.05, 0.06, 0.055, 0.058, 0.052]
    >>> result = wilcoxon_test(alg1_results, alg2_results)
    >>> print(f"p-value: {result['p_value']:.4f}")
    >>> print(result['interpretation'])
    
    References
    ----------
    Wilcoxon, F. (1945). Individual comparisons by ranking methods.
    Biometrics Bulletin, 1(6), 80-83.
    """
    if len(results1) != len(results2):
        raise ValueError("Results must have same length (paired samples)")
    
    if len(results1) < 2:
        raise ValueError("Need at least 2 paired samples")
    
    # Perform test
    try:
        statistic, p_value = stats.wilcoxon(
            results1, results2,
            alternative=alternative,
            zero_method='wilcox'
        )
    except ValueError as e:
        # Handle case where all differences are zero
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'interpretation': 'No difference detected (all paired differences are zero)'
        }
    
    # Interpretation
    significant = p_value < 0.05
    if alternative == 'two-sided':
        interpretation = (
            f"{'Significant' if significant else 'No significant'} difference "
            f"between algorithms (p={p_value:.4f})"
        )
    elif alternative == 'less':
        interpretation = (
            f"Algorithm 1 is {'significantly' if significant else 'not significantly'} "
            f"better than Algorithm 2 (p={p_value:.4f})"
        )
    else:  # greater
        interpretation = (
            f"Algorithm 2 is {'significantly' if significant else 'not significantly'} "
            f"better than Algorithm 1 (p={p_value:.4f})"
        )
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': significant,
        'interpretation': interpretation
    }


def friedman_test(
    multiple_results: Dict[str, List[float]]
) -> Dict[str, Union[float, str, List]]:
    """
    Perform Friedman test for multiple related samples.
    
    Non-parametric test for comparing more than two related samples.
    Tests whether algorithms perform differently across problems.
    
    Parameters
    ----------
    multiple_results : Dict[str, List[float]]
        Dictionary mapping algorithm names to their results across problems.
        Each list should have the same length (one result per problem).
    
    Returns
    -------
    Dict[str, Union[float, str, List]]
        Dictionary with keys:
        - 'statistic': Chi-squared statistic
        - 'p_value': P-value
        - 'significant': Boolean, True if p < 0.05
        - 'mean_ranks': Mean rank for each algorithm
        - 'interpretation': Text interpretation
    
    Examples
    --------
    >>> results = {
    ...     'DE': [0.01, 0.05, 0.02, 0.03],
    ...     'PSO': [0.02, 0.06, 0.03, 0.04],
    ...     'GA': [0.03, 0.07, 0.04, 0.05]
    ... }
    >>> result = friedman_test(results)
    >>> print(f"p-value: {result['p_value']:.4f}")
    >>> print(result['mean_ranks'])
    
    References
    ----------
    Friedman, M. (1937). The use of ranks to avoid the assumption of
    normality implicit in the analysis of variance. Journal of the
    American Statistical Association, 32(200), 675-701.
    """
    if not multiple_results:
        raise ValueError("Need at least one algorithm")
    
    # Check all have same number of results
    n_problems = len(next(iter(multiple_results.values())))
    if not all(len(results) == n_problems for results in multiple_results.values()):
        raise ValueError("All algorithms must have same number of results")
    
    if n_problems < 2:
        raise ValueError("Need at least 2 problems")
    
    if len(multiple_results) < 3:
        warnings.warn("Friedman test typically used for 3+ algorithms. "
                     "Consider Wilcoxon test for 2 algorithms.")
    
    # Organize data: each row is a problem, each column is an algorithm
    algorithm_names = list(multiple_results.keys())
    data = np.array([multiple_results[alg] for alg in algorithm_names]).T
    
    # Perform test
    statistic, p_value = stats.friedmanchisquare(*[data[:, i] for i in range(data.shape[1])])
    
    # Calculate mean ranks
    ranks = np.zeros_like(data)
    for i in range(data.shape[0]):
        ranks[i, :] = stats.rankdata(data[i, :])
    mean_ranks = np.mean(ranks, axis=0)
    
    # Create rank dictionary
    rank_dict = {alg: float(rank) for alg, rank in zip(algorithm_names, mean_ranks)}
    
    # Interpretation
    significant = p_value < 0.05
    interpretation = (
        f"{'Significant' if significant else 'No significant'} difference "
        f"among algorithms (χ²={statistic:.4f}, p={p_value:.4f}). "
    )
    
    if significant:
        best_alg = algorithm_names[np.argmin(mean_ranks)]
        interpretation += f"Best performing: {best_alg} (mean rank={rank_dict[best_alg]:.2f})"
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': significant,
        'mean_ranks': rank_dict,
        'interpretation': interpretation
    }


def nemenyi_posthoc_test(
    multiple_results: Dict[str, List[float]],
    alpha: float = 0.05
) -> Dict[str, Union[np.ndarray, List[Tuple[str, str]]]]:
    """
    Perform Nemenyi post-hoc test after Friedman test.
    
    Pairwise comparisons between algorithms using critical difference.
    
    Parameters
    ----------
    multiple_results : Dict[str, List[float]]
        Dictionary mapping algorithm names to their results.
    alpha : float, optional
        Significance level. Default is 0.05.
    
    Returns
    -------
    Dict
        Dictionary with keys:
        - 'mean_ranks': Mean ranks for each algorithm
        - 'critical_difference': CD value for significance
        - 'significant_pairs': List of (alg1, alg2) tuples with significant differences
        - 'rank_matrix': Full pairwise rank difference matrix
    
    Examples
    --------
    >>> results = {
    ...     'DE': [0.01, 0.05, 0.02],
    ...     'PSO': [0.02, 0.06, 0.03],
    ...     'GA': [0.03, 0.07, 0.04]
    ... }
    >>> posthoc = nemenyi_posthoc_test(results)
    >>> print(f"Critical difference: {posthoc['critical_difference']:.3f}")
    >>> print(f"Significant pairs: {posthoc['significant_pairs']}")
    
    References
    ----------
    Demšar, J. (2006). Statistical comparisons of classifiers over multiple
    data sets. Journal of Machine Learning Research, 7, 1-30.
    """
    if not multiple_results:
        raise ValueError("Need at least one algorithm")
    
    # Get data dimensions
    algorithm_names = list(multiple_results.keys())
    k = len(algorithm_names)  # number of algorithms
    n = len(next(iter(multiple_results.values())))  # number of problems
    
    if k < 2:
        raise ValueError("Need at least 2 algorithms for post-hoc test")
    
    # Calculate ranks
    data = np.array([multiple_results[alg] for alg in algorithm_names]).T
    ranks = np.zeros_like(data)
    for i in range(data.shape[0]):
        ranks[i, :] = stats.rankdata(data[i, :])
    mean_ranks = np.mean(ranks, axis=0)
    
    # Critical difference (Nemenyi test)
    # q_alpha values from studentized range statistic table
    # For alpha=0.05, approximate q values:
    q_alpha_values = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164
    }
    
    if k in q_alpha_values:
        q_alpha = q_alpha_values[k]
    else:
        # Approximate for k > 10
        q_alpha = 3.164 + (k - 10) * 0.05
    
    critical_difference = q_alpha * np.sqrt((k * (k + 1)) / (6 * n))
    
    # Find significant pairs
    significant_pairs = []
    rank_matrix = np.zeros((k, k))
    
    for i in range(k):
        for j in range(i + 1, k):
            rank_diff = abs(mean_ranks[i] - mean_ranks[j])
            rank_matrix[i, j] = rank_diff
            rank_matrix[j, i] = rank_diff
            
            if rank_diff > critical_difference:
                significant_pairs.append((algorithm_names[i], algorithm_names[j]))
    
    return {
        'mean_ranks': {alg: float(rank) for alg, rank in zip(algorithm_names, mean_ranks)},
        'critical_difference': float(critical_difference),
        'significant_pairs': significant_pairs,
        'rank_matrix': rank_matrix
    }


def calculate_effect_size(
    results1: List[float],
    results2: List[float],
    method: str = 'cohen'
) -> Dict[str, Union[float, str]]:
    """
    Calculate effect size between two algorithms.
    
    Effect size quantifies the magnitude of difference between algorithms,
    independent of sample size.
    
    Parameters
    ----------
    results1 : List[float]
        Results from first algorithm.
    results2 : List[float]
        Results from second algorithm.
    method : str, optional
        Effect size method: 'cohen' (Cohen's d), 'hedges' (Hedges' g),
        or 'vargha' (Vargha-Delaney A). Default is 'cohen'.
    
    Returns
    -------
    Dict[str, Union[float, str]]
        Dictionary with keys:
        - 'effect_size': Numerical effect size value
        - 'magnitude': Interpretation (small/medium/large)
        - 'method': Method used
    
    Examples
    --------
    >>> alg1 = [0.01, 0.02, 0.015, 0.018]
    >>> alg2 = [0.05, 0.06, 0.055, 0.058]
    >>> es = calculate_effect_size(alg1, alg2, method='cohen')
    >>> print(f"Cohen's d: {es['effect_size']:.3f} ({es['magnitude']})")
    
    References
    ----------
    - Cohen, J. (1988). Statistical power analysis for the behavioral sciences.
    - Vargha, A., & Delaney, H. D. (2000). A critique and improvement of the
      CL common language effect size statistic. Journal of Educational and
      Behavioral Statistics, 25(2), 101-132.
    """
    arr1 = np.array(results1)
    arr2 = np.array(results2)
    
    if method == 'cohen':
        # Cohen's d
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
        n1, n2 = len(arr1), len(arr2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            effect_size = 0.0
        else:
            effect_size = (mean1 - mean2) / pooled_std
        
        # Interpretation (absolute value)
        abs_d = abs(effect_size)
        if abs_d < 0.2:
            magnitude = 'negligible'
        elif abs_d < 0.5:
            magnitude = 'small'
        elif abs_d < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
    
    elif method == 'hedges':
        # Hedges' g (bias-corrected Cohen's d)
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        std1, std2 = np.std(arr1, ddof=1), np.std(arr2, ddof=1)
        n1, n2 = len(arr1), len(arr2)
        
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            cohens_d = 0.0
        else:
            cohens_d = (mean1 - mean2) / pooled_std
        
        # Correction factor
        df = n1 + n2 - 2
        correction = 1 - (3 / (4 * df - 1))
        effect_size = cohens_d * correction
        
        abs_g = abs(effect_size)
        if abs_g < 0.2:
            magnitude = 'negligible'
        elif abs_g < 0.5:
            magnitude = 'small'
        elif abs_g < 0.8:
            magnitude = 'medium'
        else:
            magnitude = 'large'
    
    elif method == 'vargha':
        # Vargha-Delaney A statistic
        n1, n2 = len(arr1), len(arr2)
        
        # Count how many times arr1[i] < arr2[j]
        count = 0
        for x1 in arr1:
            for x2 in arr2:
                if x1 < x2:
                    count += 1
                elif x1 == x2:
                    count += 0.5
        
        effect_size = count / (n1 * n2)
        
        # Interpretation (A=0.5 means no difference)
        if 0.44 <= effect_size <= 0.56:
            magnitude = 'negligible'
        elif effect_size < 0.44 or effect_size > 0.56:
            if effect_size < 0.38 or effect_size > 0.62:
                magnitude = 'large'
            else:
                magnitude = 'medium'
        else:
            magnitude = 'small'
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cohen', 'hedges', or 'vargha'")
    
    return {
        'effect_size': float(effect_size),
        'magnitude': magnitude,
        'method': method
    }


def mann_whitney_u_test(
    results1: List[float],
    results2: List[float],
    alternative: str = 'two-sided'
) -> Dict[str, Union[float, str]]:
    """
    Perform Mann-Whitney U test for independent samples.
    
    Non-parametric test for comparing two independent samples.
    
    Parameters
    ----------
    results1 : List[float]
        Results from first algorithm.
    results2 : List[float]
        Results from second algorithm.
    alternative : str, optional
        'two-sided', 'less', or 'greater'. Default is 'two-sided'.
    
    Returns
    -------
    Dict[str, Union[float, str]]
        Test results including statistic, p-value, and interpretation.
    
    Examples
    --------
    >>> alg1 = [0.01, 0.02, 0.015]
    >>> alg2 = [0.05, 0.06, 0.055]
    >>> result = mann_whitney_u_test(alg1, alg2)
    >>> print(result['interpretation'])
    """
    statistic, p_value = stats.mannwhitneyu(
        results1, results2,
        alternative=alternative
    )
    
    significant = p_value < 0.05
    interpretation = (
        f"{'Significant' if significant else 'No significant'} difference "
        f"(U={statistic:.2f}, p={p_value:.4f})"
    )
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': significant,
        'interpretation': interpretation
    }


class StatisticalTests:
    """
    Statistical tests for comparing optimization algorithms.
    
    Provides convenient interface for hypothesis testing and effect size
    calculation when comparing algorithm performance.
    
    Examples
    --------
    >>> tests = StatisticalTests()
    >>> alg1 = [0.01, 0.02, 0.015, 0.018, 0.012]
    >>> alg2 = [0.05, 0.06, 0.055, 0.058, 0.052]
    >>> result = tests.wilcoxon(alg1, alg2)
    >>> print(result['interpretation'])
    """
    
    @staticmethod
    def wilcoxon(
        results1: List[float],
        results2: List[float],
        alternative: str = 'two-sided'
    ) -> Dict[str, Union[float, str]]:
        """Perform Wilcoxon signed-rank test."""
        return wilcoxon_test(results1, results2, alternative)
    
    @staticmethod
    def friedman(
        multiple_results: Dict[str, List[float]]
    ) -> Dict[str, Union[float, str, List]]:
        """Perform Friedman test."""
        return friedman_test(multiple_results)
    
    @staticmethod
    def nemenyi(
        multiple_results: Dict[str, List[float]],
        alpha: float = 0.05
    ) -> Dict:
        """Perform Nemenyi post-hoc test."""
        return nemenyi_posthoc_test(multiple_results, alpha)
    
    @staticmethod
    def effect_size(
        results1: List[float],
        results2: List[float],
        method: str = 'cohen'
    ) -> Dict[str, Union[float, str]]:
        """Calculate effect size."""
        return calculate_effect_size(results1, results2, method)
    
    @staticmethod
    def mann_whitney(
        results1: List[float],
        results2: List[float],
        alternative: str = 'two-sided'
    ) -> Dict[str, Union[float, str]]:
        """Perform Mann-Whitney U test."""
        return mann_whitney_u_test(results1, results2, alternative)
