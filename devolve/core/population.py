"""
Population class for managing a collection of individuals.
"""

from typing import List, Optional, Callable
import numpy as np
from .individual import Individual


class Population:
    """
    Manages a collection of individuals in the optimization process.
    
    The population maintains a list of individuals and provides methods
    for common operations like adding, removing, sorting, and statistical analysis.
    
    Parameters
    ----------
    individuals : List[Individual], optional
        Initial list of individuals. Default is empty list.
    
    Attributes
    ----------
    individuals : List[Individual]
        The list of individuals in the population.
    
    Examples
    --------
    >>> import numpy as np
    >>> from devolve.core import Individual, Population
    >>> pop = Population()
    >>> pop.add(Individual(position=np.array([1.0, 2.0]), fitness=10.0))
    >>> print(pop.size)
    1
    >>> best = pop.get_best()
    >>> print(best.fitness)
    10.0
    """
    
    def __init__(self, individuals: Optional[List[Individual]] = None):
        """Initialize the population."""
        self.individuals: List[Individual] = individuals if individuals else []
    
    @property
    def size(self) -> int:
        """
        Get the current population size.
        
        Returns
        -------
        int
            Number of individuals in the population.
        """
        return len(self.individuals)
    
    @property
    def dimensions(self) -> int:
        """
        Get the dimensionality of the problem.
        
        Returns
        -------
        int
            Number of dimensions. Returns 0 if population is empty.
        """
        return self.individuals[0].dimensions if self.size > 0 else 0
    
    def add(self, individual: Individual) -> None:
        """
        Add an individual to the population.
        
        Parameters
        ----------
        individual : Individual
            The individual to add.
        """
        self.individuals.append(individual)
    
    def remove(self, index: int) -> Individual:
        """
        Remove and return an individual at the specified index.
        
        Parameters
        ----------
        index : int
            The index of the individual to remove.
        
        Returns
        -------
        Individual
            The removed individual.
        """
        return self.individuals.pop(index)
    
    def get_best(self, use_feasibility_rules: bool = True) -> Individual:
        """
        Get the best individual in the population.
        
        Parameters
        ----------
        use_feasibility_rules : bool, optional
            If True, use Deb's feasibility rules for comparison.
            Default is True.
        
        Returns
        -------
        Individual
            The best individual.
        
        Raises
        ------
        ValueError
            If the population is empty.
        """
        if self.size == 0:
            raise ValueError("Cannot get best individual from empty population")
        
        best = self.individuals[0]
        for ind in self.individuals[1:]:
            if ind.is_better_than(best, use_feasibility_rules):
                best = ind
        return best
    
    def get_worst(self, use_feasibility_rules: bool = True) -> Individual:
        """
        Get the worst individual in the population.
        
        Parameters
        ----------
        use_feasibility_rules : bool, optional
            If True, use Deb's feasibility rules for comparison.
            Default is True.
        
        Returns
        -------
        Individual
            The worst individual.
        
        Raises
        ------
        ValueError
            If the population is empty.
        """
        if self.size == 0:
            raise ValueError("Cannot get worst individual from empty population")
        
        worst = self.individuals[0]
        for ind in self.individuals[1:]:
            if worst.is_better_than(ind, use_feasibility_rules):
                worst = ind
        return worst
    
    def get_random(self, n: int = 1, exclude: Optional[List[int]] = None) -> List[Individual]:
        """
        Get random individuals from the population.
        
        Parameters
        ----------
        n : int, optional
            Number of random individuals to select. Default is 1.
        exclude : List[int], optional
            Indices to exclude from selection. Default is None.
        
        Returns
        -------
        List[Individual]
            List of randomly selected individuals.
        
        Raises
        ------
        ValueError
            If n is larger than available individuals.
        """
        if exclude is None:
            exclude = []
        
        available_indices = [i for i in range(self.size) if i not in exclude]
        
        if n > len(available_indices):
            raise ValueError(f"Cannot select {n} individuals from {len(available_indices)} available")
        
        selected_indices = np.random.choice(available_indices, size=n, replace=False)
        return [self.individuals[i] for i in selected_indices]
    
    def get_top_p_percent(self, p: float) -> List[Individual]:
        """
        Get the top p% individuals based on fitness.
        
        Parameters
        ----------
        p : float
            Percentage of top individuals (0 < p <= 1).
        
        Returns
        -------
        List[Individual]
            List of top individuals.
        """
        n_top = max(1, int(np.ceil(p * self.size)))
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness)
        return sorted_pop[:n_top]
    
    def sort(self, reverse: bool = False, use_feasibility_rules: bool = True) -> None:
        """
        Sort the population by fitness.
        
        Parameters
        ----------
        reverse : bool, optional
            If True, sort in descending order. Default is False (ascending).
        use_feasibility_rules : bool, optional
            If True, use Deb's feasibility rules for comparison.
            Default is True.
        """
        if use_feasibility_rules:
            self.individuals.sort(key=lambda x: (
                not x.is_feasible,  # Feasible first
                x.constraint_violation if not x.is_feasible else x.fitness
            ), reverse=reverse)
        else:
            self.individuals.sort(key=lambda x: x.fitness, reverse=reverse)
    
    def get_positions(self) -> np.ndarray:
        """
        Get all position vectors as a 2D array.
        
        Returns
        -------
        np.ndarray
            Array of shape (population_size, dimensions).
        """
        return np.array([ind.position for ind in self.individuals])
    
    def get_fitness_values(self) -> np.ndarray:
        """
        Get all fitness values as a 1D array.
        
        Returns
        -------
        np.ndarray
            Array of fitness values.
        """
        return np.array([ind.fitness for ind in self.individuals])
    
    def get_mean_position(self) -> np.ndarray:
        """
        Calculate the mean position of all individuals.
        
        Returns
        -------
        np.ndarray
            Mean position vector.
        """
        return np.mean(self.get_positions(), axis=0)
    
    def get_std_position(self) -> np.ndarray:
        """
        Calculate the standard deviation of positions.
        
        Returns
        -------
        np.ndarray
            Standard deviation per dimension.
        """
        return np.std(self.get_positions(), axis=0)
    
    def get_diversity(self) -> float:
        """
        Calculate population diversity as average distance to centroid.
        
        Returns
        -------
        float
            Average Euclidean distance to population centroid.
        
        Notes
        -----
        Diversity = (1/NP) * Σ ||x_i - x_mean||
        where NP is population size, x_i are individuals, x_mean is centroid.
        """
        if self.size == 0:
            return 0.0
        
        centroid = self.get_mean_position()
        distances = [np.linalg.norm(ind.position - centroid) 
                    for ind in self.individuals]
        return np.mean(distances)
    
    def get_feasible_count(self) -> int:
        """
        Count the number of feasible individuals.
        
        Returns
        -------
        int
            Number of feasible individuals.
        """
        return sum(1 for ind in self.individuals if ind.is_feasible)
    
    def get_feasibility_ratio(self) -> float:
        """
        Calculate the ratio of feasible individuals.
        
        Returns
        -------
        float
            Ratio of feasible individuals (0.0 to 1.0).
        """
        if self.size == 0:
            return 0.0
        return self.get_feasible_count() / self.size
    
    def resize(self, new_size: int) -> None:
        """
        Resize the population by removing worst individuals.
        
        Parameters
        ----------
        new_size : int
            Target population size.
        """
        if new_size < self.size:
            self.sort()  # Sort by fitness
            self.individuals = self.individuals[:new_size]
        elif new_size > self.size:
            raise ValueError("Cannot increase population size with resize()")
    
    def clear(self) -> None:
        """Remove all individuals from the population."""
        self.individuals.clear()
    
    def copy(self) -> 'Population':
        """
        Create a deep copy of the population.
        
        Returns
        -------
        Population
            A new Population instance with copied individuals.
        """
        return Population([ind.copy() for ind in self.individuals])
    
    def __len__(self) -> int:
        """Return the population size."""
        return self.size
    
    def __getitem__(self, index: int) -> Individual:
        """Get individual by index."""
        return self.individuals[index]
    
    def __setitem__(self, index: int, individual: Individual) -> None:
        """Set individual at index."""
        self.individuals[index] = individual
    
    def __iter__(self):
        """Iterate over individuals."""
        return iter(self.individuals)
    
    def __repr__(self) -> str:
        """String representation of the population."""
        if self.size == 0:
            return "Population(size=0)"
        best = self.get_best()
        feasible = self.get_feasible_count()
        return (f"Population(size={self.size}, "
                f"dimensions={self.dimensions}, "
                f"best_fitness={best.fitness:.6f}, "
                f"feasible={feasible})")
