"""
DEvolve: A Comprehensive Differential Evolution Library
========================================================

A production-ready Python package implementing all major Differential Evolution
variants for optimization problems.

Key Features:
- Classic DE variants (DE/rand/1, DE/best/1, etc.)
- Adaptive variants (jDE, JADE, SHADE, L-SHADE, LSHADE-EpSin)
- Hybrid algorithms (DE-PSO, DEGL, CoDE)
- Multi-objective optimization (MODE, GDE3, NSDE)
- Comprehensive benchmarking suite
- Constraint handling methods
- Machine learning integration
- Parallel execution support
- Rich visualization capabilities

Author: DEvolve Development Team
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DEvolve Development Team"
__license__ = "MIT"

# Core classes
from .core.base import BaseDifferentialEvolution
from .core.individual import Individual
from .core.population import Population
from .core.problem import Problem
from .core.boundary import BoundaryHandler
from .core.logger import OptimizationLogger

# Classic DE variants
from .algorithms.classic import (
    DErand1,
    DEbest1,
    DEcurrentToBest1,
    DErand2,
    DEbest2
)

# Adaptive DE variants
from .algorithms.adaptive import (
    JDE,
    SaDE,
    JADE,
    SHADE,
    LSHADE,
    LSHADEEpSin,
    LSHADEcnEpSin
)

# Hybrid variants
from .algorithms.hybrid import (
    DEPSO,
    DEGL,
    CoDE
)

# Multi-objective variants
from .algorithms.multiobjective import (
    MODE,
    GDE3,
    NSDE
)

# Operators
from .operators import mutation, crossover, selection

# Benchmark problems
from .benchmarks.classic import (
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
    Michalewicz,
    Zakharov
)

from .benchmarks.cec2017 import (
    CEC2017Function,
    get_cec2017_function
)

from .benchmarks.engineering import (
    PressureVesselDesign,
    WeldedBeamDesign,
    TensionCompressionSpring,
    SpeedReducerDesign,
    get_engineering_problem
)

# Constraint handling
from .constraints import (
    PenaltyMethod,
    FeasibilityRules,
    EpsilonConstraint,
    StochasticRanking
)

# Utilities
from .utils.metrics import PerformanceMetrics
from .utils.stats import StatisticalTests

# Visualization utilities (optional, import as needed)
# from .utils import visualization

# ML Integration
from .ml import SklearnOptimizer, DEFeatureSelector

__all__ = [
    # Core
    'BaseDifferentialEvolution',
    'Individual',
    'Population',
    'Problem',
    'BoundaryHandler',
    'OptimizationLogger',
    
    # Classic algorithms
    'DErand1',
    'DEbest1',
    'DEcurrentToBest1',
    'DErand2',
    'DEbest2',
    
    # Adaptive algorithms
    'JDE',
    'SaDE',
    'JADE',
    'SHADE',
    'LSHADE',
    'LSHADEEpSin',
    'LSHADEcnEpSin',
    
    # Hybrid algorithms
    'DEPSO',
    'DEGL',
    'CoDE',
    
    # Multi-objective
    'MODE',
    'GDE3',
    'NSDE',
    
    # Operators
    'mutation',
    'crossover',
    'selection',
    
    # Benchmarks - Classic
    'Sphere',
    'Rosenbrock',
    'Rastrigin',
    'Ackley',
    'Griewank',
    'Schwefel',
    'Michalewicz',
    'Zakharov',
    
    # Benchmarks - CEC2017
    'CEC2017Function',
    'get_cec2017_function',
    
    # Benchmarks - Engineering
    'PressureVesselDesign',
    'WeldedBeamDesign',
    'TensionCompressionSpring',
    'SpeedReducerDesign',
    'get_engineering_problem',
    
    # Constraints
    'PenaltyMethod',
    'FeasibilityRules',
    'EpsilonConstraint',
    'StochasticRanking',
    
    # Utils
    'PerformanceMetrics',
    'StatisticalTests',
    
    # ML
    'SklearnOptimizer',
    'DEFeatureSelector',
]
