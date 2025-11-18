"""Benchmark problems for testing DE algorithms."""

from .classic import (
    Sphere,
    Rosenbrock,
    Rastrigin,
    Ackley,
    Griewank,
    Schwefel,
    Michalewicz,
    Zakharov,
    get_benchmark,
    BENCHMARK_FUNCTIONS
)

from .cec2017 import (
    CEC2017Function,
    get_cec2017_function,
    CEC2017_FUNCTIONS,
    # F1-F3: Unimodal
    CEC2017_F1, CEC2017_F2, CEC2017_F3,
    # F4-F10: Simple Multimodal
    CEC2017_F4, CEC2017_F5, CEC2017_F6, CEC2017_F7, CEC2017_F8, CEC2017_F9, CEC2017_F10,
    # F11-F20: Hybrid
    CEC2017_F11, CEC2017_F12, CEC2017_F13, CEC2017_F14, CEC2017_F15,
    CEC2017_F16, CEC2017_F17, CEC2017_F18, CEC2017_F19, CEC2017_F20,
    # F21-F30: Composition
    CEC2017_F21, CEC2017_F22, CEC2017_F23, CEC2017_F24, CEC2017_F25,
    CEC2017_F26, CEC2017_F27, CEC2017_F28, CEC2017_F29, CEC2017_F30,
)

from .engineering import (
    PressureVesselDesign,
    WeldedBeamDesign,
    TensionCompressionSpring,
    SpeedReducerDesign,
    get_engineering_problem,
    ENGINEERING_PROBLEMS
)

__all__ = [
    # Classic benchmarks
    'Sphere',
    'Rosenbrock',
    'Rastrigin',
    'Ackley',
    'Griewank',
    'Schwefel',
    'Michalewicz',
    'Zakharov',
    'get_benchmark',
    'BENCHMARK_FUNCTIONS',
    
    # CEC2017 suite
    'CEC2017Function',
    'get_cec2017_function',
    'CEC2017_FUNCTIONS',
    'CEC2017_F1', 'CEC2017_F2', 'CEC2017_F3',
    'CEC2017_F4', 'CEC2017_F5', 'CEC2017_F6', 'CEC2017_F7', 'CEC2017_F8', 'CEC2017_F9', 'CEC2017_F10',
    'CEC2017_F11', 'CEC2017_F12', 'CEC2017_F13', 'CEC2017_F14', 'CEC2017_F15',
    'CEC2017_F16', 'CEC2017_F17', 'CEC2017_F18', 'CEC2017_F19', 'CEC2017_F20',
    'CEC2017_F21', 'CEC2017_F22', 'CEC2017_F23', 'CEC2017_F24', 'CEC2017_F25',
    'CEC2017_F26', 'CEC2017_F27', 'CEC2017_F28', 'CEC2017_F29', 'CEC2017_F30',
    
    # Engineering problems
    'PressureVesselDesign',
    'WeldedBeamDesign',
    'TensionCompressionSpring',
    'SpeedReducerDesign',
    'get_engineering_problem',
    'ENGINEERING_PROBLEMS',
]
