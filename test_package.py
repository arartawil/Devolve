"""
Quick Test Script
==================

Verify that the DEvolve package works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from devolve.algorithms.classic import DErand1, DEbest1
from devolve.benchmarks.classic import Sphere, Rastrigin


def test_sphere():
    """Test on Sphere function."""
    print("Testing DE/rand/1 on Sphere function...")
    
    problem = Sphere(dimensions=10)
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=200,
        F=0.8,
        CR=0.9,
        random_seed=42
    )
    
    optimizer.logger.verbose = 0
    best_x, best_f = optimizer.optimize()
    
    assert best_f < 1.0, f"Expected fitness < 1.0, got {best_f}"
    print(f"  ✓ Success! Best fitness: {best_f:.6e}")
    return True


def test_rastrigin():
    """Test on Rastrigin function."""
    print("Testing DE/best/1 on Rastrigin function...")
    
    problem = Rastrigin(dimensions=5)
    optimizer = DEbest1(
        problem=problem,
        population_size=50,
        max_iterations=200,
        F=0.8,
        CR=0.9,
        random_seed=42
    )
    
    optimizer.logger.verbose = 0
    best_x, best_f = optimizer.optimize()
    
    assert best_f < 50, f"Expected fitness < 50, got {best_f}"
    print(f"  ✓ Success! Best fitness: {best_f:.6e}")
    return True


def test_core_components():
    """Test core components."""
    print("Testing core components...")
    
    from devolve.core.individual import Individual
    from devolve.core.population import Population
    from devolve.core.problem import Problem
    from devolve.core.boundary import BoundaryHandler
    
    # Test Individual
    ind = Individual(position=np.array([1.0, 2.0, 3.0]), fitness=10.0)
    assert ind.dimensions == 3
    assert ind.fitness == 10.0
    print("  ✓ Individual class works")
    
    # Test Population
    pop = Population()
    pop.add(ind)
    assert pop.size == 1
    print("  ✓ Population class works")
    
    # Test Problem
    def simple_func(x):
        return np.sum(x**2)
    
    problem = Problem(
        objective_function=simple_func,
        bounds=(-5, 5),
        dimensions=3
    )
    assert problem.dimensions == 3
    assert problem.evaluate(np.zeros(3)) == 0.0
    print("  ✓ Problem class works")
    
    # Test BoundaryHandler
    handler = BoundaryHandler(strategy="clip", bounds=np.array([[-5, 5]] * 3))
    x_clipped = handler.handle(np.array([10, -10, 3]))
    assert np.allclose(x_clipped, [5, -5, 3])
    print("  ✓ BoundaryHandler class works")
    
    return True


def test_operators():
    """Test operators."""
    print("Testing operators...")
    
    from devolve.operators.mutation import rand_1
    from devolve.operators.crossover import binomial_crossover
    from devolve.core.population import Population
    from devolve.core.individual import Individual
    
    # Create test population
    pop = Population([Individual(np.random.randn(5)) for _ in range(10)])
    rng = np.random.default_rng(42)
    
    # Test mutation
    mutant = rand_1(pop, target_idx=0, F=0.8, rng=rng)
    assert mutant.shape == (5,)
    print("  ✓ Mutation operators work")
    
    # Test crossover
    target = np.random.randn(5)
    trial = binomial_crossover(target, mutant, CR=0.9, rng=rng)
    assert trial.shape == (5,)
    print("  ✓ Crossover operators work")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("DEvolve Package Test Suite")
    print("="*70)
    print()
    
    tests = [
        ("Core Components", test_core_components),
        ("Operators", test_operators),
        ("Sphere Function", test_sphere),
        ("Rastrigin Function", test_rastrigin),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
    
    print("="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("🎉 All tests passed! Package is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
