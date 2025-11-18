"""
Test script for all benchmark problems.

Tests the three categories of benchmarks:
1. Classic functions (Sphere, Rosenbrock, etc.)
2. CEC2017 suite (F1-F30)
3. Engineering problems (constrained optimization)
"""

import numpy as np
from devolve.benchmarks import (
    # Classic
    Sphere, Rosenbrock, Rastrigin, Ackley,
    # CEC2017
    get_cec2017_function,
    # Engineering
    PressureVesselDesign, WeldedBeamDesign,
    TensionCompressionSpring, SpeedReducerDesign
)


def test_classic_benchmarks():
    """Test classic benchmark functions."""
    print("=" * 70)
    print("Testing Classic Benchmarks")
    print("=" * 70)
    
    problems = [
        Sphere(dimensions=10),
        Rosenbrock(dimensions=10),
        Rastrigin(dimensions=10),
        Ackley(dimensions=10),
    ]
    
    for problem in problems:
        # Test at optimum
        x_opt = problem.optimum_position
        f_opt = problem.evaluate(x_opt)
        
        # Test at random point
        np.random.seed(42)
        x_rand = np.random.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        f_rand = problem.evaluate(x_rand)
        
        print(f"\n{problem.name} ({problem.dimensions}D):")
        print(f"  Bounds: [{problem.bounds[0, 0]:.1f}, {problem.bounds[0, 1]:.1f}]")
        print(f"  Known optimum: {problem.optimum:.6e}")
        print(f"  f(x_opt) = {f_opt:.6e}")
        print(f"  f(x_rand) = {f_rand:.6e}")
        
        assert np.abs(f_opt - problem.optimum) < 1e-5, f"Optimum mismatch for {problem.name}"
        print(f"  ✓ Optimum verification passed")


def test_cec2017_suite():
    """Test CEC2017 functions."""
    print("\n" + "=" * 70)
    print("Testing CEC2017 Suite (sample functions)")
    print("=" * 70)
    
    # Test representative functions from each category
    test_functions = [1, 4, 11, 21]  # Unimodal, Multimodal, Hybrid, Composition
    dimensions = 10
    
    for func_num in test_functions:
        problem = get_cec2017_function(func_num, dimensions)
        
        # Test at shifted optimum (should be close to func_num * 100)
        x_opt = problem.optimum_position
        f_opt = problem.evaluate(x_opt)
        
        # Test at random point
        np.random.seed(42 + func_num)
        x_rand = np.random.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        f_rand = problem.evaluate(x_rand)
        
        print(f"\n{problem.name}:")
        print(f"  Dimensions: {problem.dimensions}")
        print(f"  Target optimum: {problem.optimum:.1f}")
        print(f"  f(x_opt) = {f_opt:.6e}")
        print(f"  f(x_rand) = {f_rand:.6e}")
        print(f"  Error at optimum: {abs(f_opt - problem.optimum):.6e}")


def test_engineering_problems():
    """Test engineering design problems with constraints."""
    print("\n" + "=" * 70)
    print("Testing Engineering Problems (Constrained)")
    print("=" * 70)
    
    problems = [
        PressureVesselDesign(),
        WeldedBeamDesign(),
        TensionCompressionSpring(),
        SpeedReducerDesign(),
    ]
    
    for problem in problems:
        # Test at known optimum
        x_opt = problem.optimum_position
        f_opt = problem.evaluate(x_opt)
        
        # Check constraints
        if problem.constraint_function is not None:
            constraints = problem.constraint_function(x_opt)
            n_violated = np.sum(constraints > 1e-6)  # Allow small tolerance
            max_violation = np.max(constraints)
        else:
            n_violated = 0
            max_violation = 0.0
        
        # Test at random feasible point
        np.random.seed(42)
        x_rand = np.random.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        f_rand = problem.evaluate(x_rand)
        
        if problem.constraint_function is not None:
            constraints_rand = problem.constraint_function(x_rand)
            n_violated_rand = np.sum(constraints_rand > 0)
        else:
            n_violated_rand = 0
        
        print(f"\n{problem.name} ({problem.dimensions}D):")
        print(f"  Known optimum: {problem.optimum:.6f}")
        print(f"  f(x_opt) = {f_opt:.6f}")
        print(f"  Constraints violated at x_opt: {n_violated}")
        if n_violated > 0:
            print(f"  Max violation: {max_violation:.6e}")
        print(f"  f(x_rand) = {f_rand:.6f}")
        print(f"  Constraints violated at x_rand: {n_violated_rand}")
        
        # Known optima should be feasible
        if n_violated == 0:
            print(f"  ✓ Known optimum is feasible")
        else:
            print(f"  ⚠ Known optimum has constraint violations (may need adjustment)")


def test_problem_interfaces():
    """Test that all problems follow the common interface."""
    print("\n" + "=" * 70)
    print("Testing Problem Interface Compliance")
    print("=" * 70)
    
    problems = [
        Sphere(dimensions=5),
        get_cec2017_function(1, 10),
        PressureVesselDesign(),
    ]
    
    for problem in problems:
        print(f"\n{problem.name}:")
        
        # Check required attributes
        assert hasattr(problem, 'dimensions'), "Missing 'dimensions'"
        assert hasattr(problem, 'bounds'), "Missing 'bounds'"
        assert hasattr(problem, 'evaluate'), "Missing 'evaluate'"
        assert hasattr(problem, 'name'), "Missing 'name'"
        print(f"  ✓ Has required attributes")
        
        # Check bounds shape
        assert problem.bounds.shape == (problem.dimensions, 2), "Invalid bounds shape"
        print(f"  ✓ Bounds shape correct: {problem.bounds.shape}")
        
        # Check evaluation works
        x_test = np.random.uniform(problem.bounds[:, 0], problem.bounds[:, 1])
        f_test = problem.evaluate(x_test)
        assert isinstance(f_test, (int, float, np.number)), "Evaluate should return scalar"
        print(f"  ✓ Evaluation returns scalar: {f_test:.6e}")
        
        # Check optional attributes
        if hasattr(problem, 'optimum'):
            print(f"  ✓ Has known optimum: {problem.optimum}")
        if hasattr(problem, 'optimum_position'):
            print(f"  ✓ Has optimum position")
        if hasattr(problem, 'constraint_function'):
            if problem.constraint_function is not None:
                c_test = problem.constraint_function(x_test)
                print(f"  ✓ Has {len(c_test)} constraints")


def main():
    """Run all benchmark tests."""
    print("\n" + "=" * 70)
    print("DEvolve Benchmark Test Suite")
    print("=" * 70)
    
    try:
        # Test each category
        test_classic_benchmarks()
        test_cec2017_suite()
        test_engineering_problems()
        test_problem_interfaces()
        
        print("\n" + "=" * 70)
        print("✓ All benchmark tests completed successfully!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
