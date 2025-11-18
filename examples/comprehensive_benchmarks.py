"""
Comprehensive Benchmark Suite Example

This example demonstrates all three categories of benchmarks:
1. Classic optimization functions
2. CEC2017 competition suite
3. Engineering design problems

Shows how to use them with adaptive DE algorithms.
"""

import numpy as np
from devolve import (
    # Adaptive algorithms
    JDE, JADE, SHADE, LSHADE, LSHADEEpSin,
    # Classic benchmarks
    Sphere, Rosenbrock, Rastrigin,
    # CEC2017 benchmarks
    get_cec2017_function,
    # Engineering problems
    PressureVesselDesign, WeldedBeamDesign,
    TensionCompressionSpring, SpeedReducerDesign
)


def test_classic_benchmarks():
    """Test adaptive algorithms on classic benchmarks."""
    print("=" * 70)
    print("CLASSIC BENCHMARKS")
    print("=" * 70)
    
    problems = [
        Sphere(dimensions=10),
        Rosenbrock(dimensions=10),
        Rastrigin(dimensions=10),
    ]
    
    for problem in problems:
        print(f"\n{problem.name} (D={problem.dimensions}):")
        print(f"  Bounds: [{problem.bounds[0, 0]:.1f}, {problem.bounds[0, 1]:.1f}]")
        print(f"  Known optimum: {problem.optimum:.6e}")
        
        # Test with JADE (best performer from previous tests)
        optimizer = JADE(
            problem=problem,
            population_size=50,
            max_iterations=100,
            random_seed=42
        )
        
        best_x, best_f = optimizer.optimize()
        
        print(f"  JADE result: {best_f:.6e} ({optimizer.function_evaluations} FEs)")
        print(f"  Error: {abs(best_f - problem.optimum):.6e}")


def test_cec2017_suite():
    """Test adaptive algorithms on CEC2017 functions."""
    print("\n" + "=" * 70)
    print("CEC2017 BENCHMARK SUITE")
    print("=" * 70)
    
    # Test representative functions from each category
    test_functions = {
        1: "Unimodal (Shifted Bent Cigar)",
        4: "Simple Multimodal (Rosenbrock)",
        11: "Hybrid (Zakharov+Rosenbrock+Rastrigin)",
        21: "Composition (F1-F3 composition)"
    }
    
    dimensions = 10
    
    for func_num, description in test_functions.items():
        problem = get_cec2017_function(func_num, dimensions)
        
        print(f"\nF{func_num}: {description}")
        print(f"  Target: {problem.optimum:.1f}")
        
        # Use L-SHADE for better performance on complex functions
        optimizer = LSHADE(
            problem=problem,
            initial_population_size=100,
            max_iterations=200,
            random_seed=42
        )
        
        best_x, best_f = optimizer.optimize()
        error = best_f - problem.optimum
        
        print(f"  L-SHADE: {best_f:.6e} ({optimizer.function_evaluations} FEs)")
        print(f"  Error from target: {error:.6e}")


def test_engineering_problems():
    """Test adaptive algorithms on constrained engineering problems."""
    print("\n" + "=" * 70)
    print("ENGINEERING DESIGN PROBLEMS (Constrained)")
    print("=" * 70)
    
    problems = [
        PressureVesselDesign(),
        WeldedBeamDesign(),
        TensionCompressionSpring(),
        SpeedReducerDesign(),
    ]
    
    for problem in problems:
        print(f"\n{problem.name} (D={problem.dimensions}):")
        print(f"  Known optimum: {problem.optimum:.6f}")
        print(f"  Constraints: {len(problem.constraints)}")
        
        # Use LSHADE-EpSin which has epsilon constraint handling
        optimizer = LSHADEEpSin(
            problem=problem,
            initial_population_size=100,
            max_iterations=300,
            random_seed=42
        )
        
        best_x, best_f = optimizer.optimize()
        
        # Check constraint satisfaction
        if problem.constraints:
            violations = problem.constraint_function(best_x)
            n_violated = np.sum(violations > 1e-6)
            max_violation = np.max(violations) if len(violations) > 0 else 0
            feasible = n_violated == 0
        else:
            feasible = True
            n_violated = 0
            max_violation = 0
        
        print(f"  LSHADE-EpSin: {best_f:.6f} ({optimizer.function_evaluations} FEs)")
        print(f"  Feasible: {feasible} ({n_violated}/{len(problem.constraints)} violated)")
        if not feasible:
            print(f"  Max violation: {max_violation:.6e}")
        print(f"  Gap from optimum: {abs(best_f - problem.optimum):.6f}")


def compare_algorithms_on_problem():
    """Compare all adaptive algorithms on a single problem."""
    print("\n" + "=" * 70)
    print("ALGORITHM COMPARISON (Rastrigin 10D)")
    print("=" * 70)
    
    problem = Rastrigin(dimensions=10)
    algorithms = [
        ("jDE", JDE),
        ("JADE", JADE),
        ("SHADE", SHADE),
        ("L-SHADE", LSHADE),
        ("LSHADE-EpSin", LSHADEEpSin),
    ]
    
    results = []
    
    for name, AlgorithmClass in algorithms:
        # Configure based on algorithm type
        if name in ["L-SHADE", "LSHADE-EpSin"]:
            optimizer = AlgorithmClass(
                problem=problem,
                initial_population_size=50,
                max_iterations=100,
                random_seed=42
            )
        else:
            optimizer = AlgorithmClass(
                problem=problem,
                population_size=50,
                max_iterations=100,
                random_seed=42
            )
        
        best_x, best_f = optimizer.optimize()
        results.append((name, best_f, optimizer.function_evaluations))
    
    # Sort by fitness
    results.sort(key=lambda x: x[1])
    
    print("\nRanking:")
    for rank, (name, fitness, fes) in enumerate(results, 1):
        print(f"  {rank}. {name:15s} f={fitness:.6e}  ({fes} FEs)")


def main():
    """Run all benchmark examples."""
    print("\n" + "=" * 70)
    print("DEvolve Comprehensive Benchmark Examples")
    print("=" * 70)
    print("\nDemonstrating three benchmark categories:")
    print("  1. Classic functions (Sphere, Rosenbrock, Rastrigin, etc.)")
    print("  2. CEC2017 suite (30 competition functions)")
    print("  3. Engineering problems (constrained real-world designs)")
    print()
    
    # Run all tests
    test_classic_benchmarks()
    test_cec2017_suite()
    test_engineering_problems()
    compare_algorithms_on_problem()
    
    print("\n" + "=" * 70)
    print("Example completed!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  - Classic benchmarks: Good for quick algorithm testing")
    print("  - CEC2017: Standardized competition functions (unimodal, multimodal, hybrid, composition)")
    print("  - Engineering: Real-world constrained optimization problems")
    print("  - JADE: Best for simple problems (fast convergence)")
    print("  - L-SHADE: Best for complex problems (adaptive population)")
    print("  - LSHADE-EpSin: Best for constrained problems (epsilon handling)")
    print()


if __name__ == "__main__":
    main()
