"""
Quick Demo - DEvolve Package
=============================
Demonstrates the DEvolve package working on a simple problem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from devolve.algorithms.classic import DErand1, DEbest1, DEcurrentToBest1
from devolve.benchmarks.classic import Sphere, Rastrigin

def main():
    print("="*70)
    print("DEvolve Package - Quick Demonstration")
    print("="*70)
    print()
    
    # Demo 1: Simple optimization
    print("Demo 1: Optimizing 10D Sphere Function with DE/rand/1")
    print("-" * 70)
    problem = Sphere(dimensions=10)
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=200,
        F=0.8,
        CR=0.9,
        random_seed=42
    )
    
    best_x, best_f = optimizer.optimize()
    print(f"\n✓ Optimization Complete!")
    print(f"  Best fitness: {best_f:.6e}")
    print(f"  Error from optimum: {best_f:.6e}")
    print(f"  Function evaluations: {optimizer.function_evaluations}")
    
    # Demo 2: Compare algorithms
    print("\n" + "="*70)
    print("Demo 2: Comparing 3 DE Variants on Rastrigin Function")
    print("-" * 70)
    
    problem = Rastrigin(dimensions=10)
    algorithms = [
        ('DE/rand/1', DErand1),
        ('DE/best/1', DEbest1),
        ('DE/current-to-best/1', DEcurrentToBest1),
    ]
    
    results = []
    for name, AlgoClass in algorithms:
        optimizer = AlgoClass(
            problem=problem,
            population_size=50,
            max_iterations=300,
            F=0.8,
            CR=0.9,
            random_seed=42
        )
        optimizer.logger.verbose = 0
        _, fitness = optimizer.optimize()
        results.append((name, fitness))
        print(f"  {name:25s} → {fitness:12.6e}")
    
    best_algo = min(results, key=lambda x: x[1])
    print(f"\n✓ Best Algorithm: {best_algo[0]} (fitness: {best_algo[1]:.6e})")
    
    # Demo 3: Custom problem
    print("\n" + "="*70)
    print("Demo 3: Custom Optimization Problem")
    print("-" * 70)
    
    from devolve.core.problem import Problem
    
    def custom_func(x):
        """Combination of polynomial and trigonometric terms."""
        return np.sum(x**2) + 0.1 * np.sum(np.sin(5*x)**2)
    
    problem = Problem(
        objective_function=custom_func,
        bounds=(-2, 2),
        dimensions=5,
        name="CustomFunction"
    )
    
    optimizer = DErand1(problem=problem, population_size=40, max_iterations=200)
    optimizer.logger.verbose = 0
    best_x, best_f = optimizer.optimize()
    
    print(f"  Problem: {problem.name}")
    print(f"  Dimensions: {problem.dimensions}")
    print(f"  Best solution found: {best_x}")
    print(f"  Best fitness: {best_f:.6f}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ All Demos Complete!")
    print("="*70)
    print("\nKey Features Demonstrated:")
    print("  • Classic DE algorithms (rand/1, best/1, current-to-best/1)")
    print("  • Standard benchmark functions (Sphere, Rastrigin)")
    print("  • Custom problem definition")
    print("  • Algorithm comparison")
    print("  • Flexible configuration (F, CR, population size, iterations)")
    print("\nPackage Status: ✓ FULLY OPERATIONAL")
    print("="*70)

if __name__ == "__main__":
    main()
