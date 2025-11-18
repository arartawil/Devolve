"""
Example 1: Basic Optimization
==============================

This example demonstrates basic usage of DEvolve with different
benchmark functions and classic DE variants.
"""

import numpy as np
from devolve.algorithms.classic import DErand1, DEbest1, DEcurrentToBest1
from devolve.benchmarks.classic import Sphere, Rastrigin, Rosenbrock, Ackley


def example_basic_optimization():
    """Run basic optimization on Sphere function."""
    print("="*70)
    print("Example 1: Basic Optimization on Sphere Function")
    print("="*70)
    
    # Create problem
    problem = Sphere(dimensions=10)
    
    # Create optimizer
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=500,
        F=0.8,
        CR=0.9,
        random_seed=42
    )
    
    # Run optimization
    best_position, best_fitness = optimizer.optimize()
    
    print(f"\nOptimization completed!")
    print(f"Best fitness: {best_fitness:.6e}")
    print(f"Distance to optimum: {np.linalg.norm(best_position):.6e}")
    print(f"Function evaluations: {optimizer.function_evaluations}")
    

def example_compare_algorithms():
    """Compare different DE variants on Rastrigin function."""
    print("\n" + "="*70)
    print("Example 2: Comparing DE Variants on Rastrigin Function")
    print("="*70)
    
    problem = Rastrigin(dimensions=20)
    
    algorithms = {
        'DE/rand/1': DErand1(problem, population_size=100, max_iterations=500, random_seed=42),
        'DE/best/1': DEbest1(problem, population_size=100, max_iterations=500, random_seed=42),
        'DE/current-to-best/1': DEcurrentToBest1(problem, population_size=100, max_iterations=500, random_seed=42),
    }
    
    results = {}
    
    for name, algo in algorithms.items():
        print(f"\nRunning {name}...")
        _, fitness = algo.optimize()
        results[name] = fitness
        print(f"  Final fitness: {fitness:.6e}")
    
    print("\n" + "-"*70)
    print("Summary:")
    best_algo = min(results, key=results.get)
    print(f"Best algorithm: {best_algo} (fitness: {results[best_algo]:.6e})")


def example_multiple_problems():
    """Test DE/rand/1 on multiple benchmark functions."""
    print("\n" + "="*70)
    print("Example 3: Testing on Multiple Benchmark Functions")
    print("="*70)
    
    dimensions = 10
    problems = {
        'Sphere': Sphere(dimensions=dimensions),
        'Rosenbrock': Rosenbrock(dimensions=dimensions),
        'Rastrigin': Rastrigin(dimensions=dimensions),
        'Ackley': Ackley(dimensions=dimensions),
    }
    
    print(f"\nDimensions: {dimensions}")
    print(f"Algorithm: DE/rand/1")
    print(f"Population size: 50, Iterations: 500\n")
    
    for name, problem in problems.items():
        optimizer = DErand1(
            problem=problem,
            population_size=50,
            max_iterations=500,
            F=0.8,
            CR=0.9,
            random_seed=42
        )
        
        optimizer.logger.verbose = 0  # Silent mode
        _, fitness = optimizer.optimize()
        
        error = fitness - problem.optimum if problem.optimum is not None else fitness
        print(f"{name:15s} - Final error: {error:.6e}")


def example_with_settings():
    """Demonstrate various algorithm settings."""
    print("\n" + "="*70)
    print("Example 4: Testing Different Settings")
    print("="*70)
    
    problem = Rastrigin(dimensions=10)
    
    # Test different F values
    print("\nEffect of F (Scaling Factor):")
    for F in [0.4, 0.6, 0.8, 1.0]:
        optimizer = DErand1(
            problem=problem,
            population_size=50,
            max_iterations=300,
            F=F,
            CR=0.9,
            random_seed=42
        )
        optimizer.logger.verbose = 0
        _, fitness = optimizer.optimize()
        print(f"  F={F:.1f}: fitness={fitness:.6e}")
    
    # Test different CR values
    print("\nEffect of CR (Crossover Rate):")
    for CR in [0.3, 0.5, 0.7, 0.9]:
        optimizer = DErand1(
            problem=problem,
            population_size=50,
            max_iterations=300,
            F=0.8,
            CR=CR,
            random_seed=42
        )
        optimizer.logger.verbose = 0
        _, fitness = optimizer.optimize()
        print(f"  CR={CR:.1f}: fitness={fitness:.6e}")
    
    # Test different crossover strategies
    print("\nEffect of Crossover Strategy:")
    for strategy in ['binomial', 'exponential']:
        optimizer = DErand1(
            problem=problem,
            population_size=50,
            max_iterations=300,
            F=0.8,
            CR=0.9,
            crossover_strategy=strategy,
            random_seed=42
        )
        optimizer.logger.verbose = 0
        _, fitness = optimizer.optimize()
        print(f"  {strategy:12s}: fitness={fitness:.6e}")


def example_convergence_analysis():
    """Analyze convergence behavior."""
    print("\n" + "="*70)
    print("Example 5: Convergence Analysis")
    print("="*70)
    
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
    best_position, best_fitness = optimizer.optimize()
    
    history = optimizer.best_fitness_history
    
    print(f"\nConvergence Progress:")
    print(f"{'Iteration':<12} {'Best Fitness':<15} {'Improvement':<15}")
    print("-" * 42)
    
    milestones = [0, 10, 50, 100, 150, 199]
    for i in milestones:
        improvement = history[i-1] - history[i] if i > 0 else 0
        print(f"{i:<12} {history[i]:<15.6e} {improvement:<15.6e}")
    
    print(f"\nFinal best fitness: {best_fitness:.6e}")
    print(f"Total improvement: {history[0] - history[-1]:.6e}")


if __name__ == "__main__":
    # Run all examples
    example_basic_optimization()
    example_compare_algorithms()
    example_multiple_problems()
    example_with_settings()
    example_convergence_analysis()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
