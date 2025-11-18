"""
Test all adaptive DE algorithms
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\Users\ROG SRTIX\Desktop\DEvolve')

from devolve.algorithms.adaptive import JDE, JADE, SHADE, LSHADE, LSHADEEpSin
from devolve.benchmarks import Sphere, Rastrigin

print("=" * 70)
print("ADAPTIVE DE ALGORITHMS - COMPREHENSIVE TEST")
print("=" * 70)

# Test configuration
DIM = 10
POP_SIZE = 20
MAX_ITER = 50
SEED = 42

algorithms = [
    ("jDE", JDE, {}),
    ("JADE", JADE, {"c": 0.1, "p": 0.05}),
    ("SHADE", SHADE, {"H": 5, "p": 0.11}),
    ("L-SHADE", LSHADE, {"H": 5, "NP_min": 4}),
    ("LSHADE-EpSin", LSHADEEpSin, {"H": 5, "epsilon_0": 0.25}),
]

results = []

for idx, (name, AlgorithmClass, params) in enumerate(algorithms, 1):
    print(f"\n[{idx}/{len(algorithms)}] Testing {name}...")
    print("-" * 70)
    
    try:
        # Create problem
        problem = Sphere(dimensions=DIM)
        
        # Create optimizer
        optimizer = AlgorithmClass(
            problem=problem,
            population_size=POP_SIZE,
            max_iterations=MAX_ITER,
            seed=SEED,
            **params
        )
        
        print(f"  ✓ {name} initialized")
        print(f"    Parameters: {params}")
        
        # Run optimization
        best_solution, best_fitness = optimizer.optimize()
        
        print(f"  ✓ Optimization completed")
        print(f"    Best fitness: {best_fitness:.6e}")
        print(f"    Function evaluations: {optimizer.function_evaluations}")
        
        # Get parameter statistics
        stats = optimizer.get_parameter_statistics()
        
        # Check convergence
        if best_fitness < 1.0:
            status = "✓ Good"
        elif best_fitness < 10.0:
            status = "○ Fair"
        else:
            status = "△ Slow"
        
        print(f"    Convergence: {status}")
        
        # Algorithm-specific info
        if name == "jDE":
            print(f"    Final F: {stats['F_mean']:.3f} ± {stats['F_std']:.3f}")
            print(f"    Final CR: {stats['CR_mean']:.3f} ± {stats['CR_std']:.3f}")
        elif name == "JADE":
            print(f"    Final μ_F: {stats['mu_F']:.3f}")
            print(f"    Final μ_CR: {stats['mu_CR']:.3f}")
            print(f"    Archive size: {stats['archive_size']}")
        elif name in ["SHADE", "L-SHADE", "LSHADE-EpSin"]:
            print(f"    Mean M_F: {stats['mean_F']:.3f}")
            print(f"    Mean M_CR: {stats['mean_CR']:.3f}")
            if 'current_NP' in stats:
                print(f"    Final NP: {stats['current_NP']} (from {optimizer.NP_init})")
            if 'current_epsilon' in stats:
                print(f"    Final ε: {stats['current_epsilon']:.3f}")
                print(f"    Final p: {stats['current_p']:.3f}")
        
        results.append((name, best_fitness, "✓"))
        
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        results.append((name, float('inf'), "✗"))

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("\nAlgorithm Performance on Sphere 10D:")
print("-" * 70)
print(f"{'Algorithm':<20} {'Best Fitness':>15} {'Status':>10}")
print("-" * 70)

for name, fitness, status in results:
    print(f"{name:<20} {fitness:>15.6e} {status:>10}")

print("-" * 70)

# Count successes
successes = sum(1 for _, _, status in results if status == "✓")
print(f"\n✓ {successes}/{len(algorithms)} algorithms working correctly!")

if successes == len(algorithms):
    print("\n🎉 All adaptive DE algorithms implemented successfully!")
else:
    print(f"\n⚠ {len(algorithms) - successes} algorithm(s) need attention")

print("\n" + "=" * 70)
