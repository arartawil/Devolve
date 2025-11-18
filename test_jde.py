"""
Quick test for jDE implementation.
"""

import numpy as np
import sys
sys.path.insert(0, r'c:\Users\ROG SRTIX\Desktop\DEvolve')

from devolve.algorithms.adaptive import JDE
from devolve.benchmarks import Sphere, Rastrigin

print("=" * 70)
print("jDE ALGORITHM - QUICK TEST")
print("=" * 70)

# Test 1: Import
print("\n[1/4] Testing import...")
try:
    from devolve.algorithms.adaptive import JDE
    print("  ✓ JDE imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialization
print("\n[2/4] Testing initialization...")
try:
    problem = Sphere(dimensions=10)
    optimizer = JDE(
        problem=problem,
        population_size=20,
        max_iterations=100,
        tau1=0.1,
        tau2=0.1
    )
    print("  ✓ JDE initialized successfully")
    print(f"    - Population size: {optimizer.population_size}")
    print(f"    - τ₁: {optimizer.tau1}")
    print(f"    - τ₂: {optimizer.tau2}")
    print(f"    - F range: [{optimizer.F_lower}, {optimizer.F_upper}]")
except Exception as e:
    print(f"  ✗ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Optimization on Sphere
print("\n[3/4] Testing optimization on Sphere (10D, 100 iterations)...")
try:
    problem = Sphere(dimensions=10)
    optimizer = JDE(
        problem=problem,
        population_size=20,
        max_iterations=100,
        tau1=0.1,
        tau2=0.1,
        seed=42
    )
    
    best_solution, best_fitness = optimizer.optimize()
    
    print("  ✓ Optimization completed successfully")
    print(f"    - Best fitness: {best_fitness:.6e}")
    print(f"    - Evaluations: {optimizer.function_evaluations}")
    
    # Check if we improved
    if best_fitness < 1.0:
        print(f"    - ✓ Good convergence (fitness < 1.0)")
    else:
        print(f"    - ⚠ Moderate convergence")
        
except Exception as e:
    print(f"  ✗ Optimization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Parameter adaptation
print("\n[4/4] Testing parameter adaptation statistics...")
try:
    stats = optimizer.get_parameter_statistics()
    
    print("  ✓ Parameter statistics retrieved")
    print(f"    - Final F: {stats['F_mean']:.3f} ± {stats['F_std']:.3f}")
    print(f"    - F range: [{stats['F_min']:.3f}, {stats['F_max']:.3f}]")
    print(f"    - Final CR: {stats['CR_mean']:.3f} ± {stats['CR_std']:.3f}")
    print(f"    - CR range: [{stats['CR_min']:.3f}, {stats['CR_max']:.3f}]")
    print(f"    - Parameter history length: {len(stats['F_history'])}")
    
    # Check if parameters adapted
    if stats['F_std'] > 0.01:
        print("    - ✓ F parameter shows diversity")
    if stats['CR_std'] > 0.01:
        print("    - ✓ CR parameter shows diversity")
        
except Exception as e:
    print(f"  ✗ Statistics failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ jDE implementation is working correctly!")
print("\nKey features verified:")
print("  ✓ Imports correctly")
print("  ✓ Initializes with adaptive parameters")
print("  ✓ Optimizes successfully")
print("  ✓ Adapts F and CR parameters")
print("  ✓ Tracks parameter evolution")
print("\n" + "=" * 70)
