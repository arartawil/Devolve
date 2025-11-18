"""
Example 2: Custom Problem Definition
=====================================

This example shows how to define custom optimization problems
with constraints and custom objective functions.
"""

import numpy as np
from devolve.core.problem import Problem
from devolve.algorithms.classic import DErand1


def example_simple_custom_problem():
    """Define and solve a simple custom problem."""
    print("="*70)
    print("Example 1: Simple Custom Function")
    print("="*70)
    
    # Define objective function
    def custom_function(x):
        """Sum of squares plus product of cosines."""
        return np.sum(x**2) + np.abs(np.prod(np.cos(x)))
    
    # Create problem
    problem = Problem(
        objective_function=custom_function,
        bounds=(-5.0, 5.0),
        dimensions=5,
        name="CustomFunction"
    )
    
    # Optimize
    optimizer = DErand1(
        problem=problem,
        population_size=30,
        max_iterations=300,
        F=0.8,
        CR=0.9
    )
    
    best_x, best_f = optimizer.optimize()
    
    print(f"\nBest solution: {best_x}")
    print(f"Best fitness: {best_f:.6f}")


def example_constrained_problem():
    """Solve a constrained optimization problem."""
    print("\n" + "="*70)
    print("Example 2: Constrained Optimization")
    print("="*70)
    
    # Minimize x^2 + y^2 subject to x + y >= 1
    def objective(x):
        return x[0]**2 + x[1]**2
    
    def constraint1(x):
        # Constraint: x + y >= 1, rewrite as: -(x + y - 1) <= 0
        return -(x[0] + x[1] - 1)
    
    problem = Problem(
        objective_function=objective,
        bounds=[(-5, 5), (-5, 5)],
        dimensions=2,
        constraints=[constraint1],
        name="ConstrainedCircle"
    )
    
    optimizer = DErand1(
        problem=problem,
        population_size=40,
        max_iterations=200
    )
    
    best_x, best_f = optimizer.optimize()
    
    print(f"\nBest solution: x={best_x[0]:.6f}, y={best_x[1]:.6f}")
    print(f"Best fitness: {best_f:.6f}")
    print(f"Sum x+y: {best_x[0] + best_x[1]:.6f} (should be >= 1)")
    
    # Verify constraint
    is_feasible, violation = problem.evaluate_constraints(best_x)
    print(f"Feasible: {is_feasible}, Violation: {violation:.6f}")


def example_different_bounds():
    """Problem with different bounds per dimension."""
    print("\n" + "="*70)
    print("Example 3: Different Bounds Per Dimension")
    print("="*70)
    
    def objective(x):
        # Weighted sum with different scales
        weights = np.array([1, 2, 3, 4, 5])
        return np.sum(weights * x**2)
    
    # Different bounds for each dimension
    bounds = [
        (-10, 10),   # dim 0
        (-5, 5),     # dim 1
        (-2, 2),     # dim 2
        (-1, 1),     # dim 3
        (-0.5, 0.5)  # dim 4
    ]
    
    problem = Problem(
        objective_function=objective,
        bounds=bounds,
        dimensions=5,
        name="VariableBounds"
    )
    
    optimizer = DErand1(problem=problem, population_size=30, max_iterations=200)
    optimizer.logger.verbose = 0
    
    best_x, best_f = optimizer.optimize()
    
    print(f"\nBounds per dimension:")
    for i, (lower, upper) in enumerate(bounds):
        print(f"  Dim {i}: [{lower:6.1f}, {upper:6.1f}] -> best_x[{i}] = {best_x[i]:8.6f}")
    
    print(f"\nBest fitness: {best_f:.6e}")


def example_multiple_constraints():
    """Problem with multiple constraints."""
    print("\n" + "="*70)
    print("Example 4: Multiple Constraints")
    print("="*70)
    
    def objective(x):
        return (x[0] - 3)**2 + (x[1] - 2)**2
    
    def constraint1(x):
        # x + y <= 5
        return x[0] + x[1] - 5
    
    def constraint2(x):
        # x >= 1
        return -(x[0] - 1)
    
    def constraint3(x):
        # y >= 0.5
        return -(x[1] - 0.5)
    
    problem = Problem(
        objective_function=objective,
        bounds=[(0, 10), (0, 10)],
        dimensions=2,
        constraints=[constraint1, constraint2, constraint3],
        name="MultiConstraint"
    )
    
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=300
    )
    
    best_x, best_f = optimizer.optimize()
    
    print(f"\nBest solution: x={best_x[0]:.6f}, y={best_x[1]:.6f}")
    print(f"Best fitness: {best_f:.6f}")
    print(f"\nConstraint values:")
    print(f"  x + y = {best_x[0] + best_x[1]:.6f} (should be <= 5)")
    print(f"  x = {best_x[0]:.6f} (should be >= 1)")
    print(f"  y = {best_x[1]:.6f} (should be >= 0.5)")


def example_engineering_problem():
    """Pressure vessel design problem (engineering application)."""
    print("\n" + "="*70)
    print("Example 5: Engineering Design Problem")
    print("="*70)
    print("Pressure Vessel Design Optimization\n")
    
    def pressure_vessel_cost(x):
        """
        Minimize cost of a cylindrical pressure vessel.
        x[0] = thickness of shell (Ts)
        x[1] = thickness of head (Th)
        x[2] = inner radius (R)
        x[3] = length of cylindrical section (L)
        """
        Ts, Th, R, L = x
        
        # Material cost
        material_cost = 0.6224 * Ts * R * L
        # Forming cost
        forming_cost = 1.7781 * Th * R**2
        # Welding cost
        welding_cost = 3.1661 * Ts**2 * L
        # Head cost
        head_cost = 19.84 * Ts**2 * R
        
        total_cost = material_cost + forming_cost + welding_cost + head_cost
        return total_cost
    
    def constraint1(x):
        # Shell thickness constraint
        Ts, _, R, _ = x
        return -(Ts - 0.0193 * R)
    
    def constraint2(x):
        # Head thickness constraint
        _, Th, R, _ = x
        return -(Th - 0.00954 * R)
    
    def constraint3(x):
        # Volume constraint (minimum volume)
        Ts, Th, R, L = x
        volume = np.pi * R**2 * L + (4/3) * np.pi * R**3
        return -(volume - 1296000)  # cm³
    
    def constraint4(x):
        # Length constraint
        _, _, _, L = x
        return L - 240  # max 240 inches
    
    # Bounds: Ts, Th in [0.0625, 99.75], R in [10, 200], L in [10, 200]
    bounds = [(0.0625, 99.75), (0.0625, 99.75), (10, 200), (10, 200)]
    
    problem = Problem(
        objective_function=pressure_vessel_cost,
        bounds=bounds,
        dimensions=4,
        constraints=[constraint1, constraint2, constraint3, constraint4],
        name="PressureVessel"
    )
    
    optimizer = DErand1(
        problem=problem,
        population_size=60,
        max_iterations=500,
        F=0.8,
        CR=0.9
    )
    
    optimizer.logger.verbose = 1
    best_x, best_cost = optimizer.optimize()
    
    print(f"\nOptimal Design:")
    print(f"  Shell thickness (Ts): {best_x[0]:.4f} in")
    print(f"  Head thickness (Th):  {best_x[1]:.4f} in")
    print(f"  Inner radius (R):     {best_x[2]:.4f} in")
    print(f"  Length (L):           {best_x[3]:.4f} in")
    print(f"\nMinimum cost: ${best_cost:.2f}")
    
    is_feasible, violation = problem.evaluate_constraints(best_x)
    print(f"Solution feasible: {is_feasible}")


if __name__ == "__main__":
    example_simple_custom_problem()
    example_constrained_problem()
    example_different_bounds()
    example_multiple_constraints()
    example_engineering_problem()
    
    print("\n" + "="*70)
    print("All custom problem examples completed!")
    print("="*70)
