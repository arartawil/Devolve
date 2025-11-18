"""
Example 3: Comprehensive Visualization Features

This script demonstrates all visualization capabilities of the DEvolve package.

Run this after installing:
    pip install -e .
    pip install matplotlib seaborn tqdm

Usage:
    python examples/03_visualization_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import DEvolve
from devolve import DErand1, DEbest1, DEcurrent_to_best1
from devolve.benchmarks import Sphere, Rastrigin, Rosenbrock, Ackley

# Import visualization
from devolve.utils import (
    set_publication_style,
    setup_figure_folders,
    plot_convergence,
    plot_convergence_with_ci,
    plot_algorithm_comparison,
    plot_statistical_comparison,
    plot_population_2d,
    plot_3d_landscape,
    generate_comparison_table,
    create_comprehensive_report,
    generate_all_figures
)


def demo_1_basic_convergence():
    """Demo 1: Basic convergence plot."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Convergence Plot")
    print("="*70)
    
    # Run optimization
    problem = Sphere(dimensions=10)
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=100,
        F=0.8,
        CR=0.9
    )
    
    best_solution, best_fitness = optimizer.optimize()
    
    # Plot convergence
    history = optimizer.logger.get_history()
    fig = plot_convergence(
        history=history,
        title="DE/rand/1 on 10D Sphere Function",
        xlabel="Iterations",
        ylabel="Best Fitness",
        log_scale=True,
        show_mean=True,
        show_std=True
    )
    
    # Save figure
    Path("figures/examples").mkdir(parents=True, exist_ok=True)
    fig.savefig("figures/examples/demo1_convergence.png", dpi=300, bbox_inches='tight')
    print(f"✓ Convergence plot saved to: figures/examples/demo1_convergence.png")
    print(f"  Best fitness: {best_fitness:.6e}")
    
    plt.show()


def demo_2_algorithm_comparison():
    """Demo 2: Compare multiple algorithms."""
    print("\n" + "="*70)
    print("DEMO 2: Algorithm Comparison")
    print("="*70)
    
    # Define algorithms to compare
    algorithms = {
        'DE/rand/1': DErand1,
        'DE/best/1': DEbest1,
        'DE/current-to-best/1': DEcurrent_to_best1
    }
    
    # Run each algorithm
    problem = Rastrigin(dimensions=10)
    results = {}
    
    for name, AlgoClass in algorithms.items():
        print(f"  Running {name}...", end=" ")
        optimizer = AlgoClass(
            problem=problem,
            population_size=50,
            max_iterations=200,
            F=0.8,
            CR=0.9,
            seed=42
        )
        _, fitness = optimizer.optimize()
        results[name] = optimizer.logger.get_history()
        print(f"✓ (Best: {fitness:.4f})")
    
    # Plot comparison
    fig = plot_algorithm_comparison(
        results_dict=results,
        title="Algorithm Comparison on 10D Rastrigin",
        log_scale=False
    )
    
    # Save figure
    fig.savefig("figures/examples/demo2_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: figures/examples/demo2_comparison.png")
    
    plt.show()


def demo_3_multiple_runs_with_ci():
    """Demo 3: Multiple runs with confidence intervals."""
    print("\n" + "="*70)
    print("DEMO 3: Multiple Runs with Confidence Intervals")
    print("="*70)
    
    # Run algorithm multiple times
    problem = Sphere(dimensions=10)
    n_runs = 10
    runs_data = []
    
    print(f"  Running {n_runs} independent trials...")
    for run in range(n_runs):
        optimizer = DErand1(
            problem=problem,
            population_size=50,
            max_iterations=150,
            F=0.8,
            CR=0.9,
            seed=42 + run
        )
        _, _ = optimizer.optimize()
        history = optimizer.logger.get_history()
        runs_data.append(history['best_fitness'])
        print(f"    Run {run+1}/{n_runs} ✓")
    
    # Plot with confidence intervals
    fig = plot_convergence_with_ci(
        runs_data=runs_data,
        confidence_level=0.95,
        title=f"DE/rand/1 on 10D Sphere ({n_runs} runs)",
        log_scale=True,
        show_median=True,
        show_mean=True,
        show_best=True,
        show_worst=False
    )
    
    # Save figure
    fig.savefig("figures/examples/demo3_confidence_intervals.png", dpi=300, bbox_inches='tight')
    print(f"✓ CI plot saved to: figures/examples/demo3_confidence_intervals.png")
    
    plt.show()


def demo_4_statistical_comparison():
    """Demo 4: Statistical comparison with box plots."""
    print("\n" + "="*70)
    print("DEMO 4: Statistical Comparison (Box Plot)")
    print("="*70)
    
    # Run multiple algorithms multiple times
    algorithms = {
        'DE/rand/1': DErand1,
        'DE/best/1': DEbest1,
        'DE/current-to-best/1': DEcurrent_to_best1
    }
    
    problem = Rastrigin(dimensions=10)
    n_runs = 10
    final_results = {name: [] for name in algorithms.keys()}
    
    print(f"  Running {len(algorithms)} algorithms × {n_runs} runs...")
    for name, AlgoClass in algorithms.items():
        for run in range(n_runs):
            optimizer = AlgoClass(
                problem=problem,
                population_size=50,
                max_iterations=200,
                F=0.8,
                CR=0.9,
                seed=100 + run
            )
            _, fitness = optimizer.optimize()
            final_results[name].append(fitness)
        print(f"    {name}: Mean = {np.mean(final_results[name]):.4f} ± {np.std(final_results[name]):.4f}")
    
    # Plot statistical comparison
    fig = plot_statistical_comparison(
        results=final_results,
        metric='best_fitness',
        title=f"Statistical Comparison ({n_runs} runs)",
        ylabel="Final Best Fitness",
        plot_type='box',
        show_significance=True
    )
    
    # Save figure
    fig.savefig("figures/examples/demo4_statistical.png", dpi=300, bbox_inches='tight')
    print(f"✓ Statistical plot saved to: figures/examples/demo4_statistical.png")
    
    plt.show()


def demo_5_population_2d():
    """Demo 5: Population scatter plot in 2D."""
    print("\n" + "="*70)
    print("DEMO 5: Population Visualization (2D)")
    print("="*70)
    
    # Run on 2D problem
    problem = Rastrigin(dimensions=2)
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=100,
        F=0.8,
        CR=0.9
    )
    
    best_solution, best_fitness = optimizer.optimize()
    
    # Get final population
    pop = optimizer.population
    pop_array = np.array([ind.position for ind in pop.individuals])
    fitness_array = np.array([ind.fitness for ind in pop.individuals])
    
    # Plot population
    fig = plot_population_2d(
        population=pop_array,
        fitness_values=fitness_array,
        best_solution=best_solution,
        iteration=optimizer.current_iteration,
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        contour_function=lambda x: problem.objective_function(x),
        show_contour=True
    )
    
    # Save figure
    fig.savefig("figures/examples/demo5_population_2d.png", dpi=300, bbox_inches='tight')
    print(f"✓ Population plot saved to: figures/examples/demo5_population_2d.png")
    print(f"  Best fitness: {best_fitness:.6f}")
    
    plt.show()


def demo_6_3d_landscape():
    """Demo 6: 3D fitness landscape."""
    print("\n" + "="*70)
    print("DEMO 6: 3D Fitness Landscape")
    print("="*70)
    
    # Create 3D landscape
    problem = Rastrigin(dimensions=2)
    
    # Run optimization to get population
    optimizer = DErand1(
        problem=problem,
        population_size=30,
        max_iterations=50,
        F=0.8,
        CR=0.9
    )
    
    best_solution, _ = optimizer.optimize()
    
    pop_array = np.array([ind.position for ind in optimizer.population.individuals])
    
    # Plot 3D landscape
    fig = plot_3d_landscape(
        function=lambda x: problem.objective_function(x),
        bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        population=pop_array,
        best_solution=best_solution,
        resolution=80,
        elevation=30,
        azimuth=45
    )
    
    # Save figure
    fig.savefig("figures/examples/demo6_3d_landscape.png", dpi=300, bbox_inches='tight')
    print(f"✓ 3D landscape saved to: figures/examples/demo6_3d_landscape.png")
    
    plt.show()


def demo_7_latex_table():
    """Demo 7: Generate LaTeX comparison table."""
    print("\n" + "="*70)
    print("DEMO 7: LaTeX Comparison Table")
    print("="*70)
    
    # Simulate results from multiple algorithms
    results_dict = {
        'DE/rand/1': {
            'Mean': 12.345e-05,
            'Std': 4.567e-06,
            'Best': 8.901e-06,
            'Worst': 23.456e-05,
            'Success Rate': 85.0
        },
        'DE/best/1': {
            'Mean': 9.876e-06,
            'Std': 3.210e-06,
            'Best': 7.654e-06,
            'Worst': 19.876e-05,
            'Success Rate': 90.0
        },
        'JADE': {
            'Mean': 3.456e-06,
            'Std': 1.234e-06,
            'Best': 2.345e-06,
            'Worst': 5.678e-06,
            'Success Rate': 100.0
        }
    }
    
    # Generate LaTeX table
    Path("figures/tables").mkdir(parents=True, exist_ok=True)
    latex_str = generate_comparison_table(
        results_dict=results_dict,
        metrics=['Mean', 'Std', 'Best', 'Worst', 'Success Rate'],
        save_path="figures/tables/comparison_table.tex",
        bold_best=True,
        format_scientific=True,
        caption="Algorithm Comparison on Benchmark Functions",
        label="tab:comparison"
    )
    
    print("✓ LaTeX table generated:")
    print(latex_str)
    print(f"\n  Saved to: figures/tables/comparison_table.tex")


def demo_8_comprehensive_report():
    """Demo 8: Comprehensive report with all subplots."""
    print("\n" + "="*70)
    print("DEMO 8: Comprehensive Report")
    print("="*70)
    
    # Run optimization
    problem = Sphere(dimensions=10)
    optimizer = DErand1(
        problem=problem,
        population_size=50,
        max_iterations=150,
        F=0.8,
        CR=0.9
    )
    
    best_solution, best_fitness = optimizer.optimize()
    
    # Create results object
    class Results:
        def __init__(self, optimizer):
            self.history = optimizer.logger.get_history()
            self.best_fitness_history = self.history['best_fitness']
            self.final_population = np.array([ind.position for ind in optimizer.population.individuals])
            self.final_fitness = np.array([ind.fitness for ind in optimizer.population.individuals])
            self.best_solution = best_solution
            self.dimension = problem.dimensions
            self.f_history = None  # Not adaptive
            self.cr_history = None
            self.diversity_history = None
    
    results = Results(optimizer)
    
    # Generate comprehensive report
    fig = create_comprehensive_report(
        results=results,
        algorithm_name="DE/rand/1",
        problem_name=f"Sphere_{problem.dimensions}D",
        save_path="figures/examples/demo8_comprehensive_report",
        file_formats=['png']
    )
    
    print(f"✓ Comprehensive report saved to: figures/examples/demo8_comprehensive_report.png")
    print(f"  Best fitness: {best_fitness:.6e}")
    
    plt.show()


def demo_9_automatic_generation():
    """Demo 9: Automatic generation of all figures."""
    print("\n" + "="*70)
    print("DEMO 9: Automatic Figure Generation")
    print("="*70)
    
    # Run optimization
    problem = Rastrigin(dimensions=10)
    optimizer = DErand1(
        problem=problem,
        population_size=75,
        max_iterations=200,
        F=0.8,
        CR=0.9
    )
    
    best_solution, best_fitness = optimizer.optimize()
    
    # Create results object
    class Results:
        def __init__(self, optimizer, problem):
            self.history = optimizer.logger.get_history()
            self.best_fitness_history = self.history['best_fitness']
            self.final_population = np.array([ind.position for ind in optimizer.population.individuals])
            self.final_fitness = np.array([ind.fitness for ind in optimizer.population.individuals])
            self.best_solution = best_solution
            self.dimension = problem.dimensions
            self.bounds = [(problem.bounds[0], problem.bounds[1]) for _ in range(problem.dimensions)]
            self.f_history = None
            self.cr_history = None
            self.diversity_history = None
            self.iteration = optimizer.current_iteration
    
    results = Results(optimizer, problem)
    
    # Generate all figures automatically
    folders = generate_all_figures(
        results=results,
        algorithm_name='DErand1',
        problem_name=f'Rastrigin_{problem.dimensions}D',
        base_save_path='figures/auto_generated',
        formats=['png', 'pdf'],
        dpi=300,
        generate_animation=False
    )
    
    print(f"\n✓ All figures generated successfully!")
    print(f"  Base folder: figures/auto_generated/")
    print(f"  Best fitness: {best_fitness:.6f}")


def main():
    """Run all visualization demos."""
    print("\n" + "="*70)
    print("DEVOLVE VISUALIZATION DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates all visualization capabilities.")
    print("Figures will be saved to the 'figures/' directory.")
    print("\nPress Enter to start...")
    input()
    
    # Set publication style globally
    set_publication_style()
    print("✓ Publication style applied")
    
    # Setup figure folders
    folders = setup_figure_folders("figures")
    print(f"✓ Figure folders created: {list(folders.keys())}")
    
    # Run demos
    try:
        demo_1_basic_convergence()
    except Exception as e:
        print(f"✗ Demo 1 failed: {e}")
    
    try:
        demo_2_algorithm_comparison()
    except Exception as e:
        print(f"✗ Demo 2 failed: {e}")
    
    try:
        demo_3_multiple_runs_with_ci()
    except Exception as e:
        print(f"✗ Demo 3 failed: {e}")
    
    try:
        demo_4_statistical_comparison()
    except Exception as e:
        print(f"✗ Demo 4 failed: {e}")
    
    try:
        demo_5_population_2d()
    except Exception as e:
        print(f"✗ Demo 5 failed: {e}")
    
    try:
        demo_6_3d_landscape()
    except Exception as e:
        print(f"✗ Demo 6 failed: {e}")
    
    try:
        demo_7_latex_table()
    except Exception as e:
        print(f"✗ Demo 7 failed: {e}")
    
    try:
        demo_8_comprehensive_report()
    except Exception as e:
        print(f"✗ Demo 8 failed: {e}")
    
    try:
        demo_9_automatic_generation()
    except Exception as e:
        print(f"✗ Demo 9 failed: {e}")
    
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETED!")
    print("="*70)
    print("\nAll figures have been saved to the 'figures/' directory.")
    print("Check the following subdirectories:")
    for folder_name in folders.keys():
        print(f"  - figures/{folder_name}/")
    print("\nYou can now use these functions in your own scripts!")


if __name__ == '__main__':
    main()
