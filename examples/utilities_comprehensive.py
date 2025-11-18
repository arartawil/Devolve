"""
Comprehensive Utilities Example

Demonstrates all utility modules in a realistic optimization workflow:
1. Reproducibility with seed management
2. Performance metrics calculation
3. Statistical comparison of algorithms
4. Parallel execution
5. Result saving and export
"""

import numpy as np
from pathlib import Path

# Import DEvolve components
from devolve import JADE, SHADE, LSHADE, Sphere

# Import utilities
from devolve.utils import (
    # Seed management
    set_seed, get_seed_sequence, ensure_reproducibility,
    # Metrics
    calculate_success_rate, calculate_ert, calculate_convergence_speed,
    calculate_stability, PerformanceMetrics,
    # Statistics
    wilcoxon_test, friedman_test, nemenyi_posthoc_test,
    calculate_effect_size, StatisticalTests,
    # Parallel
    parallel_optimize,
    # I/O
    save_results, export_to_csv, export_to_latex_table,
    export_comparison_table
)


def example_reproducibility():
    """Example 1: Ensuring reproducibility."""
    print("=" * 70)
    print("Example 1: Reproducibility")
    print("=" * 70)
    
    # Set seed for complete reproducibility
    info = ensure_reproducibility(seed=42)
    print(f"\n✓ Reproducibility ensured:")
    print(f"  - Seed: {info['seed']}")
    print(f"  - NumPy: {info['numpy_version']}")
    
    # Generate seed sequence for multiple runs
    seeds = get_seed_sequence(n_runs=5, base_seed=42)
    print(f"\n✓ Generated {len(seeds)} seeds: {seeds}")
    
    # Run with same seed twice to verify
    problem = Sphere(dimensions=5)
    
    set_seed(42)
    optimizer1 = JADE(problem, population_size=20, max_iterations=10, random_seed=42)
    _, fitness1 = optimizer1.optimize()
    
    set_seed(42)
    optimizer2 = JADE(problem, population_size=20, max_iterations=10, random_seed=42)
    _, fitness2 = optimizer2.optimize()
    
    print(f"\n✓ Reproducibility verified:")
    print(f"  - Run 1: {fitness1:.6e}")
    print(f"  - Run 2: {fitness2:.6e}")
    print(f"  - Match: {np.isclose(fitness1, fitness2)}")


def example_metrics():
    """Example 2: Calculating performance metrics."""
    print("\n" + "=" * 70)
    print("Example 2: Performance Metrics")
    print("=" * 70)
    
    # Simulate multiple runs
    problem = Sphere(dimensions=10)
    seeds = get_seed_sequence(5, base_seed=100)
    
    print("\n✓ Running JADE 5 times...")
    results = []
    for seed in seeds:
        optimizer = JADE(problem, population_size=30, max_iterations=50, random_seed=seed)
        best_x, best_f = optimizer.optimize()
        fes = optimizer.function_evaluations
        results.append((best_f, fes))
    
    # Extract data
    fitness_values = [f for f, _ in results]
    
    # Calculate metrics
    metrics = PerformanceMetrics()
    
    # Success rate (target: error < 0.01)
    sr = metrics.success_rate(fitness_values, target_error=0.01, optimal_value=0.0)
    print(f"\n✓ Success rate (error < 0.01): {sr:.2%}")
    
    # Expected Running Time
    ert = metrics.ert(results, target_error=0.01, optimal_value=0.0)
    if ert:
        print(f"✓ Expected Running Time: {ert:.0f} FEs")
    
    # Stability
    stability = metrics.stability(fitness_values)
    print(f"\n✓ Stability metrics:")
    print(f"  - Mean: {stability['mean']:.6e}")
    print(f"  - Std: {stability['std']:.6e}")
    print(f"  - CV: {stability['cv']:.4f}")
    print(f"  - Min: {stability['min']:.6e}")
    print(f"  - Max: {stability['max']:.6e}")


def example_statistical_comparison():
    """Example 3: Statistical comparison of algorithms."""
    print("\n" + "=" * 70)
    print("Example 3: Statistical Comparison")
    print("=" * 70)
    
    # Run multiple algorithms on same problem
    problem = Sphere(dimensions=10)
    seeds = get_seed_sequence(10, base_seed=200)
    
    algorithms = {
        'JADE': JADE,
        'SHADE': SHADE,
        'L-SHADE': LSHADE
    }
    
    results = {}
    
    print("\n✓ Running 3 algorithms, 10 runs each...")
    for alg_name, AlgClass in algorithms.items():
        fitness_values = []
        for seed in seeds:
            if alg_name == 'L-SHADE':
                optimizer = AlgClass(problem, initial_population_size=30, 
                                    max_iterations=50, random_seed=seed)
            else:
                optimizer = AlgClass(problem, population_size=30, 
                                    max_iterations=50, random_seed=seed)
            _, best_f = optimizer.optimize()
            fitness_values.append(best_f)
        
        results[alg_name] = fitness_values
        print(f"  - {alg_name}: Mean={np.mean(fitness_values):.6e}")
    
    # Friedman test (overall comparison)
    tests = StatisticalTests()
    friedman_result = tests.friedman(results)
    
    print(f"\n✓ Friedman test (overall comparison):")
    print(f"  - χ²={friedman_result['statistic']:.4f}, p={friedman_result['p_value']:.4f}")
    print(f"  - Significant: {friedman_result['significant']}")
    print(f"  - Mean ranks: {friedman_result['mean_ranks']}")
    
    # Pairwise Wilcoxon tests
    print(f"\n✓ Pairwise comparisons (Wilcoxon):")
    for i, alg1 in enumerate(algorithms.keys()):
        for alg2 in list(algorithms.keys())[i+1:]:
            result = tests.wilcoxon(results[alg1], results[alg2])
            print(f"  - {alg1} vs {alg2}: p={result['p_value']:.4f} "
                  f"({'significant' if result['significant'] else 'not significant'})")
    
    # Effect sizes
    print(f"\n✓ Effect sizes (Cohen's d):")
    jade_results = results['JADE']
    for alg_name in ['SHADE', 'L-SHADE']:
        es = tests.effect_size(jade_results, results[alg_name], method='cohen')
        print(f"  - JADE vs {alg_name}: d={es['effect_size']:.3f} ({es['magnitude']})")
    
    # Nemenyi post-hoc test
    nemenyi = tests.nemenyi(results, alpha=0.05)
    print(f"\n✓ Nemenyi post-hoc test:")
    print(f"  - Critical difference: {nemenyi['critical_difference']:.3f}")
    if nemenyi['significant_pairs']:
        print(f"  - Significant pairs: {nemenyi['significant_pairs']}")
    else:
        print(f"  - No significant pairwise differences")
    
    return results


def example_parallel_execution():
    """Example 4: Parallel optimization runs."""
    print("\n" + "=" * 70)
    print("Example 4: Parallel Execution")
    print("=" * 70)
    
    problem = Sphere(dimensions=10)
    
    # Run 20 independent optimizations in parallel
    print("\n✓ Running 20 independent optimizations in parallel...")
    results = parallel_optimize(
        optimizer_class=JADE,
        problem=problem,
        n_runs=20,
        n_jobs=4,  # Use 4 parallel processes
        population_size=30,
        max_iterations=50
    )
    
    fitness_values = [f for _, f in results]
    
    print(f"\n✓ Parallel execution completed:")
    print(f"  - Number of runs: {len(results)}")
    print(f"  - Mean fitness: {np.mean(fitness_values):.6e}")
    print(f"  - Best fitness: {np.min(fitness_values):.6e}")
    print(f"  - Worst fitness: {np.max(fitness_values):.6e}")


def example_save_and_export():
    """Example 5: Saving and exporting results."""
    print("\n" + "=" * 70)
    print("Example 5: Save and Export Results")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("utility_examples_output")
    output_dir.mkdir(exist_ok=True)
    
    # Run a single optimization
    problem = Sphere(dimensions=10)
    optimizer = JADE(problem, population_size=50, max_iterations=100, random_seed=42)
    best_x, best_f = optimizer.optimize()
    
    # Prepare results
    results = {
        'algorithm': 'JADE',
        'problem': 'Sphere',
        'dimensions': 10,
        'best_fitness': best_f,
        'best_position': best_x,
        'function_evaluations': optimizer.function_evaluations,
        'history': {
            'iteration': list(range(len(optimizer.logger.history['best_fitness']))),
            'best_fitness': optimizer.logger.history['best_fitness']
        }
    }
    
    # Save as JSON
    json_path = output_dir / "results.json"
    save_results(results, json_path, format='json')
    print(f"\n✓ Saved results to JSON: {json_path}")
    
    # Save as Pickle
    pkl_path = output_dir / "results.pkl"
    save_results(results, pkl_path, format='pickle')
    print(f"✓ Saved results to Pickle: {pkl_path}")
    
    # Export history to CSV
    csv_path = output_dir / "history.csv"
    export_to_csv(results['history'], csv_path)
    print(f"✓ Exported history to CSV: {csv_path}")
    
    # Create comparison table (simulate multiple algorithms)
    comparison_data = {
        'Algorithm': ['JADE', 'SHADE', 'L-SHADE'],
        'Mean': [0.00123, 0.00245, 0.00156],
        'Std': [0.00012, 0.00023, 0.00015],
        'Min': [0.00089, 0.00178, 0.00112],
        'Max': [0.00167, 0.00334, 0.00223]
    }
    
    # Export to CSV
    comp_csv = output_dir / "comparison.csv"
    export_to_csv(comparison_data, comp_csv)
    print(f"✓ Exported comparison to CSV: {comp_csv}")
    
    # Export to LaTeX table
    latex_path = output_dir / "comparison_table.tex"
    export_to_latex_table(
        comparison_data,
        latex_path,
        caption='Algorithm Comparison on Sphere Function (D=10)',
        label='tab:sphere_comparison',
        format_spec='.6f',
        bold_best=True
    )
    print(f"✓ Exported comparison to LaTeX: {latex_path}")
    
    print(f"\n✓ All outputs saved to: {output_dir}/")


def example_complete_workflow(algorithm_results):
    """Example 6: Complete analysis workflow."""
    print("\n" + "=" * 70)
    print("Example 6: Complete Analysis Workflow")
    print("=" * 70)
    
    output_dir = Path("utility_examples_output")
    output_dir.mkdir(exist_ok=True)
    
    # 1. Calculate comprehensive statistics
    print("\n✓ Calculating comprehensive statistics...")
    stats_table = {'Algorithm': []}
    for key in ['Mean', 'Std', 'Min', 'Max', 'Median', 'CV']:
        stats_table[key] = []
    
    for alg_name, values in algorithm_results.items():
        stability = calculate_stability(values)
        stats_table['Algorithm'].append(alg_name)
        stats_table['Mean'].append(stability['mean'])
        stats_table['Std'].append(stability['std'])
        stats_table['Min'].append(stability['min'])
        stats_table['Max'].append(stability['max'])
        stats_table['Median'].append(stability['median'])
        stats_table['CV'].append(stability['cv'])
    
    # 2. Export statistics table
    stats_path = output_dir / "algorithm_statistics.csv"
    export_to_csv(stats_table, stats_path)
    print(f"  - Saved statistics: {stats_path}")
    
    # 3. Statistical tests summary
    print("\n✓ Performing statistical tests...")
    tests = StatisticalTests()
    
    test_results = []
    alg_names = list(algorithm_results.keys())
    for i, alg1 in enumerate(alg_names):
        for alg2 in alg_names[i+1:]:
            wilcoxon = tests.wilcoxon(algorithm_results[alg1], algorithm_results[alg2])
            effect = tests.effect_size(algorithm_results[alg1], algorithm_results[alg2])
            
            test_results.append({
                'Comparison': f"{alg1} vs {alg2}",
                'P-value': wilcoxon['p_value'],
                'Significant': wilcoxon['significant'],
                'Effect Size': effect['effect_size'],
                'Magnitude': effect['magnitude']
            })
    
    # 4. Export test results
    tests_path = output_dir / "statistical_tests.csv"
    export_to_csv(test_results, tests_path)
    print(f"  - Saved test results: {tests_path}")
    
    # 5. Export LaTeX table
    latex_path = output_dir / "algorithm_comparison.tex"
    export_to_latex_table(
        stats_table,
        latex_path,
        caption='Algorithm Performance Comparison',
        label='tab:algorithm_performance',
        format_spec='.6e'
    )
    print(f"  - Saved LaTeX table: {latex_path}")
    
    print(f"\n✓ Complete analysis saved to: {output_dir}/")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DEvolve Utilities Comprehensive Examples")
    print("=" * 70)
    print("\nDemonstrating:")
    print("  1. Reproducibility with seed management")
    print("  2. Performance metrics calculation")
    print("  3. Statistical comparison of algorithms")
    print("  4. Parallel execution")
    print("  5. Saving and exporting results")
    print("  6. Complete analysis workflow")
    print()
    
    # Run examples
    example_reproducibility()
    example_metrics()
    algorithm_results = example_statistical_comparison()
    example_parallel_execution()
    example_save_and_export()
    example_complete_workflow(algorithm_results)
    
    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  - Use set_seed() at the start for reproducibility")
    print("  - PerformanceMetrics class provides all common metrics")
    print("  - StatisticalTests class for rigorous algorithm comparison")
    print("  - parallel_optimize() for efficient multiple runs")
    print("  - export_to_latex_table() for publication-ready tables")
    print("  - Always save experiment configurations with results")
    print()


if __name__ == "__main__":
    main()
