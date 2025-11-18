"""
Run All Algorithms on Classic Benchmarks with Visualization

Executes all adaptive DE algorithms on classic benchmark functions with:
- Population size: 10
- Iterations: 10
- Generates comprehensive metric visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import algorithms
from devolve import JDE, JADE, SHADE, LSHADE, LSHADEEpSin

# Import benchmarks
from devolve.benchmarks import Sphere, Rosenbrock, Rastrigin, Ackley

# Import utilities
from devolve.utils import (
    set_seed,
    calculate_stability,
    calculate_convergence_speed,
    PerformanceMetrics,
    set_publication_style
)


def run_all_algorithms():
    """Run all algorithms on all benchmarks."""
    print("=" * 70)
    print("Running All Adaptive DE Algorithms")
    print("=" * 70)
    print(f"Configuration: Population=10, Iterations=10, Dimensions=10")
    print()
    
    # Setup
    set_seed(42)
    dimensions = 10
    pop_size = 10
    max_iter = 10
    
    # Algorithms
    algorithms = {
        'jDE': JDE,
        'JADE': JADE,
        'SHADE': SHADE,
        'L-SHADE': LSHADE,
        'LSHADE-EpSin': LSHADEEpSin
    }
    
    # Benchmarks
    benchmarks = {
        'Sphere': Sphere(dimensions=dimensions),
        'Rosenbrock': Rosenbrock(dimensions=dimensions),
        'Rastrigin': Rastrigin(dimensions=dimensions),
        'Ackley': Ackley(dimensions=dimensions)
    }
    
    # Run experiments
    results = {}
    
    for bench_name, problem in benchmarks.items():
        print(f"\nBenchmark: {bench_name}")
        results[bench_name] = {}
        
        for alg_name, AlgClass in algorithms.items():
            # Configure algorithm
            optimizer = AlgClass(
                problem=problem,
                population_size=pop_size,
                max_iterations=max_iter,
                seed=42
            )
            
            # Run optimization
            best_x, best_f = optimizer.optimize()
            
            # Collect results
            history = optimizer.logger.history['best_fitness']
            fes = optimizer.function_evaluations
            
            results[bench_name][alg_name] = {
                'best_fitness': best_f,
                'best_position': best_x,
                'history': history,
                'function_evaluations': fes
            }
            
            print(f"  {alg_name:15s}: {best_f:.6e} ({fes} FEs)")
    
    return results, benchmarks


def create_convergence_plots(results, benchmarks, output_dir):
    """Create convergence plots for all algorithms on each benchmark."""
    print("\n" + "=" * 70)
    print("Creating Convergence Plots")
    print("=" * 70)
    
    set_publication_style()
    
    # Create 2x2 grid for 4 benchmarks
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']
    
    for idx, (bench_name, bench_results) in enumerate(results.items()):
        ax = axes[idx]
        
        for color_idx, (alg_name, alg_data) in enumerate(bench_results.items()):
            history = alg_data['history']
            iterations = range(len(history))
            
            ax.semilogy(iterations, history, 
                       label=alg_name, 
                       marker='o', 
                       color=colors[color_idx],
                       linewidth=2,
                       markersize=4)
        
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Best Fitness (log scale)', fontsize=11)
        ax.set_title(f'{bench_name} Function', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'convergence_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_final_fitness_comparison(results, output_dir):
    """Create bar chart comparing final fitness values."""
    print("\n" + "=" * 70)
    print("Creating Final Fitness Comparison")
    print("=" * 70)
    
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    benchmarks = list(results.keys())
    algorithms = list(next(iter(results.values())).keys())
    
    x = np.arange(len(benchmarks))
    width = 0.15
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']
    
    for idx, alg_name in enumerate(algorithms):
        fitness_values = [results[bench][alg_name]['best_fitness'] 
                         for bench in benchmarks]
        
        offset = width * (idx - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, fitness_values, width, 
               label=alg_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Benchmark Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Best Fitness (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Comparison: Final Fitness Values', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'final_fitness_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_function_evaluations_plot(results, output_dir):
    """Create plot showing function evaluations used by each algorithm."""
    print("\n" + "=" * 70)
    print("Creating Function Evaluations Plot")
    print("=" * 70)
    
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    benchmarks = list(results.keys())
    algorithms = list(next(iter(results.values())).keys())
    
    x = np.arange(len(benchmarks))
    width = 0.15
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']
    
    for idx, alg_name in enumerate(algorithms):
        fes_values = [results[bench][alg_name]['function_evaluations'] 
                     for bench in benchmarks]
        
        offset = width * (idx - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, fes_values, width, 
               label=alg_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Benchmark Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Function Evaluations', fontsize=12, fontweight='bold')
    ax.set_title('Function Evaluations Used by Each Algorithm', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'function_evaluations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_ranking_plot(results, output_dir):
    """Create ranking plot showing which algorithm performed best on each problem."""
    print("\n" + "=" * 70)
    print("Creating Algorithm Ranking Plot")
    print("=" * 70)
    
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    benchmarks = list(results.keys())
    algorithms = list(next(iter(results.values())).keys())
    
    # Calculate ranks (1=best, 5=worst)
    rank_matrix = np.zeros((len(algorithms), len(benchmarks)))
    
    for bench_idx, bench_name in enumerate(benchmarks):
        fitness_values = [(alg_name, results[bench_name][alg_name]['best_fitness'])
                         for alg_name in algorithms]
        fitness_values.sort(key=lambda x: x[1])  # Sort by fitness
        
        for rank, (alg_name, _) in enumerate(fitness_values, 1):
            alg_idx = algorithms.index(alg_name)
            rank_matrix[alg_idx, bench_idx] = rank
    
    # Create heatmap
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=5)
    
    # Set ticks
    ax.set_xticks(np.arange(len(benchmarks)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(benchmarks, fontsize=11)
    ax.set_yticklabels(algorithms, fontsize=11)
    
    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(benchmarks)):
            rank = int(rank_matrix[i, j])
            text = ax.text(j, i, rank,
                          ha="center", va="center", 
                          color="black", fontsize=12, fontweight='bold')
    
    ax.set_title('Algorithm Rankings (1=Best, 5=Worst)', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rank', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    
    output_path = output_dir / 'algorithm_rankings.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_convergence_speed_analysis(results, benchmarks, output_dir):
    """Analyze and visualize convergence speed metrics."""
    print("\n" + "=" * 70)
    print("Creating Convergence Speed Analysis")
    print("=" * 70)
    
    set_publication_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (bench_name, problem) in enumerate(benchmarks.items()):
        ax = axes[idx]
        
        algorithms = list(results[bench_name].keys())
        convergence_rates = []
        
        for alg_name in algorithms:
            history = results[bench_name][alg_name]['history']
            
            # Calculate convergence speed
            metrics = calculate_convergence_speed(
                history, 
                target_error=0.01,
                optimal_value=problem.optimum
            )
            
            convergence_rates.append(abs(metrics['linear_rate']))
        
        # Bar plot
        colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']
        bars = ax.bar(algorithms, convergence_rates, color=colors, alpha=0.8)
        
        ax.set_ylabel('Convergence Rate (abs)', fontsize=11)
        ax.set_title(f'{bench_name} - Convergence Speed', 
                    fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars, convergence_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'convergence_speed_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_error_from_optimum(results, benchmarks, output_dir):
    """Create plot showing error from known optimum."""
    print("\n" + "=" * 70)
    print("Creating Error from Optimum Plot")
    print("=" * 70)
    
    set_publication_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    benchmark_names = list(results.keys())
    algorithms = list(next(iter(results.values())).keys())
    
    x = np.arange(len(benchmark_names))
    width = 0.15
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']
    
    for idx, alg_name in enumerate(algorithms):
        errors = []
        for bench_name in benchmark_names:
            best_f = results[bench_name][alg_name]['best_fitness']
            optimum = benchmarks[bench_name].optimum
            error = abs(best_f - optimum)
            errors.append(error)
        
        offset = width * (idx - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, errors, width, 
               label=alg_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Benchmark Function', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error from Optimum (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Distance from Known Optimum', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_names, fontsize=11)
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = output_dir / 'error_from_optimum.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(results, benchmarks, output_dir):
    """Create and save summary statistics table."""
    print("\n" + "=" * 70)
    print("Creating Summary Statistics Table")
    print("=" * 70)
    
    # Prepare data
    summary_lines = []
    summary_lines.append("=" * 100)
    summary_lines.append("SUMMARY: All Algorithms on Classic Benchmarks")
    summary_lines.append("Configuration: Population=10, Iterations=10, Dimensions=10")
    summary_lines.append("=" * 100)
    summary_lines.append("")
    
    for bench_name, problem in benchmarks.items():
        summary_lines.append(f"\n{bench_name} Function (Optimum: {problem.optimum})")
        summary_lines.append("-" * 100)
        summary_lines.append(f"{'Algorithm':<15} {'Best Fitness':>15} {'Error':>15} {'FEs':>10} {'Final Iter':>12}")
        summary_lines.append("-" * 100)
        
        for alg_name, alg_data in results[bench_name].items():
            best_f = alg_data['best_fitness']
            error = abs(best_f - problem.optimum)
            fes = alg_data['function_evaluations']
            final_iter = len(alg_data['history']) - 1
            
            summary_lines.append(
                f"{alg_name:<15} {best_f:>15.6e} {error:>15.6e} {fes:>10} {final_iter:>12}"
            )
    
    summary_lines.append("\n" + "=" * 100)
    
    # Rankings
    summary_lines.append("\nALGORITHM RANKINGS (by average rank across all benchmarks)")
    summary_lines.append("=" * 100)
    
    algorithms = list(next(iter(results.values())).keys())
    avg_ranks = {alg: 0 for alg in algorithms}
    
    for bench_name in benchmarks.keys():
        fitness_values = [(alg_name, results[bench_name][alg_name]['best_fitness'])
                         for alg_name in algorithms]
        fitness_values.sort(key=lambda x: x[1])
        
        for rank, (alg_name, _) in enumerate(fitness_values, 1):
            avg_ranks[alg_name] += rank
    
    for alg in avg_ranks:
        avg_ranks[alg] /= len(benchmarks)
    
    sorted_algs = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    for rank, (alg_name, avg_rank) in enumerate(sorted_algs, 1):
        summary_lines.append(f"{rank}. {alg_name:<15} (Average Rank: {avg_rank:.2f})")
    
    summary_lines.append("=" * 100)
    
    # Print to console
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    
    # Save to file
    output_path = output_dir / 'summary_statistics.txt'
    with open(output_path, 'w') as f:
        f.write(summary_text)
    
    print(f"\n✓ Saved: {output_path}")


def create_all_in_one_dashboard(results, benchmarks, output_dir):
    """Create comprehensive dashboard with all metrics."""
    print("\n" + "=" * 70)
    print("Creating Comprehensive Dashboard")
    print("=" * 70)
    
    set_publication_style()
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#CC79A7']
    algorithms = list(next(iter(results.values())).keys())
    benchmark_names = list(results.keys())
    
    # 1. Convergence curves (top row - 4 plots)
    for idx, bench_name in enumerate(benchmark_names):
        ax = fig.add_subplot(gs[0, idx])
        
        for color_idx, alg_name in enumerate(algorithms):
            history = results[bench_name][alg_name]['history']
            ax.semilogy(history, label=alg_name, color=colors[color_idx], 
                       linewidth=2, alpha=0.8)
        
        ax.set_title(bench_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('Fitness (log)', fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')
    
    # 2. Final fitness comparison (middle left - span 2 columns)
    ax = fig.add_subplot(gs[1, :2])
    x = np.arange(len(benchmark_names))
    width = 0.15
    
    for idx, alg_name in enumerate(algorithms):
        fitness_values = [results[bench][alg_name]['best_fitness'] 
                         for bench in benchmark_names]
        offset = width * (idx - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, fitness_values, width, 
               label=alg_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Benchmark', fontsize=9)
    ax.set_ylabel('Final Fitness (log)', fontsize=9)
    ax.set_title('Final Fitness Comparison', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_names, fontsize=8)
    ax.set_yscale('log')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Function evaluations (middle right - span 2 columns)
    ax = fig.add_subplot(gs[1, 2:])
    
    for idx, alg_name in enumerate(algorithms):
        fes_values = [results[bench][alg_name]['function_evaluations'] 
                     for bench in benchmark_names]
        offset = width * (idx - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, fes_values, width, 
               label=alg_name, color=colors[idx], alpha=0.8)
    
    ax.set_xlabel('Benchmark', fontsize=9)
    ax.set_ylabel('Function Evaluations', fontsize=9)
    ax.set_title('Function Evaluations Used', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmark_names, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Rankings heatmap (bottom left - span 2 columns)
    ax = fig.add_subplot(gs[2, :2])
    
    rank_matrix = np.zeros((len(algorithms), len(benchmark_names)))
    for bench_idx, bench_name in enumerate(benchmark_names):
        fitness_values = [(alg_name, results[bench_name][alg_name]['best_fitness'])
                         for alg_name in algorithms]
        fitness_values.sort(key=lambda x: x[1])
        
        for rank, (alg_name, _) in enumerate(fitness_values, 1):
            alg_idx = algorithms.index(alg_name)
            rank_matrix[alg_idx, bench_idx] = rank
    
    im = ax.imshow(rank_matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=5)
    ax.set_xticks(np.arange(len(benchmark_names)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(benchmark_names, fontsize=8)
    ax.set_yticklabels(algorithms, fontsize=8)
    ax.set_title('Rankings (1=Best)', fontsize=10, fontweight='bold')
    
    for i in range(len(algorithms)):
        for j in range(len(benchmark_names)):
            ax.text(j, i, int(rank_matrix[i, j]),
                   ha="center", va="center", color="black", fontsize=9)
    
    # 5. Average performance bar chart (bottom center-right)
    ax = fig.add_subplot(gs[2, 2])
    
    avg_fitness = []
    for alg_name in algorithms:
        avg = np.mean([np.log10(results[bench][alg_name]['best_fitness'] + 1e-15)
                      for bench in benchmark_names])
        avg_fitness.append(avg)
    
    bars = ax.barh(algorithms, avg_fitness, color=colors, alpha=0.8)
    ax.set_xlabel('Average Log10(Fitness)', fontsize=10)
    ax.set_title('Overall Performance (lower is better)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, val in zip(bars, avg_fitness):
        ax.text(val, bar.get_y() + bar.get_height()/2., f'{val:.2f}',
               ha='left', va='center', fontsize=9, fontweight='bold')
    
    # 6. Win count (bottom right)
    ax = fig.add_subplot(gs[2, 3])
    
    win_counts = {alg: 0 for alg in algorithms}
    for bench_name in benchmark_names:
        best_alg = min(algorithms, 
                      key=lambda a: results[bench_name][a]['best_fitness'])
        win_counts[best_alg] += 1
    
    ax.pie([win_counts[alg] for alg in algorithms],
           labels=algorithms,
           colors=colors,
           autopct='%1.0f%%',
           startangle=90)
    ax.set_title('Best Performance Count', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle('Adaptive DE Algorithms: Comprehensive Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    output_path = output_dir / 'comprehensive_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("All Algorithms Benchmark with Complete Visualization")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path("algorithm_benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    # Run all experiments
    results, benchmarks = run_all_algorithms()
    
    # Generate all visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    create_convergence_plots(results, benchmarks, output_dir)
    create_final_fitness_comparison(results, output_dir)
    create_function_evaluations_plot(results, output_dir)
    create_ranking_plot(results, output_dir)
    create_convergence_speed_analysis(results, benchmarks, output_dir)
    create_error_from_optimum(results, benchmarks, output_dir)
    create_summary_table(results, benchmarks, output_dir)
    create_all_in_one_dashboard(results, benchmarks, output_dir)
    
    # Final summary
    print("\n" + "=" * 70)
    print("✅ All Analyses Complete!")
    print("=" * 70)
    print(f"\nGenerated visualizations:")
    print(f"  1. convergence_plots.png - Convergence curves for each benchmark")
    print(f"  2. final_fitness_comparison.png - Final fitness values")
    print(f"  3. function_evaluations.png - FEs used by each algorithm")
    print(f"  4. algorithm_rankings.png - Ranking heatmap")
    print(f"  5. convergence_speed_analysis.png - Convergence rate analysis")
    print(f"  6. error_from_optimum.png - Distance from known optimum")
    print(f"  7. comprehensive_dashboard.png - All-in-one dashboard")
    print(f"  8. summary_statistics.txt - Complete text summary")
    print(f"\n📁 All files saved to: {output_dir.absolute()}/")
    print()


if __name__ == "__main__":
    main()
