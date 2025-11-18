"""
Comprehensive test suite for DEvolve utilities.

Tests all utility modules:
- metrics.py
- stats.py (statistics.py)
- parallel.py
- seed.py
- io.py
"""

import numpy as np
import tempfile
import os
from pathlib import Path

# Test metrics
from devolve.utils.metrics import (
    calculate_error,
    calculate_success_rate,
    calculate_ert,
    calculate_convergence_speed,
    calculate_auc,
    calculate_stability,
    PerformanceMetrics
)

# Test statistics
from devolve.utils.stats import (
    wilcoxon_test,
    friedman_test,
    nemenyi_posthoc_test,
    calculate_effect_size,
    mann_whitney_u_test,
    StatisticalTests
)

# Test parallel
from devolve.utils.parallel import (
    parallel_evaluate,
    get_optimal_n_jobs
)

# Test seed management
from devolve.utils.seed import (
    set_seed,
    get_seed,
    get_seed_sequence,
    ensure_reproducibility,
    create_rng,
    split_seed
)

# Test I/O
from devolve.utils.io import (
    save_results,
    load_results,
    export_to_csv,
    export_to_latex_table,
    export_comparison_table
)


def test_metrics():
    """Test performance metrics functions."""
    print("=" * 70)
    print("Testing Performance Metrics")
    print("=" * 70)
    
    # Test calculate_error
    error = calculate_error(0.001, 0.0)
    print(f"\n✓ calculate_error: {error:.6f}")
    assert error == 0.001
    
    # Test success rate
    runs = [0.001, 0.005, 0.1, 0.002]
    sr = calculate_success_rate(runs, target_error=0.01, optimal_value=0.0)
    print(f"✓ calculate_success_rate: {sr:.2%} (expected 75%)")
    assert sr == 0.75
    
    # Test ERT
    runs_with_fes = [(0.001, 1000), (0.005, 1500), (0.1, 2000), (0.002, 1200)]
    ert = calculate_ert(runs_with_fes, target_error=0.01)
    print(f"✓ calculate_ert: {ert:.0f} FEs")
    assert ert is not None
    
    # Test convergence speed
    history = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]
    metrics = calculate_convergence_speed(history, target_error=0.01)
    print(f"✓ calculate_convergence_speed:")
    print(f"  - Iterations to target: {metrics['iterations_to_target']}")
    print(f"  - Final error: {metrics['final_error']:.6e}")
    assert metrics['iterations_to_target'] == 4
    
    # Test AUC
    auc = calculate_auc(history, normalize=True)
    print(f"✓ calculate_auc: {auc:.2f}")
    assert auc > 0
    
    # Test stability
    stability = calculate_stability(runs)
    print(f"✓ calculate_stability:")
    print(f"  - Mean: {stability['mean']:.6f}")
    print(f"  - Std: {stability['std']:.6f}")
    print(f"  - CV: {stability['cv']:.4f}")
    
    # Test PerformanceMetrics class
    pm = PerformanceMetrics()
    error2 = pm.error(0.005, 0.0)
    print(f"✓ PerformanceMetrics class: {error2:.6f}")
    
    print("\n✅ All metrics tests passed!")


def test_statistics():
    """Test statistical functions."""
    print("\n" + "=" * 70)
    print("Testing Statistical Tests")
    print("=" * 70)
    
    # Test Wilcoxon
    alg1 = [0.01, 0.02, 0.015, 0.018, 0.012]
    alg2 = [0.05, 0.06, 0.055, 0.058, 0.052]
    
    result = wilcoxon_test(alg1, alg2)
    print(f"\n✓ wilcoxon_test:")
    print(f"  - Statistic: {result['statistic']:.2f}")
    print(f"  - P-value: {result['p_value']:.6f}")
    print(f"  - Significant: {result['significant']}")
    print(f"  - {result['interpretation']}")
    
    # Test Friedman
    results = {
        'DE': [0.01, 0.05, 0.02, 0.03],
        'PSO': [0.02, 0.06, 0.03, 0.04],
        'GA': [0.03, 0.07, 0.04, 0.05]
    }
    
    result = friedman_test(results)
    print(f"\n✓ friedman_test:")
    print(f"  - Statistic: {result['statistic']:.4f}")
    print(f"  - P-value: {result['p_value']:.6f}")
    print(f"  - Mean ranks: {result['mean_ranks']}")
    
    # Test Nemenyi
    result = nemenyi_posthoc_test(results)
    print(f"\n✓ nemenyi_posthoc_test:")
    print(f"  - Critical difference: {result['critical_difference']:.3f}")
    print(f"  - Significant pairs: {result['significant_pairs']}")
    
    # Test effect size
    es = calculate_effect_size(alg1, alg2, method='cohen')
    print(f"\n✓ calculate_effect_size:")
    print(f"  - Cohen's d: {es['effect_size']:.3f}")
    print(f"  - Magnitude: {es['magnitude']}")
    
    # Test Mann-Whitney
    result = mann_whitney_u_test(alg1, alg2)
    print(f"\n✓ mann_whitney_u_test:")
    print(f"  - {result['interpretation']}")
    
    # Test StatisticalTests class
    tests = StatisticalTests()
    result = tests.wilcoxon(alg1, alg2)
    print(f"\n✓ StatisticalTests class works")
    
    print("\n✅ All statistics tests passed!")


def test_parallel():
    """Test parallel evaluation."""
    print("\n" + "=" * 70)
    print("Testing Parallel Execution")
    print("=" * 70)
    
    # Define test function
    def sphere(x):
        return np.sum(x**2)
    
    # Create population
    np.random.seed(42)
    population = np.random.randn(20, 5)
    
    # Test sequential
    fitness_seq = parallel_evaluate(population, sphere, n_jobs=1, use_tqdm=False)
    print(f"\n✓ Sequential evaluation: {len(fitness_seq)} individuals")
    print(f"  - Best fitness: {np.min(fitness_seq):.6f}")
    
    # Test parallel (2 jobs)
    fitness_par = parallel_evaluate(population, sphere, n_jobs=2, use_tqdm=False)
    print(f"✓ Parallel evaluation (2 jobs): {len(fitness_par)} individuals")
    print(f"  - Best fitness: {np.min(fitness_par):.6f}")
    
    # Verify same results
    assert np.allclose(fitness_seq, fitness_par), "Parallel results differ!"
    print("✓ Parallel results match sequential")
    
    # Test optimal n_jobs
    n_jobs = get_optimal_n_jobs(population_size=100, function_cost='medium')
    print(f"\n✓ get_optimal_n_jobs: {n_jobs} (for 100 individuals, medium cost)")
    
    print("\n✅ All parallel tests passed!")


def test_seed_management():
    """Test seed management functions."""
    print("\n" + "=" * 70)
    print("Testing Seed Management")
    print("=" * 70)
    
    # Test set_seed
    set_seed(42)
    x1 = np.random.randn(5)
    set_seed(42)
    x2 = np.random.randn(5)
    print(f"\n✓ set_seed: Reproducibility verified")
    assert np.allclose(x1, x2)
    
    # Test get_seed
    seed = get_seed()
    print(f"✓ get_seed: {seed}")
    assert seed == 42
    
    # Test get_seed_sequence
    seeds = get_seed_sequence(5, base_seed=100)
    print(f"✓ get_seed_sequence: {seeds}")
    assert len(seeds) == 5
    assert seeds == [100, 101, 102, 103, 104]
    
    # Test ensure_reproducibility
    info = ensure_reproducibility(seed=42)
    print(f"✓ ensure_reproducibility:")
    print(f"  - Seed: {info['seed']}")
    print(f"  - NumPy version: {info['numpy_version']}")
    
    # Test create_rng
    rng = create_rng(seed=42)
    x = rng.standard_normal(5)
    print(f"✓ create_rng: Generated {len(x)} random numbers")
    
    # Test split_seed
    split_seeds = split_seed(42, 3)
    print(f"✓ split_seed: {len(split_seeds)} seeds generated")
    assert len(split_seeds) == 3
    
    print("\n✅ All seed management tests passed!")


def test_io():
    """Test input/output functions."""
    print("\n" + "=" * 70)
    print("Testing I/O Functions")
    print("=" * 70)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test save/load results (JSON)
        results = {
            'best_fitness': 0.001,
            'best_position': np.array([0.1, 0.2, 0.3]),
            'algorithm': 'JADE',
            'problem': 'Sphere',
            'history': {'iteration': [0, 1, 2], 'fitness': [10.0, 1.0, 0.1]}
        }
        
        json_path = tmpdir / 'results.json'
        save_results(results, json_path, format='json')
        print(f"\n✓ save_results (JSON): {json_path}")
        assert json_path.exists()
        
        loaded = load_results(json_path)
        print(f"✓ load_results (JSON): Loaded {len(loaded)} keys")
        assert loaded['best_fitness'] == 0.001
        assert loaded['algorithm'] == 'JADE'
        
        # Test save/load results (Pickle)
        pkl_path = tmpdir / 'results.pkl'
        save_results(results, pkl_path, format='pickle')
        print(f"✓ save_results (Pickle): {pkl_path}")
        
        loaded = load_results(pkl_path)
        print(f"✓ load_results (Pickle): Loaded successfully")
        assert np.allclose(loaded['best_position'], results['best_position'])
        
        # Test export_to_csv
        csv_data = {
            'Algorithm': ['JADE', 'SHADE', 'L-SHADE'],
            'Mean': [0.001, 0.002, 0.0015],
            'Std': [0.0001, 0.0002, 0.00015]
        }
        
        csv_path = tmpdir / 'comparison.csv'
        export_to_csv(csv_data, csv_path)
        print(f"✓ export_to_csv: {csv_path}")
        assert csv_path.exists()
        
        # Test export_to_latex_table
        latex_path = tmpdir / 'table.tex'
        export_to_latex_table(
            csv_data,
            latex_path,
            caption='Test Table',
            label='tab:test'
        )
        print(f"✓ export_to_latex_table: {latex_path}")
        assert latex_path.exists()
        
        # Check LaTeX content
        with open(latex_path, 'r') as f:
            content = f.read()
            assert '\\begin{table}' in content
            assert 'Test Table' in content
        
        # Test export_comparison_table
        alg_results = {
            'JADE': [0.001, 0.002, 0.0015],
            'SHADE': [0.002, 0.003, 0.0025]
        }
        
        comp_path = tmpdir / 'comparison_stats.csv'
        export_comparison_table(alg_results, comp_path, format='csv')
        print(f"✓ export_comparison_table: {comp_path}")
        assert comp_path.exists()
        
    print("\n✅ All I/O tests passed!")


def test_integration():
    """Integration test using multiple utilities together."""
    print("\n" + "=" * 70)
    print("Integration Test: Complete Workflow")
    print("=" * 70)
    
    # Set reproducibility
    set_seed(42)
    print("\n✓ Set seed for reproducibility")
    
    # Simulate algorithm comparison
    np.random.seed(42)
    jade_runs = np.random.uniform(0.001, 0.01, 10)
    shade_runs = np.random.uniform(0.002, 0.015, 10)
    
    # Calculate metrics
    jade_stability = calculate_stability(jade_runs.tolist())
    shade_stability = calculate_stability(shade_runs.tolist())
    
    print(f"\n✓ JADE stability: Mean={jade_stability['mean']:.6f}, Std={jade_stability['std']:.6f}")
    print(f"✓ SHADE stability: Mean={shade_stability['mean']:.6f}, Std={shade_stability['std']:.6f}")
    
    # Statistical comparison
    stat_result = wilcoxon_test(jade_runs.tolist(), shade_runs.tolist())
    print(f"\n✓ Statistical comparison: p-value={stat_result['p_value']:.4f}")
    
    # Effect size
    effect = calculate_effect_size(jade_runs.tolist(), shade_runs.tolist())
    print(f"✓ Effect size: {effect['effect_size']:.3f} ({effect['magnitude']})")
    
    # Export results
    with tempfile.TemporaryDirectory() as tmpdir:
        results = {
            'algorithm': 'JADE',
            'mean_fitness': jade_stability['mean'],
            'std_fitness': jade_stability['std'],
            'runs': jade_runs.tolist()
        }
        
        output_path = Path(tmpdir) / 'integration_results.json'
        save_results(results, output_path)
        print(f"\n✓ Saved results to {output_path.name}")
        
        # Load and verify
        loaded = load_results(output_path)
        assert loaded['algorithm'] == 'JADE'
        print(f"✓ Loaded and verified results")
    
    print("\n✅ Integration test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("DEvolve Utilities Test Suite")
    print("=" * 70)
    
    try:
        test_metrics()
        test_statistics()
        test_parallel()
        test_seed_management()
        test_io()
        test_integration()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nUtilities tested:")
        print("  ✓ metrics.py - Performance metrics")
        print("  ✓ stats.py - Statistical tests")
        print("  ✓ parallel.py - Parallel execution")
        print("  ✓ seed.py - Reproducibility")
        print("  ✓ io.py - Save/load/export")
        print("  ✓ Integration - Combined workflow")
        print()
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    main()
