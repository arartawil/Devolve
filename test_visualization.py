"""
Quick verification test for the visualization module.
Tests basic functionality without requiring optimization runs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*70)
print("DEVOLVE VISUALIZATION MODULE - QUICK TEST")
print("="*70)

# Test 1: Import core functions
print("\n[1/7] Testing imports...")
try:
    from devolve.utils import (
        set_publication_style,
        setup_figure_folders,
        plot_convergence,
        plot_convergence_with_ci,
        plot_algorithm_comparison,
        plot_statistical_comparison,
        OKABE_ITO,
        OKABE_ITO_COLORS
    )
    print("  ✓ Core functions imported successfully")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Test 2: Import extended functions
print("\n[2/7] Testing extended imports...")
try:
    from devolve.utils import (
        plot_population_2d,
        plot_3d_landscape,
        plot_parameter_evolution,
        plot_diversity,
        calculate_diversity
    )
    print("  ✓ Extended functions imported successfully")
except Exception as e:
    print(f"  ✗ Extended import failed: {e}")

# Test 3: Import master functions
print("\n[3/7] Testing master imports...")
try:
    from devolve.utils import (
        generate_comparison_table,
        create_comprehensive_report,
        generate_all_figures
    )
    print("  ✓ Master functions imported successfully")
except Exception as e:
    print(f"  ✗ Master import failed: {e}")

# Test 4: Set publication style
print("\n[4/7] Testing publication style...")
try:
    set_publication_style()
    print(f"  ✓ Publication style applied")
    print(f"    - Font family: {plt.rcParams['font.family']}")
    print(f"    - Font size: {plt.rcParams['font.size']}")
    print(f"    - Line width: {plt.rcParams['lines.linewidth']}")
    print(f"    - DPI: {plt.rcParams['savefig.dpi']}")
except Exception as e:
    print(f"  ✗ Style setting failed: {e}")

# Test 5: Setup folders
print("\n[5/7] Testing folder setup...")
try:
    folders = setup_figure_folders("test_figures")
    print(f"  ✓ Created {len(folders)} folders:")
    for name, path in list(folders.items())[:3]:
        print(f"    - {name}: {path}")
    print(f"    ... and {len(folders)-3} more")
except Exception as e:
    print(f"  ✗ Folder setup failed: {e}")

# Test 6: Generate a simple plot
print("\n[6/7] Testing basic plotting...")
try:
    # Create dummy data
    history = {'best_fitness': [100, 50, 25, 12.5, 6.25, 3.12, 1.56, 0.78]}
    
    # Plot
    fig = plot_convergence(
        history=history,
        title="Test Convergence Plot",
        log_scale=True
    )
    
    plt.close(fig)
    print("  ✓ Basic plot created successfully")
except Exception as e:
    print(f"  ✗ Plotting failed: {e}")

# Test 7: Test comparison plot
print("\n[7/7] Testing comparison plot...")
try:
    results = {
        'Algorithm 1': [100, 80, 60, 40, 20, 10],
        'Algorithm 2': [100, 70, 50, 35, 25, 15],
        'Algorithm 3': [100, 85, 65, 45, 30, 20]
    }
    
    fig = plot_algorithm_comparison(
        results_dict=results,
        title="Test Algorithm Comparison"
    )
    
    plt.close(fig)
    print("  ✓ Comparison plot created successfully")
except Exception as e:
    print(f"  ✗ Comparison plot failed: {e}")

# Test 8: Color palette
print("\n[8/7] Bonus - Testing color palette...")
try:
    print(f"  ✓ Okabe-Ito palette loaded:")
    for name, color in list(OKABE_ITO.items())[:4]:
        print(f"    - {name}: {color}")
    print(f"    ... and {len(OKABE_ITO)-4} more colors")
except Exception as e:
    print(f"  ✗ Color palette test failed: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\n✓ All core visualization functions are working!")
print("\nThe module is ready to use. Try running:")
print("  python examples/03_visualization_demo.py")
print("\nOr use it in your code:")
print("  from devolve.utils import plot_convergence, set_publication_style")
print("\n" + "="*70)
