"""Utilities module."""

# Performance metrics
from .metrics import (
    PerformanceMetrics,
    calculate_error,
    calculate_success_rate,
    calculate_ert,
    calculate_convergence_speed,
    calculate_auc,
    calculate_stability
)

# Statistical tests
from .stats import (
    StatisticalTests,
    wilcoxon_test,
    friedman_test,
    nemenyi_posthoc_test,
    calculate_effect_size,
    mann_whitney_u_test
)

# Parallel execution
from .parallel import (
    parallel_evaluate,
    parallel_optimize,
    get_optimal_n_jobs,
    benchmark_parallel_speedup
)

# Random seed management
from .seed import (
    set_seed,
    get_seed,
    get_seed_sequence,
    ensure_reproducibility,
    create_rng,
    split_seed,
    get_reproducible_state,
    restore_reproducible_state
)

# Input/Output
from .io import (
    save_results,
    load_results,
    export_to_csv,
    export_to_latex_table,
    export_comparison_table,
    save_experiment_config,
    load_experiment_config
)

# Visualization functions
from .visualization import (
    set_publication_style,
    setup_figure_folders,
    plot_convergence,
    plot_convergence_with_ci,
    plot_algorithm_comparison,
    plot_statistical_comparison,
    OKABE_ITO,
    OKABE_ITO_COLORS
)

# Extended visualization (population, 3D, parameters)
try:
    from .visualization_extended import (
        plot_population_2d,
        animate_population_2d,
        plot_3d_landscape,
        plot_parameter_evolution,
        plot_diversity,
        calculate_diversity
    )
    EXTENDED_VIZ_AVAILABLE = True
except ImportError:
    EXTENDED_VIZ_AVAILABLE = False

# Master generation functions
try:
    from .visualization_master import (
        generate_comparison_table,
        create_comprehensive_report,
        generate_all_figures
    )
    MASTER_VIZ_AVAILABLE = True
except ImportError:
    MASTER_VIZ_AVAILABLE = False

__all__ = [
    # Metrics
    'PerformanceMetrics',
    'calculate_error',
    'calculate_success_rate',
    'calculate_ert',
    'calculate_convergence_speed',
    'calculate_auc',
    'calculate_stability',
    
    # Statistics
    'StatisticalTests',
    'wilcoxon_test',
    'friedman_test',
    'nemenyi_posthoc_test',
    'calculate_effect_size',
    'mann_whitney_u_test',
    
    # Parallel
    'parallel_evaluate',
    'parallel_optimize',
    'get_optimal_n_jobs',
    'benchmark_parallel_speedup',
    
    # Seed management
    'set_seed',
    'get_seed',
    'get_seed_sequence',
    'ensure_reproducibility',
    'create_rng',
    'split_seed',
    'get_reproducible_state',
    'restore_reproducible_state',
    
    # I/O
    'save_results',
    'load_results',
    'export_to_csv',
    'export_to_latex_table',
    'export_comparison_table',
    'save_experiment_config',
    'load_experiment_config',
    
    # Basic visualization
    'set_publication_style',
    'setup_figure_folders',
    'plot_convergence',
    'plot_convergence_with_ci',
    'plot_algorithm_comparison',
    'plot_statistical_comparison',
    'OKABE_ITO',
    'OKABE_ITO_COLORS',
]

# Add extended visualization if available
if EXTENDED_VIZ_AVAILABLE:
    __all__.extend([
        'plot_population_2d',
        'animate_population_2d',
        'plot_3d_landscape',
        'plot_parameter_evolution',
        'plot_diversity',
        'calculate_diversity'
    ])

# Add master functions if available
if MASTER_VIZ_AVAILABLE:
    __all__.extend([
        'generate_comparison_table',
        'create_comprehensive_report',
        'generate_all_figures'
    ])
