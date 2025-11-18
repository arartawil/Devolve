"""
Input/Output Utilities

Utilities for saving, loading, and exporting optimization results in various formats.
"""

import json
import pickle
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = 'json'
) -> None:
    """
    Save optimization results to file.
    
    Supports JSON and pickle formats. JSON is human-readable but may lose
    precision for floating-point numbers. Pickle preserves exact values.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing results. Common keys:
        - 'best_fitness': float
        - 'best_position': np.ndarray
        - 'history': dict of lists
        - 'algorithm': str
        - 'problem': str
        - 'parameters': dict
    filepath : str or Path
        Path to save file.
    format : str, optional
        Format to use: 'json' or 'pickle'. Default is 'json'.
    
    Examples
    --------
    >>> results = {
    ...     'best_fitness': 0.001,
    ...     'best_position': np.array([0.1, 0.2, 0.3]),
    ...     'algorithm': 'JADE',
    ...     'problem': 'Sphere',
    ...     'history': {'iteration': [0, 1, 2], 'fitness': [10.0, 1.0, 0.1]}
    ... }
    >>> save_results(results, 'results.json')
    
    See Also
    --------
    load_results : Load saved results
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = _make_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    elif format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'pickle'.")


def load_results(
    filepath: Union[str, Path],
    format: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load optimization results from file.
    
    Auto-detects format from file extension if not specified.
    
    Parameters
    ----------
    filepath : str or Path
        Path to results file.
    format : str, optional
        Format: 'json' or 'pickle'. If None, inferred from extension.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded results.
    
    Examples
    --------
    >>> results = load_results('results.json')
    >>> print(f"Best fitness: {results['best_fitness']}")
    
    See Also
    --------
    save_results : Save results to file
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    # Auto-detect format from extension
    if format is None:
        suffix = filepath.suffix.lower()
        if suffix == '.json':
            format = 'json'
        elif suffix in ['.pkl', '.pickle']:
            format = 'pickle'
        else:
            raise ValueError(f"Cannot infer format from extension: {suffix}")
    
    if format == 'json':
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        results = _restore_numpy_arrays(results)
        return results
    
    elif format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    else:
        raise ValueError(f"Unknown format: {format}")


def export_to_csv(
    results: Union[Dict[str, List], List[Dict]],
    filepath: Union[str, Path],
    delimiter: str = ','
) -> None:
    """
    Export results to CSV file.
    
    Parameters
    ----------
    results : Dict[str, List] or List[Dict]
        Results to export. Either:
        - Dict mapping column names to value lists (all same length)
        - List of dicts (one dict per row)
    filepath : str or Path
        Path to CSV file.
    delimiter : str, optional
        Column delimiter. Default is ','.
    
    Examples
    --------
    >>> # Export comparison table
    >>> results = {
    ...     'Algorithm': ['JADE', 'SHADE', 'L-SHADE'],
    ...     'Mean': [0.001, 0.002, 0.0015],
    ...     'Std': [0.0001, 0.0002, 0.00015]
    ... }
    >>> export_to_csv(results, 'comparison.csv')
    
    >>> # Export history
    >>> history = [
    ...     {'iteration': 0, 'fitness': 10.0},
    ...     {'iteration': 1, 'fitness': 1.0},
    ...     {'iteration': 2, 'fitness': 0.1}
    ... ]
    >>> export_to_csv(history, 'history.csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        if isinstance(results, dict):
            # Dict of lists format
            writer = csv.writer(f, delimiter=delimiter)
            
            # Write header
            headers = list(results.keys())
            writer.writerow(headers)
            
            # Write rows
            n_rows = len(next(iter(results.values())))
            for i in range(n_rows):
                row = [results[key][i] for key in headers]
                writer.writerow(row)
        
        elif isinstance(results, list):
            # List of dicts format
            if not results:
                return
            
            writer = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=delimiter)
            writer.writeheader()
            writer.writerows(results)
        
        else:
            raise ValueError("Results must be Dict[str, List] or List[Dict]")


def export_to_latex_table(
    results: Dict[str, List],
    filepath: Union[str, Path],
    caption: str = "Results",
    label: str = "tab:results",
    format_spec: str = ".4f",
    bold_best: bool = True
) -> None:
    """
    Export results to LaTeX table format.
    
    Parameters
    ----------
    results : Dict[str, List]
        Dictionary mapping column names to value lists.
    filepath : str or Path
        Path to .tex file.
    caption : str, optional
        Table caption. Default is "Results".
    label : str, optional
        LaTeX label for referencing. Default is "tab:results".
    format_spec : str, optional
        Number format specification. Default is ".4f".
    bold_best : bool, optional
        Bold the best (minimum) value in each numeric column. Default is True.
    
    Examples
    --------
    >>> results = {
    ...     'Algorithm': ['JADE', 'SHADE', 'L-SHADE'],
    ...     'Mean': [0.0012, 0.0023, 0.0015],
    ...     'Std': [0.0001, 0.0002, 0.00015]
    ... }
    >>> export_to_latex_table(
    ...     results,
    ...     'comparison_table.tex',
    ...     caption='Algorithm Comparison on Sphere Function',
    ...     label='tab:sphere_comparison'
    ... )
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Find best values in numeric columns
    best_indices = {}
    if bold_best:
        for col_name, values in results.items():
            if col_name == 'Algorithm' or not isinstance(values[0], (int, float)):
                continue
            try:
                numeric_values = [float(v) for v in values]
                best_idx = np.argmin(numeric_values)
                best_indices[col_name] = best_idx
            except (ValueError, TypeError):
                pass
    
    # Generate LaTeX
    n_cols = len(results)
    n_rows = len(next(iter(results.values())))
    
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{'l' + 'r' * (n_cols - 1)}}}")
    lines.append("\\toprule")
    
    # Header
    headers = list(results.keys())
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    
    # Rows
    for i in range(n_rows):
        row_values = []
        for col_name in headers:
            value = results[col_name][i]
            
            # Format value
            if isinstance(value, (int, float)):
                formatted = f"{value:{format_spec}}"
            else:
                formatted = str(value)
            
            # Bold if best
            if col_name in best_indices and best_indices[col_name] == i:
                formatted = f"\\textbf{{{formatted}}}"
            
            row_values.append(formatted)
        
        lines.append(" & ".join(row_values) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def export_comparison_table(
    algorithm_results: Dict[str, List[float]],
    filepath: Union[str, Path],
    format: str = 'csv',
    include_statistics: bool = True
) -> None:
    """
    Export algorithm comparison table with statistics.
    
    Parameters
    ----------
    algorithm_results : Dict[str, List[float]]
        Dictionary mapping algorithm names to lists of fitness values from runs.
    filepath : str or Path
        Output file path.
    format : str, optional
        Format: 'csv' or 'latex'. Default is 'csv'.
    include_statistics : bool, optional
        Include mean, std, min, max columns. Default is True.
    
    Examples
    --------
    >>> results = {
    ...     'JADE': [0.001, 0.002, 0.0015, 0.0018],
    ...     'SHADE': [0.002, 0.003, 0.0025, 0.0028],
    ...     'L-SHADE': [0.0015, 0.0025, 0.002, 0.0023]
    ... }
    >>> export_comparison_table(results, 'comparison.csv')
    """
    # Calculate statistics
    table_data = {'Algorithm': []}
    
    if include_statistics:
        table_data.update({
            'Mean': [],
            'Std': [],
            'Min': [],
            'Max': [],
            'Median': []
        })
    
    for alg_name, values in algorithm_results.items():
        table_data['Algorithm'].append(alg_name)
        
        if include_statistics:
            values_array = np.array(values)
            table_data['Mean'].append(float(np.mean(values_array)))
            table_data['Std'].append(float(np.std(values_array)))
            table_data['Min'].append(float(np.min(values_array)))
            table_data['Max'].append(float(np.max(values_array)))
            table_data['Median'].append(float(np.median(values_array)))
    
    # Export
    if format == 'csv':
        export_to_csv(table_data, filepath)
    elif format == 'latex':
        export_to_latex_table(
            table_data,
            filepath,
            caption='Algorithm Comparison',
            label='tab:algorithm_comparison',
            format_spec='.6e'
        )
    else:
        raise ValueError(f"Unknown format: {format}")


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def _restore_numpy_arrays(obj: Any) -> Any:
    """Attempt to restore numpy arrays from lists."""
    if isinstance(obj, dict):
        # Check for special keys that should be arrays
        array_keys = ['best_position', 'position', 'individuals', 'population']
        
        result = {}
        for key, value in obj.items():
            if key in array_keys and isinstance(value, list):
                result[key] = np.array(value)
            else:
                result[key] = _restore_numpy_arrays(value)
        return result
    
    elif isinstance(obj, list):
        return [_restore_numpy_arrays(item) for item in obj]
    
    else:
        return obj


def save_experiment_config(
    config: Dict[str, Any],
    filepath: Union[str, Path]
) -> None:
    """
    Save experiment configuration to JSON file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary with experiment parameters.
    filepath : str or Path
        Path to config file.
    
    Examples
    --------
    >>> config = {
    ...     'algorithm': 'JADE',
    ...     'problem': 'Sphere',
    ...     'dimensions': 10,
    ...     'population_size': 50,
    ...     'max_iterations': 100,
    ...     'n_runs': 30,
    ...     'seed': 42
    ... }
    >>> save_experiment_config(config, 'experiment_config.json')
    """
    save_results(config, filepath, format='json')


def load_experiment_config(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment configuration from JSON file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to config file.
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    
    Examples
    --------
    >>> config = load_experiment_config('experiment_config.json')
    >>> print(f"Algorithm: {config['algorithm']}")
    """
    return load_results(filepath, format='json')
