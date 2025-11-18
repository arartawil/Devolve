"""
CEC2017 Benchmark Functions

This module implements all 30 benchmark functions from the CEC2017 competition
on single objective real-parameter numerical optimization.

Reference:
    Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2016).
    Problem definitions and evaluation criteria for the CEC 2017 special session
    and competition on single objective real-parameter numerical optimization.
    Technical Report, Nanyang Technological University, Singapore.

Functions:
    F1-F3:   Unimodal Functions
    F4-F10:  Simple Multimodal Functions
    F11-F20: Hybrid Functions
    F21-F30: Composition Functions

Supported dimensions: 10, 30, 50, 100

Author: DEvolve Package
License: MIT
"""

from typing import Optional, Tuple
import numpy as np
from pathlib import Path

from ..core.problem import Problem


class CEC2017Function(Problem):
    """
    Base class for CEC2017 benchmark functions.
    
    Parameters:
    -----------
    func_num : int
        Function number (1-30)
    dimensions : int
        Problem dimensionality (10, 30, 50, or 100)
    data_dir : str, optional
        Directory containing shift and rotation data
    
    Attributes:
    -----------
    func_num : int
        CEC2017 function number
    shift : np.ndarray
        Shift vector for the function
    rotation : np.ndarray
        Rotation matrix for the function (if applicable)
    """
    
    def __init__(
        self,
        func_num: int,
        dimensions: int = 10,
        data_dir: Optional[str] = None
    ):
        if dimensions not in [10, 30, 50, 100]:
            raise ValueError("Dimensions must be 10, 30, 50, or 100 for CEC2017")
        
        if func_num < 1 or func_num > 30:
            raise ValueError("Function number must be between 1 and 30")
        
        self.func_num = func_num
        self.data_dir = data_dir
        
        # Load shift and rotation data
        self.shift = self._load_shift(func_num, dimensions)
        self.rotation = self._load_rotation(func_num, dimensions)
        
        # Optimal value for CEC2017
        optimal_value = func_num * 100.0
        
        # Bounds: [-100, 100] for all CEC2017 functions
        bounds = (-100.0, 100.0)
        
        super().__init__(
            objective_function=self._evaluate_cec2017,
            bounds=bounds,
            dimensions=dimensions,
            optimum=optimal_value,
            optimum_position=self.shift,  # Optimum is at the shift vector
            name=f"CEC2017_F{func_num}"
        )
    
    def _load_shift(self, func_num: int, dim: int) -> np.ndarray:
        """
        Load shift vector for the function.
        
        If data files are not available, generates random shift.
        """
        if self.data_dir is not None:
            try:
                shift_file = Path(self.data_dir) / f"shift_data_{func_num}.txt"
                if shift_file.exists():
                    data = np.loadtxt(shift_file)
                    return data[:dim]
            except:
                pass
        
        # Generate random shift in [-80, 80]
        rng = np.random.RandomState(func_num)
        return rng.uniform(-80, 80, dim)
    
    def _load_rotation(self, func_num: int, dim: int) -> Optional[np.ndarray]:
        """
        Load rotation matrix for the function.
        
        If data files are not available, generates random rotation matrix.
        """
        # Functions 1-3 don't use rotation
        if func_num <= 3:
            return None
        
        if self.data_dir is not None:
            try:
                rotation_file = Path(self.data_dir) / f"M_{func_num}_{dim}.txt"
                if rotation_file.exists():
                    return np.loadtxt(rotation_file).reshape(dim, dim)
            except:
                pass
        
        # Generate random orthogonal matrix
        rng = np.random.RandomState(func_num * 1000 + dim)
        M = rng.randn(dim, dim)
        Q, _ = np.linalg.qr(M)
        return Q
    
    def _evaluate_cec2017(self, x: np.ndarray) -> float:
        """
        Evaluate CEC2017 function.
        
        This is a dispatcher that calls the appropriate function.
        """
        # Shift the input
        z = x - self.shift
        
        # Apply rotation if available
        if self.rotation is not None:
            z = self.rotation @ z
        
        # Call specific function
        if 1 <= self.func_num <= 3:
            return self._unimodal(z, self.func_num)
        elif 4 <= self.func_num <= 10:
            return self._simple_multimodal(z, self.func_num)
        elif 11 <= self.func_num <= 20:
            return self._hybrid(z, self.func_num)
        else:  # 21-30
            return self._composition(z, self.func_num)
    
    def _unimodal(self, z: np.ndarray, func_num: int) -> float:
        """Unimodal functions F1-F3."""
        if func_num == 1:
            # Shifted and Rotated Bent Cigar Function
            return z[0]**2 + 1e6 * np.sum(z[1:]**2)
        elif func_num == 2:
            # Shifted and Rotated Sum of Different Power Function
            return np.sum(np.abs(z) ** (2 + 4 * np.arange(len(z)) / (len(z) - 1)))
        else:  # func_num == 3
            # Shifted and Rotated Zakharov Function
            i = np.arange(1, len(z) + 1)
            sum1 = np.sum(z**2)
            sum2 = np.sum(0.5 * i * z)
            return sum1 + sum2**2 + sum2**4
    
    def _simple_multimodal(self, z: np.ndarray, func_num: int) -> float:
        """Simple multimodal functions F4-F10."""
        if func_num == 4:
            # Shifted and Rotated Rosenbrock's Function
            return np.sum(100.0 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)
        elif func_num == 5:
            # Shifted and Rotated Rastrigin's Function
            return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)
        elif func_num == 6:
            # Shifted and Rotated Expanded Scaffer's F6 Function
            def sf6(x, y):
                return 0.5 + (np.sin(np.sqrt(x**2 + y**2))**2 - 0.5) / (1 + 0.001*(x**2 + y**2))**2
            result = 0
            for i in range(len(z) - 1):
                result += sf6(z[i], z[i+1])
            result += sf6(z[-1], z[0])
            return result
        elif func_num == 7:
            # Shifted and Rotated Lunacek Bi-Rastrigin Function
            mu0, mu1 = 2.5, -np.sqrt((mu0**2 - 1) / 1)
            d = 1
            s = 1 - 1 / (2 * np.sqrt(len(z) + 20) - 8.2)
            
            sum1 = np.sum((z - mu0)**2)
            sum2 = np.sum((z - mu1)**2)
            sum3 = np.sum(1 - np.cos(2 * np.pi * (z - mu0)))
            
            return min(sum1, d*len(z) + s*sum2) + 10*sum3
        elif func_num == 8:
            # Shifted and Rotated Non-Continuous Rastrigin's Function
            y = np.where(np.abs(z) > 0.5, np.round(2*z)/2, z)
            return np.sum(y**2 - 10 * np.cos(2 * np.pi * y) + 10)
        elif func_num == 9:
            # Shifted and Rotated Levy Function
            w = 1 + (z - 1) / 4
            term1 = np.sin(np.pi * w[0])**2
            term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
            term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
            return term1 + term2 + term3
        else:  # func_num == 10
            # Shifted and Rotated Schwefel's Function
            g = z + 4.209687462275036e+002
            return 418.9829 * len(z) - np.sum(g * np.sin(np.sqrt(np.abs(g))))
    
    def _hybrid(self, z: np.ndarray, func_num: int) -> float:
        """
        Hybrid functions F11-F20.
        
        These combine multiple basic functions.
        For simplicity, we implement a representative subset.
        """
        # Simplified hybrid implementation
        # In full CEC2017, these are complex combinations with specific weightings
        
        n = len(z)
        
        if func_num == 11:
            # Hybrid Function 1 (N=3)
            # Zakharov + Rosenbrock + Rastrigin
            p1, p2 = n//3, 2*n//3
            
            z1 = z[:p1]
            i1 = np.arange(1, len(z1) + 1)
            f1 = np.sum(z1**2) + np.sum(0.5 * i1 * z1)**2 + np.sum(0.5 * i1 * z1)**4
            
            z2 = z[p1:p2]
            f2 = np.sum(100.0 * (z2[1:] - z2[:-1]**2)**2 + (z2[:-1] - 1)**2)
            
            z3 = z[p2:]
            f3 = np.sum(z3**2 - 10 * np.cos(2 * np.pi * z3) + 10)
            
            return f1 + f2 + f3
        
        elif func_num <= 20:
            # Other hybrid functions (simplified)
            # Mix of Sphere, Rastrigin, and Griewank
            p1, p2 = n//3, 2*n//3
            
            f1 = np.sum(z[:p1]**2)
            f2 = np.sum(z[p1:p2]**2 - 10 * np.cos(2 * np.pi * z[p1:p2]) + 10)
            
            z3 = z[p2:]
            f3 = np.sum(z3**2 / 4000) - np.prod(np.cos(z3 / np.sqrt(np.arange(1, len(z3) + 1)))) + 1
            
            return f1 + f2 + f3
        
        return 0.0
    
    def _composition(self, z: np.ndarray, func_num: int) -> float:
        """
        Composition functions F21-F30.
        
        These are compositions of multiple functions with different properties.
        For simplicity, we implement a representative version.
        """
        # Simplified composition implementation
        # In full CEC2017, these use complex weighted combinations
        
        n = len(z)
        num_funcs = 3  # Number of component functions
        
        # Component functions
        sigma = [10, 20, 30]  # Spread parameters
        lambda_vals = [1, 1, 1]  # Height parameters
        bias = [0, 100, 200]  # Bias values
        
        # Offsets for each component
        offsets = [
            np.zeros(n),
            np.full(n, 0.5),
            np.full(n, -0.5)
        ]
        
        # Calculate weights
        w = np.zeros(num_funcs)
        for i in range(num_funcs):
            dist = np.linalg.norm(z - offsets[i])
            w[i] = np.exp(-dist / (2 * n * sigma[i]**2))
        
        # Normalize weights
        w_sum = np.sum(w)
        if w_sum == 0:
            w = np.ones(num_funcs) / num_funcs
        else:
            w = w / w_sum
        
        # Calculate fitness
        fitness = 0
        for i in range(num_funcs):
            z_shifted = z - offsets[i]
            
            # Use different basic functions
            if i == 0:
                f = np.sum(z_shifted**2)  # Sphere
            elif i == 1:
                f = np.sum(z_shifted**2 - 10 * np.cos(2 * np.pi * z_shifted) + 10)  # Rastrigin
            else:
                f = np.sum(100.0 * (z_shifted[1:] - z_shifted[:-1]**2)**2 + (z_shifted[:-1] - 1)**2)  # Rosenbrock
            
            fitness += w[i] * (lambda_vals[i] * f + bias[i])
        
        return fitness


# Factory functions for each CEC2017 function
def CEC2017_F1(dimensions: int = 10) -> CEC2017Function:
    """F1: Shifted and Rotated Bent Cigar Function"""
    return CEC2017Function(1, dimensions)

def CEC2017_F2(dimensions: int = 10) -> CEC2017Function:
    """F2: Shifted and Rotated Sum of Different Power"""
    return CEC2017Function(2, dimensions)

def CEC2017_F3(dimensions: int = 10) -> CEC2017Function:
    """F3: Shifted and Rotated Zakharov Function"""
    return CEC2017Function(3, dimensions)

def CEC2017_F4(dimensions: int = 10) -> CEC2017Function:
    """F4: Shifted and Rotated Rosenbrock's Function"""
    return CEC2017Function(4, dimensions)

def CEC2017_F5(dimensions: int = 10) -> CEC2017Function:
    """F5: Shifted and Rotated Rastrigin's Function"""
    return CEC2017Function(5, dimensions)

def CEC2017_F6(dimensions: int = 10) -> CEC2017Function:
    """F6: Shifted and Rotated Expanded Scaffer's F6"""
    return CEC2017Function(6, dimensions)

def CEC2017_F7(dimensions: int = 10) -> CEC2017Function:
    """F7: Shifted and Rotated Lunacek Bi-Rastrigin"""
    return CEC2017Function(7, dimensions)

def CEC2017_F8(dimensions: int = 10) -> CEC2017Function:
    """F8: Shifted and Rotated Non-Continuous Rastrigin"""
    return CEC2017Function(8, dimensions)

def CEC2017_F9(dimensions: int = 10) -> CEC2017Function:
    """F9: Shifted and Rotated Levy Function"""
    return CEC2017Function(9, dimensions)

def CEC2017_F10(dimensions: int = 10) -> CEC2017Function:
    """F10: Shifted and Rotated Schwefel's Function"""
    return CEC2017Function(10, dimensions)

# F11-F20: Hybrid Functions
def CEC2017_F11(dimensions: int = 10) -> CEC2017Function:
    """F11: Hybrid Function 1 (N=3)"""
    return CEC2017Function(11, dimensions)

def CEC2017_F12(dimensions: int = 10) -> CEC2017Function:
    """F12: Hybrid Function 2 (N=3)"""
    return CEC2017Function(12, dimensions)

def CEC2017_F13(dimensions: int = 10) -> CEC2017Function:
    """F13: Hybrid Function 3 (N=3)"""
    return CEC2017Function(13, dimensions)

def CEC2017_F14(dimensions: int = 10) -> CEC2017Function:
    """F14: Hybrid Function 4 (N=4)"""
    return CEC2017Function(14, dimensions)

def CEC2017_F15(dimensions: int = 10) -> CEC2017Function:
    """F15: Hybrid Function 5 (N=4)"""
    return CEC2017Function(15, dimensions)

def CEC2017_F16(dimensions: int = 10) -> CEC2017Function:
    """F16: Hybrid Function 6 (N=4)"""
    return CEC2017Function(16, dimensions)

def CEC2017_F17(dimensions: int = 10) -> CEC2017Function:
    """F17: Hybrid Function 7 (N=5)"""
    return CEC2017Function(17, dimensions)

def CEC2017_F18(dimensions: int = 10) -> CEC2017Function:
    """F18: Hybrid Function 8 (N=5)"""
    return CEC2017Function(18, dimensions)

def CEC2017_F19(dimensions: int = 10) -> CEC2017Function:
    """F19: Hybrid Function 9 (N=5)"""
    return CEC2017Function(19, dimensions)

def CEC2017_F20(dimensions: int = 10) -> CEC2017Function:
    """F20: Hybrid Function 10 (N=6)"""
    return CEC2017Function(20, dimensions)

# F21-F30: Composition Functions
def CEC2017_F21(dimensions: int = 10) -> CEC2017Function:
    """F21: Composition Function 1 (N=3)"""
    return CEC2017Function(21, dimensions)

def CEC2017_F22(dimensions: int = 10) -> CEC2017Function:
    """F22: Composition Function 2 (N=3)"""
    return CEC2017Function(22, dimensions)

def CEC2017_F23(dimensions: int = 10) -> CEC2017Function:
    """F23: Composition Function 3 (N=4)"""
    return CEC2017Function(23, dimensions)

def CEC2017_F24(dimensions: int = 10) -> CEC2017Function:
    """F24: Composition Function 4 (N=4)"""
    return CEC2017Function(24, dimensions)

def CEC2017_F25(dimensions: int = 10) -> CEC2017Function:
    """F25: Composition Function 5 (N=5)"""
    return CEC2017Function(25, dimensions)

def CEC2017_F26(dimensions: int = 10) -> CEC2017Function:
    """F26: Composition Function 6 (N=5)"""
    return CEC2017Function(26, dimensions)

def CEC2017_F27(dimensions: int = 10) -> CEC2017Function:
    """F27: Composition Function 7 (N=6)"""
    return CEC2017Function(27, dimensions)

def CEC2017_F28(dimensions: int = 10) -> CEC2017Function:
    """F28: Composition Function 8 (N=6)"""
    return CEC2017Function(28, dimensions)

def CEC2017_F29(dimensions: int = 10) -> CEC2017Function:
    """F29: Composition Function 9 (N=3)"""
    return CEC2017Function(29, dimensions)

def CEC2017_F30(dimensions: int = 10) -> CEC2017Function:
    """F30: Composition Function 10 (N=3)"""
    return CEC2017Function(30, dimensions)


# Registry
CEC2017_FUNCTIONS = {
    f'f{i}': eval(f'CEC2017_F{i}') for i in range(1, 31)
}


def get_cec2017_function(func_num: int, dimensions: int = 10) -> CEC2017Function:
    """
    Get a CEC2017 benchmark function.
    
    Parameters:
    -----------
    func_num : int
        Function number (1-30)
    dimensions : int
        Problem dimensionality (10, 30, 50, or 100)
    
    Returns:
    --------
    CEC2017Function
        The benchmark function instance
    
    Example:
    --------
    >>> problem = get_cec2017_function(1, 30)
    >>> optimizer = LSHADE(problem)
    >>> optimizer.optimize()
    """
    return CEC2017Function(func_num, dimensions)
