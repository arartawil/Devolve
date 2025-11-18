"""
Engineering Design Optimization Problems

This module implements classical constrained engineering design optimization
problems commonly used for testing constrained optimization algorithms.

Problems:
- Pressure Vessel Design
- Welded Beam Design
- Tension/Compression Spring Design
- Speed Reducer Design

Each problem includes:
- Objective function (cost minimization)
- Constraint functions (inequality/equality)
- Design variable bounds
- Optimal known solutions (where available)

Reference:
    Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for
    engineering optimization problems. Computers in Industry, 41(2), 113-127.

Author: DEvolve Package
License: MIT
"""

from typing import Tuple, List, Callable
import numpy as np

from ..core.problem import Problem


class PressureVesselDesign(Problem):
    """
    Pressure Vessel Design Problem
    
    Minimize the cost of a cylindrical pressure vessel with a spherical cap.
    
    Design Variables:
    -----------------
    x[0]: Thickness of shell (Ts) - discrete (multiple of 0.0625 in)
    x[1]: Thickness of head (Th) - discrete (multiple of 0.0625 in)
    x[2]: Inner radius (R) - continuous [10, 200]
    x[3]: Length of cylindrical section (L) - continuous [10, 200]
    
    Objective:
    ----------
    Minimize: f(x) = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]^2 +
                     3.1661*x[0]^2*x[3] + 19.84*x[0]^2*x[2]
    
    Constraints:
    ------------
    g1: -x[0] + 0.0193*x[2] ≤ 0
    g2: -x[1] + 0.00954*x[2] ≤ 0
    g3: -π*x[2]^2*x[3] - (4/3)*π*x[2]^3 + 1296000 ≤ 0
    g4: x[3] - 240 ≤ 0
    
    Known Optimum:
    --------------
    x* = [0.8125, 0.4375, 42.0984, 176.6366]
    f(x*) = 6059.714
    
    Example:
    --------
    >>> from devolve import LSHADE
    >>> problem = PressureVesselDesign()
    >>> optimizer = LSHADE(problem, population_size=100)
    >>> best_x, best_f = optimizer.optimize()
    >>> print(f"Cost: ${best_f:.2f}")
    """
    
    def __init__(self):
        # Bounds: [Ts_min, Th_min, R_min, L_min], [Ts_max, Th_max, R_max, L_max]
        bounds = [
            (0.0625, 99*0.0625),    # Ts
            (0.0625, 99*0.0625),    # Th
            (10.0, 200.0),           # R
            (10.0, 200.0)            # L
        ]
        
        def objective(x):
            """Calculate total cost."""
            return (0.6224 * x[0] * x[2] * x[3] +
                    1.7781 * x[1] * x[2]**2 +
                    3.1661 * x[0]**2 * x[3] +
                    19.84 * x[0]**2 * x[2])
        
        def constraint_g1(x): return -x[0] + 0.0193 * x[2]
        def constraint_g2(x): return -x[1] + 0.00954 * x[2]
        def constraint_g3(x): return -np.pi * x[2]**2 * x[3] - (4/3) * np.pi * x[2]**3 + 1296000
        def constraint_g4(x): return x[3] - 240
        
        super().__init__(
            objective_function=objective,
            bounds=bounds,
            dimensions=4,
            constraints=[constraint_g1, constraint_g2, constraint_g3, constraint_g4],
            optimum=6059.714,
            optimum_position=np.array([0.8125, 0.4375, 42.0984, 176.6366]),
            name="Pressure Vessel Design"
        )
        
        # Helper for batch constraint evaluation
        self.constraint_function = lambda x: np.array([c(x) for c in self.constraints])
    
    def repair_solution(self, x: np.ndarray) -> np.ndarray:
        """
        Repair solution to satisfy discrete variable constraints.
        
        x[0] and x[1] must be multiples of 0.0625.
        """
        x_repaired = x.copy()
        x_repaired[0] = np.round(x[0] / 0.0625) * 0.0625
        x_repaired[1] = np.round(x[1] / 0.0625) * 0.0625
        return x_repaired


class WeldedBeamDesign(Problem):
    """
    Welded Beam Design Problem
    
    Minimize the cost of a welded beam subject to constraints on shear stress,
    bending stress, buckling load, end deflection, and side constraints.
    
    Design Variables:
    -----------------
    x[0]: h - Height of weld (0.1 to 2.0)
    x[1]: l - Length of clamped bar (0.1 to 10.0)
    x[2]: t - Width of beam (0.1 to 10.0)
    x[3]: b - Thickness of beam (0.1 to 2.0)
    
    Objective:
    ----------
    Minimize: f(x) = 1.10471*h^2*l + 0.04811*t*b*(14.0 + l)
    
    Constraints:
    ------------
    g1: τ(x) - τ_max ≤ 0            (shear stress)
    g2: σ(x) - σ_max ≤ 0            (bending stress)
    g3: h - b ≤ 0                    (geometric)
    g4: 0.10471*h^2 + 0.04811*t*b*(14+l) - 5.0 ≤ 0  (cost limit)
    g5: 0.125 - h ≤ 0                (minimum weld size)
    g6: δ(x) - δ_max ≤ 0            (deflection)
    g7: P - P_c(x) ≤ 0              (buckling load)
    
    Known Optimum:
    --------------
    x* = [0.2057, 3.4705, 9.0366, 0.2057]
    f(x*) = 1.7249
    
    Example:
    --------
    >>> problem = WeldedBeamDesign()
    >>> optimizer = JADE(problem, population_size=50)
    >>> best_x, best_f = optimizer.optimize()
    """
    
    def __init__(self):
        bounds = [
            (0.1, 2.0),    # h
            (0.1, 10.0),   # l
            (0.1, 10.0),   # t
            (0.1, 2.0)     # b
        ]
        
        # Constants
        self.P = 6000  # Load (lb)
        self.L = 14    # Length (in)
        self.E = 30e6  # Modulus of elasticity (psi)
        self.G = 12e6  # Shear modulus (psi)
        self.tau_max = 13600  # Maximum shear stress (psi)
        self.sigma_max = 30000  # Maximum normal stress (psi)
        self.delta_max = 0.25  # Maximum deflection (in)
        
        def objective(x):
            """Calculate total fabrication cost."""
            h, l, t, b = x
            return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
        
        def _compute_stresses(x):
            h, l, t, b = x
            tau_prime = self.P / (np.sqrt(2) * h * l)
            M = self.P * (self.L + l/2)
            R = np.sqrt(l**2/4 + ((h + t)/2)**2)
            J = 2 * (np.sqrt(2) * h * l * (l**2/12 + ((h + t)/2)**2))
            tau_double_prime = M * R / J
            tau = np.sqrt(tau_prime**2 + tau_double_prime**2 + l * tau_prime * tau_double_prime / R)
            sigma = 6 * self.P * self.L / (b * t**2)
            delta = 4 * self.P * self.L**3 / (self.E * t**3 * b)
            P_c = 4.013 * self.E * np.sqrt(t**2 * b**6 / 36) / self.L**2 * (1 - t / (2*self.L) * np.sqrt(self.E / (4*self.G)))
            return tau, sigma, delta, P_c
        
        self._compute_stresses = _compute_stresses
        
        def c1(x):
            tau, _, _, _ = self._compute_stresses(x)
            return tau - self.tau_max
        def c2(x):
            _, sigma, _, _ = self._compute_stresses(x)
            return sigma - self.sigma_max
        def c3(x): return x[0] - x[3]
        def c4(x): return 0.10471 * x[0]**2 + 0.04811 * x[2] * x[3] * (14 + x[1]) - 5.0
        def c5(x): return 0.125 - x[0]
        def c6(x):
            _, _, delta, _ = self._compute_stresses(x)
            return delta - self.delta_max
        def c7(x):
            _, _, _, P_c = self._compute_stresses(x)
            return self.P - P_c
        
        super().__init__(
            objective_function=objective,
            bounds=bounds,
            dimensions=4,
            constraints=[c1, c2, c3, c4, c5, c6, c7],
            optimum=1.7249,
            optimum_position=np.array([0.2057, 3.4705, 9.0366, 0.2057]),
            name="Welded Beam Design"
        )
        
        self.constraint_function = lambda x: np.array([c(x) for c in self.constraints])


class TensionCompressionSpring(Problem):
    """
    Tension/Compression Spring Design Problem
    
    Minimize the weight of a tension/compression spring subject to constraints
    on minimum deflection, shear stress, surge frequency, and limits on outside
    diameter.
    
    Design Variables:
    -----------------
    x[0]: d - Wire diameter [0.05, 2.0]
    x[1]: D - Mean coil diameter [0.25, 1.3]
    x[2]: N - Number of active coils [2, 15]
    
    Objective:
    ----------
    Minimize: f(x) = (N + 2) * D * d^2
    
    Constraints:
    ------------
    g1: 1 - (D^3 * N) / (71785 * d^4) ≤ 0
    g2: (4*D^2 - d*D)/(12566*(D*d^3 - d^4)) + 1/(5108*d^2) - 1 ≤ 0
    g3: 1 - 140.45*d/(D^2 * N) ≤ 0
    g4: (D + d)/1.5 - 1 ≤ 0
    
    Known Optimum:
    --------------
    x* = [0.0516, 0.3565, 11.2885]
    f(x*) = 0.0126652
    
    Example:
    --------
    >>> problem = TensionCompressionSpring()
    >>> optimizer = SHADE(problem)
    >>> best_x, best_f = optimizer.optimize()
    """
    
    def __init__(self):
        bounds = [
            (0.05, 2.0),    # d
            (0.25, 1.3),    # D
            (2.0, 15.0)     # N
        ]
        
        def objective(x):
            """Minimize weight of spring."""
            d, D, N = x
            return (N + 2) * D * d**2
        
        def c1(x): return 1 - (x[1]**3 * x[2]) / (71785 * x[0]**4)
        def c2(x): return (4*x[1]**2 - x[0]*x[1]) / (12566 * (x[1]*x[0]**3 - x[0]**4)) + 1/(5108*x[0]**2) - 1
        def c3(x): return 1 - 140.45*x[0] / (x[1]**2 * x[2])
        def c4(x): return (x[1] + x[0]) / 1.5 - 1
        
        super().__init__(
            objective_function=objective,
            bounds=bounds,
            dimensions=3,
            constraints=[c1, c2, c3, c4],
            optimum=0.0126652,
            optimum_position=np.array([0.0516, 0.3565, 11.2885]),
            name="Tension/Compression Spring"
        )
        
        self.constraint_function = lambda x: np.array([c(x) for c in self.constraints])


class SpeedReducerDesign(Problem):
    """
    Speed Reducer Design Problem
    
    Minimize the weight of a speed reducer subject to constraints on bending
    stress of gear teeth, surface stress, transverse deflections of shafts,
    and stresses in the shafts.
    
    Design Variables:
    -----------------
    x[0]: b - Face width [2.6, 3.6]
    x[1]: m - Module of teeth [0.7, 0.8]
    x[2]: z - Number of teeth on pinion [17, 28]
    x[3]: l1 - Length of first shaft [7.3, 8.3]
    x[4]: l2 - Length of second shaft [7.3, 8.3]
    x[5]: d1 - Diameter of first shaft [2.9, 3.9]
    x[6]: d2 - Diameter of second shaft [5.0, 5.5]
    
    Objective:
    ----------
    Minimize weight: f(x) = 0.7854*b*m*z^2*(3.3333*l1^2 + 14.9334*l1 - 43.0934)
                           - 1.508*b*(d1^2 + d2^2)
                           + 7.4777*(d1^3 + d2^3) + 0.7854*(l1*d1^2 + l2*d2^2)
    
    Constraints: 11 inequality constraints
    
    Known Optimum:
    --------------
    x* = [3.5, 0.7, 17, 7.3, 7.715, 3.350, 5.287]
    f(x*) = 2994.471
    
    Example:
    --------
    >>> problem = SpeedReducerDesign()
    >>> optimizer = LSHADE(problem, population_size=100)
    >>> best_x, best_f = optimizer.optimize()
    """
    
    def __init__(self):
        bounds = [
            (2.6, 3.6),   # b
            (0.7, 0.8),   # m
            (17, 28),     # z
            (7.3, 8.3),   # l1
            (7.3, 8.3),   # l2
            (2.9, 3.9),   # d1
            (5.0, 5.5)    # d2
        ]
        
        def objective(x):
            """Minimize weight of speed reducer."""
            b, m, z, l1, l2, d1, d2 = x
            
            return (0.7854 * b * m * z**2 * (3.3333*l1**2 + 14.9334*l1 - 43.0934) -
                    1.5079 * b * (d1**2 + d2**2) +
                    7.4777 * (d1**3 + d2**3) +
                    0.7854 * (l1*d1**2 + l2*d2**2))
        
        def c1(x): return 27 / (x[0] * x[1] * x[2]) - 1
        def c2(x): return 397.5 / (x[0] * x[1] * x[2]**2) - 1
        def c3(x): return 1.93 * x[3]**3 / (x[1] * x[2] * x[5]**4) - 1
        def c4(x): return 1.93 * x[4]**3 / (x[1] * x[2] * x[6]**4) - 1
        def c5(x): return np.sqrt((745*x[3]/(x[1]*x[2]))**2 + 16.9e6) / (110*x[5]**3) - 1
        def c6(x): return np.sqrt((745*x[4]/(x[1]*x[2]))**2 + 157.5e6) / (85*x[6]**3) - 1
        def c7(x): return x[1] * x[2] / 40 - 1
        def c8(x): return 5*x[1] / x[0] - 1
        def c9(x): return x[0] / (12*x[1]) - 1
        def c10(x): return (1.5*x[5] + 1.9) / x[3] - 1
        def c11(x): return (1.1*x[6] + 1.9) / x[4] - 1
        
        super().__init__(
            objective_function=objective,
            bounds=bounds,
            dimensions=7,
            constraints=[c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
            optimum=2994.471,
            optimum_position=np.array([3.5, 0.7, 17, 7.3, 7.715, 3.350, 5.287]),
            name="Speed Reducer Design"
        )
        
        self.constraint_function = lambda x: np.array([c(x) for c in self.constraints])


# Registry
ENGINEERING_PROBLEMS = {
    'pressure_vessel': PressureVesselDesign,
    'welded_beam': WeldedBeamDesign,
    'spring': TensionCompressionSpring,
    'speed_reducer': SpeedReducerDesign,
}


def get_engineering_problem(name: str) -> Problem:
    """
    Get an engineering design problem by name.
    
    Parameters:
    -----------
    name : str
        Problem name: 'pressure_vessel', 'welded_beam', 'spring', 'speed_reducer'
    
    Returns:
    --------
    Problem
        The engineering problem instance
    
    Example:
    --------
    >>> problem = get_engineering_problem('pressure_vessel')
    >>> optimizer = LSHADE(problem, population_size=100)
    >>> best_x, best_f = optimizer.optimize()
    """
    name_lower = name.lower().replace(' ', '_').replace('-', '_')
    
    if name_lower not in ENGINEERING_PROBLEMS:
        available = list(ENGINEERING_PROBLEMS.keys())
        raise ValueError(f"Unknown problem '{name}'. Available: {available}")
    
    return ENGINEERING_PROBLEMS[name_lower]()
