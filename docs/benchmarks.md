# DEvolve Benchmark Suite

Comprehensive benchmark problems for testing Differential Evolution algorithms.

## Overview

The DEvolve package includes three categories of benchmark problems:

1. **Classic Benchmarks** - Standard optimization test functions
2. **CEC2017 Suite** - IEEE CEC 2017 competition functions
3. **Engineering Problems** - Constrained real-world design optimization

---

## 1. Classic Benchmarks

### Available Functions

| Function | Dimensions | Bounds | Optimum | Characteristics |
|----------|-----------|--------|---------|----------------|
| Sphere | Any | [-5.12, 5.12] | 0.0 | Unimodal, convex, separable |
| Rosenbrock | Any | [-2.048, 2.048] | 0.0 | Unimodal, valley-shaped, non-separable |
| Rastrigin | Any | [-5.12, 5.12] | 0.0 | Multimodal, highly rugged |
| Ackley | Any | [-32.768, 32.768] | 0.0 | Multimodal, nearly flat outer region |
| Griewank | Any | [-600, 600] | 0.0 | Multimodal, many local minima |
| Schwefel | Any | [-500, 500] | 0.0 | Multimodal, deceptive |
| Michalewicz | Any | [0, π] | Varies | Multimodal, steep valleys |
| Zakharov | Any | [-5, 10] | 0.0 | Unimodal, plate-shaped |

### Usage

```python
from devolve import Sphere, Rastrigin, JADE

# Create problem
problem = Sphere(dimensions=10)

# Optimize
optimizer = JADE(problem, population_size=50)
best_x, best_f = optimizer.optimize()

print(f"Best fitness: {best_f}")
```

---

## 2. CEC2017 Benchmark Suite

IEEE Congress on Evolutionary Computation 2017 competition functions.

### Function Categories

#### Unimodal (F1-F3)
- **F1**: Shifted and Rotated Bent Cigar
- **F2**: Shifted and Rotated Sum of Different Power  
- **F3**: Shifted and Rotated Zakharov

#### Simple Multimodal (F4-F10)
- **F4**: Shifted and Rotated Rosenbrock
- **F5**: Shifted and Rotated Rastrigin
- **F6**: Shifted and Rotated Expanded Scaffer's F6
- **F7**: Shifted and Rotated Lunacek Bi-Rastrigin
- **F8**: Shifted and Rotated Non-Continuous Rastrigin
- **F9**: Shifted and Rotated Levy
- **F10**: Shifted and Rotated Schwefel

#### Hybrid (F11-F20)
Combinations of different functions with partitioned dimensions:
- **F11-F20**: Various combinations of Zakharov, Rosenbrock, Rastrigin, Weierstrass, Griewank, Scaffer, HappyCat, HGBat, Katsuura, Ackley, Bent Cigar, Elliptic

#### Composition (F21-F30)
Weighted combinations of multiple functions:
- **F21-F30**: Compositions of 3-5 functions with different rotation and shift patterns

### Properties

- **Dimensions**: 10, 30, 50, or 100
- **Bounds**: [-100, 100]
- **Optimal Value**: func_num × 100 (e.g., F1 optimum = 100, F15 optimum = 1500)
- **Optimal Position**: Randomly shifted within search space

### Usage

```python
from devolve import get_cec2017_function, LSHADE

# Get a specific CEC2017 function
problem = get_cec2017_function(func_num=5, dimensions=10)

# Optimize
optimizer = LSHADE(problem, initial_population_size=100, max_iterations=200)
best_x, best_f = optimizer.optimize()

print(f"F5 (Rastrigin) result: {best_f}")
print(f"Error from optimum: {best_f - 500}")
```

### Shift and Rotation Data

The implementation automatically generates random shift vectors and rotation matrices if data files are not provided. For official CEC2017 data:

1. Download from: http://www.ntu.edu.sg/home/EPNSugan/index_files/CEC2017/CEC2017.htm
2. Place in directory and specify:

```python
problem = get_cec2017_function(1, 10, data_dir="path/to/cec2017_data")
```

---

## 3. Engineering Design Problems

Real-world constrained optimization problems from engineering design.

### 3.1 Pressure Vessel Design

Minimize the total cost of a cylindrical pressure vessel.

**Variables**: 4
- Ts: Shell thickness (discrete, multiples of 0.0625 in)
- Th: Head thickness (discrete, multiples of 0.0625 in)  
- R: Inner radius (continuous, 10-200 in)
- L: Length (continuous, 10-200 in)

**Constraints**: 4 (shell stress, head stress, volume, length limit)

**Known Optimum**: $6,059.71

```python
from devolve import PressureVesselDesign, LSHADEEpSin

problem = PressureVesselDesign()
optimizer = LSHADEEpSin(problem, initial_population_size=100, max_iterations=300)
best_x, best_f = optimizer.optimize()

print(f"Minimum cost: ${best_f:.2f}")
```

### 3.2 Welded Beam Design

Minimize fabrication cost of a welded beam subject to structural constraints.

**Variables**: 4
- h: Weld height (0.1-2.0 in)
- l: Clamped bar length (0.1-10.0 in)
- t: Beam width (0.1-10.0 in)
- b: Beam thickness (0.1-2.0 in)

**Constraints**: 7 (shear stress, bending stress, buckling, deflection, geometric)

**Known Optimum**: $1.7249

```python
from devolve import WeldedBeamDesign, JADE

problem = WeldedBeamDesign()
optimizer = JADE(problem, population_size=50)
best_x, best_f = optimizer.optimize()
```

### 3.3 Tension/Compression Spring Design

Minimize the weight of a tension/compression spring.

**Variables**: 3
- d: Wire diameter (0.05-2.0 in)
- D: Mean coil diameter (0.25-1.3 in)
- N: Number of active coils (2-15)

**Constraints**: 4 (minimum deflection, shear stress, surge frequency, diameter limits)

**Known Optimum**: 0.0126652 lb

```python
from devolve import TensionCompressionSpring, SHADE

problem = TensionCompressionSpring()
optimizer = SHADE(problem)
best_x, best_f = optimizer.optimize()
```

### 3.4 Speed Reducer Design

Minimize the weight of a speed reducer (gearbox).

**Variables**: 7
- b: Face width (2.6-3.6 cm)
- m: Module of teeth (0.7-0.8 mm)
- z: Number of teeth (17-28)
- l1, l2: Shaft lengths (7.3-8.3 in)
- d1, d2: Shaft diameters (2.9-3.9, 5.0-5.5 in)

**Constraints**: 11 (bending stress, surface stress, shaft deflections, shaft stresses)

**Known Optimum**: 2994.47 lb

```python
from devolve import SpeedReducerDesign, LSHADE

problem = SpeedReducerDesign()
optimizer = LSHADE(problem, initial_population_size=100)
best_x, best_f = optimizer.optimize()
```

---

## Constraint Handling

Engineering problems have constraints. The Problem class provides:

```python
# Check if solution is feasible
is_feasible, total_violation = problem.evaluate_constraints(x)

# Get individual constraint values (should be ≤ 0)
violations = problem.constraint_function(x)
```

Recommended algorithms for constrained problems:
- **LSHADE-EpSin**: Has built-in epsilon constraint handling
- **JADE**: Works well with penalty methods
- **SHADE**: Good balance of exploration/exploitation

---

## Testing All Benchmarks

Run comprehensive test suite:

```bash
python test_benchmarks.py
```

Test categories:
- ✓ Classic benchmark functions
- ✓ CEC2017 suite (sample functions)
- ✓ Engineering problems (with constraints)
- ✓ Problem interface compliance

---

## Performance Comparison

From testing on Sphere 10D (50 iterations, seed=42):

| Algorithm | Final Fitness | FEs | Characteristics |
|-----------|---------------|-----|-----------------|
| JADE | 3.17e-02 | 1020 | Best convergence |
| SHADE | 2.16e-02 | 1020 | Second best |
| L-SHADE | 3.43e-02 | 700 | Fewer FEs (adaptive NP) |
| jDE | 1.66e-01 | 1020 | Good baseline |
| LSHADE-EpSin | 1.14e-01 | 700 | Best for constraints |

**Recommendations**:
- Simple problems → **JADE**
- Complex/multimodal → **L-SHADE**
- Constrained problems → **LSHADE-EpSin**
- Limited budget → **L-SHADE** (adaptive population)

---

## References

### Classic Functions
- Jamil, M., & Yang, X. S. (2013). A literature survey of benchmark functions for global optimization problems. *Journal of Mathematical Modelling and Numerical Optimisation*, 4(2), 150-194.

### CEC2017
- Awad, N. H., Ali, M. Z., Liang, J. J., Qu, B. Y., & Suganthan, P. N. (2016). Problem definitions and evaluation criteria for the CEC 2017 special session and competition on single objective real-parameter numerical optimization. Technical Report, Nanyang Technological University, Singapore.

### Engineering Problems
- Coello, C. A. C. (2000). Use of a self-adaptive penalty approach for engineering optimization problems. *Computers in Industry*, 41(2), 113-127.
- Deb, K. (1991). Optimal design of a welded beam via genetic algorithms. *AIAA Journal*, 29(11), 2013-2015.

---

## See Also

- **examples/comprehensive_benchmarks.py** - Full usage examples
- **test_benchmarks.py** - Comprehensive test suite
- **docs/algorithms.md** - Algorithm selection guide
