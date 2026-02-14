# Claude Code Task: Build a Lateral Pile Analysis Module (COM624P Methods)

## Project Context

This is part of an ongoing project building Python tools for LLM-based geotechnical engineering agents. We already have groundhog integrated. Now we need to build a lateral pile analysis module based on the public-domain methods from FHWA's COM624P program. This will be a standalone Python module that can be used alongside groundhog.

**Do NOT try to find or port original Fortran code.** Instead, reimplement from the published engineering methods. The theory is fully documented in public FHWA publications.

## What COM624P Does

COM624P solves the problem of a single pile or drilled shaft subjected to lateral loads and moments at the pile head. It computes:
- Lateral deflection vs depth
- Rotation vs depth  
- Bending moment vs depth
- Shear force vs depth
- Soil reaction (p) vs depth

The method uses **p-y curves** (nonlinear soil springs) coupled with a **finite difference solution** of the beam-column differential equation.

## Architecture

Create a Python package called `lateral_pile` with this structure:

```
lateral_pile/
    __init__.py
    pile.py           # Pile definition (geometry, EI, sections)
    soil.py           # Soil layer definitions
    py_curves.py      # All p-y curve models
    solver.py         # Finite difference beam-column solver
    analysis.py       # Top-level analysis runner
    results.py        # Results container and plotting
    validation.py     # Validation against published examples
```

## Phase 1: p-y Curve Models (py_curves.py)

Implement these p-y curve formulations. Each should be a function that takes depth, soil properties, and a deflection value y, and returns the soil resistance p (force/length). Also provide a function that returns the full p-y curve as arrays for a given depth.

### 1. Soft Clay Below Water Table — Matlock (1970)
- Ultimate resistance: pu = min(3*c + gamma'*z + J*c*z/b, 9*c*b) where:
  - c = undrained shear strength
  - gamma' = effective unit weight  
  - z = depth
  - b = pile diameter
  - J = empirical constant (0.5 for soft clay, 0.25 for medium clay)
- Reference deflection: y50 = 2.5 * eps50 * b
  - eps50 = strain at 50% of max stress (typical values: soft clay 0.02, medium clay 0.01, stiff clay 0.005)
- Static loading: p = 0.5 * pu * (y/y50)^(1/3) for y <= 8*y50, then p = pu
- Cyclic loading: p follows Matlock's cyclic degradation rules:
  - For z > zr (deep): p = 0.72*pu for y >= 3*y50
  - For z < zr (shallow): p linearly reduces from 0.72*pu to 0.72*pu*(z/zr) for y >= 3*y50
  - zr = critical depth where wedge and flow-around mechanisms transition

### 2. Stiff Clay Below Water Table — Reese et al. (1975)
- Uses five-segment p-y curve construction
- Requires ks (initial modulus of subgrade reaction) in addition to c, gamma', eps50
- Static case has parabolic transition; cyclic case has degraded plateau
- Coefficients As and Ac are functions of z/b ratio (tabulated/interpolated)
- pu computed from two mechanisms: wedge failure near surface, flow-around at depth

### 3. Stiff Clay Above Water Table — Welch & Reese (1972)  
- Shape: p = 0.5*pu*(y/y50)^(1/4) — note 1/4 power, not 1/3
- Same pu formulation as Matlock but with total unit weight
- Cyclic: p remains constant beyond y = 16*y50

### 4. Sand — Reese, Cox & Koop (1974)
- Three-part p-y curve construction
- Ultimate resistance from two mechanisms:
  - Shallow (wedge): pus uses coefficients C1, C2 (functions of friction angle phi)
  - Deep (flow): pud uses coefficient C3 (function of phi)
  - pu = min(pus, pud) at each depth
- Initial slope: k * z (k = modulus of subgrade reaction, function of phi and density)
- Parabolic transition between initial line and ultimate resistance
- Requires: phi (friction angle), gamma (unit weight), k (subgrade reaction modulus)

### 5. API Sand — O'Neill & Murchinson (1983) / API RP2A
- Simplified version of Reese sand using hyperbolic tangent:
  - p = A * pu * tanh(k * z * y / (A * pu))
  - A = 0.9 for cyclic, (3.0 - 0.8*z/b) >= 0.9 for static
- Same pu as Reese sand (C1, C2, C3 coefficients)
- Widely used for offshore/transportation piles

### 6. Weak Rock — Reese (1997)
- For weak/weatherite rock with qu (unconfined compressive strength)
- Uses initial modulus kir and ultimate resistance pur
- Include if time permits; lower priority than clay and sand models

For each model, include:
- Parameter validation with typical ranges
- Both static and cyclic loading versions
- Docstrings citing the original reference
- A helper function to generate a table of p vs y values at a given depth

## Phase 2: Finite Difference Solver (solver.py)

The governing differential equation for a beam-column (pile) is:

    EI * d4y/dz4 + Q * d2y/dz2 + Es * y = 0

Where:
- EI = pile flexural rigidity (can vary with depth for nonprismatic piles)
- Q = axial load on the pile  
- y = lateral deflection
- z = depth
- Es = secant modulus of soil reaction (p/y) — this is what makes it nonlinear

**Finite Difference Method:**
1. Divide pile into n equal segments of length h
2. At each node i, the 4th derivative is approximated as:
   (y[i-2] - 4*y[i-1] + 6*y[i] - 4*y[i+1] + y[i+2]) / h^4
3. The 2nd derivative for axial load effect:
   (y[i-1] - 2*y[i] + y[i+1]) / h^2
4. This creates a system of n+1 equations with n+5 unknowns
5. Four boundary conditions close the system (two at top, two at bottom)

**Boundary Conditions at Pile Head (z=0):**
- Free head: specify shear Vt and moment Mt
- Fixed head: specify shear Vt and rotation St = 0
- Partially fixed: specify shear Vt and rotational stiffness

**Boundary Conditions at Pile Tip:**
- Typically: moment = 0 and shear = 0 (free tip)

**Iterative Solution:**
1. Assume initial Es values (e.g., from linear portion of p-y curves)
2. Solve the linear system [K]{y} = {F} for deflections
3. For each node, use the computed y to look up p from the p-y curve
4. Compute new Es = p/y at each node
5. Repeat until convergence (deflections change less than tolerance)
6. Typical convergence in 5-20 iterations

**Important implementation details:**
- Use numpy for matrix assembly and solving
- The stiffness matrix is pentadiagonal (bandwidth of 5)
- Handle the case where y ≈ 0 carefully (avoid division by zero in Es = p/y)
- Allow variable EI along the pile length
- Include axial load effect (P-delta) for combined loading
- Default to 100 pile segments (n=100) for good accuracy

## Phase 3: Analysis Runner (analysis.py)

Create a high-level `LateralPileAnalysis` class that:

1. Accepts a Pile object and a list of SoilLayer objects
2. Accepts loading: lateral load (Vt), moment (Mt), axial load (Q)
3. Accepts boundary condition type at pile head
4. Runs the iterative solver
5. Returns a Results object with deflection, moment, shear, rotation, and soil reaction profiles

The API should be clean enough for an LLM agent to use. Example usage:

```python
from lateral_pile import Pile, SoilLayer, LateralPileAnalysis
from lateral_pile.py_curves import SoftClayMatlock, SandAPI

# Define pile
pile = Pile(
    length=20.0,        # meters
    diameter=0.6,       # meters  
    thickness=0.012,    # wall thickness for pipe pile (m)
    E=200e6,            # Young's modulus (kPa)
    moment_of_inertia=None  # computed from geometry if not provided
)

# Define soil layers
layers = [
    SoilLayer(
        top=0.0, bottom=5.0,
        py_model=SoftClayMatlock(
            c=25.0,           # undrained shear strength (kPa)
            gamma=8.0,        # effective unit weight (kN/m3)
            eps50=0.02,       # strain at 50% stress
            J=0.5,
            loading='static'
        )
    ),
    SoilLayer(
        top=5.0, bottom=20.0,
        py_model=SandAPI(
            phi=35.0,         # friction angle (degrees)
            gamma=10.0,       # effective unit weight (kN/m3)
            k=16000,          # subgrade reaction modulus (kN/m3)
            loading='static'
        )
    ),
]

# Run analysis
analysis = LateralPileAnalysis(pile, layers)
results = analysis.solve(
    Vt=100.0,     # lateral load at pile head (kN)
    Mt=0.0,       # moment at pile head (kN-m)  
    Q=500.0,      # axial load (kN)
    head_condition='free',
    n_elements=100,
    tolerance=1e-5,
    max_iterations=100
)

# Access results
print(f"Pile head deflection: {results.y_top:.4f} m")
print(f"Max bending moment: {results.max_moment:.1f} kN-m")
results.plot_deflection()
results.plot_moment()
results.plot_shear()
results.plot_soil_reaction()
```

## Phase 4: Validation (validation.py)

Build validation tests against known solutions:

1. **Matlock's original test**: 12.75-inch OD steel pipe pile in soft clay at Lake Austin. This is well-documented and COM624P's manual includes the comparison.

2. **Simple elastic check**: For a pile in uniform elastic soil (constant Es), the closed-form Hetenyi solution exists. Use this to verify the finite difference solver independently of the p-y curves.

3. **COM624P manual examples**: The manual (FHWA-SA-91-048) includes 4 worked examples with full input/output. Use Examples 1 and 2 as validation targets. The manual is available at: https://rosap.ntl.bts.gov/view/dot/40955/dot_40955_DS1.pdf

Include pytest-style tests that verify:
- p-y curves generate correct values at known points
- Solver converges for elastic case to within 1% of closed-form solution
- Full analysis matches COM624P examples to within 5% (allowing for minor differences in implementation)

## Implementation Notes

- Use **numpy** for all numerical work (matrix assembly, linear algebra)
- Use **matplotlib** for plotting (but make it optional — don't fail if not installed)
- Use **dataclasses** for clean data structures
- All units should be **SI (kN, m, kPa)** internally, but provide a unit conversion utility
- Include comprehensive docstrings with references to the original papers
- Each p-y model should cite its source paper and page in the COM624P manual
- Parameter validation: warn if soil parameters are outside typical ranges
- The module should have NO dependencies beyond numpy (matplotlib optional for plotting)

## Key References (all publicly available)

1. COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993) — full theory in Part II
2. FHWA GEC-13: Design, Analysis, and Testing of Laterally Loaded Deep Foundations (FHWA-HIF-18-031) — current state of practice
3. Matlock, H. (1970) "Correlations for Design of Laterally Loaded Piles in Soft Clay" — OTC 1204
4. Reese, L.C., Cox, W.R. & Koop, F.D. (1974) "Field Testing and Analysis of Laterally Loaded Piles in Sand" — OTC 2080
5. Reese, L.C., Cox, W.R. & Koop, F.D. (1975) "Field Testing and Analysis of Laterally Loaded Piles in Stiff Clay" — OTC 2312
6. API RP2A — Recommended Practice for Planning, Designing and Constructing Fixed Offshore Platforms

## Priority Order

Build in this order, testing each phase before moving on:
1. `pile.py` and `soil.py` — data structures
2. `py_curves.py` — Matlock soft clay first, then API sand, then others
3. `solver.py` — finite difference solver, validate with elastic closed-form
4. `analysis.py` and `results.py` — tie it together
5. `validation.py` — test against published examples
6. Iterate and fix until validation passes

Start with Phase 1 and 2. Get the solver working with Matlock soft clay and verify it produces reasonable results before adding more p-y models.
