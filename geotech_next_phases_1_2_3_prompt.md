# Claude Code Task: Geotechnical Agent Phases 1-3 — Integration, Judgment, and New Modules

## Project Context

This project already contains the following working modules:
- **groundhog** — general-purpose geotechnical library (external package)
- **DM7 library** — Navy Design Manual 7 equations
- **lateral_pile** — COM624P p-y method for laterally loaded piles
- **bearing_capacity** — CBEAR methods for shallow foundation bearing capacity
- **settlement** — CSETT methods for consolidation and elastic settlement
- **axial_pile** — SPILE/DRIVEN methods for driven pile axial capacity
- **sheet_pile** — CWALSHT methods for sheet pile wall design
- **pile_group** — CPGA methods for pile group analysis under rigid cap
- **wave_equation** — WEAP87 methods for pile driving analysis

We are now building three things simultaneously:
1. A **unifying soil profile** that all modules share
2. An **engineering judgment / QA layer** that checks results for reasonableness
3. **New calculation modules** that fill workflow gaps

The end user is an LLM agent (Claude Code) acting as a staff geotechnical engineer. Every API should be unambiguous, well-typed, and self-documenting.

---

## PHASE 1: Unifying Layer and Engineering Judgment

### 1A. Shared SoilProfile Class

Build `geotech_common/soil_profile.py` (or extend it if it already exists). This is the single canonical representation of the subsurface that every module consumes.

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class SoilLayer:
    """A single layer in the subsurface profile."""
    top_depth: float              # depth to top of layer (m), 0 = ground surface
    bottom_depth: float           # depth to bottom of layer (m)
    description: str              # e.g. "Soft gray clay (CH)"
    
    # Classification
    uscs: Optional[str] = None    # USCS symbol: CH, CL, SP, SM, etc.
    
    # Strength parameters — provide what you have, agent selects appropriate method
    cu: Optional[float] = None              # undrained shear strength (kPa)
    phi: Optional[float] = None             # effective friction angle (degrees)
    c_prime: Optional[float] = None         # effective cohesion (kPa)
    
    # Index and physical properties
    gamma: Optional[float] = None           # total unit weight (kN/m3)
    gamma_sat: Optional[float] = None       # saturated unit weight (kN/m3)
    e0: Optional[float] = None              # initial void ratio
    wn: Optional[float] = None              # natural water content (%)
    LL: Optional[float] = None              # liquid limit (%)
    PL: Optional[float] = None              # plastic limit (%)
    PI: Optional[float] = None              # plasticity index (%)
    
    # Consolidation parameters
    Cc: Optional[float] = None              # compression index
    Cr: Optional[float] = None              # recompression index
    Cv: Optional[float] = None              # coefficient of consolidation (m2/year)
    sigma_p: Optional[float] = None         # preconsolidation pressure (kPa)
    C_alpha: Optional[float] = None         # secondary compression index
    
    # Stiffness / deformation
    Es: Optional[float] = None              # elastic modulus (kPa)
    eps50: Optional[float] = None           # strain at 50% max stress (for p-y curves)
    k_py: Optional[float] = None            # p-y subgrade reaction modulus (kN/m3)
    
    # Field test data
    N_spt: Optional[float] = None           # average SPT N-value (blows/ft)
    N60: Optional[float] = None             # energy-corrected SPT N
    N160: Optional[float] = None            # corrected (N1)60
    qc: Optional[float] = None              # CPT tip resistance (kPa)
    fs: Optional[float] = None              # CPT sleeve friction (kPa)
    
    # Soil behavior type
    is_cohesive: Optional[bool] = None      # True = clay/silt, False = sand/gravel
    is_rock: bool = False
    qu: Optional[float] = None              # unconfined compressive strength for rock (kPa)
    RQD: Optional[float] = None             # Rock Quality Designation (%)

    @property
    def thickness(self) -> float:
        return self.bottom_depth - self.top_depth
    
    @property 
    def mid_depth(self) -> float:
        return (self.top_depth + self.bottom_depth) / 2.0


@dataclass
class GroundwaterCondition:
    """Groundwater levels for the site."""
    depth: float                            # depth to groundwater (m below ground surface)
    is_artesian: bool = False
    artesian_head: Optional[float] = None   # piezometric head if artesian (m above ground)


@dataclass 
class SoilProfile:
    """Complete subsurface profile for a location."""
    layers: List[SoilLayer]
    groundwater: GroundwaterCondition
    ground_elevation: Optional[float] = None  # elevation of ground surface (m, datum)
    location_name: Optional[str] = None
    boring_id: Optional[str] = None
    
    def layer_at_depth(self, z: float) -> SoilLayer:
        """Return the soil layer at a given depth."""
        ...
    
    def effective_stress_at_depth(self, z: float) -> float:
        """Compute vertical effective stress at depth z."""
        ...
    
    def total_stress_at_depth(self, z: float) -> float:
        """Compute total vertical stress at depth z."""
        ...
    
    def pore_pressure_at_depth(self, z: float) -> float:
        """Compute pore water pressure at depth z."""
        ...
    
    def effective_unit_weight_at_depth(self, z: float) -> float:
        """Return effective unit weight at depth z (accounts for GW)."""
        ...
    
    def to_bearing_capacity_input(self, footing_depth: float) -> dict:
        """Convert to input format for bearing_capacity module."""
        ...
    
    def to_settlement_input(self, footing_depth: float, footing_width: float) -> dict:
        """Convert to input format for settlement module."""
        ...
    
    def to_axial_pile_input(self, pile_length: float, pile_diameter: float) -> dict:
        """Convert to input format for axial_pile module."""
        ...
    
    def to_lateral_pile_input(self, pile_length: float, pile_diameter: float) -> dict:
        """Convert to input format for lateral_pile module."""
        ...
    
    def fill_missing_from_correlations(self):
        """
        Use standard correlations to estimate missing parameters from available data.
        For example:
        - If N_spt is available but phi is not (sand): estimate phi from Peck/Meyerhof
        - If N_spt is available but cu is not (clay): estimate cu from Terzaghi & Peck
        - If cu is available but eps50 is not: use typical values by consistency
        - If gamma is missing: estimate from USCS and N_spt
        - If Cc is missing but LL is available: Cc ≈ 0.009*(LL - 10) (Terzaghi)
        - If Es is missing: estimate from N_spt or cu
        
        Uses groundhog and DM7 correlations where available.
        Marks estimated values with a flag so the agent knows what's measured vs estimated.
        """
        ...
    
    def validate(self) -> List[str]:
        """
        Check the profile for internal consistency and flag issues:
        - Layers must be continuous (no gaps or overlaps)
        - Unit weights should be in typical ranges (14-24 kN/m3)
        - Friction angles should be 0-50 degrees
        - cu should be positive
        - SPT N should be 0-100 (flag refusal)
        - GW depth should be consistent with saturated unit weights
        Returns a list of warning strings.
        """
        ...
```

**Adapter methods** (`to_bearing_capacity_input`, etc.) are critical — they translate the universal profile into each module's specific input format, selecting the right soil parameters and p-y models automatically. This is what lets the agent say "analyze this profile" without manually reformatting inputs for each module.

Also build a `SoilProfileBuilder` that constructs a profile from common input formats:
- From a list of SPT boring log entries (depth, N, description)
- From CPT data (depth, qc, fs)
- From a simple table (depth, soil type, key parameters)

### 1B. Engineering Judgment / QA Module

Build `geotech_common/engineering_checks.py`. This module reviews analysis results and flags concerns.

**Result Reasonableness Checks:**

```python
def check_bearing_capacity(qult_kPa: float, soil_type: str, footing_width_m: float) -> List[str]:
    """
    Flag if bearing capacity is outside typical ranges.
    
    Typical ranges:
    - Soft clay: 50-150 kPa
    - Stiff clay: 150-500 kPa  
    - Loose sand: 100-300 kPa
    - Dense sand: 300-800 kPa
    - Gravel: 400-1200 kPa
    - Weathered rock: 500-2000 kPa
    
    Also flag:
    - qult < 75 kPa (very low — verify soft soil conditions)
    - qult > 2000 kPa on soil (unusually high — check parameters)
    - qallowable < applied stress (failure condition)
    """
    ...

def check_settlement(total_settlement_mm: float, differential_settlement_mm: float,
                     structure_type: str) -> List[str]:
    """
    Compare settlement to typical tolerances.
    
    Typical limits (FHWA / AASHTO):
    - Bridges: 50mm total, 1/500 angular distortion
    - Buildings (steel frame): 50mm total, 1/300 angular distortion
    - Buildings (concrete frame): 50mm total, 1/500 angular distortion
    - Industrial / warehouses: 75mm total, 1/300 angular distortion
    - MSE walls: 50-100mm total depending on facing type
    
    Also flag:
    - Primary consolidation time > 5 years (consider wick drains)
    - Secondary compression > 20% of total (long-term concern)
    """
    ...

def check_pile_capacity(capacity_kN: float, pile_type: str, pile_diameter_m: float,
                        pile_length_m: float, soil_description: str) -> List[str]:
    """
    Flag unusual pile capacity results.
    
    Rules of thumb:
    - Typical H-pile capacity: 400-1500 kN
    - Typical 300mm pipe pile: 300-1200 kN
    - Typical 600mm drilled shaft: 1000-5000 kN
    - Skin friction in clay typically 20-100 kPa
    - End bearing in sand typically 5-15 MPa
    - L/D ratio > 60 is unusual (flag)
    - L/D ratio < 10 for friction pile (flag — may behave as short rigid pile)
    
    Also flag:
    - Capacity increasing with depth at unreasonable rate
    - Single layer providing > 80% of total capacity (concentration risk)
    """
    ...

def check_lateral_pile(deflection_mm: float, max_moment_kNm: float,
                       pile_type: str, pile_diameter_m: float,
                       service_or_ultimate: str) -> List[str]:
    """
    Flag unusual lateral pile results.
    
    Typical limits:
    - Service deflection at pile head: 6-25mm depending on structure
    - Bridge foundations per AASHTO: typically 25mm max lateral
    - Sound wall / sign structure: 50mm may be acceptable
    
    Also flag:
    - Max moment location > 10*diameter below ground (unusually deep)
    - Zero-crossing of deflection > 15*diameter (pile may be too short)
    - Negative moment at pile head with free-head condition (solver error)
    """
    ...

def check_sheet_pile(embedment_m: float, retained_height_m: float,
                     max_moment_kNm_per_m: float, soil_type: str) -> List[str]:
    """
    Flag unusual sheet pile results.
    
    Rules of thumb:
    - Cantilever embedment typically 1.5-2.5x retained height in sand
    - Cantilever embedment typically 1.0-2.0x retained height in clay
    - Anchored embedment typically 0.5-1.5x retained height
    - Cantilever walls rarely practical above 5-6m retained height
    - If FOS < 1.5 on passive, warn
    """
    ...

def check_wave_equation(blow_count: float, max_comp_stress_MPa: float,
                        max_tension_stress_MPa: float, pile_type: str) -> List[str]:
    """
    Flag driving stress concerns.
    
    Typical limits (FHWA GEC-12):
    - Steel piles: compression < 0.9*fy (typically 0.9*248 = 223 MPa)
    - Steel piles: tension < 0.9*fy
    - Concrete piles: compression < 0.85*f'c (typically 0.85*35 = 30 MPa)
    - Concrete piles: tension < 0.7*sqrt(f'c) + effective prestress
    - Timber: compression < 3*allowable static stress
    
    Also flag:
    - Blow count > 240 blows/foot (practical refusal)
    - Blow count < 10 blows/foot at required capacity (easy driving — may lose capacity from setup)
    """
    ...

def check_pile_group(max_pile_load_ratio: float, min_pile_load_ratio: float,
                     n_piles: int, spacing_over_diameter: float) -> List[str]:
    """
    Flag pile group concerns.
    
    Check:
    - Any pile in tension when not designed for it
    - Max/min load ratio > 2:1 (uneven distribution)
    - Spacing < 3*diameter (group effects significant, verify p-multipliers used)
    - Spacing > 8*diameter (cap spanning concern)
    - Any pile loaded > 90% of capacity (low margin)
    """
    ...
```

**Cross-Module Consistency Checks:**

```python
def check_foundation_selection(profile: SoilProfile, 
                               shallow_results: dict, 
                               deep_results: dict) -> List[str]:
    """
    Help the agent decide between shallow and deep foundations.
    
    Flag when:
    - Shallow foundation works but settlement is > 80% of limit (marginal)
    - Pile length < 5m (consider shallow foundation instead)
    - Shallow foundation FOS > 5 (overdesigned — could reduce footing size)
    - Soft layer within 2B below footing (check for punch-through)
    - Liquefiable layer in profile (deep foundations usually required)
    """
    ...

def check_parameter_consistency(profile: SoilProfile) -> List[str]:
    """
    Check that soil parameters are internally consistent.
    
    Flag:
    - cu and phi both specified for same layer (pick one analysis type)
    - N_spt = 0 but cu > 50 (inconsistent)
    - phi > 40 but N_spt < 20 (inconsistent)
    - gamma_sat < gamma (impossible)
    - Cc > 1.0 (very high — verify, typical organic soils only)
    - sigma_p < current effective stress (should be >= for OC soils)
    - OCR > 10 (very high — verify)
    """
    ...
```

**Implementation pattern:** Each check function returns `List[str]` — empty list means no concerns, populated list contains plain-English warnings the agent can include in its response. Warnings are categorized as INFO, WARNING, or CRITICAL.

### 1C. Refactor Existing Modules

Add adapter methods so each existing module can accept a `SoilProfile` directly:

```python
# Example: bearing_capacity module gets a new entry point
from geotech_common.soil_profile import SoilProfile

def analyze_from_profile(profile: SoilProfile, 
                         footing_width: float, 
                         footing_length: float,
                         footing_depth: float,
                         applied_load: float,
                         **kwargs) -> BearingCapacityResults:
    """
    Run bearing capacity analysis directly from a SoilProfile.
    Automatically extracts soil parameters at footing depth,
    selects appropriate method (phi for sand, cu for clay),
    and runs the analysis.
    """
    ...
```

Do this for: bearing_capacity, settlement, axial_pile, lateral_pile, sheet_pile, pile_group. The wave_equation module can be deferred since it takes different inputs (hammer/driving system).

---

## PHASE 2: New Calculation Modules

### Module 8: drilled_shaft — Axial Capacity of Drilled Shafts (GEC-10)

Build `drilled_shaft/` for axial capacity of drilled shafts (bored piles) using FHWA GEC-10 methods.

**Source:** FHWA GEC-10: Drilled Shafts: Construction Procedures and LRFD Design Methods (FHWA-HIF-18-070, Brown et al., 2018)

**Methods to implement:**

**1. Side Resistance in Cohesive Soil — Alpha Method (O'Neill & Reese 1999, updated Brown et al. 2010):**
- fs = alpha * cu
- alpha depends on cu/pa ratio (pa = atmospheric pressure = 101.3 kPa):
  - cu/pa <= 1.5: alpha = 0.55
  - 1.5 < cu/pa <= 2.5: alpha = 0.55 - 0.1*(cu/pa - 1.5)
  - cu/pa > 2.5: see GEC-10 Figure 13-5 (reduce further, minimum ~0.35 practically)
- Exclude top 1.5m (5 ft) and bottom 1 diameter from side resistance
- Exclude any zone within casing

**2. Side Resistance in Cohesionless Soil — Beta Method:**
- fs = beta * sigma_v'
- beta = function of depth and N60:
  - Brown et al. (2010): beta = 1.5 - 0.245*sqrt(z) for z in meters, with 0.25 <= beta <= 1.2
  - Alternative: beta from N60 per O'Neill & Reese (1999)
- Limit fs to 200 kPa maximum
- Exclude top 1.5m (5 ft) from side resistance

**3. Side Resistance in Rock — Horvath & Kenney / O'Neill et al. (1996) IGM Method:**
- fs = C * sqrt(qu) where C is a reduction factor for socket roughness
- For rough sockets: C = 1.0 (with qu in kPa, fs in kPa)
- For smooth sockets: reduced C per O'Neill et al. IGM method
- Apply alpha_E factor for jointed/fractured rock
- Use RQD to assess rock quality

**4. End Bearing — Cohesive Soil:**
- qb = Nc * cu_tip
- Nc = 9.0 for L/D >= 3 (full Nc), reduced for shorter shafts
- Limit settlement-based mobilization: qb at 5% of base diameter

**5. End Bearing — Cohesionless Soil:**
- qb = 57.5 * N60_tip (kPa) for N60 <= 50 (from GEC-10)
- Limited to settlement-based value at 5% of base diameter
- For large diameter shafts (D > 1.27m): use reduced qb per O'Neill & Reese

**6. End Bearing — Rock:**
- qb = 2.5 * qu for intact rock
- Reduced for fractured rock based on RQD
- Check settlement-based limit

**7. Load-Settlement Curve (t-z method):**
- Normalized side load transfer: follows GEC-10 Figure 13-12
- Normalized base load transfer: follows GEC-10 Figure 13-14
- Produces load vs settlement curve for service limit state check

```
drilled_shaft/
    __init__.py
    shaft_geometry.py    # Shaft dimensions (straight, belled, rock socket)
    side_resistance.py   # Alpha, beta, and rock socket methods
    end_bearing.py       # Tip resistance for clay, sand, rock
    capacity.py          # Combined analysis with exclusion zones
    load_settlement.py   # t-z based load-settlement curve
    lrfd.py              # AASHTO resistance factors
    results.py           # Capacity summary, capacity vs depth profile
```

**Validation:** GEC-10 worked examples (Chapters 13-14 have extensive examples for clay, sand, mixed profiles, and rock sockets).

---

### Module 9: retaining_walls — MSE and Cantilever Concrete Walls

Build `retaining_walls/` covering the two most common wall types beyond sheet piles.

**Sources:**
- FHWA GEC-11: Design of Mechanically Stabilized Earth Walls and Reinforced Soil Slopes (FHWA-NHI-10-024)
- AASHTO LRFD Bridge Design Specifications, Section 11
- FHWA GEC-7: Soil Nail Walls (for future expansion)

**Part A: Cantilever Retaining Walls**

Classical gravity/cantilever wall design checks:

1. **External Stability:**
   - Sliding: FOS = (sum resisting forces) / (sum driving forces) >= 1.5
     - Resisting: base friction = V*tan(delta_b) + ca*B (V = vertical resultant, delta_b = base friction angle, ca = base adhesion)
     - Driving: horizontal component of active earth pressure + water
   - Overturning: FOS = (sum stabilizing moments) / (sum overturning moments) >= 2.0
     - Moments taken about toe
   - Bearing capacity: eccentricity check (resultant within middle third for soil, middle half for rock), then bearing pressure vs allowable
   - Global stability: flag for slope stability check (don't solve — just warn if conditions suggest it's needed)

2. **Earth Pressures:**
   - Active: Rankine or Coulomb (reuse from sheet_pile earth_pressure module)
   - Include surcharge, sloping backfill, broken back slope
   - Seismic increment via Mononobe-Okabe (see Phase 3)

3. **Geometry:**
   - Standard proportions: base width ~ 0.5-0.7*H, toe ~ 0.1*B, heel ~ remainder
   - Stem thickness: 0.08*H at top minimum, battered back face typical
   - Key (shear key) for sliding resistance if needed

**Part B: MSE Walls (Mechanically Stabilized Earth)**

Per FHWA GEC-11 / AASHTO:

1. **External Stability** (same checks as cantilever but reinforced zone acts as gravity block):
   - Sliding at base
   - Overturning (eccentricity of resultant)
   - Bearing capacity of foundation
   - Global stability

2. **Internal Stability:**
   - Tensile resistance of reinforcement: Tmax = sigma_h * Sv
     - sigma_h = Kr * sigma_v + delta_sigma_h (Kr varies with depth, different for metallic vs geosynthetic)
     - Kr/Ka ratio: varies from 1.7 at top to 1.0 at 6m+ depth (metallic) or constant (geosynthetic)
   - Pullout resistance: Pr = F* * alpha * sigma_v' * Le * C
     - F* = pullout resistance factor (function of depth for metallic strips)
     - Le = length of reinforcement in resistant zone (beyond failure surface)
     - C = 2 for strips, 1 for grids
   - Check at each reinforcement level: Tmax <= min(Tallowable, Pr/FOS)
   - Connection strength at face

3. **Reinforcement Layout:**
   - Minimum length: 0.7*H
   - Vertical spacing: 0.2-0.8m typical
   - Select reinforcement type and strength grade

```
retaining_walls/
    __init__.py
    earth_pressure.py     # Import/extend from sheet_pile module
    cantilever/
        __init__.py
        geometry.py       # Wall geometry definition and standard proportions
        external.py       # Sliding, overturning, bearing checks
        design.py         # Iterative design to meet all FOS requirements
    mse/
        __init__.py
        geometry.py       # Wall geometry, reinforcement layout
        reinforcement.py  # Metallic strip and geosynthetic properties database
        external.py       # External stability checks
        internal.py       # Tensile, pullout, and connection checks
        design.py         # Design reinforcement layout to meet all checks
    results.py            # Pressure diagrams, FOS summary, reinforcement schedule
```

**Validation:** GEC-11 worked examples for MSE walls, standard textbook examples for cantilever walls.

---

### Module 10: seismic_geotech — Seismic Earth Pressures and Site Classification

Build `seismic_geotech/` for the most common seismic geotechnical analyses.

**Sources:**
- AASHTO LRFD Bridge Design Specifications, Section 3 and 11
- FHWA GEC-3: Geotechnical Earthquake Engineering (FHWA-NHI-11-032)

**1. AASHTO/NEHRP Site Classification:**
- Site Class A through F based on:
  - Vs30 (shear wave velocity in top 30m) — primary method
  - N-bar (average SPT N in top 30m) — common alternative
  - su-bar (average undrained shear strength in top 30m)
- Compute directly from SoilProfile if SPT or Vs data available
- Output: Site Class letter and associated site coefficients Fpga, Fa, Fv

**2. Mononobe-Okabe Seismic Earth Pressures:**
- Active case: KAE = [cos²(phi - theta - beta)] / [cos(theta)*cos²(beta)*cos(delta + beta + theta) * (1 + sqrt(sin(phi+delta)*sin(phi-theta-i) / (cos(delta+beta+theta)*cos(i-beta))))²]
  - theta = arctan(kh / (1 - kv))
  - kh = horizontal seismic coefficient
  - kv = vertical seismic coefficient (often taken as 0)
  - beta = wall batter angle
  - i = backfill slope angle
  - phi = friction angle
  - delta = wall-soil friction angle
- Seismic increment: delta_PAE = 0.5 * gamma * H² * (KAE - KA)
- Applied at 0.6*H from base (higher than static 1/3*H)
- Passive case: KPE (reduced passive during earthquake)

**3. Simplified Liquefaction Triggering (Seed & Idriss simplified procedure):**
- CSR = 0.65 * (amax/g) * (sigma_v/sigma_v') * rd
  - rd = stress reduction factor (function of depth)
- CRR from (N1)60cs using Youd et al. (2001) curve or Boulanger & Idriss (2014)
- FOS_liq = CRR / CSR
- Evaluate at each layer where N_spt data available
- Output: FOS_liq vs depth profile, flag layers with FOS < 1.0

**4. Post-Liquefaction Residual Strength:**
- Sr from (N1)60cs per Seed & Harder (1990) or Idriss & Boulanger (2008)
- Needed for post-liquefaction stability and lateral spreading estimates

```
seismic_geotech/
    __init__.py
    site_class.py           # AASHTO/NEHRP site classification from Vs30 or N-bar
    mononobe_okabe.py       # Seismic earth pressures
    liquefaction.py         # Simplified liquefaction triggering (Seed & Idriss, Boulanger & Idriss)
    residual_strength.py    # Post-liquefaction strength
    results.py              # Summary output
```

**Validation:** FHWA GEC-3 worked examples, AASHTO site class examples.

---

## PHASE 3: Ground Improvement Evaluation

### Module 11: ground_improvement — Basic Evaluation Methods

Build `ground_improvement/` to help the agent evaluate whether ground improvement is viable before jumping to deep foundations.

**Sources:**
- FHWA NHI-06-019/020: Ground Improvement Methods (Volumes I and II)
- FHWA GEC-13: Ground Modification Methods Reference Manual

**1. Aggregate Piers / Rammed Aggregate Piers:**
- Area replacement ratio: as = Ac/A (column area / tributary area)
- Composite modulus: E_comp = as*Ec + (1-as)*Es
- Stress concentration ratio: n = sigma_c/sigma_s (typically 3-8 for aggregate piers)
- Improved bearing capacity: q_improved ≈ q_unreinforced * improvement_factor
- Settlement reduction factor: SRF ≈ 1 / (1 + as*(n-1))

**2. Prefabricated Vertical Drains (Wick Drains):**
- Barron's theory for radial consolidation
- Equivalent diameter of drain: dw
- Influence zone diameter: de (function of spacing and pattern — triangular or square)
  - Triangular: de = 1.05 * s
  - Square: de = 1.13 * s
- Time factor for radial drainage: Tr = ch*t / de²
- Degree of consolidation: Ur = 1 - exp(-8*Tr/F(n))
  - F(n) = ln(n) - 0.75 + smear and well resistance corrections
  - n = de/dw
- Combined vertical + radial: U_total = 1 - (1-Uv)*(1-Ur)
- Design: find drain spacing for target U% in target time

**3. Surcharge Preloading:**
- Simple: apply surcharge, monitor settlement, remove when target U% reached
- With wick drains: combined analysis
- Equivalent surcharge for overconsolidation: sigma_surcharge to achieve sigma_p' >= final stress

**4. Vibratory Densification (Vibro-compaction):**
- Applicable to clean sands (< 15% fines)
- Improvement from initial N_spt to target N_spt
- Probe spacing based on soil type and equipment
- Simple go/no-go assessment based on grain size

```
ground_improvement/
    __init__.py
    aggregate_piers.py     # Composite modulus, settlement reduction, bearing improvement
    wick_drains.py         # Barron's theory, drain spacing design, time to target consolidation
    surcharge.py           # Preloading with and without drains
    vibro.py               # Vibro-compaction feasibility assessment
    feasibility.py         # Decision support: which method for which conditions
    results.py             # Improved parameters, time-settlement curves
```

The `feasibility.py` module is key for the agent — it takes a `SoilProfile` and a design problem (required bearing capacity, allowable settlement, time constraint) and recommends which ground improvement methods are applicable and provides preliminary sizing.

---

## Build Order

1. **Phase 1A**: `SoilProfile` class and `fill_missing_from_correlations`
2. **Phase 1B**: Engineering checks for bearing_capacity and axial_pile (template for rest)
3. **Phase 1C**: Adapter methods on 2-3 existing modules (bearing_capacity, settlement, axial_pile)
4. **Phase 2**: drilled_shaft module (high priority — very common in practice)
5. **Phase 2**: retaining_walls — cantilever first, then MSE
6. **Phase 2**: seismic_geotech — site classification and Mononobe-Okabe first, then liquefaction
7. **Phase 3**: ground_improvement — wick drains and aggregate piers first
8. **Phase 1B continued**: Add engineering checks for all remaining modules
9. **Phase 1C continued**: Adapter methods on remaining modules

This is a large scope. It is fine to build across multiple sessions. Prioritize getting Phase 1A + 1B + drilled_shaft done in the first session, as those have the highest impact.
