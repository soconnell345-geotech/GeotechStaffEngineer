# geolysis_agent Design

## Overview

Wraps the geolysis library (v0.24.1) for soil classification, SPT corrections, and bearing capacity.

**External Library**: [geolysis](https://github.com/patrickboateng/geolysis) by Patrick Boateng
**License**: MIT
**Installation**: `pip install geolysis`

## Modules

### 1. Classification (`classification.py`)

#### USCS (Unified Soil Classification System)
- Classifies soil based on liquid limit, plastic limit, fines, sand, and gradation
- Returns symbol (e.g., 'SW-SC', 'CL', 'Pt') and description
- May return dual symbols for borderline cases (e.g., 'SW-SC,SP-SC')
- All percentages are 0-100, not decimals

#### AASHTO
- Classifies soil based on liquid limit, plastic limit, and fines
- Returns symbol with group index in parentheses (e.g., 'A-7-6(20)')
- Group index indicates soil quality (higher = worse subgrade)

### 2. SPT Corrections (`spt_corrections.py`)

#### Correction Chain
1. **Energy correction**: N → N60 (corrects for energy ratio, hammer, sampler, borehole, rod length)
2. **Overburden correction**: N60 → N1_60 (normalizes to 100 kPa overburden)
3. **Dilatancy correction** (optional): N1_60 → N_corrected (for fine sands)

#### Methods
- **Energy**: Automatic based on hammer/sampler type
- **Overburden**: 5 methods (gibbs, bazaraa, peck, liao, skempton)
- **Dilatancy**: Optional, see geolysis.spt.DilatancyCorrection

#### Design N-value
- **Weighted (wgt)**: Conservative, weights lower values more heavily
- **Minimum (min)**: Most conservative
- **Average (avg)**: Arithmetic mean

### 3. Bearing Capacity (`bearing.py`)

#### Allowable (SPT-based)
- For cohesionless soils only
- 3 methods: Bowles, Meyerhof, Terzaghi
- Returns allowable BC (kPa) and allowable load (kN)
- Based on corrected SPT N-value and tolerable settlement

#### Ultimate
- For all soil types (cohesive, cohesionless, mixed)
- 2 methods: Vesic (recommended), Terzaghi
- Returns ultimate BC (kPa) and bearing capacity factors (Nc, Nq, Nγ)
- Also returns allowable BC = ultimate / factor_of_safety

## Units

All inputs/outputs follow project SI convention:
- Length: meters (m)
- Pressure: kPa
- Unit weight: kN/m³
- Angle: degrees
- Percentages: 0-100 (not decimals)

**Exception**: geolysis uses mm for borehole_diameter (not m)

## Sign Conventions

- All depths positive downward from surface
- All pressures positive (compression)

## Validation Rules

### Classification
- Liquid limit: 0-200%
- Plastic limit: 0-200%, must be ≤ liquid limit
- Fines: 0-100%
- Sand: 0-100%
- d_10, d_30, d_60: ≥ 0 or None

### SPT
- Recorded N: ≥ 0
- Effective overburden pressure: > 0 kPa
- Energy percentage: 0 < E ≤ 1.0 (decimal)
- Borehole diameter: > 0 mm
- Rod length: > 0 m
- Corrected N values: ≥ 0

### Bearing Capacity
- Depth: ≥ 0 m
- Width: > 0 m
- SPT N: ≥ 0
- Settlement: > 0 mm
- Friction angle: 0-50 degrees
- Cohesion: ≥ 0 kPa
- Unit weight: > 0 kN/m³
- Factor of safety: > 1.0

## Edge Cases

### Classification
- **Non-plastic soil**: Set LL and PL to None
- **Organic soil**: Set `organic=True` for USCS
- **Missing gradation**: Set d_10/d_30/d_60 to None (fine-grained classification still works)
- **Dual symbols**: geolysis returns comma-separated string for borderline cases

### SPT
- **Very low N**: Corrected N can be < 1.0 (valid for very loose sand)
- **High overburden**: Use appropriate OPC method for deep deposits (liao recommended for σ'v > 300 kPa)
- **Dilatancy**: Only applicable to fine sands; leave as None for medium/coarse sand

### Bearing Capacity
- **Pure cohesion (φ=0)**: Valid for undrained clay analysis
- **Pure friction (c=0)**: Valid for clean sand
- **Very small width**: BC increases as width decreases (geolysis handles)
- **Deep embedment**: Some methods have depth/width ratio limits

## Testing Strategy

### Tier 1 (no geolysis required)
- Result dataclass defaults and construction
- Input validation (ranges, types, consistency)
- Utilities (has_geolysis)
- Foundry metadata (list/describe methods)

### Tier 2 (requires geolysis)
- USCS classification: known symbols
- AASHTO classification: known group indices
- SPT correction: intermediate values (N60, N1_60)
- Allowable BC: reasonable values for typical inputs
- Ultimate BC: bearing capacity factors, allowable = ultimate / FoS
- Design N: weighted vs min vs avg

## References

- ASTM D2487: USCS classification
- AASHTO M145: AASHTO classification
- Bowles (1996): Foundation Analysis and Design
- Meyerhof (1956): SPT-based bearing capacity
- Terzaghi & Peck (1967): Bearing capacity
- Vesic (1973): Bearing capacity factors
- Gibbs & Holtz (1957): SPT overburden correction

## Implementation Notes

- geolysis returns result objects with `.symbol` and `.description` attributes
- AASHTO result also has `.group_index` (string)
- SPT correction uses method chaining internally (Energy → OPC → Dilatancy)
- Bearing capacity objects have `.n_c`, `.n_q`, `.n_gamma` properties
- All geolysis imports are lazy (via utils module) for optional dependency
