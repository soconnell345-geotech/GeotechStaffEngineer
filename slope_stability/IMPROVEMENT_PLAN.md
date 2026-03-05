# Slope Stability Module: Deep Dive & Improvement Plan

## Research Summary: HYRCAN vs SSAP 2010 vs Our Module

### HYRCAN (v3.0.4, Nov 2025)
- **Language**: C++ with Qt GUI (not Python as initially thought)
- **License**: LGPLv3, fully free for commercial use
- **Developer**: Roozbeh Geraili Mikola, Ph.D., P.E. (geowizard.org)
- **Downloads**: 5,000+ worldwide since Apr 2021

#### Analysis Methods (5)
| Method | Notes |
|--------|-------|
| Fellenius/Ordinary | Added v3.0.3 (Oct 2025) |
| Bishop Simplified | Original release |
| Spencer | Added v1.25 |
| Janbu Simplified | Added v1.10 |
| GLE / Morgenstern-Price | Original release, user-selectable interslice force function |

#### Strength Models (3)
- Mohr-Coulomb (effective or total stress)
- Generalized Hoek-Brown (rock slopes — GSI, mi, sigma_ci, D)
- SHANSEP (undrained clays — su/sigma'v depends on OCR)

#### Search Algorithms
- **Circular**: Entry/exit region method (similar to Slope/W) — user defines entry zone and exit zone on the ground surface, program generates trial arcs between them
- **Non-circular**: Particle Swarm Optimization (PSO) — metaheuristic optimization over control point coordinates

#### Key GUI Features
- CAD-type modeling interface (draw geometry point-by-point)
- DXF import from AutoCAD
- **All trial surfaces visible** after analysis with critical surface highlighted
- **Color coding** of surfaces by FOS
- **Click-on-slice inspection** — dialog shows weight, base normal, shear, pore pressure, interslice forces
- **Multi-method comparison** — run all methods simultaneously, compare FOS side-by-side
- Run-time monitoring (see results as analysis progresses)
- Python/JavaScript scripting for parametric studies
- Portable .exe (no install, no admin rights)
- 9-language support

#### Reinforcement Types (5)
- End-anchored rock bolts
- Grouted tiebacks
- Soil nails
- Geotextiles/geosynthetics
- Piles (passive resistance)

#### Pore Pressure Options
- Water table (drawn or coordinate-based)
- Ponded water (auto-detected when WT > ground surface)
- Pore pressure ratio (Ru) per material
- Rapid drawdown analysis

---

### SSAP 2010 (v6.1, Feb 2026)
- **Language**: Object Pascal / FreePascal with Lazarus IDE
- **License**: Full freeware
- **Developer**: Dr. Lorenzo Borselli (35 years of development since 1991)
- **Platform**: Windows 64-bit

#### Analysis Methods (7 rigorous methods)
| Method | Type |
|--------|------|
| Janbu Rigorous (1973) | Full equilibrium |
| Spencer (1973) | Full equilibrium |
| Sarma I (1973) | Full equilibrium |
| Morgenstern & Price (1965) | Full equilibrium |
| Chen & Morgenstern (1983) | Full equilibrium |
| Sarma II (1979) | Full equilibrium |
| Borselli (2016) | Original rigorous method |

**Note**: SSAP uses ONLY rigorous methods. No simplified methods (no Bishop Simplified, no Janbu Simplified). This is a deliberate design philosophy — the author argues simplified methods give misleading results.

#### Failure Criteria (5)
- Mohr-Coulomb
- Tresca
- GSI-GHB (Generalized Hoek-Brown, 1997/2019)
- Barton-Bandis JRC (1990) — for rock discontinuities
- Liquefaction — Olson-Stark (2003) for both dynamic and static conditions

#### Search Algorithms (5 engines)
1. **SNIFF RANDOM SEARCH** (Borselli 2002, 2023) — *flagship innovation*
   - Hybrid expert-system + Monte Carlo
   - Biases random surfaces toward weak layers ("sniffs" for worst properties)
   - Generic (non-circular) surface shapes
   - Currently v3.3
   - Claimed comparable or better than multidimensional optimization
2. **NEW RANDOM SEARCH** — alternative random generator
3. **MIXED RANDOM SEARCH** — combination approach
4. **Dynamic Range** — dynamic attractor-based exploration
5. **Dynamic Attractor** — exploration/exploitation in (Lambda0, FS0) space

#### Key GUI/Visualization Features
- **2D color maps of local safety factor** — qFEM/p-qFEM hybrid LEM-FEM method
  - Generates a mesh over the slope cross-section
  - Computes local FOS at each point using Strength Reduction Method
  - Renders as a heatmap — immediately shows where the slope is weakest
- **DXF output** (R12 format) — export results to AutoCAD
- Internal pressure diagrams
- Internal force distribution plots
- Critical surface mapping over geometry
- MAKEFILE utility for batch model creation
- Bilingual (English/Italian)

#### Reinforcement (7+ types)
- Reinforced earth / geosynthetics
- Piles (up to 12 lines, models: Ito & Matsui, Hassiotis, Kumar & Hall)
- Anchors/bolts
- Gabions
- Walls
- Anchored steel wire mesh
- External surcharge

#### Groundwater (advanced)
- Phreatic line / piezometric levels
- Multi-groundwater mode for pressurized aquifer systems
- Aquiclude layer exclusion
- Custom fluid unit weight
- Pore pressure dissipation simulation
- Filtration force calculations
- Tension crack water effects

---

### Our Module (Current State)

#### Analysis Methods (3)
| Method | Status |
|--------|--------|
| Fellenius | Working |
| Bishop Simplified | Working (circular only) |
| Spencer | Working (circular + noncircular) |

#### Strength Models (1)
- Mohr-Coulomb only

#### Search Algorithms (2)
1. Grid search (2D grid over xc, yc) + golden-section radius optimization
2. Random noncircular search (random polylines, Spencer FOS)

#### Visualization (Dash GUI + Qt GUI)
- Cross-section plot with soil layers and GWT
- Circular slip arc rendering
- Entry/exit markers
- Slice boundary lines
- Stress distribution tab (sigma_n, tau_mob, tau_avail vs distance)

#### Known Bugs (5 critical)
1. **GUI crash**: References `is_stable`, `n_nails_active`, `nail_resisting_kN_per_m` — properties that don't exist on `SlopeStabilityResult`
2. **GUI crash**: Passes invalid `FOS_required` parameter to `analyze_slope()`
3. **Noncircular (0,0,0)**: `search_noncircular()` hardcodes `xc=0, yc=0, radius=0` — no polyline point storage
4. **Invisible slip surface**: Dash GUI draws circle arc from (xc, yc, radius) — produces nothing for noncircular
5. **Stress tab broken**: Qt GUI can't rebuild slices from (0,0,0), shows "Run analysis" even after running

#### What's Disconnected
- `nails.py` exists with SoilNail class but never called from analysis pipeline
- No integration with fem2d for SRM comparison
- No probabilistic analysis integration with pystra_agent

---

## Feature Gap Analysis

| Feature | HYRCAN | SSAP 2010 | Ours | Priority |
|---------|--------|-----------|------|----------|
| **Fellenius** | Yes | No* | Yes | - |
| **Bishop Simplified** | Yes | No* | Yes | - |
| **Spencer** | Yes | Yes | Yes | - |
| **Morgenstern-Price / GLE** | Yes | Yes | No | HIGH |
| **Janbu Rigorous** | No | Yes | No | MEDIUM |
| **Sarma** | No | Yes | No | LOW |
| **Hoek-Brown strength** | Yes | Yes | No | MEDIUM |
| **SHANSEP** | Yes | No | No | MEDIUM |
| **Barton JRC** | No | Yes | No | LOW |
| **Liquefaction criterion** | No | Yes | No | LOW (have liquepy) |
| **PSO search** | Yes | No | No | HIGH |
| **Weak-layer biased search** | No | Yes | No | HIGH |
| **Entry/exit region search** | Yes | No | No | HIGH |
| **Show all trial surfaces** | Yes | Implicit | No | HIGH |
| **Color-code by FOS** | Yes | Yes (qFEM map) | No | HIGH |
| **Click-to-inspect slice** | Yes | Partial | No | HIGH |
| **Multi-method comparison** | Yes | Yes (7 methods) | Partial | MEDIUM |
| **Local FOS heatmap** | No | Yes (qFEM) | No | MEDIUM |
| **DXF import** | Yes | No (DXF export) | Yes | - |
| **Soil nails** | Yes | Partial | Disconnected | MEDIUM |
| **Piles in slope** | Yes | Yes (3 models) | No | LOW |
| **Rock bolts** | Yes | Yes | No | LOW |
| **Geosynthetics** | Yes | Yes | No | LOW |
| **Ponded water** | Yes | Yes | No | LOW |
| **Pore pressure ratio (Ru)** | Yes | Yes | No | MEDIUM |
| **Rapid drawdown** | Yes | Yes | No | LOW |
| **Tension crack** | No | Yes | No | LOW |
| **Python scripting** | Yes | No (but batch files) | Built-in | - |
| **Force diagrams per slice** | Yes | Yes | No | HIGH |

*SSAP deliberately excludes simplified methods

---

## Improvement Roadmap

### Phase 1: Fix Critical Bugs (DONE)

1. ~~**Add `is_stable` property** to `SlopeStabilityResult`~~
2. ~~**Remove `FOS_required`** from all `analyze_slope()` calls in GUI~~
3. ~~**Add `slip_points` field** to `SlopeStabilityResult` for noncircular surfaces~~
4. ~~**Store polyline points** in `search_noncircular()` result~~
5. ~~**Draw polyline slip surfaces** in Dash GUI (not just circles)~~
6. ~~**Fix Qt stress distribution** — reconstruct PolylineSlipSurface from stored points~~

### Phase 2: Visualization Upgrades (DONE)

7. ~~**Show all trial surfaces** — `add_trial_surfaces()` renders up to 50 arcs, color-coded red→green by FOS~~
8. ~~**Color-code surfaces by FOS** — gradient from red (low FOS) to green (high FOS) with toggle~~
9. ~~**Click-to-inspect slice** — `build_slice_detail_popup()` shows force breakdown for most critical slice~~
10. ~~**Force diagram per slice** — `build_force_diagram()` plots sigma'_n, tau_mob, tau_avail, Weight vs x~~
11. ~~**Interslice force plot** — Combined into force diagram (stresses + weight on dual axis)~~
12. **Entry/exit region visualization** — Draw the constraint regions on the cross-section when search limits are set

### Phase 3: Search Algorithm Improvements (DONE)

13. ~~**Particle Swarm Optimization (PSO)** for noncircular search — `search_pso()`, 19 tests~~
14. ~~**Weak-layer biased search** (SNIFF-inspired) — `search_weak_layer_biased()`, biases toward weak layers~~
15. ~~**Entry/exit region search** — `search_entry_exit()`, circles through entry/exit points on perpendicular bisector~~

GUI now supports 6 search modes: Single Circle, Grid Search, Entry/Exit, Random Noncircular, PSO Noncircular, Weak-Layer Biased.

### Phase 4: Additional Analysis Methods (Medium Priority)

16. **Morgenstern-Price / GLE** method
    - User-selectable interslice force function f(x): constant, half-sine, trapezoidal
    - Solves for both FOS and lambda (scaling factor)
    - Most general rigorous method
17. **Janbu Rigorous** method
    - Full force equilibrium with line of thrust
    - Better for non-circular surfaces than Bishop
18. **Multi-method comparison table** — run all available methods, display FOS comparison table in results

### Phase 5: Strength Models (Medium Priority)

19. **Generalized Hoek-Brown** for rock slopes
    - Input: GSI, mi, sigma_ci, D (disturbance)
    - Convert to equivalent Mohr-Coulomb along slip surface (stress-dependent)
20. **Pore pressure ratio (Ru)** per layer
    - Simple alternative to explicit water table
    - u = Ru * gamma * h

### Phase 6: Advanced Visualization (Medium Priority)

21. **Local FOS heatmap** (inspired by SSAP's qFEM)
    - Generate a 2D mesh over the slope cross-section
    - At each point, compute local FOS from stress state and strength
    - Render as color contour map
    - This could integrate with our fem2d module's SRM capability
22. **Progress monitoring** — show surfaces being evaluated in real-time during search
23. **DXF export** of results (slip surfaces, FOS labels)

### Phase 7: Reinforcement Integration (Lower Priority)

24. **Connect nails.py** to the analysis pipeline
    - Modify slice force calculations to include nail tension
    - Show nail forces on cross-section plot
25. **Geosynthetic reinforcement** (simple tension along slip surface)
26. **Pile resistance** in slope (passive wedge model)

---

## Summary of Key Takeaways

### From HYRCAN:
- **Entry/exit region search** is more intuitive than center-grid search for most engineers
- **PSO for noncircular** is far superior to random sampling
- **Click-to-inspect slices** is expected by users
- **Show all surfaces** gives confidence that the search was thorough
- **Multi-language** and **portable .exe** are nice-to-haves
- **SHANSEP** is valuable for embankment-on-clay problems

### From SSAP 2010:
- **Rigorous methods only** philosophy (no simplified methods) — we should at minimum add M-P/GLE
- **SNIFF weak-layer biasing** is a brilliant idea — bias search toward weak layers
- **Local FOS heatmap** via qFEM is the most visually impressive feature — shows WHERE failure is developing
- **5 failure criteria** including rock (Hoek-Brown, Barton) and liquefaction
- **7 analysis methods** gives great cross-validation
- **35 years of development** — we're early in our journey but on a good foundation

### From Our Codebase:
- **The analysis engine is mathematically correct** — validates against Duncan textbook examples
- **The bugs are all in the GUI layer** — the core is solid
- **Noncircular support exists but results can't be visualized** — a few fields and drawing logic fix this
- **169+ tests** is excellent coverage for the analysis engine
- **Dict-based I/O** is a unique advantage for LLM integration that neither HYRCAN nor SSAP has
