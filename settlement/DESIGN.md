# settlement — Foundation Settlement Analysis

## Purpose
Computes immediate (elastic) and consolidation settlement for shallow
foundations, matching CSETT program output. Includes Schmertmann CPT method,
1-D consolidation, and time-rate analysis.

## References
- Schmertmann (1970, 1978) — CPT-based immediate settlement
- Hough (1959) — granular (sand/gravel) C'-index settlement
- Terzaghi 1-D consolidation theory
- FHWA GEC-6 (Shallow Foundations)

## Files
- `immediate.py` — elastic_settlement(), schmertmann_settlement()
- `hough.py` — hough_settlement() granular C'-index (HoughLayer/HoughResult)
- `consolidation.py` — consolidation_settlement() with Cc/Cr/Csec
- `time_rate.py` — time_factor(), degree_of_consolidation()
- `results.py` — SettlementResult with summary()/to_dict()
- `__init__.py` — exports analyze_settlement(), analyze_consolidation()

## Public API
```python
analyze_settlement(q_net, B, L, footing_depth, ...) -> SettlementResult
analyze_consolidation(layers, delta_sigma, ...) -> ConsolidationResult
```

## Key Notes
- Schmertmann uses SchmertmannLayer list with depth_top/bottom/Es/Izp
- Hough (granular): `hough_settlement(layers, q_net, B, L=None)` over HoughLayer
  (thickness/depth_to_center/sigma_v0/C_prime); per-layer
  dH=H/C'·log10[(σ'vo+Δσ)/σ'vo] with a 2:1 Δσ (reuses `approximate_2to1`). C' is
  the Hough bearing-capacity index (from corrected SPT N', GEC-6 Fig 5-19), NOT
  Cc/(1+e0). sigma_v0 is supplied per layer (not computed from a γ-profile).
  Validated vs GEC-6 Ex B-1 Tables B1-2/B1-3 (V-022).
- time_rate module reused by ground_improvement/surcharge.py and wick_drains.py
- Parameter `q0` (not `q_overburden`) for Schmertmann overburden correction
