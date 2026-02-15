# settlement — Foundation Settlement Analysis

## Purpose
Computes immediate (elastic) and consolidation settlement for shallow
foundations, matching CSETT program output. Includes Schmertmann CPT method,
1-D consolidation, and time-rate analysis.

## References
- Schmertmann (1970, 1978) — CPT-based immediate settlement
- Terzaghi 1-D consolidation theory
- FHWA GEC-6 (Shallow Foundations)

## Files
- `immediate.py` — elastic_settlement(), schmertmann_settlement()
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
- time_rate module reused by ground_improvement/surcharge.py and wick_drains.py
- Parameter `q0` (not `q_overburden`) for Schmertmann overburden correction
- 39 tests
