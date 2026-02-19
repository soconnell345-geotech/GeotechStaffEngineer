# liquepy Agent — Design Notes

## Purpose

Wraps the liquepy library for CPT-based liquefaction triggering analysis
using the Boulanger & Idriss (2014) simplified procedure. Complements the
existing `seismic_geotech` module which provides SPT-based triggering
(Youd et al. 2001).

## Architecture

```
liquepy_agent/
  __init__.py              # exports public API
  liquepy_utils.py         # has_liquepy(), import helpers
  cpt_liquefaction.py      # analyze_cpt_liquefaction()
  field_correlations.py    # analyze_field_correlations()
  results.py               # 2 result dataclasses
  tests/
    test_liquepy_agent.py
  DESIGN.md
liquepy_agent_foundry.py   # Foundry wrapper (project root)
```

## Key Design Decisions

1. **Optional dependency** — liquepy is checked with `has_liquepy()` following
   the same pattern as opensees_agent and pystrata_agent. Tier 1 tests work
   without liquepy installed.

2. **numpy 2.0 workaround** — liquepy's `calc_ldi()` uses the removed
   `np.trapz`. We implement `_calc_ldi_safe()` using `np.trapezoid` instead
   of calling liquepy's broken function directly.

3. **Two public functions**:
   - `analyze_cpt_liquefaction()` — full triggering + post-triggering
   - `analyze_field_correlations()` — Vs, Dr, su/σv', permeability from CPT

4. **All units SI/kPa** — consistent with project conventions. liquepy uses
   kPa natively for most functions (exception: Robertson 2009 Vs uses Pa).

## Boulanger & Idriss (2014) Procedure

The B&I 2014 CPT-based simplified procedure:

1. Correct cone tip resistance: `q_t = q_c + (1 - a_ratio) * u_2`
2. Compute stress profile: σ_v, u, σ'_v from unit weights
3. Normalize cone resistance: Q, F, I_c (iterative)
4. Clean sand correction: Δq_c1n from fines content
5. Compute CSR: `CSR = 0.65 * (σ_v/σ'_v) * a_max * r_d / MSF / K_σ`
6. Compute CRR from q_c1n_cs (Eq. 2.24)
7. Factor of safety: `FoS = CRR / CSR`

## Post-Triggering Indices

| Index | Formula | Reference | Threshold |
|-------|---------|-----------|-----------|
| LPI | Σ w·F·Δz, w=10-0.5z, F=1-FoS if FoS<1 | Iwasaki et al. 1982 | >15 = high risk |
| LSN | Σ (εv/z)·Δz·10 | van Ballegooy et al. 2014 | >30 = severe |
| LDI | ∫ γ dz | Zhang et al. 2004 | Lateral displacement (m) |

## Post-Triggering Strains

- **Volumetric strain** (Zhang et al. 2002): function of FoS and q_c1n_cs
- **Shear strain** (Zhang et al. 2004): function of FoS and relative density
- **Relative density** (Zhang 2002): `Dr = (-85 + 76·log10(q_c1n)) / 100`

## Vs Correlations

| Method | Reference | Notes |
|--------|-----------|-------|
| mcgann2015 | McGann et al. 2015 | `Vs = 18.4·qc^0.144·fs^0.0832·z^0.278` |
| robertson2009 | Robertson 2009 | Function of I_c, σ'_v, q_t; uses Pa units |
| andrus2007 | Andrus et al. 2007 | Function of I_c, depth, q_t |

## Edge Cases

- **Above GWL**: FoS set to 2.0 (not liquefiable, no pore pressure)
- **I_c > limit**: FoS set to 2.25 (claylike behavior, not liquefiable)
- **Very shallow depths**: r_d close to 1.0, unit weight assumed = gamma_predrill
- **Zero u_2**: Default when no piezocone data; affects q_t correction minimally
  for most soils (significant for clays with high u_2)

## Comparison: liquepy Agent vs seismic_geotech

| Feature | liquepy Agent | seismic_geotech |
|---------|---------------|-----------------|
| Input data | CPT (q_c, f_s, u_2) | SPT (N) |
| Triggering | B&I 2014 | Youd et al. 2001 |
| Soil classification | I_c (SBT) | Fines content (manual) |
| Post-triggering | Zhang strains, LPI, LSN, LDI | Residual strength only |
| Field correlations | Vs, Dr, su, permeability | None |
| Continuous profile | Yes (every CPT reading) | No (discrete SPT values) |
