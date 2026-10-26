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

2. **numpy 2.4 workaround** — liquepy's `calc_ldi()` uses `np.trapz`,
   removed in numpy 2.4. We implement `_calc_ldi_safe()` with `trapezoid`
   from `geotech_common.numpy_compat` (numpy 1.26 and 2, INF-53) instead
   of calling liquepy's broken function directly.

3. **Two public functions**:
   - `analyze_cpt_liquefaction()` — full triggering + post-triggering
   - `analyze_field_correlations()` — Vs, Dr, su/σv', permeability from CPT

4. **All units SI/kPa** — consistent with project conventions. liquepy uses
   kPa natively for most functions (exception: Robertson 2009 Vs uses Pa).

5. **Site inputs have no default** (BACKLOG GM-133, 2026-10-17). The water
   table (`gwl`) and, for triggering, the peak ground acceleration (`pga`)
   are required; a call without one is refused with the reason. Until then
   `analyze_cpt_liquefaction` took 1.0 m and 0.25 g, and
   `analyze_field_correlations` 1.0 m, without saying so (the unified
   `liquefaction` tool did not even declare them). The water table sets
   σ'v in CSR = 0.65 (σv/σ'v) amax rd (Youd et al. 2001; B&I 2014) and
   which soil is saturated; GEC 5 (FHWA-NHI-16-072, p. 12-23) has the site
   investigation "establish the location of the groundwater level", with
   piezometers where it fluctuates; liquepy itself takes it with no
   default (`field.CPT`, and `run_bi2014`'s "depth to ground water from
   surface at time of earthquake"); and the SPT routes always required it.
   A CPT result lists the water table among its `choices` as given, and
   the CRR curve's C0 = 2.8 (liquepy's deterministic value, not an input
   on this route) as the module's.

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

## Depth profile for the agent's tool results (GM-201)

`cpt_liquefaction.depth_profile(depth, factor_of_safety, volumetric_strain)`
gives the CPT's FS_liq and post-liquefaction volumetric strain by depth (the
strain is liquepy's `calc_volumetric_strain_zhang_2002`: Zhang, Robertson
and Brachman (2002), "Estimating liquefaction-induced ground settlements
from CPT for level ground", Canadian Geotechnical Journal 39(5), 1168-1180,
doi 10.1139/t02-047, the doi liquepy gives) as
rows of `top_m`, `bottom_m`, `FOS_liq`, `volumetric_strain`; the
`liquefaction` tool's CPT route returns them as `depth_profile`, and
`downdrag.post_liquefaction_downdrag` takes that result as it is. Each
reading stands for the depth halfway to its neighbours. Up to
`PROFILE_MAX_ROWS` (40) readings give one row each; a longer CPT (hundreds
of readings) is summarised in depth bands of whole readings so the result
stays under the agents' 8,000-character tool-result cap (about 5,100
characters measured for 600 and 1,500 readings):

- a band never holds readings on both sides of FS_liq = 1.0 (GEC-10
  §9.4.2.2: "FSliq ≤ 1.0" liquefies), so at that threshold the liquefied
  depths are the readings' own; only when the readings cross 1.0 more often
  than there are rows are the thinnest runs ABOVE 1.0 counted in the
  liquefied bands around them, and `depth_profile_note` gives that
  thickness. A reading at or below 1.0 is never put in a band above it,
  however thin its layer: a lens one reading thick stays in a liquefied
  band (review of GM-201: the first build merged the thinnest run of
  either kind, and a noisy 25 m CPT lost 1.0 m of liquefied readings into
  bands above 1.0);
- the rest of the rows split the runs into bands of about equal thickness;
- `FOS_liq` is the band's lowest reading on its side of 1.0, rounded UP to
  three decimals (so 1.0004 never reads as 1.0), and `volumetric_strain` is
  the band's thickness-weighted average (strain x thickness = the band's
  settlement), never below zero (Zhang's interpolation dips a hair below
  zero where liquepy caps FS at 2).

The SPT routes compute no post-liquefaction volumetric strain: no reference
in the library prints an SPT method (see `downdrag/DESIGN.md`,
"Post-liquefaction downdrag"), and liquepy's Zhang et al. (2002) is CPT
only.

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
