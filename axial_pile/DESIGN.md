# axial_pile — Driven Pile Axial Capacity

## Purpose
Computes the nominal axial resistance (shaft + toe) of a driven pile by the
static analysis methods of FHWA GEC-12 (FHWA-NHI-16-009, 2016) Chapter 7:
Nordlund in cohesionless soil, the alpha-method in cohesive soil (method
"auto", by layer), or the effective stress (beta) method in every layer;
open-end pipe plugging; uplift.

## References
- FHWA GEC-12 Vol 1 (FHWA-NHI-16-009), Sec. 7.2.1.3 (static analysis
  methods), 7.2.3.1 (uplift), 7.10.7 (open pile sections). Citations below
  are to its PRINTED page numbers (PDF page index = printed page + 45).
- GEC-12 Vol 3 (FHWA-NHI-16-064), Appendix D: the worked example used for
  validation (V-001 Table D-6, V-002 Table D-100).
- Nordlund (1963, 1979); Tomlinson (1979, 1994); API RP 2A (1993);
  Paikowsky and Whitman (1990); Fellenius (1991) for beta from phi'.

## Files
- `nordlund.py` — the Nordlund charts and equations: `displaced_volume`,
  `delta_phi_ratio_fig_7_9`, `k_delta`, `correction_factor`,
  `nordlund_side_factors`, `skin_friction_cohesionless` (Eq. 7-8),
  `alpha_t`, `n_q_prime`, `_limiting_tip_resistance` (Fig. 7-15),
  `toe_limit_chart_note`, `end_bearing_cohesionless` (Eq. 7-9 with the
  Step 8 and Step 9 limits).
- `tomlinson.py` — the alpha-method: `adhesion_fig_7_17`, `api_alpha`
  (Eqs. 7-13/7-14), `adhesion` (the chart, at most su; past it the
  curve's end or API alpha), `alpha_tomlinson`, `skin_friction_cohesive` (Eq. 7-10),
  `end_bearing_cohesive` (Eq. 7-11).
- `beta_method.py` — `beta_from_phi`, `Nt_from_phi`, `skin_friction_beta`,
  `end_bearing_beta`.
- `capacity.py` — `AxialPileAnalysis` (the orchestrator, its choices and
  its GEC-12 options).
- `calc_steps.py` — the calc-package steps; `results.py` —
  `AxialPileResult` with summary()/to_dict()/get_tables().
- `pile_types.py`, `soil_profile.py` — sections and layers.

## Public API
```python
AxialPileAnalysis(pile, soil, pile_length, method="auto",
                  factor_of_safety=2.5, include_uplift=False,
                  cohesive_phi=25.0, uplift_skin_fraction=1.0,
                  pile_weight=None, head_depth=0.0,
                  toe_stress_limit=143.64, inside_friction_ratio=1/3,
                  open_pipe_toe_limit=5027.4, open_pipe_method="auto",
                  cohesive_Nt=3.0, h_pile_toe_area="box",
                  adhesion_curve="auto", adhesion_beyond_chart="chart_end",
                  displaced_volume=None).compute() -> AxialPileResult
```
`AxialSoilLayer` takes, besides thickness, soil_type, unit_weight,
friction_angle and cohesion: `toe_friction_angle` (a separate toe phi),
`delta_phi_ratio` (None = Fig. 7-9), and for the beta method `beta` and `Nt`
(local experience, Steps 2a/5a). The result carries `choices` (method,
given/default/module, basis, alternatives), `toe_details`, `plug_weight`,
`warnings` and a per-layer breakdown with each layer's chart values.

## GEC-12 Chapter 7, step by step (GM-121 audit, 2026-10-18)

### Nordlund (Sec. 7.2.1.3.1, Steps 1-11, pp. 242-245)
| Step | GEC-12 | Module |
|---|---|---|
| 2a | V, displaced volume per unit length | `displaced_volume`: the steel area for an H-pile (Fig. 7-9 curve f; Table D-6 uses it), the gross area otherwise, an open pipe included (GEC-12 gives no V for an open pipe; its steel area is below curve a); or given (`displaced_volume`, m3/m). Reported. |
| 2b | delta/phi from Fig. 7-9 by pile type and V | Fig. 7-9 curves a (pipe), b (timber), c (precast concrete), f (H-piles); held at the curve's end with a note outside it; a layer's `delta_phi_ratio` wins. Before GM-121: 0.75 steel / 0.90 concrete, no V. |
| 3 | K_delta from Tables 7-6/7-7, linear in phi, log-linear in V (Steps 3c-3e) | As printed at V = 0.1, 1, 10 ft3/ft, phi 25-40; held with a note outside. Before: a fit that ignored V. |
| 4 | C_F from Fig. 7-14 by phi and delta/phi | Fig. 7-14 measured off the printed figure. Before: a linear fit on delta/phi. |
| 5-6 | sigma'd at the layer midpoint, no limit; fs = K_delta C_F sigma'd sin delta (Eq. 7-8); H-piles on the box perimeter | As stated; midpoint rule split at the water table. No limit on sigma'd or fs (GEC-12 gives none). |
| 7 | alpha_t from Fig. 7-16(a) by phi and D/b (D/b 20, 30, 45); N'q from Fig. 7-16(b) | Both measured off the printed figure; D/b below 20 or above 45 takes the nearest curve. Before: alpha_t saturating at 1.0, no phi. |
| 8 | "The limiting value of sigma'p is 3 ksf" (p. 244; Nordlund, p. 234) | Applied by default (`toe_stress_limit` 143.6 kPa, None = no limit). Before: not applied (a 25 m sand run used 475 kPa). |
| 9 | Rp = alpha_t N'q sigma'p Ap at most qL Ap (Fig. 7-15); the toe area of an H or open pile "based on past experience and local correlations" | qL from Fig. 7-15 (re-measured 2026-10-08), held at the chart's ends with a warning. H-pile toe area: box by default (`h_pile_toe_area`), as Sec. 7.10.7 (p. 408) uses in soil. |
| 7c | toe phi averaged from the toe to about 3b below | Not done: the toe layer's phi (or its `toe_friction_angle`). Backlog candidate. |
| p. 232 | widths over 18 in were outside the correlation database; Nordlund tends to over predict over 24 in | Warnings on the result. |

### alpha-method (Sec. 7.2.1.3.2, Steps 1-6, pp. 246-250)
| Step | GEC-12 | Module |
|---|---|---|
| 1 | Ca from Fig. 7-17 by su, pile type and D/b; D from the ground surface to the bottom of the clay layer, or to the toe if less (p. 246) | Fig. 7-17 measured off the printed figure: smooth steel (dashed) and concrete/timber/corrugated (solid) curves at D = 10b and 40b, linear in D/b between, held outside. Before: one simplified alpha-vs-cu curve, no D/b. |
| p. 247 | H-piles: the corrugated pile curve times the box area | `adhesion_curve` "auto": pipe piles smooth steel, H, concrete and timber piles the solid curve; either can be given. |
| p. 247, Step 1 | su past the charts (highly overconsolidated clays): alpha by API RP 2A recommended (Eqs. 7-13/7-14) | `adhesion_beyond_chart` "chart_end" by default (Ca held at the curve's end, as GEC-12's worked example, Vol 3 Table D-100, did), or "api". At the chart's end API alpha makes Ca jump (solid curves at su 3.45 ksf, sigma'v 4 ksf: 0.83-1.37 ksf to 1.86 ksf). Before review: "api" by default. |
| p. 252, p. 247 | API alpha at most 1.0 ("An alpha value of 1.0 is recommended for unconsolidated clays"); su on an H-pile's soil-to-soil faces | Ca at most su within the chart (Fig. 7-17 as drawn passes above Ca = su at low su: the D = 40b solid curve to about 1.5 ksf). Stated in `choices`; uncapped is not offered. |
| 4 | qp = 9 su (Eq. 7-11) | As stated. |
| p. 246-247 | Fig. 7-18 for three stratigraphy cases; where Fig. 7-17 or 7-18 could be used, "the inexperienced user should select and use the smaller value"; no shaft resistance in a shrink-swell zone | Not done; Fig. 7-18 is named in `choices` as an alternative. Backlog candidates. |

### beta method (Sec. 7.2.1.3.4, pp. 255-258)
beta and Nt from local experience first (a layer's `beta`, `Nt`), else from
phi' (beta = (1 - sin phi') tan phi'; Nt within Table 7-9's ranges); a
cohesive toe takes `cohesive_Nt` (default 3: "the use of a relatively low Nt
coefficient in clays is recommended", p. 256; before GM-121 an Nt from the
25 deg clay phi, about 22); no limiting values (GEC-12 gives none). Beta from
Table 7-9 by soil class (sand 0.30-0.90) is a backlog candidate.

### Open-end pipes (Sec. 7.10.7, pp. 405-408)
- The lesser of plugged (Eq. 7-88: fso Aso + qp App) and unplugged (Eq. 7-89:
  fso Aso + fsi Asi + qp Ap - Wp) governs (Paikowsky and Whitman, p. 406).
- fsi = `inside_friction_ratio` x fso, default 1/3, the low end of GEC-12's
  "typically on the order of 1/3 to 1/2" (p. 407). Before: equal to fso.
- Wp, the effective weight of the soil plug from the head to the toe, is
  subtracted in the unplugged case. Before: not subtracted.
- In cohesionless soil the unit toe resistance is limited to 105 ksf
  (Tomlinson 1994, p. 407; `open_pipe_toe_limit`, None = none).
- A pipe under 24 in in predominantly cohesive soil uses Eq. 7-90
  (Rn = 0.8 Ca As + 4.5 su Ap): `open_pipe_method` "auto" takes it when the
  toe is in clay and more than half the embedded length is in clay (the
  module's reading of "predominantly"); "paikowsky_whitman" or "tomlinson"
  force one; uplift then takes the reduced shaft (0.8 Ca As). A note says
  plugging is uncertain at 24 in and over.

### Uplift (Sec. 7.2.3.1, pp. 288-289)
The nominal shaft resistance with phi_up (Eq. 7-38): "the shaft resistance
does not need to be further reduced for the uplift case".
`uplift_skin_fraction` default 1.0 (before: 0.75); 0.7, the customary share
in free-draining soils De Nicola and Randolph report, is the alternative.
Side resistance inside an open pipe is left out (under Eq. 7-90 the
shaft is its 0.8 Ca As); the pile weight is added when given.

## Chart provenance
Figs. 7-9, 7-14, 7-16(a), 7-16(b) and 7-17 were measured off the printed
GEC-12 Vol 1 figures on 2026-10-18 (pixel reads against each figure's own
gridlines, overlaid on the figure to check); Fig. 7-15 on 2026-10-08;
Tables 7-6/7-7 transcribed as printed. Against Vol 3 Table D-6
(DrivenPiles, HP12x74, V = 0.151 ft3/ft) the charts give fs/sigma'd 0.454 at
phi 33 and 0.601 at phi 36 (table: 0.447, 0.592).

## Datum
Model the profile from the ground surface and put the pile head at its
depth with `head_depth`: layers above the head give no shaft resistance,
and the soil above it still counts in sigma'v. GEC-12's Vol 3 example
counts it (Table D-6's toe at 0.01 ft below the footing is 16.4 kips), and
Fig. 7-17's D runs from the ground surface. Start the profile at the
footing only where the soil above it is gone for good.

## Validation
- V-001 (Table D-6): shaft +1.1 % to +1.2 %; toe with phi 40 and no sigma'p
  limit +0.3 %; with Step 8 (default) 359.7 kips, -16 %, because the
  example did not apply Step 8.
- V-002 (Table D-100): toe 9 su within 2 %; shaft within 1 % with the
  example's choices (smooth steel curves, Ca held at the chart's end); +13 %
  to +14 % with the defaults (corrugated curve, p. 247; Ca held at the
  chart's end); +25 % to +53 % with API alpha past the chart.
See `validation_examples/RESULTS.md`.

## Not done (engineering choices or gaps; backlog candidates from GM-121)
- V for an open pipe: the gross area by default; the steel area (driven
  unplugged) lies below Fig. 7-9 curve a, and Nordlund's database had no
  open end pipes (p. 232).
- Step 7c's toe zone (phi averaged to about 3b below the toe).
- Fig. 7-18 stratigraphy cases and the shrink-swell zone (alpha-method).
- beta and Nt from Table 7-9 by soil class rather than from phi'.
- Piles driven into rock (Sec. 7.2.1.4) and the API method for
  large-diameter open-end pipes (Sec. 7.2.1.3.3, 36 in and over).
- `capacity_vs_depth` in the agent tool takes the module's defaults only.
- Integration: midpoint rule per layer, split at the water table (exact for
  Eq. 7-8 and the beta method, linear in sigma'v).
