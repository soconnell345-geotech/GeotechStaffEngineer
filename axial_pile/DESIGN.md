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
  (Eqs. 7-13/7-14), `adhesion` (the chart as drawn; past it the
  curve's end or API alpha), `alpha_fig_7_18` and `adhesion_fig_7_18`
  (Fig. 7-18's three stratigraphy cases), `alpha_tomlinson`,
  `skin_friction_cohesive` (Eq. 7-10), `end_bearing_cohesive` (Eq. 7-11).
- `beta_method.py` — `beta_from_phi`, `Nt_from_phi`, `TABLE_7_9` and
  `table_7_9` (beta and Nt by soil type), `skin_friction_beta`,
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
                  adhesion_curve="auto",
                  adhesion_beyond_chart="api_if_highly_overconsolidated",
                  highly_overconsolidated_su_ratio=None,
                  displaced_volume=None, adhesion_chart="fig_7_17",
                  shrink_swell_depth=None, beta_coefficients="from_phi",
                  toe_phi_zone="toe_layer").compute() -> AxialPileResult
```
`AxialSoilLayer` takes, besides thickness, soil_type, unit_weight,
friction_angle and cohesion: `toe_friction_angle` (a separate toe phi),
`delta_phi_ratio` (None = Fig. 7-9), for the beta method `beta` and `Nt`
(local experience, Steps 2a/5a) and `soil_class` (Table 7-9's soil type),
and for a clay layer `fig_7_18_case` (Fig. 7-18's stratigraphy case).
The agent tool also takes a custom section (`pile_area`, `pile_perimeter`,
`pile_width`, and `pile_tip_area` for an H-pile or a closed pipe) for any
pile type but an open pipe; a timber pile is always one (GM-144). The result carries `choices` (method,
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
| 8 | "The limiting value of sigma'p is 3 ksf" (p. 244; Nordlund, p. 234) | Applied by default (`toe_stress_limit` 143.6 kPa, None = no limit), stated in `choices` as "sigma'p at the toe (Nordlund, Step 8)" with p. 244; the owner kept it (OWN-63, 2026-10-20). Before: not applied (a 25 m sand run used 475 kPa). |
| 9 | Rp = alpha_t N'q sigma'p Ap at most qL Ap (Fig. 7-15); the toe area of an H or open pile "based on past experience and local correlations" | qL from Fig. 7-15 (re-measured 2026-10-08), held at the chart's ends with a warning. H-pile toe area: box by default (`h_pile_toe_area`), as Sec. 7.10.7 (p. 408) uses in soil. |
| 7c | "If phi angle is estimated from SPT data, compute the average corrected SPT N value, (N1)60, over the zone from the pile toe to 3 diameters below the pile toe" | `toe_phi_zone`: "toe_layer" (default; the toe layer's phi or `toe_friction_angle`, as GEC-12's worked examples, Tables D-6 and D-51, took it) or "average_3b" (the thickness-weighted average of the cohesionless layers' toe phi from the toe to 3b below, the profile's last layer continuing; cohesive soil in the zone is left out of the average with a note). The module takes phi, not N, so it averages phi (the same where phi is linear in N). Used for Fig. 7-16 and Fig. 7-15 alike (Step 9a.2). GM-144. |
| p. 232 | widths over 18 in were outside the correlation database; Nordlund tends to over predict over 24 in | Warnings on the result. |

### alpha-method (Sec. 7.2.1.3.2, Steps 1-6, pp. 245-250)
| Step | GEC-12 | Module |
|---|---|---|
| 1 | Ca from Fig. 7-17 by su, pile type and D/b; D from the ground surface to the bottom of the clay layer, or to the toe if less (p. 246; the figure's legend, p. 248) | D as p. 246 gives it, stated in `choices` ("D for Fig. 7-17 (alpha method)") with pp. 246 and 248; the owner kept it (OWN-63). Fig. 7-17 measured off the printed figure: smooth steel (dashed) and concrete/timber/corrugated (solid) curves at D = 10b and 40b, linear in D/b between, held outside. Before: one simplified alpha-vs-cu curve, no D/b. |
| p. 247 | H-piles: the corrugated pile curve times the box area | `adhesion_curve` "auto": pipe piles smooth steel, H, concrete and timber piles the solid curve; either can be given. The H-pile basis is stated in `choices` with p. 247; the owner kept it (OWN-63). |
| p. 247, Step 1 (p. 250) | su past the charts in highly overconsolidated clays: alpha by API RP 2A recommended (Sec. 7.2.1.3.3, Eqs. 7-13 to 7-15, p. 251); GEC-12 gives no su / sigma'vo at which a clay becomes highly overconsolidated | `adhesion_beyond_chart` "api_if_highly_overconsolidated" by default: API alpha in a clay layer past the curves whose su / sigma'vo (sigma'vo at the middle of the layer's length along the pile) is at or above `highly_overconsolidated_su_ratio`, Ca held at the curve's end below it. The line is the engineer's judgment (owner, OWN-63, 2026-10-20), so a run with such a layer and no line is refused (`JudgmentNeeded`, a ValueError) with the layer's su / sigma'vo in the message. The 1.0 in Eqs. 7-13/7-14 only chooses between API's two expressions and is not offered as the line. "chart_end" (every such layer, as GEC-12's worked example, Vol 3 Table D-100, did) and "api" (every such layer) need no line. `choices` says, per run of layers, which side of the line each fell on. At the chart's end API alpha can make Ca jump (solid curves at su 3.45 ksf, sigma'v 4 ksf: 0.83-1.37 ksf to 1.86 ksf). GM-229; before it: "chart_end" by default. |
| Steps 1-2 (p. 250), Fig. 7-17 (p. 248); p. 252 | Ca from Fig. 7-17 and fs = Ca (Eq. 7-10), with no limit on Ca; API alpha at most 1.0 ("An alpha value of 1.0 is recommended for unconsolidated clays", p. 252) belongs to the API method | Fig. 7-17 as drawn: Ca may exceed su (alpha about 1.1 at su 1.0 ksf on the D = 40b solid curve, 1.04 at the solid curve's start), stated in `choices` ("Ca above su (alpha method)") with how many layers and the largest alpha. API's alpha <= 1.0 applies only where API alpha is taken. GM-229 (owner, OWN-63); before it Ca was held at su within the chart (GEC-12's Example 9-3 of NHI-06-089, WE-SF-2: shaft 715.3 kN at alpha 1.0, 761.9 kN as drawn). |
| 4 | qp = 9 su (Eq. 7-11) | As stated. |
| p. 246-247 | Fig. 7-18 (Tomlinson 1980): alpha against su for three stratigraphy cases; Fig. 7-17 generally; where either could be used, "the inexperienced user should select and use the smaller value" | `adhesion_chart`: "fig_7_17" (default), "fig_7_18" or "smaller" (the smaller Ca of the two, layer by layer). A clay layer's case is its `fig_7_18_case`, else from the profile: contiguous cohesive layers are one clay stratum; under a cohesionless layer the top graph (sands or sandy gravels over stiff clay), from the ground surface the bottom graph (stiff clay without different overlying strata); the middle graph (soft clay over stiff clay) only where a layer gives it, since GEC-12 gives no su that separates a soft clay from a stiff one. A given case starts a stratum at the layer's top. D for Fig. 7-18 is the penetration into the clay (the figure's sketches): from the top of the stratum to the bottom of the layer, or to the toe if less; linear in D/b between the curves, the nearest held outside them (below the middle and bottom graphs' 10b curve with a note: alpha rises with penetration, so it is overstated there). Past the curves (5.0 ksf; 4.56-4.76 ksf in the middle graph) `adhesion_beyond_chart` as for Fig. 7-17. GM-144. |
| p. 247 | "In clays with large shrink-swell potential, static resistance calculations should ignore the shaft resistance from the adhesion in the shrink-swell zone" | `shrink_swell_depth` (m below the ground surface, default none): no shaft resistance in cohesive soil above it, in the beta method too (the gap is the same). GM-144. |

### beta method (Sec. 7.2.1.3.4, pp. 255-258)
beta and Nt from local experience first (a layer's `beta`, `Nt`), else from
phi' (beta = (1 - sin phi') tan phi'; Nt within Table 7-9's ranges); a
cohesive toe takes `cohesive_Nt` (default 3: "the use of a relatively low Nt
coefficient in clays is recommended", p. 256; before GM-121 an Nt from the
25 deg clay phi, about 22); no limiting values (GEC-12 gives none).
`beta_coefficients` "table_7_9" (GM-144) reads Steps 2b and 5b ("use Table
7-9 to estimate beta coefficient from phi' angle") as Table 7-9 by the
layer's soil type (`soil_class`; clay for a cohesive layer and sand for a
cohesionless one by default), linear in phi' across the type's phi' range
and held at its ends with a note: clay 25-30 deg, beta 0.15-0.35, Nt 3-30;
silt 28-34, 0.25-0.50, 20-40; sand 32-40, 0.30-0.90, 30-150; gravel 35-45,
0.35-0.80, 60-300 (p. 256). A clay toe keeps `cohesive_Nt` either way; a
silt toe takes the table's Nt.

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
  module's reading of "predominantly", which GEC-12 does not measure; the
  result's `choices` state it with the clay share, "predominantly cohesive
  soil (Eq. 7-90)", GM-144); "paikowsky_whitman" or "tomlinson" force one;
  Ca is from Fig. 7-17 as Eq. 7-90 prints it (p. 407), whatever
  `adhesion_chart` is (that chart still sets Eqs. 7-88/7-89);
  uplift then takes the reduced shaft (0.8 Ca As). A note says plugging is
  uncertain at 24 in and over.

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
Fig. 7-18 on 2026-10-19 (the embedded 150 dpi raster, pdf page 294: each
curve's stroke traced column by column between the fitted gridlines, su
every 0.2 ksf and alpha every 0.05, about 0.004 in alpha a pixel, and
overlaid on the figure to check); Tables 7-6/7-7 and 7-9 transcribed as
printed. Against Vol 3 Table D-6
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
  to +14 % with the defaults and a highly overconsolidated line of 1.0
  (corrugated curve, p. 247; Layer 3's su / sigma'vo 0.32-0.41 is below the
  line, so Ca is held at the chart's end); +25 % to +53 % with API alpha in
  every layer past the chart. With no line the default run is refused
  (GM-229). No V-002 layer reads alpha above 1, so the as-drawn reading moves
  none of it.
See `validation_examples/RESULTS.md`.

## Not done (engineering choices or gaps)
- V for an open pipe: the gross area by default; the steel area (driven
  unplugged) lies below Fig. 7-9 curve a, and Nordlund's database had no
  open end pipes (p. 232). GEC-12 gives none.
- Piles driven into rock (Sec. 7.2.1.4, pp. 269-275: soft rock by the soil
  methods or Eq. 7-34 with Fig. 7-22's bearing capacity factors; hard rock
  structural) and the API method for large-diameter open-end pipes (Sec.
  7.2.1.3.3, pp. 251-255, 36 in and over: Table 7-8's delta, limiting fs,
  Nq and limiting qp by density class, K 0.8 unplugged / 1.0 plugged).
- No GEC-12 worked example (Vol 3) uses Fig. 7-18, Table 7-9, the 3b toe
  zone or the shrink-swell zone: the DrivenPiles runs take Fig. 7-17 and
  the toe layer's phi, so the GM-144 options are checked against the
  printed figure, table and text, not against an example.
- Integration: midpoint rule per layer, split at the water table and at a
  shrink-swell depth (exact for Eq. 7-8 and the beta method, linear in
  sigma'v).

The agent tool's `capacity_vs_depth` takes the same GEC-12 options and
custom section as `axial_pile_capacity` (GM-144) and returns the choices of
its deepest run.
