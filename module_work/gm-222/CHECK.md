# GM-222: fem2d seepage flow is a flow through a section

Branch `claude/gm-222-seepage-flow`, 2026-11-05.

## The problem

`fem2d.analyze_seepage` (tool `fem2d_seepage`) reported
`total_flow_m3_per_s_per_m` = Σ |v|·A·t over the elements. That is a
volume-weighted speed, not a flow: it has units of m³/s × m, grows with the
slice thickness t and with the size of the mesh's domain, and is not the
flow through any section. The calc package printed it as "Q" and the
seepage plot as "Q = ... m³/s per m".

## What the print says

UFS 3-220-10, Soil Mechanics (DM 7.1), 16 January 2026 (held in
geotech-references as `docs/ufc_3_220_10_2026.pdf`):

- **Eq. 6-3 and 6-4, p. 286** (Sec. 6-2.2): Darcy's law q = k i A; for a
  region of constant height and unit width the flow per unit length is
  q = k i y.
- **Eq. 6-7, p. 287** (Sec. 6-2.3): steady two-dimensional flow is the
  Laplace equation (what fem2d solves).
- **Eq. 6-9, p. 290** (Sec. 6-2.4.2): the flow through a flow net section,
  q = k h_L (N_f / N_d) W, W often a unit width. So "the flow" of a 2-D
  seepage problem is the flow per unit width across the section from the
  upstream (inflow) boundary to the downstream one.
- **Figure 6-5 "Flow Net Example Calculations", p. 292**, on the flow net of
  **Figure 6-3, p. 288** (impervious dam or cofferdam on isotropic
  permeable soil over an impervious base): h_L = 36 ft, k = 1×10⁻³ cm/s =
  2.8 ft/day, permeable soil 40 ft thick, embedment 20 ft, N_f = 3,
  N_d = 9: **q = 2.8 × 36 × 3/9 × 1 = 34 ft³/day per ft**. The total head
  upstream is 76 ft with the datum at the base of the permeable soil (the
  same example's pore-pressure step), so 40 ft downstream.
- Not printed: the dam's width. It scales 20 ft off Figure 6-3 (139 px
  against 280 px for the printed 40 ft of soil; the printed 20 ft
  embedment scales 19.6 ft the same way). The soil is drawn about 60 ft
  either side of the dam's centre.

## What changed and why

1. `fem2d/porewater.py`: `solve_seepage` keeps the prescribed nodal flows
   q, solves as before (penalty heads), and calls the new
   `seepage_boundary_flow(H, head, q, head_bcs, t)`. With H assembled
   **without** the head constraints, the reaction at a fixed-head node is
   R_i = (H h)_i − q_i: the flow that crosses the boundary there (row i of
   the Galerkin equations). R / t is per metre run. Returned:
   - `flow_rate` / `flow`: the inflow, Σ R_i > 0 (the outflow instead where
     a prescribed inflow, a recharge well, makes it the larger);
   - `inflow`, `outflow`, `prescribed_flow`, `nodal_flux`;
   - `flow_balance`: |inflow + prescribed − outflow| / max(in, out) against
     `FLOW_BALANCE_TOLERANCE` = 1e-6, and `passed`. The interior rows are
     met to the direct solver's round-off (measured 1e-13 to 1e-11 on the
     meshes below); 1e-6 leaves room for large k contrasts and still
     catches a boundary that leaks;
   - `flow_boundaries`: per prescribed head value, its node count and net
     flow (+ in, − out);
   - `flow_measured_on`: a sentence naming the boundary set measured.
2. `fem2d/results.py` `SeepageResult`: `total_flow_m3_per_s_per_m` removed;
   `flow_m3_per_s_per_m`, `outflow_m3_per_s_per_m`, `flow_balance`,
   `flow_boundaries`, `flow_measured_on` added (summary and `to_dict`).
   The old key was dropped, not kept: every caller is in this repo
   (`grep -rn total_flow`: analysis, calc_steps, plotting, the adapter's
   description, one test, DESIGN.md) and each is updated; keeping the name
   would have kept the misreading.
3. `fem2d/analysis.py` `analyze_seepage`: passes the new fields; docstring
   says what the flow is and that t changes nothing.
4. `fem2d/calc_steps.py` (calc package): the seepage section now has the
   head solution (Eq. 6-7) and a "Flow Through the Section" step citing
   Eq. 6-3 and 6-9 with the boundary sentence, a per-boundary flow table,
   the outflow and the balance check in the summary. The old step cited
   "Smith & Griffiths (2004)", which is not in the reference library.
5. `fem2d/plotting.py`: the plot label reads "q = ... (inflow at fixed
   heads)".
6. `funhouse_agent/adapters/fem2d_adapter.py`: `fem2d_seepage`'s `t`
   description no longer says the flow grows with t; `returns` describes
   the five new keys. `docs/user_manual/manual.html` carries the same `t`
   text (a build artifact; the one string edited to match).
7. `fem2d/DESIGN.md`: the GM-204 note that left the flow scaling with t,
   and the Steady-State Seepage section, describe the new flow.
8. `fem2d/tests/test_seepage_flow.py` (new, 9 tests): exact 1-D Darcy flow
   and boundary table; independence of t; the Figure 6-5 case; `to_dict`;
   a prescribed inflow leaving at the fixed heads.

## Before and after

**Figure 6-5** (`_figure_6_3` in the new test: SI, k = 1e-5 m/s as printed
in cm/s, dam 20 ft wide, soil ±60 ft, CST grid). Printed q = 34 ft³/day per
ft = 3.656e-5 m³/s per m.

| Mesh | t | Before: Σ\|v\|At (reported as m³/s/m) | After: q (ft³/day/ft) | After / printed | Balance mismatch |
|---|---|---|---|---|---|
| 2 ft | 1.0 | 1.058e-3 (= 985 "ft³/day/ft") | 32.38 | 0.952 | 2.6e-13 |
| 1 ft | 1.0 | 1.056e-3 | 32.24 | 0.948 | 1.2e-12 |
| 2 ft | 2.5 | 2.646e-3 | 32.38 | 0.952 | - |
| 1 ft | 2.5 | 2.640e-3 | 32.24 | 0.948 | - |
| 0.5 ft | 1.0 | - | 32.18 | 0.947 | 7.0e-12 |

Before: about 29 times the printed flow at t = 1, and 2.5 times more at
t = 2.5. After: unchanged by t to 1e-12, converged with the mesh.

Sensitivity to what the print does not fix (1 ft mesh): soil ±120 ft,
33.05 (0.972); ±240 ft, 33.06; dam width 16 ft, 34.58; 24 ft, 30.19.

**Tolerance accepted: 10 %.** The printed q is a hand-drawn net counted in
whole flow channels (N_f = 3, so one channel is a third of the flow), the
printed k is rounded (1e-3 cm/s = 2.835 ft/day, printed 2.8, q 33.6
printed 34), and the dam's width is not printed: ±4 ft on the scaled
20 ft moves the FEM flow by about ±7 %. The FEM sits 5 % below the printed
value on the drawn geometry and 3 % below with the far sides moved out.

**1-D Darcy** (12 m × 3 m, k = 2e-5 m/s, heads 8 and 5 m): q = k i y =
1.5e-5 m³/s/m; the module gives it to 1e-9 relative, inflow and outflow
both.

## Tests run (Python 3.12 venv, `pip install -e .`)

- `pytest fem2d calc_package -n 8`: 855 passed, 3 skipped (pdflatex not installed), 0 failed.
- `pytest fem2d/tests/test_seepage_flow.py fem2d/tests/test_groundwater.py`:
  59 passed, 0 failed.
- `pytest webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  19 passed, 0 failed. The fingerprints did not change (the fem2d method
  catalog is not part of a page fingerprint), so nothing was re-recorded.
- `pytest validation_examples/test_published_v023_v024.py` (the one example
  file that imports fem2d; it does not use seepage): 13 passed, 0 failed.
- Adapter: `call_agent('fem2d', 'fem2d_seepage', ...)` on the
  `method_probes.json` probe (unit square, unit gradient, k = 1e-5) returns
  `flow_m3_per_s_per_m` = 1e-5, balance passed.

## Release note

fem2d seepage (`analyze_seepage`, tool `fem2d_seepage`): the result's
`total_flow_m3_per_s_per_m` is replaced by `flow_m3_per_s_per_m`, the flow
through the section per metre run: the inflow across the fixed-head
boundaries, from the nodal reactions at the constrained-head nodes, with
`outflow_m3_per_s_per_m`, a `flow_balance` check (inflow = outflow within
1e-6), the flow at each prescribed head (`flow_boundaries`) and the
boundary set named (`flow_measured_on`). The old value summed |v| × element
area × t; it grew with t and was not a flow (about 29 times the flow on
the UFS 3-220-10 Figure 6-5 example). Calc packages and seepage plots print
the new flow. Heads, pore pressures and velocities are unchanged.

## Noted, not fixed

- `seepage_heads` (T3/T6/Q4, used by the dewatered excavation) reports no
  flow at all; the excavation result does not print one, so nothing in a
  saved report moves. `seepage_boundary_flow` could serve it if a flow is
  ever wanted there.
- `solve_seepage` still uses the penalty method (1e20) for the heads; the
  reactions are taken from the unconstrained H, so the penalty does not
  enter the flow.
- `SeepageResult` carries no `choices` / `defaults_used` of its own (the
  adapter adds `defaults_used`); the flow basis is stated in
  `flow_measured_on`.
