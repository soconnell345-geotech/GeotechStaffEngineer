# GM-309: Meyerhof's own depth factor beyond D_f/B = 1, as a cited option

Branch `claude/gm-309-meyerhof-depth`. Source: DR-11 slice 3 review,
finding F5 (`module_work/dr11/CHECK_SLICE3.md`).

## What the print says

**UFC 3-220-20, Foundations (16 January 2025), Table 5-6 "Bearing and
Correction Factors for the Meyerhof (1963) Method", p. 294** (PDF page
index 331 of `geotech-references/docs/ufc_3_220_20_2025.pdf`, read off the
page with PyMuPDF; the library's text, `dm7_2/text/chapter05.json`
Sec. 5-3.3.2, and `dm7_2.chapter5.meyerhof_depth_factors` say the same):

- Depth, phi = 0: d_c = 1 + 0.2 k (K_P)^0.5, d_q = 1, d_gamma = 1
- Depth, phi' > 10 deg: d_q = 1 + 0.1 k (K_P)^0.5, d_gamma = 1 + 0.1 k (K_P)^0.5
  (the d_c cell spans both rows)
- Notes: K_P = tan^2(pi/4 + phi'/2), **k = Df / B**, with no limit on Df/B.

Brinch Hansen's k = arctan(Df/B) for Df/B >= 1 is printed only in Tables
5-7 and 5-8 note B (pp. 295-296), for the Brinch Hansen factors.

Not checked further: FHWA NHI-06-089 ch. 8 and DM 7.02 were not needed
once UFC 3-220-20 was found to print the form (the DM 7.02 successor is
UFC 3-220-20 itself).

## What changed

- `bearing_capacity/factors.py`: `DEPTH_FACTOR_METHODS` gains `"linear"`
  with its basis in `DEPTH_FACTOR_BASIS`; `depth_factors(...,
  depth_method="linear")` takes k = Df/B at every depth; refused with
  `method="vesic"` (Table 5-8 note B gives Vesic/Hansen the arctan).
- `bearing_capacity/capacity.py`: `depth_factor_method="linear"` accepted
  with `factor_method="meyerhof"` only (ValueError otherwise). The "depth
  factors" choice reports it (`how` = given, basis = Table 5-6, k = Df/B at
  every depth) and lists the other methods as alternatives. With the
  default, the old alternative line "k = Df/B ... not an option of this
  tool" is replaced by the `'linear'` option's line with its values.
- `bearing_capacity/calc_steps.py`: the depth-factor equation reads
  "k = D_f/B at every depth"; the reference cites Table 5-6 alone.
- `funhouse_agent/adapters/bearing_capacity.py`: `allowed_values` and the
  parameter description. The calc package (`bearing_capacity_package`)
  runs the tool's own code path and parameter docs, so it takes the option
  with no change.
- `bearing_capacity/DESIGN.md`; the three `depth_factor_method` rows of
  `docs/user_manual/manual.html` (hand-edited, no rebuild).
- Default (`"hansen"`) unchanged.

## Before and after (made-up footing: square 1.6 m, Meyerhof)

| soil | Df | Df/B | method | d_c | d_q | q_ult kPa |
|---|---|---|---|---|---|---|
| phi 34, gamma 18.5 | 4.0 | 2.5 | hansen (default) | 1.448 | 1.224 | 4,373.0 |
| phi 34, gamma 18.5 | 4.0 | 2.5 | linear | 1.940 | 1.470 | 5,253.1 |
| phi 34, gamma 18.5 | 0.96 | 0.6 | either | 1.226 | 1.113 | 1,482.1 |
| c 60, phi 0 | 4.0 | 2.5 | hansen (default) | 1.238 | 1.000 | 532.2 |
| c 60, phi 0 | 4.0 | 2.5 | linear | 1.500 | 1.000 | 629.1 |

By hand at phi 34: K_P = tan^2 62 deg = 3.5371, sqrt 1.8807; d_c = 1 + 0.2
x 2.5 x 1.8807 = 1.9404; d_q = 1 + 0.1 x 2.5 x 1.8807 = 1.4702.

## Tests

- New `bearing_capacity/tests/test_meyerhof_linear_depth.py` (23 tests):
  by hand at Df/B 2.5 (phi 34, phi 0) and 0.6 (phi 28); linear equals the
  default up to Df/B = 1 (12 cases); refused with Vesic; the analysis
  carries the factors; choices report linear as given and the default
  lists linear; Vesic does not list it; calc steps; the adapter.
- `test_own_ngamma.py::test_meyerhof_states_the_depth_parameter`: its last
  alternative is now the `'linear'` option, not the "not an option" note.
- `bearing_capacity`: 393 passed.
- 2,280 tests over the funhouse_agent files that name bearing_capacity
  (dispatch, native tools, defaults_used, method aliases, top-level
  params, calc-package parity/adapter/assumed inputs, render tables,
  citations), `calc_package/tests`, `validation_examples/test_published_v021_v022.py`,
  `report_ingest` recompute slices 1-3 and calc reader,
  `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  2,278 passed, 26 skipped (pdflatex, pydiggs, python-ags4 missing),
  **2 failed, both failing identically on master e8065ac in this cloud
  environment** (not this change):
  `test_citations.py::test_every_cited_number_is_in_its_manual` (GEC-12
  table D-*, GEC-10 eq A-2, GEC 9 table A-* not in the installed
  geotech-references chapter files; the PyPI refs 1.5.0, not the pinned
  submodule) and `test_defaults_used.py::...[profile_figure|plot_data]`.
- Page fingerprints pass without re-recording (the bearing tool
  description is not in a fingerprinted page build), so no fingerprint
  commit. `module_work/doc10/check_manual.py`: OK.

## Release note

Default unchanged. New option: `bearing_capacity_analysis` (and
`bearing_capacity_package`) take `depth_factor_method="linear"` with
`factor_method="meyerhof"`: Meyerhof's (1963) depth factors with k = Df/B
at every depth, as UFC 3-220-20 Table 5-6 (p. 294) prints them. Up to
Df/B = 1 it equals the default; beyond it the depth factors and q_ult are
larger (square 1.6 m at Df 4.0 m, phi 34 deg: q_ult 4,373 -> 5,253 kPa).
The depth choice lists what the option gives.

## Left / noted, not fixed

- **Changes saved output, for the lead:** `report_ingest/recompute.py`
  still runs only the module's `'hansen'` depth factors and `'none'` as
  bearing options, so the 2 F5 rows (Meyerhof at Df/B = 2.5) still read
  'differs'. Adding `'linear'` as a Meyerhof option there would move them
  to 'within_options'. Not done here (outside this item's scope).
- Noted, not fixed: Table 5-6 prints depth rows for phi = 0 and phi' > 10
  deg only; for 0 < phi <= 10 the module keeps d_q = d_gamma = 1 under
  both methods, as before.
