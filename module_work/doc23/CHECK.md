# DOC-23: the user manual's module prose against the code and the prints (CHECK)

Branch `claude/doc-23-manual-prose`, from master `9ae6560` (version 5.34.0).
BACKLOG.md, HANDOFF.md and CLAUDE.md are not edited; the lead writes the notes.
No code, tool text or calculation changed: only `docs/user_manual/manual_content.py`,
the rebuilt `manual.html` / `toc.json`, and `module_work/doc10/check_manual.py`.

## How it was checked

Every sentence of `MODULE_NARRATIVE` (problems, methods, limits) and every
`WORKED` note was set against the module's code, its adapter's `METHOD_INFO`
(descriptions, defaults, the `choices` results list), DESIGN.md / VALIDATION.md,
`validation_examples/RESULTS.md`, and the print-check notes (BACKLOG_DONE
GM-281 to GM-303, GM-237; BACKLOG Done GM-296, GM-299, GM-304 to GM-308, and
each item's `module_work/<dir>/CHECK.md`). Three read-only audit passes were
run in parallel (foundations; retention, slope and fem2d; seismic,
characterization, reliability, structural, I/O), and every finding used here was
re-checked by the lead against the cited file before it was applied (e.g.
`wave_equation/time_integration.py:1-12`, `bearing_capacity/capacity.py:50-52,
152-153, 183`, `axial_pile/capacity.py:282-292`, `downdrag/analysis.py:488-489`,
`slope_stability/geometry.py:158, 354-355`, `fem2d/porewater.py:453-455`,
`seismic_geotech/site_class.py:1-3`, `pyproject.toml` extras at lines 258-282,
`subsurface_characterization/diggs_parser.py:5`, the worked-example JSON).
The worked examples were re-run for retaining_walls (FOS_sliding 1.143),
sheet_pile (D 5.911 m, M 218.1 kN-m/m) and slope_stability (2,300 surfaces,
FOS 1.5831): all match `worked_examples.json`, so no worked number moved.

## Sentences changed (before -> after; what settles it)

### bearing_capacity
| Before | After | Settled by |
|---|---|---|
| N factors "by Vesic, Meyerhof, or Hansen (selectable)"; "AASHTO/GEC-6 factor set" | Two correction-factor sets, Vesic (default; GEC-6 Eq. 5-7, Table 5-2) or Meyerhof (UFC 3-220-20 Table 5-6); N-gamma the set's own or Brinch Hansen's by name | `capacity.py:50-52` FACTOR_METHODS / GENERAL_NGAMMA_METHODS; bearing_factors CHECK ("Brinch Hansen ... not offered as a factor set") |
| Correction factors "(Vesic form)" | Shape GEC-6 Table 5-2; depth by Brinch Hansen's k (default), GEC-6 or none (`depth_factor_method`); inclination EN 1997-1 D.4; base tilt GEC-6 Table 5-5; slope Vesic 1975; phi = 0 adds d_c (UFC Eq. 5-21); Meyerhof refuses tilt/slope | `capacity.py:153`, `factors.py:22, 228-258, 614-645`; GM-281 |
| Terzaghi: "inclined loads and slopes are refused" | ... "inclined loads, slopes and tilted bases are refused, UFC Table 5-3" | `capacity.py:205-215` |
| Two-layer "strong-over-weak / weak-over-strong ... with load spread" | strong over weak by 2V:1H spread; weak over strong a bounded estimate, not a closed form | `capacity.py:698-730` docstring |
| References incl. "AASHTO LRFD" | GEC-6 ch. 5; UFC 3-220-20 ch. 5; EN 1997-1 Annex D; Terzaghi; Meyerhof; Brinch Hansen; Vesic | no bearing_capacity file cites AASHTO LRFD |
| Limit: "applies Vesic depth factors ... AASHTO C_w" | Brinch Hansen's depth factors by default, +17.6 % on GEC-6 Ex B-1 (V-021); `depth_factor_method` "gec6"/"none"; C_w not offered | `capacity.py:153`; RESULTS V-021 |
| Limit: "General shear failure is assumed; local shear not modelled" | Local shear offered with Terzaghi's factors only (UFC Table 5-4); punching not modelled | `capacity.py:183-191` (`failure_mode="local"`) |

The same "Vesic depth factors" phrase in chapter 8's CONVENTION callout and
chapter 9's boundaries list was corrected the same way.

### settlement
- Schmertmann: added NHI-06-089 Eqs. 8-16 to 8-18, the L/B interpolation and
  `gamma_soil` required (GM-285; adapter `gamma_soil` required).
- Westergaard: "read off UFS 3-220-10 Fig. 4-10" (GM-285; the old formula was a
  Boussinesq term).

### axial_pile (the unverified Tomlinson bullet)
| Before | After | Settled by |
|---|---|---|
| "Tomlinson / alpha-method total-stress skin friction ...; 9 cu end bearing" | alpha method (GEC-12 Sec. 7.2.1.3.2, Eq. 7-10): Ca off Fig. 7-17 by su, pile type and D/b; Fig. 7-18 or the smaller on request; API alpha past the curves in a clay called highly overconsolidated, else Ca at the curve's end; a layer's own alpha or Ca with its basis; toe 9 su (Eq. 7-11) | `capacity.py:282-292` (`adhesion_chart="fig_7_17"`, `adhesion_beyond_chart`), `calc_steps.py:308`; GM-237 |
| Limit: "alpha~0.42 vs DrivenPiles ~0.76 at su~1.9 ksf ... (V-002)" | **False**: the simplified curve is gone (GM-121). Now two items: with the example's choices (smooth_steel, chart_end) the V-002 shaft is within 1 %; the defaults (corrugated curve for an H-pile, p. 247) read 13-14 % higher; past the curves the run is refused until `highly_overconsolidated_su_ratio` or the layer's flag is given | RESULTS V-002 rows; `capacity.py:290, 546` |
| Nordlund "(1963/1979) ... Meyerhof q_L limit" | GEC-12 Sec. 7.2.1.3.1 (Eqs. 7-8, 7-9) charts named; sigma'p at the toe limited to 3 ksf (Step 8); q_L off Fig. 7-15 after Meyerhof 1976 | `capacity.py:282` `toe_stress_limit`; axial_pile_print CHECK |
| "beta-method effective-stress skin friction for all layers" | beta and Nt from Table 7-9 where a layer gives none; new bullet: Q_factored (Eq. 7-3, Table 7-1; none for beta), factored uplift Eq. 7-38, FS 2.5 (NHI-06-089 Table 9-5) | GM-282 |
| Sections list; "References: FHWA GEC-12 (FHWA-NHI-16-064)" | adds timber, custom, open pipe Eqs. 7-88 to 7-90; GEC-12 Vol. I (FHWA-NHI-16-009) ch. 7, validated against Vol. III (FHWA-NHI-16-064) Tables D-6, D-100, D-51 (V-001, V-002, V-059) | covers (below); `axial_pile/__init__.py:11` |

### lateral_pile
- References: added FHWA GEC 9 (FHWA-HIF-18-031) App. A and GEC 9 Example B.1
  (pinned, `lateral_pile/validation.py:1123`).

### pile_group
- Equivalent footing "to where the stress falls to 10 % of the footing's" was
  credited to GEC-12 Sec. 7.3.5.3; now UFC 3-220-20 p. 449, with GEC-12 step
  1c's 10 % of the effective stress as `influence_depth_m` (gm286 CHECK).
- Added the group uplift bullet (GEC-12 Sec. 7.2.3.2; UFC Eqs. 6-40 to 6-43;
  GM-289; `pile_group/__init__.py:13`).

### wave_equation
- "characteristic-line time integration" -> lumped masses about 1 m long,
  one cushion spring with the helmet lumped with the first segment, explicit
  central-difference integration (`time_integration.py:1-12`).
- Drivability bullet: `embedded_length`, permanent set (Vol. 2 p. 158), refusal
  120 blows/ft (p. 161), 14 Appendix C hammers, default cushion for Vulcan 014
  only (GM-294, GM-299, GM-301).
- Diesel limit: rigid ram at v = sqrt(2 g h eta); the result's warning (p. 206);
  V-003 numbers (RESULTS). Chapter 9's diesel bullet aligned.

### downdrag
- Limit "the module's intrinsic beta-shaft" -> alpha in clay (UFC Table 6-16,
  Fig. 6-14), Table 7-9 beta in sand; `pile_area` also sets AE and the weight.
- Added: Nt from Table 7-9 or Cheng; pile weight left out by default
  (`include_pile_weight=False`); UFC 10 %-of-bearing-stress below the toe
  (`influence_depth_criterion="bearing_stress"`) (GM-283; `analysis.py:488-489`).

### drilled_shaft
No change: every bullet of the DOC-10 rewrite matches the code. V-008's "0.60
N60" is GEC-10 (2018) Eq. 10-14 and stays.

### sheet_pile
- Log-spiral passive: UFC 3-220-20 Eq. 4-12 by default, Figs. 4-12/4-13 or
  Caquot-Kerisel 1948 as options (adapter `log_spiral_source`).
- Solver bullet: Simplified Method steps with FOS_passive and D = 1.2 D0;
  anchored wall by free earth support; water standing in the excavation
  counts (GM-290, GM-296).
- References: V-013 validates Ka and log-spiral Kp, not the anchored
  embedment (RESULTS: CONVENTION, ~7.2 ft against 6.09 ft); UFC Fig. 4-43 added.
- Limit "continuous per-metre free-earth-support solver ... toe-moment cubic not
  modelled" (**false**) -> the Caltrans Simplified Method per metre run; no
  soldier-pile widths (V-012); not UFC Fig. 4-41's full method (15.6 ft against
  17 ft on Fig. 4-43); moment basis differs from soe (gm290 P1, P3, P4; gm296).

### soe
- "Cantilever and single-anchor free-earth solutions" -> Simplified Method
  cantilever (moment at FS 1.0); free earth support below the lowest of one or
  more supports (GM-193); GEC-4 Sec. 5.3 anchors; Fb 0.60 Fy.
- Basal heave adds Terzaghi, the default (GEC-4 Eqs. 35/37); Bjerrum-Eide Nc off
  Caltrans Fig. 10-15; blowout and piping in the module's own forms (gm290 S9-S11).

### retaining_walls
- "Gravity/cantilever" -> cantilever (semi-gravity) only (no gravity-wall method).
- "Rankine/Coulomb thrust decomposition and Meyerhof bearing" -> thrust on the
  heel plane with Coulomb delta = beta (NHI-06-089 p. 10-20); no tension in
  cohesive backfill; GEC-6 sliding, FS 2.0 with passive; e within B/6 (B/4 on
  rock); q_eq = V/(B - 2e) against a given allowable (GEC-6 Eq. 6-6); optional
  water table (GM-288, GM-292).
- "MSE coherent-gravity" -> GEC-11's Simplified Method, geosynthetics included,
  sloping/broken backslopes and battered faces (GM-297, GM-307).
- References: FHWA-NHI-10-024 / -025 (was -025 only); GEC-6, NHI-06-089, UFC,
  V-061 added.
- Limit `lrfd_external=True` -> the agent tool `mse_lrfd_external_stability`;
  LRFD and broken backslope vertical face only; no MSE water table.

### ground_improvement
- PVD: F(n) is GEC-13 Eq. 2-2 (was attributed to Eq. 2-1) (gm291 CHECK).

### slope_stability
- Strength models add `anisotropic` (adapter `_STRENGTH_MODELS`).
- Seismic displacement adds the yield coefficient and Bray & Travasarou 2007;
  rapid drawdown adds Lowe-Karafiath and the search (adapter METHOD_INFO).
- Limit "not an arbitrary interpolated flow-net grid" (**false**) ->
  `pore_pressure_points` Delaunay TIN since 5.4.0 (RESULTS V-029).
- Limit "tension crack is entry-side" (**false**) -> entry or exit side,
  strength or truncation model (V-026 within 0.5 %). Chapter 9 aligned.
- WORKED note "in the entry-exit search" -> a toe circle governs (1.583), the
  entry-exit minimum 1.587 (worked_examples.json `governing_surface`).

### fem2d
- "Steady/transient seepage" (**false**) -> steady seepage (`porewater.py:455`).
- New footing-capacity bullet: flexible default; frictional q_ult a lower
  bound; mesh-set with no surcharge on sand; rigid not converged; graded mesh for
  clay only (GM-304 to GM-308).
- "a slightly higher FOS than before" -> "than the narrower model" (no referent
  in a manual).

### seismic_geotech
- "ASCE-7 site class" / "References: ASCE 7" (**false**) -> AASHTO site class
  as GEC-12 Vol I Table 7-26 prints it; Fa/Fv Tables 7-28/7-29; Fpga and As
  only with a PGA (Eq. 7-78) (`site_class.py:1-3`; GM-293).
- M-O: KAE UFC Eq. 4-15, dPAE at 0.6 H, KPE GEC-4 Eq. 44 at 0.4 phi by default.
- Liquefaction: NCEER/Youd with CSR and rd as GEC-13 Eq. 5-9; residual strength
  Seed & Harder or Idriss & Boulanger off GEC-12 Fig. 7-78.
- Added the Gmax and modulus-reduction bullet (adapter methods `gmax`,
  `modulus_reduction`).

### liquepy, seismic_signals, subsurface, reliability, pystra, section_props, pynite, dxf, pdf_import, calc_package
- Every "Requires the [X] extra" (liquepy, opensees, pystrata, seismic-signals,
  salib, pystra, dxf twice, pdf, calc/plot/interactive) was **false**: those
  extras are empty aliases and the packages are core (`pyproject.toml`
  258-282). Now "installed with the app". subsurface: GEF core; AGS4 needs
  `[ags4]`, DIGGS validation `[pydiggs]`.
- liquepy "Field correlations (Vs, Dr, G0 from N)" -> CPT correlations (Vs,
  Dr, su/sigma'v, permeability, Ic) (`field_correlations.py:27-44`).
- seismic_signals "Fourier processing" -> bandpass filtering, baseline
  correction, PGV and PGD (eqsig).
- subsurface "20 test types" -> 18 (the parser's own docstring).
- reliability V-030 "scope edge" (stale; RESULTS says closed by `linear_su`)
  replaced; wrappers called library functions (not agent tools).
- pystra "full distribution library" -> the eight distributions the tool takes;
  "port of FERUM" -> built on PyRe and FERUM.
- section_props "steel and generic shapes" -> the five parametric shapes and
  polygons (no steel-shape table).
- pynite "nonlinear/dynamic problems belong to the OpenSees module" -> no
  nonlinear frame analysis; PyNiteFEA left out on numpy 1.26.
- dxf_import / dxf_export "reinforcement", "nails/anchors" -> soil nails.
- pdf_import: the tool has no vision extraction or grid overlay (planlens and
  the legacy agent do); its cross-check takes a vision result made elsewhere.
- calc_package "about a dozen" -> 17 (`_IMPORT_MAP`); "PDF needs pdflatex"
  (**false**) -> PyMuPDF HTML-engine fallback; tables list adds soe and
  reliability. Chapter 6's dxf/pdf/dxf_export/calc paragraphs aligned.

### Chapters outside the module entries (same false claims)
- Chapter 2: "the optional extras pull in the third-party libraries" and the
  "[full]" callout replaced (empty aliases since 5.8.1; three real extras).
- Chapter 3: "the 21 reference modules" -> "the reference modules".
- Chapter 8: KPIs 42/75/53 -> 63/107/79 (RESULTS.md: 63 distinct problem IDs,
  107 result rows, 79 with a PASS verdict; INVENTORY.md has 64 entries); the
  "V-001...V-040" range dropped from the intro and the section title.
- Chapter 9: slope, diesel, bearing and soe/sheet_pile bullets aligned.

## The references table (REFERENCE_INFO), from the covers

Each PDF in `geotech-references/docs` was opened with PyMuPDF and its cover
and report-documentation page read. Rows now give the designation and date as
printed:

| Module | Was | Now (cover) |
|---|---|---|
| dm7 | NAVFAC DM7.01 / DM7.02 | UFS 3-220-10 (DM 7.1), 16 Jan 2026; UFC 3-220-20 (DM 7.2), 16 Jan 2025 |
| gec4 | FHWA-IF-99-015 | FHWA-IF-99-015, June 1999 (unchanged number) |
| gec5 | FHWA-NHI-10-016, "soil and rock properties" | FHWA NHI-16-072, April 2017, Geotechnical Site Characterization |
| gec6 | FHWA-SA-02-054 | FHWA-SA-02-054, September 2002 |
| gec7 | FHWA-0-IF-03-017 | FHWA-NHI-14-007, February 2015 |
| gec8 | FHWA-NHI-13-026 | FHWA-HIF-07-03, April 2007 |
| gec9 | FHWA-NHI-16-009, "driven piles" | FHWA-HIF-18-031, April 2018, laterally loaded deep foundations |
| gec10 | FHWA-NHI-18-024 | FHWA NHI-18-024, September 2018 |
| gec11 | FHWA-NHI-10-025 | FHWA-NHI-10-024 (Vol. I) and -025 (Vol. II), November 2009 |
| gec12 | FHWA-NHI-16-064 | FHWA-NHI-16-009 (Vols. I and II, both covers), July 2016; Vol. III is FHWA-NHI-16-064, September 2016 |
| gec13 | "FHWA ground-modification manual" | FHWA-NHI-16-027 (Vol. I) and -028 (Vol. II), April 2017 |
| gec14 | "FHWA-NHI ground-improvement" | FHWA-HIF-17-016, August 2016, Assuring Quality in Geotechnical Reporting Documents |
| micropile | FHWA-NHI-05-039 | FHWA NHI-05-039, December 2005 |
| ufc_backfill | UFC 3-220-04N | UFC 3-220-04FA, 16 January 2004 |
| ufc_expansive | UFC 3-220-07 | UFC 3-220-07, 16 January 2004 |
| ufc_pavement | UFC 3-260-02, airfield | UFC 3-250-01, Pavement Design for Roads and Parking Areas, 14 November 2016 |
| fema_p2082 | NEHRP provisions | FEMA P-2082-1, 2020 Edition Vol. I, September 2020 |
| california_trenching | Caltrans T&S Manual | June 2011, Revision 2, July 2025 |
| fhwa_pavements | FHWA-NHI-05-037 | FHWA NHI-05-037, May 2006 |

Thirteen reference modules were missing from the table and are added, each
from its cover: fhwa_soils_foundations (NHI-06-088/089, Dec 2006),
eurocode_7_1 (EN 1997-1:2004), eurocode_7_2 (EN 1997-2:2007), aashto_1993
(the cover is a scan with no date; "1993" from the package), em_2104 (1 Nov
2023), em_2107 (1 Aug 2022), ufc_structural (UFC 3-301-01, 11 Apr 2023,
Change 4, 3 Jun 2025), ufc_collapse (UFC 4-023-03, 14 Jul 2009, Change 4, 10
Jun 2024), gsa_collapse (24 Oct 2013, Rev. 1, 28 Jan 2016), ufc_flexible_practice
(UFC 3-250-03, O&M Manual: Standard Practice for Flexible Pavements, 30 May
2018), ufc_concrete_practice (UFC 3-250-04, 29 Jan 2024), ufc_stabilization
(UFC 3-250-11, 30 Nov 2020), and wood_handbook (FPL-GTR-282, 2021; **its PDF is
not in docs/, so this row is from the package docstring, not a cover**).
Because the Appendix A loop treats any module not in REFERENCE_INFO as an
analysis module, those 13 had been printed in Appendix A with full parameter
tables; they now sit in the Chapter 5 and Appendix A reference tables with a
method count, like the other references. That is most of the 1,345 deleted
lines of `manual.html`.

One audit pass said GEC-12 Vol. II is FHWA-NHI-16-010; its cover reads
FHWA-NHI-16-009 (checked again), so the table says 16-009 for both volumes.

## Rebuild and drift from master

`python build_manual.py` (version still 5.34.0, so the cover does not move);
a second build gives the same bytes (`check_manual.py --rebuild` reports both
files unchanged). Before the edits, `--rebuild` on master's files showed three
changed lines of drift, all from master's tool text since the last rebuild: the
"documented parameters" box (5,792 -> 5,808) and the `mse_wall` and
`retaining_wall_package` parameter tables (GM-307's `face_batter`,
`wall_type='mse_lrfd'`, hand-edited at the time). The rebuild picks those up;
everything else in the diff is this item's rows. `toc.json` changes only in
section 8.1's title.

## `check_manual.py --rebuild`

New options: `--rebuild` builds the manual in memory with
`build_manual.build()` and compares it byte for byte with the committed
`manual.html` and `toc.json`, printing a unified diff (80 lines per file) and
exiting 1 on any difference; `--write` with it writes the rebuild in place.
The name checks still run after it.

**Not put in CI.** It needs no network (the worked examples come from the
committed JSON, the catalog from the registry), and two runs here gave the
same bytes, but the catalog is introspected from the installed environment:
an optional package that will not import on a host can hide a method or change
a description, so the result can differ between CI's install and the laptop's.
**For the lead, as a release step:** after the version bump, in the release
venv, run `python module_work/doc10/check_manual.py --rebuild`; if it exits 1,
run it again with `--write` (or `python docs/user_manual/build_manual.py`),
read the diff, and commit the rebuilt `manual.html` and `toc.json` with the
release. (Rebuilding after the bump is what moves the cover version.)

## Tests

- `python module_work/doc10/check_manual.py`: exit 0 (70 modules, 195
  analysis methods, 2,638 parameters, 1,105 registered defaults);
  `--rebuild`: both files unchanged, exit 0.
- `pytest webapp/tests/test_public_text.py webapp/tests/test_docs_currency.py`:
  11 passed, exit 0.

## Left / noted, not fixed

- Chapters 1, 3, 4 and 7 were read only for the claims above, not sentence by
  sentence; chapter 2's extras table still lists the empty-alias extras (its
  "Pulls in" column, generated from pyproject, shows them empty).
- VALIDATION_ROWS: V-001 (toe under Step 8 is CONVENTION, -13 %) and V-015
  (Bishop row CONVENTION) could read "PASS / CONVENTION"; V-020's "GEC-13 Eq.
  2-1" is not in INVENTORY.md. Not changed (table rows, not module prose).
- The calc_package adapter's own `format` text still says PDF "requires
  pdflatex installed" (tool text; not changed here by rule).
- The PDF itself is not rebuilt (DOC-10's open item).
- Optional sharpenings not taken: subsurface `spt_correction` /
  `stress_dilatancy`, opensees backbone curves, section_props warping only for
  I-sections and circles, aggregate piers named as GEC-13 Eq. 5-4.

## Release note

The user manual's module descriptions now match the code and the printed
references, and its references table gives each manual's designation and date
as printed on its cover. No calculation changes.
