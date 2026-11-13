# GM-310: the default changes from the 2026-11-12 judgement calls

Branch `claude/gm-310-judgement-call-defaults`, from master 2feea3d.
Source, binding: `module_work/owner_calls_2026-11-12/CALLS.md` section B
(OWN-84, OWN-85, OWN-82, OWN-68, OWN-108, OWN-112, OWN-114, OWN-118,
OWN-131) and, from section A, OWN-105. Nothing in `slope_stability/`.
OWN-19 (moment-curvature EI) is GM-311 and is not here.

Every number below is from a made-up case, run on master (before) and on
this branch (after) through the tool layer (`funhouse_agent.dispatch.call_agent`)
unless it says otherwise.

## The ten changes

| # | Call | Module | What the print says | What changed | Before | After |
|---|------|--------|---------------------|--------------|--------|-------|
| 1 | OWN-84 | `drilled_shaft` | GEC-10 (FHWA-NHI-18-024) p. 10-22: N*cr = 2.5 (Eq. 10-23) is recommended only where the rock below the base is massive or tightly jointed to one diameter below it, with no cavities and a clean base; for a fractured rock mass the Hoek-Brown base, Eqs. 10-24 to 10-29 (pp. 10-23, 10-24). | A rock layer's RQD left out is no longer read as 100 for the base. Base on rock with no RQD, no `rock_base_conditions_met` and no GSI with m_i: refused, naming the three. | 1.2 m shaft, 8 m clay (cu 75) over rock (qu 20 MPa), base at 14 m, no RQD: Q_ult 89,722 kN (Eq. 10-23) | refused, naming RQD, `rock_base_conditions_met`, GSI and m_i |
| 2 | OWN-85 | `retaining_walls` (MSE) | GEC-11 Vol. I Table 3-6 note (i), p. 3-17: the empirical F* is acceptable for Table 3-1 fill with Cu >= 4; "For fill outside these limits, tests must be run." | `Cu_backfill` below 4 with no `F_star_tested`: refused, asking for `F_star_tested`, whatever the reinforcement. | 6 m strip wall, Cu 2.5: F* 1.541 at the top, FOS_pullout 1.959, flagged in the choices | refused; with `F_star_tested` 1.2 it runs (FOS_pullout 1.526, unchanged) |
| 3 | OWN-82 | `sheet_pile` | UFC 3-220-20 Fig. 4-37 step 1 (p. 237): "Passive pressures for clean coarse grained soils include wall friction (see Table 4-1). For active or passive pressures in all other soil types, ignore wall friction." Table 4-1 (p. 184): clean sand 17 deg, clean gravel 22 deg. | Under `log_spiral` a layer that names no wall friction takes none (its Kp is Rankine's) and the choice says "not marked clean sand or gravel". A layer marked `interface_soil` 'clean_sand' (17) or 'clean_gravel' (22) takes the table's value, as before. Coulomb's delta stays 0. | 4 m cantilever, sand phi' 32, water at the excavation level, `log_spiral`, nothing marked: D 6.08 m, M 173.9 kN-m/m (17 deg taken on the passive side) | D 8.65 m, M 251.7 kN-m/m; marked 'clean_sand': D 5.67 m, M 147.2 (unchanged) |
| 4 | OWN-68 | `geotech_common` (`SoilProfile.to_lateral_pile_input`, the lateral-pile shortcut) | GEC-9 p. 219: stiff clays (Reese et al. 1975) are those of 1,000 to 2,000 psf (47.9 to 95.8 kPa); p. 216: soft clays 250 to 500 psf. COM624P's stiff clay Tables 3.2 and 3.3 start at 0.5 ton/ft2 (47.9 kPa). | `STIFF_CLAY_CU` 100 -> 47.9 kPa (1,000 psf). `stiff_clay_cu` stays a setting. | 60 kPa clay: `SoftClayMatlock` | `StiffClayAboveWT`. Same clay through the lateral-pile tool, 0.6 m pipe, 100 kN at a free head: head deflection 4.3 mm (Matlock) against 3.5 mm (Welch and Reese) |
| 5 | OWN-108 | `ground_improvement` | GEC-13 (FHWA-NHI-16-027): stone columns "between 2.0 and 5.0" measured (p. 5-31), "a ratio of 2.5 is often conservatively used" for preliminary design (p. 5-32); rammed aggregate piers 5 to 10 under flexible embankment loading, 10 under rigid footings (p. 5-43). | n has no default. New `column_type`: 'stone_column' 2.5, 'rammed_aggregate_pier' 5. Neither given where n is used: refused, naming `column_type` (and n). | a_s 0.2, S 100 mm, no n: n 5, SRF 0.5556, 55.6 mm | refused; 'stone_column': n 2.5, SRF 0.7692, 76.9 mm; 'rammed_aggregate_pier': n 5, SRF 0.5556, 55.6 mm |
| 6 | OWN-112 | `geotech_common` (`spt_overburden_correction`; the soil profile's phi from N1,60; the `spt_correction` tool) | GEC-6 Fig. 5-18 (p. 90) draws Liao and Whitman's C_N to "2.0 max."; GEC-12 Vol. I p. 108 Eq. 5-3 and NHI-06-088 Eq. 3-3 (p. 3-56) print Peck's. The 1.7 cap is Youd et al. (2001), not held. | Liao and Whitman's cap 1.7 -> 2.0. Peck's correction, at most 2.0, was already offered and stays. The liquefaction procedures do not use this function. | N 10: sigma'v 20 kPa, C_N 1.70, N1,60 17.0; 30 kPa, 1.70, 17.0; 40 kPa, 1.581 | 20 kPa, 2.00, 20.0; 30 kPa, 1.826, 18.3; 40 kPa, 1.581 (unchanged at and above 34.6 kPa) |
| 7 | OWN-114 | `seismic_geotech` | FHWA NHI-06-089 p. 10-19: a planar passive wedge is unreliable for wall friction above phi'/3. UFC 3-220-20 Fig. 4-8 (p. 187): Coulomb's Kp "unconservative for delta/phi' > 0.4". | KPE's wall friction when `delta_passive` is not given: 0.4 phi -> phi/3. The warning above 0.4 phi stays. | phi 30, kh 0.2: delta 12.0 deg, KPE 3.7307 | delta 10.0 deg, KPE 3.4995 |
| 8 | OWN-118 | `pile_group` | UFC 3-220-20 Table 6-28 note A (p. 434): factors reduced by 20 % "for small pile groups (<5 piles)". GEC-12 Vol. I p. 225: "less than 3 piles per substructure unit in this manual". | Driven piles take UFC's fewer than 5 (GEC-12's fewer than 3 before); the choice gives GEC-12's value and the block at it for 3 or 4 piles. | 2 x 2 driven piles, block 2,609 kN: phi_ug 0.50, factored block 1,304.6 kN | phi_ug 0.40, 1,043.7 kN; alternative "GEC-12 ... phi_ug 0.50, the block 1304.6 kN" |
| 9 | OWN-131 | `bearing_capacity` | UFC 3-220-20 Table 5-6 (p. 294), Meyerhof: k = D_f/B, no limit printed. The arctangent form is printed for Brinch Hansen only (Table 5-8 note B, p. 295). | With `factor_method` 'meyerhof' a left-out `depth_factor_method` is 'linear' (his Table 5-6 at every depth); 'hansen' stays the option. Beyond D_f/B = 1 the result warns (Python warning and in the depth choice) that the factors grow without limit and gives Brinch Hansen's factors and q_ult. Vesic's default is unchanged. | 1.6 m square footing at 4 m (D_f/B 2.5), phi 34, gamma 18.5, Meyerhof: d_q 1.2239, q_ult 4,373 kPa | d_q 1.4702, q_ult 5,253 kPa (+20 % on this case, a footing 2.5 times deeper than wide; the warning gives 4,373 kPa beside it). D_f/B 0.8: unchanged |
| 10 | OWN-105 | `downdrag` | UFC 3-220-20 Sec. 6-7.4 step 3 (p. 477): the neutral plane at the intersection of the load and resistance curves. Sec. 6-5.8.4.3 (p. 449): the plane where pile and soil settle equally. (Pages as CALLS.md and the module's own cites give them; the sections are in the library's `dm7_2/text/chapter06.json`.) | Summary text only. Where the headline is the intersection and the curves meet below the depth at which the soil stops settling more than the pile, `summary()` gives the settlement-compatible plane, drag load, maximum load and drag force under the headline numbers, and `to_dict()` carries the same words as `settlement_compatible_beside`. | 18 m steel H-pile, 3 m fill over 10 m soft clay over sand, Q_dead 200 kN (library call): headline plane 14.21 m; the 12.97 m plane only in the comparison table | the same headline, and under it "Neutral plane depth: 12.97 m ... Drag force: 307.0 kN" with both section cites. No number changes |

## Calls made (for the owner to revisit)

1. **OWN-84, a rock base with no RQD but with GSI and m_i.** The call
   lists three inputs that settle it. With GSI and m_i given and no RQD
   and no statement, the base is now the Hoek-Brown base (Eq. 10-29), not
   Eq. 10-23 with Hoek-Brown beside it. Before, the defaulted RQD of 100
   selected Eq. 10-23 here too (89,722 kN against 47,378 kN on the probe).
   An RQD given as 100 still selects Eq. 10-23.
2. **OWN-84, `rock_base_method` 'rqd'.** This base is a function of the
   RQD, so it is refused when the tip rock layer gives none (it took 2.5 qu
   at the defaulted 100 before).
3. **OWN-84, which layers.** The refusal looks at every rock layer within
   one base diameter below the base, the same zone the RQD-below-100 check
   already used.
4. **OWN-82, how a layer is "marked".** The existing `interface_soil`
   input is the mark: 'clean_sand' 17 deg, 'clean_gravel' 22 deg (the
   table's own row for gravel, not 17), used on both sides of the wall as
   before. The call's "17 degrees" is the clean sand row. The choice gives,
   on the clean sand row's alternative, the embedment with the unmarked
   cohesionless layers marked clean sand.
5. **OWN-68, 47.9 not 48.0.** The default is 47.9 kPa, 1,000 psf as GEC-9
   and COM624P print it (the call's "48 kPa" rounded), so a clay entered
   as 47.9 or 48 kPa is stiff.
6. **OWN-108, when n is not used.** With a Fig. 5-21 settlement method and
   no bearing asked, n enters nothing; then neither n nor `column_type` is
   required, the result reports n as 0.0 and the summary says "not given,
   not used". A given n always wins over `column_type`.
7. **OWN-108, 5 for a rammed aggregate pier.** The low end of the printed
   5 to 10, as the call says.
8. **OWN-131, the low-level `depth_factors` function** takes the same
   default by method as the analysis (it took 'hansen' for Meyerhof).
9. **OWN-105, the tool result.** The agent's tool returns `to_dict()`, not
   `summary()`, so the same words are also in a new result key,
   `settlement_compatible_beside`, present only when it applies.

## What changed, by commit

One commit per default (code, tests, `choices` and `defaults_used`, tool
description, DESIGN.md, manual row). No fingerprints commit was needed
(see "Tests"). The commits, in order:

- `92cef5d` GM-310 (OWN-84): a drilled shaft's base on rock with no RQD is refused, naming the inputs that settle it
- `fd6f6f1` GM-310 (OWN-85): MSE backfill with Cu below 4 and no tested F* is refused
- `bc65d78` GM-310 (OWN-82): sheet piles take passive wall friction under log-spiral only for a layer marked clean sand or gravel
- `1f9a172` GM-310 (OWN-68): a clay takes the stiff-clay p-y curve from 47.9 kPa in the soil-profile shortcut
- `1392f51` GM-310 (OWN-108): aggregate piers take the stress ratio n by column type, with no one default
- `360709b` GM-310 (OWN-112): Liao and Whitman's SPT overburden correction is capped at 2.0
- `42c01c4` GM-310 (OWN-114): Mononobe-Okabe KPE takes wall friction phi/3 when none is given
- `e738693` GM-310 (OWN-118): the small-group cut in uplift starts below 5 driven piles
- `2337f06` GM-310 (OWN-131): with Meyerhof's method the depth factors follow his own table at every depth
- `1391464` GM-310 (OWN-105): the downdrag summary gives the settlement-compatible plane beside the headline
- `cd00c05` GM-310 (OWN-105): the settlement-compatible sentence in the downdrag tool result reads as one sentence
- this file, last.

What each touched:

- **OWN-84** `drilled_shaft/soil_profile.py` (`RQD` optional, `RQD_given`
  kept through `dataclasses.replace`), `drilled_shaft/capacity.py`
  (`_rock_rqd_not_given`, the refusals, the conditions choice),
  `funhouse_agent/adapters/drilled_shaft.py`,
  `geotech_common/soil_profile.py` (`to_drilled_shaft_input` passes an
  RQD not given as None), DESIGN.md, tests.
- **OWN-85** `retaining_walls/mse.py` (`CU_BELOW_4_REFUSAL`, raised in
  `check_internal_stability`; the tested-F* choice says no empirical F* is
  worked out below Cu 4), adapter, DESIGN.md, tests.
- **OWN-82** `sheet_pile/cantilever.py` (`wall_friction`,
  `clean_sand_where_left_out` in place of `smooth_where_left_out`, the
  choice), `sheet_pile/anchored.py`, `sheet_pile/calc_steps.py`, adapter,
  DESIGN.md, tests.
- **OWN-68** `geotech_common/soil_profile.py` (`STIFF_CLAY_CU`, the
  boundary choice), `geotech_common/DESIGN.md`, `lateral_pile/DESIGN.md`,
  tests.
- **OWN-108** `ground_improvement/aggregate_piers.py` (`COLUMN_TYPE_N`,
  `N_NEEDS_COLUMN_TYPE`, `column_type`), `results.py`, adapter
  (`column_type`, `when_absent` and `applies_when`), DESIGN.md, tests,
  the foundry harness scenario.
- **OWN-112** `geotech_common/soil_properties.py` (`OVERBURDEN_METHODS`),
  `geotech_common/soil_profile.py` (the estimate's note),
  `funhouse_agent/adapters/subsurface_adapter.py`, tests.
- **OWN-114** `seismic_geotech/mononobe_okabe.py`
  (`passive_delta_choice`), adapter, DESIGN.md, tests.
- **OWN-118** `pile_group/group_efficiency.py` (`small_group_reduction`),
  adapter, DESIGN.md, tests.
- **OWN-131** `bearing_capacity/factors.py`
  (`default_depth_factor_method`), `capacity.py` (`_deep_meyerhof_note`,
  the warning), adapter (`default_depends_on`), DESIGN.md, tests.
- **OWN-105** `downdrag/results.py` (`settlement_compatible_beside`),
  adapter (`returns`), DESIGN.md, test.

## Probes, worked examples, validation examples and fixtures that move

Every probe of the changed modules was run on master and on this branch
and the scalar results compared (43 probes in `method_probes.json`); the
`docs/user_manual` examples and the before/after probes above the same way.

**Probes (`funhouse_agent/tests/method_probes.json`):**

| Probe | What happened | What was done |
|---|---|---|
| `drilled_shaft\|drilled_shaft_capacity#rock` | refused as written (rock with no RQD) | the rock layer now gives `"RQD": 100.0`; its answer is unchanged (Q_ult 89,722.1 kN). Its `rock_base_method` LABEL_ONLY reason reworded |
| `ground_improvement\|aggregate_piers` | refused as written (no n, no column type) | now gives `"column_type": "stone_column"`; n 5 -> 2.5, SRF 0.6562 -> 0.8358, composite modulus 7,619 -> 5,982 kPa |
| `ground_improvement\|aggregate_piers#by_moduli` | `stress_concentration_ratio` 5.0 -> 0.0 | none: n is not used there (Fig. 5-21 settlement, no bearing) and none is given; nothing else moves |
| `sheet_pile\|cantilever_wall#log_spiral_left_out` | answer moves: D 3.838 -> 5.226 m, M 119.1 -> 151.7 kN-m/m (delta 17 -> 0) | none to the probe; `log_spiral_source` is now LABEL_ONLY for it (at delta 0 every source gives Rankine's Kp), reached by the `#log_spiral` and `#interface_soil` probes |

No other probe of `drilled_shaft`, `retaining_walls` (MSE),
`sheet_pile`, `ground_improvement`, `subsurface` (spt_correction),
`seismic_geotech`, `pile_group` (group_uplift), `bearing_capacity`,
`downdrag` or `lateral_pile` moves.

**User manual worked examples (`docs/user_manual/worked_examples.json`,
`manual.html`):** `ground_improvement.aggregate_piers` was refused and
now gives `column_type="stone_column"` (outputs listed under "User
manual"); `seismic_geotech.seismic_earth_pressure` moves (KPE 3.7307 ->
3.4995). The other examples of the changed modules do not move
(bearing capacity is Vesic's, the sheet pile Rankine's, the drilled shaft's
base is in sand, the downdrag example gains only the new
`settlement_compatible_beside` key, which the manual does not print and
whose JSON entry was left as it was).

**Validation examples (`validation_examples/`):** none moves; the suite
passes unchanged.

**Worked-example corpus (`funhouse_agent/worked_examples.json`) and the
eval suite (`funhouse_agent/geotech_test_suite.json`):** no call in either
uses a changed default (searched for aggregate piers, `include_passive`,
Meyerhof, `spt_correction`, `group_uplift`, `log_spiral`, `Cu_backfill`,
rock layers).

**Tests whose fixtures needed the input a refusal now asks for** (the
numbers they assert are unchanged unless noted):

- rock layers given `RQD=100.0`: `drilled_shaft/tests/` (`test_calc_tables`,
  `test_concrete_bearing_base`, `test_drilled_shaft`, `test_fractured_rock`,
  `test_gec10_2018_printed`, `test_hoek_brown_base`,
  `test_steps_match_the_analysis`), `funhouse_agent/tests/test_gm220_printed_options.py`,
  `test_gm251_drilled_shaft_base.py`;
- aggregate piers given `n=5.0` or a `column_type`:
  `ground_improvement/tests/` (`test_choices`, `test_settlement_ratio`),
  `calc_package/tests/test_steps_from_a_minimal_analysis.py`,
  `foundry_test_harness/scenarios.py`, and in `funhouse_agent/tests/`:
  `test_adapter_ergonomics`, `test_calc_package_adapter`,
  `test_calc_package_derived_inputs`, `test_calc_package_tool_parity`,
  `test_choices_stated_defaults`, `test_gm220_printed_options`,
  `test_gm221_printed_options`;
- MSE Cu below 4: `retaining_walls/tests/test_strip_pullout_factor.py` and
  `funhouse_agent/tests/test_mse_strip_pullout_factor.py` now assert the
  refusal (they asserted the flagged F* before).

**Tests whose asserted numbers or words moved with a default:**
`sheet_pile/tests/test_sheet_pile.py` and
`funhouse_agent/tests/test_earth_pressure_sources.py` (delta 17 -> 0 when
left out), `geotech_common/tests/test_adapters.py` (60 kPa clay stiff),
`geotech_common/tests/test_gm295_correlations.py` and
`test_soil_properties.py` (cap 2.0), `seismic_geotech/tests/test_print_gm293.py`
and `funhouse_agent/tests/test_calc_package_tool_parity.py` (phi/3),
`pile_group/tests/test_group_uplift.py` and
`funhouse_agent/tests/test_gm289_group_uplift.py` (4 driven piles 0.40),
`bearing_capacity/tests/` (`test_bearing_capacity`,
`test_meyerhof_linear_depth`, `test_own_ngamma`: Meyerhof's default is
linear), `funhouse_agent/tests/test_module_followup_adapters.py` (the new
conditional downdrag key).

## User manual

Hand-edited rows of `docs/user_manual/manual.html` only (no rebuild):
the changed parameter descriptions (drilled shaft `layers` and
`rock_base_conditions_met`, `Cu_backfill`, sheet pile `layers`,
aggregate piers `n` with its default cell removed and a new `column_type`
row, `overburden_method`, `delta_passive`, group uplift `pile_type`,
`depth_factor_method` with its default cell removed), the drilled-shaft
module paragraph (`manual_content.py` too), and two worked examples whose
output moves:

- `ground_improvement.aggregate_piers`: was refused as written, so it now
  gives `column_type="stone_column"` (`gen_worked_examples.py`); its output
  is the stone column's (n 2.5, SRF 0.7412, 74.1 mm, improved bearing
  202.4 kPa; was n 5, 0.5178, 51.8 mm, 289.7 kPa).
- `seismic_geotech.seismic_earth_pressure`: KPE 3.4995 at delta 10.0 deg
  (was 3.7307 at 12.0).

`worked_examples.json` carries those two and the downdrag example's new
`settlement_compatible_beside` key (not among the 14 lines the manual
prints).

## Tests

All judged by pytest's exit code, on the branch's final tree, Python 3.12,
the venv of COMMON_MODULES (planlens and geotech-references from PyPI).

| Run | Result |
|---|---|
| The changed modules and their neighbours: `bearing_capacity drilled_shaft retaining_walls sheet_pile ground_improvement geotech_common seismic_geotech pile_group downdrag lateral_pile subsurface_characterization foundry_test_harness soe` | 3,403 passed, 29 skipped, 0 failed (exit 0) |
| `calc_package validation_examples webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py` | 621 passed, 3 skipped, 1 xfailed, 0 failed (exit 0) |
| `funhouse_agent/tests` (all of it, so `test_defaults_used.py` and every file that names a changed module) | 4,236 passed, 30 skipped, 2 failed (exit 1); both failures are on master 2feea3d too, see below |
| `webapp/tests/test_docs_currency.py webapp/tests/test_backlog.py` | passed (in a 52-test run with the two web guards, exit 0) |
| `module_work/doc10/check_manual.py` (the manual's name check, no rebuild) | OK |

The two `funhouse_agent/tests` failures are not from this branch: each
fails the same way on master 2feea3d in the same venv.

- `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
  (`logx`, `logy` change nothing in the probe's result);
- `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`.

**Fingerprints.** `webapp/tests/test_page_fingerprints.py` passes without
a re-record, and `python -m webapp.tests.page_builds record` writes a file
identical to the one checked in: the page fingerprints do not cover the
tools' `METHOD_INFO` text. So there is no fingerprints commit.

**Skipped here for a missing dependency** (named by the test run):
python-ags4 (9 tests of `subsurface_characterization`); openseespy and the
other optional backends as the `funhouse_agent` skips list them.

## Findings table

GM-310 takes no rows of the audit's section 4: its ten changes are the
calls of `CALLS.md`, one row each in "The ten changes" above (all ten
done: six defaults changed, three refusals added, one summary text).

## Release note

Ten defaults follow the judgement calls of 2026-11-12, and answers move.
Drilled shafts: a base on rock is refused unless the profile gives the
rock's RQD, `rock_base_conditions_met`, or GSI with m_i (an RQD left out
was read as 100 and took 2.5 qu); with GSI and m_i and no RQD the base is
now the Hoek-Brown base. MSE walls: backfill with Cu below 4 is refused
without `F_star_tested`. Sheet piles under `log_spiral`: a layer takes
wall friction on the passive side only where it is marked
`interface_soil` 'clean_sand' or 'clean_gravel'; unmarked sand now takes
none, so its embedment and moment rise (4 m cantilever in phi 32 sand:
6.08 m to 8.65 m). Lateral piles from a soil profile: a clay is stiff from
47.9 kPa, not 100, so clays of 48 to 100 kPa take a stiff clay curve (a
stiffer response). Aggregate piers: n has no default; give `column_type`
('stone_column' 2.5, 'rammed_aggregate_pier' 5) or n (a stone column at
a_s 0.2: settlement reduction factor 0.56 to 0.77). SPT overburden
correction: Liao and Whitman capped at 2.0, not 1.7, which moves N1,60
only below about 35 kPa. Seismic passive pressure: KPE takes wall friction
phi/3 when none is given, not 0.4 phi (3.731 to 3.500 at phi 30, kh 0.2).
Pile group uplift: groups of 3 or 4 driven piles take the 20 % cut (phi_ug
0.50 to 0.40). Bearing capacity by Meyerhof: his own depth factors at
every depth, so footings deeper than they are wide gain capacity (the
gain grows with D_f/B, with no upper limit: 20 % at D_f/B 2.5 and 27 % at
D_f/B 3, a strip at phi' 34), with a warning
and Brinch Hansen's value beside it. Downdrag: no number changes; the
summary gives the settlement-compatible plane beside the headline where
the two differ.

## Noted, not fixed

- `drilled_shaft`, `rock_side_method` 'caving': alpha_E of Table 10-3
  still reads a left-out RQD as 100 (the largest alpha_E) and says so in
  its choice. The call is about the base; the same reasoning would refuse
  it. It changes a value (side resistance under 'caving' with no RQD).
- `foundry/` (retired wrappers): `ground_improvement_agent_foundry.py`
  still documents n's default as 5.0, and its harness scenario now gives
  `column_type`. `foundry/calc_package_agent_foundry.py` passes n = 5.0
  itself.
- `spt_overburden_correction`: the 'iso' form (EN ISO 22476-3, not held)
  is untouched.
- The lateral-pile boundary is 47.9 kPa; a clay entered as 47.88 kPa
  (1,000 psf to four figures) is soft by 0.02 kPa.
- `sheet_pile`: the chart sources (`log_spiral_source` Figs. 4-12, 4-13)
  are drawn for delta = 0.66 phi' only, so with them every layer must now
  give `wall_friction_deg` (a left-out layer took 17 deg and was refused
  the same way before; the message's wording is updated).
