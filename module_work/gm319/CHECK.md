# GM-319: pile_group, three things the tools say and do not do

Branch `claude/gm-319-pile-group-interface`, from master 2feea3d. Source: the module audit of 2026-11-12, section 4 `pile_group` (scout file `results/deep_axial.md`).

## Findings taken

| Audit row | Finding | Confirmed on master? | What was done |
|---|---|---|---|
| 1 (A) | The manual offers an equivalent-footing settlement; `group_settlement_equivalent_raft` exists and no tool reaches it | Yes | **Fixed.** Function read against the print (below): it matches. New tool `equivalent_footing_settlement`, with its description, two probes and the manual rows. |
| 2 (A) | `pile_group_6dof` takes batter angles and leaves out the kxz/kyz terms; the tool text does not say so | Yes (`rigid_cap.py` assembled `k3` with zeros in the x-z and y-z places) | **Fixed**: the terms are assembled (a call, see "Calls made"). What is still left out (pinned pile heads) is said in the tool text and in the result's `notes`. |
| 3 (A) | `pile_group_simple` takes per-pile `batter_x/batter_y` and stiffness and ignores them; cap `dz` from `piles[0]`'s stiffness times n | Yes | **Refused.** A `piles[]` entry with a batter other than 0, `axial_stiffness` or `lateral_stiffness` is refused by the tool, as the top-level stiffness already was. The library function refuses a battered pile and unequal stiffness. |
| 3, second half (same function) | `dz` reported as 0.0 when no stiffness is given (always, on the agent's path) | Yes | **Stated.** With no stiffness the library returns no cap displacements and a note; the tool returns `cap_displacements: null`. With one equal stiffness on every pile (library callers only) dz, rx and ry are worked out and match the 6-DOF analysis. |
| 5 (B) | `block_failure_capacity` and the raft function exported with no adapter method | Yes | The raft function is row 1. `block_failure_capacity` is a wrapper of `block_failure_resistance`, which `group_efficiency` reaches: **not done**, nothing is missing from the agent. |
| 4 (B to C) | DESIGN.md's "Public API" names `analyze_pile_group` | No: today's DESIGN.md lists the real exports (fixed by DOC-24) | Wrong on today's master; nothing to do. |

## What the print says

**Equivalent-footing settlement** (item 1).

- FHWA GEC-12 (FHWA-NHI-16-009) Vol. I Sec. 7.3.5.3, pp. 320-325: the equivalent footing at 1/3 D above the pile toe (Fig. 7-44), plan area B x Z the perimeter of the group (Fig. 7-31), the load spread at 1H:2V, and Eq. 7-55 (p. 320): `delta_sigma = Q / [(B + z)(Z + z)]`, Q the unfactored permanent load. Step 1c (p. 324) sums to where the increase is under 10 % of the existing vertical effective stress. Step 3 adds the pile's elastic compression.
- UFC 3-220-20 Sec. 6-5.8.4.1: Eq. 6-51, the same 2V:1H spread. Sec. 6-5.8.4.3: Eq. 6-54, `delta = (H0 / Es) [(1 + nu)(1 - 2 nu) / (1 - nu)] delta_sigma` for coarse-grained soil, summed by sublayers, to the depth "where the change in stress is less than 10% of the bearing stress of the equivalent footing".

The function does exactly this: footing to the outer pile faces at 2/3 D, Eq. 7-55 at each sublayer's mid-depth, `delta_sigma dz / M` with M the constrained modulus (Eq. 6-54's `Es (1 - nu) / [(1 + nu)(1 - 2 nu)]`), summed to UFC's 10 % depth or to a given depth. Checked against the closed form of Eq. 7-55 for a square footing, `S = (Q/M)[1/B - 1/(B + z)]` (test `test_equivalent_footing_settlement_is_the_printed_sum`). No change to its numbers. One change: its depth choice read "given" by its own word; it now asks `how_given`, as GM-180 requires.

What the function does not do, and the tool says so (`not_included`, the footing's `choices` entry, the description): the piles' own elastic compression (step 3), a clay's consolidation settlement (GEC-12 Eqs. 7-56 to 7-58; UFC Eq. 6-53), the other footing positions of GEC-12 Figs. 7-45 and 7-46 and UFC Table 6-32, the neutral-plane footing (UFC Sec. 6-5.8.4.2), batter (GEC-12 step 1b).

**Batter in the 6-DOF analysis** (item 2).

- The method's source is USACE EM 1110-2-2906 Ch. 4. It is **not held in the reference library**. The repo's own earlier check against the owner's copy (`module_work/wiki_verification/wave7_em2906_pti_mo.md`) records para. 4-5b(2), pp. 4-28/4-29: `[K] = SUM [K]_i`, "[K]_i = stiffness of ith pile transformed to global coordinates", and Saul's method "incorporates position and batter of piles". The EM's equations there are unnumbered. I could not re-read those pages in this session.
- Held print: UFC 3-220-20 Sec. 6-6.3, "Lateral capacity can be conservatively approximated by assuming that the piles only carry axial load as depicted in Figure 6-20" (force decomposition along the batter).

The transformation itself is plain arithmetic. A spring ka along the unit vector l and a spring kl across it have the global stiffness `k = kl I + (ka - kl) l l^T`. The code already used its xx, yy, zz and xy terms and already took the axial force as `ka (u . l)`; it left the xz and yz terms at zero. So the assembled head forces and the reported axial forces disagreed for a battered pile.

## What changed

- `pile_group/rigid_cap.py`
  - `analyze_group_6dof`: `k3 = kl I + (ka - kl) l l^T`, all terms. `PileGroupResult.notes` (new, in `to_dict` and `summary`) says the piles are pinned to the cap and batter is carried in full.
  - `analyze_vertical_group_simple`: refuses a battered pile and unequal (or partly given) `axial_stiffness`. Cap `dz = Vz/(n ka)`, `ry = My/(ka SUM x^2)`, `rx = Mx/(ka SUM y^2)` only when every pile carries the same ka (the displacements that give the formula's own forces; they match the 6-DOF analysis). Otherwise `cap_displacements` is empty and `notes` says why.
- `pile_group/group_efficiency.py`: the depth choice's "given" through `how_given`.
- `pile_group/calc_steps.py`: the 6-DOF step says how a pile is modelled; the simple method's section says the cap displacements are not calculated in place of a table of zeros.
- `funhouse_agent/adapters/pile_group.py`: the new tool; the `piles[]` refusal for the simple method; the briefs, the `piles` descriptions and the `returns` of both group tools (the 6-DOF `returns` named a key `cap_displacement` that does not exist).
- `funhouse_agent/tests/method_probes.json`: two probes for the new tool; the simple probe's battered pile taken out (it is now refused).
- Tests: `pile_group/tests/test_pile_group.py` (9 new), `funhouse_agent/tests/test_gm319_pile_group.py` (14 new), one case of `test_nested_keys.py` moved to the 6-DOF tool.
- `docs/user_manual/manual_content.py` and the matching rows of `manual.html` (rendered with the builder's own table functions; no rebuild). `pile_group/DESIGN.md`.

No fingerprint re-record was needed: `webapp/tests/test_page_fingerprints.py` passes unchanged.

## Before and after (made-up values)

Three piles on the x axis, ka = 200,000 kN/m, no lateral stiffness: A at x = -1 m and B at x = +1 m vertical, C at x = 0 battered 15 degrees toward +x. Loads Vx = 100 kN, Vz = 1,000 kN, My = 200 kN-m. The group is statically determinate: only C has a horizontal component, so P_C = 100 / sin 15 = 386.4 kN; then P_A + P_B = 1,000 - 386.4 cos 15 = 626.8 and P_B - P_A = 200.

| | P_A (kN) | P_B (kN) | P_C (kN) | Vertical sum of the axial forces (kN) | dx (mm) |
|---|---|---|---|---|---|
| Statics | 213.4 | 413.4 | 386.4 | 1,000 | |
| Before (master) | 240.9 | 440.9 | 715.7 | 1,373 | 7.46 |
| After | 213.4 | 413.4 | 386.4 | 1,000 | 1.62 |

A second test checks, for a four-pile group with batter in both directions and lateral stiffness under all six loads, that the head forces sum to the applied loads and that each head force's component along its pile is the reported axial force.

Vertical groups: unchanged (the worked example in the manual, V-060 and the module's existing tests give the same numbers).

Equivalent footing, 3 x 3 group, s = 0.9 m, b = 0.3 m, D = 12 m, Q = 3,000 kN, M = 40,000 kPa: footing 2.1 m x 2.1 m at 8 m, bearing stress 680.3 kPa, summed to 4.54 m, settlement 24.3 mm (closed form to that depth: 24.4 mm). The function's value is the same before and after; before, no tool returned it.

`pile_group_simple` with `piles: [{x: -1, y: 0}, {x: 1, y: 0, batter_x: 5}]`, Vz = 1,000: before, 500 kN each with `cap_displacements.dz = 0.0`; after, refused, naming `piles[1]`, `batter_x` and `pile_group_6dof`. Without the batter: 500 kN each, `cap_displacements: null`, and the note.

## Calls made

1. **Batter terms added, not only stated.** The section says to add them if the print gives them, otherwise to say they are left out. The EM is not in the reference library, so I could not cite its page from the print itself; I relied on the repo's recorded reading of EM pp. 4-28/4-29 ("stiffness of ith pile transformed to global coordinates") and on plain arithmetic, checked against statics. The other form, keeping the old numbers and stating the omission, would have left a tool whose axial forces do not balance its loads. The earlier owner decision (calc-QC PG-1) deferred a "full CPGA-style coupled rewrite" for want of a benchmark; this is not that rewrite: no rotational head stiffness is added, and GM-28's CPGA benchmark is still open. If the owner prefers the old behaviour until that benchmark, the change is the two lines that build `k3`.
2. **The simple method refuses** per-pile batter and stiffness rather than using them. Its formula has no place for either; `pile_group_6dof` takes both.
3. **No stiffness, no displacement**: `cap_displacements` is null for the simple tool, not zeros. The library still works out dz, rx, ry when given one equal stiffness, since that is the same arithmetic as the forces.
4. **The settlement tool takes M, or Es with nu** (UFC Eq. 6-54 is printed in Es and nu). No default for either; nu = 0.5 is refused (Eq. 6-54 then gives no compression).
5. **The tool's name** is `equivalent_footing_settlement`, the manual's and the references' words; the library function keeps its name.

## Tests run

Venv per COMMON_MODULES (Python 3.12). Judged by pytest's exit code.

- `webapp/tests/test_public_text.py` and `webapp/tests/test_page_fingerprints.py`: 18 passed.
- `module_work/doc10/check_manual.py` (the manual's name check): OK.
- Wide run, `funhouse_agent/tests webapp/tests calc_package validation_examples pile_group`: 6,317 passed, 43 skipped, 1 xfailed, 3 failed (12 min). One of the three was mine and is fixed: `test_citations.py` read "UFC Eq. 6-54" in the new tool's brief as a GEC-12 cite; the brief now says "UFC 3-220-20 Eq. 6-54". Re-run after the fix, `test_citations.py`, `test_gm319_pile_group.py`, `test_defaults_used.py`, the public-text and fingerprint tests and `pile_group`: 1,011 passed, 23 skipped, 1 failed (the `plot_data` one below).

Two tests fail on master 2feea3d as well, in this environment, and are not touched by this branch: `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` (logx/logy) and `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table` (a figure-layout note; it fails the same way with this branch's changes stashed).

## Review round (2026-11-13)

Independent review of 7ddbd73 (verdict: merge after fixes), fixed on `fix/gm319-review` from 9e5f439 (7ddbd73 with that morning's master merged in). The lead's rulings are taken as given.

| # | Grade | Finding | What it became |
|---|---|---|---|
| 1 | A (older than the branch) | `pile_group_simple` skipped Mx when every pile is on y = 0 (one row) and My on one column, without a word; the 6-DOF refusal of the same layout said "lateral force or torsion" and pointed to the simple method | **Fixed.** `analyze_vertical_group_simple` refuses a non-zero Mx when SUM(y²) = 0 and My when SUM(x²) = 0: "piles pinned to the cap on one line cannot carry a moment about that line; give piles off that line, or leave Mx out if the cap is restrained against that rotation another way". A zero moment runs. `analyze_group_6dof` says load by load what the group cannot carry and why (Vx/Vy: no stiffness along that axis; Mz: no horizontal stiffness; Mx/My: the one-line reason) and no longer points to the simple method. Both tools' Mx/My descriptions say so. Tests both directions, zero moment, moment across the row, both tools. |
| 2 | B | The settlement tool did not say Eq. 6-54 is for coarse-grained soil; it cited GEC-12 Sec. 7.3.5.3 (cohesive soils, consolidation) for it; nu up to 0.49 passed silently | **Fixed.** Brief, `constrained_modulus_kPa`, `youngs_modulus_kPa` and `poissons_ratio` say "coarse-grained (drained) soil" (UFC 3-220-20 Eq. 6-54, pp. 448-449) and send a group in cohesive soil to consolidation, GEC-12 Sec. 7.3.5.3, with the settlement module's `consolidation_settlement` (it takes each layer's `delta_sigmas`, so the footing's Q/[(B + z)(Z + z)] can be given). Sec. 7.3.5.3 is cited only as that method; the footing cites GEC-12 Fig. 7-44 and Eq. 7-55, the piles' compression Sec. 7.3.5.1, Eq. 7-48. No new refusal below 0.5. With Es and nu the modulus choice states M and M/Es. Above nu = 0.40 a `warnings` entry (printed basis below). Library docstring, DESIGN.md and the manual narrative follow. |
| 3 | C | `num_sublayers` 2.7 ran as 2 | **Fixed.** Refused unless a whole number of 1 or more, as `n_rows`/`n_cols`. |
| 4 | C | Spacing below the pile diameter accepted | **Fixed.** `group_settlement_equivalent_raft` refuses s < b when the group has more than one pile in a direction (the tool reaches it). |
| 5 | C | sin²bx + sin²by > 1 clamped lz to 0 | **Fixed.** `GroupPile.direction_cosines` refuses sin²bx + sin²by ≥ 1, naming the pile and the sum; the 6-DOF `piles` description says the limit. |
| 6 | C | Simple note "…and none was given" | **Fixed.** "…use no pile stiffness (the pile_group_simple tool takes none). The 6-DOF analysis (pile_group_6dof) gives the cap's displacements…" |
| 7 | C | Alias `soil_modulus_kPa` read a "soil modulus" as M; `D` for `pile_length` | **Fixed.** Both dropped (nothing else used them; grep of tests, probes, manual, worked examples). The library keyword `soil_modulus_kPa` of `group_settlement_equivalent_raft` is unchanged. |
| 8 | C | CHECK said 15 new tests in `test_gm319_pile_group.py` | **Fixed**: 14 (above). |
| 9 | C | No opposed-batter symmetry test | **Added** (`TestOpposedBatter`): battered piles at x = ±1.2 m, 12° outward, axial only; vertical piles at (0, ±0.9) m with kl = 4,000 kN/m. Vz alone: dx = ry = 0, equal pairs. Vx = 120 kN alone: dx = 120/(2 × 4,000) = 15.0 mm, ry = −dx/h with h = 1.2/tan 12° = 5.65 m, battered axial forces 0; with Vz added they equal the Vz-alone forces. |
| 10 | C | Simple tool refuses an equal per-pile stiffness | Nothing (the review says none is needed). |

**Before and after** (made-up values; "before" is 9e5f439):

| Call | Before | After |
|---|---|---|
| `pile_group_simple` n_rows 1, n_cols 3, s 1.5, Vz 900, Mx 300 | 300 / 300 / 300 kN, no word | refused: "Mx = 300 kN-m cannot be carried: every pile is on y = 0, one line along the x axis…" |
| the same with n_rows 3, n_cols 1, My 300 | 300 / 300 / 300 kN, no word | refused, the same for My on x = 0 |
| the same with Mx 0 | 300 / 300 / 300 kN | 300 / 300 / 300 kN |
| `pile_group_6dof` n_rows 1, n_cols 3, ka 2e5, Vz 900, Mx 300 | refused: "…cannot carry lateral force or torsion… or use analyze_vertical_group_simple" | refused: "…cannot carry the applied Mx. Mx: every pile is on y = 0…" |
| `pile_group_6dof`, a pile at 50° and 50° | lz clamped to 0 (singular in the review's layout) | refused: "sin^2(batter_x) + sin^2(batter_y) = 1.174, at or above 1" |
| `equivalent_footing_settlement` 3 x 3, s 0.9, b 0.3, D 12, Q 3,000, Es 20,000 kPa, nu 0.49 | 2.84 mm, M 342,282 kPa, no word | 2.84 mm, choice "342,282 kPa = 17.11 Es", and the warning |
| the same with nu 0.3 | 36.09 mm | 36.09 mm, choice "26,923 kPa = 1.35 Es", no warning |
| the same with M 40,000 kPa, `num_sublayers` 2.7 | 21.89 mm (run as 2) | refused |
| the same with s 0.2 m | 72.9 mm (0.7 m footing) | refused: "Spacing 0.2 m is less than the pile diameter 0.3 m" |
| the same with `soil_modulus_kPa` 40,000 | run as M, "given" | refused as an unknown parameter |

No calculated value of a call that was right changes.

**Calls made in this round**

6. **The Poisson's ratio warning has a printed basis.** FHWA GEC-5 (FHWA-NHI-16-072, 2016) p. 8-26, Table 8-10 (after AASHTO 2014, read from the rendered page): sand 0.2 to 0.36 (loose) to 0.3 to 0.4 (dense), gravel 0.2 to 0.35 to 0.3 to 0.4; the same page: "approximately 0.1 to 0.2 for drained loading of coarse-grained soils". The warning starts above 0.40, the top of the table's coarse-grained rows (the text's 0.1 to 0.2 would flag the table's own values). So the lead's no-basis fallback was not needed; the threshold is the same 0.40.
7. **Eq. 6-54's page.** Its scope sentence ("Elastic compression of coarse-grained soil can be approximated…") is at the foot of UFC 3-220-20 p. 448 and the equation on p. 449 (checked in the PDF), so the tool cites pp. 448-449.
8. **Where the refusals sit.** The one-line moment in the library (both tools reach it; "on the line" is SUM(y²) or SUM(x²) at or below 1e-12 m²); the batter limit in `GroupPile.direction_cosines` (every caller); the spacing in the library function, only when a direction has more than one pile (a single pile has no spacing); `num_sublayers` in the tool.
9. **The manual's cover counts** were copied in with the rows, so the rebuild differs from the committed manual only at line 1596 (formerly 1594), the older difference on master.

**Tests run in this round** (venv `C:\dev\GeotechStaffEngineer\.venv`, PYTHONPATH the worktree and its `packages/aec-agent-core`, one file at a time, judged by exit code): `pile_group/tests/test_pile_group.py` 100 passed (13 new); `funhouse_agent/tests/test_gm319_pile_group.py` 29 passed (15 new); `funhouse_agent/tests/test_nested_keys.py` 66 passed; `funhouse_agent/tests/test_adapter_ergonomics.py` 170 passed; `funhouse_agent/tests/test_defaults_used.py -k pile_group` 42 passed; `webapp/tests/test_page_fingerprints.py` 13 passed (no re-record needed); `webapp/tests/test_public_text.py` 5 passed; `funhouse_agent/tests/test_citations.py` 5 passed (run for the new cites, against the installed reference library). `module_work/doc10/check_manual.py`: OK, exit 0; `--rebuild`: differs at line 1596 only.

**Follow-up** (changes a saved report's output; pre-existing, not in the review; for the lead to file): `pile_group_simple` takes `piles[]` coordinates as measured from the piles' centroid (`GroupLoad`: loads "at the centroid of the pile group") but neither shifts nor refuses an off-centre layout. Piles at x = −1, 1 and 3 m, Vz 900 kN, My 200 kN-m give 281.8 / 318.2 / 354.5 kN, which sum to 954.5 kN, not 900; with My 0 they give 300 kN each, where `pile_group_6dof` (equal ka) gives 525 / 300 / 75 kN.

## Release note

pile_group, 6-DOF analysis with battered piles: the pile forces and cap displacements change. Each pile's stiffness is now turned to the global axes in full; before, the terms between the horizontal and vertical head components were left out and a battered pile's axial force did not balance the applied loads (in the example above, 715.7 kN where statics gives 386.4 kN). Groups of vertical piles are unchanged. `pile_group_simple` now refuses a pile entry with a batter or a stiffness, and returns no cap displacement (null) where it returned zeros. Both rigid-cap tools refuse a moment about the one line every pile is on (Mx with every pile on y = 0, My with every pile on x = 0); before, `pile_group_simple` returned equal shares as if the moment were not there. A pile battered both ways with sin²(batter_x) + sin²(batter_y) at or above 1 is refused. New tool `equivalent_footing_settlement`: the elastic compression of coarse-grained soil below the equivalent footing (UFC 3-220-20 Sec. 6-5.8.4, Eq. 6-54; the footing of GEC-12 Fig. 7-44 and Eq. 7-55); it sends a group in cohesive soil to consolidation (`consolidation_settlement`), warns above a Poisson's ratio of 0.40, and states M/Es when M is worked out from Es and nu. Both group tools' results carry a new `notes` list.

## Noted, not fixed

- The manual's cover and KPI counts were one short after the first round; the review round copied the rebuilt counts in (1535 methods, 5818 parameters).
- `docs/user_manual/worked_examples.json`'s `pile_group_6dof` result has no `notes` key until the examples are regenerated; its numbers are unchanged.
- `foundry/pile_group_agent_foundry.py` (retired route) still describes the old behaviour.
- The calc package has no section for the equivalent-footing settlement (audit row 6, C).
- `analyze_vertical_group_simple`'s Python warning for Vx, Vy, Mz is left as it is (the agent's tool refuses them first; GM-316/317 owns warnings).
- A fixed-head (moment-carrying) pile model and the CPGA benchmark stay with GM-28.
- Batter direction cosines are (sin batter_x, sin batter_y, rest), now said in the tool text; a pile battered in both directions at once is read that way, not as two successive rotations.
