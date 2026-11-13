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
- Tests: `pile_group/tests/test_pile_group.py` (9 new), `funhouse_agent/tests/test_gm319_pile_group.py` (15 new), one case of `test_nested_keys.py` moved to the 6-DOF tool.
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

## Release note

pile_group, 6-DOF analysis with battered piles: the pile forces and cap displacements change. Each pile's stiffness is now turned to the global axes in full; before, the terms between the horizontal and vertical head components were left out and a battered pile's axial force did not balance the applied loads (in the example above, 715.7 kN where statics gives 386.4 kN). Groups of vertical piles are unchanged. `pile_group_simple` now refuses a pile entry with a batter or a stiffness, and returns no cap displacement (null) where it returned zeros. New tool `equivalent_footing_settlement` (GEC-12 Sec. 7.3.5.3; UFC 3-220-20 Sec. 6-5.8.4). Both group tools' results carry a new `notes` list.

## Noted, not fixed

- The manual's cover and KPI counts ("1534 agent-callable methods") are one short until the next rebuild; a hand edit of the cover was left to the release rebuild.
- `docs/user_manual/worked_examples.json`'s `pile_group_6dof` result has no `notes` key until the examples are regenerated; its numbers are unchanged.
- `foundry/pile_group_agent_foundry.py` (retired route) still describes the old behaviour.
- The calc package has no section for the equivalent-footing settlement (audit row 6, C).
- `analyze_vertical_group_simple`'s Python warning for Vx, Vy, Mz is left as it is (the agent's tool refuses them first; GM-316/317 owns warnings).
- A fixed-head (moment-carrying) pile model and the CPGA benchmark stay with GM-28.
- Batter direction cosines are (sin batter_x, sin batter_y, rest), now said in the tool text; a pile battered in both directions at once is read that way, not as two successive rotations.
