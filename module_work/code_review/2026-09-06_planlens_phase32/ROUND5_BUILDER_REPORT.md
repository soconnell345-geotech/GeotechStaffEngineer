# planlens round-5 repair — builder's report (2026-09-10)

Relayed verbatim in substance from the builder (Fable, fresh agent, no
round-4 context) via five sectioned messages; the harness refused its report
file, so this is the durable copy. **Independent verification was in flight
when this was written — see ROUND5_VERIFICATION.md for the verdict.** All
artifacts cited (patches, tree snapshots, probe scripts, outputs) were under
the session scratch dir `C:\Users\socon\.claude\jobs\69be0e95\tmp\`.

**Tree:** planlens `main` @ `1f6551c`, working tree uncommitted, nothing
staged; frozen 20:32:54 at `queries.py` md5 `c0c67b3407a67568f5148c810c6a3ab8`,
`test_end_ownership.py` md5 `667c7e5a8250bced9013be2d329d4631`. App repo not
modified by the builder.

## 1. Measurements — required → observed on the final tree

| acceptance row | required | observed |
|---|---|---|
| corpus @0.3 leaders / dims | 25/25, 16/16 | **25/25, 16/16** |
| corpus @0.5 leaders / dims | 23/25, 16/16 | **23/25, 16/16** |
| leader FPs @0.5 (2000a/2000b/3000/5003/10.17a/10.25a/11.01) | 2/10/6/19/23/20/38 | **identical** |
| dim precision @0.5 (21.01/3001/11.01/10.31A) | 5/10/12/10 | **identical** |
| 41 matched residuals @0.3 | identical to 3 dp | **every value identical** (leader med 7.197 max 10.814; dim med 0.026 max 0.036) |
| blunt proposals @0.5 | 0 | **0** |
| full called set @0.5 (322 constructs, path + both ends + confidence, ten sheets) | byte-identical | **byte-identical** after-A vs final, and prune-on vs prune-off |
| leaders at 0.0 / 0.3 / 0.5 | identical | **identical on every sheet** |
| README blunt figure | — | 19/17/0 → **18/16/0 by definition**; `doc_claims_check.py` prints 18/16/0 (11.01 1/1/0, 21.01 17/15/0) |
| planlens suite | green | **724 passed** (632 start → 671 after A → 702 after B → 724 final; +40 `test_tipless_terminators.py`, +52 `test_end_ownership.py`); 0 failures |
| app drawing tests | 58 green | **61 passed** (the two files now collect 61, superset of 58) |
| `test_readme_claims.py` | green | **10 passed** |

`diff corpus_before.txt corpus_after_B.txt` differs only in the blunt-count rows.

## 2. Changes (all `planlens/ir/queries.py` unless noted)

### Change A — finding 1 (landed and verified alone; `A_delta_queries.patch`, 282 lines)
- `:1274-1324` tipless branch of `_arrow_attach`. **Coordinate:** `along = max(0.0, …)` at `:1299` — the centroid projection never falls behind the shaft tip; a centred box block is inert (centroid = defpoint), a diamond / flat-tipped arrow drawn inside the line now publishes the line's end. **Policy** at `:1322`: `_cluster_alignment` PCA elongation ≥ 1.6 **and** fold ≥ `_ARROW_AXIS_MIN` (cos 30°) → state `"oriented"` (a cluster's standing: admitted, scored sign-blind, in `arrowhead_ids`, tier 1); otherwise `"blunt"` (isotropic; 0.45 cap unconditional, withheld from `arrowhead_ids`). No new constant.
- `:1049` `_Attach` states documented ok / oriented / blunt / contradicted. `:1368` `_end_tier` — oriented sits at tier 1 with clusters (sign-blind, never above a directional arrowhead).
- `:1745-1752` `find_leaders` **keeps capping BOTH tipless states** (`arrow_blunt`): a dimension terminator marks an END, a leader arrowhead must POINT. Leader FPs provably untouched.
- `:2543`, `:2760` — the no-text "at least one drawn shape" rule now means a **pointing** drawn shape (`kind == "triangle" and state == "ok"`); behaviour-identical for every pre-existing state, excludes oriented.
- `:2613`, `:2825` — new evidence key `oriented_terminator_ids`; `blunt_terminators` / `blunt_terminator_ids` stay isotropic-only; oriented ends ARE named in `arrowhead_ids`.
- `:2013-2058` `find_dimensions` docstring rewritten (tipless split + end ownership). `:734` `_arrowhead_candidates` — finding 6 named as a known sharp edge.

### Change B — findings 2, 3, 4 together (`B_plus_docs_delta_queries.patch`, 273 lines)
- `:1424` `_seat_distance` — within-tier ranking metric: pointed → `min(|apex−tip|, |base_centre−tip|)` (0.0 for a real terminator in either drafted style); cluster → centroid; tipless → min of centroid / either axial extreme vertex.
- `:1482` `_SEAT_DOMINANCE = 10.0`; `:1492` `_award_end` — the best-seated of tier 0 and tier 1 are compared once (order-independent); tier 1 wins iff `seat1 · 10 < seat0`; contradicted (tier 2) never dominates or is dominated.
- `:2266-2340` per-end loop keeps the best-seated per tier and calls `_award_end`.
- **Exact prune restored** (`:1489` `_EXACT_SEAT_PRUNE`, `:2305-2325`): skip a candidate when `d − R ≥ tier-0 incumbent seat`, R = the candidate's own radius (farthest vertex from its centroid; 0 for a cluster; memoized). Exact by the triangle inequality — every seat point (apex, base centre, vertex extreme, centroid) lies within R of the centroid, so `seat ≥ d − R`; a skipped candidate cannot beat the incumbent within tier 0 (strict <), cannot dominate from tier 1 (needs `seat < s0/10`), and tier 2 never wins over tier 0. Proven output-identical: band probes prune-on vs prune-off byte-identical on all ten sheets, and `test_the_prune_never_changes_the_answer` toggles the flag over 10 scenes. The OLD centroid prune was NOT exact: `test_the_old_prune_would_have_skipped_the_true_arrow` measures the premise (junk chevron at apex (103,4) nearer by centroid 4.39 vs 4.80 but farther by seat 4.24 vs 0.0 than the true arrow whose apex IS the end).

### Docs
- `README.md:237-259` tipless paragraph re-DEFINED and re-measured (18/16/0; 21.01 17/15/0; 11.01 1/1/0; the six oriented corpus proposals, none called). `README.md:272-303` new "Sharp edges the corpus cannot falsify" (findings 6, 5, and B's two documented consequences).
- `planlens/ir/DESIGN.md:223` `find_dimensions` bullet: end ownership + oriented/blunt definition.
- `planlens/tests/test_readme_claims.py:84-91` guard constants 19/17 → 18/16, 11.01 (2,2,0) → (1,1,0), dated note.

### Fixtures
- `planlens/ir/tests/test_tipless_terminators.py` (40 collected): clamp on diamond / flat-tip / centred box, vertex-order invariance (4 rotations), shape wholly beyond the tip documented; oriented vs blunt (along = oriented, across = blunt, box/rect blunt, cos-30 cone); diamond and trapezoid called at 0.952 with exact defpoints, order-invariant, control triangle unchanged, box dimension never called, one-oriented-one-box capped, no-text oriented pair capped; **the trade both ways** (rectangle scene not called + leader survives; two rectangles not called; the SAME scene with an oblong along the crossing line IS called and the leader is arbitrated away exactly as with a chevron; oblong across the line → blunt, cap holds); leader flow keeps capping a diamond arrowhead.
- `planlens/ir/tests/test_end_ownership.py` (52 collected): `TestSeatDistance`; `TestAwardEnd` (tie → sound; seated tier-0 unbeatable; 10× threshold; contradicted never dominates/dominated); `TestASeatedClusterKeepsItsEnd` (F2: foreign chevron changes nothing, cluster wins with ids `['e4','cluster:e10']`, the LEADER owning the chevron survives `exclude_dimensions` and no dimension names it at any threshold, seat ratio 23×); `TestASplashOffTheEndCannotLiftACappedConstruct` (round 4's measured case, arrows-outside at 12° and 20°); `TestTheDocumentedResidualOfTheTrade`; `TestTheTrueArrowOutranksJunkAtTheSameEnd` (F3 (108,0)/(108,1)/(108,5) and F4 (103,2..5) → `[100,0]` 0.952, award independent of entity order); `TestALoneOutwardChevronIsAnArrowsOutsideEnd`; `TestTheSeatPruneIsExactAndTheOldOneWasNot` (bound `seat ≥ d − R` over 6 shapes × 3 tips; identical output with the prune toggled over 10 scenes).
- Existing suites green unchanged, including `TestSkewedDimensionDoesNotStealANeighbouringLeader` at all skews and both junk distances.

## 3. `round4_repro.py` — round-4 tree → repaired

| fixture | round-4 tree | repaired |
|---|---|---|
| slender triangle (control) | `[100,0]` 100.0 0.952 | `[100,0]` 100.0 **0.952** |
| diamond / trapezoid | `[97,0]` 94.0 0.45 | **`[100,0]` 100.0 0.952** (called) |
| dart (finding 6) | no proposal | no proposal — documented |
| cluster + foreign chevron | `[108,0.5]` 0.944, ids `['e4','e26']` | **`[101.53,0]` 0.951, ids `['e4','cluster:e10']`** |
| on-spine junk (108,0) / (108,1) / (108,5) | `[108,0]` 0.942 / `[108,1]` 0.945 | **`[100,0]` 0.952** all three |
| off-spine junk (103,2..5) | `[100,0]` **0.45** (invisible) | **`[100,0]` 0.952** (called) |
| 60°+ triangles (finding 5) | 0.25 | 0.25 — documented, not fixed |

## 4. Timing (best of 3, seconds; dims @0.0 / @0.3 / @0.5 / leaders @0.5 exclude_dimensions)

| sheet | after A | final, prune ON | final, prune OFF |
|---|---|---|---|
| 5003 | 0.01/0.01/0.01/0.02 | same | same |
| 10.17a | 0.06/0.03/0.01/0.04 | 0.06/0.04/0.01/0.04 | same |
| 11.01 | 0.11/0.07/0.04/0.17 | 0.12/0.07/0.04/0.20 | 0.12/0.07/0.04/0.11 |
| 21.01 | 0.85/0.62/0.36/0.86 | 0.80/0.54/0.30/0.86 | 0.72/0.54/0.29/0.72 |
| 3001 | 3.38/1.56/1.02/1.58 | 4.05/1.93/1.31/1.96 | 3.62/1.54/1.02/1.53 |
| TOTAL | 4.68/2.49/1.57/3.15 | 5.42/2.80/1.82/3.63 | 4.88/2.41/1.53/2.97 |

3001 @0.0 reads +20% with the prune ON, but identical code varied more than that run to run on this machine (other jobs running) — the noise floor is at least the size of the effect. What is real: the only extra work is in the 0.0 band; at 0.3 / 0.5 (the scorer's thresholds) the three dense sheets are ≤ 0.2 s and flat. Nothing approaches the 600 s regime. The exact prune's benefit is unmeasurable at this noise level; kept ON.

## 5. Observational movement, accounted per row

**The ≤0.25 band grew by 60 constructs** (dims @0.0, ten sheets, after-A → final): 0.0: 16→25, 0.2: 2→2, 0.214: 42→47, 0.25: 304→350, 0.376: 1→1, 0.377: 1→1, 0.429: 24→24, 0.45: 289→289, 0.571: 1→1, 1.0: 53→53. **Not from the prune** (prune-on vs prune-off byte-identical). Mechanism: letterform strokes whose two ends previously resolved to the SAME chevron (skipped as non-distinct) now resolve to distinct best-seated chevrons; every new row carries detached + direction-violation + off-spine + below-extent, all ≤ 0.25.

**Change A — six re-labelled proposals** (sheet | shaft | candidate old→new | rung holding it | conf):

| | | | | |
|---|---|---|---|---|
| 11.01 | e199 | e791 blunt→oriented (e790 stays blunt) | detached (seat 7.76 > bound) + off-spine + one blunt end — this is the `blunt_terminators` 2→1 | 0.45 |
| 11.01 | e203 | e790 blunt→oriented | contradicted other end (e788) + off-spine + below-extent | 0.25 |
| 11.01 | e488 | e778 blunt→oriented | contradicted (e774) + off-spine | 0.25 |
| 11.01 | e543 | e778 blunt→oriented | contradicted + off-spine | 0.25 |
| 11.01 | e551 | e778 blunt→oriented | contradicted + off-spine | 0.25 |
| 21.01 | e329 | e7447 blunt→oriented | detached (seat 8.33) + off-spine + below-extent; ends 223/231 pt from any native defpoint | 0.45 |

Coordinates unchanged on all six; 11.01 has no native dims.

**Change B — six changed @0.3 rows**, all 0.45, all off-spine + below-extent, glyph-scale 5–15 pt spans (sheet | shaft | end old (tier, seat, centroid d) → new | coordinate | nearest native defpoint):

| | | | | |
|---|---|---|---|---|
| 11.01 | e3058 | e3056 (0, 4.74, 4.84) → e3753 (0, 4.17, 5.23) | unchanged `[328.5, 358.02]` | n/a |
| 21.01 | e4676 | e4674 (0, 4.26, 4.65) → e4694 (0, 4.01, 5.28) | unchanged | 34.5 pt |
| 21.01 | e4678 | e4697 (0, 3.30, 5.02) → e4680 (0, 2.88, 5.15) | `[274.38,126.48]` → `[282.72,124.80]` | 44.3 → 37.8 pt |
| 21.01 | e4727 | e4742 (0, 5.73, 6.49) → e4737 (0, 4.93, 6.50) | unchanged | 45.7 pt |
| 3001 | e2857 | e2855 (0, 4.92, 5.36) → e2877 (0, 3.96, 5.95) | `[346.74,136.26]` → `[337.2,139.02]` | 156.7 → 150.6 pt |
| 3001 | e3181 | e3178 (1, 7.14, 7.09) → e3195 (1, 6.17, 7.91) | unchanged | 207.0 pt |

In every row the new winner is nearer by seat and farther by centroid — the metric change and nothing else; none within 30 pt of a native defpoint (scorer tolerance 18), so none is a suppressed true positive.

## 6. Judgement calls (each with the fixture that fails under the alternative)

1. **Oriented gate** = PCA elongation ≥ 1.6 AND fold ≥ cos 30° along the line — the existing cluster gate and the signed test's cone, no new constant. Alternatives: unconditional cap (fails `test_called_at_the_default_with_the_exact_defpoints`); round-1 witnesses+text hatch (fails `test_the_rectangle_scene_is_not_called`). Cost pinned in `TestWhatTheOrientedRankNewlyAdmits`: an oblong at a crossing line's end founds a called dim and the leader sharing its arrowhead is arbitrated away — same outcome as a chevron there.
2. **Leader flow keeps capping oriented shapes** (a leader arrowhead must point) — leader FPs untouched.
3. **Tipless coordinate** = tip-clamped centroid projection. A shape wholly beyond the tip publishes its projected centroid — exact for centred blocks, half a length short for tip-at-defpoint blocks; nothing distinguishes them. Pinned as documented.
4. **Seat metric** = min(apex, base-centre) distance. Alternative (centroid + dominance only) leaves F3/F4 open (`TestTheTrueArrowOutranksJunkAtTheSameEnd` fails). Known cost: a foreign outward arrow whose base sits inside a >17°-skewed true arrow's seat could still take that end; the round-4 skew fixture passes 0–25° at both junk distances.
5. **Dominance ratio 10×** — the round-4 verifier's number, coarse on purpose; real corpus clusters seat 0.5–2 pt so no attached directional candidate is dominated on the corpus (called set byte-identical). Alternative (absolute seating within a fraction of scale) would differ on a cluster 0.5 pt off vs a chevron 3 pt off — **not pinned; builder not confident which way that should go.** Both directions of the round-4 splash trade ARE pinned (`TestASplashOffTheEndCannotLiftACappedConstruct` / `TestTheDocumentedResidualOfTheTrade`).
6. **No axial-gap bound for finding 3.** An outward chevron based at a shaft end IS the arrows-outside anatomy (corpus "T=" construct; the skew fixture at 0–9° asserts the apex 7.2 pt beyond the tip, called). A gap tolerance would be an invented constant and would have moved the 0.5 band. Fixed by ranking instead; the lone-chevron reading pinned as documented.

## 7. Not done, stated plainly

- Finding 5 documented in README, not fixed: applying the quad margin test to triangles would move wide junk triangles from the 0.25 rung into the 0.45 band on the corpus — needs its own measured round.
- Finding 6 documented (README sharp edges + `queries.py:734`), not fixed.
- `FINDINGS.md` corrections supplied for the lead (app repo was read-only to the builder): round 1 "316 → 375" should read "350 → 375"; round 4 "316 at the start" should read "350" — the tip collects 350; 632 reproduces. **Applied by the lead in the same commit as this file.**
- Prune timing benefit unmeasurable at this machine's noise floor.
- `queries.py`, `DESIGN.md`, `ocr.py` are CRLF in the working tree (i/lf w/crlf, autocrlf=true) — pre-existing from round 4, not introduced; normalised on commit. README and the builder's tests stayed LF; the delta patches were built CR-stripped.
- Nothing committed, staged, tagged or pushed in either repo.
