# planlens round-6 repair — builder's report (2026-09-11)

Round 6 answers the round-5 verification (`ROUND5_VERIFICATION.md`): D1
blocker, D2 cost, D3 policy, D4 tie, D5 docs. Same builder (Fable, resumed
with full context after a session-limit reset; the tree was verified
byte-identical to the round-5 freeze before it resumed). Relayed from six
sectioned messages; the harness refused its report file. **Independent
verification was in flight when this was written — see
ROUND6_VERIFICATION.md.**

**Tree (FROZEN 07:07:32, verified by the lead):**

| file | md5 |
|---|---|
| `planlens/ir/queries.py` | `e2b2b3de5820fe12f8400348108e098b` |
| `planlens/ir/tests/test_end_ownership.py` | `a561fd2d7dc219fe09c3814e22eaeff2` |
| `planlens/ir/tests/test_tipless_terminators.py` | `0547923eebb0a11061ded24a245cfa46` |
| `README.md` | `4d62c94fc3db252745ba34ffccee69a7` |
| `planlens/ir/DESIGN.md` | `231c91f034fe693c4dc9221040d29861` |
| `planlens/tests/test_readme_claims.py` | `4ff1904515e2aa491dfaa2bf1bdaa848` |
| app: `module_work/drawing_ground_truth/doc_claims_check.py` (lead-directed: adds the ORIENTED row; blunt logic unchanged — lead reviewed the diff) | `2dcaf502ed50ad45457b678d782723f4` |

Nothing staged; `ocr.py` + its test untouched; no app full gate run. Delta from
the round-5 freeze: `round6_delta_queries.patch` (436 lines, scratch dir).

## 1. Measurements — final tree vs round 4 (= baseline)

| row | required | observed |
|---|---|---|
| @0.3 leaders / dims | 25/25, 16/16 | **identical** |
| @0.5 leaders / dims | 23/25, 16/16 | **identical** |
| leader FPs @0.5 | 2/10/6/19/23/20/38 | **identical** |
| dim precision @0.5 | 5/10/12/10 | **identical** |
| 41 residuals @0.3 | identical to 3 dp | **every value** (leader med 7.197 max 10.814; dim med 0.026 max 0.036) |
| called set @0.5 (322 constructs, coords+conf) | byte-identical | **byte-identical to round 4** (no line of the vcheck section differs) |
| leaders @0.0/0.3/0.5 | identical | **identical every sheet** |
| blunt @0.5 | 0 | **0** |
| **oriented corpus population** | — | **0 / 0 / 0** at 0.0/0.3/0.5 — every former member was a 0.6–2 pt fragment below the 4.4–4.7 pt floor |
| **blunt README figure** | — | **19 / 17 / 0** (21.01 17/15/0, 11.01 2/2/0) — **back to round 4 exactly**; round 5's 18/16/0 is superseded |
| README vs `doc_claims_check.py` | agree | **agree** — the script now prints the ORIENTED row (`"README form: oriented-terminator proposals: 0 / 0 / 0"`), the README carries that exact string plus 19/17/0 with its definition; layers 557/158/1696 unchanged; `test_readme_claims` pins both (11 passed) |
| planlens suite | green | **798 passed / 0 failed** (632 start of round 5 → 724 round-5 freeze → 798) |
| app drawing tests | 58 green | **61 passed** |
| @0.3 dimension set | — | n 369 → 370; counts move on 10.31A 50→51, 21.01 124→123, 10.17a 44→45; **15 rows differ** (§3), all 0.45, all below-extent-floor and/or detached/off-spine, 30–457 pt from any native defpoint |
| @0.0 band | — | n 733 → 790 (**+57**); histogram r4→r6: 0.0: 16→25, 0.2: 2→2, 0.214: 42→47, 0.25: 304→346, 0.376: 1→1, 0.377: 1→1, 0.429: 24→24, 0.45: 289→290, 0.571: 1→1, 1.0: 53→53 |

Verifier's `attack.py` on the final tree (`attack_work_r6b.txt`): P2 cluster
wins at all 6 shifts × 7 seats and the leader survives at all 6 shifts; P4 same
answer in both orders; P1/P5/P6 all blunt, none called. Appendix C's pytest
passes at shifts 0.1/0.5/1.0/2.0 (`test_appendix_c_the_leader_that_owns_the_chevron_survives`);
the verifier's D1 table passes at 6 shifts × 4 seats × both enumeration orders
(48 cases).

## 2. What changed (all `planlens/ir/queries.py`)

- **D1 — award rule.** `_seat_distance`: a fill cluster seats at its **NEAREST MEMBER** (was centroid). `_award_end`: between the two sound tiers the **better-seated candidate wins outright**, tie → directional; contradicted (tier 2) never wins on distance. `_SEAT_DOMINANCE` **removed** (a test asserts it is absent).
- **Prune.** Bounds against the best SOUND seat (strict); cluster radius is the **real** radius — a zero radius was pruning real clusters at shift 1–2 (caught by the verifier's D1 table).
- **D3 — oriented gate.** Oriented now requires long-axis-along (as before) **AND** vertex-set diagonal ≥ `_MIN_ARROW_SIZE_SCALE` (0.5, now the single home shared with the open-3 gate) × arrowhead scale **AND** a taper toward the marked end measured in the shape's own PCA frame (far-extreme half-width ≤ `_OPEN3_BASE_RATIO` (0.55) × half-width). Rectangles never taper however turned (scale-bar blocks 6×3, 3×3, tiles → blunt); fragments fail the floor; diamond / flat-tip at arrow scale still called at the exact defpoint.
- **D4 — tie-break.** Within-tier key `(seat, −alignment, centroid d)`; order-independent.
- **D2 — cost.** `_ending_near_from_grid(limit=50)` for the witness search: same hits, same order, dicts built for 50 instead of ~1,050; proven output-inert (vcheck byte-identical before/after, `vcheck_work_r6.txt` = `vcheck_work_r6b.txt`).

## 3. Timing — FINAL tree vs round 4 (alternated r4/work/r4/work, best of 3 per run, best of two runs; dims @0.0 / @0.3 / @0.5 / leaders @0.5 exclude_dimensions, s)

| sheet | round 4 | round 6 |
|---|---|---|
| 5003 | 0.007/0.007/0.006/0.025 | 0.007/0.007/0.007/0.025 (flat) |
| 10.17a | 0.055/0.035/0.012/0.039 | 0.041/0.027/0.013/0.039 |
| 11.01 | 0.108/0.065/0.035/0.115 | 0.091/0.059/0.038/0.113 |
| 21.01 | 0.795/0.617/0.301/0.838 | 0.444/0.337/0.191/0.590 |
| 3001 | 3.588/1.843/0.952/1.476 | **0.679/0.493/0.403/0.845** (−81% / −73% / −58% / −43%) |
| TOTAL | 4.846/2.739/1.442/2.978 | **1.554/1.114/0.805/2.187** (−68% / −59% / −44% / −27%) |

Where the round-5 cost was: cProfile of `find_dimensions(3001, 0.5)` on the
frozen round-5 tree (`profile_3001_dims5_work.txt`) put **64% of 1.16 s in
`_witness_at` → `_ending_near_from_grid`**, which built and rounded ~1,050
reference dicts per tip (560k `round()` calls) to keep 50; the per-end
candidate loop (`_arrow_attach` 0.13 s, `_seat_distance` 0.035 s) was minor.
The exact prune's own effect is within noise; the witness limit is what moved
the numbers. Every band on every sheet is now at or below round 4; the 3001
cost the verifier flagged (+45%/+21–26%/+29%) is replaced by −81%/−58%/−43%.

## 4. Per-row accounting — @0.3 rows that differ round 4 → round 6 (all 0.45; "truth" = nearest native defpoint)

Re-arbitrations among junk chevrons (new winner nearer by seat, farther by
centroid), unchanged from round 5:
- 11.01 e3058: e3056 (0, 4.74, 4.84) → e3753 (0, 4.17, 5.23); coords unchanged; no native dims
- 21.01 e4676: e4674 → e4694 (4.26→4.01); e4678: e4697 → e4680 (3.30→2.88), end b [274.4,126.5]→[282.7,124.8]; e4727: e4742 → e4737 (5.73→4.93); truth 30–49 pt
- 3001 e3181: e3178 → e3195 (7.14→6.17, tier 1); e2857: e2855 → e2877 (4.92→3.96), end b [346.7,136.3]→[337.2,139.0]; truth 150–207 pt

New in round 6 (nearest-member cluster seat):
- 10.17a e1225 ADDED: cluster:e1226 + cluster:e1399, detached + below-extent; no native dims
- 10.31A e2702 ADDED: cluster:e2703 + cluster:e2876, detached + below-extent; truth 450–457 pt
- 21.01 e3901: cluster:e3879 → cluster:e3878 at one end (coords unchanged; two overlapping splashes); e3879 REMOVED (its cluster pair re-sorted); e329 REMOVED [351.0,502.4] and e14 ADDED [353.9,512.3] (cluster:e149 now seats nearer e14's end; e7447 blunt at the other end on both); truth 63–234 pt
- 3001 e7868: cluster:e7650 → cluster:e2845 at end a ([325.2,558.4]→[325.2,560.3]); e7789 and e7790 ADDED (cluster:e2845 + cluster:e7650 pairs) — three overlapping stipple splashes near (325,558) re-assorted; e6292 REMOVED (cluster:e4340 end lost); e9791 REMOVED (e9789 seat 6.12 lost to blunt e9793 seat 2.94, construct falls below 0.3); truth 65–97 pt

None is a true positive: recall, precision and all 41 residuals are identical;
the nearest truth for any changed row is 30 pt (tolerance 18). **Change-A
re-labels: none remain** — the oriented class is empty on the corpus, so 11.01
e199 etc. are blunt again exactly as in round 4.

The +57 at ≤0.25 (was +60 in round 5): the seat ordering lets the two ends of
short junk shafts elect DIFFERENT contradicted candidates, so pairs the
DISTINCT rule used to drop now publish at ≤0.25 (the verifier's row-12
mechanism, confirmed). Not from the prune.

## 5. Judgement calls (each with the fixture that fails under the alternative)

1. **D1 award rule — nearer-wins between sound tiers, tie → directional; NOT a ratio, NOT a scale-stated dominance.** With cluster seat = nearest member, the verifier's D1 table needs the cluster to win at shift 2.0 (nearest member 0.5 pt) against a chevron seated 0.94 — any ratio > 1.9 fails it, and a scale-stated threshold would have to sit between 0.5 and 0.94 pt (0.1 × scale = 0.93, a 0.01 pt margin). Nearer-wins has no constant, matches the tip's behaviour on this anatomy, and the principle round 4 wanted the tier for is carried by the seat: a real arrow seats at 0.0 and is unbeatable; round 4's splash 5 pt off seats 3.5 and loses to a 1.5–2.5 pt crooked arrow (pinned). **Residual, pinned as documented:** a splash whose nearest member is nearer than a crooked arrow's base centre takes the end. Fails under "tier first + 10×": `TestARealClusterOwnsItsEndAtEveryOffset` shift 1.0/2.0; under "seat always, contradicted included": `TestNearerJunkDoesNotShadowTheTrueArrow` (round-2 blocker 2).
2. **D3 — kept the oriented rank (not the clamp-only fallback)** because two reused statements separate the cases cleanly: the open-3 gate's 0.5 × scale floor (now `_MIN_ARROW_SIZE_SCALE`, one home) and a taper — far-extreme half-width ≤ `_OPEN3_BASE_RATIO` (0.55) × half-width, in the shape's own PCA frame. Rectangles never taper however turned; fragments fail the floor; diamond / flat-tip at arrow scale still called at the exact defpoint. Fallback would fail `test_called_at_the_default_with_the_exact_defpoints`; without the taper, `TestAScaleBarIsNotADimension` fails (0.944). **Cost accepted, pinned:** a flat tip wider than 0.55 of its body is blunt.
3. **D4 — tie-break `(seat, −alignment, d)`.** Alignment before centroid because two base-seated chevrons tie on centroid too (both h/3); the straighter one wins in both orders (pinned).
4. **D2 — the witness-search limit is the fix** (output-inert, proven); the exact prune stays as a correctness-neutral bound.

## 6. D5 docs done
README tipless paragraph re-defines the figure (blunt = every terminator blunt under the split; oriented = arrow-scale + taper), re-measured 19/17/0 + "oriented-terminator proposals: 0 / 0 / 0" (the exact string `doc_claims_check` prints and `test_readme_claims` pins); README sharp-edge "Who owns a dimension end" rewritten for nearer-wins + nearest-member seat, and names the 0.05–0.4 pt window the 10× rule had and that round 4's 5-pt splash seats 3.5 and loses; DESIGN.md `find_dimensions` bullet updated; `_award_end` / `_end_tier` / `find_dimensions` docstrings state the true rule (no dominance window anywhere; `_SEAT_DOMINANCE` gone, asserted absent by a test). The "+60 from the pre-prune" line: corrected — it is the DISTINCT-rule mechanism. "6 rows" → 8 at round 5 → 15 at round 6.

## 7. Unfinished / unverified
- Finding 5 (equilateral / wide triangles) and finding 6 (concave dart) still documented, not fixed.
- The @0.3 count movement (+1 10.17a, +1 10.31A, −1 21.01) and +57 at ≤0.25 are corpus movement in the observational bands, **reported not tuned**; none touches a called or matched construct.
- FINDINGS.md: the lead's (316 → 350 applied; round-5 and round-6 entries from these parts).
- CRLF as before (`queries.py`, `DESIGN.md` pre-existing w/crlf; nothing new).

**Verifier pointers (scratch dir):** `round6_delta_queries.patch`, `queries_round6.py`, `attack_work_r6b.txt`, `vcheck_work_r6b.txt` (= `vcheck_work_r6.txt` pre-witness-limit), `corpus_after_r6.txt`, `accounting_r6_{v_r4,work}.txt`, `timing_r6_{v_r4,work}_run{1,2}.txt`, `profile_3001_dims5_work.txt`, `doc_claims_r6.txt`.
