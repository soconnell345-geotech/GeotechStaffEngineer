# planlens round-6 repair — independent adversarial verification

**Verifier** independent (did not build round 6; read only the diff and the tree) · **Date** 2026-09-11
**Under review** the frozen working tree of `C:/Users/socon/OneDrive/dev/planlens` (branch `main`, tip `1f6551c`)
**Tree state measured** — every md5 matches the lead's freeze list, re-checked unchanged after the last run:
`planlens/ir/queries.py` e2b2b3de5820fe12f8400348108e098b (07:07:32) · `test_end_ownership.py` a561fd2d… · `test_tipless_terminators.py` 0547923e… · `README.md` 4d62c94f… · `ir/DESIGN.md` 231c91f0… · `tests/test_readme_claims.py` 4ff19045… · app `doc_claims_check.py` 2dcaf502…
**Baselines** my own scratch copies from round 5, verified intact: `tmp/v_r4` (working-tree package + `queries_round4.py`, the pre-repair round-4 file) and `tmp/v_tip` (+ `queries_tip.py` = `git archive 1f6551c`). `queries_round6.py` in the scratch dir is byte-equal to the working tree.
**Python** `GeotechStaffEngineer/.venv/Scripts/python.exe`. Both repos untouched; scratch only under the job tmp dir. **RUN** = a measurement I executed; **READ** = static reading.

---

## VERDICT: SHIP — every round-5 defect is closed and measured; one documented residual carries a wrong threshold in the README (fix the sentence, no code change required to ship)

Round 6 fixes all four of my round-5 findings on the fixtures that found them and on the corpus (RUN): the cluster keeps its end at every offset against every attached foreign chevron and the leader survives with its CORRECT arrowhead (D1); the tree is now 35-54 % FASTER than round 4 rather than 15-34 % slower (D2); sub-scale glyph fragments and the 2:1-block scale bar are blunt and never called, the oriented class has zero members on the corpus, and diamond/flat-tip terminators are still called at the exact defpoint (D3); seat ties are order-independent (D4). The exact prune is exact with the real cluster radius (proof holds, on/off byte-identical for dims @0.0 AND leaders @0.5 on all ten sheets). Corpus figures, the 322-row called set and all 41 residuals are unchanged. Suites 798 / 61 / 11 green.

What remains is the residual the builder chose and documented — a stipple splash whose nearest member is nearer to a shaft end than a base-anchored arrow's own base centre takes the end — and the README states its threshold as "an arrow drafted more than ~9.6 deg off axis". **Measured, the threshold is any non-zero crookedness: 0.5 deg (seat 0.06 pt) already loses the end to a splash centred on it, and the construct is CALLED at 0.95 with the end 4.2-5.7 pt short.** The 9.6 deg figure is the off-spine cap's angle, not this rule's. Corpus-inert (corpus arrows seat at ≤ 0.036 pt), pinned only at 12 deg, and not a regression against the tip (which gave the splash the end at every angle) — but a regression against round 4 for arrows crooked 0.3-9.6 deg, where round 4 published the exact apex. Ship with the sentence corrected and the 0.5 deg fixture added; the design decision itself is the lead's.

---

## 1. Observed vs claimed — every row

| # | Claim | Observed | Source | |
|---|---|---|---|---|
| 1 | D1: `_seat_distance` cluster = nearest member; `_award_end` better-seated wins between tiers 0/1, tie → directional; `_SEAT_DOMINANCE` removed | READ: `queries.py:1509-1510` (nearest member), `:1567-1568` (`blind if blind["seat"] < sound["seat"] else sound`), `hasattr(q, "_SEAT_DOMINANCE")` is False (RUN) | READ+RUN | ✓ |
| 2 | attack.py P2 passes at every shift and every chevron seat | RUN: cluster wins all 42 (s1 ∈ 0.04-2.0 × s0 ∈ 0.94-11) cells; published 101.57-103.53 (the splash's reach), conf 0.95-0.951, foreign chevron never named | RUN | ✓ |
| 3 | the leader that owns the chevron survives as a CORRECT reading at every shift | RUN P13: "CB #4" read with `arrowhead=e26` (the chevron) at shift 0/0.1/0.5/1.0/2.0, conf 0.931 — not the cluster-misattributed reading of round 5 | RUN | ✓ |
| 4 | round 4's splash-5-pt-off case still capped | RUN P9: at 12 deg and 20 deg with the splash 5 pt off, the arrow keeps the end, b = 92.8 (tip fallback), 0.45, not called. At 3 pt off: 12 deg still the arrow (0.45); **20 deg → splash, 0.89, CALLED** (nearest member 1.5 < seat 2.5) | RUN | ✓ at 5 pt; boundary at 3 pt |
| 5 | tie → directional is the right direction | RUN P11: (0.0, 0.0) → tier 0; (0.30, 0.30) → tier 0; (0.30 vs 0.29) → tier 1; splash centred on the end vs a straight apex-anchored arrow → arrow, b = 100.0; vs a straight base-anchored arrow → arrow, b = 107.2 (the arrows-outside reading). Right direction: a real arrow seats at 0.0 in both styles and is unbeatable | RUN | ✓ |
| 6 | clusters straddling vs beside the end | RUN P10: beside the end laterally, a splash still takes the end from a 12 deg arrow up to 2 pt lateral offset (published (94.3, 0) — the reach clamps the lateral), and from a 2.4 deg arrow only at 0 offset; a straight arrow never loses | RUN | see §2 R1 |
| 7 | prune exact with the real cluster radius | READ: seat points — apex (vertex), base (centroid of a vertex subset → in the hull), axial extremes (vertices), centroid, and now the nearest MEMBER (a vertex of the cluster's point set) — all lie within R = max vertex/member distance from the vertex centroid, so seat ≥ d − R; pruned iff d − R > s_best (strict), hence seat > s_best ≥ the winner's seat: cannot win in-tier, cannot win the cross-tier nearer-wins, and cannot tie (strict); tier 2 never beats a sound candidate; s_best only decreases, so early pruning is monotone-safe. RUN: prune on/off identical for dims @0.0 AND leaders @0.5 (exclude_dimensions) on all ten sheets | READ+RUN | ✓ |
| 8 | D3 gate: axis-along AND diagonal ≥ 0.5 × scale AND taper (far-extreme half-width ≤ 0.55 × max) | READ `:1352-1373`. RUN P1/P5/P6/P12: all rectangles blunt (untapered) at every elongation and angle; 2×0.8, 1.98×0.96 fragments blunt (sub-scale); 2:1 and 1:1 scale bars blunt 0.45; diamond leg 6 oriented, apex (100, 0); trapezoid tip 0.8 and 1.6 (ratio 0.533) oriented, tip 1.7 (0.567) blunt; diamond crooked 20 deg oriented (PCA frame); 5-gon "house" oriented; sheared parallelogram blunt; diamond diag 4.94 oriented / 4.62 blunt at scale 9.31 (floor 4.655) | READ+RUN | ✓ |
| 9 | P5/P6 scenes now blunt and not called | RUN: no-text [chevron + oblong fragment] continuous 0.45 and split 0.45; text-sheet oblong 0.45; scale bars 0.45 | RUN | ✓ |
| 10 | diamond / flat-tip at arrow scale still called at the exact defpoint | RUN: `round4_repro.py` output byte-identical to my round-5 run — diamond and trapezoid `[100.0, 0.0]` 100.0 0.952 | RUN | ✓ |
| 11 | D4: within-tier key (seat, −alignment, d); P4 both orders | RUN P4/P14: straight-first and skew-first both publish `[107.2007, 0]` 0.944. A FULL tie (mirror-image chevrons ±20 deg, equal seat/score/d) resolves to the same candidate in both entity orders — by grid-cell iteration order, deterministic and arbitrary; no right answer exists there | RUN | ✓ |
| 12 | D2: `_ending_near_from_grid(limit=50)` output-inert | READ: same stable sort on distance, refs built for the first 50 only. RUN: called set @0.5 (322 rows) byte-identical to round 4; full leader set @0.0 (1662 rows) byte-identical | READ+RUN | ✓ |
| 13 | timing target < 10 % at 0.3/0.5 vs round 4 | RUN §3: totals **−54 % / −35 % / −10 %** (dims@0.0 / dims@0.5 / leaders@0.5); 3001 **−69 % / −47 % / −21 %** | RUN | ✓ (target exceeded) |
| 14 | corpus 25/25, 16/16 @0.3; 23/25, 16/16 @0.5; FPs 2/10/6/19/23/20/38; precision 5/10/12/10 | RUN: all exactly | RUN | ✓ |
| 15 | 41 residuals identical to `1f6551c` to 3 dp | RUN: residual lines byte-identical to my `v_tip` run (and to round 4) | RUN | ✓ |
| 16 | 322-row called set identical | RUN: identical to round 4 (which is what "baseline" means for the called set — the tip's is 350 rows at the documented 19/15/18/15 precision) | RUN | ✓ |
| 17 | oriented corpus population under the size floor | RUN: **0 / 0 / 0** at 0.0 / 0.3 / 0.5 (the six round-5 fragments are blunt again); blunt figure back to 19 / 17 / 0 (21.01 17/15/0, 11.01 2/2/0) | RUN | ✓ |
| 18 | README figures agree with `doc_claims_check.py` | RUN: my `doc_claims_check.py` output byte-identical to the builder's; README carries 19 / 17 / 0, 17/15/0, 2/2/0, "oriented-terminator proposals: 0 / 0 / 0", 557/158/1696, 3→2/3→2/4, 4/8/3; `test_readme_claims` 11 passed and pins `ORIENTED_TOTALS = (0, 0, 0)` | RUN | ✓ — every published corpus figure now has a command and a guard |
| 19 | docs: +60 attribution corrected; "8 rows"; dominance sentence | READ: `ROUND6_BUILDER_REPORT.md:100` attributes the band growth to the DISTINCT-rule mechanism (correct) and gives +57 — RUN: 52 removed / 109 added @0.0 vs round 4 = +57 ✓; "8 → 15 rows" at 0.3 — RUN: 8 added / 7 removed by (sheet, path, ends, conf) = 15 ✓; README "Who owns a dimension end" rewritten for nearer-wins with the 0.05-0.4 pt window named ✓ — **but the residual's threshold is misstated** (§2 R1) | READ+RUN | ✓ / ✗ one sentence |
| 20 | planlens suite 798, app 61, `test_readme_claims` 11 | RUN: **798 passed** (174 s), **61 passed**, **11 passed** | RUN | ✓ |
| 21 | the 15 changed @0.3 rows are junk, none within the scorer's tolerance of truth | RUN: all eight added rows are 0.45 capped (extent floor / detached / off-spine), spans 4.9-16.0 pt; on annotated sheets nearest native defpoint ≥ 33.7 pt; one 21.01 row's end sits 2.6 pt from a native leader TIP (it names that leader's own cluster arrowhead at 0.45) — leaders are byte-identical at every band, so it changes nothing | RUN | ✓ |

---

## 2. Findings, most severe first

### R1 — MEDIUM (residual by design) + LOW (documentation): the seat-only award hands a base-anchored arrow's end to a stipple splash at ANY non-zero crookedness, and the README says "more than ~9.6 deg"
`queries.py:1567-1568` (`_award_end`), `:1509-1510` (cluster seat = nearest member); README "Who owns a dimension end" bullet; fixture pinned only at 12 deg (`test_end_ownership.py::TestTheDocumentedResidualOfTheTrade`).

Fixture (`attack_r6.py` P9): the builder's own arrows-outside anatomy — shaft from H to 100−H, apexes at the defpoints 0 and 100, witnesses, text — right arrow rotated `off_deg` about its apex (base-centre seat = 2H·sin(off/2)), a 7×3 stipple splash centred `off` pt beyond the shaft end. Truth b = [100, 0]. S = splash owns the end, A = arrow; * = called at 0.5.

| arrow off_deg (seat) | off 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|---|---|
| 0.0 (0.00) | A 100.0* | A 100.0* | A 100.0* | A 100.0* | A 100.0* | A 100.0* | A 100.0* |
| **0.5 (0.06)** | **S 94.3*** | **S 94.8*** | **S 95.3*** | **S 95.8*** | A 100.0* | A 100.0* | A 100.0* |
| 1.0 (0.13) | S 94.3* | S 94.8* | S 95.3* | S 95.8* | A 100.0* | A 100.0* | A 100.0* |
| 2.4 (0.30) | S 94.3* | S 94.8* | S 95.3* | S 95.8* | A 100.0* | A 100.0* | A 100.0* |
| 5.0 (0.63) | S 94.3* | S 94.8* | S 95.3* | S 95.8* | S 96.3* | A 100.0* | A 100.0* |
| 12.0 (1.51) | S 94.3* | S 94.8* | S 95.3* | S 95.8* | S 96.3* | A 92.8 (0.45) | A 92.8 (0.45) |
| 20.0 (2.50) | S 94.3* | S 94.8* | S 95.3* | S 95.8* | S 96.3* | S 96.3 (0.89*) | A 92.8 (0.45) |

Laterally (P10): a splash centred on the end but 1-2 pt beside the line still takes it from a 12 deg arrow (published (94.3, 0)); from a 2.4 deg arrow only at 0 offset; from a straight arrow never.

Reading: the rule is exactly "nearest splash member nearer than the arrow's base-centre error", so a splash sitting on the shaft end (nearest member 0.04 pt) beats any arrow whose base is more than 0.04 pt off — 0.33 deg. Consequences: the end is published inside the splash, 4.2-5.7 pt short of the defpoint, at 0.95, uncapped (the off-spine cap belonged to the arrow, which lost). Three-tree standing: the **tip** gave the splash the end at every angle including 0 deg (centroid 0.04 vs 2.4), so this is not a regression against what is installed; **round 4** kept the arrow at every angle and published the exact apex up to ~9.6 deg, so for 0.3-9.6 deg this scene regresses from an exact 100.0 to 94.3-96.3; **round 5** lost only above ~3.3 deg (the 10× window). Realism: it needs a base-anchored drawn arrow drafted imperfectly AND a cluster candidate at that shaft end, which the cluster builder issues only for a fragment-density spike ≥ 2× the annulus (uniform texture is rejected) — hand-rotated arrow blocks in a hatched detail is the scene. Corpus-inert (RUN: corpus arrows seat at 0.007-0.036 pt; called set identical).

The **documentation defect** is concrete: the README bullet says the splash takes the end from "an arrow drafted more than ~9.6 deg off axis in the arrows-outside style" — 9.6 deg is `asin(half_width / leg)`, the angle at which the OFF-SPINE cap engages; it has nothing to do with this rule, which triggers at 0.5 deg in the same fixture. The `_end_tier` / `_award_end` docstrings describe it correctly ("a splash whose nearest member sits closer … than a crooked arrow's base centre"); only the README carries the number. Fix: drop "more than ~9.6 deg", state the real condition, and pin the 0.5 deg row so it cannot move silently. If the lead wants the rule itself narrowed (e.g. a directional candidate keeps its end when seated within its own drafting tolerance — half-width, 1.2 pt — and the splash's reach does not enclose the end), that is a design choice with the same both-ways fixtures the builder already has.

### R2 — LOW (observation, consistent with existing gates): diamond / flat-tip callability now depends on the sheet's arrowhead-scale statistic
`queries.py:1356-1360`. The size floor uses `max_arrowhead_size`, which without an explicit value is 25 % of the median open-segment length (floored at 1 % of the page diagonal) — a sheet statistic, not a property of the arrows. RUN P12: the finding-1 diamond dimension on `core()` alone (scale 10.0, floor 5.0) is called at 0.94; add five 400 pt lines (median 400 → scale 100 → floor 50) and it is blunt-capped at 0.45 with `arrowhead_ids` emptied, while the same scene with closed triangles keeps its ids (closed 3-5-gons have no lower size floor). The open-3 chevron gate has had this dependence since Phase 3.1, so it is not new — but the diamond now inherits it where a closed triangle does not. Worth one line in the sharp-edges list; no fixture disagrees with the docs.

### R3 — INFO: a full tie (equal seat, alignment and centroid distance) resolves by grid-cell iteration order
RUN P14: mirror-image chevrons at ±20 deg, both base-seated, both orders → the same (lower) candidate. Deterministic, order-independent, arbitrary; there is no correct answer for a genuine mirror pair. The docstring's "short of an exact tie on both keys" (`:2364-2367`) is honest.

### R4 — INFO: the observational band churn vs round 4 is 52 removed / 109 added at 0.0 (net +57) and 7 / 8 at 0.3 (15 rows)
All ≤ 0.45; the 0.3-band additions are cluster-cluster pairs and pattern triangles under the extent floor; the 0.0-band additions are the DISTINCT-rule mechanism I measured in round 5 plus the nearest-member seat electing different clusters. Builder's numbers (+57, 15) reproduce. The builder's ledger now attributes the growth correctly.

---

## 3. Timing — frozen round-6 tree vs the round-4 baseline (RUN `timing_v.py`, best-of-3 per run; order work → r4 → work → r4; both runs of each shown, min in bold)

| sheet | dims@0.0 r6 | r4 | Δ (min/min) | dims@0.5 r6 | r4 | Δ | leaders@0.5 r6 | r4 | Δ |
|---|---|---|---|---|---|---|---|---|---|
| 5003 | **0.007** / 0.007 | **0.007** / 0.007 | 0 % | 0.007 | 0.007 | 0 % | 0.025 | 0.025 | 0 % |
| 10.17a | 0.040 / **0.039** | **0.055** / 0.055 | **−29 %** | 0.014 / **0.013** | **0.012** / 0.012 | +0.001 s (noise) | **0.039** / 0.039 | **0.037** / 0.037 | +0.002 s |
| 11.01 | **0.088** / 0.095 | **0.107** / 0.107 | **−18 %** | **0.037** / 0.039 | **0.036** / 0.036 | +0.001 s | **0.113** / 0.121 | 0.115 / **0.111** | +0.002 s |
| 21.01 | 0.555 / **0.533** | **0.757** / 0.882 | **−30 %** | **0.260** / 0.286 | **0.311** / 0.341 | **−16 %** | **0.795** / 0.798 | **0.780** / 0.890 | +2 % |
| **3001** | **0.885** / 1.064 | **2.826** / 2.905 | **−69 %** | **0.525** / 0.661 | **0.983** / 0.990 | **−47 %** | **1.136** / 1.253 | **1.446** / 1.451 | **−21 %** |
| 10.31A | 0.244 / **0.240** | 0.271 / **0.239** | 0 % | **0.077** / 0.085 | 0.089 / **0.058** | +0.02 s (r4 itself varied 0.058-0.089) | 0.321 / **0.316** | 0.319 / **0.246** | +0.07 s (r4 itself varied 0.246-0.319) |
| **TOTAL** | **1.889** / 2.046 | **4.089** / 4.261 | **−54 %** | **0.975** / 1.146 | **1.492** / 1.497 | **−35 %** | **2.684** / 2.801 | **2.970** / 3.013 | **−10 %** |

Round 5 (my previous table) was +34 % / +15 % / +16 % on the same totals; the witness-search limit reverses it. Every cell that appears slower is a sub-0.1 s number inside the run-to-run spread of the baseline itself. The builder's report claims larger gains (3001 −81 %, total −68 %) from a higher r4 baseline (3.59 s); direction and conclusion agree, and the < 10 % target is met on every total with margin. Cost question closed.

---

## 4. What verified SOUND beyond the table (RUN unless noted)

- **`_seat_distance` invariance** (P3, round-5 fixtures re-run): chevron / base-leg / closed triangle in all vertex rotations seat at 0.000000 with the apex at the end and with the base at the end; duplicate closing vertex and reversed traversal of a diamond leave state, score, apex and seat unchanged.
- **Finding 5** (wide-triangle apex vote) still open and documented; **finding 6** (concave dart) still deleted and documented — unchanged by round 6.
- **Hoisted ceiling**: for every sheet and both thresholds `find_dimensions(min_confidence=c)` equals the @0.0 result filtered at `c` (vcheck cross-check, no mismatch).
- **No-text "pointing shape" rule** unchanged; with the size/taper gates the [arrow + fragment] and split-leg fragment scenes that it admitted in round 5 are blunt and capped (P5).
- **`exclude_dimensions` arbitration**: with the cluster owning its end, no dimension names the leader's chevron at any threshold (builder's test, and P13 shows the leader read with `e26` at every shift).

---

## 5. Could not verify, and why

- **Whether R1 bites on a real non-Mecklenburg drawing.** No corpus with hand-rotated base-anchored arrows in stipple exists in either repo; the fixture is the only falsification. The corpus's own arrows are exact to ≤ 0.036 pt, so it cannot see the rule at all.
- **The builder's profiling attribution** ("64 % of 3001's default run in dict building") — not re-profiled; only the end-to-end timing above, which confirms the effect.
- **The app's full gate** — not run (instructed).
- **Render adjudication, DXF truth regeneration, the 9.59 deg arrows-outside sharp edge** — untouched by round 6; not re-verified.
- **Tree state after 07:07:32** — md5s unchanged through my last run; anything edited after this file is written is unverified.

---

## Appendix A — scripts and raw outputs (all under `C:/Users/socon/.claude/jobs/69be0e95/tmp`)

- `vcheck.py work` → `vcheck_work_r6v.txt` (recall, residuals, FPs, precision, blunt/oriented counts, band counts, full sets) — diffed against `vcheck_v_r4.txt` and `vcheck_v_tip.txt` (round-5 baselines, verified intact).
- `acct_r6.py` → `acct_r6_out.txt`: prune on/off equality (dims @0.0 + leaders @0.5, ten sheets), oriented population, the added @0.3 rows against truth.
- `timing_v.py` → `timing_verifier_r6.txt` (§3).
- `round4_repro.py work` → `round4_verifier_r6.txt` (byte-identical to round 5).
- `attack.py work` → `attack_work_r6v.txt` (round-5 fixtures P1-P8 on the frozen tree); `attack_r6.py work` → `attack_r6_work.txt` (P9-P14, source below).
- `suite_planlens_r6.txt` (798 passed), `suite_app_r6.txt` (61 passed), `doc_claims_verifier_r6.txt` (byte-identical to the builder's `doc_claims_r6.txt`).

## Appendix B — `attack_r6.py` (runnable: `<venv>/python attack_r6.py work`)

```python
"""Verifier's round-6 fixtures: nearest-member cluster seat, seat-only award, taper/size gates. Usage: attack_r6.py <tree>"""
import math, os, sys
TREE = sys.argv[1]; TMP = r"C:\Users\socon\.claude\jobs\69be0e95\tmp"
if TREE != "work": sys.path.insert(0, os.path.join(TMP, TREE))
import planlens
from planlens.ir import queries as q
from planlens.ir.results import DrawingIR, Line, Polyline, TextItem
print("[%s] %s  has _SEAT_DOMINANCE=%s  has _MIN_ARROW_SIZE_SCALE=%s" % (TREE, planlens.__file__, hasattr(q, "_SEAT_DOMINANCE"), hasattr(q, "_MIN_ARROW_SIZE_SCALE")))
MAS = 9.31
H = math.sqrt(7.3 ** 2 - 1.2 ** 2)

def ir_of(ents):
    ir = DrawingIR(units="pt", coordinate_space="page", origin="bottom_left", source="pdf_vector", width=612.0, height=792.0)
    for i, e in enumerate(ents): e.id = "e%d" % i; ir.add(e)
    return ir
def chevron(apex, d, leg=7.3, base=2.4):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    return Polyline(vertices=[(bx+px*base/2, by+py*base/2), apex, (bx-px*base/2, by-py*base/2)], closed=False)
def base_leg(apex, d, leg=7.3, base=2.4):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    return Polyline(vertices=[(bx+px*base/2, by+py*base/2), (bx-px*base/2, by-py*base/2), apex], closed=False)
def tri_closed(apex, d, leg=6.0, base=2.0):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    return Polyline(vertices=[apex, (bx+px*base/2, by+py*base/2), (bx-px*base/2, by-py*base/2)], closed=True)
def diamond(apex, d, leg=6.0, w=2.4):
    dx, dy = d; px, py = -dy, dx
    return Polyline(vertices=[apex, (apex[0]-dx*leg/2+px*w/2, apex[1]-dy*leg/2+py*w/2), (apex[0]-dx*leg, apex[1]-dy*leg), (apex[0]-dx*leg/2-px*w/2, apex[1]-dy*leg/2-py*w/2)], closed=True)
def trapezoid(apex, d, leg=6.0, tip_w=0.8, base=3.0):
    dx, dy = d; px, py = -dy, dx; bx, by = apex[0]-dx*leg, apex[1]-dy*leg
    return Polyline(vertices=[(apex[0]+px*tip_w/2, apex[1]+py*tip_w/2), (apex[0]-px*tip_w/2, apex[1]-py*tip_w/2), (bx-px*base/2, by-py*base/2), (bx+px*base/2, by+py*base/2)], closed=True)
def house(apex, d, leg=6.0, w=2.4, body=0.5):
    dx, dy = d; px, py = -dy, dx; sx, sy = apex[0]-dx*leg*(1-body), apex[1]-dy*leg*(1-body); bx, by = apex[0]-dx*leg, apex[1]-dy*leg
    return Polyline(vertices=[apex, (sx+px*w/2, sy+py*w/2), (bx+px*w/2, by+py*w/2), (bx-px*w/2, by-py*w/2), (sx-px*w/2, sy-py*w/2)], closed=True)
def fill_cluster(tip, d, n=7, span=3.0, lateral=0.0):
    out = []; px, py = -d[1], d[0]
    for i in range(n):
        t = (i/(n-1.0))*span - 0.5*span
        for k in (-0.6, 0.0, 0.6):
            x = tip[0]+d[0]*t+px*(k+lateral); y = tip[1]+d[1]*t+py*(k+lateral)
            out.append(Line(start=(x, y), end=(x+0.06, y+0.06)))
    return out
def core(text=True):
    e = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0))]
    if text: e.append(TextItem(content="100'", position=(50.0, 6.0), height=6.0))
    return e
def dims(ents, mc=0.0, shaft="e0", mas=MAS):
    ps = q.find_dimensions(ir_of(ents), max_arrowhead_size=mas, min_confidence=mc)
    return [p for p in ps if p["shaft_id"] == shaft or p["evidence"]["path"] == "split_shaft"]
def show(tag, ents, mc=0.0, shaft="e0", mas=MAS):
    ps = dims(ents, mc, shaft, mas)
    if not ps: print("  %-62s NO PROPOSAL" % tag); return None
    p = ps[0]; ev = p["evidence"]
    flags = ",".join(sorted(k for k in ev if k.startswith("arrow_") or "blunt" in k or "oriented" in k or k == "below_extent_floor"))
    print("  %-62s b=%-18s len=%-8s conf=%-6s ids=%s [%s]" % (tag, p["end_b_xy"], p["length"], p["confidence"], p["arrowhead_ids"], flags))
    return p
def attach(shape, sdir=(1.0, 0.0), tip=(100.0, 0.0), mas=MAS):
    return q._arrow_attach([tuple(v) for v in shape.vertices], "triangle", sdir, tip, mas)

A = chevron((0.0, 0.0), (-1.0, 0.0))

print("\nP9  SPLASH vs CROOKED ARROW (arrows-outside anatomy): right arrow crooked off_deg about its apex, splash centred `off` pt beyond the shaft end. Truth b=[100,0]")
for off_deg in (0.0, 0.5, 1.0, 2.4, 5.0, 12.0, 20.0):
    a = math.radians(off_deg); seat_arrow = 2*H*math.sin(a/2)
    row = []
    for off in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        ents = [Line(start=(H, 0.0), end=(100.0 - H, 0.0)), Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0)), TextItem(content="100'", position=(50.0, 6.0), height=6.0),
                base_leg((0.0, 0.0), (-1.0, 0.0)), base_leg((100.0, 0.0), (math.cos(a), math.sin(a)))] + fill_cluster((100.0 - H + off, 0.0), (1.0, 0.0))
        p = dims(ents, 0.0)[0]
        who = "S" if any(i.startswith("cluster") for i in p["arrowhead_ids"]) else "A"
        row.append("off=%.1f:%s b=%.1f c=%.2f%s" % (off, who, p["end_b_xy"][0], p["confidence"], "*" if p["confidence"] >= 0.5 else ""))
    print("  off_deg=%4.1f (arrow seat %.2f): %s" % (off_deg, seat_arrow, " | ".join(row)))

print("\nP10 SPLASH BESIDE the end (lateral offset, centred on the shaft end axially) vs the same arrows")
for off_deg in (0.0, 2.4, 12.0):
    a = math.radians(off_deg); seat_arrow = 2*H*math.sin(a/2)
    row = []
    for lat in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
        ents = [Line(start=(H, 0.0), end=(100.0 - H, 0.0)), Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0)), TextItem(content="100'", position=(50.0, 6.0), height=6.0),
                base_leg((0.0, 0.0), (-1.0, 0.0)), base_leg((100.0, 0.0), (math.cos(a), math.sin(a)))] + fill_cluster((100.0 - H, 0.0), (1.0, 0.0), lateral=lat)
        ps = dims(ents, 0.0)
        if not ps: row.append("lat=%.1f:NONE" % lat); continue
        p = ps[0]; who = "S" if any(i.startswith("cluster") for i in p["arrowhead_ids"]) else "A"
        row.append("lat=%.1f:%s b=(%.1f,%.1f) c=%.2f%s" % (lat, who, p["end_b_xy"][0], p["end_b_xy"][1], p["confidence"], "*" if p["confidence"] >= 0.5 else ""))
    print("  off_deg=%4.1f (arrow seat %.2f): %s" % (off_deg, seat_arrow, " | ".join(row)))

print("\nP11 TIES and direction of the tie-break")
for tag, f in (("tier0 seat 0.0 vs tier1 seat 0.0", {0: {"tier": 0, "seat": 0.0}, 1: {"tier": 1, "seat": 0.0}}), ("tier0 seat 0.30 vs tier1 seat 0.30", {0: {"tier": 0, "seat": 0.3}, 1: {"tier": 1, "seat": 0.3}}), ("tier0 seat 0.30 vs tier1 seat 0.29", {0: {"tier": 0, "seat": 0.3}, 1: {"tier": 1, "seat": 0.29}}), ("tier1 seat 5 vs tier2 seat 0", {1: {"tier": 1, "seat": 5.0}, 2: {"tier": 2, "seat": 0.0}})):
    print("  %-40s -> tier %d" % (tag, q._award_end(f)["tier"]))
show("straight arrow apex@end + splash centred on the end", core() + [A, chevron((100.0, 0.0), (1.0, 0.0))] + fill_cluster((100.0, 0.0), (1.0, 0.0)))
show("straight arrow base@end + splash centred on the end", core() + [A, base_leg((100.0 + H, 0.0), (1.0, 0.0))] + fill_cluster((100.0, 0.0), (1.0, 0.0)))

print("\nP12 TAPER / SIZE gate")
for tag, shape in (("diamond leg6 w2.4 inside", diamond((100.0, 0.0), (1.0, 0.0))),
                   ("diamond leg6 w2.4 BEYOND the tip (100..106)", diamond((106.0, 0.0), (1.0, 0.0))),
                   ("diamond crooked 20 deg", diamond((100.0, 0.0), (math.cos(math.radians(20)), math.sin(math.radians(20))))),
                   ("trapezoid tip0.8/base3 inside", trapezoid((100.0, 0.0), (1.0, 0.0))),
                   ("trapezoid tip1.6/base3 (0.533)", trapezoid((100.0, 0.0), (1.0, 0.0), tip_w=1.6)),
                   ("trapezoid tip1.7/base3 (0.567)", trapezoid((100.0, 0.0), (1.0, 0.0), tip_w=1.7)),
                   ("trapezoid REVERSED (wide end at the tip)", trapezoid((94.0, 0.0), (-1.0, 0.0))),
                   ("house 5-gon leg6 w2.4", house((100.0, 0.0), (1.0, 0.0))),
                   ("rect 6x1 (untapered oblong)", Polyline(vertices=[(94.0, -0.5), (100.0, -0.5), (100.0, 0.5), (94.0, 0.5)], closed=True)),
                   ("parallelogram 6x2 sheared (slash-like)", Polyline(vertices=[(94.0, -1.0), (99.0, -1.0), (100.0, 1.0), (95.0, 1.0)], closed=True)),
                   ("diamond leg4.6 w1.8 (diag 4.94 > floor 4.655)", diamond((100.0, 0.0), (1.0, 0.0), leg=4.6, w=1.8)),
                   ("diamond leg4.3 w1.7 (diag 4.62 < floor 4.655)", diamond((100.0, 0.0), (1.0, 0.0), leg=4.3, w=1.7))):
    a = attach(shape); vs = [tuple(v) for v in shape.vertices]; g = q._arrow_geometry(vs)
    print("  %-48s pointed=%-5s state=%-8s score=%.3f apex=%s" % (tag, g.pointed, a.state, a.score, tuple(round(x, 2) for x in a.apex)))
for tag, extra in (("core only (scale 10.0, floor 5.0)", []), ("plus five 400 pt lines (scale 100, floor 50)", [Line(start=(0.0, 200.0 + 20*i), end=(400.0, 200.0 + 20*i)) for i in range(5)])):
    dia = core() + [diamond((0.0, 0.0), (-1.0, 0.0)), diamond((100.0, 0.0), (1.0, 0.0))] + extra
    tri = core() + [tri_closed((0.0, 0.0), (-1.0, 0.0)), tri_closed((100.0, 0.0), (1.0, 0.0))] + extra
    print("   " + tag)
    show("    diamond ends, default scale", dia, mas=None)
    show("    closed-triangle ends, default scale", tri, mas=None)

print("\nP13 the finding-2 leader: which ARROWHEAD does the surviving 'CB #4' leader carry")
for s1 in (0.0, 0.1, 0.5, 1.0, 2.0):
    ents = core() + [A] + fill_cluster((100.0 + s1, 0.0), (1.0, 0.0)) + [chevron((108.0, 0.5), (1.0, 0.0)), Line(start=(108.0 - H, 0.5), end=(60.0, -10.0)), TextItem(content="CB #4", position=(54.0, -12.0), height=3.0)]
    ir = ir_of(ents)
    L = [(l.get("arrowhead_id"), l["confidence"], l["tip_xy"]) for l in q.find_leaders(ir, max_arrowhead_size=MAS, min_confidence=0.5, exclude_dimensions=True) if l.get("text") == "CB #4"]
    print("  shift=%.1f -> CB #4 readings: %s" % (s1, L))

print("\nP14 order independence with the (seat, -score, d) key")
for order in ("straight-first", "skew-first"):
    a20 = math.radians(20.0)
    c1 = chevron((100.0 + H, 0.0), (1.0, 0.0)); c2 = chevron((100.0 + H*math.cos(a20), H*math.sin(a20)), (math.cos(a20), math.sin(a20)))
    show("order=%s" % order, core() + [A] + ([c1, c2] if order == "straight-first" else [c2, c1]))
for order in ("up-first", "down-first"):
    a20 = math.radians(20.0)
    cu = chevron((100.0 + H*math.cos(a20), H*math.sin(a20)), (math.cos(a20), math.sin(a20))); cd = chevron((100.0 + H*math.cos(a20), -H*math.sin(a20)), (math.cos(a20), -math.sin(a20)))
    show("mirrored pair order=%s" % order, core() + [A] + ([cu, cd] if order == "up-first" else [cd, cu]))
```

## Appendix C — the R1 fixture as a pytest (drop beside `test_end_ownership.py`; it FAILS on the frozen tree at every row, which is the point — it pins the residual at its real threshold once the README is corrected and the expected values are flipped to the documented ones)

```python
import math
import pytest
from planlens.ir import queries as q
from planlens.ir.results import Line, TextItem
from planlens.ir.tests.test_arrow_direction import MAS, _arrow_base_leg, _ir
from planlens.ir.tests.test_end_ownership import fill_cluster, _H

@pytest.mark.parametrize("off_deg", [0.5, 1.0, 2.4, 5.0])   # far below the README's "~9.6 deg"
def test_a_splash_on_the_end_takes_it_from_a_barely_crooked_arrow(off_deg):
    a = math.radians(off_deg)
    ents = [Line(start=(_H, 0.0), end=(100.0 - _H, 0.0)),
            Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0)),
            TextItem(content="100'", position=(50.0, 6.0), height=6.0),
            _arrow_base_leg((0.0, 0.0), (-1.0, 0.0)),
            _arrow_base_leg((100.0, 0.0), (math.cos(a), math.sin(a)))] + fill_cluster((100.0 - _H, 0.0), (1.0, 0.0))
    p = [p for p in q.find_dimensions(_ir(ents), max_arrowhead_size=MAS, min_confidence=0.5) if p["shaft_id"] == "e0"][0]
    # As the README currently reads (arrow keeps its end below ~9.6 deg):
    assert p["arrowhead_ids"] == ["e4", "e5"]                       # FAILS on the frozen tree at all four rows: ['e4', 'cluster:e10'], end 94.33, conf 0.947 (RUN)
    assert abs(p["end_b_xy"][0] - 100.0) < 0.05                     # FAILS: 94.33
```
