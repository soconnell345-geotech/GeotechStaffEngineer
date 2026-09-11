# planlens round-5 repair — independent adversarial verification

**Verifier** independent (did not build the repair; did not read the builder's reasoning) · **Date** 2026-09-10
**Under review** the uncommitted working tree of `C:/Users/socon/OneDrive/dev/planlens` (branch `main`, tip `1f6551c`)
**Tree state measured** `planlens/ir/queries.py` md5 `c0c67b3407a67568f5148c810c6a3ab8`, mtime 2026-09-10 20:32:35 — re-checked unchanged after the last measurement. **The tree moved once during this pass** (queries.py 20:32:35, `test_end_ownership.py` 20:32:54, after I had begun reading): the version that ships carries an `_EXACT_SEAT_PRUNE` the briefing said had been REMOVED. Everything below is measured on the 20:32 state.
**Baselines** scratch copies of the working-tree package with `queries.py` swapped: `tmp/v_r4` (= `queries_round4.py`, the pre-repair round-4 file) and `tmp/v_tip` (= `queries_tip.py`, verified byte-equal to `git archive 1f6551c` after CRLF normalisation). `sys.path.insert(0, …)` beats the editable finder (it is appended to `meta_path`); every run prints the `planlens.__file__` it resolved.
**Python** `GeotechStaffEngineer/.venv/Scripts/python.exe`. Both repos untouched; scratch only under `C:/Users/socon/.claude/jobs/69be0e95/tmp`.
**Runs vs reading** — every table row says which. "RUN" = a measurement I executed; "READ" = static code reading.

---

## VERDICT: DO NOT SHIP as-is — one blocking defect (finding 2 is repaired only inside a 0.05-0.4 pt window), one cost flag, three policy consequences to accept explicitly or fix

The repair is sound on the corpus and its fixtures: every headline figure reproduces (RUN), the called set at 0.5 is byte-identical to the pre-repair tree (RUN), all 41 matched defpoint/tip residuals are identical to `1f6551c` AND to round 4 to 3 dp (RUN), the new exact prune is provably and measurably inert (READ + RUN), findings 1, 3 and 4 are fixed by the previous pass's own fixtures (RUN), and both suites are green (RUN: 724 / 61).

But the central claim of Change B — "finding 2 fixed" — holds only when the fill cluster is seated within `s0/10 - 0.04` pt of the shaft end, i.e. **< 0.05 pt against the finding-2 chevron and < 0.42 pt against any attached foreign chevron at all**. The builder's own constant docstring says real corpus clusters sit 0.5-2 pt off; at those offsets the foreign chevron still takes the end, publishes 108-110.5 for a 100 pt line at 0.94, is named in `arrowhead_ids`, and the leader that owns it is deleted by `exclude_dimensions` — finding 2 verbatim, on the tip's own cluster anatomy, where the tip got it right at every offset (three-tree table in §2). The README sentence written for it ("a stipple-rendered arrowhead centred on its own end is not lost to a foreign chevron") is true only in that window and does not say so.

---

## 1. Observed vs claimed — every row

| # | Claim (builder) | Claimed | Observed | Source | |
|---|---|---|---|---|---|
| 1 | corpus @0.3 leaders | 25/25 | **25/25** | RUN `vcheck.py work` | ✓ |
| 2 | corpus @0.3 dims | 16/16 | **16/16** | RUN | ✓ |
| 3 | corpus @0.5 leaders | 23/25 | **23/25** | RUN | ✓ |
| 4 | corpus @0.5 dims | 16/16 | **16/16** | RUN | ✓ |
| 5 | leader FPs @0.5 (2000a/2000b/3000/5003/10.17a/10.25a/11.01) | 2/10/6/19/23/20/38 | **2/10/6/19/23/20/38** | RUN | ✓ |
| 6 | dim precision @0.5 (21.01/3001/11.01/10.31A) | 5/10/12/10 | **5/10/12/10** | RUN | ✓ |
| 7 | 41 matched defpoint residuals identical to `1f6551c` to 3 dp | identical | **identical** — also identical to round 4; leader med 7.197 / dim med 0.026, max 0.036 | RUN, all three trees | ✓ |
| 8 | called set @0.5, 322 constructs, byte-identical to baseline | identical | **322 rows, byte-identical to round 4** (vs tip: 36 diff lines, the documented 19/15/18/15 → 5/10/12/10 precision change) | RUN, `diff` | ✓ |
| 9 | @0.3 dim set 369 rows, per-sheet counts identical | identical | **identical** (124/66/50/67/44/8/6/4 per sheet) | RUN | ✓ |
| 10 | "6 rows changed at 0.3, uncorroborated 0.45 junk, none matched to truth" | 6 | **8 rows differ**: the 6 candidate swaps + 2 re-labels (11.01 blunt→oriented row, 21.01 `cluster:e149` row now also naming `e7447`). All eight at 0.45; the six on annotated sheets are 23-207 pt from any truth point; the two 11.01 rows have no truth to match (sheet carries no native annotation) but are sub-6-pt pattern triangles under the extent floor | RUN `vcheck` diff + `prune_acct.py` | ✗ count, ✓ substance |
| 11 | ≤0.25 band grew by 60 (0.25: 304→350, 0.214: 42→47, 0.0: 16→25) | +60 | **+60 net, exactly those histograms** — but as churn: 60 rows removed, 120 added | RUN | ✓ number |
| 12 | …"attributed to removing the pre-prune" | pre-prune | **wrong attribution.** The shipped tree HAS a prune (`_EXACT_SEAT_PRUNE`), and it is inert: on/off identical on all ten sheets (RUN). The growth is the seat ordering electing DIFFERENT contradicted candidates at the two ends of short junk shafts where centroid ordering elected the same one and the "two arrowheads must be DISTINCT" rule dropped the pair — measured: DISTINCT-skips 2000a 41→36 (+5 proposals), 2000b 41→39 (+2), 3001 215→192 (+23), 10.31A 131→117 (+14); each delta equals the sheet's proposal growth | RUN (instrumented scratch copies) | ✗ |
| 13 | leaders identical at every band | identical | **identical**: full @0.0 leader set (1662 rows) byte-identical to round 4 | RUN | ✓ |
| 14 | blunt figure 19/17/0 → 18/16/0 (21.01 17/15/0, 11.01 1/1/0) | 18/16/0 | **18/16/0**, per sheet as claimed; `doc_claims_check.py` output byte-identical to the builder's | RUN | ✓ |
| 15 | six proposals re-labelled blunt→oriented (5 on 11.01, 1 on 21.01), all capped on other rungs, none ≥0.5, per-sheet proposal counts unchanged | — | **6 (5+1)**, conf 0.45×2 (detached+off-spine) and 0.25×4 (contradicted); none ≥0.5. Per-sheet counts @0.3/@0.5 unchanged; @0.0 counts changed (row 11) | RUN | ✓ |
| 16 | …"none matched to truth" | — | 21.01 one: 13-15 pt from the nearest leader tip, 223 pt from any defpoint (unmatched at the 18 pt tolerance). 11.01 five: **no native truth exists on 11.01**, so the claim is vacuous there; by geometry all six "oriented" candidates are 0.6-2.0 pt 4-vertex glyph fragments (elong 1.73-3.51), none a terminator | RUN `prune_acct.py` | ✓ (vacuous on 11.01) |
| 17 | `round4_repro.py` on the repaired tree | (builder's `round4_after_B.txt`) | **byte-identical** to the builder's output: findings 1/3/4 fixed, finding 5 open (documented), finding 6 deleted (documented), item 4 sound | RUN | ✓ |
| 18 | planlens suite | (not stated; was 632) | **724 passed**, 0 failed, 174 s | RUN | ✓ |
| 19 | app drawing tests | 58 | **61 passed** (3 new since the brief), 1.6 s | RUN | ✓ |
| 20 | timing of the repaired tree | (never measured — `timing_work.txt` is 0 bytes) | see §3: **+34 % dims@0.0, +15 % dims@0.5, +16 % leaders@0.5 vs pre-repair**, 3001 +45 %/+21 %/+29 % | RUN | flag |
| 21 | README figures agree with `doc_claims_check.py` | — | all agree (18/16/0; 17/15/0; 1/1/0; 557/158/1696; 3→2, 3→2, 4; 4/8/3). The "six oriented (five/one)" figure agrees with my run but **has no command** — `doc_claims_check.py` does not produce it | RUN + READ | ✓ / gap |

---

## 2. Confirmed defects, most severe first

### D1 — BLOCKING. Finding 2 is repaired only inside a 0.05-0.4 pt seat window; on realistic cluster offsets the foreign chevron still takes the end and the leader is still deleted
`planlens/ir/queries.py:1482` (`_SEAT_DOMINANCE = 10.0`), `:1485-1504` (`_award_end`), `:1424-1450` (`_seat_distance`, cluster seat = centroid distance)

The rule: a tier-1 cluster beats the tier-0 chevron only if `seat_cluster * 10 < seat_chevron`. A fill cluster's seat is its centroid distance, which for the Mecklenburg stipple anatomy is the splash offset **+ 0.04 pt** (member centres). A foreign chevron pointing outward along the line is tier 0 whenever its centroid is within `0.75·max(scale, h)` = 6.98 pt, i.e. base-centre seat up to ~4.6 pt. So the cluster wins only when `offset < seat_chevron/10 - 0.04`: **< 0.054 pt** against the finding-2 chevron (seat 0.94) and **< 0.42 pt** against the farthest attached one. The constant's own docstring (`:1470-1474`) concedes it: "at the 0.5-2 pt centroid offsets real corpus clusters carry, no attached directional candidate can be dominated".

Fixture (`attack.py` P2): 100 pt witnessed, texted dimension, true chevron at 0, fill cluster straddling `(100 + s1, 0)`, foreign chevron with base-centre gap `s0` on-axis. Truth `end_b ≈ [100, 0]`.

| cluster offset s1 | chevron seat s0 | **repaired** | pre-repair round 4 | tip `1f6551c` |
|---|---|---|---|---|
| 0.04 | 0.94 | cluster, b=[101.57, 0] 0.951 | chevron b=[108.14, 0] 0.942 | cluster b=[101.57, -0.57] 0.949 |
| 0.10 | 0.94 | **chevron b=[108.14, 0] 0.942** | chevron | cluster |
| 0.30 | 0.94 / 2.7 / 3.0 / 3.3 | **chevron b=108.1 / 109.9 / 110.2 / 110.5, conf 0.936-0.942** | chevron | cluster at all four |
| 0.50 | 0.94 / 2.7 / 3.0 / 3.3 | **chevron** (same numbers) | chevron | cluster |
| 1.00 / 2.00 | 0.94 / 2.7 / 3.0 / 3.3 | **chevron** | chevron | cluster |
| 0.30-2.00 | 5.0 / 9.0 / 11.0 | cluster — only because a chevron seated ≥ 5 pt is DETACHED (tier 1), so seat decides within the tier, not the dominance rule | cluster | cluster |

The exact finding-2 anatomy (foreign apex `(108, 0.5)`, seat 0.94) with the leader that owns the chevron added (`Line (108-H, 0.5)→(60, -10)`, text "CB #4"), cluster shifted by `s1`:

| shift | repaired `end_b` | conf | `arrowhead_ids` | leader "CB #4" after `exclude_dimensions` |
|---|---|---|---|---|
| 0.00 | [101.53, 0] | 0.951 | e4, cluster:e10 | survives ✓ |
| 0.05 | [101.58, 0] | 0.951 | e4, cluster:e10 | survives ✓ |
| **0.10** | **[108.0, 0.5]** | 0.944 | **e4, e26** | **DELETED** |
| **0.50** | **[108.0, 0.5]** | 0.944 | **e4, e26** | **DELETED** |
| 1.00 / 2.00 | [108.0, 0.5] | 0.944 | e4, e26 | "survives" — but as a WRONG reading: the leader is re-proposed with `arrowhead=cluster:e10`, tip `[100.8, 0.5]` (the cluster misattributed to the leader's shaft end); the e26 reading is excluded |

Tip `1f6551c`: cluster wins and the leader survives at every shift (RUN). Round 4: chevron wins at every shift (RUN). The repair moves the boundary from "never" to "< 0.05-0.4 pt"; the mechanism that deletes a real leader through a cluster-terminated dimension end is otherwise unchanged. This is the line the previous pass called the highest-risk in round 4, and it remains open for the corpus's own terminator anatomy. Corpus-inert (RUN: called set identical) because no corpus cluster end has an outward-pointing foreign arrow within 4.6 pt — the same "the corpus did not move" non-evidence the last three rounds recorded.

**What would close it** (design decision, not mine to make): a cluster seat that measures the splash's reach on the ray rather than its centroid (the code already publishes `_reach_on_ray` for clusters, so "seated" would mean "the end lies inside the splash's axial extent" — 0.0 for every real cluster), or a dominance rule stated in arrowhead-scale rather than ratio. Either needs fixtures both ways (round 4's splash-5-pt-off case must stay capped) and a corpus re-run. If the lead instead accepts the residual as policy, the README's dominance sentence and `_SEAT_DOMINANCE`'s docstring must state the window, and a fixture must pin it — `test_end_ownership.py` currently pins only the 0.04 pt case.

### D2 — COST FLAG (> 10 %). The repaired tree is 34 % slower on the observational band and 15-16 % slower at the default, driven by sheet 3001
`queries.py:2298-2322` (per-candidate `_seat_distance` + `_radius_of` + the weaker exact prune), plus the 60 extra capped constructs that now reach the witness hunt.

Best-of-3 wall clock, seconds (RUN `timing_v.py`, order work → r4 → tip → work; both work runs shown):

| sheet | dims@0.0 work | r4 | Δ | dims@0.5 work | r4 | Δ | leaders@0.5 work | r4 | Δ | tip (0.0 / 0.5 / L) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5003 | 0.007 / 0.007 | 0.007 | 0 % | 0.007 | 0.007 | 0 % | 0.025 | 0.025 | 0 % | 0.006 / 0.006 / 0.023 |
| 10.17a | 0.057 / 0.057 | 0.055 | +4 % | 0.013 | 0.012 | +8 % | 0.038 | 0.038 | 0 % | 0.010 / 0.010 / 0.035 |
| 11.01 | 0.122 / 0.117 | 0.109 | +7-12 % | 0.039 / 0.037 | 0.035 | +6-11 % | 0.127 / 0.120 | 0.112 | +7-13 % | 0.030 / 0.030 / 0.106 |
| 21.01 | 0.835 / 0.911 | 0.805 | +4-13 % | 0.363 / 0.344 | 0.333 | +3-9 % | 0.877 / 0.861 | 0.852 | +1-3 % | 0.081 / 0.090 / 0.681 |
| **3001** | **4.296 / 4.248** | **2.940** | **+45 %** | **1.290 / 1.239** | **1.022** | **+21-26 %** | **1.924 / 1.898** | **1.476** | **+29-30 %** | 0.389 / 0.357 / 0.953 |
| 10.31A | 0.311 / 0.221 | 0.221 | 0-41 % | 0.079 / 0.064 | 0.060 | +7-32 % | 0.319 / 0.247 | 0.224 | +10-42 % | 0.058 / 0.073 / 0.341 |
| **TOTAL** | **5.698 / 5.631** | **4.204** | **+34-36 %** | **1.846 / 1.759** | **1.524** | **+15-21 %** | **3.560 / 3.447** | **2.982** | **+16-19 %** | 0.623 / 0.615 / 2.375 |

Nothing near the 600 s cliff (max 4.3 s), and the tip column shows the round-1-4 cap-not-delete policy is the far larger cost (the tip deleted what the band now carries). But the repair's own marginal cost exceeds the ~10 % bar on every total and is +45 % on the dense sheet, and the builder never measured it (`timing_work.txt` is empty; the builder's `timing_afterA.txt` is Change A only, 4.49 s total, which the shipped tree has since exceeded by 25 %). Where it goes: the exact prune only bites once a tier-0 incumbent exists (`cur0 is not None`), and on junk-dense sheets many ends have no tier-0 candidate, so every candidate in the attach radius now pays `_geom_of` + `_arrow_attach` + `_seat_distance`; and the 23 extra 3001 constructs each pay the witness search.

### D3 — POLICY CONSEQUENCE (medium; corpus-inert; partly documented). The "oriented" rank admits sub-2-pt SHX glyph fragments and 2:1 rectangles as callable terminators
`queries.py:1329-1331` (`oriented = deg >= 0.0 and align >= _ARROW_AXIS_MIN`, i.e. PCA elongation ≥ 1.6 of the VERTICES and fold ≥ cos 30°), `:951-961` (`_cluster_alignment`'s 1.6 gate, reused), `:699-712` (the open-4-vertex near-ring candidate path that delivers glyph strokes).

All six corpus proposals that carry an oriented end are 0.60×0.72 to 1.08×0.84 pt 4-vertex glyph fragments (RUN, §4) — the real population of the new class on this corpus is 6/6 junk and 0/6 terminators. They are held by other rungs today. Three fixtures show what the rank does when those rungs are absent (RUN, `attack.py` P5/P6; all on-spine, attached, witnessed):

| scene | repaired | pre-repair round 4 | tip `1f6551c` |
|---|---|---|---|
| NO-TEXT sheet, true chevron at 0, a 2.0×0.8 pt closed oblong the line runs into at 100, witnesses both ends | **CALLED 1.0**, b=[100.6, 0], `arrowhead_ids` names the oblong | 0.45 (blunt) | called 0.98, b=[101.6, -0.4] (elected a corner) |
| same, open 4-corner "re" ring / corpus-like 1.98×0.96 ring | **CALLED 1.0** | 0.45 | 0.98 / 0.971 |
| same fragment ACROSS the line, or a 1×1 tile | 0.45 (blunt) ✓ | 0.45 | no proposal |
| split leg, no text: real arrows-outside half + a half whose "arrow" is a 2×0.8 oblong beyond its tip | **CALLED 1.0** (split_shaft, `oriented_terminator_ids`) | 0.45 | no proposal |
| TEXT sheet, graphic scale bar: 100 pt baseline, filled 6×3 end blocks inside the line, ticks every 25, labels 0/50/100 | **CALLED 0.944** as a dimension, text "50" | 0.45 (blunt), b=[97, 0] | no proposal |
| same with 3×3 (1:1) blocks | 0.45 (blunt) ✓ | 0.45 | no proposal |
| same with 20×3 blocks | no proposal (bbox > size cap) | no proposal | no proposal |

So versus the pre-repair tree the oriented rank re-opens, on the corpus's own no-text sheet class, a [real arrow + oblong glyph fragment] call at 1.0 that rounds 1-4 had capped, and it newly calls a 2:1-block scale bar as a dimension at 0.944 (a scale bar is on most civil sheets; the tip returned nothing for it). Versus the tip it is not a regression for the fragment case (the tip called it too, with a worse coordinate). The builder documented the class as "unmeasured on real drafting above the call threshold" and pinned the intended direction (`TestWhatTheOrientedRankNewlyAdmits`); it did not pin the scale bar or the no-text fragment. Gate boundaries behave as coded (RUN P1): elongation 1.59 blunt / 1.60 oriented; 29.9° oriented / 30.0° blunt (the diamond at exactly 30° falls to blunt in floating point — boundary is inclusive by intent, exclusive in practice; harmless). Diamond width: `leg/w ≥ 1.6` oriented, so a 1:1 rhombus (a square rotated 45°) is blunt — the only standard AutoCAD tipless terminators (Dot, Box) are isotropic and stay capped; the class the rank was built for (diamond, flat tip) is rare in practice while the class it actually admits (oblong glyph strokes, hatch tiles, scale-bar blocks) is common.

### D4 — LOW. An exact seat tie is entity-order dependent
`queries.py:2340-2341` (`if cur is not None and seat >= cur["seat"]: continue` — first-wins).

Two tier-0 chevrons both base-seated at the same shaft end (seat 0.0, 0.0), one on-axis, one 20° up (RUN, `attack.py` P4): repaired tree publishes `[107.20, 0]` 0.944 with straight-first entity order and `[106.77, 2.46]` 0.941 with skew-first — a 2.5 pt swing on entity order. Round 4 and the tip publish the same answer in both orders (centroid distance breaks the tie). Deterministic, but the docstring's "the outcome does not depend on the order candidates arrive in" (`:2288-2291`) is false at a tie. Rare geometry (two arrows sharing a base point within 30° of each other); a tie-break on `d` would restore the old behaviour.

### D5 — LOW (documentation / accounting)
- `_SEAT_DOMINANCE` docstring (`:1462-1481`) and README ("a stipple-rendered arrowhead centred on its own end is not lost to a foreign chevron that merely sits near it") state the fix without its window (D1).
- "+60 attributed to removing the pre-prune" — wrong; see row 12. The correct statement: the seat ordering lets the two ends of short junk shafts elect different contradicted candidates, so pairs the DISTINCT rule used to drop now publish at ≤ 0.25 (60 removed / 120 added, all ≤ 0.45; 112 of the 120 are contradicted).
- "6 rows changed at 0.3" — 8 (row 10).
- README's "Six corpus proposals carry an oriented end … five on 11.01, one on 21.01" is true (RUN) but is a published figure with no command: `doc_claims_check.py` does not compute it, and `test_readme_claims.py` does not pin it. Same class as the last pass's finding 7.
- README caption vs code for the blunt figure: **consistent.** The prose now defines the figure as proposals carrying `blunt_terminators` evidence under the 2026-09-10 split (isotropic only), says it was re-measured, and names the construct that moved. The number is re-measured under a CHANGED and stated definition, not under an unchanged caption.
- `test_readme_claims.py` constants 18 / 16 / (17,15,0) / (1,1,0) match `doc_claims_check.py` (RUN).

### D6 — PROCESS. The tree moved during verification
`queries.py` and `test_end_ownership.py` were rewritten at 20:32 while this pass was reading them; the briefing's "the tier-0 centroid PRE-PRUNE was REMOVED" describes the 20:21 state, not the shipping one. I re-derived from the 20:32 file and confirmed the md5 after the last run. Any further edit invalidates §1.

---

## 3. What verified SOUND (RUN unless noted)

- **Findings 1, 3, 4** — fixed on the previous pass's fixtures: diamond/trapezoid `[100, 0]` 100.0 0.952; junk at (108, 0/1/5) and (103, 2..5) all `[100, 0]` 0.952. `round4_repro.py` output byte-identical to the builder's.
- **Finding 5** still open, **finding 6** still a deletion — both now documented in the README sharp-edges block, as the previous pass asked.
- **Exact seat prune** (`:2318-2322`). READ: every seat point (apex, base = centroid of a vertex subset, vertex extremes, centroid) lies in the convex hull, which lies in the ball of radius R about the vertex centroid, so `seat ≥ d - R` by the triangle inequality; a candidate with `d - R ≥ s0` can neither beat the tier-0 incumbent (strict `<`), dominate it from tier 1 (needs `< s0/10`), nor win from tier 2; the incumbent only improves, so early pruning is monotone-safe; only the winner is consumed downstream. RUN: on/off identical on all ten sheets @0.0 (22/15/0/5/85/12/144/219/154/137).
- **`_seat_distance` invariance**: chevron `[b,apex,b]`, base-leg `[b,b,apex]`, closed triangle in all three vertex rotations — seat 0.000000 with the apex at the end AND with the base at the end. Duplicate closing vertex and reversed traversal of a diamond: state, score, apex and seat unchanged.
- **`_award_end`**: tier 0 vs tier 2 → 0; tier 1 vs 2 → 1; tie `seat*10 == s0` → sound (checked at 0.1/1.0, 0.3/3.0, 0.07/0.7 — the `0.7000000000000001` float case also resolves to sound); empty → None. Non-tie award is order-independent (builder's test and P2 rows agree under swapped entity order).
- **Corpus invariance at the default**: called set @0.5 byte-identical to round 4 (322 rows); leader set identical at every band; residuals identical to the tip.
- **No-text "pointing shape" rule** (`:2547-2551`, `:2721-2725`): reads `kind == "triangle" and state == "ok"`, so a cluster or an oriented shape never satisfies it on its own; two oriented ends on a no-text sheet stay capped (builder's test, and P5's across-the-line row). Its consequence when the OTHER end is a real arrow is D3.
- **Hoisted ceiling**: for every sheet and both thresholds, `find_dimensions(min_confidence=c)` equals the @0.0 result filtered at `c` (RUN, `vcheck.py` cross-check — no mismatch on any sheet).
- **Docs vs `doc_claims_check.py`**: all published corpus figures agree (row 21).

---

## 4. Per-proposal accounting

### The six proposals that carry an oriented end (@0.0)

| sheet | shaft | conf | held by | oriented candidate | geometry (RUN, from the IR) | truth |
|---|---|---|---|---|---|---|
| 11.01 | e199 | 0.45 | detached + off-spine | e791 | 4-vertex open ring, bbox 0.96×1.98 pt, elong 1.73 | no native annotation on 11.01 |
| 11.01 | e203 | 0.25 | contradicted (other end) + off-spine + extent floor | e790 | bbox 1.08×0.84, elong 1.84 | — |
| 11.01 | e488 | 0.25 | contradicted + off-spine | e778 | bbox 1.02×0.66, elong 1.97 | — |
| 11.01 | e543 | 0.25 | contradicted + off-spine | e778 | same fragment | — |
| 11.01 | e551 | 0.25 | contradicted + off-spine | e778 | same fragment | — |
| 21.01 | e329 | 0.45 | detached + off-spine + extent floor | e7447 | bbox 0.60×0.72, elong 3.51 | ends 13.4 / 14.7 pt from the nearest native leader tip, 223-231 pt from any defpoint — unmatched |

All six: **junk**, not true positives merely unmatched. None is a terminator; all are ≤ 2 pt glyph fragments the open-4-vertex candidate path admits.

### The eight @0.3 rows that differ from round 4 (all 0.45, all `below_extent_floor` or detached/off-spine)

| sheet | shaft | old ids → new ids | coordinate change | candidates (RUN) | nearest truth |
|---|---|---|---|---|---|
| 11.01 | e3058 | e3054, e3056 → e3054, **e3753** | none | 4.5×2.6 pt right-triangle pattern glyphs | no truth on sheet |
| 11.01 | e199 | [] → **e791** | none | re-label (D3 population) | no truth |
| 21.01 | e4727 | e4728, e4742 → e4728, **e4737** | none | 5.0×2.9 / 5.0×3.1 pattern triangles | 46-49 pt from a defpoint, 62-66 from a tip |
| 21.01 | e4678 | e4677, e4697 → e4677, **e4680** | end b [274.4, 126.5] → [282.7, 124.8] (8.5 pt), len 13.5 → 5.0 | pattern triangles | 34-38 pt / 31-35 pt |
| 21.01 | e4676 | e4674, e4677 → e4677, **e4694** | none | pattern triangles | 30-34 / 27-32 pt |
| 21.01 | e329 | cluster:e149 → cluster:e149, **e7447** | none | re-label (D3 population) | 14-15 pt from a tip (unmatched at 18? no — the END is; the construct was already unmatched) |
| 3001 | e3181 | e3174, e3178 → e3174, **e3195** | none | 5.8×3.3 / 5.8×3.5 pattern triangles | 204-207 / 73-84 pt |
| 3001 | e2857 | e2855, e2858 → e2858, **e2877** | end b [346.7, 136.3] → [337.2, 139.0] (9.9 pt), len 5.8 → 15.5 | pattern triangles | 151-159 / 23-39 pt |

Verdict per row: junk, correctly capped, correctly unmatched; two coordinate moves of 8.5 and 9.9 pt on 0.45 junk. Note the 21.01 e329 end sits 13-15 pt from a native leader tip — inside the scorer's 18 pt tolerance, so a greedy match could in principle ride it; it did not (leader residuals identical).

---

## 5. Could not verify, and why

- **Whether D1 or D3 bites on any real non-Mecklenburg drawing.** No such corpus exists in either repo; the six-line fixtures are the only falsification available. The corpus cannot see either (D1 needs an outward foreign arrow within 4.6 pt of a cluster end; D3 needs an oriented fragment that is attached AND on-spine AND unopposed by a tier-0 candidate — 0/6 today).
- **The builder's "corpus cannot move on it (verified)"** — I verified the called set is identical; I did not enumerate every cluster-terminated end's nearest tier-0 candidate to state the corpus margin against D1. The docstring's own 0.5-2 pt figure is the builder's, not mine.
- **The app's full gate** — not run (instructed).
- **The render-adjudication of survivors, the DXF truth regeneration, the 9.59° arrows-outside sharp edge** — unchanged by this repair; not re-verified.
- **Tree state after 20:42** — md5 unchanged through my last run; anything edited after this file is written is unverified.

---

## Appendix A — measurement scripts (all under `C:/Users/socon/.claude/jobs/69be0e95/tmp`)

- `vcheck.py <work|v_r4|v_tip>` → `vcheck_<tree>.txt`: recall/residuals @0.3 and @0.5, FPs, precision, blunt/oriented counts, band counts, full called set @0.5, full dim sets @0.3 and @0.0, full leader set @0.0, plus the hoisted-ceiling cross-check. Diff two outputs for rows 7-13.
- `prune_acct.py`: exact-prune on/off equality on all ten sheets; the six-plus-eight accounting against the transformed native truth (§4).
- `timing_v.py <tree>`: best-of-3 per sheet (§3 / D2). Run order was work, v_r4, v_tip, work.
- `attack.py <tree>` → `attack_<tree>.txt`: the fixtures of D1, D3, D4 and the sound checks (source below).
- `round4_verifier_work.txt`, `doc_claims_verifier.txt`, `suite_planlens.txt`, `suite_app58.txt`: raw outputs.
- `v_tip/`, `v_r4/`: the scratch baselines (working-tree package + swapped `queries.py`); `v_r4i/`, `v_worki/`: the DISTINCT-skip instrumented copies for row 12.

## Appendix B — `attack.py` (runnable: `<venv>/python attack.py work|v_r4|v_tip`)

```python
"""Verifier's adversarial fixtures for the round-5 repair. Usage: attack.py <tree>  (work | v_r4 | v_tip)"""
import math, os, sys
TREE = sys.argv[1]; TMP = r"C:\Users\socon\.claude\jobs\69be0e95\tmp"
if TREE != "work": sys.path.insert(0, os.path.join(TMP, TREE))
import planlens
from planlens.ir import queries as q
from planlens.ir.results import DrawingIR, Line, Polyline, TextItem
print("[%s] %s" % (TREE, planlens.__file__))
MAS = 9.31
H = math.sqrt(7.3 ** 2 - 1.2 ** 2)   # 7.2007 apex-to-base of the standard arrow

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
def tri_closed(apex, d, leg=7.3, base=2.4, rot=0):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    v = [apex, (bx+px*base/2, by+py*base/2), (bx-px*base/2, by-py*base/2)]
    return Polyline(vertices=v[rot:]+v[:rot], closed=True)
def rect(cx, cy, w, h, ang=0.0, open_ring=False):
    """w along `ang`, h across. open_ring: drop one SHORT edge (a PDF 're' ingests as an open 4-corner chain)."""
    c, s = math.cos(ang), math.sin(ang)
    loc = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    v = [(cx + x*c - y*s, cy + x*s + y*c) for x, y in loc]
    if open_ring:
        v = [v[1], v[0], v[3], v[2]]
        return Polyline(vertices=v, closed=False)
    return Polyline(vertices=v, closed=True)
def diamond(apex, d, leg=6.0, w=2.4):
    dx, dy = d; px, py = -dy, dx
    return Polyline(vertices=[apex, (apex[0]-dx*leg/2+px*w/2, apex[1]-dy*leg/2+py*w/2), (apex[0]-dx*leg, apex[1]-dy*leg), (apex[0]-dx*leg/2-px*w/2, apex[1]-dy*leg/2-py*w/2)], closed=True)
def fill_cluster(tip, d, n=7, span=3.0):
    out = []; px, py = -d[1], d[0]
    for i in range(n):
        t = (i/(n-1.0))*span - 0.5*span
        for k in (-0.6, 0.0, 0.6):
            x = tip[0]+d[0]*t+px*k; y = tip[1]+d[1]*t+py*k
            out.append(Line(start=(x, y), end=(x+0.06, y+0.06)))
    return out
def core(text=True):
    e = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0))]
    if text: e.append(TextItem(content="100'", position=(50.0, 6.0), height=6.0))
    return e
def dims(ents, mc=0.0, shaft="e0"):
    ps = q.find_dimensions(ir_of(ents), max_arrowhead_size=MAS, min_confidence=mc)
    return [p for p in ps if p["shaft_id"] == shaft or p["evidence"]["path"] == "split_shaft"]
def leaders(ents, mc=0.5):
    return q.find_leaders(ir_of(ents), max_arrowhead_size=MAS, min_confidence=mc, exclude_dimensions=True)
def show(tag, ents, mc=0.0, shaft="e0"):
    ps = dims(ents, mc, shaft)
    if not ps: print("  %-58s NO PROPOSAL" % tag); return None
    p = ps[0]; ev = p["evidence"]
    flags = ",".join(sorted(k for k in ev if k.startswith("arrow_") or "blunt" in k or "oriented" in k or k == "below_extent_floor"))
    print("  %-58s b=%-16s len=%-8s conf=%-6s ids=%s [%s]" % (tag, p["end_b_xy"], p["length"], p["confidence"], p["arrowhead_ids"], flags))
    return p
class _NA:
    state = "n/a"; score = float("nan"); apex = (float("nan"), float("nan"))
def attach(shape, sdir=(1.0, 0.0), tip=(100.0, 0.0)):
    if not hasattr(q, "_arrow_attach"):
        return _NA()
    return q._arrow_attach([tuple(v) for v in shape.vertices], "triangle", sdir, tip, MAS)

A = chevron((0.0, 0.0), (-1.0, 0.0))
B = chevron((100.0, 0.0), (1.0, 0.0))

print("\nP1  ORIENTED/BLUNT gate: closed rectangle terminators (w along line x h across) at both ends, drawn INSIDE the line (far edge at the defpoint); truth b=[100,0]")
for w, h in ((3.0, 3.0), (3.9, 3.0), (4.5, 3.0), (4.77, 3.0), (4.8, 3.0), (4.83, 3.0), (6.0, 3.0), (6.0, 1.0), (6.0, 0.75)):
    ents = core() + [rect(w/2, 0.0, w, h), rect(100.0 - w/2, 0.0, w, h)]
    st = attach(rect(100.0 - w/2, 0.0, w, h)).state
    show("rect %.2fx%.2f (elong %.2f) state=%s" % (w, h, w/h, st), ents)
print("  -- same 6x3 rect rotated about its centre (fold cone: cos30 = %.4f)" % math.cos(math.radians(30)))
for deg in (25.0, 29.0, 29.9, 30.0, 30.1, 31.0, 35.0, 45.0):
    r = rect(97.0, 0.0, 6.0, 3.0, math.radians(deg))
    print("     %5.1f deg -> state=%s score=%.4f" % (deg, attach(r).state, attach(r).score))
print("  -- diamond width sweep (leg 6): elong = leg/w")
for w in (2.4, 3.0, 3.6, 3.75, 3.8, 4.0, 6.0):
    d = diamond((100.0, 0.0), (1.0, 0.0), leg=6.0, w=w)
    a = attach(d); print("     w=%.2f elong=%.2f -> state=%s apex=%s" % (w, 6.0/w, a.state, tuple(round(x, 3) for x in a.apex)))

print("\nP2  SEAT DOMINANCE: cluster (tier 1) genuinely terminates the line, foreign chevron (tier 0) beyond the end on-axis; cluster seat s1 = splash-centre offset from the end; chevron seat s0 = base-centre gap. truth b~[100,0]; foreign apex = 100+s0+H")
for s1 in (0.04, 0.1, 0.3, 0.5, 1.0, 2.0):
    for s0 in (0.94, 2.7, 3.0, 3.3, 5.0, 9.0, 11.0):
        ents = core() + [A] + fill_cluster((100.0 + s1, 0.0), (1.0, 0.0)) + [chevron((100.0 + s0 + H, 0.0), (1.0, 0.0))]
        p = dims(ents, 0.0)[0]
        who = "CLUSTER" if any(i.startswith("cluster") for i in p["arrowhead_ids"]) else "chevron"
        print("  s1=%.2f s0=%5.2f ratio=%5.1f -> %-7s b=%-16s conf=%-6s ids=%s" % (s1, s0, s0/max(s1, 1e-9), who, p["end_b_xy"], p["confidence"], p["arrowhead_ids"]))
print("  -- finding-2 anatomy exactly (foreign apex (108,0.5), seat 0.94) with the cluster shifted by s1 along the line, PLUS the leader that owns the chevron:")
for s1 in (0.0, 0.05, 0.1, 0.5, 1.0, 2.0):
    ents = core() + [A] + fill_cluster((100.0 + s1, 0.0), (1.0, 0.0)) + [chevron((108.0, 0.5), (1.0, 0.0)), Line(start=(108.0 - H, 0.5), end=(60.0, -10.0)), TextItem(content="CB #4", position=(54.0, -12.0), height=3.0)]
    p = dims(ents, 0.0)[0]; L = leaders(ents, 0.5)
    print("  shift=%.2f -> b=%-16s conf=%-6s ids=%-28s leader 'CB #4' survives exclude_dimensions: %s" % (s1, p["end_b_xy"], p["confidence"], p["arrowhead_ids"], any(l.get("text") == "CB #4" for l in L)))

print("\nP3  SEAT for a pointed terminator: vertex order / flavour invariance (apex at end, and base at end)")
for name, mk in (("chevron [b,apex,b]", chevron), ("base-leg [b,b,apex]", base_leg), ("closed rot0", lambda a, d: tri_closed(a, d, rot=0)), ("closed rot1", lambda a, d: tri_closed(a, d, rot=1)), ("closed rot2", lambda a, d: tri_closed(a, d, rot=2))):
    for tag, apex in (("apex@end", (100.0, 0.0)), ("base@end", (100.0 + H, 0.0))):
        v = [tuple(x) for x in mk(apex, (1.0, 0.0)).vertices]
        if hasattr(q, "_seat_distance"):
            s = q._seat_distance(v, "triangle", (1.0, 0.0), (100.0, 0.0), q._arrow_geometry(v))
            print("  %-22s %-9s seat=%.6f" % (name, tag, s))
        else:
            print("  %-22s %-9s (no _seat_distance on this tree)" % (name, tag))

print("\nP4  TIE at one end: two tier-0 chevrons both base-seated at the end (seat 0, 0), one on-axis, one 20 deg up; entity ORDER swapped")
for order in ("straight-first", "skew-first"):
    a20 = math.radians(20.0)
    c1 = chevron((100.0 + H, 0.0), (1.0, 0.0)); c2 = chevron((100.0 + H*math.cos(a20), H*math.sin(a20)), (math.cos(a20), math.sin(a20)))
    ents = core() + [A] + ([c1, c2] if order == "straight-first" else [c2, c1])
    show("order=%s" % order, ents)

print("\nP5  NO-TEXT sheet (the corpus's own class): true chevron at one end, an ORIENTED oblong glyph fragment at the other, witnesses both ends")
for tag, frag in (("closed 2.0x0.8 oblong on the end", rect(100.6, 0.0, 2.0, 0.8)), ("open near-ring 2.0x0.8 (PDF re)", rect(100.6, 0.0, 2.0, 0.8, open_ring=True)), ("corpus-like 1.98x0.96 open ring", rect(100.5, 0.0, 1.98, 0.96, open_ring=True)), ("same fragment ACROSS the line (90 deg)", rect(100.4, 0.0, 2.0, 0.8, math.radians(90))), ("square 1.0x1.0 tile", rect(100.5, 0.0, 1.0, 1.0))):
    ents = core(text=False) + [A, frag]
    p = show(tag, ents)
    if p is not None:
        print("      -> CALLED at 0.5: %s" % (p["confidence"] >= 0.5))
print("  -- same on a TEXT-bearing sheet")
show("text sheet: closed 2.0x0.8 oblong on the end", core() + [A, rect(100.6, 0.0, 2.0, 0.8)])
print("  -- split leg on a no-text sheet: real half (arrow outside, base at tip) + a half whose 'arrow' is an oblong fragment beyond its tip")
ents = [Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0)),
        Line(start=(H, 0.0), end=(40.0, 0.0)), base_leg((0.0, 0.0), (-1.0, 0.0)),
        Line(start=(60.0, 0.0), end=(98.0, 0.0)), rect(99.0, 0.0, 2.0, 0.8)]
ps = [p for p in q.find_dimensions(ir_of(ents), max_arrowhead_size=MAS, min_confidence=0.0) if p["evidence"]["path"] == "split_shaft"]
for p in ps: print("  split: a=%s b=%s conf=%s ids=%s kinds=%s flags=%s" % (p["end_a_xy"], p["end_b_xy"], p["confidence"], p["arrowhead_ids"], p["evidence"]["arrowhead_kinds"], sorted(k for k in p["evidence"] if k.startswith("arrow_") or "blunt" in k or "oriented" in k)))
if not ps: print("  split: NO PROPOSAL")

print("\nP6  GRAPHIC SCALE BAR (text sheet): baseline 0-100 with filled end blocks INSIDE the line, ticks every 25, labels")
ticks = [Line(start=(x, -4.0), end=(x, 4.0)) for x in (0.0, 25.0, 50.0, 75.0, 100.0)]
labels = [TextItem(content="0", position=(0.0, 7.0), height=3.0), TextItem(content="50", position=(50.0, 7.0), height=3.0), TextItem(content="100", position=(100.0, 7.0), height=3.0)]
ents = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), rect(10.0, 0.0, 20.0, 3.0), rect(90.0, 0.0, 20.0, 3.0)] + ticks + labels
show("scale bar (20x3 blocks, 6.7:1)", ents)
ents2 = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), rect(3.0, 0.0, 6.0, 3.0), rect(97.0, 0.0, 6.0, 3.0)] + ticks + labels
show("scale bar (6x3 blocks, 2:1)", ents2)
ents3 = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), rect(1.5, 0.0, 3.0, 3.0), rect(98.5, 0.0, 3.0, 3.0)] + ticks + labels
show("scale bar (3x3 blocks, 1:1)", ents3)

print("\nP7  EXACT threshold: tier-1 seat*10 == tier-0 seat (float)")
if hasattr(q, "_award_end"):
    for s1, s0 in ((0.1, 1.0), (0.3, 3.0), (0.07, 0.7), (1e-9, 1e-8)):
        r = q._award_end({0: {"tier": 0, "seat": s0}, 1: {"tier": 1, "seat": s1}})
        print("  s1=%g s0=%g -> tier %d wins  (s1*10=%r)" % (s1, s0, r["tier"], s1 * 10))
else: print("  (no _award_end on this tree)")

print("\nP8  Tipless attach/seat with a duplicate closing vertex and a reversed traversal")
d0 = diamond((100.0, 0.0), (1.0, 0.0)); v = [tuple(x) for x in d0.vertices]
for tag, vv in (("plain", v), ("dup-closing", v + [v[0]]), ("reversed", v[::-1])):
    if not hasattr(q, "_arrow_attach"):
        print("  (no _arrow_attach on this tree)"); break
    a = q._arrow_attach(vv, "triangle", (1.0, 0.0), (100.0, 0.0), MAS)
    extra = ""
    if hasattr(q, "_seat_distance"):
        extra = " seat=%.4f" % q._seat_distance(vv, "triangle", (1.0, 0.0), (100.0, 0.0), q._arrow_geometry(vv))
    print("  %-12s state=%s score=%.4f apex=%s%s" % (tag, a.state, a.score, tuple(round(x, 3) for x in a.apex), extra))
```

## Appendix C — the D1 fixture as a pytest, ready to drop beside `test_end_ownership.py`

```python
import math
import pytest
from planlens.ir import queries as q
from planlens.ir.results import Line, TextItem
from planlens.ir.tests.test_arrow_direction import MAS, _arrow_chevron, _ir
from planlens.ir.tests.test_end_ownership import core, fill_cluster, A, _H

@pytest.mark.parametrize("shift", [0.1, 0.5, 1.0, 2.0])   # the "0.5-2 pt" real-cluster offsets the constant's docstring names
def test_a_realistically_seated_cluster_still_loses_its_end_to_a_foreign_chevron(shift):
    ents = core() + [A] + fill_cluster((100.0 + shift, 0.0), (1.0, 0.0)) + [
        _arrow_chevron((108.0, 0.5), (1.0, 0.0)),
        Line(start=(108.0 - _H, 0.5), end=(60.0, -10.0)),
        TextItem(content="CB #4", position=(54.0, -12.0), height=3.0)]
    ir = _ir(ents)
    p = [p for p in q.find_dimensions(ir, max_arrowhead_size=MAS, min_confidence=0.5)
         if p["shaft_id"] == "e0"][0]
    # EXPECTED (tip 1f6551c behaviour): the cluster keeps the end, the chevron is not named.
    assert any(i.startswith("cluster:") for i in p["arrowhead_ids"]), p["arrowhead_ids"]   # FAILS on the repaired tree: ['e4', 'e26']
    assert abs(p["end_b_xy"][0] - 100.0) < 4.0                                            # FAILS: 108.0
    leads = q.find_leaders(ir, max_arrowhead_size=MAS, exclude_dimensions=True, min_confidence=0.5)
    real = [l for l in leads if l.get("text") == "CB #4" and not str(l.get("arrowhead_id")).startswith("cluster:")]
    assert real, "the leader that owns the chevron was arbitrated away"                    # FAILS at shift 0.1 and 0.5; at 1.0/2.0 the leader is re-read with the cluster as its arrowhead
```
