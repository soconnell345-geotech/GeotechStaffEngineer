# planlens round-4 remediation — independent adversarial verification

**Verifier** independent (did not build any of this) · **Date** 2026-09-09
**Under review** the uncommitted working tree of `C:/Users/socon/OneDrive/dev/planlens`
(branch `main`, tip `1f6551c`), proposed as **0.2.0** for PyPI
**Comparison trees** committed tip `1f6551c` ("tip") and published PyPI **v0.1.0**,
both extracted read-only with `git archive` into
`C:/Users/socon/.claude/jobs/69be0e95/tmp/{base,v010}`
**Python** `C:/Users/socon/OneDrive/dev/GeotechStaffEngineer/.venv/Scripts/python.exe`
**Repro fixtures** `C:/Users/socon/.claude/jobs/69be0e95/tmp/round4_repro.py` (see Appendix A)
**Both repos were left untouched.** All scratch under the job tmp directory.

---

## VERDICT: DO NOT SHIP as 0.2.0 without addressing findings 1 and 2

Every measured number the ledger publishes **reproduces** — several of them I
confirmed against a fresh baseline run of the reviewed tip, which is stronger
evidence than the ledger itself offers. One documentation figure does not
reproduce (finding 7), and no README/DESIGN figure disagrees with
`doc_claims_check.py`.

But two round-4 behaviours are **regressions against both the published 0.1.0
and the committed tip**, each reproducible in a six-line fixture, both on
arrowhead styles the Mecklenburg corpus does not contain — which is exactly the
cross-cutting risk the original review named ("the gates are calibrated to one
agency's plotter"), now measurable rather than speculative.

The remediation is otherwise sound and, on the corpus, provably inert: not one
published coordinate on any matched corpus construct moved (see the residual row
in §3).

---

## 1. Three-tree comparison — the headline evidence

Same fixtures, three trees. Truth in every row: `end_b=[100.0, 0.0]`, length `100.0`.

| fixture | published **v0.1.0** | committed tip **`1f6551c`** | **round-4 tree** |
|---|---|---|---|
| slender triangle (control) | `[100.0,0.0]` 100.0 **0.952** | `[100.0,0.0]` 100.0 **0.952** | `[100.0,0.0]` 100.0 **0.952** |
| **diamond terminator** (finding 1) | `[100.0,0.0]` 100.0 **0.952** | `[100.0,0.0]` 100.0 **0.952** | `[97.0,0.0]` **94.0** **0.45** |
| **trapezoid / flat-tip** (finding 1) | `[100.0,0.0]` 100.0 **0.948** | *deleted* | `[97.0,0.0]` **94.0** **0.45** |
| **cluster + foreign chevron** (finding 2) | `[100.0,0.0]` **0.952** | `[101.53,-0.57]` **0.949** | `[108.0,0.5]` **0.944** |
| **on-spine axial outlier** (finding 3) | `[100.0,0.0]` **0.952** | `[108.0,0.0]` 0.942 | `[108.0,0.0]` 0.942 |
| **off-spine shadow** (finding 4) | `[100.0,0.0]` **0.952** | `[103.0,3.0]` 0.962 | `[100.0,0.0]` **0.45** |
| **60° triangle, vertex-rot 1** (finding 5) | `[100.0,0.0]` **0.952** | *deleted* | `[94.8,3.0]` **89.81** **0.25** |
| **63° triangle, any order** (finding 5) | `[100.0,0.0]` **0.952** | *deleted* | `[94.89,3.15]` **90.01** **0.25** |
| **concave dart** (finding 6) | *deleted* | *deleted* | *deleted* |

**One honest caveat so this table is not misread as "0.1.0 was better".** It was
not. 0.1.0 is immune to this whole failure class only because it published
*shaft endpoints*, not arrow apexes. That cruder contract scored **13/16**
dimension recall on the real corpus; the apex contract is what bought 16/16 —
and what created the failure class. Reverting is not the fix.

---

## 2. Findings

### Finding 1 — a box-like (diamond / trapezoid) terminator loses 6% of the measured length and drops below the call threshold
**Regression vs both v0.1.0 and the tip.** Introduced by the uncommitted work
(the blunt branch, rounds 1/3), and confirmed at the "evidence unobservable"
rank by round 4.
`planlens/ir/queries.py:1157` (`pointed`), `:1259-1276` (blunt branch), `:1362` (`_end_tier`)

A 6 pt diamond terminator at each end of a 100 pt dimension, witness lines and
value text present:

| | published `end_b` | length | confidence |
|---|---|---|---|
| v0.1.0 / tip | `[100.0, 0.0]` | **100.0** | **0.952** — called |
| round-4 tree | `[97.0, 0.0]` | **94.0** | **0.45** — never called |

`_arrow_geometry` computes `apex=(100.0, 0.0)` **correctly** and the code then
**discards it**: a centrally symmetric quad ties the apex vote (`pointed=False`),
and the blunt branch substitutes the *centroid* projected on the shaft ray
(`:1275`) — half a terminator behind the true defpoint, at each end. The cap at
`:1362` → `_UNCORROBORATED_CAP` is unconditional by design ("NO escape hatch",
`:2328-2340`), so no amount of corroboration recovers it.

This is F1's own demonstration case (the architectural box terminator that scored
0.859 on v0.1.0). The remediation converted *deleted* into *capped*, which is
progress, but the style is still **never called at the default threshold** and
now also carries a wrong coordinate. The docstring at `:1136-1147` names
"BOX-LIKE QUADS" as the family this branch is *for* — for a diamond the vote
gets the tip exactly right and the code throws it away anyway.

### Finding 2 — a fill-cluster arrowhead loses its end to a farther foreign arrowhead, and the arrowhead-steal re-opens
**Regression vs both v0.1.0 and the tip.** Introduced by round 4 specifically
(item 3, demoting fill clusters to tier 1).
`planlens/ir/queries.py:1362` (`_end_tier`), ranking at `:2111-2130`

Dimension `(0,0)-(100,0)` whose right terminator is a micro-dot fill cluster —
**the Mecklenburg anatomy** — plus one foreign chevron pointing outward 8 pt away:

| | winning candidate | published `end_b` | length | `arrowhead_ids` |
|---|---|---|---|---|
| v0.1.0 | cluster | `[100.0, 0.0]` | 100.0 | `['e4','cluster:e10']` |
| tip | `cluster:e10` (d = **0.04 pt**) | `[101.53, -0.57]` | 101.53 | `['e4','cluster:e10']` |
| round-4 | foreign chevron (d = **3.24 pt**) | `[108.0, 0.5]` | 108.0 | `['e4','e26']` |

Confidence stays 0.944 — **no cap, no evidence flag**. The cluster is nearer by
two orders of magnitude and loses purely on tier. Two consequences:

1. the published defpoint moves by up to ~9.6 pt (bounded by
   `0.75·max(scale, h)` plus the candidate's own `2h/3`), silently;
2. the foreign arrowhead is named in `arrowhead_ids`, so
   `find_leaders(exclude_dimensions=True)` will **delete** whatever leader owns
   it — the arbitration steal round 4's headline says is closed, reached by a
   route the round-4 fixture (`TestSkewedDimensionDoesNotStealANeighbouringLeader`)
   does not cover.

On a corpus whose real arrowheads *are* clusters, this is the highest-risk line
in round 4.

### Finding 3 — F2 is fixed only in its lateral half; the ledger's "F2's *verified* continuous-leg fix" overstates it
**Regression vs v0.1.0; pre-existing at the tip** (introduced by the already-committed
Phase-3.2 commit `2b4eee1`), **not closed by the remediation.**
`planlens/ir/queries.py:2227`, rationale at `:2211-2225`

The off-spine → publish-the-tip fallback fixes the original F2 report *only
because* its junk chevron sat 5 pt off axis, outside the arrow's own 1.2 pt
half-width. Move the same junk on-spine and F2 reproduces verbatim, identically
on the tip and the round-4 tree, and **correctly on v0.1.0**:

```
junk chevron apex=(108, 0)  -> end_b=[108.0, 0.0]  length 108.0    conf 0.942
junk chevron apex=(108, 1)  -> end_b=[108.0, 1.0]  length 108.0046 conf 0.945
junk chevron apex=(108, 5)  -> end_b=[100.0, 0.0]  length 100.0    conf 0.952   (the F2 report; fixed)
```

The code comment at `:2222` even names `(108, 0)` as the wrong answer, and that
is what the code publishes. Nothing caps or flags an on-spine axial outlier.
Per §1 this is an open regression against the version users have installed
today, which matters more for a 0.2.0 than for an internal round.

### Finding 4 — removing `on_spine` from the tier lets nearer off-spine junk cap a true dimension out of the called band
**vs v0.1.0: regression in callability** (0.952 → 0.45). **vs the tip: a mixed
trade** — the tip published a wrong coordinate at 0.962 (called), round 4
publishes the right coordinate at 0.45 (not called).
`planlens/ir/queries.py:1360-1366` vs the cap ladder at `:2264-2272`

`_end_tier` declares off-spine "not an identity signal"; the cap ladder still
treats it as a hard 0.45 ceiling. **The rank therefore cannot express the thing
that decides callability.** A junk chevron at `(103, 2..4)`, marginally nearer
than the true arrow (centroid 4.39 pt vs 4.80 pt), takes the end:

| | `end_b` | confidence |
|---|---|---|
| v0.1.0 | `[100.0, 0.0]` | 0.952 — called |
| tip | `[103.0, 3.0]` (3 pt error) | 0.962 — called |
| round-4 | `[100.0, 0.0]` (correct) | **0.45 — invisible at the 0.5 default** |

A deliberate trade the ledger does not mention. The comment at `:2101-2108` —
"a nearer candidate that the construct's own preconditions REJECT must not
shadow a farther one they accept" — is exactly the invariant this violates,
because off-spine is no longer one of the "preconditions" the tier can see.

### Finding 5 — equilateral / wide triangles: the apex axis is order-dependent at 60° and inverted above it
**Regression vs v0.1.0; pre-existing at the tip.** Round 4 restated an
invariance claim that measurement contradicts.
`planlens/ir/queries.py:1102-1159`

The round-4 docstring claims invariance "under vertex ORDER (a max and a
runner-up over a set)" and that for a triangle "the signed direction test
already judges it" — the stated reason 3-vertex shapes are exempt from the
blunt test. Round-4 tree:

```
base/leg 1.00 (60 deg tip), vertex-rot 0 -> b=[100.0, 0.0]  len 100.00  conf 0.952
base/leg 1.00,              vertex-rot 1 -> b=[94.80, 3.00] len  89.81  conf 0.25  VIOLATION
base/leg 1.00,              vertex-rot 2 -> b=[94.80,-3.00] len  89.81  conf 0.25  VIOLATION
base/leg 1.05 (63 deg tip), any order    -> b=[94.89, 3.15] len  90.01  conf 0.25  VIOLATION
base/leg 1.20 (74 deg tip), any order    -> b=[95.20, 3.60] len  90.69  conf 0.25  VIOLATION
```

v0.1.0 returns `[100.0, 0.0]` / 100.0 / 0.952 for every one of those rows. The
invariance argument holds only when the max is unique; at a tie it is false, and
above a 60° included tip angle a base corner is genuinely the farthest vertex,
so the inversion is deterministic. The tip *deleted* these; round 4 caps and
publishes them with a wrong coordinate — better discipline, still wrong number,
and still invisible at both 0.5 and the 0.3 observational band.

### Finding 6 — a concave dart arrowhead is deleted at every `min_confidence`
**Pre-existing in all three trees.**
`planlens/ir/queries.py:740` (the area / perimeter² ≥ 0.02 non-degeneracy gate)

A swallowtail/barbed arrowhead never becomes a candidate at all
(`_arrowhead_candidates_all` yields nothing), so no proposal exists at any
threshold. Not introduced here, but it is a silent deletion of a real
terminator family in a train whose stated discipline is cap-not-delete, and it
is undocumented. Worth a line in the sharp-edges section rather than a fix.

### Finding 7 — `"632 passed (from 316)"`: the baseline figure does not reproduce
**Documentation defect, new.**
`FINDINGS.md` Round-4 section, `HANDOFF.md` §0a-current, `CLAUDE.md`

The reviewed tip `1f6551c` collects **350** tests in this venv, not 316:

```
planlens/ir/tests   206
planlens/pdf/tests  114
planlens/dxf/tests   18
planlens/tests       12   (test_ocr.py)
                    ---
                    350
```

`632` is correct and reproduces (229 s, 0 failures). The growth claim is not,
and `doc_claims_check.py` does not cover it. On a project whose rule is that a
published number carries a command, this one carries none.

### Finding 8 — `doc_claims_check.py` models `ingest` rather than calling it
**New (the script is new).**
`module_work/drawing_ground_truth/doc_claims_check.py:60-88` vs `planlens/ir/ingest.py:371-381`

The script re-implements the layer-`"0"` inheritance walk instead of reading it
off the IR. Two divergences from the code it claims to measure:

* **nested INSERTs** — the script propagates the *outermost* INSERT's layer
  (`here = placed if placed is not None else lay`); `ingest._handle` resolves to
  the *nearest enclosing named* layer, because each INSERT has itself already
  been resolved before it recurses.
* **no entity budget** — `ingest` stops at `max_block_entities` (50 000 walked);
  the script walks unbounded.

I implemented both rules and ran them over all ten corpus DXFs: **identical**
(557 / 158 / 1696), because the corpus contains **0** nested INSERTs on named
layers. So the published number is right today — but the script is a model of
the code, not a measurement of it, and is one nested block away from diverging.

### Finding 9 — the README guards do not do what the ledger says they do
**New.** `planlens/tests/test_readme_claims.py`

The ten tests assert that `README.md` *contains* constants that are hardcoded
**in the test file** (`BLUNT_TOTAL_AT_ZERO = 19`, `REHOMED = {...}`), not that
`doc_claims_check.py` still produces them. The test docstring is honest about
this ("Deliberately narrow: this asserts the README quotes the live constants,
NOT that any particular measured count is still reproducible"). The ledger's
"ten guard tests pin the published figures against it [the script]" is not: if
the measured value drifts, the guards stay green. `test_layer_rehoming_counts_are_the_measured_ones`
in particular only checks that the substring `**557**` appears somewhere in the
README.

### Finding 10 — the staging trap is real, and confirmed
**New.**

`git ls-files planlens/testing` returns nothing; the whole package is untracked.
The **tracked** `planlens/ir/tests/leader_fixtures.py` and `construct_fixtures.py`
are now pure re-export shims over it, and the app imports `planlens.testing.*`
at module level:

* `funhouse_agent/tests/test_drawing_ir_adapter.py:348, :351, :377-378, :492`
* `funhouse_agent/tests/test_render_region.py:29`

Six paths must land in one commit or the source tree breaks:
`planlens/testing/{__init__,leader_fixtures,construct_fixtures}.py`,
`planlens/tests/{test_packaging,test_readme_claims}.py`,
`planlens/ir/tests/test_text_bearing_scenes.py`.

Packaging itself checks out: `include = ["planlens*"]` with
`exclude = ["*.tests", "*.tests.*"]` ships `planlens.testing` and excludes
`planlens.tests` / `planlens.ir.tests`, which is the point of the move.
`pyproject.toml` and `planlens/__init__.py` are both bumped to `0.2.0`.

---

## 3. Measurements — observed vs claimed

Every row measured by me, in this venv, on this tree.

| Claim | Claimed | Observed | |
|---|---|---|---|
| planlens suite | 632 passed | **632 passed**, 0 failed (229.25 s) | ✓ |
| …"was 316" | 316 | **350** collected at `1f6551c` | ✗ **finding 7** |
| app drawing tests | 58 passed | **58 passed** (7.0 s) | ✓ |
| corpus @0.3 leaders | 25/25 | **25/25** | ✓ |
| corpus @0.3 dimensions | 16/16 | **16/16** | ✓ |
| corpus @0.5 leaders | 23/25 | **23/25** | ✓ |
| corpus @0.5 dimensions | 16/16 | **16/16** | ✓ |
| leader FPs @0.5 — 2000a | 2 | **2** | ✓ |
| leader FPs @0.5 — 2000b | 10 | **10** | ✓ |
| leader FPs @0.5 — 3000 | 6 | **6** | ✓ |
| leader FPs @0.5 — 5003 | 19 | **19** | ✓ |
| leader FPs @0.5 — 10.17a | 23 | **23** | ✓ |
| leader FPs @0.5 — 10.25a | 20 | **20** | ✓ |
| leader FPs @0.5 — 11.01 | 38 | **38** | ✓ |
| …"all seven identical to baseline" | — | **identical** in my own `1f6551c` run | ✓ |
| dim precision @0.5 — 21.01 | → 5 | **5** (baseline 19) | ✓ |
| dim precision @0.5 — 3001 | → 10 | **10** (baseline 15) | ✓ |
| dim precision @0.5 — 11.01 | → 12 | **12** (baseline 18) | ✓ |
| dim precision @0.5 — 10.31A | → 10 | **10** (baseline 15) | ✓ |
| blunt terminators, all ten sheets | 19 @0.0, 17 @0.3, 0 @0.5 | **19 / 17 / 0** | ✓ |
| …per sheet 21.01 | 17 / 15 / 0 | **17 / 15 / 0** | ✓ |
| …per sheet 11.01 | 2 / 2 / 0 | **2 / 2 / 0** | ✓ |
| layer-0 inheritance re-homed | 557 / 158 / 1696 | **557 / 158 / 1696** | ✓ |
| README "geometry layers 3→2 on 10.17a and 5003, 11.01 stays 4" | — | reproduces | ✓ |
| README `n_layers` metadata 4 / 8 / 3 | — | **4 / 8 / 3** | ✓ |
| README "+561 and +1700 entities" | — | **+561 / +1700** | ✓ |
| README "21.01 and 3001 return exactly their natives (5/5, 10/10); the tip returned 14 and 5 unmatched" | — | reproduces (baseline 19 and 15 proposals) | ✓ |
| README "10.17a 17 proposals, 11.01 12" | — | **17 / 12** | ✓ |
| *(not claimed — my own check)* matched-defpoint residuals, 41 constructs, 3 annotated sheets | — | **identical to baseline to 3 dp**: leader med 7.197 max 10.814; dim med 0.026 max 0.036 | ✓✓ |

**No README or `ir/DESIGN.md` figure disagrees with `doc_claims_check.py`.** The
only non-reproducing number anywhere is the "316" of finding 7, which that
script does not cover.

**The last table row is the strongest evidence in the ledger's favour and nobody
wrote it down.** Not one published coordinate on any matched corpus construct
moved between `1f6551c` and the round-4 tree. It is also precisely why findings
1–5 escaped: no diamond terminators in the corpus, no foreign arrowhead inside a
cluster's attach radius, no >60° arrowheads, and an 18 pt greedy match tolerance
that would hide a 9 pt error anyway. **"The corpus did not move" is not
evidence, for the third round running** — the same lesson round 2 recorded about
the missing text layer.

### Round-4 claims that verified SOUND

* **Item 1 — `on_spine` no longer feeds the soundness tier.** Confirmed by code:
  `_end_tier` (`:1360-1366`) reads only `state` and `attached`. (Its cost is
  finding 4.)
* **Item 2 — split leg projects the apex, continuous leg keeps the shaft-end
  fallback.** Confirmed by code (`:1286`, `:2227`) and by the F2 fixture at
  `y=5`, which is genuinely fixed. (Its gap is finding 3.)
* **Item 4 — native suppression is a CAP on both legs, duplication is the same
  SPAN.** Verified exhaustively and it is **clean**:

  ```
  same span                   -> native 1.0 | continuous 0.45 /superseded_by_native
  same span REVERSED          -> native 1.0 | continuous 0.45 /superseded_by_native
  same span, min_conf=0.5     -> native 1.0                      (composed dropped, native called)
  ONE shared end (0,0)-(60,0) -> native 1.0 | continuous 0.952   (NOT capped — correct)
  enclosing (-9,0)-(109,0)    -> native 1.0 | continuous 0.45 /superseded_by_native
  enclosing (-10,0)-(110,0)   -> native 1.0 | continuous 0.952
  ```

  Reversed spans handled; partial overlap correctly not treated as duplication;
  a degenerate single-defpoint native affects nothing. Residual limitation only:
  the tolerance is `max_arrowhead_size` **per end**, so a genuinely different
  composed dimension nested inside a native (both ends within one arrowhead
  scale) is capped. Strictly better than the shared-endpoint *delete* it
  replaced.
* **Ranking determinism.** `(tier, d)` tuple comparison with first-wins ties over
  a deterministic candidate iteration order; `pairs.sort` on
  `(gap, fold, i, j, style)` is a total order. No non-determinism found. No tier
  is unreachable (`_CONTRADICTED_CAP` is reachable on the split leg via the
  founder-on-contradicted path at `:2148-2156`).
* **Ceiling hoist consistency.** The pre-witness-search `ceiling` (`:2264-2272`)
  uses the same predicates in the same order as the final `min(confidence,
  ceiling)` (`:2342`); the later no-text cap only ever lowers, so the hoist
  cannot wrongly drop a construct. No 0.3-vs-0.5 asymmetry found.
* **Prune tightness.** `attach_radius = min(search_radius, 0.75 · 1.5 · scale)`
  exactly equals the loosest bound `_arrow_attach` can grant given the candidate
  gate's `1.5 ×` size cap. No off-by-one. (Caveat: a caller passing
  `search_radius < 1.125 × scale` silently tightens attachment; the default
  `4 × scale` does not.)

---

## 4. Repair sizing, and whether these are one change or four

### Findings 1–2: regression status
* **Finding 1** — clean on v0.1.0 **and** on the tip → **strictly a regression
  introduced by the uncommitted work.**
* **Finding 2** — clean on v0.1.0 **and** on the tip → **strictly a regression
  introduced by round 4.**
* **Finding 3** — regression vs v0.1.0; **pre-existing at the tip**; not closed.
* **Finding 4** — callability regression vs v0.1.0; **mixed trade** vs the tip.
* **Finding 5** — regression vs v0.1.0; **pre-existing at the tip**.
* **Finding 6** — **pre-existing in all three**.
* **Findings 7–10** — documentation / process, introduced by the remediation itself.

### Finding 1 — contained, two parts, independent of the rest
The coordinate half is a **one-line substitution at `queries.py:1275`**: the
blunt branch invents a centroid-projection apex where the fill-cluster leg
already has the right primitive (`_reach_on_ray`, `:1296`); on the diamond that
returns `(100, 0)` exactly. Low risk, mechanically verifiable.

The second half is the **unconditional 0.45 cap on blunt** (`:1362`,
`:2264-2272`, plus `arrow_blunt` in `find_leaders` at `:1638-1641`). That is a
deliberate policy — it exists to stop a plain rectangle founding a dimension —
so it is a judgement call plus a corpus re-run, not a code problem. The
`blunt_terminators` count is 0 at 0.5 on all ten sheets, so relaxing it cannot
move the published corpus figures; it needs new fixtures instead.

**Size: half a day. Touches only the blunt clauses. Does not touch the ranking.**

### Finding 2 — one function, but it needs a new rule, not a tweak
`_end_tier` collapses three unrelated conditions into tier 1 (blunt / fill
cluster / detached), and distance is consulted only *within* a tier — so a
candidate 0.04 pt from the tip and one 3.24 pt away are interchangeable. The
minimal repair is a **distance-dominance rule in the comparison at
`:2111-2130`** (a candidate an order of magnitude nearer wins regardless of
tier) — roughly ten lines, no redesign of the ordering. But it re-opens the
trade round 4 decided (their measured case: a stipple splash 5 pt off an end
lifting a construct 0.45 → 0.908), so it needs fixtures pinning **both**
directions and a corpus re-run.

**Size: one day. Contained in `_end_tier` + the per-end comparison, but it is a
design decision the round-4 rationale did not make.**

### Finding 3 — small, but it lands on the same line as finding 4
The apex published when `on_spine` is `g.apex` with **no axial bound** relative
to the shaft tip; `attached` bounds only `|centroid − tip|`, which leaves the
apex free to sit up to `0.75·max(scale, h) + 2h/3` beyond the end. The repair is
to widen the continuous leg's publish test at **`:2227`** from "on spine" to "on
spine **and** axially plausible" — the apex's along-shaft reach beyond the tip
within a small fraction of the arrowhead scale — falling back to the tip (and
the existing cap) otherwise. One line plus a small helper; it reuses the
`arrow_off_spine` cap machinery already there (the evidence key would want
renaming, e.g. `arrow_apex_implausible`).

Crucially the bound **cannot** be shared with the split leg: a split half's arrow
legitimately sits *outside* its half-shaft, so its apex is genuinely beyond the
tip by ~one arrow length (measured 7.2 pt). This is exactly where the two legs
already differ on purpose, so the fix belongs there and nowhere else.

**Size: two hours of code. But see the interaction below.**

### Finding 4 — smallest code, largest coupling
Nothing to add: the fix is to make the rank see what the cap ladder cares about,
i.e. re-admit a coordinate-quality signal into candidate selection *without*
re-introducing round 2's blocker #3 (the leader-arrowhead steal that removing
`on_spine` from the tier was meant to close). That is a **design decision, not an
edit** — and it is the same decision finding 2 forces.

### Are these one change or four? — **two changes, not four**

* **Change A (independent, safe to land alone): finding 1.** The blunt branch's
  published coordinate and the blunt cap policy. Touches the `blunt` clauses
  only. It cannot move the corpus (0 blunt proposals at 0.5) and it does not
  read or write the ranking.

* **Change B (one coordinated change): findings 2, 3 and 4.** These are three
  halves of one question — *which candidate owns this end, and how far do we
  trust its coordinate.* They are **not independent**:
  * finding 2 changes **who wins** the end;
  * finding 4 changes **what a win costs** (the cap);
  * finding 3 changes **whether the winner's coordinate is publishable at all**.

  Fixing any one alone shifts the other two, and two of them pull in opposite
  directions: repairing 3 by widening what triggers the tip-fallback-plus-cap
  makes 4 strictly **worse** (more true constructs capped out of the band),
  while repairing 4 by preferring on-spine candidates in the rank re-opens the
  arrowhead steal unless 2's distance-dominance rule lands with it. They share
  four sites — `_end_tier` `:1360-1366`, the per-end comparison `:2111-2130`,
  the continuous apex publication `:2227`, and the cap ladder `:2264-2272`.

  **Land them together, with fixtures pinning every direction that was measured
  in rounds 2–4, and re-run the corpus + the residual check.**

* **Sequencing:** A first (independently verifiable, corpus-inert), then B.
  Both edit `_end_tier`, but different clauses — A the `blunt` clause, B the
  `fill_cluster` / `attached` clauses and the ordering — so they do not
  textually conflict if A lands first.

* **Findings 5 and 6** are separable and lower priority: 5 wants a slenderness
  precondition on closed-3 candidates (or an apex-vote margin test that works
  for triangles) and can wait; 6 wants a documented sharp edge, not a fix.
  **Findings 7–10 are documentation and staging** and cost minutes.

---

## 5. What I could not verify, and why

* **Phase-3.1 "before" figures** (295→2, 273→10, 91→6, 77→19, 57→23, 151→20,
  137→38 — the left-hand numbers). That revision is not in the tree; only the
  right-hand column is reproducible, and it reproduces exactly.
* **Round 4's own blocker claim** — "across skews of 0–25 degrees at both junk
  distances, the dimension never claims the neighbouring leader's arrowhead". I
  did not sweep that specific fixture. I reached the same steal by a different
  route (finding 2, via a cluster-terminated end), so the claim may well hold
  for the geometry it was written against; it does not hold for all geometry.
* **The 9.59° arrows-outside sharp edge.** Not independently derived; consistent
  with the `asin(half_width / leg)` reading of the code but not measured.
* **"10/10 byte-identical truth regeneration"** (`planlens.dxf.truth`) — not
  re-run. Round 2 verified it independently.
* **The render-adjudication of detail-sheet survivors** ("~14/15 semantic
  precision on the curb-ramp sheet") — requires visual inspection; not attempted.
* **Whether findings 1–5 bite on any real non-Mecklenburg drawing.** No such
  corpus exists in either repo. That absence *is* the finding, not a caveat on it:
  the corpus cannot falsify a change to a terminator style it does not contain,
  and five of the six code findings live in exactly that blind spot.
* **The app's full gate** — deliberately not run (instructed; a full run was in
  flight elsewhere on this machine).

---

## Appendix A — repro fixtures

Saved and runnable at
`C:/Users/socon/.claude/jobs/69be0e95/tmp/round4_repro.py`. It prints findings
1–6 and round-4 item 4 against any of the three trees:

```
<venv>/python round4_repro.py work    # uncommitted round-4 working tree
<venv>/python round4_repro.py tip     # committed tip 1f6551c
<venv>/python round4_repro.py v010    # published PyPI 0.1.0
```

The comparison trees are produced **read-only** from the planlens repo:

```bash
cd C:/Users/socon/OneDrive/dev/planlens
git archive 1f6551c | tar -x -C C:/Users/socon/.claude/jobs/69be0e95/tmp/base
git archive v0.1.0  | tar -x -C C:/Users/socon/.claude/jobs/69be0e95/tmp/v010
```

Full source:

```python
"""Self-contained repro fixtures for the planlens round-4 adversarial review."""
import math
import sys

WORK_DIR = r"C:\Users\socon\OneDrive\dev\planlens"
TIP_DIR = r"C:\Users\socon\.claude\jobs\69be0e95\tmp\base"
V010_DIR = r"C:\Users\socon\.claude\jobs\69be0e95\tmp\v010"

which = sys.argv[1] if len(sys.argv) > 1 else "work"
sys.path.insert(0, {"work": WORK_DIR, "tip": TIP_DIR, "v010": V010_DIR}[which])

import planlens
print("[%s] planlens: %s" % (which, planlens.__file__))
from planlens.ir import queries as q
from planlens.ir.results import Dimension, DrawingIR, Line, Polyline, TextItem

MAS = 9.31  # the validation sheets' arrowhead scale, points


def ir_of(entities, source="pdf_vector"):
    ir = DrawingIR(units="pt", coordinate_space="page", origin="bottom_left",
                   source=source, width=612.0, height=792.0)
    for i, e in enumerate(entities):
        e.id = "e%d" % i
        ir.add(e)
    return ir


def chevron(apex, d, leg=7.3, base=2.4):
    """[barb, apex, barb] open chain - the native-leader arrowhead flavor."""
    dx, dy = d
    px, py = -dy, dx
    h = math.sqrt(leg * leg - (base / 2) ** 2)
    bx, by = apex[0] - dx * h, apex[1] - dy * h
    return Polyline(vertices=[(bx + px * base / 2, by + py * base / 2), apex,
                              (bx - px * base / 2, by - py * base / 2)],
                    closed=False)


def triangle(apex, d, leg=6.0, ratio=0.33, rot=0):
    """Closed filled triangle. ratio = base/leg. rot rotates the VERTEX ORDER
    (a plotter is free to start the outline at any corner)."""
    dx, dy = d
    px, py = -dy, dx
    base = ratio * leg
    h = math.sqrt(max(leg * leg - (base / 2) ** 2, 1e-9))
    bx, by = apex[0] - dx * h, apex[1] - dy * h
    v = [apex, (bx + px * base / 2, by + py * base / 2),
         (bx - px * base / 2, by - py * base / 2)]
    return Polyline(vertices=v[rot:] + v[:rot], closed=True)


def diamond(apex, d, leg=6.0, w=2.4):
    """Rhombus/diamond terminator (a centrally symmetric quad)."""
    dx, dy = d
    px, py = -dy, dx
    return Polyline(vertices=[
        apex,
        (apex[0] - dx * leg / 2 + px * w / 2,
         apex[1] - dy * leg / 2 + py * w / 2),
        (apex[0] - dx * leg, apex[1] - dy * leg),
        (apex[0] - dx * leg / 2 - px * w / 2,
         apex[1] - dy * leg / 2 - py * w / 2),
    ], closed=True)


def trapezoid(apex, d, leg=6.0, tip_w=0.8, base=3.0):
    """Truncated-triangle ('flat tip') terminator."""
    dx, dy = d
    px, py = -dy, dx
    bx, by = apex[0] - dx * leg, apex[1] - dy * leg
    return Polyline(vertices=[
        (apex[0] + px * tip_w / 2, apex[1] + py * tip_w / 2),
        (apex[0] - px * tip_w / 2, apex[1] - py * tip_w / 2),
        (bx - px * base / 2, by - py * base / 2),
        (bx + px * base / 2, by + py * base / 2)], closed=True)


def dart(apex, d, leg=7.3, base=3.0, notch=0.45):
    """Concave swallowtail / barbed arrowhead."""
    dx, dy = d
    px, py = -dy, dx
    bx, by = apex[0] - dx * leg, apex[1] - dy * leg
    nx, ny = apex[0] - dx * leg * notch, apex[1] - dy * leg * notch
    return Polyline(vertices=[apex, (bx + px * base / 2, by + py * base / 2),
                              (nx, ny),
                              (bx - px * base / 2, by - py * base / 2)],
                    closed=True)


def fill_cluster(tip, d, n=7, span=3.0):
    """Micro-dot stipple arrowhead straddling `tip` (the Mecklenburg anatomy)."""
    out = []
    for i in range(n):
        t = (i / (n - 1.0)) * span - 0.5 * span
        px, py = -d[1], d[0]
        for k in (-0.6, 0.0, 0.6):
            x = tip[0] + d[0] * t + px * k
            y = tip[1] + d[1] * t + py * k
            out.append(Line(start=(x, y), end=(x + 0.06, y + 0.06)))
    return out


def core():
    """A 100 pt dimension: shaft, two witness lines, value text. Terminators
    are added by each scenario. TRUTH: end_a=[0,0] end_b=[100,0] length 100."""
    return [Line(start=(0.0, 0.0), end=(100.0, 0.0)),
            Line(start=(0.0, -5.0), end=(0.0, 25.0)),
            Line(start=(100.0, -5.0), end=(100.0, 25.0)),
            TextItem(content="100'", position=(50.0, 6.0), height=6.0)]


def report(tag, entities, min_confidence=0.0):
    props = q.find_dimensions(ir_of(entities), max_arrowhead_size=MAS,
                              min_confidence=min_confidence)
    if not props:
        print("  %-46s NO PROPOSAL" % tag)
        return
    p = props[0]
    flags = ",".join(k for k in p["evidence"]
                     if k.startswith("arrow_") or "blunt" in k)
    print("  %-46s b=%-18s len=%-9s conf=%-6s [%s]"
          % (tag, p["end_b_xy"], p["length"], p["confidence"], flags))


# FINDING 1 - box-like terminators: correct apex discarded, construct capped
print("\nFINDING 1  box-like terminators (truth b=[100,0] len 100)")
for name, mk in (("slender triangle (control)", lambda a, d: triangle(a, d)),
                 ("diamond terminator", diamond),
                 ("trapezoid (flat tip)", trapezoid),
                 ("dart (concave)  [= FINDING 6]", dart)):
    report(name, core() + [mk((0.0, 0.0), (-1.0, 0.0)),
                           mk((100.0, 0.0), (1.0, 0.0))])

# FINDING 2 - a fill cluster loses its end to a FARTHER directional arrowhead
print("\nFINDING 2  fill-cluster arrowhead vs foreign directional chevron")
A = triangle((0.0, 0.0), (-1.0, 0.0))
report("cluster alone (centroid 0.04 pt from the tip)",
       core() + [A] + fill_cluster((100.0, 0.0), (1.0, 0.0)))
report("cluster + foreign chevron apex=(108,0.5)",
       core() + [A] + fill_cluster((100.0, 0.0), (1.0, 0.0))
       + [chevron((108.0, 0.5), (1.0, 0.0))])
ir = ir_of(core() + [A] + fill_cluster((100.0, 0.0), (1.0, 0.0))
           + [chevron((108.0, 0.5), (1.0, 0.0))])
for p in q.find_dimensions(ir, max_arrowhead_size=MAS, min_confidence=0.5):
    print("    arrowhead_ids=%s   <- named for exclude_dimensions arbitration"
          % p["arrowhead_ids"])

# FINDING 3 - ON-SPINE axial outlier: F2's other half, uncapped
print("\nFINDING 3  on-spine axial outlier (F2's axial half)")
B = chevron((100.0, 0.0), (1.0, 0.0))
report("clean", core() + [chevron((0.0, 0.0), (-1.0, 0.0)), B])
for ax, ay in ((108.0, 0.0), (108.0, 1.0), (108.0, 5.0)):
    report("junk chevron apex=(%g,%g)" % (ax, ay),
           core() + [chevron((0.0, 0.0), (-1.0, 0.0)), B,
                     chevron((ax, ay), (1.0, 0.0))])

# FINDING 4 - nearer OFF-SPINE junk caps a true dimension out of the band
print("\nFINDING 4  nearer off-spine junk shadows the true arrow")
for oy in (2.0, 3.0, 4.0, 5.0):
    report("junk chevron apex=(103,%g)" % oy,
           core() + [chevron((0.0, 0.0), (-1.0, 0.0)), B,
                     chevron((103.0, oy), (1.0, 0.0))])

# FINDING 5 - apex vote order-dependent at 60 deg, inverted above it
print("\nFINDING 5  equilateral / wide triangle apex vote")
for rot in (0, 1, 2):
    report("base/leg=1.00 (60 deg tip) vertex-rot=%d" % rot,
           core() + [triangle((0.0, 0.0), (-1.0, 0.0), ratio=1.0, rot=rot),
                     triangle((100.0, 0.0), (1.0, 0.0), ratio=1.0, rot=rot)])
for ratio in (0.95, 1.05, 1.20):
    tip_deg = 2 * math.degrees(math.asin(min(ratio / 2.0, 1.0)))
    report("base/leg=%.2f (%.0f deg tip) any order" % (ratio, tip_deg),
           core() + [triangle((0.0, 0.0), (-1.0, 0.0), ratio=ratio),
                     triangle((100.0, 0.0), (1.0, 0.0), ratio=ratio)])

# ROUND-4 ITEM 4 (verified SOUND) - native-dimension suppression
print("\nROUND-4 ITEM 4  native suppression = CAP on the same SPAN (sound)")
if not hasattr(q, "_native_span_at"):
    print("  (_native_span_at absent on this tree - round-4 code only)")
else:
    def native_scene(defpoints):
        e = core() + [chevron((0.0, 0.0), (-1.0, 0.0)),
                      chevron((100.0, 0.0), (1.0, 0.0))]
        if defpoints is not None:
            e.append(Dimension(defpoints=defpoints, measurement=100.0,
                               text="100'"))
        return e

    cases = (("same span", [(0.0, 0.0), (100.0, 0.0)], 0.0),
             ("same span REVERSED", [(100.0, 0.0), (0.0, 0.0)], 0.0),
             ("same span, min_conf=0.5", [(0.0, 0.0), (100.0, 0.0)], 0.5),
             ("ONE shared end (0,0)-(60,0)", [(0.0, 0.0), (60.0, 0.0)], 0.0),
             ("enclosing (-9,0)-(109,0)", [(-9.0, 0.0), (109.0, 0.0)], 0.0),
             ("enclosing (-10,0)-(110,0)", [(-10.0, 0.0), (110.0, 0.0)], 0.0))
    for tag, dps, mc in cases:
        props = q.find_dimensions(ir_of(native_scene(dps), source="dxf"),
                                  max_arrowhead_size=MAS, min_confidence=mc)
        bits = []
        for p in props:
            sup = p["evidence"].get("superseded_by_native")
            bits.append("%s=%s%s" % (p["evidence"].get("path"),
                                     p["confidence"],
                                     "/sup:" + sup if sup else ""))
        print("  %-46s %s" % (tag, " | ".join(bits)))
```

## Appendix B — corpus measurement commands

```bash
cd C:/Users/socon/OneDrive/dev/planlens
<venv>/python -m pytest -q -p no:cacheprovider                 # 632 passed

cd C:/Users/socon/OneDrive/dev/GeotechStaffEngineer
.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider \
    funhouse_agent/tests/test_drawing_ir_adapter.py \
    funhouse_agent/tests/test_render_region.py                 # 58 passed
.venv/Scripts/python.exe module_work/drawing_ground_truth/score_compositions.py
.venv/Scripts/python.exe module_work/drawing_ground_truth/doc_claims_check.py
```

The @0.5 corpus run, the per-sheet FP/precision counts and the defpoint-residual
comparison were made with three short scratch scripts alongside the repro file:
`score05.py`, `fp05.py`, `residual.py` (each takes `work` / `base` to select the
tree). `layer_probe.py` implements both layer-inheritance rules for finding 8;
`meta.py` reads the `n_layers` / `n_block_entities` metadata claims.
