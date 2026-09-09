# Code review — planlens Phase 3.2 (unreleased)

**Reviewed** 2026-09-06 · **Scope** `git diff v0.1.0..HEAD` in
`C:/Users/socon/OneDrive/dev/planlens` — 4 unreleased commits, 8 files,
~1066 lines. Working tree was clean at review time.

| commit | subject |
|---|---|
| `2b4eee1` | Signed arrow-direction attach + leader letterform caps (Phase 3.2) |
| `c32e003` | DXF ingest: INSERT block-geometry explosion with depth/entity caps |
| `6ba4e90` | truth.py: extract paper-space layouts (byte-identical corpus claim) |
| `1f6551c` | README: Phase-3.2 measured capability story |

**Method** — 10 finder angles, an adversarial verification pass and a gap
sweep, with empirical probes run against the live pipeline (350-test baseline
green). Findings 1–8 were demonstrated by running the code, including an
old-vs-new comparison against a v0.1.0 archive for F1 and `ezdxf` 1.4.4 probes
for F7. Findings 9–13 are code-read confirmations. 14–15 are packaging and
maintainability.

**Two candidates were refuted and dropped**, recorded here so they are not
re-raised:

- `truth.py`'s `layout_name == "Model"` check — `ezdxf` guarantees that name
  for the model layout (`doc.modelspace().name == "Model"`,
  `layout_names_in_taborder()` returns `['Model', ...]`).
- A suspected scorer break from the new paper-space keys —
  `score_compositions.py:195` pins `truth["spaces"]["model"]`, so the addition
  is backward-safe.

---

## The cross-cutting concern

Findings **F1, F3 and F5 are the same shape**. Phase 3.2 raised corpus recall
to 25/25 leaders and 16/16 dimensions, but each of those three mechanisms
*deletes or caps genuine constructs* whose anatomy differs from the 10-sheet
Mecklenburg corpus — non-triangular terminators, arrows larger than the
estimated sheet scale, apex-closed outlines. The measured gains are real; the
risk is that the gates are calibrated to one agency's plotter and now fail
closed on drawings from anywhere else.

**F2 and F4 are the reverse trade**: the apex adoption and the shaft-length
floor bypass both *widened* what qualifies, and both are demonstrably
exploitable by glyph junk.

That pairing is the thing to hold onto — the fix for one side must not be paid
for out of the other.

---

## Findings

Ranked most-severe first, as delivered.

### F1 — `_dim_arrow_attach` deletes rather than caps
`planlens/ir/queries.py:1010`

Returning `None` makes any non-slender or >30°-crooked terminator
unattachable, so whole dimension styles vanish at every `min_confidence` —
violating the file's own documented cap-not-delete discipline.

*Demonstrated:* an architectural box-terminator dimension (small filled square
at each shaft end, witness lines, value text) scores 0.859 "continuous" on
v0.1.0 and returns **no proposal at any `min_confidence`** on HEAD. Slash,
tick, dot and box terminators, plus hand-drafted arrows off-axis by more than
30°, are now invisible rather than down-ranked.

### F2 — continuous-dimension ends adopt an unconstrained apex
`planlens/ir/queries.py:1700`

The ends adopt the candidate's apex with no bound tying that apex back to the
shaft end, so one sign-passing junk candidate drags `end_a/b_xy`, `midpoint`,
`length`, `angle_deg` and the witness-search point onto the junk's geometry.

*Demonstrated:* shaft (0,0)–(100,0), genuine arrow at end A, glyph chevron
with apex at (108,5) pointing (1,0) near end B — centroid 5.9 pt from the tip,
so it passes the 0.75× attach test. Proposal reports `end_b=[108.0,5.0]`
(9.4 pt off the true end, beyond the scorer's 18 pt match), length 108.1
instead of 100, angle 2.65 instead of 0. Attach constrains only
|centroid − tip|; the apex is unconstrained.

### F3 — detach cap and candidate gate disagree about scale
`planlens/ir/queries.py:1265`

The `arrow_detached` cap uses 0.75× `max_arrowhead_size` while the
open-3-vertex candidate gate deliberately admits arrows up to 1.5× that scale
— so genuine leaders whose arrows exceed the sheet-statistic estimate are
capped to 0.45, below the default call threshold.

*Demonstrated* on the exact case the candidate gate documents (12.3 pt arrows
vs a 10.0 pt estimate): an apex-anchored chevron leg=12.3 with a clean shaft
and tail text scores 0.45 with `arrow_detached=True` and
`signed_axis_alignment=1.0` — alignment perfect, text present, still capped.
The identical anatomy at corpus scale (7.3 @ 9.31) scores 0.953.

### F4 — the narrow-dimension bypass removes the only length guard
`planlens/ir/queries.py:1647`

The `min_shaft_length` bypass for two-triangle shafts removes the only length
guard against glyph-scale strokes pairing into dimension proposals, and keys
on exact list equality so mixed kinds hit an arbitrary cliff.

*Demonstrated:* two letterform chevrons (leg 5.0) at the ends of a 5.2 pt
stroke with glyph verticals as witnesses produces a "continuous" proposal at
confidence **0.985** (separation 5.2 vs the 18.62 floor that previously
blocked it). On the SHX notes sheets whose dimension count the README reports
as zero at 0.3+, every short letter stroke between two chevrons is now a
near-1.0 dimension. Also `['triangle','fill_cluster']` narrow dims stay
rejected while `['triangle','triangle']` passes, with no principled
difference.

### F5 — a repeated apex vertex inverts the apex axis
`planlens/ir/queries.py:980`

`_intrinsic_apex_axis` picks the vertex farthest from the centroid of the
others, which a repeated apex vertex defeats — the duplicate pulls that
centroid onto itself and elects a base corner as the apex, inverting the axis.

*Demonstrated:* an arrow chain `[A, b1, b2, A]` (an apex-closed 4-vertex
outline — exactly what the open-4 "near-ring" candidate class was added to
accept) yields intrinsic apex (107.2, 101.2), a base corner, and
`signed_axis_alignment = −0.949` against its own shaft. The genuine leader is
flagged `arrow_direction_violation` and capped to 0.45; as a dimension arrow
it would be rejected outright. No dedup of coincident vertices before the
apex vote.

### F6 — ingest and ground truth disagree about block-nested annotations
`planlens/ir/ingest.py:313` (with `planlens/dxf/truth.py`)

Block explosion promotes LEADER/MULTILEADER/DIMENSION found inside blocks to
first-class native annotations at confidence 1.0, but `truth.py` walks only
top-level entities.

*Demonstrated:* a block containing a DIMENSION and a LEADER, inserted once,
yields a confidence-1.0 `native_dxf` dimension proposal from `find_dimensions`
while `extract_native_annotations` reports 0 dimensions and 0 leaders for the
same file. On corpus sheets with detail blocks these are unmatchable
confidence-1.0 detections, and their defpoints/tips additionally **suppress**
nearby composed proposals through the native-dedup filters at
`queries.py:1730` and `:1310`.

### F7 — exploded block geometry ignores layer-0 / ByBlock inheritance
`planlens/ir/ingest.py:164`

Exploded entities keep the block-definition layer verbatim; geometry on layer
`'0'` should inherit the INSERT's layer.

*Confirmed against `ezdxf` 1.4.4:* a block whose LINE is on layer `'0'`,
inserted on layer `'X'`, yields a virtual entity with `dxf.layer == '0'`, and
`_handle` records `'0'`. Every standard detail drawn the normal CAD way lands
on layer `'0'` in the IR instead of its placed layer, so `discover_layers`,
layer filters and any layer-scoped query miss the newly exploded linework —
the very geometry (+561 and +1700 entities) this commit was written to
surface. `n_layers` metadata is also inflated by block-internal layer names.

### F8 — the curve branch discards block provenance
`planlens/ir/ingest.py:204`

The ELLIPSE/SPLINE branch overwrites `style` with `approx_from_<etype>`,
discarding the `block:` provenance the INSERT explosion just applied.

*Demonstrated:* a block `DETAIL` containing a SPLINE yields a Polyline with
`style='approx_from_spline'` while its sibling LINE correctly carries
`'block:DETAIL'`. Any caller using the documented `style.startswith('block:')`
discriminator — including this commit's own new tests — misclassifies exploded
curve geometry as directly drawn model-space work.

### F9 — one bad entity aborts a block, with a misleading warning
`planlens/ir/ingest.py:317`

The recursive `_handle` calls sit inside the parent INSERT's `try` block, so
one bad virtual entity aborts the rest of that block's explosion and reports
`Skipped INSERT on layer X: ...` while partial geometry is already in the IR.

A block with 800 entities where entity 300 raises (malformed spline, missing
dxf attribute, `ezdxf` transform error on a non-uniformly-scaled entity)
ingests the first 300, silently drops 500, and warns "Skipped INSERT" — which
reads as *the INSERT was skipped*, the opposite of what happened. Callers
cannot distinguish a fully-skipped INSERT from a half-exploded one, and
`n_block_entities` includes the truncated run.

### F10 — the explosion budget counts emissions, not ingests
`planlens/ir/ingest.py:314`

`n_exploded` increments per virtual entity emitted rather than per entity
actually ingested, so unsupported types and nested INSERT containers burn the
budget and overstate the metadata. A block of 50,000 unsupported entities
(SOLID/3DFACE/ATTDEF/nested INSERT wrappers) exhausts `max_block_entities` and
truncates the explosion while contributing zero IR entities;
`metadata['n_block_entities']` then reports 50,000 for an IR that gained
nothing. The new test asserting `n_block_entities >= 4` passes either way, so
the count's meaning is untested.

### F11 — rejected continuous dimensions become split-pair candidates
`planlens/ir/queries.py:1614`

Shafts whose second arrow now fails the signed attach test fall through into
the `n_arrowed == 1` split-half pool, so rejected continuous dimensions become
new split-shaft pairing candidates rather than being discarded. There they can
pair with any collinear opposed half within 40× `max_arrowhead_size`. The
signed gate's precision gain on the continuous leg is partly re-spent as new
false-pair opportunities — the split leg's arrows never go through the signed
test at all.

### F12 — a stipple dot is published as a CAD defpoint
`planlens/ir/queries.py:1006`

For fill-cluster candidates the "apex" is just the member center farthest
along `sdir` — an arbitrary stipple dot — yet it is now published as a CAD
defpoint end and used as the witness-search anchor. A mixed triangle+cluster
continuous dimension on a stippled sheet reports `end_b_xy` at whichever
micro-dot sits farthest along the shaft direction (anywhere within the ~0.6×
`max_arrowhead_size` cluster span, ~5.6 pt at corpus scale), and `_witness_at`
runs at that dot rather than the true defpoint. Before this change both ends
were the shaft endpoints, which at least sat on the drawn geometry.

### F13 — `explode_blocks` silently changes app input population
`GeotechStaffEngineer/funhouse_agent/adapters/drawing_ir_adapter.py:129`

It defaults to `True` and the app's DXF adapter exposes no way to turn it off,
so every downstream geotech flow silently changes its input population when
planlens 0.2 lands (the app pins `planlens[raster]>=0.1`).
`from_dxf(filepath=..., units=...)` at adapter line 129 (and line 85) now
ingests title-block borders, north arrows and standard-detail linework that
were previously invisible. `candidate_ground_surface`, `build_slope_geometry`
and the FEM-input builders consume the same IR: a title-block rectangle or
detail-block linework can outrank the real ground surface, and dense sheets
grow by thousands of entities in the composition loops.

### F14 — version and test-fixture packaging
`planlens/pyproject.toml:7`

Version is still 0.1.0 while four unreleased commits changed detection
behavior; 0.1.0 is already published and the app pins `planlens[raster]>=0.1`,
so publishing this tree needs a bump. Relatedly,
`[tool.setuptools.packages.find]` excludes `*.tests`, yet the app's suites
import `planlens.ir.tests.leader_fixtures` and `.construct_fixtures` — those
pass only against the editable dev install, not a released wheel.

### F15 — one rule, two implementations
`planlens/ir/queries.py:1227`

`find_leaders` reimplements the signed-attach physics inline instead of
calling `_dim_arrow_attach`, and the 0.75×-scale constant is magic-numbered in
both legs. The same physics now lives at `queries.py:1227-1268` (inline apex +
signed dot + 0.75× detach bound) and at `:1004-1012`/`:1501`
(`_dim_arrow_attach` + `attach_radius`). A future tolerance change must be
made in both places or leaders and dimensions will disagree about which arrows
attach — exactly the arbitration inconsistency this commit was written to fix.
The per-candidate apex is also recomputed for every shaft it is tested
against, though it depends only on the candidate.

---

## Disposition

### Round 1 — fixes applied (2026-09-06)

Five design agents reproduced every finding before proposing anything, three
builders applied fixes split by file, and an integration pass re-measured.
Result: all fifteen addressed, planlens suite 316 → 375 tests green, and the
corpus **bit-for-bit unchanged** — 25/25 and 16/16 at the 0.3 observational
threshold, 23/25 and 16/16 at the 0.5 default, all seven false-positive counts
identical.

Two builder-reported honesty notes proved load-bearing later: the new
blunt-terminator branch fired on **0 of 155** dimension proposals across all
ten sheets, and the `truth.py` block-annotation walk promoted nothing (all ten
truth files regenerate byte-identical). Both unchanged numbers were *silence,
not evidence*.

### Round 2 — independent verification (2026-09-07)

Six independent agents re-ran every original failure scenario and then attacked
the fixes. **Four of six returned do-not-ship**, with 6 blockers, 19 majors and
28 minors. The corpus headline was reproduced independently by two of them, so
the measurement itself is sound. What failed was the fixes.

**The finding that matters most, and the reason it was missed.** All ten
Mecklenburg sheets are SHX-stroked with **no text layer**, so `text_score` is
always 0 on them. Every behavior gated on a text layer is invisible to the
corpus *by construction*. That is how a fix that silently drops a real leader
on any ordinary text-bearing drawing measured "zero change." On this project,
"the corpus did not move" is not evidence that a text-dependent change is safe.
A permanent text-bearing regression fixture is now required.

**The six blockers.** Three are new deletions introduced by a fix, in a train
whose entire purpose was replacing deletions with caps:

1. The blunt-terminator branch re-opened the arrowhead steal Phase 3.2 existed
   to close. An ordinary 3×3 rectangle at the far end of a crossing line founds
   a 0.915 "dimension" that claims a real leader's arrowhead; that genuine
   0.976 leader, tail text and all, is then silently dropped by
   `exclude_dimensions`. The guarding cap is escapable through
   `ext_ends == 2 and text_score > 0` — the normal condition on any drawing
   with a text layer.
2. A nearer *contradicted* candidate now shadows the true arrow. One letterform
   chevron 1.02 pt from an end, against the real arrow's 2.40 pt, takes a
   genuine dimension from 0.976 to 0.25 with its published end 7.96 pt off the
   drawn geometry — and it disappears at the default threshold.
3. The split-leg `on_spine` refusal deletes real split-shaft dimensions from
   about 10° off-axis; baseline returned those out to 30° at ~0.95. The
   tolerance is `asin(base_ratio/2)`, so **the incentive is inverted** — the
   slenderer and more arrow-like the arrowhead, the tighter the tolerance. The
   code comment concedes the deletion was chosen because the alternative
   "measured four NEW false pairs," which is the corpus-tuning the house rules
   forbid.
4. The extent floor reintroduces exactly the defect F1 was raised about: a real
   narrow both-triangle dimension at 18.0 pt span that HEAD proposed at 0.968
   is now deleted at every `min_confidence`. Its comment claims it can "only be
   stricter"; it is in fact looser for non-`[triangle,triangle]` kinds.
5. The app pin `planlens[raster]>=0.1` breaks at **runtime**, not just in
   tests: the adapter unconditionally passes `explode_blocks=` to a `from_dxf`
   that 0.1.0 does not accept, so every DXF `digitize_drawing` raises
   `TypeError` against a released install. Neither repo's tests reach this,
   because both run editable installs.
6. `planlens/testing/` and `planlens/tests/test_packaging.py` are untracked
   while the tracked-and-modified shims now delegate to them, so a commit that
   misses either **breaks the source tree**, not merely the wheel — and the
   guard test that would catch it is in the same untracked set.

**The invented pointed-vs-blunt criterion is unsound as written.** It measures
against the interior angle of the regular polygon with the same vertex count,
so its watershed sits exactly *on* the regular form. It is neither
order-invariant nor translation-invariant; a real drafted equilateral "datum
triangle filled" lands on the blunt side, costing 0.676 → 0.45 and moving the
defpoint 4.04 pt; a pentagon dot is misclassified at 24 of 25 rotations; 2%
plot jitter flips a box from 0.907 to 0.250. The dot and oblique-tick styles
its docstring claims to serve are structurally unreachable — what it actually
admits is box-like quadrilaterals, which is the shape class letterforms and
hatch tiles supply.

**One piece of good news the builders under-reported.** False dimension
proposals at the 0.5 default fell sharply with recall unchanged: 21.01 from 19
to 6, 3001 from 15 to 10, 11.01 from 18 to 12, 10.31A from 15 to 10. That is a
real precision win nobody wrote down.

**Verified clean and not to be disturbed:** F3, F5, F2's spine fallback, F12's
`_reach_on_ray`, all four ingest fixes (F7–F10), F14's packaging fix (verified
by building the wheel, inspecting it, installing into a clean venv and running
a fixture), and F6's truth walk (agreement to 5e-5 in across plain, rotated,
scaled, non-uniform, mirrored and nested placements).

### Round 3 — blocker round (in progress)

Directed at the six blockers plus eight majors, with the lead's judgment calls
made in advance rather than left to the builders: remove the escape hatch and
bar non-directional terminators from winning arbitration against directional
ones; rank candidates by soundness before distance; replace every new deletion
with a capped fallback; give the blunt criterion a real margin or scope its
claims down to what it demonstrably does; and correct the documentation, where
at least seven published claims are now measurably stale.

### Round 4 — closed by the lead (2026-09-07)

Two builders landed four of the five directed items before the round was
stopped for a machine restart; the lead finished the work directly.

**The blocker is closed.** Across skews of 0-25 degrees at both junk
distances, the dimension never claims the neighbouring leader's arrowhead,
nothing *called* ever claims it, and the real leader survives at every
angle. `on_spine` no longer feeds the soundness tier: identity is direction
and attachment only, and the spine is a statement about the coordinate.
Fill clusters and blunt terminators sit at the "evidence unobservable" rank,
so absence of evidence can no longer outrank a directional arrowhead.

**One over-application caught and corrected.** The builder applied the
projection at the shared attach level, which silently undid F2's *verified*
continuous-leg fix: a glyph chevron beside a 100 pt shaft went back to
publishing 108. Projection recovers only the lateral half of that defect.
The two legs now differ deliberately — the continuous leg publishes the
shaft's own end (its arrows sit inside), the split leg the projected apex
(its arrows sit outside) — and the reasoning is in the code.

**F6's suppression is now a cap on both legs,** with a correction the
original finding did not ask for: duplication now means the same SPAN, not
a shared endpoint. Two dimensions sharing a witness line is ordinary
drafting, and matching on loose defpoints was deleting real constructs for it.

**Measured on the close-out tree** — planlens 595 tests green (316 at the
start of this remediation), app drawing tests 58 green; corpus 25/25 and
16/16 at 0.3, 23/25 and 16/16 at the 0.5 default; leader false positives
2/10/6/19/23/20/38, all seven identical to baseline; the dimension precision
win preserved exactly at 21.01 5, 3001 10, 11.01 12, 10.31A 10.

**A fourth sharp edge, new and documented.** In the arrows-outside style a
continuous dimension drafted past 9.59 degrees reports its end at the shaft
tip rather than the defpoint — one arrow length short — and is capped below
the call threshold. Nothing is lost at `min_confidence=0.0`. The witness
line standing at the true defpoint is the evidence that would settle it;
using it requires reordering the confidence hoist that took sheet 3001 from
8.5 s to 0.4 s, which is not a change to make at the end of a four-round
remediation. Pinned by `TestSkewedDimensionDoesNotStealANeighbouringLeader`.

**The documentation was re-measured by the lead, not by an agent.** The
layer figures (557 / 158 / 1696 re-homed, and 3 to 2 distinct geometry
layers on two sheets) DO reproduce — the verifier's contradiction of them
was itself wrong. But the prose conflated two different quantities: the
layers *carrying geometry*, which the inheritance moves, and the `n_layers`
*metadata field*, which counts every layer name seen during ingest and does
not follow it. The blunt-terminator figures did NOT reproduce: the published
28 / 18 / 22 / 4 are actually **19 proposals at `min_confidence=0.0`, 17 at
0.3 and none at the 0.5 default**, from just two sheets.

So that this cannot happen a third time, the numbers now have a command —
`module_work/drawing_ground_truth/doc_claims_check.py`, beside the recall
scorer — and ten guard tests pin the published figures against it.

**Staging note (the release trap).** These six paths must enter the same
commit or the source tree breaks, because the tracked shims already delegate
to them: `planlens/testing/{__init__,leader_fixtures,construct_fixtures}.py`,
`planlens/tests/{test_packaging,test_readme_claims}.py`,
`planlens/ir/tests/test_text_bearing_scenes.py`.

*Remaining before a tag: the app pin requires planlens 0.2.0 to publish
first, which is the owner's sequencing call.*
