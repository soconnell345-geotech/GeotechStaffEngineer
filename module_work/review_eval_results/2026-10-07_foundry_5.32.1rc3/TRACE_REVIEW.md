# Foundry brief 4 (5.32.1rc3 + planlens 0.12.0rc1): every run read in full

The AI FDE ran brief 4 on Palantir Foundry on 2026-10-07 against the test
wheels **geotech-staff-engineer 5.32.1rc3** (app `a4ef417`) and **planlens
0.12.0rc1** (`08a1d53`). Two models, both through the FDE's Responses-route
wrapper with `detail: "original"` sent as AUTO and every `GEOTECH_VISION_*`
variable unset, so the app's own 2,048 px cap was in force:

| part | what | GPT-5.4 | GPT-5.6 Sol |
|---|---|---|---|
| A | location re-measure, no agent (23 calls) | run | run |
| B | `produce-circle-tags` + `produce-markup`, × 3 | run | run |
| C | the whole suite, `baseline` arm, 37 tasks | held by the owner | run |
| D | report ingest stage d (logs, lab, narrative) | run | — |

**Sources.** Raw hand-back (git-ignored):
`module_work/field_feedback/2026-10-08_foundry_brief4_5.32.1rc3/raw/`
(`part_a/`, `part_b/<model>/runs/<arm>/<task>/`, `part_c/sol/runs/baseline/<task>/`,
`part_d/RESULTS.md`, the operator's `review/` and `glue/`). Comparison runs:
run 4 (Sol, 5.32.0rc2/rc3, full resolution, same wrapper) in
`module_work/field_feedback/2026-10-04_foundry_evidence/raw/brief3/out_sol_532/runs/`;
live check 1 (GPT-5.4 on Funhouse, 5.32.0) in
`module_work/review_eval_results/2026-10-07_funhouse_5.32.0_check1/TRACE_REVIEW.md`
(cited below as **check 1**). The suite's own tables are next to this file:
`RESULTS_partB_gpt54.md`, `RESULTS_partB_sol.md`, `RESULTS_partC_sol.md`.
Part D's RESULTS.md stays in `raw/` (private report data); this review quotes
only its report IDs, counts and rates. Code is cited at `a4ef417` (planlens
at `08a1d53`).

**Method.** I read every event of all 49 agent runs (12 in part B, 37 in
part C: 3,826 records), every vision side call's own text, every
`annotate_document` call with its check verdicts, and every final answer.
Scripts then:
* placed every ring written in part B and C against the fixture's truth
  (`build_synthetic_tag_set()`), with the view each ring's box came from;
* compared every part C task with its run-4 twin (outcome, tokens, time,
  model calls, tiles, zooms);
* read the produce PDFs (rings) and the operator's arrow table and images
  (`review/markup_arrows.csv`, `review/markup_arrows_png/`), and the four
  Word files part C produced;
* replayed `find_like` offline on the fixture with the exact example boxes
  the two Sol runs used, on both the numpy and the OpenCV matcher.

`Lnn` is a line of that run's `activity.jsonl`, `t=` seconds into the turn.

---

## In short

**Verdict: release 5.32.1 from `a4ef417` as measured.** Nothing found here
is a regression from 5.32.0 that needs fixing first; the defects below are
older (most since 5.32.0 or earlier) or cost time rather than correctness.
The one cherry-pick worth considering is unrelated to this run's findings:
the activity-log lock (§6).

1. **The location fix works on both models.** Pixel boxes came back on every
   whole page within a few points (GPT-5.4 repeats 2–3: 5.1 and 3.5 pt
   median; Sol: 1.1–1.5 pt), against 28.6 and 8.8 pt for the old 0-999
   prompt on the same image. Every one of the 1,574 boxes the vision calls
   wrote in parts B and C was a `px=` box; none fell back to the grid.
2. **Every ring was placed from a zoom,** and every padded zoom in part B
   held its target. GPT-5.4 passed the circle task 2 of 3 (it failed 3 of 3 on
   Funhouse with 5.32.0 and with rc1); Sol 3 of 3, plus part C's run.
   No "zoom on the thing first" refusal fired because no agent ever tried to
   anchor a mark on a view wider than 300 pt (§2.3).
3. **GPT-5.4's one failure (3 of 7 tags)** had three causes, none of them a
   position: the rotated tag T5 was found only by a tile and never pursued;
   T3 and T4 were dropped after hedged readings ("[G/C]CE", "G[C/O]E") on
   zooms the agent itself had made small with `dpi`; T6 was dropped after
   the markup check twice rejected a ring that did enclose it (§2.3).
4. **The markup check cannot confirm a review comment.** Of 10 checks on
   comments, boxes and highlights placed on the ramp-slope note, none was
   confirmed and 8 of the 10 "misplaced" verdicts were wrong — the comment
   was on the right line. The check compares what is under the mark with
   the comment's own words ("Please confirm the 8.33 % …"), not with the
   thing the comment is about. Every agent then switched to a quote or a
   sticky note, which the check does not look at (§2.4). On rings it is
   mostly right: 66 of 70 good rings confirmed, 4 rejected when a leader
   stroke touched the tag.
5. **The suite passed a comment on the wrong note.** GPT-5.4 baseline's
   arrow points at the section label "RAMP SLOPE UP TO 7.5 % (8.3 % MAX.)";
   the check only asks for a comment containing "8.33". A general fix is in
   §2.6.
6. **Part C: 37 of 37, no task changed outcome.** Input tokens are down 10 %
   on the 35 original tasks (4.09 M against 4.53 M), output up 38 %, time up
   13 % once a single 15-minute hung call is set aside. Tiling now fires on
   90 of 100 page looks (1 of 120 in run 4). The two newer tasks cost more
   (§3.2).
7. **An agent-set `dpi` quarters zoom resolution.** 45 of 106 zooms in parts
   B and C carried a `dpi`; those rendered at a median 4.2 px per point
   against 15–17 px per point without it (§3.6). A given `dpi` can only lower
   the budget's size, never raise it.
8. **`find_like` ran three times (Sol only).** Once it was perfect in 46 s;
   twice its example was the tag that sits on a heavy grid line, which floods
   the matcher (the same on OpenCV, checked offline) and cost 180–289 s and
   18–20 vision calls (§3.5).
9. **Part D** improved where the fixes aimed (narrative 83 % recall / 85 %
   precision; `siteResponseMention` 8/8). Gradation's 80 % → 37 % is a LINK
   failure on four sheets, not values dropped by the merge; the logs' index
   values 44 % → 0 % (blind) are the same as run 1 and the run files hold
   what would settle it (§4).
10. **The record is complete but silent on "why".** No lost or glued lines
    in 3,826 records, every tool call ended. But the primary agent's text on
    tool-calling steps is empty in all 49 runs, and no reasoning summary is
    requested on the Responses route (§5).

---

## 1. Part A — the location re-measure (no agent)

### 1.1 The numbers

Median / max error against the fixture, pt; y scale fitted per answer.

| look | GPT-5.4 | Sol |
|---|---|---|
| page as shipped, repeat 1 | 19.2 / 22.6, y 1.015 | 1.1 / 123.7, y 0.960 |
| page as shipped, repeat 2 | 5.1 / 11.8, y 1.017 | 1.1 / 121.7, y 0.959 |
| page as shipped, repeat 3 | 3.5 / 8.7, y 1.014 | 1.5 / 122.1, y 0.955 |
| old 0-999 prompt, same image | 28.6 / 71.2 | 8.8 / 13.0 |
| tool tiles (9 fired) | 1.1–3.5 per tile | 0.1–0.4 (one 90.7, see §1.3) |
| zoom on the page box, T6 / T4 | 0.6 / 1.3, window holds the tag | 0.2 / 1.0, window holds the tag |
| 80 pt windows, T4 / T7 | 0.4 / 0.4 | 0.1 / 0.1 |

`view_px [2048, 1325]`, `detail original`, `tiles: 9` on both; zoom padding
122.4 × 79.2 pt. GPT-5.4 used 22.3 K input tokens in 32.5 s, Sol 25.2 K in
118.7 s. Probe: GPT-5.4 `high` 2,664 / `original` 4,260 image tokens; Sol
692 / 4,916, so AUTO delivered the full 2,048 px on both and no host edge
was found (`max_edge: null`): the 2,048 px cap on Foundry is the app's
choice, not the host's.

### 1.2 GPT-5.4 repeat 1: was going on right?

**Yes.** The operator's own gate ("every repeat under 10 pt") is stricter
than the brief's ("tens of points off — stop"), whose purpose was to catch
the 57–91 pt grid shrink of check 4. Repeat 1 is not that:

* Its y scale is 1.015, the same as repeats 2 and 3. Nothing is shrunk.
* Five of its six boxes are 71–101 px wide, against 12–28 px in repeats
  2–3: the model boxed **the tag and its leader**. Their centres therefore sit 15–20 pt
  left of the tag (fitted x offset −36 px).
* Four of its six boxes contain the whole tag. The other two (T4 and T7,
  whose leaders run down-left) cover the leader and arrowhead and stop just
  below the tag.
* It also left out T5, the tag drawn turned 90°.

**Model, prompt or scorer?** All three, in that order. It is the model:
three identical calls, one answered with leader boxes. The question invites
it ("every GCE penetration tag **that has a leader drawn from it**") even
though `PIXEL_INSTRUCTION` says "Box the thing itself, tightly"
(`vision_view.py:145-151`). And the scorer measures centre distance, which
counts a box that holds the tag as 15–20 pt off. For the product it is
harmless: a whole-page box is never used to place a mark (the precision line
and planlens' refusal both say zoom first), and the padded zoom window
(±122 × 79 pt) holds the tag from any of these boxes.

### 1.3 Sol's ~122 pt outlier and y scale 0.955–0.960: no shrink

Every Sol repeat lists **eight** GCE boxes. The eighth,
`px=[160, 741, 181, 753]`, is the **QCE look-alike** at (97–107, 444–448 pt),
to within 1 pt. The scoring cell only matches a box labelled GCE to a GCE
tag, so it paired it with T2, 122 pt away, and that single false pair drags
the fitted y scale to 0.96. Refitted without it, Sol's pixel boxes have
slope 0.997–1.001 in y and 0.998–0.999 in x, offset under 1 px: **exact**.
The seven true tags are all within 2.8 pt. What the outlier does show is a
**reading** error at 2,048 px — Q read as G in 3 of 3 repeats, but not in the
old-prompt answer. In the agent runs that look-alike was caught by tiles,
zooms or the markup check every time it was considered (§2.2).

The same artefact explains the Sol tile r3c1 maximum of 90.7 pt: that answer
boxed the GCG callout cut off at the tile's top edge and called it GCE.

### 1.4 A small residual in GPT-5.4's pixel boxes

GPT-5.4's pixel boxes are stretched in y by **1.4–1.7 %** with no offset,
and not at all in x:

| image sent (px) | fitted y slope | height rounded up to a multiple of 32 px, ratio |
|---|---|---|
| 2048 × 1325, whole page, repeats 2–3 | 1.017, 1.015 | 1344 / 1325 = 1.014 |
| 2048 × 1424, tile r2c1 | 1.011 | 1440 / 1424 = 1.011 |
| 2048 × 1303, zoom on T6 | 1.006 | 1312 / 1303 = 1.007 |
| 2048 × 1229, zoom on T4 (3 points) | 1.024 | 1248 / 1229 = 1.016 |

x is 1.000 because 2,048 is already a multiple of 32. That pattern fits
GPT-5.4 working in a frame whose sides are rounded up to whole 32-px patches.
Sol shows nothing of the kind. The effect is up to ~10 pt at the bottom of
an 11 × 17 sheet (T7's 8.7 and 11.5 pt in repeats 2–3 are almost exactly
this) and under 4 pt in a 300 pt zoom. It is small next to what it replaced,
and it is a hypothesis from four images. A cheap test and a general remedy
are in §7 (item K).

### 1.5 What the answers themselves say

* GPT-5.4 tile r1c1 answered "GCE" for an FBG, a GCG and a turned FBG: the
  cell's format ("GCE <box>") forced a label onto every tag in a tile with
  no GCE. In the agent runs, with the agent's own wording, the same tile
  named them FBG and GCG correctly.
* Sol's zoom on T6 read the turned GCG as "OOO"; GPT-5.4 read it GCG.
* GPT-5.4's 80 pt windows give `px=[246, 804, 424, 877]` on a 1334 px image:
  a box of 10.7 × 4.4 pt round a 10.2 × 4.3 pt tag. At that size both models
  are exact.

---

## 2. Part B — circling the tags, and the markup comment

### 2.1 Truth, briefly

Sheet 1 of the tag fixture: seven GCE callouts T1–T7 (centres as in check 1
§1). **T5 is turned 90°.** **T1's lettering sits on a heavy vertical grid
line**, and on T1 and T6 the leader's horizontal shoulder ends a stroke
width from the "G". Look-alikes: GCG × 6 (one turned 90° near T6), QCE, GPE,
FBG; one GCE with no leader; the legend row.

### 2.2 The circle task, run by run

| run | rings on target (suite) | where the final rings' boxes came from | annotate calls | check: confirmed / misplaced (of which wrong) | in / out tokens, time |
|---|---|---|---|---|---|
| GPT-5.4 baseline | **3/7** ✗ (precision 3/3) | zooms 30–46 pt (`dpi` 600–900) of zooms 100–144 pt off **tile** boxes | 5 (1 `color` error) | 8 / 9 (3 wrong) | 348 K / 7.1 K, 159 s |
| GPT-5.4 r2 | 7/7 ✓ (8 rings: T6 twice) | zooms 94–110 pt off tile boxes | 2 (1 `color`) | 8 / 0 | 206 K / 4.9 K, 105 s |
| GPT-5.4 r3 | 6/7 ✓ (T5 missing) | zooms 100–107 pt off tile boxes; T6 re-zoom 42 × 24 pt | 2 | 11 / 1 (1 wrong) | 238 K / 5.2 K, 117 s |
| Sol baseline | 7/7 ✓ | zooms 94–107 pt off tile boxes | 2 (1 `color`) | 7 / 0 | 164 K / 18.0 K, 157 s |
| Sol r2 | 7/7 ✓ | first try: padded zooms 169–298 pt off whole-page boxes; final: `find_like`'s page boxes | 3 (1 `color`) | 12 / 3 (0 wrong) | 252 K / 19.9 K, 219 s |
| Sol r3 | 6/7 ✓ (T5 missing) | padded zooms 236–261 pt off whole-page boxes | 3 (1 `color`) | 13 / 0 | 386 K / 57.9 K, 342 s |
| Sol, part C | 7/7 ✓ | zooms 59–117 pt round `find_like`'s hits | 1 | 7 / 0 | 286 K / 88.6 K, 428 s |

Every ring that was meant tightly for a tag landed **0.1–3.8 pt** from the
tag's centre (0.1–0.5 pt for Sol and GPT-5.4 r2). The exceptions are not
position errors: GPT-5.4 baseline's first rings were drawn round tag and
leader on purpose (centres 10–24 pt from the tag), and three of Sol r2's
first rings went round a leader's arrowhead (twice, 20–26 pt from the tag)
and round the QCE — the side call had boxed the wrong thing. No ring
anywhere came from a whole page or a tile.

**GPT-5.4 baseline (L1–L172).**
1. t=4–25: `analyze_pdf_page(tiles="auto")`. The whole-page answer lists six
   GCE with tag-plus-leader boxes and misses T5. Tile r2c1 (L37) lists T5 as
   "vertical GCE callout … px=[1399, 833, 1533, 983]" and flags the turned
   GCG as "GC[G/C] … not a definite GCE".
2. t=29: six zooms, `dpi: 300`, each on a **tile** box (views of 100–144 pt,
   rendered 418–600 px wide). All six windows hold their tag. T5 is not
   among them: the agent took the whole page's six and used the tiles only
   for their boxes.
3. t=37: annotate #1 fails: `unknown markup field(s) ['color']`.
4. t=40: annotate #2: six rings from the zooms' "callout extent" boxes
   (tag + leader): 41–81 × the tag's area. Check: five "N times the area of
   the thing", one "arrow leader line" — **right**: these are blankets.
5. t=73: six re-zooms at `dpi: 600` (30–46 pt windows, 300–380 px images).
   T3 comes back "**[G/C]CE** … I would not confidently confirm it", T4
   "**G[C/O]E**".
6. t=77: annotate #3 with four rings (T1, T2, T6, T7), **T3 and T4 left out**.
   Check: T2, T7 confirmed; T1 "encloses: false, inside GCE" and T6
   "encloses: false, inside **- GCE**" — **both wrong**: the rings, 18 × 14
   and 24 × 14 pt, are 1.6 and 2.3 pt from the tags' centres and contain the
   whole tag.
7. t=98: re-zooms on T1 and T6 at `dpi: 900`; annotate #4: T1 now confirmed,
   T6 again "**-GCE**", again wrong (0.9 pt off).
8. t=134: annotate #5 with T1, T2, T7 only: all confirmed. Answer: "The
   delivered copy has 3 verified GCE circles". Honest, and short of the job.

**GPT-5.4 r2.** The whole page found eight (T5 included, plus the turned
GCG); eight zooms off tile boxes (no `dpi`, 1,600–1,800 px images, 16.7 px
per pt). Two of the zooms (one off tile r2c1, one off r3c1) resolved to T6,
and the agent ringed **T6 twice**. Check confirmed all eight. The answer
says "8 GCE circles … the annotation check confirmed all 8": the sheet has
seven; the duplicate was never noticed.

**GPT-5.4 r3.** As baseline in step 1: the whole page listed six and missed
T5; tile r2c1 named T5 ("vertical text near the vertical grid line: GCE.
px=[1492, 832, 1517, 897]", L35); the agent zoomed the six. Rings from
"tag plus immediate leader shoulder" boxes, 10–30 × the tag's area: five
confirmed (the 30 × one at the limit), T6 "**— GCE**", encloses false
(4.6 pt off, wrong). A re-zoom on the ring's own box fixed T6. Final 6/7;
answer: "I circled … the 6 GCE penetration callouts" — as if complete.

**Sol baseline.** The whole page listed eight (the eighth is the QCE);
tiles gave the true seven; eight zooms off tile boxes, each answered
"Confirmed text: **GCE**" without hedging (one, at the tile edge, "GCG").
Seven rings, seven confirmed.

**Sol r2.** Sol asked for "the **circular** GCE tag (not its leader)" — the
tags are plain lettering. It zoomed on whole-page boxes, padded to windows
of 169–298 pt, and passed `dpi: 300` (images 965–1,240 px, 4.2 px/pt). Four
of eight answers said there is no circle and gave no box; two boxed the
leader's **arrowhead** as the "tag". Annotate #2: the check rejected exactly
the three bad rings ("black arrowhead", "mouse cursor arrow", "QCE") —
**right**. Then `find_like` with T3 as example (46 s, one vision call):
seven callouts, exact boxes. Seven rings from those boxes, all confirmed.

**Sol r3.** The same "circular" premise blinded the tiles: all nine said "no
qualifying circular tags". The whole page missed T5 and listed a GCG
(574–585, 185–190) as GCE. `find_like` with T1 as example ran 180 s (§3.5)
and was ignored. Sol then zoomed the whole page's six candidates: the
window aimed at the GCG (259 × 169 pt) also held T1, and the side call
answered about **T1** ("Yes — exactly GCE"), so T1 was ringed twice. A
tight re-zoom (L158) read the candidate as GCG and annotate #3 dropped the
duplicate. Final 6/7; T5 never seen.

**Sol, part C.** First zoom off tile r1c2, then `find_like` with T1 as
example: 289 s and 18 vision calls, nine "instances" (the seven, the bare
GCE, and a 7 × 3 pt fragment of T1's leader shoulder labelled a callout).
Sol made its own 59–117 pt zooms round the hits and ringed the seven. The
answer names the QCE as excluded.

### 2.3 The four questions

**Where did each ring's position come from?** From a zoom in every run —
`render_region` views of 30–298 pt — or, once, from `find_like`'s
image-matched page boxes (Sol r2's final rings). GPT-5.4 and Sol baseline
zoomed off **tile** boxes; Sol r2/r3 off **whole-page** boxes through the
new padding (windows ~250–300 pt).

**Did the agents zoom first?** Yes, all seven circle runs: each zoomed
every candidate before its first `annotate_document`. On the markup task
GPT-5.4 (×3) and Sol r3 zoomed on the note before their first comment; Sol
baseline, r2 and part C went straight to a `quote` anchor, which needs no
zoom. The precision line did its
job: every whole-page and tile result said "NOT for placing a mark — zoom
until the thing is legible in a view of 300 pt or less", and no agent tried.

**Why did no "zoom on the thing first" refusal fire?** Because nothing gave
it a reason. planlens refuses a small mark only when its `view` is wider
than 300 pt; the widest view any ring used was 298 pt (Sol r2's padded zoom
on T3). The refusal also never sees a typed `bbox`/`page_bbox` (Sol r2's
final rings were typed, from `find_like`, and exact). So the refusal remains
tested offline only. One forward note: on a larger sheet the padded zoom of
a whole-page box is itself wider than 300 pt (10 % of a 2,592 pt ARCH E
sheet is 259 pt each side), so the agent will always need a second zoom
there before it may mark; the precision line already says so.

**Why did GPT-5.4 baseline fail recall (3 of 7)?** Not position: every
tight ring it wrote was within 3.8 pt of its tag (its first, blanket rings
went round tag and leader on purpose). Four tags were lost, in three ways:
* **T5, never pursued.** Only a tile found it; the agent's candidate list was
  the whole page's six. The same happened in r3 (and to Sol r3, whose tiles
  were blinded by its own prompt). The tool returns the tiles nested under
  the whole-page answer; nothing says "the tiles found one the page did not".
* **T3 and T4, dropped after hedged readings.** The agent's own `dpi: 600`
  rendered 30–46 pt windows at 300–380 px; GPT-5.4 wrote "[G/C]CE" and
  "G[C/O]E" under `READING_INSTRUCTION` ("write the alternatives in brackets
  … rather than picking one", `vision_view.py:129-133`). GPT-5.4 hedges even
  at 70 px lettering (r2: "[G/C/O][C/E]E"; r3: "[G/C][C/G]E[F] … likely
  intended as GCE") — r2 and r3 kept such tags, baseline dropped them.
* **T6, dropped after two wrong check verdicts.** The check's side call took
  the end of the leader shoulder as part of the thing ("- GCE") and said the
  18–24 pt ring did not enclose it. The ring contained the whole tag both
  times.
* One round lost to `color`, one to blanket rings.

GPT-5.4 stopped after the check confirmed the three it kept: it did not try
again for T3, T4, T5 or T6. That is better than check 1 (where it stopped after two
rejected attempts with nothing confirmed), but the same habit.

### 2.4 The markup check, measured against truth

| marks checked | confirmed | rejected rightly | rejected wrongly |
|---|---|---|---|
| rings that enclosed their tag within 30 × its area (all runs) | **66 of 70** | — | 4 (T1 once, T6 three times; leader stroke touching the tag) |
| rings that were blankets or on the wrong thing | 0 | 9 of 9 | — |
| comments, boxes, highlights on the ramp-slope note | **0 of 10** | 2 (on the wrong note) | **8** (on note 4's line) |

* **Rings.** No confirmed ring was wrong (two were duplicates of a right
  one). The four wrong rejections were all tight (18–32 × 14 pt) rings on
  T1 and T6; the side call's "inside" text ("- GCE", "— GCE") shows it
  counted the leader's end as part of the tag. The check measures size
  geometrically from its own box (`_too_wide`, `markup_check.py:92-122`) but
  still takes "encloses" as the model's yes/no (`:212-223`).
* **Comments.** For a callout, box or highlight the check asks whether the
  mark is on what its **label or comment** names (`_expected`,
  `markup_check.py:140-142`; `_PROMPT`, `:49-58`). A review comment's text
  is a request ("DRAFT: Please confirm the 8.33 % maximum …"), not the name
  of a thing, so the side call answers false whatever is there: "E", "T",
  "N" (a letter at a callout's tip, `_crop_box` `:145-155`), "nothing", and
  twice, for a box drawn exactly round the line, "**RAMP SLOPE CANNOT EXCEED
  8.33 % MAX**" — the right text, judged not to be what the mark "names".
* **The route around it.** Quote anchors and sticky notes are not checked
  (`CHECK_KINDS`, `:29`; `GEOMETRY_ANCHORS`). After two or four "misplaced"
  verdicts, every agent that had tried a checked anchor re-placed the
  comment by `quote` or as a `note`, and the check went quiet. In GPT-5.4
  baseline that is how a comment on the **wrong** note — which the check had
  rejected twice, rightly — reached the file.

### 2.5 The markup comment, run by run

| run | how the final comment is anchored | on note 4's bottom line? | before that |
|---|---|---|---|
| GPT-5.4 baseline | callout by `quote` "RAMP SLOPE UP TO 7.5% (8.3% MAX.)", tip (491.8, 434.4) | **no — the section label** | 2 callouts off zooms on the section label, check "7.5%" / "nothing" (right) |
| GPT-5.4 r2 | callout by `quote` "RAMP SLOPE CANNOT EXCEED 8.33% MAX.", tip (60.0, 227.4) | yes | 2 callouts on note 4's line, check "E" / "T" (wrong) |
| GPT-5.4 r3 | sticky note by `quote`, at (60, 227) | yes | 2 callouts on the line, check "T" / "N" (wrong); kind `text` refused |
| Sol baseline | sticky note by `quote` | yes | kind `comment` refused |
| Sol r2 | sticky note by `quote` | yes | field `anchor` refused |
| Sol r3 | sticky note by `bbox`, at (143, 227) | yes | callout ("nothing"), box ×2, highlight: all on the line, all rejected (wrong) |
| Sol, part C | callout by `quote`, tip (60.0, 227.4) | yes | — |

The rc3 quote fix works: every quote on "RAMP SLOPE CANNOT EXCEED 8.33% MAX."
landed on its own printed row (y 224–230), where 5.32.0 put it 50–80 pt
higher. One cosmetic point from the images: the callout's comment box sits
over note 3's text.

**Why GPT-5.4 baseline chose the section label.** Its whole-page look
answered "the ramp slope note … in the section view: RAMP SLOPE UP TO 7.5%
(8.3[3/8]% MAX.)"; three tiles then named note 4, the slope "B" row and the
detail's "(8.33% MAX)" labels. The agent followed the whole-page answer —
the same pattern as T5 above. The question is also genuinely loose: the sheet
has four places that state a maximum ramp slope, and only one of them reads
8.3 %.

### 2.6 A general fix to the suite's markup check

`produce-markup` checks `pdf_markups` with `text_contains: "8.33"`
(`review_eval/tasks.py`, the `produce-markup` task;
`review_eval/checks.py:305-341`): any comment mentioning 8.33 anywhere on the
page passes. The general fix is the comment-mark counterpart of
`markups_on_targets`:

* **A new check type, `markups_point_at`** (`review_eval/checks.py`): read
  the produced PDF's markups with planlens (as `check_pdf_markups` does),
  keep those that match `text_contains`, and take each one's **anchor**:
  a callout's arrow tip (`points_at`, which planlens already reads from the
  `/CL` line), a sticky note's icon, a highlight's or box's rectangle. Pass
  when the anchor lies on, or within `pad` points of, one of the task's
  **target boxes**. Report recall and the distance per markup, as
  `markups_on_targets` does.
* **Targets as task data, measured once** (`review_eval/tasks.py`): the
  printed line boxes of every acceptable target, in displayed-page points,
  measured independently of the code under test (from the DWG text or by
  hand off the rendered sheet), not from planlens' own quote anchoring.
  For 10.31A that is note 4's second line (shown y 224–230); whether the
  slope "B" row also counts is the owner's call; the section label "(8.3 %
  MAX.)" should not.
* Nothing in it is about this sheet: any future produce task ("comment on
  the dimension that …", "flag the title-block date") gets a target list the
  same way. The 10.31A target box should be checked against the operator's
  rendering (`review/markup_arrows_png/`).
* Optionally tighten the question to name note 4, or keep it loose and let
  the target list say which notes count — the owner's choice; a loose
  question is a fair test of "which note did you mean".

---

## 3. Part C — the whole suite against run 4

### 3.1 Outcomes

37 of 37 (130/130 checks); run 4's baseline also passed all 37. **No task
changed outcome.** I read every answer beside its run-4 twin: the same facts,
in places fuller (bioretention section adds the 3H:1V and 0.5 % slopes; the
monument adds the brass plate's 4 in. dowel; calc-wall-check adds that the
"Bearing OK" has no capacity behind it). Two answers carry the same
check-lenient error as run 4:

* `set-long-rare-tag`: pages 4, **9**, 12, 20 (run 4: 4, 12, **19**, 20).
  Page 9 holds a **turned FBG**, which Sol read "FPG" in a 122 × 87 pt zoom at
  16.7 px/pt; page 19's look-alike was read correctly this time. Same
  precision (0.75), a different look-alike.
* `set-3600-psi`: names 30.01 while saying its 3600 is cubic feet per acre;
  the check counts the page as named (precision 0.75), exactly as in run 4.

### 3.2 Tokens and time

| | run 4 (rc2/rc3, full size) | this run (rc3, 2,048 px) |
|---|---|---|
| 35 original tasks: input / output tokens | 4.53 M / 0.36 M | **4.09 M / 0.50 M** |
| 35 original tasks: minutes | 45.5 | 66.5, of which one hung call 15.1 → **51.4** |
| `produce-circle-tags` (one run) | 110 K / 11 K, 81 s (r2: 779 K, 1,220 s; r3: 204 K, 166 s) | 286 K / 89 K, 428 s (`find_like` 289 s) |
| `set-long-rare-tag` (one run) | 442 K / 24 K, 104 s (r2/r3 similar) | **997 K** / 44 K, 173 s |
| model calls, all 37 | 540 | 977 |

Where it went (all 37 tasks, vision side calls by tool):

| | run 4 | this run |
|---|---|---|
| primary agent | 205 calls, 2.90 M in | 180 calls, 2.67 M in |
| `analyze_pdf_page` (page + tiles) | 229 calls, 1.72 M in, 235 K out | 584 calls, 2.24 M in, 450 K out |
| probe | (within the above) | 140 calls, 281 K in |
| `render_region` | 84 calls, 419 K in | 44 calls, 117 K in |
| `find_like` reads | — | 18 calls, 47 K in, 74 K out |

Per task (input K, run 4 → this run): see the table in §8. The biggest
swings: `set-long-rare-tag` 442 → 997 (24 sheets × 9 tiles), `produce-circle-tags`
110 → 286 (`find_like`), `set-sheet-index` 644 → 297 and `set-find-bioretention`
278 → 123 (fewer page looks, each at 2,048 px instead of 3,600–6,000), `fixture-markups` 167 → 28
(it no longer looks at the page; the markup data answers it).

* **Time.** 15 minutes of the increase is one `render_region` side call in
  `set-cross-references` that hung for 903 s (L188) after a 500 from the
  model service (L68); both are host faults. The rest is tiles: Sol writes
  5,000–9,000 output tokens on some tiles of the Mecklenburg sheets and they
  take ~100 s each.
* **The probe runs once per task** here (every task is its own process on
  Foundry), now five calls: 140 calls, 5 % of input. In the app it runs once
  per process. Not a product cost.

### 3.3 Tiling

Run 4 tiled **1** of 120 page looks (9 tiles). This run tiled **90 of 100**
(485 tiles): 2 × 2 on every Mecklenburg sheet (792 × 612 pt at 2,048 px:
lettering below 12 px), 3 × 3 on the 11 × 17 tag sheets, none on the UFC and
calculation pages. That is the rc3 change working as designed ("tiling works
from the size actually received"). On this suite it bought no score and cost
time and output tokens; on the long tag set it more than doubled the input.
Worth recording in FUTURE_IDEAS "VISION EFFICIENCY" — robust first, as the
owner asked, but now measured.

### 3.4 Pixel boxes, precision lines, padded zooms

* **Pixel boxes:** 327 of Sol's 628 page/zoom answers in part C gave
  boxes — 1,223 boxes, every one `px=`; 107 tool results carry the note that
  the boxes were converted in code. Parts B: GPT-5.4 214 of 214, Sol 137 of
  137. No answer fell back to the 0-999 grid.
* **Precision lines:** on 143 of 144 page and zoom results; 115 said the
  view was too wide to mark from. Only the two produce tasks place marks;
  both obeyed.
* **Padded zooms:** 24 in part C, 4–8 per circle run in part B. Windows off a
  whole page are ~250–300 pt; off a tile ~95–145 pt. Every one in part B
  held what it was aimed at (checked against the fixture); part C has no
  box-level truth, and in every trace its zoom answers read what they were
  sent to read. Two side effects, both in part B: a 259 pt window holding
  two candidates was answered about the wrong one (Sol r3, T1 ringed twice),
  and Sol r2's padded windows rendered at the agent's `dpi: 300`.

### 3.5 `find_like`

Offered in every run (numpy matcher; `review/find_like_offered.json`).
**Used three times, all by Sol on the circle task; never by GPT-5.4; not on
`set-long-rare-tag`.**

| run | example (from) | candidates | vision calls, time | result |
|---|---|---|---|---|
| Sol B r2 | T3, from a 298 pt zoom | 43 | 1, 46 s | 7 callouts + the bare GCE, exact boxes; used for the final rings |
| Sol B r3 | T1, from a 258 pt zoom (6.2 pt tall) | **400 (the cap)**, warning returned | ~20, 180 s | "found": T1 and five 3 × 6 pt fragments called GCE callouts; ignored by the agent |
| Sol C | T1, from a 59 pt zoom (4.8 pt tall) | 367, no warning | 18, 289 s | 9 "instances": the seven, the bare GCE and a 7 × 3 pt leader fragment |

**Replayed offline** on the fixture with the same example boxes, planlens
`08a1d53`, both matchers. The replay reproduces the live runs exactly:
the T3 example gives 43 candidates with all seven GCE callouts; Sol C's T1
box gives 367 with all seven; Sol r3's T1 box (a point taller) gives 400 and
only one of the seven; even T1's exact true box gives 400 and two. **Numpy and
OpenCV agree candidate for candidate**, so the new matcher is not at fault.
The cause is the example: T1's lettering sits on a heavy grid line (§2.1),
so the template is mostly that line and matches grid ticks everywhere, and a
fraction of a point in the box decides between 367 and 400 candidates. The
app wrapper (`funhouse_agent/find_like.py:106-235`) then reads every
candidate's contact sheet, 20 to a sheet, whether or not planlens warned.
Fixes in §7 (H).

### 3.6 An agent-set `dpi` makes zooms smaller

| | zooms with `dpi` | px per pt (median, range) | zooms without | px per pt |
|---|---|---|---|---|
| GPT-5.4, part B | 17 | 4.2 (2.8–12.6) | 18 | 16.7 (7.6–16.7) |
| Sol, part B | 10 | 4.2 (4.2–7.0) | 17 | 16.7 |
| Sol, part C | 18 | 4.2 (2.9–5.6) | 26 | 14.9 |

planlens lowers a given `dpi` if it would overshoot the budget but never
raises it (`dpi = fit_dpi if dpi is None else min(float(dpi), fit_dpi)`,
`planlens/document/document.py:1030`), and the app passes the agent's value
through (`vision_tools.py:779, 807`; `deep/tools.py:850-885`, whose own
description says `dpi` "is chosen automatically"). Agents ask for 300–900 dpi
believing it sharpens the image; on a 100 pt window that is a quarter of
what the budget would give. Not new in 5.32.1 (the budget arrived in 5.27),
but it fed GPT-5.4's hedged readings above.

---

## 4. Part D — report ingest stage d, against FOUNDRY RUN 1

GPT-5.4, 85 calls, 1.10 M input / 149 K output tokens, pilot 261 s + full
1,574 s, no retries.

| reader | set | FOUNDRY RUN 1 (5.31.0) | this run (5.32.1rc3) |
|---|---|---|---|
| narrative | 8 reports | recall 75 %, precision 77 % | **83 %, 85 %** (open 84/87, blind 83/85) |
| | `siteResponseMention` / `soilCorrosion` | 0/8, 2/6 | **8/8, 5/6** |
| | `propertyType` / `recommendedFoundations` | 3/8, 3/8 | 8/8, 4/8 |
| lab | open 16, link | 55 % | 61 % |
| | open 16, before → after | 83 % → 46 % | 83 % → 51 % |
| | blind 15, before → after (link) | 55 % → 82 % (100 %) | 55 % → **78 %** (98 %) |
| logs | open 6 | 91 % → 98 % | 91 % → 98 % |
| | blind 9 | 73 % → 73 % | 73 % → **78 %** |
| | blind recovery / index / n_value | 53 % / 0 % / 84 % | **100 %** / **0 %** / 96 % |

What moved and why:
* **Narrative:** the conventions re-read from the owner's keys (`8e428ce`)
  fixed exactly the two fields they aimed at. Still weak: `earthHazardsExposed`
  0/3 (all three wrong), `structureList` 3/7 (+1 invented), `structureCount`
  3/8, `strata` and `recommendedFoundations` 4/8, `figureCount` and
  `bearingCapacity` 5/8; list recall 51 %.
* **Logs recovery 53 → 100 %** is the scorer change in rc3 (`46be026`,
  recovery printed as a length now counts), not a reader change.
* **Lab blind fell 82 → 78 %** with link still 98 %: per kind, chemical
  (54 %), compaction (1 of 10) and one consolidation sheet (R36_p56, 10 %) are
  the low ones. Not explained by the RESULTS alone.

### 4.1 Gradation: before 80 %, after 37 %

* The two columns measure different things (`lab_scoring.py`): **before**
  asks only whether each true number appears somewhere in the page's
  detected tables; **after** asks whether it sits in the right slot of a
  test **linked** to the right hole and a depth within 0.15 m
  (`_tests_near`, `:713-725`). A specimen whose link misses loses every
  value on it.
* Four of the nine gradation sheets collapsed to 1–5 values: R15_p82 (blind,
  28/28 → 1/30), R17_p114 (46/46 → 5/56), R28_p176 and R28_p177 (→ 1/37 each).
  "Everything but `kind`" is the signature of a **missed link**, not of
  values dropped by the merge: a merge drop would leave the link and lose
  some values.
* FOUNDRY RUN 1 found `after == model_alone` on every lab sheet, i.e. the
  merge did not cause the open-set collapse. The hole-ID folding (`5e70823`,
  "LB-2" = "LB2") is in rc3 and lifted the open link only 55 → 61 %. What
  remains is something the folding does not cover: a depth printed as an
  interval or in feet that misses the 0.15 m window, several specimens on one
  sheet linked to one hole, or a hole label folding cannot match.
* **So:** not a scoring artefact in the sense of a wrong check (a gradation
  tied to the wrong sample is wrong in the record and in DIGGS), but the
  80 → 37 contrast overstates it, because before has no link requirement.

### 4.2 Logs index: 44 % before, 0 % after (blind)

* Unchanged from run 1 (7/16 → 0/16), while the open set's index is 100 %
  (32/32): the scorer can see index values on samples when they are there.
* **The scorer's two columns again differ** (`log_scoring.py`): before
  counts a number in the right column family inside the depth window
  (`:400-410`); after needs it on a record `Sample`, in the named attribute
  (`water_content`, `liquid_limit` …), at a sample whose top is in the
  window (`:636-655`).
* **The merge cannot drop a value the floor held** if the model omits it
  (`floor.settle`: floor only → kept). It can **replace** it: wherever the
  model returns a different value, the model wins if its object's
  provenance has a box and a note (`has_evidence`, `floor.py:117-145`, is
  judged per object, not per value, and passes for any model object with a
  box and a note, or a picture-read one with a note). The reader asks for a note on every value, so in practice the model
  wins every disagreement.
* Two explanations fit the numbers; the RESULTS alone cannot choose:
  1. **The floor never held them.** `seed_from_grid` turns a cell into an
     index attribute only when it sits in the sample's row group; on forms
     where index values print a line or two from the sample row, the grid's
     window credits them and the floor's record does not. Nothing was
     dropped; the model did not add them either.
  2. **The model overruled them** with values from the wrong row (one row
     off), carrying a note, so `model_wins` replaced every floor value.
* **What settles it, at no model cost.** The run files from this rc3 run keep
  the merged record, the model's record and the floor's record (`46be026`).
  Run `report_ingest.log_scoring.rescore_saved` over the 9 blind logs and
  report, per log: index `floor_alone`, `model_alone` and merged scores, and
  the count of `model_wins` / `floor_only` / `model_only` verdicts on the
  index slots (from the run file's disagreements). For lab, run
  `lab_scoring.rescore_saved` on the four collapsed gradation sheets and
  report per specimen whether the hole matched and the depth difference in
  metres (the score's misses already record which hole the reader wrote).
  IDs, counts and distances only. The RESULTS.md should print
  `floor_alone` and `model_alone` beside before/after every time
  (`cluster_scoring.py`) — the numbers are computed and saved but not shown.

---

## 5. The record

* **Integrity: clean.** 49 files, 3,826 lines, none unparseable, none glued;
  463 tool starts and 463 ends; 1,401 model starts and 1,401 ends. rc3 does
  not have the activity-log lock (`6c81cc5`), and no loss happened here
  although one task launched 24 page looks (216 tiles) at once, with up to
  eight calls in flight — luck, not safety: the lock should ship.
* **The model's own words: absent where they matter.** All side calls'
  texts are there (vision answers, the check's raw JSON, `find_like` reads)
  — an improvement on check 1, where the check's JSON was missing. But the
  primary agent wrote **no text on any of its ~300 tool-calling steps** in
  49 runs, and **no reasoning summary** is recorded anywhere. The FDE's
  wrapper reads only `output_message` items from the Responses result
  (`glue/foundry_responses_model.py`, `_generate`) and does not ask for a
  reasoning summary; the package's own Responses engine
  (`webapp/palantir_sdk_engine.py`, `_generate_responses`) does not either.
  So why GPT-5.4 ignored the tile that found T5, why it dropped T3 and T4,
  and why every agent switched to a quote after the check rejected a callout
  are read from behaviour, not from reasons.
* **Still missing from `run.json`:** the vision probe profile and the app
  commit (check 1 §6, items 4 and 6). Versions are in `results.json`.
* **Infra in the record:** one 500 from the model service and one 903 s hung
  vision call (both `set-cross-references`); GPT-5.4's 45-per-minute bucket
  waited 85 times (370 s) in part B. None changed a score.

---

## 6. The release

**Release 5.32.1 from `a4ef417` as measured.** Reasons:

* Everything 5.32.1 changed that the runs reached was exercised and held:
  pixel boxes (every box, both models), the 2,048 px cap and tiling at the
  size sent, padded zooms (every part B window held its target), the
  quote-row anchor (all 5 quotes of note 4's line landed on that line, and
  the sixth quote on its own text), `find_like` on the numpy matcher (it ran,
  and matches OpenCV exactly). Not reached: the refusal (never had cause)
  and `"NxN"` tiles (all 81 page looks asked for `auto`).
* GPT-5.4's circle task went from 0 of 3 (5.32.0 and rc1, Funhouse) to 2 of
  3; Sol held 3 of 3; the full suite held 37 of 37 with fewer input tokens.
* The defects found are older than 5.32.1 (the markup check on comments, the
  ring "encloses" judgement, `dpi`, the strict markup fields, tile findings
  buried under the page answer) or cost time without changing answers
  (`find_like` on a poor example, tiling cost). None makes 5.32.1 worse than
  5.32.0, and each fix changes behaviour that should be measured before it
  ships.
* **One optional cherry-pick onto `release/5.32.1`:** the activity-log lock
  — only the `_FILE_LOCKS` / `_lock_for` hunk of `webapp/activity_log.py`
  from `6c81cc5` and its lock test, not the coverage work in the same commit.
  It touches logging only, and the post-release live checks depend on
  complete records. If the lead prefers the wheel to be byte-for-byte what
  was measured, skip it: this hand-back lost nothing.

---

## 7. Proposed general fixes (for 5.32.2, each to be measured)

Priority order. None is tuned to the tag sheet or 10.31A.

* **A. The markup check judges what a comment is about, and checks every
  anchor** (`funhouse_agent/markup_check.py`: `_expected`, `_PROMPT`,
  `CHECK_KINDS`/`GEOMETRY_ANCHORS`, `marks_to_check`). Name the target
  separately from the comment: the `label`, else the `quote`, else an
  optional `target` field the agent fills (planlens
  `document/markup_writer.py` field list; app `deep/tools.py:1036`
  `annotate_document`); and ask "is this mark on the thing the comment is
  about" rather than "does what is here match the comment". For a callout,
  ask for the whole line or object at the tip, not the glyph. Check
  quote-anchored marks and sticky notes too, so there is no unchecked route
  around a verdict.
* **B. Decide "encloses" from geometry when the check returns a box**
  (`markup_check.py:_check_one`, beside `_too_wide`): the ring encloses the
  thing when the thing's centre lies inside the ring with a few points of
  slack — the suite's own rule (`review_eval/checks.py:409-460`). Keep the
  side call's "inside" text for identity (QCE, arrowhead). Secondary:
  planlens could give small rings a margin larger than 2 pt
  (`markup_writer.py`, ring geometry) so a leader end does not cross the
  ring.
* **C. A given `dpi` never lowers a zoom below the budget** (planlens
  `document/document.py:1030`: use the budget's fit when it is larger; or
  the app drops the agent's `dpi` when a budget is in force,
  `vision_tools.py:779`), and the `render_region` description stops
  offering `dpi` (`deep/tools.py:850-885`).
* **D. `annotate_document` forgives the obvious guesses** (`color`, kinds
  `comment`/`text`, an `anchor` object): accept or ignore with a note, and
  say in the tool description that marks are drawn in red. 8 of 14 produce
  runs lost a round trip to these (`deep/tools.py:1036`; planlens
  `markup_writer.py` validation).
* **E. Surface what only the tiles found** (`vision_tools.py`,
  `_dispatch_analyze_pdf_page` around `:1139`): one merged list of located
  items, page box and source, with items found only in tiles marked as such.
  T5 (twice) and note 4 were lost under the whole-page answer.
* **F. Say how far a zoom's answer is from where it was aimed**
  (`vision_tools.py`, `_dispatch_render_region`): the distance from the
  window's aim to the nearest box the answer gives, with a note when it is
  more than the source view's error — a padded window can hold two
  candidates.
* **G. Reading instruction asks for a best reading** (`vision_view.py:129-133`):
  the best reading plus alternatives only where genuinely ambiguous, so a
  legible zoom does not come back bracketed; a bracketed reading on a zoom is
  resolved by a larger zoom, not by dropping the item.
* **H. `find_like` stops when the example floods** (`funhouse_agent/find_like.py`):
  when the candidates on a page run far past what a sheet could plausibly
  hold (part C read 367 without any warning; the cap is 400), skip the read
  pass and return the count with "box another copy, one not crossed by
  linework". In planlens (`document/findlike.py`, `_example`), drop template
  ink that continues past the example box's edge — a line crossing the
  example is not part of the mark. Test both on the T1 example, which the
  replay in §3.5 reproduces offline in seconds.
* **I. Flag duplicate marks** (`annotate_document` result or planlens
  `write_markups`): two marks of one kind and label overlapping by more than
  half are one thing marked twice (GPT-5.4 r2, Sol r3).
* **J. The suite checks where a comment points** (§2.6): `markups_point_at`
  in `review_eval/checks.py`, target boxes in `review_eval/tasks.py`.
* **K. Test, then perhaps render at whole 32-px patches** (`vision_view.py`,
  `render_view`): extend the clip by at most 31 px so both image sides are
  multiples of 32, and re-run the §5.2 cell on GPT-5.4 to see whether the
  1.4–1.7 % y stretch goes. Sol is unaffected either way.
* **L. Records:** request a reasoning summary on the Responses route and log
  it (`webapp/palantir_sdk_engine.py`, `_generate_responses`; the FDE's
  `foundry_responses_model.py`), and put the vision profile and app commit
  in `run.json` (`review_eval/runner.py`).
* **M. Report ingest, after the rescore in §4.2:** print `floor_alone` and
  `model_alone` in RESULTS.md (`report_ingest/cluster_scoring.py`); if the
  rescore shows `model_wins` replacing correct floor values, judge evidence
  per value rather than per object (`report_ingest/floor.py:117-145`,
  `log_floor.py:1199-1285`); add an unlinked "values read" line beside the
  linked lab score (`report_ingest/lab_scoring.py`), so reading and linking
  are measured apart.

---

## 8. Part C per task (run 4 → this run)

| task | run 4 | now | input K | output K | seconds | model calls | page looks | tiled (tiles) |
|---|---|---|---|---|---|---|---|---|
| calc-bearing-consistency | ✓ | ✓ | 29 → 58 | 0.7 → 1.9 | 14 → 34 | 3 → 10 | 0 / 1 | 0 / 0 |
| calc-wall-check | ✓ | ✓ | 344 → 242 | 36.3 → 21.2 | 237 → 179 | 23 → 23 | 7 / 6 | 0 / 0 |
| fixture-duplicate-page | ✓ | ✓ | 28 → 41 | 0.3 → 0.3 | 7 → 10 | 3 → 4 | 0 / 0 | 0 / 0 |
| fixture-markups | ✓ | ✓ | 167 → 28 | 14.4 → 0.6 | 102 → 14 | 18 → 3 | 1 / 0 | 1 (9) / 0 |
| meck-bioretention-access | ✓ | ✓ | 60 → 74 | 4.2 → 10.1 | 55 → 121 | 10 → 15 | 1 / 1 | 0 / 1 (4) |
| meck-bioretention-section-dims | ✓ | ✓ | 101 → 95 | 20.0 → 28.9 | 163 → 268 | 18 → 17 | 2 / 1 | 0 / 1 (4) |
| meck-curb-types | ✓ | ✓ | 93 → 59 | 5.5 → 2.7 | 62 → 39 | 12 → 13 | 1 / 1 | 0 / 1 (4) |
| meck-driveway-notes | ✓ | ✓ | 70 → 74 | 3.4 → 12.9 | 54 → 138 | 13 → 16 | 1 / 1 | 0 / 1 (4) |
| meck-monument | ✓ | ✓ | 66 → 61 | 5.9 → 11.7 | 82 → 110 | 10 → 13 | 1 / 1 | 0 / 1 (4) |
| meck-pavement-section | ✓ | ✓ | 81 → 75 | 5.0 → 7.1 | 63 → 74 | 12 → 15 | 2 / 1 | 0 / 1 (4) |
| meck-ramp-detail-callouts | ✓ | ✓ | 58 → 75 | 5.0 → 8.2 | 66 → 77 | 10 → 15 | 1 / 1 | 0 / 1 (4) |
| meck-ramp-slopes | ✓ | ✓ | 80 → 60 | 4.4 → 6.8 | 60 → 55 | 11 → 13 | 1 / 1 | 0 / 1 (4) |
| meck-ramp-warning-mat | ✓ | ✓ | 69 → 63 | 5.9 → 6.6 | 79 → 73 | 10 → 13 | 1 / 1 | 0 / 1 (4) |
| meck-revision-block | ✓ | ✓ | 57 → 60 | 1.1 → 2.6 | 23 → 34 | 10 → 13 | 1 / 1 | 0 / 1 (4) |
| meck-row-sidewalk | ✓ | ✓ | 46 → 61 | 1.4 → 6.2 | 28 → 59 | 8 → 13 | 1 / 1 | 0 / 1 (4) |
| meck-sediment-trap-criteria | ✓ | ✓ | 57 → 76 | 3.3 → 11.1 | 48 → 101 | 10 → 15 | 1 / 1 | 0 / 1 (4) |
| meck-trap-dimensions | ✓ | ✓ | 62 → 95 | 8.7 → 11.6 | 92 → 120 | 12 → 19 | 1 / 1 | 0 / 1 (4) |
| meck-underdrain | ✓ | ✓ | 73 → 60 | 3.9 → 7.8 | 63 → 82 | 12 → 13 | 1 / 1 | 0 / 1 (4) |
| produce-circle-tags | ✓ | ✓ | 110 → 286 | 11.0 → 88.6 | 81 → 428 | 32 → 60 | 1 / 1 | 0 / 1 (9) |
| produce-markup | ✓ | ✓ | 61 → 75 | 1.3 → 3.8 | 26 → 49 | 9 → 14 | 1 / 1 | 0 / 1 (4) |
| produce-memo | ✓ | ✓ | 432 → 469 | 49.8 → 92.1 | 201 → 273 | 34 → 71 | 10 / 10 | 0 / 10 (40) |
| set-3600-psi | ✓ | ✓ | 233 → 252 | 22.3 → 32.8 | 101 → 82 | 23 → 58 | 10 / 10 | 0 / 10 (40) |
| set-cross-references | ✓ | ✓ | 492 → 595 | 66.9 → 120.6 | 249 → 1340 | 35 → 72 | 11 / 11 | 0 / 10 (40) |
| set-find-bioretention | ✓ | ✓ | 278 → 123 | 26.1 → 22.8 | 261 → 154 | 29 → 23 | 11 / 2 | 0 / 2 (8) |
| set-long-rare-tag | ✓ | ✓ | 442 → 997 | 23.8 → 43.9 | 104 → 173 | 44 → 258 | 24 / 24 | 0 / 24 (216) |
| set-ncdot-vs-county | ✓ | ✓ | 139 → 197 | 10.4 → 31.9 | 78 → 168 | 17 → 38 | 5 / 5 | 0 / 5 (20) |
| set-sheet-index | ✓ | ✓ | 644 → 297 | 42.2 → 19.9 | 292 → 97 | 50 → 64 | 18 / 10 | 0 / 10 (40) |
| ufc04-confined-zones | ✓ | ✓ | 51 → 49 | 0.4 → 0.4 | 10 → 11 | 4 → 4 | 0 / 0 | 0 / 0 |
| ufc04-density | ✓ | ✓ | 44 → 46 | 0.3 → 0.4 | 10 → 11 | 4 → 4 | 0 / 0 | 0 / 0 |
| ufc04-supersedes | ✓ | ✓ | 41 → 42 | 0.2 → 0.2 | 12 → 13 | 4 → 4 | 0 / 0 | 0 / 0 |
| ufc04-table-5-1 | ✓ | ✓ | 59 → 69 | 0.3 → 0.7 | 17 → 22 | 9 → 11 | 1 / 1 | 0 / 0 |
| ufc07-drilled-shaft-table | ✓ | ✓ | 120 → 117 | 1.6 → 3.0 | 33 → 39 | 13 → 17 | 2 / 2 | 0 / 1 (4) |
| ufc07-figure-1-1 | ✓ | ✓ | 66 → 85 | 4.0 → 5.5 | 42 → 46 | 9 → 18 | 2 / 2 | 0 / 2 (8) |
| ufc260-appendices | ✓ | ✓ | 108 → 90 | 2.1 → 1.4 | 35 → 36 | 6 → 5 | 0 / 0 | 0 / 0 |
| ufc260-ch12-tables | ✓ | ✓ | 125 → 95 | 1.0 → 0.9 | 36 → 22 | 6 → 5 | 0 / 0 | 0 / 0 |
| ufc301-asce7-chapters | ✓ | ✓ | 61 → 92 | 1.3 → 1.6 | 18 → 26 | 4 → 5 | 0 / 0 | 0 / 0 |
| ufc301-changes | ✓ | ✓ | 37 → 36 | 0.4 → 0.4 | 11 → 9 | 3 → 3 | 0 / 0 | 0 / 0 |

Model calls include the per-task probe (5 calls here, 4 in run 4). Every
page look in this run went at most 2,048 px on its long side; run 4's went
3,296–6,000 px.

---

## 9. What was built

Built 2026-10-08 on app `master` (on top of `e7988b7`) and planlens `main`
(on top of `4a9393e`), uncommitted for the lead's review. Nothing here is
tuned to the tag sheet or 10.31A: the regression cases replay what the
traces showed, on synthetic fixtures or the public suite sheet. **Every item
except J and M changes what an agent sees, and is to be measured on Foundry
before it ships.**

| item | built | needs a live measurement |
|---|---|---|
| A. markup check judges what a comment is about; every anchor checked | yes | yes |
| B. "encloses" decided from geometry | yes (ring margin: not built) | yes |
| C. an agent's `dpi` never shrinks a zoom | yes (app side) | yes |
| D. `annotate_document` reads the obvious guesses | yes | light |
| E. what only the tiles found is said | yes | yes |
| F. a zoom says how far its answer is from its aim | yes | yes |
| G. best reading; a bracket is settled by a closer zoom | yes | yes |
| H. `find_like` stops flooding | yes (both halves) | yes |
| I. one thing marked twice is flagged | yes | yes |
| J. the suite checks where a comment points | yes | no (rescore) |
| K. whole 32 px patches, behind a switch | yes, OFF | yes (check 7) |
| L. reasoning summary; profile and commit in run.json | yes | yes |
| M. floor / model / values read in RESULTS.md | code part only | no (rescore) |

**A.** `funhouse_agent/markup_check.py` rewritten round one question: is the
mark on the thing it is meant to mark?
- The thing is named apart from the comment: the agent's new `target`, else
  the `label`, else the `quote`. With none, the mark is meant to be "on the
  thing that comment is about".
- The comment is shown as a remark or a request ABOUT the thing, never as
  its name.
- The look answers `same_thing` (and `comment_fits` when a thing is named
  and the comment says something else).
- A callout's or note's look reads the WHOLE line or object at its spot, not
  a letter. Its crop is 120 pt either side, so a notes line fits.
- Quote-anchored marks and sticky notes are now checked too. Only replies are
  not. The note says switching anchor does not settle a verdict.
- planlens: `MarkupSpec.target` (text, not drawn; a box there is refused)
  and its line in the tool description. The app's `annotate_document` note
  and the review prompt's one sentence about the check now say "every mark".
- Tests: `funhouse_agent/deep/tests/test_markup_check_offline.py`
  - `test_a_comment_is_a_request_about_the_thing_not_its_name`;
  - `test_a_quoted_comment_on_the_wrong_line_is_caught` (the GPT-5.4 wrong
    note, now caught on its quote anchor);
  - `test_target_names_what_a_mark_is_on`, the crop test, the nested-anchor
    test;
  - two existing tests updated: notes and quote marks are now checked.
- Live: yes. The verdicts are model judgements. Brief 5 should count
  verdicts against truth for rings AND for comments.

**B.** Enclosure is measured in `markup_check._verdict`.
- Rule: the thing's centre lies inside the mark with 4 pt of slack, the
  suite's own rule. Identity comes from the look: `same_thing`, or, on an
  old-style answer, its reading containing the named thing ("- GCE" holds
  "GCE"). A QCE or an arrowhead is still rejected.
- Order: a thing outside the mark is "beside the mark", then size (30×).
- Tests: `test_a_tight_ring_whose_look_counts_the_leader_end_is_confirmed`
  (both answer shapes), `test_a_look_alike_inside_the_ring_is_still_misplaced`,
  `test_the_thing_beside_the_ring_is_misplaced`.
- **Not built:** the larger margin for small rings in planlens. A leader that
  touches a tag crosses any ring round it, whatever the margin; the
  geometric rule removes the dependence on how the look reads that end.
- Live: yes, with A (the 4 wrong rejections of 70 should go).

**C.** `vision_tools._dispatch_render_region` drops an agent's `dpi` while a
budget is in force, and says so (`dpi_note`).
- The deep and native `render_region` schemas no longer offer `dpi`. The
  text-tool description says the zoom is always drawn as large as the model
  reads.
- planlens' `render_page` / `render_region` descriptions say a dpi only makes
  the image smaller.
- `document.py:1030` is unchanged: its contract ("under the budget: as
  asked") is pinned by its own test and serves non-agent callers.
- Tests: `funhouse_agent/tests/test_vision_brief4.py`
  - `test_an_agents_dpi_does_not_shrink_a_zoom` (the same 2,048 px image with
    and without `dpi: 300`);
  - `test_the_zoom_tools_no_longer_offer_dpi`.
- Live: yes. Every zoom should go at 15-17 px/pt. Watch whether hedged
  readings fall.

**D.** planlens `MarkupSpec.from_dict` reads the obvious guesses instead of
refusing the whole call.
- What it reads: `color` / `colour` (ignored), kinds `comment` and `text`
  (written as a note; also `rectangle`, `ellipse` and the like,
  `KIND_ALIASES`), and an `anchor` object (read as its own fields; a
  contradiction or a non-anchor inside it is still refused).
- Each is reported per markup in `WriteReport.adjusted` and the tool
  result's `adjusted`. The description says each kind has a fixed colour.
- Tests:
  - planlens `test_the_obvious_guesses_are_read_with_a_note` and
    `test_a_nested_anchor_that_contradicts_itself_is_refused`;
  - the toolkit test;
  - app `test_the_obvious_guesses_cost_no_round_trip`.
  - The old planlens test that used `colour` as its unknown field now uses
    `font`.
- Live: light. Count refused `annotate_document` calls; expect none of these.

**E.** `vision_tools._merge_found`, on any tiled `analyze_pdf_page`.
- One `found` list of every thing located: label, page box, `seen_in`
  (`page`, `r2c1` …). Tile boxes are preferred, as the smaller view.
- Items seen only in tiles are marked `tiles_only` and named in
  `found_note` ("found ONLY in the tiles … treat these as found, and zoom on
  each").
- Matching: within 2 % of the sheet (24 pt on 11 × 17), with labels that
  share a code or carry none.
- Test: `test_a_thing_only_the_tiles_found_is_named`. It replays GPT-5.4
  baseline: the page answer lists six, a reader answering from the fixture's
  truth gives the tiles all seven, and T5 comes back alone as `tiles_only`.
- Live: yes. Does the agent act on `found_note` (T5 ringed)?

**F.** `vision_tools._say_how_far_from_the_aim`, on every
`render_region(view=, image_box=)`.
- Fields: `aim`, `nearest_box_from_aim_pt`, `boxes_in_answer`.
- `aim_note` when the nearest box is more than 2.5 % of the source view away
  (31 pt off a whole 11 × 17 sheet; at least 12 pt), or when the answer
  gives several boxes.
- Test: `test_a_zoom_answered_about_a_neighbour_says_so` (Sol r3's two
  candidates in one window).
- Live: yes. Does the duplicate ring of Sol r3 go?

**G.** Reading instruction and bracket note.
- `vision_view.READING_INSTRUCTION` asks for a best reading, with brackets
  only where a character truly cannot be told apart in this image.
- An answer that still holds a bracket gets `reading_note`: settle it with
  a closer zoom, not by dropping the thing. Chart read-offs never do.
- `find_like`'s own verifier prompt is unchanged: there an uncertain read is
  wanted.
- Tests: `test_the_reading_instruction_asks_for_a_best_reading`,
  `test_a_bracketed_reading_comes_with_a_closer_zoom_note`.
- Live: yes. Two things to watch:
  - hedge rate on zooms;
  - whether whole-sheet looks start committing to misreads (the 5.29 QCE
    case). The legibility line and tiling still stand against that.

**H.** Both halves.
- planlens `findlike._crossing_lines`: ink that runs straight through the
  example box and on to the edge of a margin round it is left out of the
  template. That covers a grid line under the lettering, a wall, a rule or a
  leader's shoulder. Only thin bands (a third of the box or less) count, so
  a fill keeps the old behaviour and the solid-ink refusal still fires.
  `example.linework_left_out` says how much was dropped.
- App `find_like.FLOOD_PER_PAGE` = 200: past it on any page nothing is
  read. The result is `status: example_matches_linework` and the note says
  "NOT a count … box another copy, one not crossed by linework".
- Replay of §3.5 (numpy matcher):

  | example | candidates before | after | callouts of 7 |
  |---|---|---|---|
  | T3 | 43 | 43 (template unchanged) | 7 |
  | Sol r3's T1 box | 400 | 54 | 7 (was 1) |
  | Sol C's T1 box | 367 | 67 | 7 |
  | T1's true box | 400 | 55 | 7 (was 2) |

  Through the app, Sol C's example now reads ≤ 4 contact sheets (was 19).
- Tests:
  - planlens `test_an_example_on_a_grid_line_does_not_flood_the_search`,
    `test_a_clean_example_is_unchanged`,
    `test_a_line_through_the_box_is_left_out_and_the_mark_kept`,
    `test_a_box_holding_only_a_line_is_refused`;
  - app `funhouse_agent/tests/test_find_like_brief4.py` (the T1 replay, the
    flood guard, `flooded_pages`).
- Live: yes. `find_like` time and calls on the circle task.

**I.** planlens `duplicate_marks` → `WriteReport.duplicates`.
- Rule: marks of one kind saying the same thing (label, else comment) whose
  boxes overlap by more than half. Replies never count. Two different
  comments on one spot are two comments.
- The tool result carries `duplicates` and a `duplicates_note` ("count each
  thing once").
- Only marks in one call are compared. A repeat across two appending calls
  is not.
- Tests: planlens `test_one_thing_marked_twice_is_flagged`,
  `test_two_comments_on_one_spot_are_two_comments`, the toolkit test, and app
  `test_one_tag_ringed_twice_is_flagged`.
- Live: yes. Does GPT-5.4 r2's eighth ring go?

**J.** `review_eval/checks.py` `markups_point_at`, with
`tasks.MECK_1031A_RAMP_NOTE`, scored in `produce-markup` beside the old
check.
- Where a comment points: a callout's arrow tip, a note's spot, else a box's
  or highlight's centre. It must lie on a target box or within 4 pt of it.
- Recall and precision are reported per markup.
- The target is note 4's second line, x 60.1–225.5, y 224.5–230.3. It was
  measured from the page's ink (PyMuPDF only, 8 px/pt, annotations off) and
  checked by eye — not from planlens' quote anchoring.
- The slope "B" row is left to the owner and is not a target.
- **Rescore of the seven saved brief-4 PDFs** (part B × 6, part C × 1):
  GPT-5.4 baseline FAILS (its tip 335 pt away, on the section label); the
  other six pass, 0.0–0.1 pt from the line. The suite's 37/37 for Sol is unchanged; GPT-5.4 part B's
  markup score falls from 3/3 to 2/3.
- Tests: `test_a_comment_on_the_right_line_points_at_it`,
  `test_a_comment_on_another_note_fails_however_it_mentions_the_figure`,
  `test_point_at_needs_targets_a_pdf_and_a_matching_comment`,
  `test_the_ramp_note_target_on_the_public_sheet` (10.31A, the brief-4
  anchors).
- Live: none needed; `rescore=True` applies it to saved runs.

**K.** Behind `GEOTECH_VISION_PATCH_ALIGN` (OFF): `vision_view.align_to_patches`.
- It pads every rendered image with at most 31 px of white on the right and
  bottom, so both sides are whole 32 px patches, and widens the view by
  exactly what that shows. Converted boxes land where they did.
- It is skipped when the padded side would pass the cap.
- Tests: `test_patch_alignment_is_off_by_default`,
  `test_patch_alignment_pads_to_whole_patches_and_keeps_boxes_exact`
  (2048 × 1325 → 2048 × 1344, the mark within 1.5 pt),
  `test_patch_alignment_leaves_an_aligned_or_over_cap_image_alone`.
- Live: yes. `module_work/LIVE_TEST_QUEUE.md` check 7 runs the §5.2 cell
  off then on, on GPT-5.4. The hypothesis holds if the y scale goes from
  ~1.015 to ~1.000.

**L.** Three parts.
- **Reasoning summary** (`webapp/palantir_sdk_engine.py`).
  - The Responses route asks for one on every call:
    `GEOTECH_FOUNDRY_REASONING_SUMMARY`, `auto` by default, `off` to stop.
  - The request type is found from the SDK's own names (`Reasoning` /
    `ReasoningSummary` and kin). With none, nothing is sent.
  - A refusal (a bad request naming it, or invalid-argument) is retried once
    without it and not asked again.
  - The summary the result carries goes on `additional_kwargs["reasoning"]`,
    which the activity log already writes on `model_end`.
  - **Unverified live:** the SDK's real type names for the setting. The
    first Foundry call says, in `generation_info["reasoning_summary"]`
    (`requested` / `not requested`) and in `model_end.reasoning`.
  - The FDE's own glue (`foundry_responses_model.py`, outside this repo)
    still asks for none and reads only `output_message`. Brief 5 should run
    the package's `PalantirSdkChatModel(route="responses")`, or the FDE
    should copy these lines.
- **run.json** (`review_eval/runner.py`) now carries `versions`, `commits`
  and `vision_profile`.
  - Commits come from `GEOTECH_APP_COMMIT` / `PLANLENS_COMMIT` when the
    operator sets them; else the source checkout's `git rev-parse`, marked
    `+dirty`; else PEP 610. A test wheel installed from a file says nothing
    by itself, so the FDE should set the two variables.
  - `vision_profile` is what the probe measured for the run's model in this
    process; it is never a new probe.
  - `results.json` meta and RESULTS.md also name the commits.
- Tests:
  - `webapp/tests/test_palantir_reasoning.py` (asked and kept; level and
    off; an SDK that cannot ask; a refusal dropped once; an unrelated bad
    request still raised);
  - `test_run_json_says_what_the_run_ran_on`.
- Live: yes. Brief 5's `model_end` records should carry `reasoning`.

**M (code only).** Three parts.
- `log_scoring.score_one_log` scores the floor's record alone on every run
  (`floor_alone`), not only on a rescore.
- `lab_scoring` keeps `floor_alone` / `floor_record`, and adds
  `score_unlinked`. It answers "values read": each printed value counts when
  it is anywhere in the record, linked or not. `rescore_saved` fills both.
- RESULTS.md prints, for logs, before | floor | model | after. For lab it
  prints before | floor | model | after | read, plus a per-kind and
  per-sheet `read`. A column over fewer runs than the set says so.
- Tests: the log-reader run-file test, `TestReadingAndLinkingApart` (a missed
  link loses the values in after but not in read), and two RESULTS rendering
  tests.
- **Not built:** per-value evidence in the merge. It waits on the §4.2
  rescore.
- Live: none. Old run files show "-" until `lab_scoring.rescore_saved` /
  `log_scoring.rescore_saved` are run over them (the §4.2 rescore, at no
  model cost); lab floor needs a run that kept `floor_record`.

**Gates (2026-10-08, each in the foreground, one at a time, exit codes):**
- planlens 1,644 passed (exit 0);
- `funhouse_agent/deep/tests` 595 passed, 1 skipped (0);
- `funhouse_agent/review_eval` 93 passed (0);
- `funhouse_agent/tests` 1,582 passed, 4 skipped (0);
- `webapp/tests` 458 passed (0);
- `report_ingest/tests` 1,369 passed (0).

**What brief 5 must measure** (GPT-5.4 and Sol, through the package's
Responses engine, with `GEOTECH_APP_COMMIT` / `PLANLENS_COMMIT` set):
1. **Part B again** (`produce-circle-tags` + `produce-markup`, × 3 per
   model), against brief 4:
   - rings on target (T5 found via `found_note`?);
   - markup-check verdicts against truth, for rings (wrong rejections from
     4/70 to ~0) AND comments (comments on note 4's line now confirmed; a
     comment on another note rejected even by quote);
   - duplicate rings;
   - refused `annotate_document` calls;
   - zoom px/pt (no more 4.2);
   - hedged readings and tags dropped on them;
   - `find_like` calls and seconds.
2. **Part C on Sol (the whole suite):** no task may change outcome except
   `produce-markup`'s new check. Tokens and time against this run; watch
   `set-long-rare-tag` and the tile cost.
3. **The record:** `model_end.reasoning` present on Responses calls; run.json
   with `commits` and `vision_profile`.
4. **Check 7** (K), the §5.2 cell off and on, on GPT-5.4.
5. **No model:** rescore brief 4's part D run files (§4.2) for floor, model
   and read, before deciding on per-value evidence (M).
