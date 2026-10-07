# Live check 1 on Funhouse (5.32.0, GPT-5.4): what happened in the six runs, and why

Release 5.32.0 (tag `v5.32.0`, commit `d903d54`) with planlens 0.11.0, run on
Funhouse on 2026-10-07. The model was `funhouse-gpt-high` (the vision results name
`gpt-5.4-2026-03-05`), reached through Prompter. Two Document Review suite tasks
were run three times each, under the identical arms `baseline`, `baseline_r2` and
`baseline_r3`, which use the default ("legacy") review agent:

| task | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| `produce-circle-tags`: suite's placement check | ✗ 0/7 on target | ✗ 1/7 | ✗ 2/7 (2 of 3 rings on target) |
| `produce-markup`: suite counts comments only | ✓ | ✓ | ✓ |
| where the markup comment actually points | beside note 3, about 80 pt above the line it quotes | beside note 3, about 50 pt above | on the ramp-section note it quotes ✓ |

The suite's own table is in `RESULTS.md`, next to this file.

**Sources.** The traces are in
`module_work/field_feedback/2026-10-07_funhouse_live_checks_v5.32.0/raw/check_532_markups/runs/<arm>/<task>/`.
That folder is gitignored and kept local. Each run has `run.json`, `activity.jsonl`,
`files/*.pdf` and, where the agent recorded feedback, `FEEDBACK.md`. The owner's
notebook view is `raw/notebook_output_check1.docx`. The comparison runs (GPT-5.6 Sol
on Foundry, 5.32.0rc2/rc3) are in
`module_work/field_feedback/2026-10-04_foundry_evidence/raw/brief3/out_sol_532/runs/`.
`Lnn` is a line number in that run's `activity.jsonl` and `t=` is seconds into the
turn. Code is cited as `path:line` at `v5.32.0` (planlens at its 0.11.0 commit
`059bc79`) unless HEAD is named.

**Method.** I read every event of all six Funhouse traces. A script then:
* converted every 0-999 box in every vision answer into page points through the
  look's `view`;
* compared those boxes, and every ring `annotate_document` wrote, with the fixture's
  ground truth (`build_synthetic_tag_set().tags`);
* matched each vision side call's input tokens against the size of the image it
  was sent.

I read the marked PDFs back with PyMuPDF and rendered the 10.31A comments over the
sheet. The Sol runs of the same two tasks got the same measurements: all ten
`produce-circle-tags` runs and four `produce-markup` runs. Sol `baseline` was read
event by event; the others through those measurements and their zoom and annotate
calls.

---

## 1. Ground truth (sheet 1 of the tag fixture, 1224 × 792 pt)

| | true box (pt) | centre |
|---|---|---|
| T1 | 490.0, 136.1 – 500.3, 140.4 | 495.2, 138.2 |
| T2 | 101.0, 321.0 – 111.2, 325.3 | 106.1, 323.1 |
| T3 | 362.7, 311.7 – 372.9, 316.0 | 367.8, 313.8 |
| T4 | 513.0, 312.4 – 523.2, 316.8 | 518.1, 314.6 |
| T5 (turned 90°) | 321.7, 421.2 – 326.0, 431.5 | 323.8, 426.3 |
| T6 | 270.6, 533.9 – 280.8, 538.2 | 275.7, 536.0 |
| T7 | 115.5, 608.3 – 125.8, 612.6 | 120.6, 610.5 |

* Each tag is 10.2 × 4.3 pt, an area of about 44 pt². The lettering is drawn as
  strokes, so the sheet has no text layer.
* Also on the sheet: a bare GCE with no leader at (641.7, 328.7), the legend row,
  and look-alikes (GCG ×6, one of them turned 90° at (233, 478), 43 pt left of
  and 58 pt above T6; QCE; GPE; FBG).
* The suite counts a ring as "on target" when the tag's centre is inside it (4 pt
  slack) and the ring is no more than 60 × the tag's area
  (`review_eval/checks.py:411-468`).
* **The smallest ring planlens draws round a tag is already 18 × 14 pt**, about 6 ×
  the tag's box area. It draws the ellipse through the box's corners plus 2 pt, and
  never less than 14 pt across (`planlens/document/markup_writer.py:112-116,
  523-535`). This matters in §4, cause C4.

---

## 2. Run by run

### 2.1 `baseline / produce-circle-tags`: 0/7, 165 s, 10 primary + 5 helper model calls, 2 `annotate_document` calls

1. **t=2–22.** `open_document`, then `render_page_thumbnails` + `document_page_map`.
   Then `analyze_image` on the contact sheet and, in parallel, **look 1**:
   `analyze_pdf_page(page 0, tiles="auto")`.
   * The result says `view_px [3957, 2560]`, `detail: original`, and **no tiles**
     (L26).
   * The side call read every tag as "G[C/E][E/F]" and gave 7 boxes. Against the
     truth these were 12–31 pt off (median 16), **all above the tag**.
2. **t=27–38.** Four zooms on the first four boxes, in one step (L24–L39).
   * `render_region` turns a tag-sized box into a window only 29–30 × 14 pt wide.
   * None of the four windows held a tag. All four answers: "blank / light gray".
3. **t=44–52.** **Look 2:** `analyze_pdf_page(..., tiles="6x6")`.
   * The tool silently ignored the tile request (§4, cause C3): one whole-page
     answer came back, 3,105 input tokens.
   * Its 7 boxes were 21–86 pt off (median 45), squeezed upward more the lower the
     tag sits.
4. **t=58–70.** **annotate #1** (L58): seven circles from look 2's
   `view [0,0,1224,792]` + `image_box`.
   * The rings were 33–34 × 14 pt, each 21–86 pt from the nearest true callout. The
     table in §3.3 has every mark.
   * The check: 7 × misplaced, "seen: nothing". **That was correct.**
5. **t=74.** `record_feedback` (L77). The agent blames "a coordinate conversion
   issue" in `annotate_document`. **That diagnosis is wrong:** the conversion is
   exact (`vision_view.py:490-503`); the boxes it was given were wrong.
6. **t=78–142.** `task` → deepagents' `general-purpose` helper.
   * This helper runs on deepagents' stock prompt, not the review prompt (§4,
     cause C8).
   * Look 3: errors 23–68 pt.
   * Eight zooms in one step on 50–70-grid-unit boxes round look-3 boxes
     (rendered 80–112 × 74–86 pt). Six showed nothing, one showed a GCG
     look-alike, and one contained T4. The agent had asked about that one as a
     "false positive", so the side call said "exclude it".
   * Look 4: errors 20–78 pt.
   * The helper returned look 4's whole-page boxes as the anchors ("anchor by
     region").
7. **t=147–160.** **annotate #2** (L150).
   * The primary converted the helper's 0-999 boxes into page points itself, by
     hand. For example `[400,147,417,154]` became `bbox [486,114,513,123]`; the
     exact value is `[490.1,116.5,510.9,122.1]`.
   * Seven rings, 20–78 pt off target. Check: 7 × "nothing", again correct.
8. **Answer (t=165):**
   * "I could not hand over a marked-up copy yet ... failed on all 7 circles".
   * It writes "(PDF page 1 / tool page 0)", which fails the suite's "no page 0"
     check.
   * It offers "a slower manual re-location pass with tighter zooms".
   * The file `tag_set_marked.pdf` holding the seven wrong rings is still on disk,
     and the suite scored that file.

### 2.2 `baseline_r2 / produce-circle-tags`: 1/7, 97 s, 8 primary calls, 2 annotate calls

1. **t=2–17.** `open_document`, thumbnails, **look 1** (`tiles="auto"`, none came
   back).
   * The agent asked for boxes round "the tag/leader area", so the boxes are large
     (35–45 grid units).
   * It found 6 of 7. The errors were 18–85 pt (median 49).
2. **t=23–38.** Seven zooms in one step: six on look-1 boxes (windows 53–72 × 36–51
   pt) and one on the legend.
   * L37 found T1 ("[G/C][C/G][E/F] ... likely GCE"), with its box 5 pt off.
   * One showed a tag cut off at the window edge; four were blank; the legend was
     confirmed.
3. **t=44–48.** **annotate #1** (L56): **one circle with the T1 zoom's
   `view [460.7,101.8,513.3,143.2]` and `image_box [0,0,999,999]`, i.e. the whole
   zoom window.**
   * The ring is 78 × 63 pt, 111 × the tag's area.
   * Check: "GCE — but the mark is drawn far wider than it". That was correct: the
     suite also rejects anything over 60 ×.
4. **t=52–65.** In one step:
   * `render_region` on a 110 × 85 pt box the agent chose itself round T1. It
     returned the tag-plus-leader box `[374,542,607,652]`.
   * **Look 2** ("recount"): seven boxes in quite different places from look 1,
     20–149 pt off (median 62).
5. **t=74–84.** Six zooms on look-2 boxes (windows 51–54 × 29–31 pt).
   * Five were blank.
   * One (L85) showed T6 cut by the top edge ("Tag GCE: [156,0,351,68]"). It was not
     used.
6. **t=90–92.** **annotate #2** (L107): one circle from the agent's own zoom
   (`view [413.5,63.5,556.5,181.5]`, the tag-plus-leader box).
   * Ring 51 × 22 pt (26 ×), centre 12 pt from T1's centre.
   * **The suite counts it as on target. The check said "drawn far wider" and
     rejected it.**
7. **Answer (t=97):** "I marked the one GCE callout I could place reliably".
   * It hands the file over with the caveat that the checker "flagged the placed
     circle as wider than ideal".
   * It writes "(PDF page 1 / tool page 0)" and offers a second pass.

### 2.3 `baseline_r3 / produce-circle-tags`: 2/7 (precision 0.67), 105 s, 9 primary calls, 2 annotate calls

1. **t=6–17.** **Look 1** (`tiles="auto"`, none came back): 7 boxes covering tag and
   leader, 9–38 pt off (median 18). This was the best whole-page look of the six
   runs.
2. **t=23–41.** Seven zooms on those boxes, in one step (windows 43–73 × 40–51 pt).
   **Six windows held a tag:**
   * T4, T5 and T3 were read "GCE", with boxes within 0–5 pt.
   * T1 and T7 were read "GCE", but their **boxes were 11.5 and 9.4 pt too high**
     inside 40-pt-tall crops.
   * T2 was read "[G/C]CE".
   * The seventh window (meant for T6) held the turned GCG look-alike, read
     "[G/C][G/C][G/C]".
3. **t=47–51.** `render_region` on a 100 × 80 pt box the agent chose (rendered
   130 × 110 pt, L46). The window held both the turned GCG and T6. The side call
   described only the GCG ("GC[G/C] ... No"). **T6 was never found.**
4. **t=57–66.** **annotate #1** (L52): five circles (T1, T3, T4, T5, T7).
   * Each used its own zoom's `view`, but **an `image_box` the agent had widened by
     hand**. For T1 the zoom said `[489,561,649,646]` and the agent sent
     `[300,300,760,860]`.
   * Rings were 27–47 × 16–36 pt (10–37 × the tag). **All five took their tag in, so
     the suite counts five hits.**
   * Check: four "GCE — but the mark is drawn far wider than it". The fifth (T5) was
     "seen: GCE" yet listed as misplaced.
5. **t=72–80.** Five tight re-zooms (22–35 × 14–22 pt), on sub-boxes the agent made
   from the zoom answers.
   * T3, T5 and T4 came back with boxes within 2 pt.
   * The T1 and T7 windows were blank: they carried over the 9–12 pt errors from
     step 2.
6. **t=86–91.** **annotate #2** (L105), three circles:
   * **T3: the new tight box was paired with the old T3 zoom's view** (view
     `[314.1,270.2,384.2,320.4]`, box from the `[346.6,304.4,381.2,322.4]` look).
     The ring landed 26 pt above T3. Check "nothing", correct.
   * **T4: ring 19 × 14 pt (6 ×), centred 1.5 pt from T4** → "drawn far wider".
   * **T5: ring 14 × 22 pt (7 ×), centred 1.3 pt from T5** → "seen: GCE", misplaced.
7. **t=96.** `record_feedback`: the check "repeatedly reported marks as misplaced or
   too wide even when centered on visible GCE text". **That diagnosis is correct.**
8. **Answer (t=105):** "I could not hand over a reliable marked-up copy yet ... 5
   clear GCE ... 1 additional vertical leadered tag ... excluded".
   * It offers boxes, callouts or a memo instead.
   * No "page 0".
   * The file on disk holds annotate #2's three rings: 2 on target, 1 off.

### 2.4–2.6 `produce-markup` (all three passed, which only proves that a comment containing "8.33" exists)

The sheet 10.31A has hidden CAD text. `open_document` reports
`pages_with_hidden_cad_text: "0"`, and `read_document` lists 66 CAD text objects
with boxes. **The whole notes column is ONE object**, `p0.c21` at `37,61,229,233`,
with no word boxes.

* **`baseline`, 39 s.**
  * `read_document` and `analyze_pdf_page` ran in parallel. The vision probe ran
    inside this look (L10–L18: 10 / 724 / 724 / 4,244 input tokens), the first
    vision call of the process. The look listed five ramp-slope notes.
  * The agent wrote a `callout` anchored by `quote "RAMP SLOPE CANNOT EXCEED 8.33%
    MAX."` (note 4, line 2) (L24).
  * planlens matched the quote to the whole notes object and pointed the arrow at
    the object's left edge, half-way down: **`points_at [37.0, 147.0]`. That is the
    margin beside note 3. The quoted line is at y ≈ 222–232, about 80 pt lower**, and
    the comment box covers notes 1–2.
  * Quote anchors are not checked (`markup_check.py:32`).
  * The answer says the comment is on the ramp-slope note.
* **`baseline_r2`, 25 s.**
  * `search_document("8.33%")` returned five hits. The two in the notes column both
    carry the whole column's box.
  * A look aimed at note 4 followed.
  * The agent wrote a `note` by the same quote. The sticky note landed at
    `[37,165,55,183]`, **beside note 3, about 50 pt above the quoted line**.
* **`baseline_r3`, 19 s.**
  * The look named the section note "RAMP SLOPE UP TO 7.5% (8.3% MAX.)", and the
    agent quoted that.
  * It is a separate CAD object (`p0.c33`, `490,429,543,460`), and the arrow points
    at `(490, 444.5)`, **the left edge of that note: on target.**
  * The comment box (`385.8,370.2 – 610.2,446.0`) covers part of the section drawing.

Sol's own runs of this task hit the same CAD-block effect: `baseline` and `sweep`
put a highlight over the whole notes column (`bbox [-3.5,50.2,269.5,243.8]`). The
effect comes from the anchor rule in planlens, not from either model:
`markup_writer.py:461` keeps the whole line box when there are no word boxes, and
`:466` puts the anchor point at that box's left edge, half-way down.

---

## 3. Measurements across runs

### 3.1 What the model was actually shown on Funhouse

In every vision side call, input tokens = image tokens + prompt tokens + a fixed
overhead. I counted the prompt tokens and charged one token per 32-px tile of the
image. On 37 Funhouse side calls with images up to 2,048 px, that left an overhead
of 144 tokens (median; range 51–203). **Every image wider than 2,048 px was charged
the tile count of that image shrunk to 2,048 px**, never the count at full size:

| rendered (what the tool reports) | image tokens charged | full size | shrunk to 2,048 px |
|---|---|---|---|
| tag sheet 3957 × 2560 (7 looks) | ≈ 2,747 | 9,920 | 2,688 (2048 × 1325) |
| 10.31A 3616 × 2794 (3 looks) | ≈ 3,200 | 9,944 | 3,200 (2048 × 1582) |
| zoom 2384 × 1967 (r2 L59) | ≈ 3,392 | 4,650 | 3,392 |
| legend zoom 4530 × 2240 (r2 L43) | ≈ 2,165 | 9,940 | 2,048 |

The 2384 × 1967 zoom costs more than the 2,500-tile ceiling of `detail="high"`. So
`original` was honoured up to 2,048 px, and anything larger was shrunk to fit. Sol on
Foundry was charged for the full size, at about 1.2 tokens per tile: 11,887 tokens
for the 3957 × 2560 sheet.

**So the whole sheet reached GPT-5.4 at about 2048 × 1325 px.** The 0.06-in lettering
(4.3 pt) arrived about **7 px tall, not 14**. Yet every result told the agent
`view_px [3957, 2560]`, `detail: original` (`vision_view.py:241-256`). Sol saw it at
full size. This explains why every whole-page GPT-5.4 read is bracketed
("G[C/E][E/F]", "G[C/O]E", "too small to be sure"), while Sol read "GCE" plainly.

Most zoom crops were under 2,048 px and arrived whole, at about 16.7 px per pt
(lettering about 72 px tall). The markup check's crops are 90–234 pt wide. Even when
shrunk to 2,048 px they carry at least 8.7 px per pt, so the lettering was 38 px or
taller. **Blank zooms were looking in the wrong place, and the check's verdicts were
judgements, not failures to read.**

### 3.2 How far whole-page boxes were from the tags (nearest box to each of T1–T7, pt)

| run | look | median | max | direction |
|---|---|---|---|---|
| GPT-5.4 baseline | 1 / 2 / 3 / 4 | 16 / 45 / 47 / 47 | 31 / 86 / 68 / 78 | above the tag, worse lower on the sheet |
| GPT-5.4 baseline_r2 | 1 / 2 | 49 / 62 | 85 / 149 | look 2 put tags in quite different places from look 1 |
| GPT-5.4 baseline_r3 | 1 | 18 | 38 | above |
| Sol baseline (full size) | 1 | 14 | 24 | above, worse lower down |
| Sol baseline_r2 / r3 | 1 | 36 / 36 | 50 / 58 | above, worse lower down |
| Sol sweep / sweep_r2 / sweep_r3 | 1 | 28 / 25 / 26 | 52 / 36 / 49 | above, worse lower down |
| Sol baseline_high (1979 × 1280 overview, 3 × 3 tiles) | 1 | 5 | 8 | none |

**Correction to (a):** the 40–80 pt figure holds for GPT-5.4's later looks (medians
45–62, maximum 149). Its first looks had medians of 16–49. **Sol's full-size
whole-page boxes were off too** (medians 14–36, maximum 58), with the same pattern:
too high by an amount that grows down the sheet (2–14 % of the y coordinate).

Whole-page boxes are therefore not good enough to place a mark from, for either
model. GPT-5.4's were worse and less repeatable. Sol at the smaller size, with tiles,
was within 8 pt. Nothing in the harness measures or reports this error.

### 3.3 Every mark GPT-5.4 placed (size against the 44 pt² tag)

| run · call | marks | where they were | suite | 5.32.0 check |
|---|---|---|---|---|
| baseline · #1 | 7 rings 33–34 × 14 pt (11 ×) | 21–86 pt from the nearest callout (T1 21, T2 43, T3 45, T4 45, T5 60, T5 54, T7 86) | 0 hits | 7 × "nothing" (right) |
| baseline · #2 | 7 rings, 18–42 × 15–24 pt (10–16 ×) | 20–78 pt off (T1 20, T2 45, T3 45, T4 47, T5 56, T5 58, T7 78) | 0 hits | 7 × "nothing" (right) |
| r2 · #1 | 1 ring 78 × 63 pt (111 ×) | takes T1 in, centre 18 pt off | blanket, no hit | "far wider" (right) |
| r2 · #2 | 1 ring 51 × 22 pt (26 ×) | takes T1 in, centre 12 pt off | **hit** | "far wider" (**wrong**) |
| r3 · #1 | 5 rings: T1 45 × 36 (37 ×), T3 47 × 24 (25 ×), T4 33 × 19 (14 ×), T5 27 × 27 (16 ×), T7 27 × 16 (10 ×) | each takes its tag in; centres 2–13 pt off | **5 hits** | 4 × "far wider", 1 × "seen GCE" yet misplaced (**all wrong**) |
| r3 · #2 | T3 33 × 22 (17 ×); T4 19 × 14 (6 ×); T5 14 × 22 (7 ×) | T3 26 pt off; T4, T5 centred within 1.5 pt | 2 hits | T3 "nothing" (right); T4 "far wider", T5 "seen GCE" misplaced (**wrong**) |

**24 rings in all.** The 5.32.0 check rejected the 16 that missed (correct), and
**also rejected all 8 that the suite counts as on target**, 5 of them at 6–16 × the
tag's area. **It confirmed none.**

### 3.4 Zoom windows against location error

A zoom finds the tag only if the window is wider than the location error.

| zoom window built from | size (pt) | windows holding a true callout |
|---|---|---|
| a tight whole-page box + 15 % (GPT-5.4 baseline) | 29–30 × 14 | 0 / 4 |
| the same, Sol baseline | 17–18 × 12–13 | 0 / 7 |
| whole-page box widened by the helper (GPT-5.4 baseline) | 80–112 × 74–86 | 1 / 8 (look-3 error 23–68 pt) |
| whole-page tag-plus-leader box (GPT-5.4 r2, look 1 / look 2) | 53–72 × 36–51 / 51–54 × 29–31 | 1 / 6, 0 / 6 |
| whole-page tag-plus-leader box (GPT-5.4 r3) | 43–73 × 40–51 | 6 / 7 |
| a box in points the agent chose itself (GPT-5.4 r2, r3; rendered) | 130–143 × 110–118 | 2 / 2 |
| a box in points the agent chose itself (Sol baseline; rendered) | 91–111 × 79–96 | 7 / 7 |
| full-width bands of the sheet (Sol r3) | 1224 × 449–657 | all 7 tags, boxes within about 1 pt |

---

## 4. Root causes of the circle failures

**C1. Every mark-placing attempt started from whole-page boxes. GPT-5.4's were
9–149 pt off and drift upward down the sheet. Confirms and corrects (a).**
* §3.2 has the numbers.
  * Both baseline annotate calls (14 rings) were placed straight from whole-page
    boxes.
  * In every run, every first zoom was aimed at a whole-page box: baseline L24–L39
    and the helper's L83–L114; r2 L16–L41 and L62–L85; r3 L16–L53.
* Mechanism:
  * The side call is a one-shot call. It is told only to "give it as a box ... on a
    0-999 grid" (`vision_view.py:93-100`), with no warning that the box will place
    a mark.
  * The agent copies numbers out of prose. They came in several shapes, sometimes
    covering the leader, and sometimes boxes of border lines on a blank crop.
  * `image_box_to_page` (`vision_view.py:490-503`) converts them exactly, so their
    error passes straight through. Nothing measures it.
* Sol had the same error. The difference is in what each agent did next (C5, C6).

**C2. NEW: the Funhouse route shrinks any image over 2,048 px. The app does not
know, so small lettering is never tiled.**
* §3.1 has the numbers. The probe sends at most a 2,048-px square
  (`vision_probe.py:12-14`; planlens `budget.py:167-168`). It found `original`
  honoured (714 / 714 / 4,234 image tokens, as on 2026-09-25).
* It then picked `openai-original`, which renders up to 6,000 px and 10,000 tiles
  (planlens `budget.py:107, 202-212`). That is more than this route delivers.
* The auto-tiling rule works out lettering height from the *rendered* size. It got
  4.3 pt × 3.23 px/pt ≈ 14 px, above 12 px, so it chose no tiles
  (`vision_view.py:290-308`; `vision_tools.py:1065`). At the true 1.67 px/pt it
  would have got 7 px and chosen 3 × 3.
* The prompt also tells the agent that `analyze_pdf_page` "tiles a sheet whose
  lettering is small" (`deep/prompt.py:299-300`). On this sheet it did not.
* Control: Sol at `openai-high` rendered 1979 × 1280, was tiled 3 × 3 automatically,
  and its whole-page boxes were within 8 pt (§3.2).

**C3. NEW: the tools made the first zooms fragile and silently dropped a tile
request.**
* `render_region` pads a `view` + `image_box` box by only 15 %
  (`vision_tools.py:774`). Every vision result invites zooming on the box exactly
  as given (`ZOOM_HINT`, `vision_view.py:103-105`). A tag-sized box therefore gives
  a window smaller than the location error, and the crop is blank (§3.4).
* The agent then tried `tiles="6x6"` (baseline L42). `_tile_count` does
  `int("6x6")`, hits a ValueError and returns 1 with no message
  (`vision_tools.py:1083-1100`).
* The default agent never sees which `tiles` values are allowed. It is told only
  "Render a PDF page and analyze it using vision." (`deep/tools.py:1149-1152`); the
  docstring that lists "2"/"3"/"4" (`:793-796`) is replaced. Its `render_region`
  text points at a `drawing_ir` module this page does not have (`:1161-1171`).

**C4. The 5.32.0 "drawn closely" question could not be satisfied for these tags.
Confirms (b).**
* The check asked whether the ring is "drawn closely round that one thing (the
  thing fills a fair part of it), or takes in a much wider area — several times the
  thing's size" (`markup_check.py:64-67`). A "no" turns a ring that encloses its tag
  into "misplaced" (`:157-161`).
* planlens cannot draw a ring smaller than 18 × 14 pt round a 10 × 4 pt tag (§1).
  That is already "several times" its size.
* Result: 0 confirmed out of 24, and 8 correct rings rejected (§3.3). Two of those
  verdicts contradict themselves: "seen: GCE" yet "misplaced", both on the turned
  tag T5. The trace keeps only `seen`, so the reason cannot be read.
* The check's advice for a too-wide ring is "zoom until you can box the thing
  itself" (`:208-217`). r3 did exactly that (step 5) and was still rejected:
  **no action available to the agent could pass.**
* HEAD replaces the question with a measured area test, `markup_check.py` at HEAD
  (`MAX_AREA_FACTOR` 30). Five of the eight rejected rings are 6–16 × the true tag
  area. They would very likely pass, but that depends on the box the check's own
  side call reports. Not yet verified on these marks.

**C5. The agent handled anchors badly. Confirms (c) and adds three more.**
* r2 #1 passed the whole zoom window: `image_box [0,0,999,999]` (r2 L56).
* r3 #1 widened each zoom's box by hand before passing it. For example
  `[489,561,649,646]` became `[300,300,760,860]` (r3 L52). These rings were the
  ones called "far wider".
* r3 #2 paired a box from one look with the `view` of another (r3 L105, the T3 mark,
  26 pt off).
* baseline #2 converted whole-page boxes to points by hand (L150).
* `annotate_document` accepts any `view`/`image_box` pair: it cannot tell whether the
  two came from the same look, or whether the box came from a look where the thing
  was legible (planlens `markup_writer.py:248-257`).
* The instructions point toward whole-page boxes. The prompt says to anchor "by the
  `view` and `image_box` of the look that found it" (`deep/prompt.py:376-379`), and
  the tool says the same (`deep/tools.py:592-594`; planlens `tools/specs.py:215-217`).
  The look that "found" each tag was the whole-page look. The check's note says
  "look at the page, take the view + image_box" (`markup_check.py:212-213`), and
  baseline answered it with a fresh whole-page look. Sol read the same text and
  zoomed instead, so the wording permits bad anchors without causing them on its
  own.

**C6. NEW: GPT-5.4 stopped after the second rejected attempt in all three runs.**
* Each run made exactly 2 `annotate_document` calls. Each ended with "If you want,
  I can ...", and the task, being a single turn, ended with it.
* The default agent has no model-call budget, so nothing forced the stop.
* Sol retried until confirmed: 1, 7 and 2 annotate calls in its baseline runs, up
  to 4 in sweep.
* r2 and r3 reacted to the check correctly in kind (own windows; tight re-zooms),
  but each tried only once more.

**C7. NEW: detection was incomplete at the resolution the model got.**
* Even the best run (r3) found only 5 of 7: T2 was read "[G/C]CE" and dropped, and
  T6 never appeared (its window showed the turned GCG next to it; L46–L49).
* So **a check that confirmed every correct ring would still have left r3 at 5/7**
  (recall 0.71, under the 0.8 threshold).
* The zoom side call's own boxes were mostly within 0–5 pt, but twice 9–12 pt too
  high in y (r3 L33 T7, L41 T1). Those errors made the later tight zooms blank.

**C8. NEW: the helper ran without the review rules.**
* In baseline the agent delegated to deepagents' `general-purpose` helper. The app
  re-declares it "with its stock description and prompt" (`deep/agent.py:1129-1132`).
* Its first call was 5,267 input tokens against the primary's 8,979, consistent
  with no review prompt.
* It returned whole-page boxes as anchors. Sol never delegated in 37 baseline runs.

**C9. Why two answers wrote "tool page 0".**
* baseline and r2 both write "(PDF page 1 / tool page 0)" when saying what was
  covered. The cited viewer page is right.
* The prompt says "never cite the tools' 0-based number" (`deep/prompt.py:341-346`),
  and the suite flags any "page 0" (`review_eval/tasks.py:73`).
* None of Sol's 140 recorded answers do this. It is a GPT-5.4 habit: it gives both
  numbers to be exact about coverage, perhaps prompted by the coverage rule
  (`deep/prompt.py:322-326`). Without the model's text this cannot be settled.

### How GPT-5.4 on Funhouse differed from Sol on Foundry, in short

| | GPT-5.4 (Funhouse, 5.32.0) | Sol (Foundry, rc2/rc3) |
|---|---|---|
| whole-sheet image the model got | 2048 × 1325, labelled 3957 px | 3957 × 2560 |
| whole-page box error, median | 16–62 pt; repeated looks disagree | 14–36 pt |
| reading of 0.06-in tags, whole page | always bracketed | "GCE" |
| after blank first zooms | re-look the whole page; delegate; one own window (r2, r3) | own boxes of about 80 × 60 pt; full-width bands |
| anchor passed to annotate | widened, whole-window, mismatched, hand-converted | each zoom's own view + box, unchanged |
| markup check it faced | "drawn closely" (5.32.0) | encloses only (rc2/rc3) |
| annotate attempts before stopping | 2, 2, 2 | 1–7, until confirmed |
| delegation | general-purpose helper once | never |
| "page 0" in the answer | 2 / 6 | 0 / 140 |

These are not clean A/B pairs: the model, the host, the image size and the check all
changed together. The Sol `baseline_high` run is the one partial control. At a
smaller image than GPT-5.4 got, but tiled, Sol located the tags within 8 pt. So
resolution alone does not explain GPT-5.4's error; the route not tiling (C2) and the
model's own boxes both contribute.

### And `produce-markup`

* It passed on all three runs, but **two of the three comments point at the wrong
  note.**
* A quote that matches a CAD text block with no word boxes is anchored at the
  block's left edge, half-way down (planlens `markup_writer.py:461, 466`).
* Quote-anchored marks skip the placement check (`markup_check.py:32`), and the task
  checks only that a comment saying "8.33" exists (`review_eval/tasks.py:819-821`).

---

## 5. Where the record limits these conclusions

* **The model's own text is not in these traces.** The 5.32.0 logger kept tool
  calls, usage and timings only; I checked that no `model_end` in the six runs has a
  `text` or `reasoning` field. Master now records both (`webapp/activity_log.py`,
  commit `f405765`, after this run). So why GPT-5.4 widened boxes, used
  `[0,0,999,999]`, mixed views, delegated, or stopped after two attempts is inferred
  from what it did and from its final answer. C5, C6 and C9 rest on behaviour, not on
  stated reasons.
* **Images are not logged.** C2 is inferred from token arithmetic. It is exact across
  12 calls, but the host's resize itself is not seen. Which layer does it (Prompter
  or the deployment) is unknown.
* **The side-call prompt is not logged verbatim.** I assumed the grid instruction
  from the code at the tag. The token counts are consistent with that.
* **The check's raw answer is not logged.** Only `seen` and the verdict are kept, so
  the two "seen: GCE yet misplaced" verdicts cannot be explained. Nor can I tell
  which "far wider" verdicts came from `close: false` and which from a wrong
  `encloses`.
* **The helper's prompt** is inferred from the code and its token count.
* **The Sol comparisons are confounded**: a different host, model, image size and
  check version. There are only three Funhouse runs per task, and one fixture sheet.

## 6. What the records would need for a review like this to be conclusive

1. The primary model's text for each call: the content returned with tool calls,
   and the reasoning summary where the API offers one. This is on master since
   `f405765`. The next Funhouse run should confirm that Prompter actually returns
   them for GPT-5.4.
2. The side-call prompt as sent (the agent's prompt plus everything appended), the
   `detail` value sent, and the image's size and byte count as sent. Better still,
   the provider-reported image tokens per call, and the images themselves for
   produce tasks and for any zoom that came back blank.
3. The markup check's raw JSON for each mark (`encloses`, `inside`, `close` or
   `thing_box`, `sure`) and the crop it looked at.
4. The vision probe profile (image tokens, ratios, chosen budget) and a fingerprint
   of the rendered system prompt and tool descriptions, in `run.json`.
5. For fixture tasks: one row per mark against the truth (distance, area ratio,
   hit), not only the counts. Also an overlay image of marks over truth.
6. The app commit in `run.json`, and the same harness version on both hosts for any
   cross-host comparison.

---

## 7. Recommendations (short)

1. **Size images to what the route delivers.**
   * Probe one image larger than 2,048 px (for example 4096 × 2048) at `original`
     and cap the budget at the edge the route really accepts. On Funhouse today that
     is 2,048 px.
   * Then `view_px` tells the truth and small lettering is tiled automatically
     (3 × 3 on this sheet).
2. **Never place a mark from a whole-page box.**
   * Have `annotate_document` refuse an anchor whose `view` is the whole page and
     whose box is tag-sized, refuse `[0,0,999,999]`, and refuse a `view` it has not
     seen returned with that box. It should explain why.
   * Make the first zoom from a whole-page box at least about 80 × 60 pt, or about
     3 × the expected error.
   * Change the prompt rule to: anchor by the zoom in which the thing is legible.
3. **Validate the HEAD size check on these 24 recorded rings before the next live
   run.** This can be done offline, by replaying the check on the saved PDFs.
   Expected: the on-target rings at 6–26 × pass, the 37 × ring is borderline
   against the 30 × limit, and the 111 × blanket fails. Then tell the agent to keep
   correcting until the check confirms, up to a stated number of tries, rather than
   ending with an offer.
4. Show the default agent the `tiles` values, and return an error for a bad one
   instead of ignoring it.
5. Give the `general-purpose` helper the review rules, or hide it on this page.
6. For quotes on CAD text blocks, narrow the anchor to the matched line. Add a
   placement check to `produce-markup`.
7. Log what §6 lists. The model's text is done on master; the check's raw answer
   and the image size as sent come next.
