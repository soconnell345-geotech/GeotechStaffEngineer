# Foundry brief 5, part C: the 14 Mecklenburg tasks, every run read in full

**Slice:** the 14 `meck-*` tasks of the suite, GPT-5.6 Sol, arms `baseline`
and `coverage` (`GEOTECH_COVERAGE=1`): 28 runs. Wheels
**geotech-staff-engineer 5.33.0rc1** (app `39716a8`) and **planlens
0.13.0rc1** (`c5bdb8d`). Code is cited at those commits.

**Sources.** Raw hand-back (git-ignored):
`module_work/field_feedback/2026-10-08_foundry_brief5_5.33.0rc1/raw/part_c/sol/runs/{baseline,coverage}/meck-*/`.
Each run was set beside its **run 6** twin from brief 4 (5.32.1rc3, Sol,
baseline):
`module_work/field_feedback/2026-10-08_foundry_brief4_5.32.1rc3/raw/part_c/sol/runs/baseline/meck-*/`.
That run is reviewed in `../2026-10-07_foundry_5.32.1rc3/TRACE_REVIEW.md`
(cited as **brief 4**).

**Method.** For all 28 runs I read, in full:
- every event of `activity.jsonl` (1,150 records);
- every vision side call's text: 175 page, tile and zoom answers;
- every tool result;
- every final answer and check verdict;
- the coverage arm's 14 `coverage.json` ledgers.

The run-6 twins were read where a comparison needed them (for example the
same misreading on the same sheet). Small scripts then counted the
following, in all three runs of each task:
- tokens, time and calls;
- tiles and zooms, with each zoom's window and px per pt;
- pixel boxes against 0-999 grid boxes, and bracketed readings;
- `found`, `found_note` and `aim_note`, and what each one pointed at.

`Lnn` is a line of that run's `activity.jsonl`.

---

## In short

1. **Outcomes did not change: 14 of 14 tasks in run 6, in baseline and in
   coverage.** The answers carry the same facts as run 6. Where the two
   brief-5 arms differ, it is sampling, not the arm. For example, the
   coverage arm's section A-A answer correctly leaves out a "1′-0″ gravel
   layer" that the baseline listed as a minimum.
2. **The coverage arm did nothing on these tasks, by design.**
   - Every Mecklenburg sheet is inventoried as `figure` (group `drawings`),
     so no read can arm the gate. The gate arms only on logs, lab sheets
     and field tests (`coverage.py:75-78`, `796-832`).
   - The agent never called `document_coverage`, and `gate` is empty in all
     14 ledgers.
   - The arm cost **+70.7 K input tokens (+6.4 %)** and **−100 s (−6.7 %)**
     on this slice:
     - about 14 K is the `document_coverage` description, about 250 tokens
       on each of 56 primary calls;
     - the rest is **6 more zooms** (21 against 15), at about 11.8 K tokens
       each.
   - Zoom count explains 84 % of the per-run spread in input tokens
     (r = 0.92). On this slice it follows disagreements between views
     (page, tile and zoom readings), not the arm.
   - The arm bought no answer quality and no citations: "PDF page" was
     cited 10 of 14 times in both arms.
   - **This slice carries 1.9 % of the suite-wide +3.82 M tokens and none
     of the +34 minutes.** The 47 % and 45 % come from other tasks.
3. **The new tools stayed out of the way.**
   - `measure`, `log_grid` and `document_coverage` were called 0 times in
     28 runs, and no answer mentions them.
   - Their only cost is their text in the prompt. The first primary call
     went from about 8,860 input tokens (run 6) to about 9,790 (baseline)
     and about 10,040 (coverage): +930 and +1,180 tokens on every primary
     call, or about 4–6 % of this slice's input.
4. **Fix C held: no agent `dpi` shrank a zoom.**
   - Run 6: 11 of 12 zooms carried a `dpi` (median 4.2 px/pt); 8 of the 12
     rendered at only 728–2,032 px.
   - Brief 5: 0 of 36 zooms carried a `dpi`. All 36 rendered at
     2,046–2,048 px on the long side, at 3.3–12.1 px/pt (median 6.8).
   - Windows are 73–546 pt, so 15–17 px/pt does not apply on these sheets.
   - Pixel boxes: 788 `px=` boxes, 0 grid boxes. Bracketed readings: 0 in
     175 answers (run 6: 1 in 82).
5. **Fix F (`aim_note`) gave 6 notes on the 11 zooms aimed with `view` +
   `image_box`, and all 6 are false.**
   - 3 aimed at a region (a whole detail, a whole table) and measured from
     its centre.
   - 2 are narrow zooms on a rotated note: the note's own box is dropped as
     "region-sized" (`vision_view.py:804`), so the nearest box left is a
     fragment 67–75 pt away.
   - 1 was 13.7 pt against a 12 pt tolerance.
   - The agent ignored all six.
6. **Fix E (`found` / `found_note`) adds noise on detail sheets.** It
   fired a "found ONLY in the tiles … zoom on each" note on 25 of 28
   looks, listing 136 items:
   - 33 were already in the page answer;
   - 15 were fragments cut at a tile edge;
   - 6 were "not stated here" sentences;
   - 9 were "?" or one-token labels;
   - 73 were off the question.

   It adds about 1.9 K characters to every look, which ride in every later
   primary call. No zoom went to an off-question item, and none of the
   136 changed an answer.
7. **Sol misreads or makes up small lettering in a single view, and the
   agent catches it by comparing views.** Every case was caught:
   - "DESIGNATED" for "DEDICATED" (both baseline runs, run 6 too);
   - "75 % development occupancy" where nothing is printed;
   - "3′-0″ MINIMUM" for "3:1 MAXIMUM";
   - "2′ MIN." for "21″ MIN.";
   - a zoom cut across a rotated note that wrote a whole note with a 4″
     pipe and 6″ spacing (printed: 6″ and 3″).

   Each was settled by another zoom. Separately, one whole-page answer
   (baseline `meck-driveway-notes`) gave every box in the frame of the
   sheet turned upright to read. A mark placed from it would land about
   300 pt off.
8. **The record still cannot say why a zoom was made.**
   - 0 of 109 primary calls carry a reasoning summary (the SDK enum
     mismatch named in the hand-back README).
   - The agent's own text on tool-calling steps is empty on 81 of 81 steps.

---

## 1. Per task (run 6 → baseline → coverage)

Tokens in are thousands. Minutes are the run's own. Tiles are the 2 × 2
tiles of the one page look per run. "New tools" means `measure`,
`log_grid` and `document_coverage` (baseline / coverage).

| task | outcome | tokens in (K) | model calls | minutes | tiles | zooms | new tools |
|---|---|---|---|---|---|---|---|
| meck-bioretention-access | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 74 → 85 → 65 | 15 → 15 → 13 | 2.0 → 2.0 → 1.4 | 4 → 4 → 4 | 1 → 1 → 0 | — / — |
| meck-bioretention-section-dims | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 95 → 98 → 93 | 17 → 17 → 18 | 4.5 → 3.9 → 3.5 | 4 → 4 → 4 | 2 → 2 → 4 | — / — |
| meck-curb-types | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 59 → 63 → 63 | 13 → 13 → 13 | 0.7 → 0.7 → 1.0 | 4 → 4 → 4 | 0 → 0 → 0 | — / — |
| meck-driveway-notes | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 74 → 64 → 83 | 16 → 13 → 16 | 2.3 → 2.7 → 1.8 | 4 → 4 → 4 | 2 → 0 → 2 | — / — |
| meck-monument | ✓ 4/4 → ✓ 4/4 → ✓ 4/4 | 61 → 91 → 66 | 13 → 16 → 13 | 1.8 → 2.6 → 1.7 | 4 → 4 → 4 | 0 → 2 → 0 | — / — |
| meck-pavement-section | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 75 → 84 → 104 | 15 → 15 → 17 | 1.2 → 1.6 → 1.8 | 4 → 4 → 4 | 1 → 1 → 2 | — / — |
| meck-ramp-detail-callouts | ✓ 4/4 → ✓ 4/4 → ✓ 4/4 | 75 → 84 → 86 | 15 → 15 → 15 | 1.3 → 2.0 → 1.5 | 4 → 4 → 4 | 1 → 1 → 1 | — / — |
| meck-ramp-slopes | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 60 → 65 → 64 | 13 → 13 → 13 | 0.9 → 0.9 → 0.8 | 4 → 4 → 4 | 0 → 0 → 0 | — / — |
| meck-ramp-warning-mat | ✓ 4/4 → ✓ 4/4 → ✓ 4/4 | 63 → 67 → 66 | 13 → 13 → 13 | 1.2 → 1.5 → 0.9 | 4 → 4 → 4 | 0 → 0 → 0 | — / — |
| meck-revision-block | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 60 → 76 → 77 | 13 → 15 → 15 | 0.6 → 0.7 → 0.6 | 4 → 4 → 4 | 0 → 1 → 1 | — / — |
| meck-row-sidewalk | ✓ 3/3 → ✓ 3/3 → ✓ 3/3 | 61 → 82 → 64 | 13 → 16 → 13 | 1.0 → 1.0 → 0.9 | 4 → 4 → 4 | 0 → 2 → 0 | — / — |
| meck-sediment-trap-criteria | ✓ 4/4 → ✓ 4/4 → ✓ 4/4 | 76 → 64 → 80 | 15 → 13 → 15 | 1.7 → 1.2 → 1.4 | 4 → 4 → 4 | 1 → 0 → 1 | — / — |
| meck-trap-dimensions | ✓ 4/4 → ✓ 4/4 → ✓ 4/4 | 95 → 90 → 135 | 19 → 17 → 22 | 2.0 → 2.2 → 3.2 | 4 → 4 → 4 | 4 → 3 → 6 | — / — |
| meck-underdrain | ✓ 4/4 → ✓ 4/4 → ✓ 4/4 | 60 → 97 → 135 | 13 → 17 → 21 | 1.4 → 1.9 → 2.8 | 4 → 4 → 4 | 0 → 2 → 4 | — / — |
| **all 14** | 14/14 → 14/14 → 14/14 | **988 → 1,111 → 1,182** | 203 → 208 → 217 | **22.5 → 24.9 → 23.2** | 56 → 56 → 56 | **12 → 15 → 21** | none |

Model calls include the per-task probe: 5 calls and 10 K tokens in every
run, the same in all three arms (14 % of run 6's input, 13 % of
baseline's). Every run is one page look, tiled 2 × 2, plus 0–6 zooms.

## 2. Where the tokens went

| all 14 tasks | run 6 | baseline | coverage |
|---|---|---|---|
| input / output tokens | 988 K / 134 K | 1,111 K / 153 K | 1,182 K / 145 K |
| primary agent: calls, input | 51, 532 K | 53, 630 K | 56, 682 K |
| first primary call's input (fixed prompt + tools) | 8,850–8,879 | 9,776–9,805 | 10,028–10,057 |
| page + tile side calls: input / output | 287 K / 107 K | 289 K / 118 K | 288 K / 103 K |
| zoom side calls: count, input / output | 12, 28 K / 20 K | 15, 52 K / 28 K | 21, 71 K / 34 K |
| average `analyze_pdf_page` result, characters | 5,446 | 7,709 | 7,133 |

**Run 6 → baseline: +123 K input (+12.5 %), +19 K output, +141 s (+10 %).**

- **About 49 K is the tool surface.** The prompt grew by about 930 tokens
  on each of 53 primary calls. Most of it is the `measure` and `log_grid`
  descriptions and schemas. They are offered on the review page whenever
  planlens has them (`deep/tools.py:1187-1190`; the description is
  `measure_tool.py:39-58`), and neither was called.
- **About 13 K is `found` / `found_note`.** About 2 K characters (about
  500 tokens) per look result, carried in the 1–3 later primary calls.
- **The rest is zooms.** There were 3 more of them, and each is bigger
  because no `dpi` shrinks it any more: 2.3 K → 3.5 K tokens per zoom side
  call.
- **Time follows output.** Sol wrote 10 K more output tokens on the
  baseline's tiles.

**Baseline → coverage: +71 K input (+6.4 %), −8 K output, −100 s.**

- **About 14 K is fixed:** the `document_coverage` description, about
  250 tokens × 56 primary calls.
- **The rest is 6 extra zooms.** Over the 28 brief-5 runs, input tokens =
  66.7 K + 11.8 K × zooms (r² = 0.84); seconds = 72 + 24 × zooms
  (r² = 0.47).
- **The extra zooms are not the arm's doing.** Of the net +6, +5 are in
  `meck-underdrain` (+2) and `meck-trap-dimensions` (+3). There they went
  to settling a zoom that had cut off, misread or made up text (§4.3).
- **The arm was faster.** The coverage arm's page and tile answers were
  15 K output tokens shorter.

**Suite attribution.** The suite's coverage arm cost +3.82 M input
(8.18 → 12.00 M) and +2,036 s. This slice's share is +70.7 K (1.9 %) and
−100 s. The cost is elsewhere in the suite.

## 3. The brief-4 fixes that touch these tasks

| fix | on these tasks | held? |
|---|---|---|
| C. an agent's `dpi` never shrinks a zoom | 0 of 36 zooms carried a `dpi`, against 11 of 12 in run 6. All 36 at 2,046–2,048 px; px/pt is set by the window alone. `dpi_note` never needed. | **yes** |
| E. what only the tiles found is said | Fired on 25 of 28 looks with 136 items, mostly not tile-only finds (§4.2). No zoom chased one. No answer gained from one. | built; **noisy on detail sheets** (MK2) |
| F. a zoom says how far its answer is from its aim | 6 notes on 11 aimed zooms; all 6 false (§4.1). | built; **false alarms** (MK1) |
| G. a best reading; brackets settled by a closer zoom | 0 bracketed readings in 175 answers (run 6: 1 in 82). `reading_note` never fired. Confident single-view misreads were made (In short 7), as in run 6, and were caught by comparing views. No sign G changed the rate on these sheets. | yes |
| L. records | `run.json` carries `commits` and `vision_profile` (held). `model_end.reasoning` absent on all 109 primary calls (not held; cause in the hand-back README). | **half** (MK8) |
| A, B, D, H, I, J, K, M | Not exercised: no marks, `find_like`, ingest or patch alignment in these tasks. | — |

## 4. Per-run notes

### 4.1 The aim notes (F), all six

| run | zoom (Lnn) | aimed at | note says | what the answer was about |
|---|---|---|---|---|
| baseline ramp-detail-callouts | L32 → L35 | `image_box` over the whole 2′-6″ detail (211 × 163 pt) | nearest box 21 pt from the aim | the three callouts of that detail: right |
| baseline revision-block | L32 → L35 | the whole revision table (291 × 61 pt) | nearest box 57 pt | the table's three rows: right |
| coverage section-dims | L36 → L43 | a 132 × 286 pt region of Section A-A | 36 pt, "may be about a different thing" | the five labels in that region: right |
| coverage section-dims | L33 → L45 | the 10′ / 4′ dimension lines | 13.7 pt against a 12 pt tolerance | exactly those two dimensions |
| coverage driveway-notes | L39 → L43 | Note 4 (a rotated line, window 83 × 232 pt) | 75 pt; label "= =" | Note 4, read right |
| coverage driveway-notes | L38 → L45 | Note 1 (window 73 × 182 pt) | 67 pt, nearest box "3600 P.S.I." | Note 1, read right. The full line's box spans 77 % of the window's height, so `answer_boxes` drops it as a region (`vision_view.py:775`, `804`) |

`_say_how_far_from_the_aim` (`vision_tools.py:865-905`) measures from the
aim box's **centre**. It ignores any answer box larger than 300/999 of the
zoom's view. Both rules fit brief 4's case: one tag, two candidates in a
padded window. On a detail sheet, the agent zooms on regions and on whole
lines of text, which those rules misjudge.

### 4.2 `found` and `found_note` (E)

`_merge_found` (`vision_tools.py:927-1006`) lists as "tiles only" anything
it cannot match to the page answer. It fails to match for four reasons:

- **The region filter cuts one side only.** A page answer's box round a
  whole dimension line or a whole note is often wider than 300/999 of the
  page and is dropped. The tile's tighter box for the same thing is kept.
  This is how the monument's 30″, 12″ and IRON PIN came out "tiles only"
  (baseline L33).
- **Every capitalised word counts as a "code".** `_codes`
  (`vision_tools.py:917-924`) does not separate words from codes, so
  "Appears in NOTES, Item 1" and "ALL CONCRETE TO BE 3600 P.S.I." look
  like different things. They are the same note, listed twice: once as
  page, once as tiles only (coverage driveway L35).
- **Labels go missing.** A box written on its own line gets the label "?"
  (`_label_text`, `vision_view.py:778-786`). One of these "?" items was
  the tile's made-up "3′-0″ MINIMUM" (coverage section-dims L29).
- **Tile-edge fragments and "not stated in this tile" sentences** become
  found things.

The merge also hides disagreement. Coverage curb-types: the page read
"STD. NO. 10.1" and a tile read "STD. NO. 10.17A"; they merged and the
note said the page and tiles "agree on what is there" (L29). The revision
block's real boxes were all dropped as regions. That left one entry
labelled "Revision block", which is the title block's "REV. 3" field
(baseline L29).

**Did it distract?**
- 6 of 15 baseline zooms and 6 of 21 coverage zooms overlap a tiles-only
  box.
- Every one of those zooms was aimed at the note the question was about,
  and its prompt names that note (for example "Read Note 1 exactly").
- None went to an off-question item.
- The cost is the about 2 K characters per look, carried forward.

### 4.3 Readings that disagreed, and how each was settled (M)

| run | first reading (Lnn) | what the sheet says | settled by |
|---|---|---|---|
| baseline bioretention-access | page: "DESIGNATED PUBLIC RIGHT OF WAY" (L26) | DEDICATED (tile L33, zoom L40). Run 6 made the same misread. | the zoom asked "dedicated or designated" |
| coverage bioretention-section-dims | tile: "3′-0″ MINIMUM" (L26) | "3:1 MAXIMUM" (zoom L46) | the zoom asked to confirm |
| coverage pavement-section | page: "after reaching **75 %** development occupancy" (L22) | "MEETING % DEVELOPMENT", no number (zooms L36, L42) | 2 zooms; the answer leaves the number out |
| coverage curb-types | page: "STD. NO. 10.1" (L20) | 10.17A (tile L26) | the agent took the tile (the `tiling` note says to trust a tile) |
| coverage trap-dimensions | zoom: "**2′ MIN.**", "plain L, not L′" (L40) | 21″ MIN. (zoom L48) | 1 more zoom |
| coverage underdrain | page: "3/8 inch on center" (L20; drops "SPACED 3″") | tile L27 reads it right | 4 zooms (next row) |
| coverage underdrain | zoom on a window that cut the rotated note: "UNDERDRAIN PIPE SHOULD BE MIN. **4″ DIAMETER** … **6″ ON CENTER**" (L40) | 6″ pipe, 3″ spacing (zooms L46, L52) | 2 more zooms, the last on a wider window |

The underdrain zoom is the one to remember. The window (149 × 234 pt) cut
the rotated note's lines, and the answer filled them in with plausible
values instead of saying the text ran off the image. The two zooms either
side of it, on almost the same windows, did say "cut off at the image
boundary" (L34, L46). The final answer is right because the agent
compared views.

### 4.4 Other runs worth a line

- **Baseline driveway-notes: boxes in a turned frame.** The whole-page
  answer gave every box in the frame of the sheet turned upright to read
  (L26). Note 1 came back as px [110, 220, 560, 270]; it actually lies at
  px [1339, 84, 1363, 421]. That is the same box rotated 90° (tile L33;
  coverage page answer L26). Only 1 of 28 looks did this. Nothing was
  aimed or marked from it here.
- **Zooms coarser than the tiles.** 8 of 36 zooms used windows of
  450–620 pt, at 3.3–4.7 px/pt, below the 4.8 px/pt of the tiles already
  read:
  - baseline: section-dims (L32), monument (L36, L37), pavement (L36),
    trap-dimensions;
  - coverage: pavement (L34), trap-dimensions, underdrain (L50).

  They can only re-read. In the baseline monument, two such zooms cost
  about 24 K tokens and 58 s, and the answer matches the coverage arm's,
  which made none. The underdrain's wide last zoom was useful: a wide
  window does not cut a note's lines.
- **The coverage ledger and hidden CAD text.** 10.31A carries hidden CAD
  text, and `read_document` returned its notes (baseline ramp-slopes L11).
  The inventory still says `has_text: false`, because hidden CAD text is
  not counted (`coverage.py:299-302`). A text-only read of such a page
  would be ledgered "read as text but no text layer" (`coverage.py:680`).
  Every run here also looked at the page, and the gate cannot arm on a
  figure, so nothing came of it.

---

## 5. Fixes

Classes: **(P)** plumbing or code, reproducible offline or with any model;
**(M)** model behaviour, measurable only on Foundry; **(S)** scorer or task.
None is fitted to one sheet. Each names the general failure.

| id | problem | class | code site (at `39716a8`) | proposed general fix | how to measure | Claude API check helps? |
|---|---|---|---|---|---|---|
| MK1 | `aim_note` false alarms. Measuring from the centre of a region-sized aim, and from answer boxes after region-sized ones are dropped, flags a zoom that read exactly what it was aimed at: 6 of 6 notes false here. | P | `vision_tools.py:865-905` (`_say_how_far_from_the_aim`); `vision_view.py:757-766` (`aim_tolerance`), `775`, `804` (`REGION_GRID` in `answer_boxes`) | Measure from the aim **box**: distance 0 when an answer box overlaps it, with region-sized answer boxes kept for this check. Judge "a neighbour" only when the aim box is thing-sized (smaller than the source view's location error). Otherwise say nothing, or say the zoom was aimed at a region. | Offline replay of every brief-5 zoom with `view` + `image_box` (suite-wide; the logs hold analysis, view and aim), hand-labelled. Target: 0 false notes, with brief 4's two-candidate case (`test_a_zoom_answered_about_a_neighbour_says_so`) still caught. | n: replays offline |
| MK2 | "Tiles only" mostly means "not matched". The one-sided region filter, capitalised words taken as codes, "?" labels, tile-edge fragments and "not stated" sentences gave 136 tiles-only items in 25 of 28 looks, each with "zoom on each"; about 2 K characters per look carried forward. | P | `vision_tools.py:912-1006` (`_merge_found`, `_codes` 917-924, note 996-1006); `vision_view.py:778-812` (`_label_text`, `answer_boxes`) | Match page and tile items on their quoted text (fuzzy) as well as on position. Let a tile box inside a page region box with the same text merge. Count as codes only tokens with a digit or a tag shape, not every capitalised word. Drop or mark fragments and negative sentences. Take a bare box's label from the line above. | Offline replay of all brief-5 part C looks (both arms). Count tiles-only items that are in the page answer, fragments or negatives (here 63 of 136; target near 0). Brief 4's T5 replay (`test_a_thing_only_the_tiles_found_is_named`) must still pass. | n: replays offline |
| MK3 | When the page and a tile read the same thing differently, `found` merges them and says they "agree". The disagreement is exactly where a zoom is needed (10.1 against 10.17A; DESIGNATED against DEDICATED). | P | `vision_tools.py:962-1006` (`_merge_found`) | When a merged group's quoted readings differ after normalising, say so in `found` ("readings differ: page '…', r2c1 '…'") and in the note. | Offline: count groups with differing readings in brief-5 part C, and check them against §4.3. On Foundry: does the agent zoom on those, and only those? | n (offline); Foundry for the agent's response |
| MK4 | A zoom whose window cuts lines of text completes them with made-up values instead of saying they are cut (1 of 36 zooms, the underdrain note). Single-view misreads of small lettering (In short 7) are the same family. | M | `vision_view.py:149-155` (`READING_INSTRUCTION`), `182-187` (`PIXEL_INSTRUCTION`) | One general sentence in the reading instruction: text that runs off the image edge is reported as cut, with where it is cut, never completed. Keep the cross-view check, which caught every case here. | Foundry: hand-label zoom answers that complete a cut line, before and after, on this slice's 14 tasks (1 of 36 now). | n (M) |
| MK5 | Zooms coarser than the tiles already read: 8 of 36 at 3.3–4.7 px/pt against the tiles' 4.8. They only re-read; the monument's two cost about 24 K tokens. | P | `vision_tools.py:836-856` (`_dispatch_render_region`, beside `dpi_note`) | Say in the result what the zoom rendered at (px/pt), and the window size under which small lettering would come out larger. Never refuse: robust first; a second read is allowed. Record in FUTURE_IDEAS "VISION EFFICIENCY". | Suite count of zooms below the page's tile px/pt, and their tokens, before and after (8 of 36 here). | n |
| MK6 | A whole-page answer gave every box in the frame of the sheet turned upright to read (1 of 28 looks). A mark from it would land about 300 pt off. | M | `vision_view.py:182-187` (`PIXEL_INSTRUCTION`) | Say, generally, that boxes are in the image as sent even when the drawing is turned sideways. | Foundry: the part A location re-measure on rotated public sheets (5003, 10.25a, 21.01, 3001), Sol and GPT-5.4. | n (M) |
| MK7 | Fixed prompt cost: `measure` + `log_grid` add about 930 tokens to every review-page primary call, and `document_coverage` about 250 more. None was called on 28 ordinary runs: about 4–6 % of this slice's input. | P | `deep/tools.py:1187-1190`; `measure_tool.py:39-58`; `coverage_tools.py:89-103` | No change now. If the suite-wide fixed cost matters, offer `measure` / `log_grid` only when the open document has a scale or a log page (planlens `find_scales` / page roles), on the same feature-detection pattern. | First primary call's input per run (here 8.86 K → 9.79 K → 10.04 K). | n |
| MK8 | The record cannot say why a zoom was made. Reasoning summary absent on 109 of 109 primary calls; tool-calling steps carry no text (81 of 81). | P | `webapp/palantir_sdk_engine.py:463-484` (`_REASONING_TYPES`, `_SUMMARY_TYPES`, `_reasoning_request`) | As the hand-back README proposes: `SummaryConfig` first in `_SUMMARY_TYPES`, `ReasoningConfig` preferred in `_REASONING_TYPES`. | Offline: a fake SDK module with those names gets the request built. Foundry: `model_end.reasoning` present on the next brief. | n: a fake SDK covers it; only Foundry runs the real SDK |
| MK9 | The coverage inventory counts hidden CAD text as no text layer, so a text read of such a page would be ledgered "read as text but no text layer". | P | `coverage.py:299-302` (`has_text`), `669-680` (`covered`) | Count a page as having text when planlens reports hidden CAD text on it, or when a text tool's result for that page returned text. | Offline: `read_document` on 10.31A page 0 → `covered` true; existing coverage tests still pass. | n |

**A cheap local live check (Claude API) before the next Foundry round:
not useful for this slice.**
- Every P item (MK1–MK3, MK5, MK7–MK9) replays offline from the recorded
  answers, or with a fake SDK or a fake model.
- Whether the agent acts better on cleaner notes (MK2, MK3), and every M
  item (MK4, MK6), is Sol behaviour. Only Foundry measures that.
