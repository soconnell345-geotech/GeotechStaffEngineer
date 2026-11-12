# Singles: four coordination tasks, 24 runs, read in full

Part of the brief 7 trace review. This part covers the four "singles" coordination tasks:
`coord-sheet-index`, `coord-levels-dimensions`, `coord-drawings-vs-manual` and
`coord-drawings-vs-geotech`. Each task was run once in each of five arms on GPT-5.6 Sol
(`baseline`, `show_me`, `method`, `sheet_tools`, `all_three`) and once on GPT-5.4 (`baseline`),
so 24 runs.

For every run I read the whole `activity.jsonl` (each tool call with its arguments and result, each
model call with its reasoning summary and text), the `run.json` (question, reply, check verdicts) and
the delivered files (the Word memos' text and tables, and the comments in the one marked-up PDF).
I checked every value the runs acted on against the fixture that builds the documents
(`packages/aec-document-review/aec_document_review/review_eval/coordination_fixture.py`). The raw
records are in the gitignored
`module_work/field_feedback/2026-11-12_foundry_brief7/raw/` (`b7_coord_sol/runs/<arm>/<task>/` and
`b7_coord_gpt54/runs/baseline/<task>/`).

## What I found, in short

1. **All 20 Sol runs found both planted problems in all four tasks, and every passing verdict on a
   Sol run is sound.** No Sol reply states anything about the documents that the fixture
   contradicts. Two Sol Word memos (drawings-vs-manual, `show_me` and `all_three`) miscount the
   sections the manual lists as issued separately: they say five where there are four. The replies
   themselves count correctly.
2. **One GPT-5.4 pass is wrong.** On `coord-sheet-index` the GPT-5.4 reply says the index "matches
   the set exactly" with no title mismatches, yet the `index-title` check passes. It passes because
   the reply lists the sheets in order and S-201's own title ("Second Floor Framing Plan") sits on
   the line above S-202's "Roof Framing Plan". The check finds S-201, "roof" and "second floor
   framing" within 55 characters of each other. Any reply that just lists the sheets will pass this
   check.
3. **In the `show_me` arm, the main model never looked at a page itself.** No look in the eight
   `show_me` and `all_three` runs of this group set `show_me` to true. The switch was on: the arm's
   setting is in `run.json`, and the offered tool set's fingerprint differs from baseline. Every
   look still went through the vision side calls. In this group, then, the `show_me` arm measured
   the same harness as baseline, and its numbers say nothing about the main model looking for
   itself (DR-38).
4. **The arms changed the cost, not the answers.** In the drawings-vs-manual task, the `show_me`,
   `method` and `all_three` runs finished their own comparison, then started a general-purpose
   helper to "independently" redo it. Each helper re-opened both documents and re-viewed the sheets
   and manual pages, and came back with the same findings. Each helper took 110 to 170 seconds.
   Each of those runs used 0.4 to 0.7 million more side-call tokens than the baseline run of the
   task, counting the main agent's own second look at the manual. No answer changed. The
   `sheet_tools` run of the geotech task also started a helper.
5. **The sheet index tool is correct, but it saved no looking.** Its output matches all 15 title
   blocks. In both runs that had it, the model still viewed all 15 sheets. In `sheet_tools`, the
   model asked for the index and for all 15 page views in the same step, before the index came
   back. Both runs gave the "look at every page the question covers" rule as their reason.
6. **Two vision calls hung and cost about three minutes each.** A close-up tile call that never
   returns is retried only after the 180-second call limit. Across the whole suite (84 runs, 15,545
   tile calls) a tile normally takes 5 seconds (median), and 99 % take under 22 seconds; 34 never
   returned.

## How these runs are put together (common to all 24)

- **Three kinds of model call.**
  - The main agent makes 3 to 17 calls a run.
  - A helper's calls appear only where the main agent used the `task` tool (4 runs).
  - The "side calls" are the vision calls made inside the looking tools:
    - `analyze_pdf_page`: one overview of the page plus nine close-up tiles (3 by 3) per page;
    - `render_region`: one call per zoom;
    - `analyze_image`: one call per image;
    - five probe calls at the first look in each run, which measure the vision model's image
      budget.
- **Where the side-call tokens go.** On Sol, viewing a drawing sheet costs about 35,000 to 37,000
  input tokens in side calls (one overview and nine tiles). Viewing all 15 sheets therefore costs
  about 560,000 tokens, which is the side-call total of every Sol sheet-index run.
- **What the four checks are.** Every run is scored on four checks:
  - the two planted problems (`stated_together`: the terms within 300 characters of each other in
    the reply, or 200 where set);
  - "does not give up on legibility";
  - "no 0-based page citation".

  All four are scored on the reply only, not on the files.

| Task | Run | Main calls | Helper calls | Side calls | Main tokens in | Helper tokens in | Side tokens in | Seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sheet-index | baseline | 5 | 0 | 156 | 145,420 | 0 | 560,717 | 268.9 |
| sheet-index | show_me | 10 | 0 | 155 | 418,280 | 0 | 560,574 | 150.5 |
| sheet-index | method | 9 | 0 | 156 | 408,107 | 0 | 563,422 | 164.8 |
| sheet-index | sheet_tools | 3 | 0 | 155 | 83,576 | 0 | 559,804 | 119.6 |
| sheet-index | all_three | 5 | 0 | 156 | 133,056 | 0 | 559,594 | 277.6 |
| sheet-index | GPT-5.4 | 6 | 0 | 16 | 135,013 | 0 | 45,566 | 39.2 |
| levels-dimensions | baseline | 15 | 0 | 126 | 774,615 | 0 | 446,085 | 195.0 |
| levels-dimensions | show_me | 14 | 0 | 130 | 697,467 | 0 | 465,496 | 255.9 |
| levels-dimensions | method | 14 | 0 | 105 | 724,499 | 0 | 369,862 | 182.2 |
| levels-dimensions | sheet_tools | 17 | 0 | 136 | 1,059,020 | 0 | 479,651 | 311.1 |
| levels-dimensions | all_three | 13 | 0 | 91 | 601,168 | 0 | 317,433 | 184.4 |
| levels-dimensions | GPT-5.4 | 10 | 0 | 86 | 400,903 | 0 | 277,372 | 202.8 |
| drawings-vs-manual | baseline | 13 | 0 | 162 | 807,976 | 0 | 585,856 | 230.9 |
| drawings-vs-manual | show_me | 15 | 6 | 319 | 1,069,481 | 215,100 | 1,162,844 | 414.5 |
| drawings-vs-manual | method | 16 | 7 | 354 | 1,333,848 | 253,613 | 1,284,751 | 469.8 |
| drawings-vs-manual | sheet_tools | 12 | 0 | 162 | 874,292 | 0 | 585,113 | 245.1 |
| drawings-vs-manual | all_three | 15 | 6 | 269 | 1,149,514 | 155,394 | 978,439 | 344.6 |
| drawings-vs-manual | GPT-5.4 | 14 | 0 | 66 | 605,592 | 0 | 210,590 | 222.9 |
| drawings-vs-geotech | baseline | 15 | 0 | 95 | 845,541 | 0 | 341,768 | 195.5 |
| drawings-vs-geotech | show_me | 13 | 0 | 92 | 630,059 | 0 | 332,082 | 182.7 |
| drawings-vs-geotech | method | 14 | 0 | 98 | 770,801 | 0 | 355,578 | 220.4 |
| drawings-vs-geotech | sheet_tools | 13 | 7 | 131 | 710,853 | 138,335 | 471,221 | 476.0 |
| drawings-vs-geotech | all_three | 14 | 0 | 104 | 682,766 | 0 | 371,110 | 184.1 |
| drawings-vs-geotech | GPT-5.4 | 12 | 0 | 26 | 359,142 | 0 | 78,771 | 155.0 |

Notes on the seconds column:
- **`sheet-index` baseline** lost about 180 seconds to one hung tile (page 3, started at 24 s,
  re-run at 204.6 s).
- **`sheet-index` `all_three`** spent 123 seconds between the process starting and the turn
  starting. It then spent another 56 seconds between the turn starting and its first model call.
  Nothing in the record says what either wait was. The turn itself took 154.6 seconds.
- **`drawings-vs-geotech` `sheet_tools`** waited about 175 seconds on two hung tiles (pages 2 and
  3, started at 66 s and 100 s). It then spent 117 seconds in a helper.

## The fixture's truth for these four tasks

- **Sheet index.** The cover's index has 16 rows. It lists A-202 INTERIOR ELEVATIONS, which is not
  in the set, and it calls S-201 ROOF FRAMING PLAN, while the sheet's own title is SECOND FLOOR
  FRAMING PLAN. The other 14 rows match their sheets, revisions included (G-001, A-401 and A-601 at
  revision 1, the rest at 0).
- **Levels and grids.**
  - SECOND FLOOR is EL. 104.20 on A-201 and S-201 ("T.O. SLAB") and EL. 104.50 on both sections of
    A-301.
  - Grid B to C is 7200 on A-101, A-102, S-101, S-201 and S-202, and 7500 on both enlarged core
    plans on A-401. On A-401 the 7500 is clouded, revision 1 "REVISED CORE LAYOUT"; the geometry is
    still drawn at 7200.
  - Every value that agrees: FIRST FLOOR 100.00, ROOF 108.40, T.O. PARAPET 109.30; bays A-B and C-D
    6000, 1-2 and 2-3 5400.
  - Both 7200 and 7500 are in the text layer.
- **Drawings against the manual.** G-002 general note 6 cites Section 07 84 00, which the manual's
  contents do not list. S-001 note 6 gives footings 25 MPa; Section 03 30 00 article 2.2 gives 30
  MPa. The task's truth sentence adds that "the other cited sections are in the manual". In fact the
  manual's contents bind only 01 33 00, 03 30 00 and 08 71 00 in full; 04 20 00, 07 92 00, 08 11 13
  and 09 91 00 are listed there as "issued separately".
- **Drawings against the geotechnical memo.** S-001 gives 200 kPa allowable bearing, where the memo
  says 150 kPa, and 1.2 m minimum exterior footing depth, where the memo says 1.5 m. These agree:
  site class C, design groundwater El. 96.50, subgrade modulus 40 MN/m3, removal of all fill,
  inspection of every footing subgrade, excavations over 1.2 m sloped or shored. The fixture also
  plants footing F4 at grid C/2 on S-101 with no row in the S-501 schedule. That problem is scored
  in another task, `coord-schedules`, not in this one.

---

## Task 1: coord-sheet-index

Question: "Check the sheet index on the cover sheet against the sheets that are actually in this set.
What does not match?"

### Sol, baseline

- **Path.** The run opened the set, then read the cover with `read_document` (with locations and
  tables). The index came back as a table. At 13.8 seconds, before any picture, the main model's
  reasoning already named both problems from that table. It then made 15 `analyze_pdf_page` calls,
  one per sheet, to confirm each title block (about 560,000 side-call tokens), plus a thumbnail
  sheet.
- **Errors and lost time.** One tile on page 3 never returned. It was re-run when the 180-second
  limit ran out, which is about 180 seconds of the run's 269.
- **Reply.** Both problems, with A-202 listed and absent, and the S-201 index title set against the
  sheet's own title. It notes that S-202 is correctly ROOF FRAMING PLAN, and that 16 index rows sit
  against 15 sheets. No file delivered.
- **Checks.** Both planted checks pass soundly. No false statements.

### Sol, show_me

- **Path.** The cover in 3 by 3 tiles, then `read_document`, then all 15 sheets viewed through side
  calls. Not one look used `show_me`.
- **Reply.** Both problems, stated correctly. A Word memo was delivered with the same content.
- **Checks.** Both pass soundly. No false statements.

### Sol, method

- **Path and reply.** The same path as `show_me`, with a Word memo. Both problems, correct.
- **Checks.** Both pass soundly. No false statements.

### Sol, sheet_tools

- **Path.** Three main calls in all. In its second step the model asked for `sheet_index` and for
  all 15 `analyze_pdf_page` views in the same turn, sixteen tool calls together, so the index could
  not have spared a single look. Its reasoning: "the rules mention title blocks ... I probably need
  to look at all 15 pages closely."
- **Reply.** Three items: A-202 absent, the S-201 title, and 16 index entries against 15 sheets.
  All true.
- **Checks.** Both pass soundly. No false statements.

### Sol, all_three

- **Path.** `sheet_index` and `read_document` came first. By the third main call (at about 66 s)
  the reasoning already stated both problems. It then viewed the cover, looked at a thumbnail sheet
  (too small to read), and viewed the other 14 sheets because "the instructions say to look at every
  page that the question covers".
- **Reply.** Both problems, correct. It adds that "the remaining sheet numbers, titles, and revisions
  match the index", which is true.
- **Checks.** Both pass soundly. No false statements.
- **Time.** About three minutes of this run's recorded time is unaccounted for in the record (see
  the table notes).

### GPT-5.4, baseline

- **Path.** `read_document` of the whole set (without locations), one `analyze_pdf_page` of the
  cover, and a thumbnail sheet. In the thumbnail, the side call misread page 0's title as "NORTH
  BUILD NOTES"; this was harmless. The model viewed no other sheet. It read the other title blocks
  from the text layer, which does hold them.
- **Reply.** It opens: "the sheet index matches the set exactly. I did **not** find any missing
  indexed sheets, extra unindexed sheets, or sheet-number/title mismatches." Then it lists the index
  rows, including "S-201 — Roof Framing Plan", and the sheets present, including "PDF 13 S-201 —
  Second Floor Framing Plan". It closes with "Only issue noted: ... A-202", which contradicts its
  own opening.
- **Planted problems.**
  - A-202: delivered.
  - S-201 title: seen (both titles are in its own reply) but never delivered as a finding.
- **Checks.**
  - `index-missing-sheet`: passes, and soundly.
  - `index-title`: **passes for the wrong reason.** The terms are found across two adjacent lines of
    the list of sheets present.
- **False statements.** One: the opening claim of an exact match.

### Side by side

| Run | A-202 | S-201 title | Extra true items | False statements | Side tokens | Seconds |
|---|---|---|---|---:|---:|---:|
| Sol baseline | delivered | delivered | 16 vs 15 | 0 | 560,717 | 268.9 (180 s hung) |
| Sol show_me | delivered | delivered | none | 0 | 560,574 | 150.5 |
| Sol method | delivered | delivered | none | 0 | 563,422 | 164.8 |
| Sol sheet_tools | delivered | delivered | 16 vs 15 | 0 | 559,804 | 119.6 |
| Sol all_three | delivered | delivered | none | 0 | 559,594 | 277.6 (about 180 s unrecorded) |
| GPT-5.4 | delivered | seen, not delivered (check passes anyway) | none | 1 | 45,566 | 39.2 |

**Arm against chance.**
- Every Sol arm got both problems, with one run each, so no arm effect on accuracy can be seen.
- All Sol arms spent the same side-call tokens: viewing 15 sheets is fixed by the rule to look at
  every page in scope.
- The main-call cost differs only by how many steps the model took (3 to 10); none of these
  differences is tied to an arm's feature.
- `sheet_tools` was the cheapest and fastest, but its index played no part in what it viewed.

---

## Task 2: coord-levels-dimensions

Question: "Check that the floor levels and the grid dimensions agree from sheet to sheet: plans,
elevations, sections, enlarged plans and structural plans. Where do they disagree?"

Every run, Sol and GPT-5.4, found the second-floor level mismatch the same way, from the text:
- `find_quantities` or `search_document` returned 104.2 on page 5 (A-201) and 104.5 on page 6
  (A-301) within the first 6 to 10 seconds. The runs that used `read_document` first (`sheet_tools`,
  `all_three` and GPT-5.4) had them from the text within the first minute;
- a zoom with `render_region` confirmed it.

How each run found bay B to C is under question 3 below.

### Sol, baseline

- **Path.**
  - `find_quantities` at 6 s. It put the levels in view. It also misread the footing schedule's
    rebar ("10-20M", "8-15M") as length ranges "10 to 20 m", and labelled pages 3-4, 7, 9 and 12-13
    "not searchable as text". All six have a text layer.
  - Picture views of the plans, sections and enlarged plans. For A-401 the model wrote its own
    prompt asking for "grid labels and grid-to-grid dimensions". At 89.8 s the tiles of A-401 read
    "7500" inside a revision cloud, and at 114.5 s its reasoning set 7500 against 7200.
  - Two zooms confirmed the dimension and the C bubble, and `read_document` of page 7 confirmed
    once more.
- **Reply.**
  - Both mismatches, each with the right sheets.
  - The values that agree are listed and are all correct: 100.00, 108.40, 109.30 and the bays.
  - A Word memo was delivered.
- **Checks.** Both pass soundly. No false statements.

### Sol, show_me

- **Path.** The same approach as baseline. The tiles first read 7500 at 94.1 s, and the reasoning
  used it at 120.8 s. A zoom at 223 s confirmed it. No `show_me` looks.
- **Reply.** Both mismatches, with sheet numbers, the clouding and "REVISED CORE LAYOUT". It adds
  overall grid dimensions of 19,200 by 10,800, which is correct.
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### Sol, method

- **Path.** `search_document` and `find_quantities` found the levels at 7 s. The A-401 tiles found
  7500 at 67.9 s, and zooms and `read_document` confirmed it.
- **Reply.** Both mismatches, each with the clouding.
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### Sol, sheet_tools

- **Path.** `read_document` of the set came back with 7500 and 104.50 in the text at 59 s. Picture
  views followed, then zooms, then `annotate_document`.
- **Reply.** Both mismatches, plus a note that S-101's top of footing at 98.00 has no conflicting
  value. That is true.
- **Delivered.**
  - A Word memo.
  - A marked-up PDF with three comments: a highlight on 104.50 on A-301 and two circles on the 7500
    dimensions on A-401. All three are placed on the right things.
- **Checks.** Both pass soundly. No false statements.
- **Cost.** This was the most expensive run of the task (1.06 million main-call tokens, 311 s),
  mostly because of the markup step.

### Sol, all_three

- **Path.** `read_document` gave both values in the text at 52 s. Picture views and zooms confirmed
  them.
- **Reply.** Both mismatches. It writes "differs by 0.30" with no unit, a slip in wording only.
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### GPT-5.4, baseline

- **Path.** `read_document` returned 7500 and 104.50 at 21 s, and picture views of eight sheets
  followed.
- **Reply.**
  - The level is reported as a "clear disagreement".
  - The bay is put under "Likely disagreement / needs confirmation": "either A-401 is dimensioning a
    local enlarged-plan width ... or A-401 is inconsistent with the main plans". It does flag it
    "for coordination".
- **Delivered.** A Word memo.
- **Checks.** Both pass. The bay check is sound, because the finding is in the reply, only hedged.
  No false statements.

### Side by side

| Run | Second floor | Bay B-C | First seen 7500 | False statements | Seconds |
|---|---|---|---|---:|---:|
| Sol baseline | delivered | delivered | tile of A-401, 89.8 s | 0 | 195.0 |
| Sol show_me | delivered | delivered | tile of A-401, 94.1 s | 0 | 255.9 |
| Sol method | delivered | delivered | tile of A-401, 67.9 s | 0 | 182.2 |
| Sol sheet_tools | delivered (and marked up) | delivered (and marked up) | text of page 7, 59.3 s | 0 | 311.1 |
| Sol all_three | delivered | delivered | text of page 7, 52.4 s | 0 | 184.4 |
| GPT-5.4 | delivered | delivered, hedged | text of page 7, 21.1 s | 0 | 202.8 |

**Arm against chance.** All Sol arms gave the same answer. The spread in cost (601,000 to 1.06
million main-call tokens) follows the steps taken and the optional markup, not the arm's feature.

---

## Task 3: coord-drawings-vs-manual

Question: "Check the notes on the drawings against the project manual. List any conflict between
them, and any reference to the manual that does not resolve."

All six runs reach the same core answer:
- **The conflict.** Footings at 25 MPa on S-001 note 6 against 30 MPa in 03 30 00 article 2.2.
  Every run says the other strengths agree.
- **References that do not resolve.** 07 84 00, which the contents do not list, with the manual's
  own sentence that sections not listed are not part of it.
- **Four more references, listed by every run as not resolving in the supplied volume.** 04 20 00,
  07 92 00, 08 11 13 and 09 91 00, which the contents list as issued separately. This is an
  accurate reading of the manual's contents page. The task's truth sentence ("the other cited
  sections are in the manual") is looser than the manual itself. So I do not count these as false
  alarms; see "Follow-ups" below.

Both planted items are in the text layer, so text found them in most runs; where a picture got there
first, the text confirmed it.
- **07 84 00 (G-002).** Every run had it from the text of G-002 within 6 to 29 seconds.
- **25 MPa (S-001 note 6).** `read_document` of the whole set returns pages 0 to 7 first, and the
  model has to ask again for the rest, so S-001's text arrived later:
  - `sheet_tools`: 12 s, through a search;
  - Sol baseline: 48 s;
  - GPT-5.4: 48 s;
  - `show_me`: 132 s.

  In `method` and `all_three`, a close-up tile of S-001 reached it first (123 s and 121 s), and the
  text confirmed it afterwards.

Beyond that, the picture viewing (all 15 sheets and five manual pages for every Sol run; six sheets
for GPT-5.4) found nothing new.

### Sol, baseline

- **Path.** Text of both documents, then all 15 sheets and all five manual pages viewed.
- **Reply.** The conflict, 07 84 00, and the four issued-separately sections, each with the drawing
  note that cites it.
- **Delivered.** A Word memo with a correct count ("Five drawing references do not resolve ... Four
  point to sections that ... were issued separately ... One points to Section 07 84 00"). It also
  sets door 106 at HW-1 against door 107 at HW-2, both meeting rooms, with door 107 clouded; that is
  true.
- **Checks.** Both pass soundly. No false statements.

### Sol, show_me

- **Path.**
  - The model's own comparison was done by 216.8 s, after all 15 sheets and the five manual pages
    had been viewed (the manual pages twice).
  - Its reasoning then turned to delegating: "Maybe it helps to use a subagent to analyze findings
    with source citations independently ... We seem to have enough info already. Delegating
    comparison and synthesis to a general-purpose reviewer could be useful since it can see all
    docs."
  - The `task` helper ran from 216.8 s to 385.2 s. It re-opened both PDFs and re-viewed the sheets
    and manual pages, then returned the same findings.
  - No `show_me` looks.
- **Reply.** Correct.
- **Delivered.** A Word memo, whose summary says: "Six specification references do not resolve ...
  one cited section (07 84 00) ... and five cited sections are listed ... issued separately." The
  memo's own table lists four.
- **Checks.** Both pass soundly.
- **False statements.** One: the memo's count.

### Sol, method

- **Path.**
  - The model's comparison was done by about 200 s.
  - It then re-viewed the five manual pages in 2 by 2 tiles.
  - It made four zooms: S-001 note 6, the mix table, G-002 notes 4 to 11, and the contents page.
  - Then it called the helper ("it might help to have a subagent or another process review these
    findings for sanity"). The helper ran from 271.1 s to 439.3 s.
  - In all, the same facts were checked three times.
- **Reply.** Correct, with the five unresolved references set out in a table.
- **Delivered.** A Word memo with a correct count. It also notes, correctly, that HW-3 is not
  assigned.
- **Checks.** Both pass soundly. No false statements.

### Sol, sheet_tools

- **Path.**
  - It called `sheet_index`, but nothing in the reply or the reasoning used the result.
  - It made two fuzzy text searches ("SPEC", "SECTION"), and the hits came back from every page,
    including the six pages labelled "not searchable as text".
  - It viewed all 15 sheets and then the five manual pages.
  - No helper.
- **Reply.** Correct. It also lists:
  - "FINISH AS SCHEDULED" on A-501 and S-501 with no finish schedule anywhere, which is true;
  - the geotechnical memo as cited but not supplied, which is true.
- **Delivered.** A Word memo with a correct count.
- **Checks.** Both pass soundly. No false statements.

### Sol, all_three

- **Path.**
  - It called `check_cross_references`. That found the two planted callout problems of another task
    (7/A-501 and 2/A-302), which are out of scope here and rightly left out of the reply.
  - It viewed the sheets and manual pages.
  - It searched for "07 84 00|firestop".
  - Then it called the helper (190.0 s to 299.6 s), which re-viewed 10 sheets and the manual. The
    helper also reported that door 107 at HW-2 differs from doors 106 and 206 at HW-1. That is true,
    but it did not reach the reply.
- **Reply.** Correct, including that 07 92 00 is "needed to interpret A-501's SEALANT AND BACKER ROD
  notes".
- **Delivered.** A Word memo, whose summary says "five cited sections are only listed as issued
  separately". There are four.
- **Checks.** Both pass soundly.
- **False statements.** One: the memo's count.

### GPT-5.4, baseline

- **Path.**
  - Text of both documents.
  - Pictures of six sheets only: A-101, A-102, A-401, A-601, S-201 and S-202. It prompted the S-202
    view as a "detail/schedule sheet", so it apparently meant S-501, which is PDF page 15, not 14.
  - Then 15 exact searches of the manual, one for each section number or strength.
  - It never viewed G-002, S-001 or any manual page on its own (only a thumbnail sheet of the set).
    This made no difference, because the planted items are text.
- **Reply.** Correct: "1 direct conflict and 5 unresolved manual references".
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### Side by side

| Run | 07 84 00 | 25 vs 30 MPa | Issued-separately four | Extra true items | False statements | Helper | Seconds |
|---|---|---|---|---|---:|---|---:|
| Sol baseline | delivered | delivered | listed | door 106/107 hardware (memo) | 0 | none | 230.9 |
| Sol show_me | delivered | delivered | listed | hardware sets resolve | 1 (memo count) | 168 s, same findings | 414.5 |
| Sol method | delivered | delivered | listed | HW-3 unused | 0 | 168 s, same findings | 469.8 |
| Sol sheet_tools | delivered | delivered | listed | finish schedule missing; memo not supplied | 0 | none | 245.1 |
| Sol all_three | delivered | delivered | listed | geotech memo not supplied | 1 (memo count) | 110 s, same findings | 344.6 |
| GPT-5.4 | delivered | delivered | listed | none | 0 | none | 222.9 |

**Arm against chance.**
- **The helper.** It appears in `show_me`, `method` and `all_three` and not in `baseline` or
  `sheet_tools`. `method` and `all_three` carry the written review method ("work from your notes,
  then look at both places before you report"). `show_me` does not, and it spawned the helper just
  the same.
- **What can be said.** With one run each, this cannot be told from chance. The model's reasoning
  in each case is its own choice ("for sanity", "could be useful"), not a quoted instruction.
- **The cost.** Where a helper ran, it roughly doubled the side-call tokens (586,000 against 0.98 to
  1.28 million) and added 110 to 240 seconds, and it changed no reply.

---

## Task 4: coord-drawings-vs-geotech

Question: "Do the structural drawings follow the foundation recommendations of the geotechnical
memorandum? List every difference."

All five Sol runs had both planted values in their tool results within 7 to 12 seconds, from a
`search_document` over the structural sheets. GPT-5.4 had them from its first `read_document` of
S-001, at 58 s. Every run reports both as direct conflicts, with the memo's section numbers and the
S-001 note numbers, all correct.

### Sol, baseline

- **Path.**
  - It read the memo as text and the structural sheets as text.
  - `find_quantities` on both documents.
  - Views of S-001, S-101, S-201, S-202, S-501, C-101, A-301 and A-501.
  - Four zooms on the S-501 schedule.
  - A calculation of the footing bottoms.
- **Reply.** Both conflicts (50 kPa or 33.3 % higher; 0.3 m shallower), plus:
  - **F4 missing from the schedule.** It says that F4 at grid C/2 on S-101 has no row on S-501.
    This is true. It is the fixture's `footing-not-scheduled` problem, found here unasked.
  - **Footing bottoms.** It works out bottoms of El. 97.60, 97.55 and 97.50 for F1 to F3 (top of
    footing 98.00 less 400, 450 and 500 mm). These are correct. All are above the memo's El. 97.0
    dewatering line.
  - **Finished grade.** It notes that no finished-grade elevation is given at the building, so
    actual embedment cannot be checked. This is true: C-101 gives only "slopes away at 2 % for 3 m".
  - **The memo's other recommendations.** It says they agree, and they do.
- **Delivered.** A Word memo with the same content.
- **Checks.** Both pass soundly. No false statements.

### Sol, show_me

- **Path.** Much the same, without `find_quantities`. It made five zooms on S-501. No `show_me`
  looks.
- **Reply.** Both conflicts, the embedment that cannot be verified, and F4. All true.
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### Sol, method

- **Path.** The same, with seven zooms (five on S-501, two on S-101) and a further text search.
- **Reply.** Both conflicts, plus:
  - the embedment that cannot be demonstrated;
  - F4;
  - "Dewatering condition is not carried into the drawings". This is true: the memo's 4.3 says
    "Dewatering is not expected for footings founded above El. 97.0" and no drawing repeats it. The
    reply itself calls this "an omission/verification issue rather than a demonstrated conflict".
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### Sol, sheet_tools

- **Path.**
  - `sheet_index` first; its result is correct and was not used for anything.
  - Memo pages and three structural sheets viewed.
  - At 62.8 s it viewed four more sheets. Two tiles of that batch (C-101 and A-101) never returned,
    and the model sat idle from about 112 s to 287 s until the 180-second limit re-ran them.
  - Ten zooms.
  - Then a helper (324.5 s to 441.6 s) to "Independently compare" the two documents. The helper
    re-opened both, re-read the text, re-viewed the memo and three sheets, made five zooms, and
    returned the same findings as a table.
- **Reply.** Seven items:
  - the two conflicts;
  - the dense sand bearing stratum, not named on the drawings (true);
  - the slab base "over dense sand", also not named (true);
  - the dewatering statement, not repeated (true);
  - F4 (true);
  - embedment that cannot be verified from elevations (true).

  The third to fifth items are true observations of things the drawings do not say. They are not
  differences in value.
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### Sol, all_three

- **Path.**
  - `sheet_index` and `check_cross_references` on the structural sheets. The latter found four
    callouts to S-501, all resolved, which is correct.
  - Text, views of eight sheets, three zooms, and a calculation.
- **Reply.** Both conflicts, F4, and the dewatering statement not repeated. All true.
- **Delivered.** A Word memo.
- **Checks.** Both pass soundly. No false statements.

### GPT-5.4, baseline

- **Path.**
  - Text of the memo and of S-001 and S-501.
  - Picture views of S-101 and S-501 only.
  - `search_document` for "F4". It returned the one hit, on S-101. Searches of S-501 found no F4,
    and the S-501 view listed only F1 to F3. Nothing in the model's own text ever mentions F4.
  - Its first model call took 37 seconds to return.
- **Reply.** The two conflicts, quoted exactly. Every agreeing item is listed and is true. It treats
  dense sand as an "unstated item, not a definite difference", which is true.
- **Planted F4.** Not part of this task's checks. GPT-5.4 had all the evidence for it in its tool
  results and did not report it.
- **Delivered.** No file; none was asked for.
- **Checks.** Both pass soundly. No false statements.

### Side by side

| Run | 200 vs 150 kPa | 1.2 vs 1.5 m | F4 unscheduled | Other true observations | False statements | Seconds |
|---|---|---|---|---|---:|---:|
| Sol baseline | delivered | delivered | yes | grade not given; bottoms 97.60/97.55/97.50 | 0 | 195.5 |
| Sol show_me | delivered | delivered | yes | grade not given | 0 | 182.7 |
| Sol method | delivered | delivered | yes | grade not given; dewatering not repeated | 0 | 220.4 |
| Sol sheet_tools | delivered | delivered | yes | dense sand not named (twice); dewatering; grade | 0 | 476.0 (175 s hung, 117 s helper) |
| Sol all_three | delivered | delivered | yes | dewatering not repeated | 0 | 184.1 |
| GPT-5.4 | delivered | delivered | evidence in its results, not reported | dense sand not named | 0 | 155.0 |

**Arm against chance.**
- All Sol arms gave the same answer, and all five found the unasked F4 problem.
- The `sheet_tools` run's extra time is the hung tiles and the helper. Neither is tied to the sheet
  tools: `sheet_index` was called and not used, and the helper was the model's own choice.
- None of the six runs remarked that S-001's 200 kPa is unclouded and has no revision entry. The
  fixture plants that for `coord-revisions`, and this question did not ask about it.

---

## The six questions

### 1. Are the passes sound? Do any checks pass on adjacent words?

There are 48 planted-problem verdicts here (24 runs, two each), and all 48 pass.

- **The one unsound pass.** 47 are sound: the reply states the problem as a finding. One is not:
  GPT-5.4's `index-title`. The check needs S-201, "roof" and "second floor framing" within 300
  characters. The GPT-5.4 reply's list of the sheets present has "PDF 13 S-201 — Second Floor
  Framing Plan" on one line and "PDF 14 S-202 — Roof Framing Plan" on the next, 55 characters
  apart. In this fixture S-202 is legitimately titled ROOF FRAMING PLAN and sits next to S-201. So
  any reply that lists the sheets in order passes `index-title`, whether or not it says there is a
  mismatch.
- **The same weakness elsewhere.** Every `stated_together` check has it in a reply that is an
  inventory: the level and bay checks would pass on a reply that lists every sheet's values without
  saying that any disagree. In these 24 runs only the `index-title` case happened. GPT-5.4's bay
  finding passes on two values 111 characters apart, but there the reply does state the
  disagreement, though hedged.
- **The two other checks.** All 48 verdicts of "does not give up on legibility" and "no 0-based page
  citation" are right. No run called anything unreadable. Every page citation I checked is 1-based
  and correct; for example, "A-201, Sheet 6" is the sixth sheet of 15.

### 2. False alarms, counted per run

A false alarm here is a statement about the documents that the fixture contradicts.

| Task | Sol baseline | Sol show_me | Sol method | Sol sheet_tools | Sol all_three | GPT-5.4 |
|---|---:|---:|---:|---:|---:|---:|
| sheet-index | 0 | 0 | 0 | 0 | 0 | 1 ("matches the set exactly") |
| levels-dimensions | 0 | 0 | 0 | 0 | 0 | 0 |
| drawings-vs-manual | 0 | 1 (memo: "five" issued separately) | 0 | 0 | 1 (memo: "five") | 0 |
| drawings-vs-geotech | 0 | 0 | 0 | 0 | 0 | 0 |

Items I did **not** count as false alarms, because they are true of the documents:
- the four issued-separately sections in the manual task;
- F4 in the geotech task;
- "finish as scheduled" with no finish schedule;
- the missing finished-grade elevations;
- the dewatering statement and the dense sand stratum not repeated on the drawings.

### 3. How did the levels-dimensions runs find bay B to C (7200 against 7500)?

Every run found it on A-401, and the question itself led there: it names "enlarged plans" and "grid
dimensions".

- **Sol baseline, show_me and method.** 7500 first reached these runs through the close-up tiles of
  `analyze_pdf_page` on A-401, at 67.9 to 94.1 seconds. The tiles answered a prompt the model wrote
  for that sheet. For example, the baseline asked for "grid labels and grid-to-grid dimensions" on
  the enlarged plans, and the tiles reported "7500" inside a revision cloud between the B and C
  bubbles.
- **Sol sheet_tools, all_three and GPT-5.4.** The value first came back in the text: `read_document`
  of page 7, at 21 to 59 seconds. The picture views came after.
- **Confirmation.** In every Sol run the main model then set 7500 against the 7200 it had read on the
  main plans, and confirmed it with a `render_region` zoom. The baseline also checked that the bubble
  to the right was C, and several runs re-read page 7's text.
- **GPT-5.4.** It did not zoom, and it hedged in its reply: A-401 might be dimensioning "a local
  enlarged-plan width".
- **Text tools that did not help.** `find_quantities` could not have found the dimension: it reports
  only numbers with units, and 7500 is printed bare. `search_document` would have found it in the
  text, but no run searched for it.

### 4. The extra work in show_me, method and all_three on drawings-vs-manual: did it change the answer?

No.

- **Order of work.** In all three runs the main agent first did the whole comparison itself: the text
  of both documents, all 15 sheets viewed, and the five manual pages viewed, some twice. In `method`
  it also re-viewed the manual in 2 by 2 tiles and made four zooms.
- **The helper.** Only then did it start a general-purpose helper with `task` to "independently
  audit", "validate" or "verify" the same comparison.
- **What the helpers did.** Each re-opened both PDFs and re-read their text. `show_me` and `method`
  re-viewed all 15 sheets and the manual; `all_three` re-viewed 10 sheets and the manual. Each
  returned the same conflict and the same five unresolved references.
- **The one addition.** The `all_three` helper added a true drawing-only item (door 107's hardware),
  which did not reach the reply.
- **The cost.** 110 to 170 seconds per run in the helper and 155,000 to 254,000 helper tokens. Each
  run used 0.4 to 0.7 million more side-call tokens than the baseline run of the task (this includes
  the main agent's own second look at the manual).
- **Why it happened.**
  - The written review method (in `method` and `all_three`) says to look at both places before
    reporting, which may encourage a second pass.
  - But `show_me` has no written method and did the same.
  - The baseline and `sheet_tools` runs of the same task did not.
  - With one run each, I cannot tie it to an arm.

### 5. The sheet index tool: its output, whether it is right, and why the model still viewed every sheet

The output was identical in the two sheet-index runs that had it (and in the geotech runs that
called it). It gave 15 pages, 15 sheets, 15 indexed, with the note "page is the 0-based PDF page
these tools take; sheet, title and rev are as the sheet's own title block prints them; views are
the view marks recognised on the sheet". Each row also carries the page and the 1-based PDF page.
The rows:

| Sheet | Title | Rev | Views |
|---|---|---|---|
| G-001 | COVER SHEET AND SHEET INDEX | 1 | |
| G-002 | GENERAL NOTES | 0 | |
| C-101 | SITE PLAN | 0 | 1 |
| A-101 | FIRST FLOOR PLAN | 0 | 1 |
| A-102 | SECOND FLOOR PLAN | 0 | 1 |
| A-201 | EXTERIOR ELEVATIONS | 0 | 1, 2 |
| A-301 | BUILDING SECTIONS | 0 | 1, 2 |
| A-401 | ENLARGED PLANS | 1 | 1, 2 |
| A-501 | DETAILS | 0 | 1 to 6 |
| A-601 | DOOR SCHEDULE | 1 | |
| S-001 | STRUCTURAL GENERAL NOTES | 0 | |
| S-101 | FOUNDATION PLAN | 0 | 1 |
| S-201 | SECOND FLOOR FRAMING PLAN | 0 | 1 |
| S-202 | ROOF FRAMING PLAN | 0 | 1 |
| S-501 | FOUNDATION DETAILS AND FOOTING SCHEDULE | 0 | 1 to 4 |

- **Is it right?** Every row matches the sheet's title block in the fixture, revisions and views
  included.
- **What it does not do.** It does not read the cover's index table. `n_indexed` counts the sheets
  whose own number it found, not the cover's 16 rows. So the tool does not itself report either
  planted problem. The comparison still has to be made, and both runs made it from the cover's index
  table, which `read_document` returned.
- **Why every sheet was still viewed.**
  - In `sheet_tools`, the model asked for the index and all 15 page views in one step, before the
    index came back, citing "the rules mention title blocks ... I probably need to look at all 15
    pages closely".
  - In `all_three`, the model had both problems from the index and the text by its third call, and
    then viewed the remaining 14 sheets because "the instructions say to look at every page that the
    question covers".
  - That is the page's standing rule (look at every page in scope; text is evidence, not a filter).
    It is a deliberate one. While it stands, the sheet tools cannot show a saving in looking on a
    question like this.

### 6. In show_me and all_three, how often did the main model look itself, against a side call?

Never.

- **What the record shows.** None of the eight `show_me` and `all_three` runs in this group made a
  single look with `show_me` set to true. Every one of their `analyze_pdf_page`, `render_region` and
  `analyze_image` calls went to vision side calls.
- **Across the suite.** A search of every Sol run's record in this brief found no `show_me` look at
  all.
- **Was the switch on?** Yes, as far as the record can show: `run.json` gives the arm's setting
  (`GEOTECH_VISION_SHOW_ME` = 1), and the offered tool set's fingerprint differs from baseline.
- **What the record cannot show.** The record keeps only the tool names and a fingerprint, not the
  tool descriptions. So it cannot show what the model was told about the argument.
- **What follows.** In this group, the `show_me` arm's results are a second baseline.

---

## Follow-ups

### Proposed (a defect in a scorer, a tool or the record, which would hold on any model)

1. **The `index-title` check passes on a plain list of the sheets.**
   - What happens: in this fixture, S-201's real title and S-202's ROOF FRAMING PLAN sit on adjacent
     lines of any ordered list of the sheets, so "S-201", "roof" and "second floor framing" fall
     within the window without any finding being stated (GPT-5.4, `coord-sheet-index`).
   - The proposal: require the three terms in one line or one list item, or require "roof" to fall
     nearer to S-201 than to S-202.
   - Then: check the other `stated_together` checks for the same inventory pass. Levels and bay are
     the likely ones.
2. **A hung vision tile costs the full 180-second call limit.**
   - Evidence in this group: two runs lost about three minutes each (`coord-sheet-index` baseline;
     `coord-drawings-vs-geotech` `sheet_tools`, with two tiles).
   - Across the suite's 84 runs: the median tile takes 5 seconds and 99 % take under 22 seconds; 34
     tiles never returned, and three tiles in one run (`coord-long-review`, `sheet_tools`) returned
     after 905 seconds.
   - The proposal: a tile limit nearer the observed tail (for example 60 seconds) with the existing
     retry, or a second request after a set time.
3. **`find_quantities` reads a rebar callout as a length range.**
   - What happens: "10-20M EACH WAY" and "8-15M EACH WAY" on S-501 came back as "10 to 20 m" and
     "8 to 15 m" (length) in `coord-levels-dimensions` baseline.
   - Why it matters: a wrong value in a tool result is the kind that gets copied into a deliverable.
     The proposal is to read a count and bar size ("10-20M") as one rebar callout, not as a range.
4. **The `show_me` arm needs to show in the record that it was used.**
   - What happens: `run.json`'s tool fingerprint shows only that the offered tools differed from
     baseline's, not that `show_me` was among the arguments or how it was described;
     `activity.jsonl`'s per-turn hash covers the tool names only; and no run used it.
     (Wording corrected by the coordinator: this line first said neither file could show the
     argument was offered.)
   - The proposal: record the offered tool descriptions, or at least whether `show_me` was among the
     arguments, and report in RESULTS how many looks per run used `show_me`. Then a `show_me` arm
     that never looked for itself is visible as that, not read as a measurement of DR-38.
   - This proposes no prompt change.
5. **Time the record does not explain.** In `coord-sheet-index` `all_three`, 123 seconds passed
   between the process starting and the turn starting, and 56 seconds between the turn starting and
   the first model call. Nothing is recorded in either gap. The "seconds" figure in RESULTS includes
   both, and arms are compared on it. The proposal: record what the time before the first model call
   was spent on (setup, or a wait for a busy service).
6. **The `coord-drawings-vs-manual` truth sentence is looser than the manual.** "The other cited
   sections are in the manual" leads a reader to count four true observations as false alarms in
   every run. The proposed wording: "the other cited sections are listed in the manual's contents;
   01 33 00, 03 30 00 and 08 71 00 are bound in this volume and the rest are listed as issued
   separately".

### Noted, not proposed

- **Helpers that redo finished work.** Four runs started a helper to redo, "independently", work the
  main agent had finished: three in drawings-vs-manual, one in drawings-vs-geotech. Each roughly
  doubled the side-call cost and added two to three minutes, and none changed a reply. This is the
  model's own choice.
- **Viewing every sheet after the text had already answered.** In sheet-index (all runs) and in
  drawings-vs-manual (every planted item is text), the views confirmed and found nothing new. This
  follows the deliberate rule to look at every page in scope.
- **The "not searchable as text" label.** `search_document` and `find_quantities` list the six
  ruled-form sheets (pages 3-4, 7, 9 and 12-13) under `pages_not_searchable_as_text` and call their
  numbers "ink, not text". All six carry a full text layer, and searches in these same runs returned
  hits from them. The label errs towards looking, and it changed no answer here.
- **GPT-5.4's misses.** It delivered the S-201 index title only by accident of the check. It had F4
  in its results in the geotech task and did not report it. It hedged the bay. Each is a model-level
  miss, and no prompt change is proposed for them.
- **The two Word memos' count of five.** In drawings-vs-manual (`show_me`, `all_three`), the memo
  counts five issued-separately sections where there are four. The replies count correctly. This is
  a model slip.
