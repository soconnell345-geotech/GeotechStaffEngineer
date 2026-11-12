# Callout tasks: the full read of 18 runs

Group `callouts`: the three tasks that ask whether each section, detail and
enlarged-plan callout in the NORTH YARD ANNEX drawings points to a view that
exists on the sheet it names.

- `coord-callouts`: the 15-sheet set, with a text layer.
- `coord-scan-callouts`: the same 15 sheets as a scan with no text layer.
- `coord-long-callouts`: the 40-sheet set, with a text layer.

Each task ran six times: GPT-5.6 Sol in five arms (baseline, show_me, method,
sheet_tools, all_three) and GPT-5.4 in the baseline arm. For every run I read
the whole record: every tool call with its arguments and result, every vision
side call, the model's own text and reasoning summaries, the helpers' records,
the answer, the Word file and the checks. I checked each callout the vision
calls read against the fixture.

Raw records are under
`module_work/field_feedback/2026-11-12_foundry_brief7/raw/b7_coord_sol/runs/<arm>/<task>/`
and `.../raw/b7_coord_gpt54/runs/baseline/<task>/`.

## What the drawings hold (ground truth)

The fixture is `review_eval/coordination_fixture.py` in `aec-document-review`.

**The 15-sheet set** (PDF pages 1 to 15) holds 17 callouts on six sheets:

- A-101 has 1/A-301, 2/A-301, 1/A-401 (an enlarged-plan bubble with the words
  "ENLARGED PLAN" beside it), 3/A-501 and 7/A-501.
- A-102 has 1/A-301, 2/A-302, 2/A-401 and 5/A-501.
- A-301 has 2/A-501, 4/A-501 and 6/A-501.
- A-401 has 1/A-501.
- S-101 has 1/S-501, 2/S-501 and 3/S-501.
- S-201 has 4/S-501.

Two of them are broken:

- 7/A-501 on A-101 (PDF page 4): A-501 holds details 1 to 6 only.
- 2/A-302 on A-102 (PDF page 5): there is no sheet A-302.

**The 40-sheet set** holds 39 callouts on 17 sheets. The floor plans A-103 to
A-105 repeat 1/A-301 and 2/A-301 with 3/A-501 or 5/A-501. The reflected ceiling
plans A-111 to A-115 call details on A-502. S-202 to S-204 each call 4/S-501.
There is a third broken callout:

- 9/A-502 on A-114 (PDF page 12): A-502 holds details 1 to 8 only.

The checks give one `stated_together` check per broken callout: the callout
and a word of absence within 200 characters of each other. They also check
that the answer does not give up on legibility and does not cite a 0-based
page. Pages read is recorded but not scored. False alarms are not scored.

## Results at a glance

| Task | Arm | Passed | Pages looked at | Helpers | Model calls | Tokens | Seconds | Hung vision calls | Word file |
|---|---|---|---|---|---|---|---|---|---|
| 15-sheet | baseline | 4/4 | 15 | 0 | 173 | 1.09M | 207 | 0 | yes |
| 15-sheet | show_me | 4/4 | 15 | 0 | 246 | 1.45M | 238 | 0 | yes |
| 15-sheet | method | 4/4 | 15 | 0 | 161 | 1.33M | 365 | 1 | yes |
| 15-sheet | sheet_tools | 4/4 | 15 | 0 | 172 | 1.14M | 324 | 1 | yes |
| 15-sheet | all_three | 4/4 | 12 (and page 1 as text) | 0 | 141 | 0.96M | 163 | 0 | no |
| 15-sheet | GPT-5.4 | 4/4 | 12 | 0 | 135 | 0.81M | 258 | 0 | no |
| scan | baseline | 4/4 | 15 | 3 | 289 | 1.52M | 469 | 2 | yes |
| scan | show_me | 4/4 | 15 | 0 | 174 | 1.23M | 407 | 1 | yes |
| scan | method | 4/4 | 15 | 0 | 171 | 1.07M | 235 | 0 | yes |
| scan | sheet_tools | 4/4 | 15 | 0 | 174 | 1.17M | 258 | 0 | yes |
| scan | all_three | 4/4 | 15 | 0 | 277 | 1.65M | 342 | 0 | yes |
| scan | GPT-5.4 | 4/4 | 15 | 0 | 272 | 1.42M | 456 | 0 | no |
| 40-sheet | baseline | 5/5 | 40 | 4 | 654 | 3.32M | 772 | 3 | yes |
| 40-sheet | show_me | 5/5 | 40 | 4 | 608 | 3.44M | 736 | 1 | yes |
| 40-sheet | method | 5/5 | 40 | 1 | 480 | 3.02M | 862 | 2 | yes |
| 40-sheet | sheet_tools | 5/5 | 40 | 0 | 427 | 2.67M | 841 | 3 | yes |
| 40-sheet | all_three | 5/5 | 40 | 1 | 458 | 2.41M | 572 | 1 | no |
| 40-sheet | GPT-5.4 | **4/5** | 13 | 0 | 148 | 0.90M | 254 | 0 | no |

**Sol: correct in all 15 runs.** All 15 Sol answers name every broken callout
with the right reason, and none of them names a false alarm.

**GPT-5.4: the checks are generous.** Its verdicts on the 15-sheet set and the
scan are passes, but both answers contradict themselves (set out below). On
the 40-sheet set it named 18 false alarms and missed 9/A-502.

**Where the model calls go.** Almost all of them are vision side calls. Each
`analyze_pdf_page` costs one overview call plus one call per tile: 4, 9 or 16
tiles for 2x2, 3x3 or 4x4. Each zoom costs one call, and five probe calls
measure the vision model once per run. The main model itself made 9 to 36
calls per run.

Tokens per run follow the tile choice. show_me on the 15-sheet set and
all_three on the scan chose 4x4 tiles: 212 and 240 tile calls, against 135
for 3x3 on every page. On the 40-sheet set they follow how many helpers there
were.

## Proposed follow-ups

Each of these is a defect in the scorer, a tool or the record. Each would
hold whatever model reads the result.

### 1. Scorer: an answer that calls good callouts broken still passes

The callout tasks have no check against false alarms. GPT-5.4's 40-sheet
answer lists 20 callouts as broken: every callout on A-101 to A-105 that
names A-301 or A-501, and every 4/S-501 on S-201 to S-204, all with the reason
"sheet ... not in set". Only two of the 20 are planted problems, so 18 are
false alarms.

It still passed `callout-missing-detail` and `callout-missing-sheet`. For
7/A-501 it got the credit with the wrong reason: it says A-501 is not in the
set, when A-501 is PDF page 17. It failed only the ceiling-detail check,
because it never looked at A-114.

The fixture knows every good callout: 15 on the 15-sheet set and 36 on the
40-sheet set. A check could fail an answer that names a good callout beside a
word of absence.

The window needs care. `_ABSENT` includes "only" and "unresolved", and Sol's
answers put sentences like "All other callouts resolve, including 1/A-301,
2/A-301 ..." close to the table of broken ones. So the check should match
within one line or one table row, not within 200 characters.

On these records the check would fail GPT-5.4's 40-sheet answer. It would
probably also fail its 15-sheet answer, whose body says 4/A-501 and 6/A-501
"do not point" to a view. It should pass all 15 Sol answers. That needs to be
confirmed when the check is built.

### 2. planlens tool result: the page list leaves out real drawing sheets

`open_document` and `document_page_map` mark pages to view by their page kind.

- **15-sheet set.** `pages_to_view` lists 6 of the 15 pages (A-101, A-102,
  A-401, A-601, S-201, S-202).
- **40-sheet set.** It lists 18 of the 40 pages.

The rest are classed "text" or "mixed", among them every destination sheet:
A-301, A-501, A-502 and S-501. Yet every page is a 34 x 22 in drawing sheet
with a title block. `sheet_index` indexes all 40 sheets by their title blocks.

The note beside the list says "these pages are scans, figures or drawing
sheets: their text is labels at best, so read them by viewing them". The page
map's `look` flag says the same. A reader is entitled to take that to mean the
unlisted pages are not drawings.

GPT-5.4's 40-sheet run took the list as the set:

- It asked for the page map and the thumbnails of only those 18 pages, and
  searched only for the sheet numbers that page map showed (A-401, A-601,
  A-602, S-201 to S-205).
- It looked at 13 of them and read no page as text.
- It then wrote that A-301, A-302, A-501 and S-501 are absent "anywhere in
  this 40-page set". That sentence is the source of its 18 false alarms.

Two 15-sheet runs, all_three (Sol) and GPT-5.4, skipped exactly the
three "text" pages G-001, G-002 and S-001. That was harmless there, because
those sheets have no callouts. The four other Sol runs looked at every page
whatever the list said.

Proposed fix: list every drawing-size sheet that has a title block in
`pages_to_view` and mark it `look`. Failing that, say in the note that
"text" and "mixed" pages can still be drawing sheets whose drawings must be
viewed.

### 3. Record: a vision call that times out leaves no trace

A vision side call that gives no answer within 180 s is abandoned and asked
once more (`vision_engine.invoke_side_call`, `DEFAULT_TIMEOUT_S = 180`,
`TIMEOUT_RETRIES = 1`). The only notice is a `log.warning`.

That warning is not in the task console logs, and `activity.jsonl` has no
event for it. The abandoned call leaves a `model_start` with no `model_end`.
The second ask has its own start and end and records a duration of 1 to 2 s.
The results table shows "failed calls 0".

This happened 15 times in 10 of the 15 Sol runs, and never in GPT-5.4's three. Each
abandoned call was asked again 180 s later (once 234 s), and the second ask
answered in 1 to 2 s.

The model waits for a whole batch of pages before it goes on, so each hang
held the run up.

- **sheet_tools, 40-sheet.** Three tile calls hung. Two batches of ten pages
  took 231 s and 232 s against 75 s and 91 s for the other two. That is about
  350 s of an 841 s run.
- **15-sheet runs.** The hung page took 199 s in sheet_tools and 236 s in
  method, against at most 75 s and 107 s for the pages beside it. That is why
  those two were the slowest 15-sheet Sol runs (324 s and 365 s, against
  207 s for baseline).

So the arms' times cannot be compared without this.

Proposed fix: write an event to `activity.jsonl` when a side call times out,
giving the limit, the call's kind and page, and when it was asked again.
Count those in the results' "failed calls" column.

With that in the record, the next run can measure whether 180 s is still the
right limit. Brief 5 chose it when the 99th-percentile side call took 74 s.
Here the pages beside a hung call took a median of 34 to 93 s.

### 4. Record: the activity record alone cannot show the show_me argument was offered

In all six show_me and all_three runs the model never asked to look itself
(no call with `show_me=true`). The switch line shows
`GEOTECH_VISION_SHOW_ME=1`, and the run-level tool fingerprint in `run.json`
(`fingerprint.tools.sha`, a hash of every tool's name, description and
argument schema) differs from baseline's, so the argument was offered.

But `turn_end` in `activity.jsonl` records only the names of the tools
offered (`tools_offered_sha` hashes the names), and that value is identical
in baseline and show_me. So the activity record on its own cannot show the
argument was in the schema the model saw. (Corrected by the coordinator:
this section first said the record as a whole could not confirm it.)

Proposed fix: record a hash of each tool's schema, or the argument names,
beside the names.

### 5. Task text: wrong sheet count

The truth text of `coord-long-callouts` (`tasks.py`, about line 1322) says
"39 callouts on 22 of them". The fixture puts the 39 callouts on 17 sheets.
The text is read by reviewers, not by the model.

## Noted, not proposed

- **all_three on the 15-sheet set claims to have looked at all 15 pages.** It
  looked at 12 and read page 1 as text. It never looked at or read G-002 or
  S-001, yet wrote "I visually examined all 15 PDF pages". Its reasoning had
  planned to skip the notes and cover pages. Its conclusion about those two
  sheets rests only on text tools:
  - `sheet_index`, which lists no views on them;
  - `check_cross_references`, which finds no callouts on them;
  - the page map, which classes them as text.

  All three are right here. The false coverage sentence is the model's.
  Pages read is recorded but not scored, so nothing caught it.
- **GPT-5.4's answers contradict themselves** on the 15-sheet set and the
  scan (details under each task). This is the model's behaviour.
- **Whole-sheet vision reads misread small numbers.** The overview read
  A-401's 1/A-501 as "1/A-601" (GPT-5.4 on the 15-sheet set; Sol in the
  40-sheet sheet_tools run). GPT-5.4's overview of A-301 read 2, 4 and 6/A-501
  as "/A-301". In every case the tiles read them correctly. No Sol answer was
  affected.
- **The enlarged-plan callout looks like a view title.** In Sol's 15-sheet
  baseline run, the tiles took A-101's bubble 1/A-401, with "ENLARGED PLAN"
  beside it, for a view title on A-101 ("Sheet ID: A-401"). A zoom corrected
  it ("outgoing callout on A-101").
- **Helpers run two at a time** (a deliberate setting). Queueing cost about
  195 s in the 40-sheet baseline run (two helpers waited about 192 s and
  198 s), 188 s and 329 s in the 40-sheet show_me run, and about 274 s in the
  scan baseline run.
- **Repeated work.**
  - In the 40-sheet all_three run, the helper called
    `check_cross_references` again ten seconds after the main model had. The
    main model then re-zoomed the three callouts the helper had already
    zoomed and re-viewed A-501 and A-502.
  - In the scan baseline run, each of the three helpers re-opened the PDF and
    listed the files.
  - In the 15-sheet method run, `render_region` was used on all of A-501 (a
    box over the whole sheet), which is a whole-page look made with the zoom
    tool.
- **Sheet_tools, 15-sheet: a search result's wording disagrees with itself.**
  `search_document` listed the plan pages as "pages_not_searchable_as_text"
  and in the same result found A-302 on one of them.
- **Sheet_tools, 40-sheet: unexplained time.** 37 s pass between the start of
  the turn and the first model call, and 75 s between the end of the turn and
  the end of the run. Nothing is recorded in either gap. Every other run
  starts within 1.6 s and ends within 15 s.
- **The scan's empty result points nowhere.** On the scan, `check_cross_references`
  honestly found nothing ("15 sheet(s) have no usable text: their callouts are
  not in this list"), but gave no `look` pointer. The model looked anyway.
- **GPT-5.4's record has no reasoning summaries,** so why it skipped A-111 to
  A-115 (all in its own page list) cannot be read from the record.
- **all_three wrote no Word file** on the 15-sheet set or the 40-sheet set.
  The tasks do not ask for one; the other Sol runs wrote one unasked.

## coord-callouts: the 15-sheet set with a text layer

### Sol, baseline (207 s)

**Path.** It opened the set and looked at all 15 pages with auto tiles (3x3).
It made six zooms: the two broken callouts and A-501's view titles among
them.

**Planted problems.** Both found and delivered.

**Wrong statements.** One misreading, set right in the run: the tiles took
A-101's enlarged-plan callout for a view title, and the model zoomed and got
"outgoing callout on A-101".

**Answer.** Both broken callouts, the rest "resolve" with no list. A Word file
was written. The check verdicts are right.

### Sol, show_me (238 s)

**Path.** 4x4 tiles on 11 pages and 3x3 on 4, which is why it used the most
tokens (1.45M). Two zooms. It never used show_me.

**Answer.** Both broken callouts. It adds that A-302 was likely meant to be
A-301, which is the fixture's own note. A Word file was written. The check
verdicts are right.

### Sol, method (365 s)

**Path.** 3x3 and 2x2 tiles, `read_document` text, and seven zooms, one of
them over the whole of A-501.

**Time.** One tile on page 6 hung for 180 s. That page took 236 s against at
most 107 s for the pages beside it.

**Answer.** Both broken callouts. The Word file lists all 15 callouts that
resolve. The check verdicts are right.

### Sol, sheet_tools (324 s)

**The tool call.** `check_cross_references` was called once, 8 s in, beside
`sheet_index`, with only the document handle. The full result counted 17
callouts:

- 15 resolved;
- 7/A-501 unresolved ("view marks recognised on A-501: 1 to 6; none for 7",
  with its box on PDF page 4);
- 2/A-302 not in the document ("no sheet A-302 was read in this document; the
  document holds 15 sheets", with its box on PDF page 5).

That matches the fixture exactly. `sheet_index` listed each sheet's views
correctly.

**What the model did with it.** It did not take the result on trust:

- It still looked at all 15 pages.
- It zoomed both flagged callouts using the tool's own boxes.
- It read the cover sheet's index (the tool's note says to compare it).
- It searched for A-302 and zoomed A-501's view titles.

**Answer.** It follows the tool and is checked by eye. It is the only answer
in the group that lists, sheet by sheet, all 15 callouts that resolve. A Word
file was written.

**Time.** One hung tile, 180 s, made its page take 199 s against at most 75 s
for the pages beside it. Without it the run would have taken about as long as
baseline.

### Sol, all_three (163 s)

**The tool call.** `check_cross_references` (pages "0-14") gave the same exact
result. The answer opens with the tool's counts: "17 reference callouts. Of
these, 15 resolved and 2 did not". It zoomed both flagged callouts at the
tool's boxes.

**The render_region misreading.** `render_region` called 7/A-501 a "section
reference". The answer correctly calls it a detail.

**Coverage.** It looked at 12 pages, and never looked at or read G-002 or
S-001. Its claim to have examined all 15 pages is false (see "Noted, not
proposed").

**Answer.** Right on both broken callouts. No Word file.

### GPT-5.4, baseline (258 s)

**Path.** It looked at 12 pages, skipping G-001, G-002 and S-001, and made no
zooms.

**Misreadings.**

- Its vision overview read A-301's callouts as 2, 4 and 6/A-301, and the
  model itself judged 4/A-301 and 6/A-301 unresolved.
- The overview read A-401's callout as 1/A-601.

**Answer.** It contradicts itself:

- It first says only 2/A-302 is broken.
- It lists 6/A-501 and 4/A-501 among those that "do not point" and then says
  they are "actually valid".
- It names C-101, A-201, A-601 and S-202 as sheets with callouts. They have
  none.
- It ends with a "Correction" giving 7/A-501 and 2/A-302.

**Check verdict.** Both checks passed on that last list. Taken as a whole,
the answer is not one a reviewer could hand on.

### Side by side

All five Sol arms got the same, correct answer. Nothing in this set separates
the arms; any difference between one run and another is chance.

**Time.** It follows the hung calls (method and sheet_tools had one each) and
the tile choice. all_three was fastest and cheapest partly because it skipped
three pages and wrote no file.

**What sheet_tools changed.** The tool handed the exact answer over in under a
second, 8 s into the run, and the model then spent the rest of the run
confirming it by eye. Its zooms were aimed at the tool's boxes: four, against
six in baseline. Its answer was fuller (it lists the resolving callouts). It
was not cheaper or faster: 172 calls against 173.

## coord-scan-callouts: the same 15 sheets as a scan

`open_document` reports every page as a drawing sheet with no text layer. The
pair readings below were checked against the fixture. Sol's tiles read every
callout correctly.

### Sol, baseline (469 s)

**Path.** Three general-purpose helpers took PDF pages 1-5, 6-10 and 11-15,
two at a time; the third waited about 274 s. Each helper re-opened the PDF and
listed the files. They made 37 zooms between them, mostly on title blocks.
Two helper tile calls hung for 180 s.

**Answer.** Right; the helpers' kept outputs agree with it. The Word file
places 2/A-302 at the "upper-right / right edge" of A-102, when it is near the
top middle.

### Sol, show_me (407 s)

**Path.** Batches of five pages at 3x3. One hung tile in the first batch
stretched that batch to about 200 s. It made four zooms (both broken
callouts, A-301 and A-501's view titles) and never used show_me.

**Answer.** Right. The Word file lists the 15 callouts that resolve, sheet by
sheet.

### Sol, method (235 s)

**Path.** All 15 pages at 3x3 at once, then four zooms.

**Answer.** Right, and it says the PDF has no searchable text layer.

### Sol, sheet_tools (258 s)

**The tool calls.**

- `sheet_index` returned none indexed: "no text layer on this sheet: read its
  title block by looking".
- `check_cross_references` (pages "0-14") returned every count at zero, with
  "15 sheet(s) have no usable text: their callouts are not in this list" and
  "no callouts were recognised in the text".

Both are honest.

**What the model did.** It looked at all 15 pages and zoomed 7/A-501, 2/A-302
and A-501's titles.

**Answer.** Right; it says the PDF has no searchable text layer.

### Sol, all_three (342 s)

**Path.** It called `sheet_index` (no text layer) but not
`check_cross_references`. It used 4x4 tiles on every page (240 tile calls,
1.65M tokens) and made four zooms.

**Answer.** Right. It also notes that the cover index lists A-202 INTERIOR
ELEVATIONS, which is not in the PDF. That is true; it is the sheet-index
group's planted problem.

### GPT-5.4, baseline (456 s)

**Path.** It looked at all 15 pages (auto chose 4x4) and made no zooms.

**Answer.** It contradicts itself:

- It opens "I did not find any ... callouts that point to a non-existent
  view".
- It lists 7/A-501 among callouts that "point to sheets/views that exist".
- It then flags 7/A-501 as "the only included-sheet mismatch".
- It puts 2/A-302 under "Not counted as a mismatch within this set, but
  notable".
- It leaves A-301 out of the callout-bearing sheets, though its own tiles
  read 2, 4 and 6/A-501 there. They resolve, so the omission changes no
  finding.

**Check verdicts.** Both checks passed. The `callout-missing-sheet` pass is
generous: the answer says in so many words that it does not count 2/A-302.

### Side by side

All five Sol arms are right. The arms differ only in tile size, helpers and
hung calls. The sheet tools can do nothing without a text layer and said so.

## coord-long-callouts: the 40-sheet set with a text layer

### Sol, baseline (772 s)

**Path.**

- It ran searches for SECTION and DETAIL and a contact sheet.
- It sent four general-purpose helpers, one per ten PDF pages. Two ran at
  once and two waited about 192 s and 198 s.
- Three helper vision calls hung for 180 s or more.
- The helpers' tables list every callout on their pages correctly.
- The main model then zoomed the three exceptions, looked at A-501 and A-502
  again and read the cover sheet's text.

**Answer.** All three broken callouts, no false alarms. The Word file
matches the answer.

### Sol, show_me (736 s)

**Path.**

- Four helpers. The two waiting ones queued 188 s and 329 s.
- One helper tile call hung.
- The main model added up the helpers' counts (`calculate`: 22 + 10 + 7 = 39
  callouts).
- It searched for A-302, A-501 and A-502 and looked again at A-101, A-102,
  A-114, A-501 and A-502.
- It never used show_me.

**Answer.** "Of 39 qualifying callouts, 36 resolve ... and 3 do not". Right,
and the counts are the fixture's. A Word file was written.

### Sol, method (862 s)

**Path.**

- The main model read all 40 pages as text.
- One helper audited the whole set: it looked at all 40 pages in 744 s, with
  two hung calls.
- The main model then re-viewed the three source sheets and A-501 and A-502.

**Answer.** Right. The Word file lists the destination views it checked.

### Sol, sheet_tools (841 s)

**The tool call.** `check_cross_references` was called once, 47 s in, with
only the handle. The full result counted 39 callouts:

- 36 resolved;
- 7/A-501 unresolved ("view marks recognised on A-501: 1 to 6; none for 7");
- 9/A-502 unresolved ("view marks recognised on A-502: 1 to 8; none for 9");
- 2/A-302 not in the document ("the document holds 40 sheets").

Each came with its page and box. That is exact.

**What the model did.** It then looked at all 40 pages itself, ten at a
time, with no helpers. It zoomed the three flagged callouts at the tool's
boxes, the lower halves of A-501 and A-502, and the cover index.

**Time.** Three tile calls hung, adding about 350 s. There are also the 37 s
before the first model call and the 75 s after the turn that nothing explains.

**Answer.** Right. The Word file says every sheet from G-001 to E-601 was
examined.

### Sol, all_three (572 s)

**Path.**

- `check_cross_references` (pages "0-39") at 10.6 s gave the same exact
  result.
- One general-purpose helper "independently" audited the set. It called
  `sheet_index` and `check_cross_references` again at 20.5 s, with the same
  result.
- The helper looked at all 40 pages in 508 s, with one hung call, and zoomed
  the three callouts, both rows of A-501 and A-502, and the index.
- The main model then repeated the three zooms and the two destination
  looks.

**Answer.** It gives the tool's counts as its own coverage: "checked all 39
... 36 resolve correctly; 3 do not". The counts are right. No Word file.

### GPT-5.4, baseline (254 s): the failed run

**What it did.**

1. It opened the PDF. `open_document` returned `pages_to_view`
   "3-12,15,18-19,22-26" (PDF pages 4-13, 16, 19-20, 23-27) with the note that
   these are the drawing sheets to view.
2. It asked for the page map and the thumbnails of those pages only.
3. The thumbnail read called pages 3-12 and 22-26 plans, and 15, 18 and 19
   (A-401, A-601, A-602) the likely destination sheets.
4. It searched the text for "A-401|A-601|A-602|S-201 ... S-205". Those are the
   sheet numbers the restricted page map had shown it, so the search could
   never find A-301, A-501, A-502 or S-501.
5. It looked at PDF pages 4-8 (A-101 to A-105), 16, 19, 20 and 23-27. It asked
   for the destination views on A-601 and A-602, which are schedules.
6. It skipped pages 9-13 (A-111 to A-115), although they were in its own list
   and its thumbnail read had called them plans. It also skipped S-101 (a
   "mixed" page) and never viewed or read A-301, A-501, A-502 or S-501.
7. It wrote the answer at 249 s.

**Why it stopped.** Nothing stopped it: there was no error, no budget limit
and no step cap. The record holds no reasoning summary from GPT-5.4, so it
cannot say why A-111 to A-115 were dropped.

**Answer.**

- It lists 20 callouts as broken because "A-301 / A-501 / S-501 not in set".
  Only 2/A-302 is broken for the reason it gives. 7/A-501 is broken, but not
  because A-501 is missing. The other 18 are false alarms.
- It calls 1/A-401 and 2/A-401 valid.
- It never mentions 9/A-502, the A-111 to A-115 callouts, S-101's callouts or
  the callouts on A-301 and A-401.
- Its coverage note is honest that it looked only at "A-101 to A-105 and
  S-201 to S-205" and the destination sheets present. But it states the
  absence of A-301, A-302, A-501 and S-501 "anywhere in this 40-page set" as
  fact.

**Check verdicts.** The ceiling-detail failure is right. The two passes are
generous: one was earned with the wrong reason and both sit among 18 false
alarms (proposed follow-up 1). The tool result that steered it is proposed
follow-up 2.

### Side by side

All five Sol arms are right and agree. The spread in time (572 to 862 s) and
tokens (2.41M to 3.44M) follows three things:

- how the work was split: four helpers, one helper, or none;
- helpers waiting two at a time;
- hung vision calls.

It does not follow the arms. With one run per arm, the remaining difference
is chance.

**What the sheet tools changed.** The two runs that called
`check_cross_references` (sheet_tools and all_three) had all three broken
callouts in hand within 47 s and 11 s. Both still looked at all 40 pages,
which is what their prompts ask. Both aimed their zooms at the tool's boxes:
6 and 12 zooms, against 46 in baseline, whose helpers zoomed title blocks and
rows.

They used fewer tokens than baseline (2.67M and 2.41M against 3.32M), but
method, with one helper and no sheet tools, used 3.02M. So the saving belongs
to having fewer helpers as much as to the tool.

No answer contradicted the tool or ignored it. Every flagged callout was
checked by eye before it was reported.
