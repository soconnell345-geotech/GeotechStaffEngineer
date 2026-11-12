# Brief 7, revisions group: coord-revisions and coord-submittal

Twelve runs: each task in the five GPT-5.6 Sol arms (baseline, show_me,
method, sheet_tools, all_three) and in the GPT-5.4 baseline. Every run's
full activity record was read: each model step's reasoning summary and
text, every tool call with its arguments and result, every helper's brief
and report, and every page reading. The delivered Word files were read too.
Each value the runs acted on was checked against the fixture
(`coordination_fixture.py`) and the task definitions (`tasks.py`). A number
in square brackets is a record's line in that run's `activity.jsonl`,
counting from 0. "Page 11" means the eleventh page of the PDF.

## What this group shows

1. **coord-revisions.** Five of the six runs found all four changes and
   said correctly, for each one, whether it was clouded and whether it was
   listed. The sheet_tools run missed the one change nobody had marked: the
   bearing pressure in S-001 note 2, which went from 150 kPa to 200 kPa
   with no cloud and no revision entry. It read the text of only the sheets
   that carried revision marks. It then asked a single page reading what
   had changed on issue 1's S-001, and a reading of one page cannot answer
   that. One of the baseline run's two helpers made the same miss in the
   same way. The second helper and the primary agent put it right. So the
   record shows a method that finds only flagged changes. One run is not
   enough to call it an effect of the arm.
2. **coord-submittal.** All six runs found all four planted problems. Three
   runs also reported that the transmittal says "DEVIATIONS: NONE" while
   the mix departs from the specification. Section 01 33 00 says a
   submittal with unmarked deviations is returned. Those three runs
   (sheet_tools, all_three, GPT-5.4) are exactly the ones that read Section
   01 33 00. The method and show_me runs noticed the line in their
   reasoning but left it out of the answer. The task's truth text names
   this fact, but no check scores it.
3. **Cost.** On the submittal task, steps that did nothing but update the
   to-do list (`write_todos`) used 25 to 42 percent of each run's input
   tokens, on both models. The baseline coord-revisions run cost 1.6 to 1.8
   times what the other Sol arms cost because its two helpers each reviewed
   the whole set.
4. **Gaps in the record.** One model call took 904 seconds, and nothing in
   the record says why: the record has no retry or wait events at all.
   Every coord-revisions Sol run has one or two picture-reading calls that
   started and never ended, and no timeout event says what happened to
   them.
5. **Two answers link to the Word file with `sandbox:/mnt/data/...`.** That
   link goes nowhere in the page. The file's download card under the reply
   is the route that works.

## Scores, time and tokens

| Run | Task | Checks | Seconds | Model calls | Tokens in | Tokens out | Files delivered |
|---|---|---|---|---|---|---|---|
| Sol baseline | coord-revisions | 5/5 | 1,697 | 693 | 3,787,801 | 412,999 | change review .docx |
| Sol show_me | coord-revisions | 5/5 | 614 | 364 | 2,360,724 | 279,741 | change review .docx (plus one helper's report, kept) |
| Sol method | coord-revisions | 5/5 | 512 | 354 | 2,139,965 | 257,622 | change report .docx |
| Sol sheet_tools | coord-revisions | 4/5 (bearing pressure missed) | 654 | 328 | 2,409,927 | 183,274 | change report .docx |
| Sol all_three | coord-revisions | 5/5 | 585 | 359 | 2,367,122 | 243,117 | change report .docx |
| GPT-5.4 baseline | coord-revisions | 5/5 | 201 | 79 | 731,537 | 17,438 | none (it offered one) |
| Sol baseline | coord-submittal | 6/6 | 119 | 26 | 325,992 | 18,634 | review .docx |
| Sol show_me | coord-submittal | 6/6 | 182 | 64 | 607,674 | 37,169 | review .docx |
| Sol method | coord-submittal | 6/6 | 116 | 50 | 525,797 | 29,738 | review .docx |
| Sol sheet_tools | coord-submittal | 6/6 | 97 | 25 | 298,294 | 17,573 | review .docx |
| Sol all_three | coord-submittal | 6/6 | 166 | 41 | 472,020 | 31,897 | review .docx |
| GPT-5.4 baseline | coord-submittal | 6/6 | 86 | 25 | 319,822 | 7,980 | review .docx and the same text as .md |

Neither question asked for a file. The page's own instructions ask for a
file whenever someone asks for a review, check or comparison, so the Word
files are expected. Every file name fits the question, and every file
agrees with its run's answer, except the sheet_tools change report (below).

No run used `show_me`, the switch that lets the main model look at a
picture itself. It was offered in the show_me and all_three runs. A search
of all twelve records finds no call with `show_me` set to true.

## coord-revisions

### What changed between the issues (ground truth)

- **A-401:** the grid B–C dimension on both enlarged core plans went from
  7200 to 7500. Each is clouded with delta 1, and the revision block has
  "1 2026-03-02 REVISED CORE LAYOUT".
- **A-601:** the hardware set for door 107 (Meeting Room) went from HW-1 to
  HW-2. The cell is clouded, and the revision block has "REVISED DOOR
  HARDWARE".
- **S-001:** note 2, "ALLOWABLE NET SOIL BEARING PRESSURE", went from
  150 kPa to 200 kPa. There is no cloud and no revision entry. The sheet
  is still revision 0, and G-001's index still lists it as revision 0.
  S-001 has 182 words in both issues, so the page map cannot flag it.
- **G-001:** the index's revision column became 1 for G-001, A-401 and
  A-601, and a revision row was added: "1 2026-03-02 SHEET INDEX UPDATED".
  There is no cloud.
- **Every sheet:** the title-block line went from "ISSUE 0 - 2026-01-12" to
  "ISSUE 1 - 2026-03-02".

### Each run's path

- **Sol baseline.** The primary started two helpers, and each one compared
  the whole set. Each looked at all 30 pages (about 300 picture-reading
  calls and about 1.1 million input tokens apiece). Each read only part of
  the text. Helper `wpk5Fe` asked its issue-1 reading of S-001 to "Identify
  every substantive drawing change versus issue 0". The reading answered
  "No substantive change versus Issue 0 is identifiable in this tile"
  ([684]). The helper's text read covered page 11 for issue 0 only, so it
  reported S-001 unchanged. Helper `CPsoWO` asked its reading to report "all
  notes for changes/clouds". That reading transcribed 200 kPa, and the
  helper reported the unflagged change. The primary saw the two reports
  disagree. It read page 11's text in both issues and looked again at
  S-001, A-401, A-601 and G-001, then answered correctly.
- **Sol show_me.** The primary split the set into three helpers by page
  range (1–5, 6–10, 11–15). Each read the text and looked at both issues of
  every page, zooming where needed. Helper `tiHKev` read the text of pages
  11–15 in both issues, asked for an inventory of all notes, and zoomed on
  note 2. It reported the change and its missing cloud and entry. The
  primary then confirmed the changes with targeted text searches.
- **Sol method.** The same three-range split, with text, looking and
  zooms. Helper `gvzTNu` read pages 11–15's text in both issues, asked for
  exact wording, and zoomed on the note. The primary then read the text of
  pages 1, 8, 10 and 11 itself.
- **Sol sheet_tools.** No helpers. At [55] the model planned two helpers;
  at [59] it chose to make 30 page readings itself instead. Its path was:
  `sheet_index` on both issues ([18], [20]), which reported issue 1's
  revision as "1" on pages 1, 8 and 10 and "0" everywhere else; the page
  map; contact sheets; searches for "ISSUED" and "2026-03-02"; and the text
  of only pages 1, 8 and 10 in both issues. The text read was issued in the
  same step as the searches, straight after the index results came back.
  Then it looked at all 15 pages of both issues, made eight zooms and wrote
  the Word file. Its reading prompts differed between the issues. For issue
  0 it asked to "Read sheet ... fully", and the S-001 reading ([550])
  transcribed all twelve notes, including 150 kPa. For issue 1 it asked to
  "Identify substantive changes, revision clouds/deltas and revision block
  beyond global issue/date". The S-001 tile holding the notes answered
  "Substantive changes: None identifiable in this tile; it contains
  STRUCTURAL GENERAL NOTES 1–12 with no visible change marking" ([653]).
  The model's own reasoning at that point says: "Without prior issues for
  comparison, it's tricky to spot substantive changes... If notes changed
  without clouds, it's impossible to tell here." No record in the run
  contains "200 kPa", and its final reasoning ([727]) never comes back to
  S-001.
- **Sol all_three.** Two helpers, for pages 1–8 and 9–15. Helper `0VAT4G`
  read the text of pages 9–15 in both issues, asked to read all notes, and
  zoomed. Its brief also said "Be alert to changes not clouded or not in
  revision blocks". The primary then searched for "ALLOWABLE NET SOIL
  BEARING PRESSURE" in both issues.
- **GPT-5.4 baseline.** It read the text of all 15 pages in both issues
  first. Then it looked at only three pages (A-401, A-601 and S-001) in
  both issues. It found everything with 79 model calls and 0.73 million
  tokens, under a third of any Sol run.

**What every run that found S-001 had in common:** a text read of S-001 in
both issues, or an issue-1 reading of S-001 asked to transcribe the notes.
Neither the arm's tools nor the number of page readings decided it.
sheet_tools had the same `sheet_index` that all_three had. all_three found
the change because its helper read the text of pages 9–15.

Some methods cannot find this change, whatever model runs them. One is to
choose sheets by their revision marks (the title-block revision, the
index's revision column, the revision-row dates, clouds). Another is to ask
a one-page reading what "changed". The draft review method's own clause on
revisions also points only at flagged changes: "what the cloud and the
revision table say changed, and whether the sheets that depend on it
changed too". Even so, the method run found the change.

### Wrong or unsupported statements

- **Sol sheet_tools, Word file.** It lists S-001 under sheets with "no
  substantive content change". It also says "Text extraction was used to
  confirm" the changes, but text was read only for pages 1, 8 and 10.
- **GPT-5.4.** The answer opens with "Issue 1 changes are limited to three
  sheets" and then lists a fourth, S-001, in a second table. It says "No
  cloud seen" for G-001, but it looked at G-001 only in the contact sheet
  of thumbnails; its G-001 findings came from text. The statement is true,
  but nothing in the run saw it.
- The baseline, show_me, method and all_three answers and Word files are
  correct throughout, including each change's cloud and listing status.
  The method and all_three runs noticed that G-001's index lists an A-202
  that is not in the set, and that S-201's index title differs from its
  sheet. They rightly set both aside as unchanged between the issues.

### Cost and time

- Model calls outnumber tool calls because each whole-page reading makes
  ten picture-reading calls (one overview and nine tiles), and each zoom
  makes more.
- The baseline run's extra cost is the duplicated work. Two helpers each
  read all 30 pages (helper agent steps alone: 482K and 426K input tokens).
  Then the primary checked again with five page readings (45 tiles).
- About half of the baseline run's 1,697 seconds is one helper model call
  (`CPsoWO`, [107] to [812]): 904 seconds for 25K tokens in and 324 out.
  The record holds no wait, retry or busy event, so it cannot say why. The
  throttle's longest wait for a busy turn is 900 seconds. (Coordinator's
  note: that path writes a `model_busy` event for each wait and the record
  has none, so it is not the cause; TRACE_REVIEW §7 names the silent retry
  path that fits.)
- Every Sol run of this task has one or two picture-reading calls with a
  start and no end: all_three [755], baseline [1464] and [1511], method
  [306], sheet_tools [70] and [368], show_me [758] and [766]. Each tool's
  result then counts 11 picture-reading calls for a 10-call page, so the
  call was presumably timed out (the limit is 180 s) and sent again. No
  event records the timeout or the re-send.

## coord-submittal

### What the submittal holds (ground truth)

The submittal is for Class C concrete (columns and suspended slabs), Mix
C-35-20: cement 310, fly ash 133, total cementitious 443 and water
213 kg/m3. Four planted problems, one check each:

- **w/cm:** 0.48, against the Class C maximum of 0.45.
- **Fly ash:** 30 percent, against the maximum of 25.
- **Strength record:** 12 tests (average 40.9 MPa), against 30 consecutive
  tests or trial batches.
- **Aggregate reports:** none attached. Section 1.3 D asks for grading and
  for alkali-silica reactivity by the accelerated mortar-bar method, no
  more than 12 months old.

Not checked by the scorer:

- The transmittal says "DEVIATIONS: NONE". Section 01 33 00, 1.2, says a
  submittal with unmarked deviations is returned.
- The slump (100 mm), aggregate size (20 mm) and strength (35 MPa) comply.
- The admixture dose of 1.3 L/m3 is about 0.29 L per 100 kg, inside the
  stated 0.2–0.5.
- The mill certificate page and the admixture page name no maker or
  product.

### Each run's path and findings

Every run opened both documents and read the text of Section 03 30 00 and
all five submittal pages. Every run looked at each of those pages and
computed w/cm and fly ash from the quantities, except GPT-5.4, which
quoted the stated values. Every run wrote a Word file. All six found all
four planted problems and cited printed pages correctly.

| Run | Read Section 01 33 00? | "DEVIATIONS: NONE" reported? | Mill certificate and admixture data | Air content |
|---|---|---|---|---|
| Sol baseline | no | no | incomplete (no maker or product) | asks to confirm interior use |
| Sol show_me | no | no (noted in reasoning at [207], left out) | insufficient | asks to confirm |
| Sol method | no; its Word file says 01 33 00 was "not used" because the submittal is under 03 30 00 | no (noted in reasoning at [149], left out) | incomplete | consistent if interior |
| Sol sheet_tools | yes (all five manual pages read, page 3 looked at) | yes | inadequate | asks to confirm |
| Sol all_three | yes (text of pages 3–4, page 3 looked at; its helper briefed on both sections) | yes | present; the specification sets no contents | asks to confirm |
| GPT-5.4 | yes (text and look, pages 3–4) | yes | provided | complies |

- **The deviation finding follows one thing: whether the run read Section
  01 33 00.** The rule's wording ("marks every deviation", "unmarked
  deviations") appears nowhere in the baseline, show_me or method records.
  The question says "against the specification section it is submitted
  under", which points a reader at 03 30 00 alone. Three runs read the
  general submittal section anyway.
- **The mill certificate and the admixture page.** Four runs call them
  inadequate because no maker or product is named, and the fixture does
  name none. Two runs accept them because the specification asks only for
  "Cement mill certificate" and "Product data for each admixture", without
  contents. Both readings can be defended; neither is wrong.
- **Air content.** Four runs ask the contractor to confirm that every
  placement is interior. The method and GPT-5.4 runs accept the submittal's
  "Not air entrained (interior concrete)". Either is a reasonable call.
- **Sol sheet_tools** adds that the strength record must apply to the
  corrected mix. That is a fair reviewer's point.
- **GPT-5.4** adds "verification of report recency" as a second missing
  item. It is the same gap as the missing aggregate reports, counted twice.
- No wrong values or citations in any answer or Word file.

### Why tokens range from 0.30 to 0.61 million

- **To-do updates.** Five or six primary steps in every run did nothing but
  call `write_todos`. Each re-sends the whole conversation, so those steps
  used 25 to 42 percent of each run's input tokens (113K to 167K tokens).
  The same count on coord-revisions is 4 to 18 percent on Sol and 37
  percent on GPT-5.4. GPT-5.4's five such steps there, three of them in a
  row at the end, took 52 of its 201 seconds. Two texts push the model this way. The framework's own
  instructions for the tool say "Mark tasks complete IMMEDIATELY after
  finishing (don't batch completions)". The page's instructions say "Plan
  multi-step work with `write_todos` — a full-set review, a
  specification-versus-submittal check ... — and keep it updated".
- **Tiled page readings on text pages.** The show_me and method primaries
  asked every page reading for 2x2 tiles: five picture-reading calls a
  page, and readings of 4,500 to 11,000 characters against 1,300 to 3,100
  for a single reading. That is about 130K picture-reading input tokens
  against 26K to 31K. The longer readings also stay in the conversation,
  so later steps re-send them (primary agent-step input 378K to 384K,
  against 256K to 288K in the baseline, sheet_tools and GPT-5.4 runs). These are plain text pages with a text layer the
  runs had already read. The reasoning gives no reason for the tiles.
- **Independent-check helpers.** show_me and all_three each started one
  helper to check the review independently (about 87K tokens, 51 to 73
  seconds each). Both helpers agreed with their primary and added nothing
  to the answer. The show_me helper was briefed on Section 03 30 00 alone,
  so it could not catch the deviation either.
- The show_me record is the largest (336 KB) because of the tiled readings
  and the helper.

## Follow-ups proposed

1. **The page-reading tools should say that one reading sees one image and
   cannot compare it with another issue.** When asked what changed "versus
   issue 0", the picture reading answered "No substantive change versus
   Issue 0 is identifiable in this tile" (baseline helper, [684]) and
   "Substantive changes: None identifiable in this tile" (sheet_tools,
   [653]). Those answers address a comparison the reading cannot make, and
   two agents took them as "unchanged". The fix belongs in the reading's
   own instructions (`vision_tools.py`, the tile and page prompts): when
   asked about changes, transcribe what is there and say that a comparison
   needs the other issue. This would hold on any model.
2. **The record should log model-call waits and retries, and picture-call
   timeouts and re-sends.** A 904-second call ([107] to [812], baseline
   coord-revisions) and nine started-but-never-ended picture calls cannot
   be explained from the record. Its only event kinds are process_start,
   turn_start, model_start, model_end, tool_start, tool_end, cited_pages and
   turn_end.
3. **A `sandbox:/mnt/data/<name>` link in an answer should not reach the
   user as a dead link.** It appears in the baseline and method submittal
   answers ("[Download the Word review memorandum](sandbox:/mnt/data/...)").
   A text search finds it in 15 Sol runs of the suite. `method_call.py`'s
   note for its writers already says "a /tmp path or a sandbox: link would
   not reach them". The note from `write_docx` does not. Either give every
   writer that note, or have the page point such a link at the attached
   file of the same name.
4. **Measure the cost of to-do-only steps across all 84 runs before
   deciding anything.** In this group they are a quarter to two fifths of a
   short task's input tokens on both models. Whether to keep `write_todos`
   on the review page for short tasks, or to say that several items may be
   marked at once, is the owner's call once the number is known.

## Noted, not proposed

- The draft review method's clause on revisions covers only flagged
  changes. The method run found the unflagged change anyway.
- A tool that compares the text of two issues would make an unflagged
  change mechanical to find. Five of six runs found this one by reading
  text, so a new tool is not justified by this task.
- `sheet_index`'s revision column invites choosing sheets by revision. Its
  description does not suggest that use; the sheet_tools run did it on its
  own.
- "DEVIATIONS: NONE" is in the task's truth text but no check scores it.
  The question's wording points away from Section 01 33 00. Whether a
  review should be expected to catch it is a question about how the task
  is designed.
- The baseline coord-revisions primary gave two helpers the same whole job.
  The other Sol arms split the pages between their helpers and cost about
  60 percent as much.
- GPT-5.4 judged 12 of 15 coord-revisions pages unchanged from text alone,
  without looking at them. Here every change had words in it, so that
  worked. A change drawn only in lines would have been missed this way.
