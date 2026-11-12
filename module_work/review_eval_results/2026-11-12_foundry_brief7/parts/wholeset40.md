# Group wholeset40: coord-long-review, six runs read in full

Task `coord-long-review`: a pre-issue coordination review of the whole
40-sheet drawing set of the made-up NORTH YARD ANNEX, checked against the
project manual and the geotechnical memorandum, delivered as a comment log
with one row per comment. Nineteen problems are planted (the `_LONG_KEYS`
list in `aec_document_review/review_eval/tasks.py`); the task has 21 checks:
one per planted problem plus two general checks (no give-up on legibility,
no 0-based page citation), which every run passed.

Runs read (the full `activity.jsonl` of each, page by page with a trimmed
reader; every result that mattered printed whole; the delivered Word and
Excel files opened and read row by row; `run.json`, `FEEDBACK.md` and the
console log of each):

| run | arm | model |
|---|---|---|
| Sol baseline | baseline | GPT-5.6 Sol |
| show_me | `GEOTECH_VISION_SHOW_ME=1` | GPT-5.6 Sol |
| method | the draft review method as instructions | GPT-5.6 Sol |
| sheet_tools | `GEOTECH_REVIEW_SHEET_TOOLS=1` | GPT-5.6 Sol |
| all_three | all three switches | GPT-5.6 Sol |
| GPT-5.4 | baseline | GPT-5.4 |

Record numbers below (#752 and so on) are line numbers in that run's
`activity.jsonl`. "H1", "H2" are the run's helpers in the order the main
agent started them.

## 1. Headline findings

1. **The scorer counts the agent's own working notes as delivered.** The
   with-files checks read `.scratch/page_readings.json` and
   `.scratch/reads.json` (and, in sheet_tools, two kept helper files) as if
   they were part of the report. I rebuilt those two notes files from each
   run's `activity.jsonl` (they were not in the hand-back) and re-ran all
   114 planted-problem checks with the package's own `run_check`: with the
   notes, every recorded verdict is reproduced exactly; without them, 12
   verdicts flip from pass to fail. None of the 12 passing passages states
   the planted problem. Three rest on a pixel coordinate: door 204
   "passes" in the Sol baseline on `"issue 1 - 2026-03-02" [648, 162, 757,
   204]` next to `"project no. 24-117"`. Scores on the answer and the
   delivered files alone: **Sol baseline 15/21 (recorded 20), show_me 20
   (20), method 17 (19), sheet_tools 19 (19), all_three 19 (20), GPT-5.4
   8 (12).** The first-pass suggestion is confirmed, and it is worse than
   suggested: some of the notes that pass checks are not even sightings.
2. **The bay B to C conflict (7200 on the plans, 7500 on A-401) was missed
   by all six runs.** In every run a vision call read the clouded 7500 on
   A-401, and 7200 was read off the plans many times; no agent ever set the
   two side by side. In three Sol runs (baseline, method, sheet_tools) the
   same helper had A-101 to A-105 and A-401 in its own scope and still did
   not compare them. One tile reading in the Sol baseline even said
   "Confirm this revision is incorporated into structural and MEP
   backgrounds before issue" (#762); nothing followed it up.
3. **Most planted problems that were lost were seen and then dropped
   between a helper's readings and its reply.** Of the 15 planted problems
   the five Sol runs failed to deliver, 12 were in a page reading of the
   helper that owned the sheet, and 3 more (door 204 in the Sol baseline,
   the firestopping section in method and sheet_tools) were only in text
   results no agent ever mentioned. Only one of the 12 reached a main
   agent at all: door 509 in all_three, as a line in a helper's sheet
   inventory, which the main agent then left out.
4. **The show_me arm never used show_me.** Not one of the 113 page looks
   and zooms in show_me, nor any in all_three, passed `show_me=true`
   (the argument appears in no tool call; the model never mentions it).
   That arm differed from the baseline only in three tool descriptions
   and an argument nobody used, so its result says nothing about the
   switch.
5. **`check_cross_references` was exact and was used.** It returned the
   three planted callout problems (7/A-501, 9/A-502, 2/A-302) and nothing
   else among 39 callouts. In sheet_tools the main agent called it at
   about 1,400 s of 1,457, saw that helper 1 had not reported 7/A-501,
   zoomed on the callout and added the row. `sheet_index` read every title
   block correctly (S-201 SECOND FLOOR FRAMING PLAN, no A-202, no A-302).
   Neither tool was contradicted by any agent.
6. **One wrong statement in 149 delivered rows**: show_me's CR-08 says
   A-401 calls "1/A-601"; it calls 1/A-501, which exists. A single zoom
   reading misread the sheet number (#1690); the helper reported it as
   "Seen by zoomed look" and the main agent copied it.
7. **GPT-5.4 looked at 5 of the 40 sheets.** Its first three
   `analyze_pdf_page` calls asked for page ranges with an argument the tool
   does not take (`pdf_pages`) and were refused; it recorded that as
   feedback, handed the review to one helper that made four page looks, and
   wrote a 7-row log in 5.2 minutes from text reads and searches. It
   delivered 6 of 19 planted problems.

## 2. The six runs side by side

| | Sol baseline | show_me | method | sheet_tools | all_three | GPT-5.4 |
|---|---|---|---|---|---|---|
| minutes | 19.1 | 22.3 | 13.9 | 24.3 | 19.9 | 5.2 |
| model calls | 812 | 1,005 | 515 | 766 | 846 | 100 |
| tool calls | 200 | 231 | 118 | 175 | 225 | 51 |
| tokens in / out (millions) | 5.04 / 0.53 | 6.33 / 0.67 | 3.27 / 0.38 | 5.00 / 0.45 | 5.45 / 0.68 | 0.97 / 0.03 |
| helpers | 4 | 5 | 3 | 4 | 5 | 1 |
| sheets looked at (of 40) | 40 | 40 | 40 | 40 | 40 | 5 |
| page looks / zooms on the set | 72 / 28 | 92 / 21 | 45 / 14 | 66 / 16 | 68 / 35 | 7 / 0 |
| rows in the delivered log | 32 | 40 | 17 | 29 | 24 | 7 |
| planted problems in the log (of 19) | 13 | 18 | 15 | 17 | 17 | 6 |
| recorded score | 20/21 | 20/21 | 19/21 | 19/21 | 20/21 | 12/21 |
| score, answer + delivered files only | 15/21 | 20/21 | 17/21 | 19/21 | 19/21 | 8/21 |

Each planted problem, per run. D = in the delivered log. S = seen (in a
page reading, a text result, a helper's text or a reply) but not
delivered. N = never in any reading.

| planted problem | Sol baseline | show_me | method | sheet_tools | all_three | GPT-5.4 |
|---|---|---|---|---|---|---|
| index lists A-202, no such sheet | D | D | D | D | D | S |
| index calls S-201 ROOF FRAMING PLAN | D | D | D | D | D | D |
| callout 7/A-501 (A-501 has 1-6) | S | D | D | D (main agent, from the tool) | D | N |
| callout 2/A-302 on A-102 | D | D | D | D | D | S |
| door 109 on plans, not scheduled | S | D | D | D | D | S |
| door 204 scheduled, not on A-102 | S | D | S | D | D | S |
| second floor 104.20 vs 104.50 | D | D | D | D | D | S |
| bay B-C 7200 vs 7500 | S | S | S | S | S | S |
| section 07 84 00 not in the manual | D | D | S | S | D | D |
| bearing 200 vs 150 kPa | D | D | D | D | D | D |
| footing depth 1.2 vs 1.5 m | D | D | D | D | D | D |
| footing concrete 25 vs 30 MPa | D | D | D | D | D | D |
| F4 not in the footing schedule | D | D | D | D | D | S |
| callout 9/A-502 (A-502 has 1-8) | S | D | D | D | D | N |
| door 407 on A-104, not scheduled | D | D | D | D | D | D (gap only) |
| door 509 scheduled, not on A-105 | S | D | S | D | S | S |
| fifth floor 116.80 vs 116.50 | D | D | D | D | D | S |
| FCU-3-4 not scheduled | D | D | D | D | D | N |
| LP-4B not scheduled | D | D | D | D | D | N |

**Arm or chance.** One run per arm, and the five Sol runs spread from 13
to 18 delivered problems with the plain baseline lowest and show_me (whose
switch was never used) highest. That spread is run-to-run chance, almost
all of it in what the helpers chose to put in their replies. What can be
said per arm:

- **show_me**: the switch was not exercised, so this run cannot tell
  anything about it. Most of its extra problems came in one long reply,
  from the helper that had A-201 to A-602 (7,529 characters).
- **sheet_tools / all_three**: the callout check plausibly makes the three
  callout problems reliable. Both runs with it delivered all three; the
  baseline, without it, missed 7/A-501 and 9/A-502 although its helper's
  tile readings had both. One run each cannot prove it.
- **method**: three helpers (drawings 1-20, drawings 21-40, manual and memo
  alone), the fastest and cheapest Sol run. It lost the firestopping
  section: the helper that read G-002 never mentioned note 6, and the
  manual was read by a different helper. Whether the method's split caused
  that cannot be told from one run.
- **GPT-5.4**: a different path altogether (section 8); the gap to Sol is
  the model's path, not chance.

**First-pass suggestions.** (1) Page notes read as delivered: confirmed
(finding 1). (2) sheet_tools missing the un-clouded bearing-pressure
change: not testable in this task, which does not ask about revisions;
here every run found the 200 vs 150 kPa conflict with the memo, and none
said the change was not clouded. (3) Bay missed in the whole-set reviews:
confirmed for all five Sol runs in this group (and GPT-5.4).

## 3. Delivered rows by class (hand count)

Every row of every delivered log, classed against the fixture. "Duplicate"
here means the same point repeated on a sibling sheet (one row per floor
saying the same thing).

| run | rows | planted (rows / problems) | wrong statement | true, not planted | request for information | duplicate |
|---|---|---|---|---|---|---|
| Sol baseline | 32 | 12 / 13 | 0 | 19 | 1 | 0 |
| show_me | 40 | 18 / 18 | 1 | 16 | 1 | 4 |
| method | 17 | 14 / 15 | 0 | 3 | 0 | 0 |
| sheet_tools | 29 | 16 / 17 | 0 | 9 | 0 | 4 |
| all_three | 24 | 14 / 17 | 0 | 9 | 1 | 0 |
| GPT-5.4 | 7 | 6 / 6 | 0 | 1 | 0 | 0 |
| **total** | **149** | **80 / 86** | **1** | **57** | **3** | **8** |

Rows that carry two planted problems: both floor levels in one row (four
Sol runs; show_me gave each floor its own row); A-202 and S-201 in one
row, and bearing and depth in one row (all_three).

**The one wrong statement.** show_me CR-08: "The first-floor enlarged core
plan includes a 1/A-601 reference at the perimeter wall, but A-601 is a
door schedule and has no Detail 1." A-401's callout is 1/A-501 (the
fixture; `check_cross_references` in the two sheet-tool runs lists
"A-401 1/A-501 resolved"). Source: a `render_region` zoom of A-401 read
"a detail call circle 1 / A-601" (#1690); helper H3 reported it with
"Seen by zoomed look" (#1950); the main agent copied it into the log
(#2469). No call checked it against the sheet's text, which holds A-501.

**Close to the line, classed true:**
- Sol baseline row 26 sets the AHU against the fan coils: the scheduled
  totals it quotes (6,240 L/s, 127.5 kW) are right, but its "6,600 L/s and
  135.0 kW when the omitted FCU-3-4 is included" uses 360 L/s and 7.5 kW
  for FCU-3-4, which no document prints; the agent took them from the
  other floors' FCU-x-4 rows. The row does not say the figure is assumed
  (the Review Basis tab shows the sum without saying where the values came
  from).
- GPT-5.4 row 7: "S-201 ... revision table shows only revision 0 ... with
  no entry for the current issue". True, but it is true of all 37 sheets
  without a revision; singling out S-201 is arbitrary.
- all_three row 11 asks to confirm door 107's HW-2: that is the clouded
  revision of A-601 (fixture `REVISIONS`), so the row is a fair question
  (counted as a request for information).
- The 57 true-but-not-planted rows are mostly about the fixture's stand-in
  drawing: reflected ceiling plans with no devices, mechanical and
  electrical plans with no backgrounds or circuits, identical detail
  sketches whose notes do not match their titles, A-201 showing only two
  elevations, A-301 stopping at EL 100.00 with no footings. Each was
  checked against the fixture and is accurate.

**Wrong statements in helper text that did not reach a log:** method H3
put GROUT FRAME SOLID in S-501 detail 2 (it is in details 1 and 4; the
delivered row says "details 1 through 4", which is true); Sol baseline
H1's reasoning "Door 109 appears on sheet A601" (#849; it does not, and
this is why H1 never raised 109).

## 4. Where each lost problem was lost

Sol baseline. The main agent wrote the spreadsheet itself from the four
helper replies (plus one `calculate` call for the AHU sums); it added
nothing the replies did not carry. All six losses were helper H1's
(sheets G-001 to A-602):
- 7/A-501: H1's tile reading of A-101 read "3/A-501 and 7/A-501" and
  asked only to "confirm both detail assignments" (#380, #392); H1 also
  read A-501's six details. Its reasoning planned to "closely examine the
  A-501 detail mismatch" (#870); the reply left it out.
- 9/A-502: H1's tile readings of A-114 read "7/A-502 and 9/A-502" (#743,
  #754, #766); H1 then went after detail 7's content and never checked
  that detail 9 does not exist.
- 109: tile readings of A-101 and A-401 show door 109 (#390, #767). H1's
  reasoning assumed "Door 109 appears on sheet A601" (#849) and never
  checked.
- 509: H1 read A-602's row "509 CORE 90 MIN HW-4" three ways (#794, #804,
  zoom #900) and checked only that HW-4 suits a rated door (#849). It
  never looked for a 509 tag on A-105.
- 204: never in any agent's text. A-601's row 204 was in H1's text read
  (#858) and A-102's tag list "201 202 203 205 206" was in the main
  agent's own first text read (#34).
- bay: section 5.

show_me: only the bay (section 5). Its plans (H1) and A-401 (H3) were
with different helpers.

method. The main agent wrote the Word log from three replies and its own
three page looks.
- 204 and 509: in H1's page readings only (509 at #486 to #891). H1's
  reply checked doors from plan to schedule (it reported 109 and 407) and
  never from schedule to plan.
- 07 84 00: H1 read G-002 by text (#46) and in a tile (#160) and never
  mentioned note 6. The manual's contents page was read by H2. No agent
  compared them.
- bay: H1's readings of A-401 only (#492 to #612); H1 also had A-101 to
  A-105.

sheet_tools.
- 07 84 00: only in H1's text read of G-002 (#94); no agent ever wrote it.
  H1's reply was the shortest of any drawing helper (3,353 characters for
  20 sheets).
- bay: H1's readings (#626 to #723) and H3's (#1255); H1 had both sets of
  sheets.
- 7/A-501 was dropped by H1 too, and recovered by the main agent from
  `check_cross_references` (#1857, then zoom #1862 to #1873).

all_three.
- 509: H2's reply listed "A-602 doors 401-406, 408 and 501-509" in its
  sheet inventory (#934) without raising it. The main agent searched for
  509 and got one hit, on A-602 only (#2114); its reasoning said "doors
  like 509 might be missing plans, so I should check those; door 204 too"
  (#2119, #2123). It checked and delivered 204, never looked at A-105, and
  left 509 out.
- bay: section 5.

GPT-5.4: section 8.

A pattern across the Sol runs, noted: the two "scheduled but not on the
plan" doors were delivered less often (204 in 3 of 5, 509 in 2 of 5) than
the five "on the plan but not scheduled" marks (109 in 4 of 5; 407, F4,
FCU-3-4 and LP-4B in 5 of 5). The helpers checked from plan to schedule.

## 5. The bay B-C dimension and section 07 84 00

**Bay B to C (7200 on A-101 to A-105, S-101, S-201 to S-205; 7500 on
A-401, clouded, revision 1).** Read off a sheet in every run:
- 7500 was read by vision tile calls on A-401 in all six runs (Sol
  baseline H1 #752 to #769 and the page result #787; show_me H3's zoom
  #1690 and H5's readings; method H1 #492 to #612; sheet_tools H1 #626 to
  #723 and H3 #1255; all_three H2 #542; GPT-5.4 main agent #231 to #235),
  and by text in four Sol runs (`read_document` returns "7500").
- 7200 was read off the plans by text and tiles in every run.
- Where it stopped: at the comparison. No agent's own text ever names 7200
  and 7500 together. The nearest misses: the Sol baseline tile reading
  quoted in finding 2; all_three H2's reply to the main agent listing
  "revised/clouded 7500 dimension" on A-401 (#933, #934), where the main
  agent had 7200 only in its own early text reads (#62, #2132), not in any
  reply; and GPT-5.4's main agent, which had 7500 from A-401 tiles (#231
  to #235) and 7200 from S-201 tiles (#239, #241) in its own results a few
  calls apart.
- The tile design cannot see this conflict by itself: each tile call sees
  one ninth of one sheet, with the whole coordination question, and can
  only flag a revision cloud as a revision.

**Section 07 84 00 (G-002 note 6, "FIRESTOPPING AT RATED PENETRATIONS: SEE
SECTION 07 84 00"; the manual binds only 01 33 00, 03 30 00 and 08 71 00
and says unlisted sections are issued separately).** Read off G-002 by
text in all six runs. Delivered where an agent searched the manual or read
its contents page and set it beside the note (Sol baseline H1 searched the
manual for "07 84 00", 0 hits; GPT-5.4's main agent searched for it,
#289). In method and sheet_tools the note sat in a text result (method H1
#46, sheet_tools H1 #94) and no agent ever mentioned it; the scorer
correctly failed both.

## 6. The helpers: how the work was split

At most two helpers ran at once (`max_helpers=2`), so the later helpers
waited for a slot. No reply was cut off and no helper hit its call cap.

| run | helper: scope (seconds from start to end) | reply length (characters) |
|---|---|---|
| Sol baseline | H1: sheets G-001 to A-602, 20 sheets (44-471); H2: S-001 to S-501 plus the manual and memo (44-441); H3: M-101 to M-601 (441-914); H4: E-101 to E-601 (471-1,089) | 4,850; 5,662; 4,779; 5,087 |
| show_me | H1: G-001 to A-115 (52-314); H2: mechanical (52-553); H3: A-201 to A-602 (314-939); H4: electrical (553-979); H5: structural (939-1,275) | 4,525; 4,994; 7,529; 3,046; 5,310 |
| method | H1: PDF pages 1-20 (12-431); H2: manual and memo only (12-136); H3: PDF pages 21-40 (136-741) | 8,268; 10,783; 7,689 |
| sheet_tools | H1: G-001 to A-602 (52-550); H2: mechanical (52-1,122); H3: structural (550-1,108); H4: electrical (1,108-1,393) | 3,353; 1,965; 5,078; 3,067 |
| all_three | H1: G-001 to A-112 (49-497); H2: A-113 to A-602 (49-500); H3: S-001 to M-102 (497-1,008); H4: M-103 to E-601 (500-1,121); H5: manual and memo (1,008-1,052) | 9,038; 8,468; 13,295; 6,660; 8,124 |
| GPT-5.4 | H1: "research only", the whole set (79-185) | 2,426 |

Each brief named its pages, told the helper to open the PDFs and look at
every page, and listed the kinds of conflict to look for. Every planted
problem lies across two sheets; how the split fell decided whether one
helper could see both: the bay was inside one helper's scope in the Sol
baseline, method and sheet_tools and split across two in show_me and
all_three; it was missed either way. Noted: sheet_tools' mechanical helper
took 18 minutes for six sheets and returned the shortest reply of all
(1,965 characters).

## 7. sheet_index and check_cross_references (sheet_tools, all_three)

- `sheet_index` (main agent and each drawing helper called it; ten calls,
  identical 4,073-character results): 40 sheets, every title and revision
  as printed (G-001, A-401 and A-601 at revision 1; S-201 SECOND FLOOR
  FRAMING PLAN; detail marks 1 to 6 on A-501 and 1 to 8 on A-502). It
  gives everything needed for the two index problems. Used: both runs
  delivered them (as did the runs without it).
- `check_cross_references`: 39 callouts, 36 resolved, 7/A-501 and 9/A-502
  unresolved ("view marks recognised on A-501: 1 to 6; none for 7"), 2/A-302
  not in the document. No false flag. Its results for the helpers' page
  ranges in all_three agree (#131 pages 10-19, #529 pages 0-9, #992 pages
  20-29).
  - sheet_tools: the main agent called it only at the end (#1857, about
    1,400 s), zoomed on all three flagged callouts and added 7/A-501, which
    its helper had dropped.
  - all_three: the main agent called it at 50 s (#61); all three callout
    rows were delivered, each marked "visually confirmed".
- Neither tool bears on the bay or the firestopping section, the two
  problems these runs still missed.

## 8. GPT-5.4: what it did instead

The main agent opened the three documents, read the memo and the manual by
text, read the drawing set's first pages by text, made a contact sheet,
and then tried three `analyze_pdf_page` calls with `pdf_pages` ranges
(4-13; 16, 19-20, 23-27; 22, 29-33, 35-40). The tool refused each with a
clear message (`analyze_pdf_page has no argument 'pdf_pages' ... did you
mean pdf_page`). The model recorded that in `FEEDBACK.md` ("Worked around
by switching to subagent reading job and targeted page looks") and started
one helper with a brief that asked for a compact log and named the kinds
of conflict to look for. The helper made four page looks (A-601, A-602,
S-101, A-401) and otherwise used text searches; its reply says it "could
not fully check all plan-to-plan elevation/detail-reference consistency".
The main agent then looked at A-602, A-401 and S-201 itself, searched the
text for the S-001 values, and wrote the 7-row spreadsheet.

Sheets looked at: A-401, A-601, A-602, S-101, S-201. No floor plan,
reflected ceiling plan, elevation, section, detail, mechanical or
electrical sheet was looked at, which is why 7/A-501, 9/A-502, FCU-3-4 and
LP-4B never appear at all. What it had and did not use: A-201 and A-301's
level text in its own read (#57); A-602's row 509 and A-601's row 204;
F4 in a tile reading of S-101 (it never read the S-501 footing schedule;
its text read of pages 0-2, 13-21, 27, 33, 39 returned only 0-2 and 13-15
under the size limit); the A-202 index row. Its door 407 row rests on the
schedule's numbering gap ("Verify whether door 407 is missing from the
schedule or intentionally omitted"); it never checked A-104.

## 9. Where the model calls and tokens went

Model calls outnumber tool calls about four to one because every look is
several model calls: `analyze_pdf_page` on a sheet is one overview call
plus nine tile calls (3 x 3), each tile call given the coordination
question and one ninth of the sheet; a zoom is one call; a contact sheet
one; the vision probe five per run.

| run | main agent | helper agents | tile calls | overview calls | zoom calls | other vision | tokens: agents / vision (millions) |
|---|---|---|---|---|---|---|---|
| Sol baseline | 12 | 35 | 648 | 72 | 28 | 17 | 2.25 / 3.32 |
| show_me | 13 | 39 | 828 | 92 | 21 | 12 | 2.86 / 4.14 |
| method | 10 | 23 | 405 | 45 | 14 | 18 | 1.52 / 2.13 |
| sheet_tools | 14 | 46 | 597 | 66 | 16 | 27 | 2.41 / 3.04 |
| all_three | 13 | 37 | 675 | 68 | 35 | 18 | 2.54 / 3.59 |
| GPT-5.4 | 16 | 7 | 63 | 7 | 0 | 7 | 0.73 / 0.27 |

Tile calls are 80 per cent of the calls and about half the tokens; the
helper agents' few calls are about a third of the tokens (each carries
40,000 to 60,000 input tokens). Every Sol run looked at every sheet; from
5 (method) to 52 (show_me) looks were repeats of a sheet already looked
at. `read_document`
on the drawing set stops at about 14,000 characters, which is only three
to five sheets: asked for pages 0-39 it returns 0-4 with a `next` pointer
(Sol baseline #34), and most agents did not follow it.

## 10. Follow-ups

**Proposed (defects in the scorer, a tool or the record; they hold on any
model):**

1. Scorer: score the whole-set review checks on the answer and the files
   delivered to the user only. Today `with_files=True` also reads
   `.scratch/page_readings.json`, `.scratch/reads.json` and kept helper
   files (`run.json` lists them in `files`; every check detail says
   "(answer + page_readings.json, reads.json, ...)"). In this group that
   passed 12 checks that the report does not support, on stray numbers
   and pixel coordinates (section 1). Evidence and the re-scoring scripts:
   my scratch `rescore.py` and `why2.py` (not committed); the recorded
   verdicts reproduce exactly from the rebuilt notes.
2. Record: the hand-back lacks `page_readings.json` and `reads.json`,
   though the scorer read them. Hand back every file a check reads (or
   stop reading them, item 1); otherwise a verdict cannot be checked from
   the hand-back.
3. Tool (`search_document`, planlens): a pattern containing "|" without
   `regex=true` is matched literally and finds nothing, and the zero-hit
   hint blames drawing sheets and suggests `fuzzy`. This happened four
   times (GPT-5.4 #107 sheet numbers, #108 spec terms; Sol method #342 and
   #1105 index titles). The result should say the pattern was matched
   literally and that `regex=true` reads "|" as "or". In the same results,
   `pages_not_searchable_as_text` lists drawing pages on which that very
   search found hits (GPT-5.4 #98: hits on pages 18-19 and 22-26, field
   "3-12,15,18-19,22-26"); the name says the opposite of what the field
   means (a miss there is not proof of absence).
4. Record: the suite's summary should count looks made with `show_me=true`
   per run. The show_me and all_three arms made none, and nothing in the
   results table shows it; an arm whose switch was never used reads as a
   measurement of the switch.

**Noted, not proposed:**

- The tile design: each tile call answers the coordination question from
  one ninth of one sheet, so a conflict between sheets (the bay) cannot
  be seen by any single vision call; the agents above them never compared
  what the tiles read.
- Helpers drop what their readings found: 12 of the 15 Sol losses were in
  the owning helper's page readings, and only one of those reached a main
  agent (as an inventory line, not a comment).
- A single zoom reading was taken as fact against the sheet's own text
  (show_me CR-08).
- Helpers checked doors from plan to schedule more than from schedule to
  plan (section 4).
- `read_document`'s size limit returns three to five drawing sheets per
  call; agents asked for whole ranges and rarely followed `next`.
- The door-407 check passed GPT-5.4 on a numbering gap in the schedule,
  without the plan sheet being looked at; the check cannot tell a gap from
  a mark on a plan that is not scheduled.
- The results table shows 0 errors and 0 failed calls for GPT-5.4,
  though three of its tool calls were refused for an unknown argument and
  it filed a tool-error feedback entry.
- GPT-5.4 turned a refused argument into a change of plan (one helper,
  four looks) rather than calling the tool once per page as the error
  message offered.
