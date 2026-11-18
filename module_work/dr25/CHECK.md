# DR-25: the review suite's gaps (check notes, 2026-11-18)

## What the item asked

Every arm scored 33 to 34 of 35 on the suite as it stood on 2026-11-10, so
the suite could not show that a change, or a stronger model, reads better.
Fourteen `coord-*` tasks on one made-up building were built since. What the
brief named as still missing:

1. a specification section in the usual three-part form, with a submittal
   answering it, and tasks whose answer needs both;
2. two revisions of one small drawing set, with tasks that need both: a
   change made on one sheet and not on its twin, a revision cloud with no
   entry in the revision block, an entry with nothing changed, a schedule
   row changed while the plan's tag was not;
3. the orientation turn, which never runs unless a run asks: no change to
   the default, only what turning it on costs and what it would measure;
4. a real long drawing set: not built here (it needs the owner's documents,
   kept private).

## What was there already, and how the new tasks differ

- `coord-submittal` already holds a three-part section (03 30 00, bound in
  the project manual) and a concrete mix submittal with four problems: two
  properties over their limits, a missing test report, a short strength
  record.
- `coord-revisions` already compares issue 0 with issue 1: two clouded and
  listed changes and one silent change. Its A-401 change (bay B-C from 7200
  to 7500, the plans left at 7200) is a change made on one sheet and not on
  its twin, but no check scores the twin: the check asks for A-401 and 7500
  only.

The new tasks add the kinds those two do not have, and do not repeat
theirs: a property where the submittal gives a complying value beside the
governing one, a reference to a section that does not exist, a specified
test the report leaves out, a schedule of values whose total is not its
sum; and for a reissue, a scored twin, a cloud with no entry, an entry with
no change, and a renumbered schedule row the plan did not follow.

## What I built, and where

All in `packages/aec-document-review/aec_document_review/review_eval/`
unless named.

- `coordination_fixture.py`
  - `REISSUE_PLANTED` (four problems), `ISSUE_2_PLANTED` (what `audit`
    finds in issue 2 beyond the 13 base problems), `SPEC_PLANTED` (five
    problems), `WINDOW_U`, `WINDOW_SOV`, `WINDOW_SOV_TOTAL`; `ISSUE_DATES`
    gains issue 2's date.
  - `_project(2, 2)`: issue 2 of the 15-sheet set. A-201 raises T.O.
    PARAPET to EL. 109.60, clouded and listed (A-301 keeps 109.30); S-501
    changes F3's reinforcing to 12-20M inside a revision 2 cloud with no
    revision-block entry; A-102 gets the entry "2 DOOR 205 RELOCATED" and
    no other change; A-601 renumbers door 106 as 110, clouded and listed
    (A-101 still tags 106). `_project(5, 2)` raises `ValueError`.
  - Drawing: `_Canvas.cloud(..., at=None)` (the triangle may sit beside a
    table), `_level_marks(..., clouded=())`, and the clouds on A-601's row
    and S-501's row. `audit` reads the mark keys from one table,
    `_MARK_KEYS`, in place of two inline conditions.
  - `build_window_section()`: Section 08 51 13, aluminum windows, in three
    parts (general, products, execution). The manual's contents already
    list 08 51 13 and say the sections not bound in that volume are issued
    separately; this is that section. It uses the generic submittal
    numbering (SD-03, SD-05, SD-06, SD-11); the wording and every value are
    invented.
  - `build_window_submittal()`: Submittal 08 51 13-01, four pages: a
    transmittal, a product data sheet, a laboratory test report and a
    schedule of values.
  - Three new fixture names in `_BUILDERS`: `coord_set_2`,
    `coord_window_section`, `coord_window_submittal`.
- `packages/aec-agent-core/aec_agent_core/review_eval/documents.py`: the
  three suite documents `fixture_coord_set_2`
  (`north_yard_annex_drawings_issue_2.pdf`), `fixture_coord_window_section`
  (`section_08_51_13_aluminum_windows.pdf`) and
  `fixture_coord_window_submittal`
  (`submittal_08_51_13-01_aluminum_windows.pdf`). The core gains three
  dictionary entries and nothing else; the geotech page never reads them.
- `packages/aec-agent-core/aec_agent_core/review_eval/checks.py` (review
  fix): `stated_together` gains `scope="finding"` (`_findings`,
  `_together_in_findings`). The answer is cut as `scope="item"` cuts it (a
  line; a list item with what is nested under it), with two differences: a
  table row is read with its column heads, each cell as "head: cell", so a
  yes/no table ("Carried? | No") says what each cell answers; and the
  heading rule reads one piece at a time, so a heading over a list does not
  join two of its items. Off unless a check names it: every older check
  scores as before.
- `tasks.py`: nine checks in `_PLANTED_CHECKS`, each with
  `scope="finding"`, the regexes they use (`_LEFT_BEHIND`, `_CLOUDED`,
  `_UNLISTED`, `_NO_CHANGE`, `_WATER_TEST`, `_NOT_GIVEN`, `_NOT_REPORTED`,
  `_NOT_A_SECTION`, the cell reading `_SAYS_NO` and `_NOT_NO`, and the
  window numbers), and two tasks at the end of `COORDINATION_TASKS`:
  - `coord-spec-submittal` (check; the manual, the section and the
    submittal): "Review this window submittal against its specification
    section and the project manual. What does not comply, what is missing,
    and what in it is wrong or does not resolve?"
  - `coord-reissue` (compare; issue 1 and issue 2): "These are issues 1 and
    2 of the same drawing set. Review issue 2 as a reissue: what changed, is
    each change clouded and listed in its sheet's revision block, does each
    new revision-block entry match a change on its sheet, and was each
    change carried to the other sheets that show the same thing?"

### The planted problems and what each check asks for

Each check is the core's `stated_together` with `scope="finding"`: the
pieces in one finding of the reply (one line, one list item with what is
nested under it, or one table row read with its column heads), within 300
characters unless noted. A heading over the finding may hold some of them.
A problem's one right answer:

| key | the problem | the check wants in one finding |
|---|---|---|
| `window-u-value` | whole-window U-value 1.90 W/(m2.K) against at most 1.60 (the sheet's 1.10 is the centre of the glass) | 1.90 and 1.60 |
| `window-anchorage` | SD-05 anchorage calculations not submitted | "anchorage" (or SD-05, "design data") and a word that it was not handed in ("not submitted", "not in the submittal", "missing", "omits", "none submitted", "left out", "no ... calculations"; not any "no" or "not") |
| `window-section-missing` | the schedule of values cites Section 08 80 00, not in the manual | 08 80 00 and a word that the section is not in the manual ("does not exist", "no such section", "not in the manual", "missing from the contents") |
| `window-water-test` | the test report gives air leakage and structural tests, no water penetration test | a water penetration test not named with the field, the site or 3.3, and a word that it was not reported, in a clause that does not name the field or the site |
| `window-sov-total` | the lines add up to 174,500, the total says 178,500 | 178,500 and 174,500 |
| `parapet-twin` | A-201 raised T.O. PARAPET to 109.60; A-301 still shows 109.30 | A-301, 109.30 or "parapet", and a word that A-301 was left behind ("still shows", "not updated", "needs updating", "differ", "Carried? No"; not "unchanged", "still consistent", "no conflict") |
| `cloud-not-listed` | S-501's F3 reinforcing changed to 12-20M inside a cloud, no revision-block entry | S-501, 12-20M, a cloud said to be there (not "not clouded", "no revision cloud", "without a revision cloud", "Clouded?: No") and a word that the entry is missing ("no entry", "revision 0 only", "In rev block?: No"), within 400 |
| `entry-no-change` | A-102 lists "DOOR 205 RELOCATED" and nothing on it changed | A-102, door 205 or "relocated", and a word that nothing changed ("has not moved", "no corresponding change", "identical to issue 1"; not "nothing else changed", "otherwise unchanged", "the rest of the sheet identical") |
| `door-renumbered` | A-601 renumbered door 106 as 110; A-101 still tags 106 | A-101 or "first floor plan", 106, and a word that the plan was left behind |

## Judgement calls

- **Two new tasks, nine checks, rather than a task per problem.** It follows
  `coord-submittal` and `coord-revisions` (one question, one check per
  planted problem), and keeps each run's cost to one reading of the
  documents. The suite goes from 52 to 54 tasks.
- **Issue 2 of the same 15-sheet set, not a new small set.** The revision
  tasks then compare like with like (the same sheets, title blocks and
  revision blocks a reviewer already met in `coord-revisions`), and issue 0
  and issue 1 stay the same bytes (proved below). 15 sheets per issue, 30
  pages a run, as `coord-revisions`.
- **The submittal holds every required item but one.** The section's list
  of submittals is cut to what a product submittal carries (SD-03, SD-05,
  SD-06; SD-11 closeout at substantial completion), so the only required
  item missing is the anchorage calculations. Shop drawings or samples
  "following separately" would have broken Section 01 33 00 1.1 ("complete,
  under one transmittal") in the existing manual, a real problem no check
  scores.
- **The careful-reader separators are planted on purpose and findable by
  reading:** the U-value (the data sheet prints a complying centre-of-glass
  value above the whole-window one the section governs), the schedule of
  values' sum, the twin and the renumbered tag (both on sheets that did not
  change between issues), and the cloud with no entry (S-501's revision
  column in the cover's index still says 0, so a reviewer who picks sheets
  by that column skips it; the brief 7 miss on `coord-revisions` was that
  path).
- **The reissue question names the reviewer's four checks** (what changed,
  clouded and listed, each entry matched by a change, carried to the other
  sheets). `coord-revisions` names the first two the same way. It names no
  sheet, value or kind of fault; a test holds that.
- **Every new check is scored in one finding** (review fix). A list of
  changes puts "A-301: no change" on the line beside the next sheet's
  finding, and a window of characters joined them: the first version passed
  replies that said the opposite. `scope="item"` alone would not read a
  yes/no table, the shape the reissue question invites (a row's "No" says
  nothing without its column head), and lets a heading join two items.
- **`parapet-twin` and `door-renumbered` want a word that the other sheet
  was left behind,** not "unchanged": a list of untouched sheets names
  A-301 and A-101 without finding anything. "Still" counts only before
  what a sheet shows ("still shows 109.30", "still tags 106"), not "still
  consistent" or "the plan still agrees"; "conflict", "differs" and the
  like count only without a "no" or "not" before them. The first version
  also asked for 109.60 and 110; a finding about the sheet left behind
  need not name the new value, so they were dropped.
- **The absence words are one per check,** not the shared `_ABSENT` of the
  older tasks: any "no" or "not" within 200 characters passed the section's
  own wording ("water penetration at 450 Pa with no leakage"), a field test
  "not due", and the schedule's sum "not the 178,500".
- **The reissue's false alarms are not scored.** The issue 1 clouds on A-401
  and A-601 stay in issue 2, so a careless reader may call them issue 2
  changes; a check for that would read words like "revised" next to sheet
  numbers and would charge right answers too. The full read of each run is
  where it shows.

## Settings

None added, none changed. Nothing the page runs changes: no prompt, tool,
description or budget, so no fingerprint moves and no switch is needed. The
new code runs only when a suite run builds these documents.

## Tests

New: `packages/aec-document-review/tests/test_dr25_spec_and_reissue_offline.py`.
Each fails on master (the documents, tasks and checks do not exist there)
and passes here:

- `test_the_rules_find_exactly_what_is_planted_in_issue_2`: issue 2 holds
  the 13 base problems and the three of `ISSUE_2_PLANTED`, found again by
  rule, and no other of those kinds; the long set has no issue 2.
- `test_issue_2_differs_from_issue_1_on_five_sheets_only`: the text differs
  on G-001, A-102, A-201, A-601 and S-501 only (the issue line aside); what
  is drawn is the same on every sheet but A-201, A-601 and S-501, which gain
  their clouds (so A-102 has an entry and nothing else).
- `test_the_reissue_faults_read_back_with_the_page_tools`: each fault read
  back with the page's own `open_document` and `read_document`, from both
  issues: the parapet on A-201 and A-301, F3 on S-501 with no revision 2
  row, A-102's entry with door 205 still tagged, the renumbered row on A-601
  with A-101 still tagging 106.
- `test_issue_0_and_issue_1_keep_their_own_dates`.
- `test_the_window_problems_read_back_with_the_page_tools`: the three parts;
  the section listed and not bound in the manual; each of the five problems
  in the documents' text as the page reads it, and what complies; no line
  names a fault.
- `test_the_new_documents_resolve_by_their_suite_ids`: names, page counts,
  no telling word in a file name.
- `test_the_builds_are_the_same_bytes_every_time`.
- `test_the_new_tasks_are_listed_like_the_others`: ids, documents,
  category, one check per planted problem, no planted value in a question.
- `test_a_perfect_answer_scores_full_and_a_careless_one_nothing`: a made-up
  perfect reply scores 5 of 5 and 4 of 4; a careless reply (the centre-of-
  glass value, the printed total, every change taken at its word) and a
  wrong one (S-501 called unclouded, A-102's door taken as moved) score 0;
  so does an empty one.
- `test_each_new_check_takes_its_own_truth_and_no_other`, and
  `test_no_older_check_takes_a_new_truth`.
- `test_a_clouded_change_is_not_an_unclouded_one`,
  `test_the_twin_wants_the_other_sheet_left_behind`: on replies of several
  list items whose neighbours say "clouded and listed" and "still tags"
  of other sheets (review fix: they were one sentence each).
- `test_every_new_check_has_its_table_of_replies` and
  `test_each_check_takes_the_right_replies_and_no_wrong_one` (review fix,
  one case per check): `REPLIES` holds, for each of the nine checks, made-up
  replies that must pass (a sentence, a list item and a table row at least)
  and replies that must fail (the careless reading, the opposite, the
  all-clear, a neighbouring item that says the right word of something
  else, each case the review found). Counts below.
- `test_the_dry_run_resolves_the_new_documents`.

In the core, `packages/aec-agent-core/tests/test_review_checks_together.py`
(review fix): `test_scope_finding_reads_a_table_row_with_its_column_heads`
(a "No" under "Index matches?" passes under `"finding"` and not under
`"item"`; a "Yes" or an empty cell does not) and
`test_scope_finding_reads_a_heading_with_one_item_at_a_time` (a heading
over two items joins them under `"item"`, not under `"finding"`; a finding
with its own nested lines, a heading over one item that holds the rest, and
a plain sentence pass; a plain list of sheets does not).

Existing tests edited (each assertion my change makes untrue):

- `tests/test_package.py`: 52 tasks to 54 (the function's name too) and
  the dry run's "54 tasks".
- `tests/test_review_eval_offline.py`: 14 coordination tasks to 16.
- `tests/test_review_eval_coordination_offline.py`: 14 to 16 tasks, the dry
  run's "16 tasks", and the set of planted checks now includes the new nine.

Run here, each exit 0: the new file with the three edited ones and
`tests/test_review_fingerprints.py` (Document Review's); and from the root
`webapp/tests/test_public_text.py`,
`packages/aec-agent-core/tests/test_core_suite_and_records.py` and
`funhouse_agent/review_eval/tests/test_review_eval_offline.py` (it reads the
core's document list). After the review fixes, also from the root
`packages/aec-agent-core/tests/test_review_checks_together.py` and
`test_ev48_working_text_said.py` (the core's check tests): all exit 0.

Issue 0, issue 1, the 40-sheet set, the scan, the manual, the memorandum and
the concrete submittal build to the same bytes from this branch's module as
from the committed one (a one-off comparison of SHA-256 digests, run here;
not kept as a test, since a different PyMuPDF writes different bytes): all
seven the same. Run again after the review fixes, against the base commit
(a7a48d8): all seven the same, and the 52 older tasks the same in every
field (id, question, documents, category, document type, checks, truth).

## Review fixes (2026-11-18)

An independent review ran replies of its own through `run_check`: the
documents were fair, but the nine checks passed wrong and opposite replies
and failed right ones. What changed: every check reads one finding
(`scope="finding"`, above); the words for "left behind", "a cloud is
there", "not in the revision block" and "nothing changed" are guarded
against "no conflict", "still consistent", "no revision cloud", "without a
revision cloud", "nothing else changed", "otherwise unchanged" and "the
rest of the sheet identical", and read a yes/no cell under its head; the
three absence checks have narrow words of their own; the twin checks no
longer ask for the new value. The package's readme says 54 tasks.

The replies table, the first version of the checks (HEAD before the fix)
against this one, each count "right replies passed" and "wrong replies
passed":

| check | right, before | wrong, before | right, now | wrong, now |
|---|---|---|---|---|
| `window-u-value` | 3 of 3 | 1 of 4 | 3 of 3 | 0 of 4 |
| `window-anchorage` | 3 of 6 | 2 of 4 | 6 of 6 | 0 of 4 |
| `window-section-missing` | 4 of 4 | 1 of 4 | 4 of 4 | 0 of 4 |
| `window-water-test` | 3 of 6 | 5 of 6 | 6 of 6 | 0 of 6 |
| `window-sov-total` | 3 of 3 | 1 of 3 | 3 of 3 | 0 of 3 |
| `parapet-twin` | 2 of 6 | 1 of 8 | 6 of 6 | 0 of 8 |
| `cloud-not-listed` | 3 of 4 | 4 of 6 | 4 of 4 | 0 of 6 |
| `entry-no-change` | 3 of 4 | 3 of 5 | 4 of 4 | 0 of 5 |
| `door-renumbered` | 3 of 4 | 2 of 5 | 4 of 4 | 0 of 5 |

40 right replies and 45 wrong ones in all. The table was written with the
fix, so it shows the fix holds for these replies, not that no other reply
gets past it; the next reviewer's own replies are the test of that.

## The orientation turn (not changed)

`score_review_suite(..., orientation=True)` sends the page's own automatic
first turn before each question: the text the page sends on every upload
(`review_page.ORIENTATION_REQUEST`: open the document, take in its page map
and structure, the contact sheets if it is long, and say in under 250 words
what it is, how it is organised, what reads as text and what must be looked
at, any markups, and what could be done next; "do not review it yet").

- **What it would measure.** Testers always get that turn first: the page
  sends it on every upload (except images alone, mid-conversation). The
  suite by default asks the question cold, a path no tester takes. With it
  on, a task measures the tester's real path, and whether a bearing first
  helps (the sheet list, the index and its revision column, the documents'
  sections, known before the review starts) or hurts (anchoring on the
  index, steps and context spent before the question). On the coordination
  tasks the question is sharp: `coord-reissue` and `coord-revisions` are
  where taking the index's revision column as the map of what changed loses
  a problem.
- **What it costs.** One more turn per task: opening each uploaded document
  (two or three here), its page map, a contact sheet per long document (one
  vision call each), and a reply of up to 250 words. The larger cost is that
  the orientation exchange then rides in every later step's input, a few
  thousand tokens times the task turn's steps. The run's `tokens`, `seconds`
  and model calls include it (`turns[0]` holds it; the score reads the last
  turn's answer). The only orientation runs on record are the ten UFC tasks
  of 2026-10-02 (1.38 to 1.49 million input tokens for ten tasks, 9 to 12
  minutes); no run has measured the same tasks with and without it, so the
  cost per task is not known yet.
- **How to measure it.** Run the same arms twice, once with
  `orientation=True`, into a different `out_dir`: a finished run is keyed by
  arm and task only, so a resume into the same folder would take the
  earlier runs as done.

## The suite before and after

- Tasks: 52 before (38 single-document, 14 coordination); 54 after (16
  coordination). Checks on the coordination tasks: nine more.
- Arms: a task has no arm of its own; the new two run under every arm a run
  names, as the other coordination tasks do. The standing measures (briefs
  7 and 8) ran `baseline`, `show_me`, `method`, `sheet_tools` and
  `all_three`.

## What a first run on a model should look at

Run both under `baseline` and `sheet_tools`, and `method` (its revision
clause covers flagged changes only, brief 7). The checks expected to
separate a careful reviewer from a careless one:

- `window-u-value`: does it compare the whole-window value, or stop at the
  first U-value it reads?
- `window-sov-total`: does it add the schedule up?
- `window-section-missing`: does it check each cited section against the
  manual's contents?
- `parapet-twin` and `door-renumbered`: does it open sheets that did not
  change (A-301, A-101) to see whether a change was carried?
- `cloud-not-listed`: does it compare every sheet, or only those the index's
  revision column or the revision blocks point to?
- `entry-no-change`: does it check an entry against the sheet, or take it at
  its word?

The easier two, `window-anchorage` and `window-water-test`, are a floor: a
reviewer who reads the section's lists against the submittal's should get
both. Read every run in full before believing a score (the owner's rule):
the checks see only that the pieces were put side by side.

## Noted, not fixed

- A run's folder is keyed by arm and task only; `orientation=True` resumed
  into the folder of a run without it reuses those runs silently, and
  `run.json` has no orientation flag (the turns show it). A measurement uses
  its own `out_dir`.
- The transmittal's "DEVIATIONS: NONE" against Section 01 33 00 1.2 is
  wrong here as in the concrete submittal; no check scores it (brief 7,
  item 12, noted the same for `coord-submittal`).
- The reissue's false alarms (the issue 1 clouds reported as issue 2
  changes) are not scored; see the judgement calls.
- A value-pair check (`window-u-value`, `window-sov-total`) passes a table
  row that marks the pair "Yes"; every such check in the suite has this
  limit.
- A finding is a line: a reply written as one long paragraph is one
  finding, and the 300 characters are then all that keep two of its
  sentences apart. Replies on record write findings as list items or rows.
- The checks need the core's `scope="finding"`: a core without it reads
  the scope as none and scores by the window alone. The two packages are
  released together (one tag), so a release has both.
- Not built: a real long drawing set (the owner's documents, private).

## For the lead

- The fixture is the measure, not the design: nothing in the page was
  touched, and no rule should be added to the page because it would help
  these documents.
- A first hosted run of the two tasks costs about what `coord-revisions`
  and `coord-submittal` cost on the same arm (`coord-reissue`: 30 pages,
  as `coord-revisions`; `coord-spec-submittal`: 11 pages, the manual's 5,
  the section's 2 and the submittal's 4, against `coord-submittal`'s 10),
  and compares the arms on the nine checks above, read run by run.
