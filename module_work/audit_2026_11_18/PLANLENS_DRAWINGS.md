<!-- Filed here by the lead on 2026-11-18 from planlens' branch `claude/audit-drawings` (planlens keeps no review file on its main). Its three repairs are merged to planlens' main on 2026-11-18 after a green run of the whole suite on GitHub (after 0.17.0, not released, no tag). What it left open is the app's BACKLOG PL-47. -->

5 findings, 3 fixed here, 2 suspected and not reproduced

# Drawing-set audit, 2026-11-18 (run 2026-11-19 UTC, from main f9b5b38)

Made-up documents only (SketchSheet and `table` from `planlens/testing/`, and
PyMuPDF), called as an agent would; findings judged by what the agent is handed.

## Findings, by how often an ordinary set meets them

| # | File, function | Made-up sheet | Agent is handed | What was drawn | Fixed? | How often |
|---|---|---|---|---|---|---|
| 1 | `scalefinder._plan_scales` / `measuring._distance_scale` | one sheet, DETAIL 1 "SCALE: 1/4" = 1'-0"" at left, DETAIL 2 "SCALE: 1/2" = 1'-0"" at right, a 72 pt line in each | `measure kind=distance` in DETAIL 2: **4.0 +/- 0.3 ft, confidence 0.6**, scale p0.plan.stated (the 1/4" one), and only `reconciled`: "the stated scale disagrees with the stated by 50.0 % (the sheet may have been re-plotted at another size); the stated is used" | 2.0 ft (1 in of paper at 1/2" = 1'-0") | noted: needs each view's region; not small | most detail and section sheets (several views, several scales) |
| 2 | `scale.parse_ratio` | one detail, "SCALE: 1-1/2" = 1'-0"" | page listing: "no scale was found on this page"; distance in points | a scale of 1:8 | **fixed** (305e3e1) | common: 1-1/2" = 1'-0" is a standard detail scale, usually lettered with the hyphen; a title block printing it gave no `sheet_index` scale either (`sheets._scale_value` returned None) |
| 3 | `measuring._convert` | plan with "SCALE: 1/4" = 1'-0"" (also one with a 0.95 bar scale only) | `measure kind=line` on a drawn line: `scale_known: false`, "no scale on this page: the position is in page points", while `measure` with no box lists the scale | a page with a scale | **fixed** (9ce4eef): now `scale_known: true` and the note sending it to kind=distance, as kind=point already did | every plan an agent measures with kind=line before kind=distance |
| 4 | `markup_writer._place` (note) | 1224 x 792 sheet, note at [1220, 788] | written row bbox [1220, 788, 1236, 804], no warning; PyMuPDF and pypdf both read an icon 12 pt off the right and bottom edges | a note at the sheet's corner | **fixed** (27c04c6): the icon is moved back onto the sheet; elsewhere unchanged (tested at /Rotate 0 and 90) | uncommon: a spot within 16 pt of the right or bottom edge |
| 5 | `sheets.sheet_index` | 7 sheets numbered A-101, A1.01, S-2, C-001A, a-102, "A 103", S-201 | five indexed; "a-102" and "A 103" not indexed, with "2 of 7 sheets could not be indexed from their text" | seven sheets | noted (the index says so honestly) | rare (lower case); occasional (a space) |

## Question 2: the false-alarm count

**0 callouts reported** over two made-up sets of 6 sheets each (12 sheets):
set 1, 24 lines of ordinary text per sheet (144 lines); set 2, 30 lines per
sheet (180 lines) plus six grid bubbles (A, B, C, 1, 2, 3) and two room tags
per sheet, in the fixture's strip title blocks. The text held dates with
slashes (03/14/2026, 11/02/25, 6/30/2026), fractions of an inch (1/2", 5/8",
3/4", 1-1/2"), grid references (A/3, B/4, C/7, 1/A, "UNIT 3/A"), room numbers,
two phone numbers, ratios (1:2, 2/1, 4/12 pitch, 2/3), N/A, A/C, S/S, W/ 2/3.
Result: "no callouts were recognised in the text".

The real callouts were all judged right: 1/A-201 resolved, 9/A-201 unresolved
(no view 9), 2/A-301 not_in_document, 3/S-201 (written) resolved; and on a set
numbered A1.01, A2.01, S-2, C-001A, C-001B: 7 of 7 right (bubbles and written,
including 4/A2.01 unresolved, 5/A3.01 not_in_document, 1/S-2 unresolved).

## What else was run, and what came back right

- **Q1 sheet index.** A-101, A1.01, S-2, C-001A read (finding 5 for the rest).
  A sheet with a second, smaller title block (X-91 at 24 pt beside the 40 pt
  number): the right number. A cover listing S-101 (not in the set) and not
  A-301 (in the set): the index lists the five pages read from title blocks;
  the comparison with the cover is the agent's, as the tool says. **60 sheets:
  0.10 s**, all 60 right.
- **Q3 schedule marks** (4 sheets): FCU-01, FCU-02, FCU-03A (leading zero,
  suffix), door D-101A on two floors, FCU-05/06 in "FAN COIL UNIT SCHEDULE
  (CONTINUED)" on a second sheet. Two schedules, 8 rows, 6 found; no_row
  FCU-04 (planned, not scheduled); no_mark D-103 and FCU-05. All true; the
  continued schedule merged into one.
- **Q4 scales** (8 sheets, 9 distances of a known line): 1/4" = 1'-0" 16.0 ft
  (true 16.0), 1" = 20' 80.0 ft (80.0), 1:100 10.16 m (10.16, with the "no
  unit, metres" warning), bar scale only 16.0 ft at confidence 0.95; stated
  scales carry 0.7 (the measurement 0.6 unsnapped) and the re-plot warning.
  NTS and NOT TO SCALE: no scale, distance in points. Then findings 1 to 3.
- **Q5 marked copy.** 12 marks over 4 pages (/Rotate 0, 90, 270; a crop box
  offset 200, 150 from the media box), read back with PyMuPDF (rect x
  rotation matrix) and pypdf (/Rect against /CropBox): every box at the place
  asked, on the right page; highlights over the quoted words on the rotated
  and the cropped page; a 1,999-character comment whole in both readers;
  "Béton armé – Größe ≥ 3 m², Ø, ñ, 日本" whole. 300 marks (box, note, circle,
  callout over 4 pages): 300 written in 0.55 s, 300 read by each reader, 0 on
  the wrong page; `document_markups` counts 300. Marks off a page are refused
  with the page size named. `markup_check.check_marks` with a recording look:
  5 crops, each holding the red mark at every rotation and crop (the same red
  pixel count on all four pages); an answer it cannot parse is "unsure".

## Suspected, not reproduced

1. A stated scale and a bar that disagree (a reduced print): finding 1's
   wording ("the stated is used") suggests the bar may be passed over; not drawn.
2. A note's popup on a rotated, cropped page: the note lands right; where its
   popup opens was not checked.

## Tests

- New: `planlens/document/tests/test_audit_drawings.py`, 9 tests; each fix's
  tests fail on main (3, 2 and 1 failing) and pass here.
- Neighbouring suites after each fix: 387 passed (scale, scales, sheets,
  quantities, measuring); 146 passed, 4 skipped (markup writer, marked copy,
  markup check); 115 passed (measuring, planlocation, scalefinder, dr44).
- Whole suite `pytest planlens -q` on this branch: 2653 passed, 16 skipped,
  exit 0 (7 min 33 s). No tool name, description or schema changed.

## What was read

In part: `crossrefs.py` (docstring, bubble and text-callout readers),
`sheets.py` (`_scale`), `scale.py` (ratio parser), `measuring.py` (`measure`,
`_convert`, `_measure_lines`, `_measure_distance`, `_plan_point`),
`markup_writer.py` (`_place`, `_page_box`, `_callout_box`), `markup_check.py`
(`check_marks`, `_check_one`), `review/tools.py` (`call`,
`annotate_document`), `tools/specs.py` (measure, sheet_index), the
`schedules.py` docstring, the fixtures. **No file was read whole.** Not
reached: `scales.py`/`scalefinder.py` beyond the stated-scale code, `roles.py`,
`quantities.py`, `planlocation.py`, `annotations.py`, `marked_copy.py` beyond
its function list, `review/frames.py`, and `planlens/review/DESIGN.md`.
