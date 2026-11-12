# Whole-set review of the 15-sheet set (`coord-full-review`): six runs read in full

Reader: group `wholeset15`. Runs read: GPT-5.6 Sol in five arms (`baseline`, `show_me`, `method`, `sheet_tools`, `all_three`) and GPT-5.4 `baseline`, all under `raw/b7_coord_sol/runs/<arm>/coord-full-review/` and `raw/b7_coord_gpt54/runs/baseline/coord-full-review/` (raw = `module_work/field_feedback/2026-11-12_foundry_brief7/raw/`, local only). For each run I read every record of `activity.jsonl` (the main agent, its helpers and the vision side calls), `run.json`, the delivered comment log (Word for Sol, Excel for GPT-5.4) and the console log. "#N" below is record N of that run's `activity.jsonl`, counting from 0.

The question was: "Do a coordination review of this drawing set before it goes out, and check it against the project manual and the geotechnical memorandum. Give me a comment log: one row per comment, with the sheet and what is wrong." The set has 13 planted problems (`BASE_PLANTED` in `coordination_fixture.py`): index lists A-202 (absent); index titles S-201 ROOF FRAMING PLAN; A-101 calls 7/A-501 (A-501 has details 1 to 6); A-102 calls 2/A-302 (no such sheet); door 109 on plan, not scheduled; door 204 scheduled, not on plan; second floor EL. 104.20 on A-201/S-201 vs 104.50 on A-301; bay B-C 7200 on the plans vs 7500 on A-401 (revision 1); G-002 cites 07 84 00, not in the manual; S-001 200 kPa vs memo 150 kPa; 1.2 m vs 1.5 m footing depth; 25 MPa vs 30 MPa footing concrete; F4 on S-101 not in the S-501 schedule.

## Headline findings

1. **Delivered, counted by hand on the delivered log:** Sol baseline 11/13, show_me 10/13, method 11/13, sheet_tools 13/13, all_three 12/13; GPT-5.4 6/13. The score table says 11, 12, 11, 13, 12 and 9. The table is wrong for show_me (2 false passes) and for GPT-5.4 (3 false passes), all from scorer defects below.
2. **The scorer reads the agent's private page notes as delivered work.** `run.json` lists `.scratch/page_readings.json` and `.scratch/reads.json` among the run's files and the checks read every listed file. Every false pass in this group came out of those notes, helped by two more defects: the answer's last bold line becomes a "heading" over all the appended file text, and a door number matches inside a decimal (`204.9`, a bounding-box coordinate; `EL. 109.30`). The last one also passes a check on GPT-5.4's delivered workbook itself.
3. **Nothing planted was missed for lack of reading.** Every Sol run read both halves of every planted problem somewhere. What was lost was lost in putting two readings side by side, or after: three times a problem was named in the main model's own reasoning and then left out when it wrote the log (method: 7/A-501; all_three: the bay; baseline: the bay, "needs closer inspection", never inspected).
4. **The sheet tools did what they say.** `check_cross_references` returned exactly the two broken callouts (7/A-501 unresolved, 2/A-302 not in the document) within 15 seconds, correctly, and both runs that had it delivered both. The three Sol runs without it all read the 7/A-501 callout and A-501's six details in separate side calls and none delivered it.
5. **The `show_me` option was never used.** It was offered (the tool schemas differ from baseline's) but none of the 60 looks in `show_me` and none of the 56 in `all_three` asked for it. On this task the show_me arm measured run-to-run variation only.
6. **GPT-5.4** looked at 10 of the 15 sheets (never at G-001, G-002, A-501, S-001 or S-501, which it read as text only), never compared a plan's door tags with the schedule, never read the 7500, and 13 of its 28 rows repeat another row.
7. **Wrong statements are few:** two in Sol baseline (wrong detail numbers on A-501) and one in GPT-5.4 (a revision said to be shown "only by a cloud" when A-601's revision block names it). None in the other four runs.

## Run by run

### Sol baseline (`b7_coord_sol/runs/baseline/coord-full-review/`)

- **Size:** 272 s; 171 model calls (15 main, no helpers, 156 vision); 1.93 M tokens; 41 tool calls; no errors.
- **Path:** opened the three PDFs, page maps, a contact sheet, read the manual and memo text, looked at the contact sheet (one vision call), then 15 page readings in parallel, one per sheet. The per-page prompts were written from guessed contents (A-401 "wall sections", A-601 "door and window schedules"), not from anything read. Then: the drawing text (3 calls); searches for A-302, A-202, "EL. 104.", 109 and 204; `find_quantities`; three reasoning steps; `write_docx` with 30 rows.
- **Delivered:** a Word log of 30 rows, plus a short answer.
- **Planted problems:**
  - Delivered (11): A-202 (row 1), S-201 title (2), 07 84 00 (4), 2/A-302 (5), level (6), bearing (11), depth (12), concrete (13), F4 (15), door 109 (25), door 204 (26).
  - Seen, not delivered: 7/A-501. The A-101 reading (#342) says "Detail callouts 3/A-501 and 7/A-501 ... Confirm both details exist" and the A-501 reading (#279) lists six detail viewports. The two were never put side by side, and the main model never mentions the callout.
  - Seen, not delivered: the bay. The A-401 tile reading (#200) says "Revision-clouded dimension 7500 ... Verify the revised dimension", and the drawing text (#399) holds both 7500 and 7200. The main reasoning (#413) notes "a conflict between core dimensions in different drawings that needs closer inspection". No inspection followed.
- **Wrong statements:** rows 23 and 24 (detail numbers; see the hand count).
- **Check verdicts:** all 13 correct. The two failures are real misses. Their near-misses (closest 249 characters for 7/A-501) were measured on the page notes.
- **Waste:** little. One pass over the set, no helper. The main model's 15 calls cost 1.25 M tokens because each re-sends the whole conversation (about 105 k tokens a call by the end).

### Sol show_me (`.../show_me/coord-full-review/`)

- **Size:** 627 s; 349 model calls (11 main, 21 helper, 317 vision); 2.66 M tokens; no errors.
- **Path:** the main agent split the sheets among three helpers by page range (0-4, 5-9, 10-14). At most two ran at once; the third waited 222 s.
  - **Helper 1** wasted four calls (`list_files` twice, opened the drawing set twice). It transcribed the index with A-202 in it twice (#378, #580) without flagging it. It transcribed G-002 note 6 with 07 84 00 (#207), but its brief gave no manual section list, so it had nothing to compare it with. It reported A-302, door 109's wall line, two C-101 items and door 204 (as a "question").
  - **Helper 2** reported the level, windows, the 1/A-501 placement, door 109 unscheduled, door 107's HW-2 and placeholder details. It read 7500 on both A-401 views (#280, zoom #347), but A-101 and A-102 were outside its range.
  - **Helper 3** was given S-001 to S-501 and also re-read A-101, A-102, A-201, A-301, A-401 and A-501 (60 vision calls outside its range). It had 7500 (#657), 7200, 7/A-501 (#700) and A-501's six details (#634), and compared none of them.
  - **The main agent** read the drawing text, re-looked at 7 pages, found A-202 itself (#815) and wrote 26 rows.
- **Planted problems:**
  - Delivered (10): A-202, S-201, A-302, 109, 204, level, bearing, depth, concrete, F4.
  - Seen, not delivered (3): 7/A-501, 07 84 00 and the bay (trace in the loss table).
- **Wrong statements:** none. Two rows are questionable (10 and 23; see below).
- **Check verdicts:** two wrong. `callout-missing-detail` and `spec-section-missing` pass in the table, but neither 7/A-501 nor 07 84 00 is anywhere in the Word log. Both passes come from neutral transcriptions in the page notes, carried by the heading defect (follow-up F2). The answer's last bullet is bold ("F4 is used on S-101 but absent from the footing schedule."), holds "absent", and so becomes the heading over everything appended after it.
- **show_me:** offered, never used.

### Sol method (`.../method/coord-full-review/`)

- **Size:** 744 s; 350 model calls (17 main, 7 helper, 326 vision); 3.27 M tokens, the most of the six; no errors.
- **Path:** the main agent did the whole review itself. It looked at contact sheets (3), read all three documents' text, ran a regex search, ran `find_quantities` three times, made 15 sheet readings in two batches and 9 manual and memo page readings (the hardware page took 2 calls and 170 s). Then it asked one helper to "independently synthesize" a high-confidence list from all three files. The helper made 15 page readings and 8 zooms and returned 11 items, without 7/A-501 and without the bay. The main reasoning (#878) lists "callouts like 7/A501 are missing from A101" among the comments to log; the write (#882) has 17 rows without it.
- **Planted problems:**
  - Delivered (11): A-202, S-201, 07 84 00, A-302, level, 109, 204, bearing, depth, concrete, F4.
  - Seen, not delivered: 7/A-501, lost at the write step. The main agent's own A-101 reading (#214) says "7 / A-501 is legible ... no incorrect reference can be established from this tile alone", and the A-501 reading (#421) shows six details.
  - Seen, not delivered: the bay. The drawing text (#77) holds both numbers, and the main A-401 reading (#250) and the helper (#833) read 7500. They were never paired.
- **How 07 84 00 was found:** the G-002 page reading (#128) was asked about "references absent from the supplied TOC". The side call answered, correctly, that no TOC is visible on that page. The main agent compared the two itself and delivered it as row 3.
- **Wrong statements:** none. **Check verdicts:** all correct. The `callout-missing-detail` near-miss (209 characters) was measured on the page notes.

### Sol sheet_tools (`.../sheet_tools/coord-full-review/`)

- **Size:** 591 s; 341 model calls (14 main, 7 helper, 320 vision); 2.63 M tokens; no errors.
- **Path:**
  - At 11 s the main agent ran `sheet_index` and `check_cross_references`, together with the page map, contact sheets and the manual and memo text.
  - It made 15 sheet readings. The A-401 prompt asked for "coordination back to overall plans".
  - It read the drawing text again, ran 2 searches and made 9 manual and memo readings.
  - Its reasoning (#447) pairs the two numbers: "the enlarged core plans say 7500, but overall plans say 7200".
  - It gave one helper 14 numbered findings to confirm or reject (#456). The helper looked for itself, including A-501's "numbering runs 1-6 only; no Detail 7 panel is visible" (#701). It confirmed 1 to 12 and qualified 13 and 14 (#835).
  - `write_docx`, 18 rows.
- **Planted problems:** all 13 delivered. The bay is row 9, with the right sheets (A-101, A-102 and A-401, to be reconciled with S-101, S-201 and S-202).
- **Wrong statements:** none. **Check verdicts:** all correct. The tool results are covered in their own section below.

### Sol all_three (`.../all_three/coord-full-review/`)

- **Size:** 486 s; 352 model calls (14 main, 7 helper, 331 vision); 2.94 M tokens; no errors.
- **Path:**
  - The main agent ran `sheet_index` at 10 s, then `check_cross_references`, all three documents' text and `find_quantities` at 13 s, then 15 sheet readings.
  - It sent one helper to "independently synthesize" a punch list, with a checklist: index, cross-references, tags against the schedule, levels, S-sheets against the memo and spec, TOC against spec references. Enlarged plans against overall plans were not on it.
  - The helper ran both sheet tools again (#455), made 24 page readings and 8 zooms ("Detail 7 does not exist", #868), and returned 11 items without the bay (#871).
  - The main agent made 3 zooms and 2 page re-reads.
  - Its reasoning (#897) states "A-401 shows B-C as 7500 while parent grids say 7200". The next two steps (#901, #905) settle on "13 clear and definite comments" without it. The log has 14 rows.
- **Planted problems:** 12 delivered. The bay was lost at the write step (trace below).
- **Wrong statements:** none.
- **Check verdicts:** all correct.
- **show_me:** offered, never used.
- **Side note:** #897 also reasons that 07 84 00 is "explicitly excluded, so that's fine", yet row 3 still delivers it, correctly.

### GPT-5.4 baseline (`b7_coord_gpt54/runs/baseline/coord-full-review/`)

- **Size:** 401 s; 119 model calls (13 main, 106 vision); 1.14 M tokens; no errors.
- **Path:**
  - Opened the documents, page maps, one contact sheet. Its reading of the contact sheet (#39) calls the set "a small structural drawing set dominated by formwork / framing elevations and sections", which is wrong; it had no visible effect.
  - Read the text of only the five pages the page map called "text" (G-001, G-002, A-501, S-001, S-501), and ran seven one-word searches (door, window, concrete, soil, bearing, geotechnical, slab).
  - Looked at 10 sheets: C-101, A-101, A-102, A-201, A-301, A-401, A-601, S-101, S-201 and S-202. It never looked at G-001, G-002, A-501, S-001 or S-501, and never read the text of the ten it looked at.
  - Read manual pages 2 to 5 and the memo, and ran nine searches, two for values that are not there ("3000 psi", and "200 kPa" in the memo).
  - Wrote one `write_xlsx` with 28 rows. GPT-5.4 records no reasoning text, so the record cannot say why it left out what it had read.
- **Planted problems:**
  - Delivered (6): S-201 title, level, bearing, depth, concrete, F4.
  - Seen, not delivered (6): A-202, 7/A-501, A-302, door 109, door 204, 07 84 00 (trace in the loss table).
  - Never seen: the bay. 7200 was read on the plans many times. 7500 was never read: the A-401 reading (#281) says "Other small tags may exist in tiles, but not reliably readable here".
- **Wrong statements:** row 14 (see the hand count).
- **Check verdicts:** three wrong.
  - `callout-missing-sheet` passes on "2 / A-302" in a page note's table, whose first column holds the words "Not visible" (#188).
  - `door-not-on-plan` passes on a bounding-box coordinate "204.9" near the revision table's header "NO.".
  - `door-not-scheduled` passes on the page note "Door 109 tag appears offset from the door". Scored on the delivered workbook alone it still passes, on "T.O. PARAPET EL. 109.30" in row 23 with "not" in "sheet status is not coordinated" (follow-up F3).
  - Real score: 6/13.

## Hand count of every log row

Classes: **planted** (names a planted problem), **wrong** (a statement the fixture contradicts), **true-not-planted** (true, not planted; "weak" means a preference or practice point rather than a defect), **request** (asks to confirm, asserts no defect), **duplicate** (repeats another row of the same log).

| Run | Rows | Planted | Wrong | True-not-planted | Request | Duplicate |
|---|---|---|---|---|---|---|
| Sol baseline | 30 | 11 | 2 | 14 | 3 | 0 |
| Sol show_me | 26 | 10 | 0 | 13 | 3 | 0 |
| Sol method | 17 | 11 | 0 | 6 | 0 | 0 |
| Sol sheet_tools | 18 | 12 (13 problems; row 12 holds 109 and 204) | 0 | 6 | 0 | 0 |
| Sol all_three | 14 | 12 | 0 | 2 | 0 | 0 |
| GPT-5.4 | 28 | 6 | 1 | 7 | 1 | 13 |

Row by row:

**Sol baseline**
- Planted: rows 1, 2, 4, 5, 6, 11, 12, 13, 15, 25, 26.
- Wrong: rows 23 and 24.
- True-not-planted:
  - 7: windows untagged.
  - 8: A-201 grade unlabelled.
  - 9: C-101 has no spot grades.
  - 10: the memo's date is not cited (weak).
  - 14: the slab concrete class is not on the framing plans (weak; S-001 has it).
  - 16: footing bottoms about EL. 97.60 to 97.50, correct arithmetic.
  - 17: base material and compaction (weak).
  - 18: the inspection hold point shares a note with fill removal (weak).
  - 19: no dewatering note (weak).
  - 20: A-301 shows no footing or slab reference. Partly questionable: A-301 does call out the wall base, 6/A-501.
  - 21: A-501 and S-501 notes do not fit their titles.
  - 22: no thicknesses.
  - 29: no room numbers (weak).
  - 30: no roof drainage.
- Request: 3 (the issue convention), 27 (door 108 HW-4 with no rating), 28 (door 107 HW-2).

**Sol show_me**
- Planted: rows 1, 2, 3, 6, 7, 8, 13, 14, 15, 18.
- True-not-planted:
  - 4 and 5: C-101 location and grades.
  - 9: windows.
  - 11 and 12: A-501 placeholder details.
  - 16: S-001 lacks w/cm, aggregate and air. True to 03 30 00; weak.
  - 17: weak.
  - 20: footing depth not demonstrable.
  - 21: S-201 shows no core opening. True, and it points at the revised core.
  - 22 and 23: untagged framing.
  - 24 and 25: S-501 details.
- Request: 10, 19, 26.

**Sol method**
- Planted: rows 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14.
- True-not-planted: 4 (C-101 location), 7 (only south and east elevations), 10 (A-501 detail 4), 15 (no footing bottom elevations), 16 and 17 (S-501 details 3 and 4).

**Sol sheet_tools**
- Planted: rows 1, 2, 4, 7, 8, 9, 10, 12, 13, 14, 15, 16.
- True-not-planted: 3 (memo date, weak), 5 and 6 (C-101), 11 (A-501 detail 4), 17 (deck span and untagged members), 18 (S-501 details).

**Sol all_three**
- Planted: rows 1 to 8 and 10 to 13.
- True-not-planted: 9 (A-501 detail 1's notes), 14 (S-501 details).

**GPT-5.4**
- Planted: rows 1, 11, 15, 16, 17, 20.
- Duplicates (13):
  - 2 and 21 repeat row 1.
  - 22 repeats 11.
  - 24 repeats 20.
  - 25 and 26 repeat 18.
  - 9 repeats 8 for the other floor.
  - 3, 4, 5, 6, 7 and 23 repeat row 28's revision-block point sheet by sheet.
- True-not-planted:
  - 8: no room numbers (weak).
  - 10: no window tags or materials.
  - 12: no toilet fixtures on A-401. True; the fixture draws only a box and the word TOILET.
  - 13: A-401 revised the core layout but A-101 and A-102 stay at Rev. 0. True, and the nearest it came to the bay, without the numbers.
  - 18: S-501 copied notes.
  - 27: G-002 does not cite 05 12 00 or 05 31 00 (weak).
  - 28: unrevised sheets show only Rev. 0 on an Issue 1 set (weak). Normal practice, and the fixture intends it. GPT-5.4 asserts "out of sync" seven times.
- Wrong: row 14.
- Request: row 19.

**Wrong statements, and why:**
- Sol baseline row 23 says rigid insulation is called out on "A-501 Details 2, 3, 5 and 6". The fixture prints RIGID INSULATION only on details 3 and 6; 2 and 5 carry granular base, sealant and flashing. The comment's point (no thickness or type) is fair; its detail numbers are wrong.
- Sol baseline row 24 says sealant, backer rod and flashing are on "Details 2, 3, 5 and 6". The fixture has them only on 2 and 5.
- GPT-5.4 row 14 says door 107's hardware revision "is shown only by a cloud in the schedule" and that "the drawing set does not show a coordinated explanatory note". A-601's revision block carries Rev. 1, "REVISED DOOR HARDWARE", dated 2026-03-02.

**Questionable, not counted as wrong:**
- Sol show_me row 10 objects that 1/A-501 sits on the core plan while detail 1 is "Core Wall at Floor Slab". That is the matching place, and the row only asks to confirm.
- Sol show_me row 23 calls S-202's "internal thin-line boundary/opening" unidentified. The fixture's inner line is the wall's inner face, drawn 200 mm inside the outline; it is not an opening. The row only asks for it to be identified.
- Sol baseline row 20 (above).

## Where each seen-but-not-delivered problem was lost

| Run | Problem | Where both halves were read | Where it was lost |
|---|---|---|---|
| Sol baseline | 7/A-501 | A-101 reading #342 ("Confirm both details exist"); A-501 reading #279 (six details), in separate side calls | Never carried into the main model's reasoning. It searched the text for A-302, A-202, 109 and 204, but not for this callout. |
| Sol baseline | bay | A-401 tile #200 ("Verify the revised dimension"); drawing text #399 | Main reasoning #413 notes "a conflict between core dimensions ... needs closer inspection". No inspection; left out of the write. |
| Sol show_me | 7/A-501 | Helper 1 (#210); helper 3 (#700, with A-501 at #634) | Neither helper compared them and neither report has it. The main agent never looked at A-101. |
| Sol show_me | 07 84 00 | Helper 1 transcribed note 6 (#207); the main agent had both the G-002 text and the manual's TOC text | Helper 1's brief held no section list. The main agent never compared the two. |
| Sol show_me | bay | Helper 2 (#280, #347: 7500 on both views); helper 3 (#657, plus 7200 on the plans) | The page-range split left helper 2 without A-101 and A-102. Helper 3 had both and did not compare them. The main agent never paired them. |
| Sol method | 7/A-501 | Main A-101 reading #214; A-501 reading #421 | Main reasoning #878 lists it as a comment. The write #882 leaves it out. |
| Sol method | bay | Drawing text #77 (both numbers); main A-401 reading #250; helper #833 | Never paired anywhere. |
| Sol all_three | bay | Drawing text #81; main A-401 reading #241; helper #638 | Main reasoning #897 states it. #901 and #905 drop it before the write (#906). |
| GPT-5.4 | A-202 | The G-001 text (#56) holds it | Never named. It never looked at G-001. |
| GPT-5.4 | 7/A-501 | A-101 reading #195 transcribes it | A-501 was never looked at (text only), so there was nothing to compare with. |
| GPT-5.4 | 2/A-302 | A-102 reading #188 transcribes it | Never compared with the index. |
| GPT-5.4 | door 109 | A-101 reading #195 (its only comment: "tag appears offset from the door"); schedule reading #278 (101 to 108) | Never compared. |
| GPT-5.4 | door 204 | A-102 reading #188 (tags 201, 202, 203, 205, 206); schedule reading #278 (has 204) | Never compared. |
| GPT-5.4 | 07 84 00 | G-002 text #56; TOC text #302 | Never compared. |

Two patterns. In Sol, a cross-sheet problem whose halves come from two separate side calls is the one that gets lost, either never paired or paired and then dropped. In GPT-5.4, nothing was compared across sheets except the S-201 title (the index and the title both arrived as text).

## Bay B-C, 7200 against 7500

| Run | 7500 read by | Both numbers side by side? | Delivered |
|---|---|---|---|
| Sol baseline | A-401 tile reading #200 (flagged "verify"); drawing text #399 | Only in the main model's text (#413), which noticed and moved on | No |
| Sol show_me | Helper 2 (#280, zoom #347); helper 3 (#657) | No call held both | No |
| Sol method | Main A-401 reading #250; helper #833; drawing text #77 | The text result held both; nothing paired them | No |
| Sol sheet_tools | A-401 reading #381 ("Confirm revised 7500 dimension is coordinated with the overall plan"; its prompt asked for coordination back to the overall plans); drawing text #417 | Main reasoning #447 | Yes, row 9 |
| Sol all_three | Main A-401 reading #241 ("its right endpoint continues beyond this tile, so the complete dimension reference cannot be verified here"); helper #638; drawing text #81 | Main reasoning #897 | No (dropped at the write) |
| GPT-5.4 | Never read | No | No |

No run ever looked at A-401 and an overall plan together. Where the pairing happened it was done in the main model's text, from two separate readings. The one run that delivered it had an A-401 reading that itself pointed at the overall plans, because of how that run happened to word the page prompt.

## Detail 7/A-501: the 249- and 209-character near-misses

- Sol baseline's table entry reads "all present, never within 200 characters of each other (closest: 249)"; Sol method's reads "closest: 209".
- Neither delivered Word log says "7/A-501", "detail 7" or anything about A-101's east-wall callout. Baseline row 21 and method row 10 are about A-501's notes not matching its detail titles, which is a different problem.
- A person would say it was not reported. Both failures are correct verdicts.
- The near-miss distances were measured over the page notes, where the callout is transcribed with a "no" or "not" some 209 to 249 characters away. Had those words fallen inside 200 characters, the check would have passed on the notes, as it did in show_me.

## The sheet tools: what they returned and what the model did with them

- **`sheet_index`** (sheet_tools #24; all_three #23, and once more in each helper) returned the 15 sheets with the number, title and revision their own title blocks print, and the view marks it recognised. It matches the fixture: S-201 SECOND FLOOR FRAMING PLAN; revision 1 on G-001, A-401 and A-601; A-501 views "1 to 6"; S-501 views "1 to 4". It does not list the cover's index entries, so the A-202 and S-201-title findings still needed the model's own reading of G-001. Both runs made them.
- **`check_cross_references`** (sheet_tools #31; all_three #42 and helper #455) returned 17 callouts:
  - 7/A-501, unresolved: "view marks recognised on A-501: 1 to 6; none for 7".
  - 2/A-302, not in the document: "no sheet A-302 was read in this document; the document holds 15 sheets".
  - 15 resolved.
  - Every line is correct against the fixture. Its note says a callout to a sheet not in the document is a finding only if the file is the whole set, and to compare with the cover's index. Both runs did.
- **Use.** Neither run ignored or contradicted either tool.
  - sheet_tools put both callouts into its checker's brief as findings 4 and 5. The helper confirmed both by looking (#701, #835). Log rows 7 and 8.
  - all_three's helper ran the check again and zoomed A-501 to confirm (#868, "Detail 7 does not exist"). Log rows 4 and 5.
  - The tool's "look before you comment" advice was followed in both runs, at the cost of a few vision calls.

## Side by side, and arm against chance

| Planted problem | baseline | show_me | method | sheet_tools | all_three | GPT-5.4 |
|---|---|---|---|---|---|---|
| index lists A-202 | D | D | D | D | D | S |
| index title S-201 | D | D | D | D | D | D |
| callout 7/A-501 | S | S | S | D | D | S |
| callout 2/A-302 | D | D | D | D | D | S |
| door 109 not scheduled | D | D | D | D | D | S |
| door 204 not on plan | D | D | D | D | D | S |
| second floor 104.20/104.50 | D | D | D | D | D | D |
| bay 7200/7500 | S | S | S | D | S | N |
| 07 84 00 not in manual | D | S | D | D | D | S |
| bearing 200/150 kPa | D | D | D | D | D | D |
| depth 1.2/1.5 m | D | D | D | D | D | D |
| concrete 25/30 MPa | D | D | D | D | D | D |
| F4 not scheduled | D | D | D | D | D | D |
| **Delivered (hand)** | **11** | **10** | **11** | **13** | **12** | **6** |
| Score table | 11 | 12 | 11 | 13 | 12 | 9 |

D = delivered; S = seen, not delivered; N = never seen.

There is one run per arm, so most differences here cannot be told from chance. What the traces do show:

- **7/A-501 (sheet tools):** the tool's result is the visible cause. It arrived at 11 to 13 s and went straight onto both lists. Without it, three Sol runs out of three read both halves and none delivered it. The cause is mechanical and it holds 2 of 2 against 0 of 3, but the numbers are small.
- **Bay (chance):** every Sol run saw it. One delivered it, one stated it and dropped it, and one noted it and moved on. That looks like run-to-run variation, plus whether the A-401 page prompt happened to mention the overall plans.
- **07 84 00 in show_me (how the run split its work):** missed only there, because the work was split so that G-002 went to a helper without the manual. That is a consequence of the split, not of the show_me switch, which was never used.
- **method:** no visible effect. It delivered the same 11 as baseline. Its pattern (the main agent reads everything, then one second-opinion helper) also appears in sheet_tools, which has no method text.
- **GPT-5.4 against Sol baseline:** 6 against 11. It did about two thirds of the vision work (106 vision calls against 156), looked at 10 sheets of 15, and made no plan-to-schedule comparison. It is one run, but the gap is wide and each missing item has a visible cause in its path.

## Check verdicts, checked

| Run | Table (13 planted) | Hand count | Wrong verdicts |
|---|---|---|---|
| Sol baseline | 11 | 11 | none |
| Sol show_me | 12 | 10 | `callout-missing-detail`, `spec-section-missing` (passed on page notes, via the heading carry-over) |
| Sol method | 11 | 11 | none |
| Sol sheet_tools | 13 | 13 | none |
| Sol all_three | 12 | 12 | none |
| GPT-5.4 | 9 | 6 | `callout-missing-sheet` ("Not visible" column), `door-not-on-plan` (bbox "204.9" + "NO."), `door-not-scheduled` (page note; and on the workbook itself via "EL. 109.30") |

The two extra checks (does not give up on legibility; no 0-based page citation) pass in all six runs, and I agree with them. `door-not-on-plan` passes on 12 to 24 of the single page readings in each run taken alone, because bounding-box coordinates hold "204" and the readings' tables hold "no" or "not". In this group it gave a false pass only where the log lacked door 204 (GPT-5.4).

## Where the time, calls and tokens went

| Run | Seconds | Model calls (main / helper / vision) | Tokens | Vision tiles | Log rows |
|---|---|---|---|---|---|
| Sol baseline | 272 | 15 / 0 / 156 | 1.93 M | 135 | 30 |
| Sol show_me | 627 | 11 / 21 / 317 | 2.66 M | 252 | 26 |
| Sol method | 744 | 17 / 7 / 326 | 3.27 M | 270 | 17 |
| Sol sheet_tools | 591 | 14 / 7 / 320 | 2.63 M | 270 | 18 |
| Sol all_three | 486 | 14 / 7 / 331 | 2.94 M | 270 | 14 |
| GPT-5.4 | 401 | 13 / 0 / 106 | 1.14 M | 90 | 28 |

- **One pass over the 15 sheets** costs 150 vision calls (one overview and nine tiles a page) and about 0.55 to 0.6 M tokens.
- **The main agent's own calls** are 19 % (show_me, which handed the reading to helpers) to 66 % (the two runs without helpers) of a run's tokens. Each of its 11 to 17 calls re-sends the whole conversation, about 105 to 146 k input tokens a call by the end.
- **method, sheet_tools and all_three** each ended with a helper that re-read all 15 sheets: about 150 vision calls, 0.18 to 0.31 M helper tokens on top, 3 to 4 minutes. No helper found a planted problem the main list lacked.
  - In method and all_three the helper's list was shorter than what the main model had already reasoned out. Both missed the bay, and method's helper also missed 7/A-501.
  - Only in sheet_tools, where the helper was asked to confirm a list, was its pass a check of the main list.
- **show_me's three-way split** cost extra looking: helper 3 re-read six sheets outside its range (60 vision calls), and the main agent re-looked at seven pages after the helpers (70 more). It also cost four wasted helper calls, and it lost two cross-sheet problems.
- **GPT-5.4** spent 16 text searches, several on values that are not there, and did the least looking.
- No run hit an error, a refused call or the step cap.

## Follow-ups

Proposed: defects in the scorer and the record that hold on any model.

- **F1, scorer: page notes count as delivered.** `checks.delivered_text` reads every file `run.json` lists, and the list includes `.scratch/page_readings.json` and `.scratch/reads.json` (the vision tools' kept readings). Leave `.scratch/` out of what a check reads as delivered. In this group alone it gives 5 false passes (show_me 2, GPT-5.4 3).
- **F2, scorer: a heading carries across appended files.** `run_check` appends the files' text to the answer. `_sections()` treats an all-bold line as a heading, so the answer's last bold bullet heads everything appended after it. I reproduced this: a neutral "7/A-501" in the notes passes "under the heading that holds the rest" when it follows that bullet, and fails when it does not. Start a fresh section at each file boundary.
- **F3, scorer: a whole number matches inside a decimal.** `_num("109")` matches "109" in "EL. 109.30", and `_num("204")` matches the bounding-box coordinate "204.9". The lookahead refuses a following digit but not a following ".digit". Its docstring promises "1.5" is not in "1.55", but a whole-number code is matched inside a decimal. This passes `door-not-scheduled` on GPT-5.4's delivered workbook, where "109" appears only as part of an elevation.
- **F4, record: the hand-back lacks the two note files the scorer read.** `run.json` lists `.scratch/page_readings.json` and `.scratch/reads.json`, the verdicts were computed partly from them, and they are not in the hand-back. I could rebuild what they held only from the page readings in `activity.jsonl`. Ship them, or stop scoring them (F1).

Noted, not proposed: model behaviour, prompt choices or low impact.

- Three times a planted problem was stated in the main model's reasoning and left out of the written log (method 7/A-501 #878 → #882; all_three bay #897 → #905; baseline bay #413).
- A page-range split among helpers separates the two halves of cross-sheet problems (show_me).
- The `show_me` option was offered and never chosen: 0 of 60 looks in show_me, 0 of 56 in all_three. `activity.jsonl`'s `tools_offered_sha` hashes tool names only, so the log alone cannot show the argument was offered. `run.json`'s fingerprint can (the tool sha differs: baseline 14030f98, show_me 88ad49e3).
- End-of-run "independent" helpers re-read the whole set and added nothing planted.
- Per-page prompts written from guessed sheet contents (baseline).
- A tile that cuts a dimension makes the reading hedge ("cannot be verified here", all_three #241). It weakened the evidence but did not block it.
- GPT-5.4 read five sheets as text only and never looked at them. The page's prompt asks it to look at every drawing page.
- GPT-5.4 records no reasoning text, so its omissions cannot be explained from the record. This is a model property, not a logging gap.
- The "NO." revision-table header and a "Not visible" table column match the absent-word list. This matters only while page notes are scored (F1).
