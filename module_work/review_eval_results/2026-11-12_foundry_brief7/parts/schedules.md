# Schedules group: coord-schedules, coord-scan-schedules, coord-long-schedules

Part of the Foundry brief 7 review (2026-11-12). 18 runs: GPT-5.6 Sol in five arms
(baseline, show_me, method, sheet_tools, all_three) and GPT-5.4 in baseline, on the made-up
"North Yard Annex" drawing set.

- Records: `module_work/field_feedback/2026-11-12_foundry_brief7/raw/b7_coord_sol/runs/<arm>/<task>/`
  and `.../raw/b7_coord_gpt54/runs/baseline/<task>/`.
- "Record N" means line N of that run's `activity.jsonl`, counting the first line as 0.
- "PDF page" counts from 1, as a PDF viewer does. The tools' `page` argument counts from 0.
  Sheet numbers are given wherever it matters.

**What I read.** For every run I read the full activity record: each agent turn, every tool call
and its result, each helper's brief and return, and the vision calls made inside the page tools.
I also read the chat answer, the check details, every delivered Word file (as extracted text) and
the helper files that were kept. A script compared every mark reported by every page reading
(whole-sheet view, close-up tiles, zooms) with the fixture's truth for that sheet. I then read
each disagreement by hand.

**How the page tool reads a sheet.** It makes one whole-sheet picture, 2048 px across a
34 x 22 in sheet (about 60 px per inch), plus a grid of overlapping close-up pictures called
"tiles": 3x3 by default on a text-layer set, 4x4 by default on a scan, or 2x2/3x3/4x4 when the
agent asks. Each tile is also about 2048 px wide. Each picture is one separate model call, which
is where most of the "model calls" counted below come from (§7).

---

## 1. Headline

1. **Sol found every planted problem in all 15 of its runs, with no false alarms**, both in chat
   and in the Word file it delivered. Not one mark was misread in any Sol page reading. Every
   "extra" mark the script flagged was a sentence saying the mark is absent ("204 is not
   present", "no 407 row", "FCU-3-4: not present in this view").
2. **GPT-5.4 got the text set right. Its pass on the scanned set was luck, and on the long set it
   missed one of seven problems and added a false alarm.**
   - *Scanned set.* It never looked at the second-floor plan A-102. It reported all six
     second-floor doors (201-206) as "scheduled but not on the checked floor plan", which is five
     false alarms, and said it found no second-floor plan. The check passed because "204" sits
     near "not". Cause: the contact sheet labels each thumbnail with its 0-based tool page
     number. The vision call that read the contact sheet called those labels "PDF page 3",
     "PDF page 4". The model then asked for PDF pages 3 and 4, which are C-101 and A-101 (§5.6,
     proposal P2).
   - *Long set.* **It missed F4 because it never opened the foundation plan S-101 or the footing
     schedule S-501** (PDF pages 22 and 28). It also reported door "409" as scheduled on A-602
     and missing from A-104. No door 409 exists. The number started in the model's own prompt to
     the page tool ("the scheduled fourth-floor door range 401-409"). The whole-sheet view of
     A-104 then read door 408 as "409" (§6.6).
3. **The arms made no difference to any answer** on any of the three tasks. They differ only in
   time and tokens (§3). **The show_me argument was offered but never used.** The run-level tool
   fingerprint (`fingerprint.tools.sha` in run.json) differs between baseline (`14030f98e6b2db87`)
   and show_me (`88ad49e346764e59`) on all three tasks, with the same system prompt, so the tool
   definitions did differ. all_three (`dbdd3e8b5eace2b2`) differs from sheet_tools in the same
   way. No call in any of the 70 Sol runs passes show_me. So show_me against baseline here
   measures run-to-run chance, not the switch. The per-turn `tools_offered_sha` hashes tool
   names only and cannot show this (§9).
4. **sheet_index was right on both text-layer sets.** For the 15- and 40-sheet sets, every sheet
   number, title and revision matches the fixture. **On the scanned set it returned nothing**
   (`n_indexed: 0`, "no text layer on this sheet: read its title block by looking"). On the long
   set, the 5 calls (sheet_tools) and 6 calls (all_three) are one by the main agent plus one by
   each helper when it started. Each call returned the same 4,073-character index in under half
   a second.
5. **Helpers on the long set did not change any answer.** show_me ran 4, sheet_tools 4 and
   all_three 5: one per schedule family, plus a fifth in all_three for A-401 and the ceiling
   plans. All returned correct lists with sheet citations. They cost time: only two run at once,
   so later helpers waited 107-284 s. Total tokens also went up (§3, §6).

## 2. What the checks test, and what they miss

- **coord-schedules** has three "stated together" checks and two "does not contain" checks. Each
  "stated together" check looks for 109, 204 or F4 within 160 characters of an absence word. The
  scanned and long tasks add an information-only "pages read" line. Every check reads the chat
  answer only, not the Word file.
- **Were the verdicts right?** All but two:
  - GPT-5.4 scanned set: it passed and should not have (five false alarms, never looked at A-102).
  - GPT-5.4 long set: it failed F4, which is correct, but nothing caught the door 409 false alarm.
- **What the checks do not test:**
  - False alarms, meaning a mismatch reported for a mark that is not planted. This happened twice
    in these runs, both GPT-5.4, and both went unpenalised.
  - Direction. "204 is on the plan but not scheduled" would pass. No run made this error here.
  - The delivered Word file. Two Sol files state wrong totals (§6.1, §6.2); the chat answers do
    not.
- **Passes "under the heading that holds the rest"** were sheet_tools long LP-4B and all_three
  long 407, F4, FCU-3-4 and LP-4B. I checked each one. The mark sits in a table under "On a plan
  but not scheduled", or in the column headed "On plan but **not scheduled**". These are correct
  passes.

## 3. Side by side

Columns: "yes" means stated in the right direction in the chat answer. Calls means all model
calls, with the agent's own turns in brackets; helper turns are added after a plus sign. Input
tokens are summed over all calls.

### coord-schedules (15 sheets, text layer)

Planted: door 109 is on a plan but not scheduled; door 204 is scheduled but not on a plan;
footing F4 is on a plan but not scheduled.

| Run | 109 | 204 | F4 | False alarms | File | Calls (turns) | Input tokens | Seconds |
|---|---|---|---|---|---|---|---|---|
| Sol baseline | yes | yes | yes | none | Word | 72 (12) | 625k | 141 |
| Sol show_me | yes | yes | yes | none | Word | 70 (11) | 564k | 111 |
| Sol method | yes | yes | yes | none | Word | 72 (12) | 659k | 122 |
| Sol sheet_tools | yes | yes | yes | none | Word | 81 (12) | 629k | 160 |
| Sol all_three | yes | yes | yes | none | Word | 80 (11) | 649k | 129 |
| GPT-5.4 | yes | yes | yes | none | none | 68 (12) | 538k | 148 |

### coord-scan-schedules (same 15 sheets, scanned at 150 dpi grey, no text layer)

| Run | 109 | 204 | F4 | False alarms | Sheets looked at | Tiles asked | Calls (turns) | Input tokens | Seconds |
|---|---|---|---|---|---|---|---|---|---|
| Sol baseline | yes | yes | yes | none | 15 of 15 | 2x2 | 107 (11) | 739k | 188 |
| Sol show_me | yes | yes | yes | none | 15 of 15 | 4x4 / 3x3 | 222 (13) | 1,396k | 446 |
| Sol method | yes | yes | yes | none | 9 of 15 | 3x3 | 117 (15) | 1,022k | 224 |
| Sol sheet_tools | yes | yes | yes | none | 6 of 15 | default 4x4 | 126 (13) | 909k | 380 |
| Sol all_three | yes | yes | yes | none | 6 of 15 | 4x4 | 129 (12) | 877k | 390 |
| GPT-5.4 | yes | "yes", by luck | yes | **201, 202, 203, 205, 206** | 9 of 15, **not A-102** | auto (4x4) | 178 (15) | 1,320k | 476 |

### coord-long-schedules (40 sheets, text layer)

Seven planted: doors 109 and 407 are on a plan but not scheduled; doors 204 and 509 are
scheduled but not on a plan; F4, FCU-3-4 and LP-4B are on a plan but not scheduled.

| Run | 109 | 407 | 204 | 509 | F4 | FCU-3-4 | LP-4B | False alarms | Helpers | Calls (turns) | Input tokens | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sol baseline | yes | yes | yes | yes | yes | yes | yes | none | 0 | 245 (12) | 1,555k | 235 |
| Sol show_me | yes | yes | yes | yes | yes | yes | yes | none | 4 | 364 (12 + 23) | 2,119k | 379 |
| Sol method | yes | yes | yes | yes | yes | yes | yes | none | 0 | 250 (17) | 2,180k | 476 |
| Sol sheet_tools | yes | yes | yes | yes | yes | yes | yes | none | 4 | 336 (11 + 23) | 1,921k | 446 |
| Sol all_three | yes | yes | yes | yes | yes | yes | yes | none | 5 | 418 (13 + 26) | 2,358k | 651 |
| GPT-5.4 | yes | yes | yes | yes | **no** | yes | yes | **409** | 0 | 221 (15) | 1,471k | 406 |

Sol's cheapest arm on all three tasks was plain baseline or show_me. show_me never used its
argument, so its gap from baseline is run-to-run chance.
Helpers moved tokens out of the main agent but did not save tokens in total. In method, the main
agent alone spent 1.33M input tokens over 17 turns, because every page reading stays in its
context. In the helper arms the main agent spent 355k-408k and the helpers 426k-613k.

---

## 4. The five group questions

### 4.1 Sheets skipped: sound or lucky?

Sound in every Sol run. Lucky or wrong in two GPT-5.4 runs. The tables below show which sheets
carry the marks; every other sheet carries none.

| Set | Sheets with tags or schedules (PDF page) |
|---|---|
| 15 | A-101 (4), A-102 (5), A-401 (8, door 109 again), A-601 (10), S-101 (12), S-501 (15) |
| 40 | A-101 to A-105 (4-8), A-401 (16), A-601 (19), A-602 (20), S-101 (22), S-501 (28), M-101 to M-105 (29-33), M-601 (34), E-101 to E-105 (35-39), E-601 (40) |

- **Text set.** No "pages read" line is recorded. All five Sol arms looked at A-101, A-102,
  A-601, S-101 and S-501 by vision. sheet_tools and all_three also looked at A-401, because the
  sheet index listed it. GPT-5.4 looked at the same five.
- **Scanned set.**
  - baseline and show_me looked at all 15.
  - method skipped C-101, A-301, A-401, S-001, S-201 and S-202. Only A-401 carries a tag, and
    that is the repeat of door 109, so the answer could not change. Sound.
  - sheet_tools and all_three looked at only A-101, A-102, A-201, A-601, S-101 and S-501. They
    chose those sheets from the thumbnails, because sheet_index returned nothing on the scan.
    The only tagged sheet they skipped is A-401 (the 109 repeat). Sound for the answer, though
    the page choice was a guess. sheet_tools first sent PDF page 6 (A-201) as "the door schedule"
    and asked A-601 for footings (record 172), then corrected itself.
  - GPT-5.4 never looked at A-102. **Not sound** (§5.6).
- **Long set.**
  - The four Sol arms other than all_three each read the 22 tag and schedule sheets plus the
    cover G-001 (its sheet index read as text). They did not read G-002, C-101, the five ceiling
    plans, A-201, A-301, A-501, A-502, S-001 or the five framing plans, and none of those carries
    a mark. Sound, and chosen from the sheet index rather than by luck.
  - all_three also had its fifth helper look at C-101 and the ceiling plans to make sure they had
    no door tags. They have none.
  - GPT-5.4 did not read S-101 (PDF 22) or S-501 (PDF 28). Those two sheets hold the footing
    marks, which is exactly the F4 miss (§6.6).

### 4.2 Scanned set: how tags and schedules were read, picture sizes, misreads

- **Scale.** The scan is about 5100 x 3300 px (34 x 22 in at 150 dpi). The page tool's pictures
  are about 2048 px wide, so the detail each one shows is roughly:

  | Picture | Pixels per inch (about) | Compared with the 150 dpi scan |
  |---|---|---|
  | Whole sheet | 60 | well below |
  | 2x2 tile | 115 | below |
  | 3x3 tile | 170 | slightly enlarged |
  | 4x4 tile | 230 | enlarged; no new detail, but bigger lettering |

  Zooms (render_region) were about 2048 px wide over boxes of 400-1300 PDF points (6-18 in), so
  also enlarged from the scan.
- **2x2 was too coarse for the schedule.** Sol baseline read every sheet with 2x2 tiles. On the
  door schedule A-601 those tiles caught only 4 of the 14 door numbers (record 172). A zoom on
  the table then read all 14 (record 263). Every 3x3 and 4x4 reading of A-601 caught all 14.
- **Misreads:** none. In all six scan runs, every door number and footing mark that was reported
  as seen matches the fixture. The only "extra" marks are absence statements ("F4 is not
  listed", "204 does not appear", "no 109 row").
- **How schedules were read.** By vision only, since there is no text. All runs read the
  schedule from tiles and then zoomed on the table. GPT-5.4 zoomed on PDF pages 4, 10, 12 and 15
  (records 366-388).

### 4.3 Each answer's mismatch list against the planted problems

- **Sol, all 15 runs:** exactly the planted problems, in the right direction, with nothing extra.
  The locations they gave check out against the fixture:
  - 109 is between the Lobby and the Core on A-101, and repeated on the A-401 enlarged core plan.
  - F4 is at grid C/2.
  - Footing counts are F1 x4, F2 x6, F3 x1 and F4 x1.
  - FCU-3-4 is in bay A-B/2-3 on M-103.
  - LP-4B is in the fourth-floor core beside LP-4A on E-104.
  - A-602 skips from 406 to 408.
- **GPT-5.4 text:** exact.
- **GPT-5.4 scanned:** found 109 and F4. Its "scheduled but not on plan" list is 201-206, because
  it never looked at A-102. That is five false alarms, and 204 is right only by accident.
- **GPT-5.4 long:** six of seven. Missing F4. False alarm 409.
- **Why GPT-5.4 missed F4 on the long set:**
  - Its contact-sheet call (record 31) listed page 27 (S-501) as a schedule sheet and page 21
    (S-101) as a plan.
  - It then read the text of pages "13-19,33,39" (record 34), leaving out page 27.
  - It never searched for "SCHEDULE" and never read the sheet index on G-001.
  - It went on to look at the door, mechanical and electrical sheets only.
  - Its "Coverage" paragraph lists the door, mechanical and panel schedules and says nothing
    about footings. The miss is silent: the answer does not say the footing schedule went
    unchecked.

### 4.4 sheet_index: calls, returns, use

| Run | Calls | Returned | Used for |
|---|---|---|---|
| text set, sheet_tools / all_three | 1 each | 15 sheets, all correct | choosing pages; both then also looked at A-401 and mentioned the 109 repeat |
| scanned set, sheet_tools / all_three | 1 each (records 14 / 15) | `n_indexed: 0`, "no text layer" | nothing; both fell back to thumbnails |
| long set, sheet_tools | 5: main agent record 12; helpers records 176, 186, 431, 589 | 40 sheets, all correct (title and revision match the fixture, including Rev 1 on G-001, A-401, A-601) | main agent's plan of five schedule groups; each helper's own page lookup |
| long set, all_three | 6: main agent record 12; helpers records 65, 82, 353, 513, 819 | the same index | the same; all_three's revision note (A-401 and A-601 at Rev 1, the floor plans and A-602 at Rev 0) comes from it and is correct |

Why 5 and 6 calls: one main-agent call, then one call by each helper (4 and 5 helpers). A
helper starts with a fresh context, so it opens the document and indexes it again. The repeats
cost under half a second and about 1k tokens each. Not waste worth fixing.

### 4.5 Helpers on the long set: what they were asked, what they returned

- **show_me** (helpers at records 48, 50, 494, 715): footing, doors, mechanical, electrical.
  - Each brief named the schedule sheet and the plan sheets, with PDF page numbers, and asked
    for both directions plus citations.
  - Returns (records 490, 711, 763, 905) were all correct. The door helper also flagged that its
    zooms had called 308/408/508 "room numbers", and kept them as door tags after weighing the
    tiled view.
  - The mechanical helper was queued 132 s and the electrical helper 210 s (records 493, 714).
  - Three returns were kept as files because they were much shorter than what the helper had
    read: `files/kept/helper_analyze-*`.
- **sheet_tools** (helpers at records 158, 159, 418, 581): doors, footing, mechanical, electrical.
  - Before starting them, the main agent had already read all five schedules (text plus 3x3
    tiles, records 46-50). Each brief therefore carried the schedule list and the gap, for
    example "A-602: 401-406, 408, 501-509 (no 407)" and "FCU-3-1 to FCU-3-3 (no FCU-3-4)". The
    helpers confirmed against the plans.
  - Returns (records 414, 579, 808, 842) were correct, with grid locations.
  - One wrong side statement: the electrical helper said LP-4B was "omitted by the plan's text
    extraction". Its search (record 594) looked only for the ten scheduled names, so LP-4B could
    not appear. The plan's text does hold LP-4B; method's search found it (record 596). This did
    not reach the answer.
- **all_three** (helpers at records 42, 44, 340, 506, 808): doors on A-101 to A-105 with the
  schedules, footing, electrical, mechanical, and a fifth door supplement for A-401, C-101 and
  the five ceiling plans.
  - Returns (records 338, 502, 800, 805, 1045) were correct.
  - The electrical helper also checked whether circuit numbers could be compared. It answered
    correctly that there are none.
  - The fifth helper added about 100 s and found nothing new, which is as expected.

---

## 5. Runs: coord-schedules and coord-scan-schedules

### 5.1 coord-schedules, Sol baseline
- Path: open, page map and thumbnails, two searches, contact sheet, text of the schedule pages,
  whole-sheet plus 3x3 readings of A-101, S-101, A-601, A-102 and S-501 (records 79-147), four
  zooms (161-167), Word file.
- Delivered all three problems, correct, in chat and in the Word file.
- One wasted call (record 171). The text tool's "next" note said to continue with pages
  "9,11,14" from line 147. The model sent pages "11,14" from line 147 and got "start_line 147 is
  past the end of page 11 (83 lines)", then re-read.
- 72 calls: 12 agent turns, 45 tiles, 5 whole-sheet views, 5 start-up probes, 4 zooms and 1
  contact-sheet reading.

### 5.2 coord-schedules, Sol show_me
- Same path as baseline: readings at records 79-149, three zooms at 165-169.
- Correct. The show_me argument was offered (the run's tool fingerprint differs from
  baseline's) but never used.

### 5.3 coord-schedules, Sol method
- Readings at records 73-149, four zooms at 163-169. It used the text continuation correctly.
- Correct.

### 5.4 coord-schedules, Sol sheet_tools and all_three
- One sheet_index call each, correct.
- Both also looked at A-401 (records 121 and 189) and noted 109 repeated there.
- Correct, Word files delivered.

### 5.5 coord-schedules, GPT-5.4
- Five whole-sheet plus tile readings with tiles "auto" (records 83-169). No zooms and no file.
  The record holds no reasoning text.
- Correct.

### 5.6 coord-scan-schedules, all runs
- **Sol baseline:** read all 15 sheets with 2x2 tiles, then made six zooms (records 253-263).
  The A-601 zoom rescued the schedule the 2x2 tiles could not read. Correct.
- **Sol show_me:** 4x4 tiles on the target sheets (records 125-347), then 3x3 or 4x4 on every
  remaining sheet. That is 184 tile calls and 446 s for the same answer. Correct.
- **Sol method:** 3x3 tiles on S-101, A-102, S-501, A-101 and A-601 (records 94-237), plus the
  cover, A-501 and G-002, then zooms. Correct.
- **Sol sheet_tools:**
  - sheet_index returned nothing (record 14).
  - It guessed pages from the thumbnails and read A-201 as "the door schedule", then asked A-601
    for footings (record 172, which read 1 of 14 door numbers because the question was about
    footings).
  - It found S-501 at PDF page 15. That one reading took 204 s, mostly one wait for a vision
    call.
  - Zooms at records 258-284. Correct.
- **Sol all_three:** sheet_index returned nothing (record 15). 4x4 tiles on the five target
  sheets (records 88-256), then nine zooms. Correct.
- **GPT-5.4:**
  - The contact-sheet reading (record 31) says "PDF page 3 - `3 drawing_sheet`", "PDF page 4 -
    `4 drawing_sheet`", and so on. The labels are 0-based tool pages.
  - The model asked for pdf_page 3, 4, 9, 11 and 14 (records 34-38), which are C-101, A-101,
    A-501, S-001 and S-202, one off from what it meant.
  - Reading the title blocks, it corrected the schedule and foundation pages (pdf_page 10, 13,
    15, 12; records 216-219).
  - It never corrected the floor-plan pair. One of its two "floor plans" really was a floor plan
    (A-101), so nothing looked wrong. It never asked for PDF page 5 (A-102).
  - Answer: 109 and F4 right; 201-206 "scheduled but not on the checked floor plan"; "I did not
    find a second floor architectural plan sheet". Check passed. No file.

---

## 6. Runs: coord-long-schedules

### 6.1 Sol baseline
- **Path:**
  - Open, page map, thumbnails, searches for "SCHEDULE" and "PLAN", contact sheet, text of the
    cover and the five schedule sheets.
  - Whole-sheet plus 3x3 readings of the schedules and A-101 to A-105 (records 50-67), then of
    S-101, A-401, M-101 to M-105 and E-101 to E-105 (records 272-283).
  - Seven confirming zooms (records 538-544): 109 seen (553), 204 absent between 203 and 205
    (559), 407 confirmed (565), 509 absent (555), F4 beside a footing outline (557), FCU-3-4
    under its unit symbol (563), LP-4B (561).
- **Delivered:** all seven, correct, in chat and Word (record 572).
- **Wrong statement in the Word file:** "The cross-check found **four** plan marks that are not
  scheduled". Its own table lists five (109, 407, F4, FCU-3-4, LP-4B). The chat answer has no
  count.
- No helpers. Cheapest Sol run on this task.

### 6.2 Sol show_me
- **Path:** orientation as in baseline, then four helpers (§4.5). The main agent made no
  plan-sheet readings itself.
- **Delivered:** all seven, correct.
- **Wrong statement in the Word file:** "The review identified **eight** mismatched marks ...
  **Six** marks appear on plans but are not scheduled". It then names five, and notes that 109
  is "one unique mark". The true totals are seven and five. The chat answer is right.
- The helpers waited 132 s and 210 s in the queue.

### 6.3 Sol method
- **Path:**
  - Schedules with tiles "auto" (records 50-54).
  - Door plans and A-401 (162-167).
  - S-101 and M-101 to M-105 (297-302).
  - E-101 to E-105 (431-435).
  - Two text reads continued correctly (436, 545).
  - Seven zooms (549-556).
  - A final text search for all seven marks (records 583-589), each found on its sheet.
  - Word file (603).
- **Delivered:** all seven, correct, as a numbered comment log in the Word file.
- **Time:** one whole-sheet reading of A-102 took 202 s (record 294), about 150 s of it waiting
  on one vision call. That is a third of the run's 476 s.
- **Tokens:** the main agent's context grew to 125k by the end, giving the 1.33M agent-turn
  input total.

### 6.4 Sol sheet_tools
- **Path:**
  - sheet_index (record 12) and text of the cover and schedules (23).
  - 3x3 readings of the five schedules (46-50).
  - Four helpers whose briefs carried the schedule lists (§4.5).
  - Word file (849).
- **Delivered:** all seven, correct, with grid locations in the chat answer. The Word file total
  ("seven unique mismatched marks") is right.
- **Wrong statement:** only the helper's text-extraction remark (§4.5), which did not reach the
  answer.

### 6.5 Sol all_three
- **Path:** sheet_index, contact sheet, five helpers (§4.5), Word file twice. The first file
  (record 1056) said "seven unique plan marks". The model caught the inconsistency in its next
  turn ("the summary says seven unique plan marks, but the table shows only five") and rewrote
  the file (record 1060).
- **Delivered:** all seven, correct.
- **Extra content:** a revision note that A-401 and A-601 are Rev 1 (2026-03-02) while the floor
  plans and A-602 are Rev 0. This matches the fixture.
- Slowest Sol run at 651 s. The fifth helper (A-401, C-101 and the ceiling plans) found nothing
  new.

### 6.6 GPT-5.4
- **Path:**
  - Page map, thumbnails, contact sheet (record 31).
  - Text of pages 13-19, 33 and 39 (record 34; no page 27).
  - Tile readings of A-601, A-602, M-601 and E-601 (48-51).
  - Door plans A-101 to A-103 and A-401 (142-145).
  - M-101, M-102, E-101 and E-102 (232-235).
  - A-104, A-105, M-103 to M-105 and E-103 to E-105 (322-330).
  - No zooms, no file.
- **Delivered:** 109, 407, 204, 509, FCU-3-4 and LP-4B, correct. F4 never seen, because S-101
  and S-501 were never opened.
- **Wrong statement:** "Door 409 is listed in the door schedule on A-602 ... but I did not find
  it on the Fourth Floor Plan." The same answer lists A-602 as "401, 402, 403, 404, 405, 406,
  408, 501 ... 509", so it contradicts itself. How the 409 arose:
  - No tool result mentions a door 409 before the model's own prompt for A-104 (record 322),
    which asked for "tags outside the scheduled fourth-floor door range 401-409".
  - In the A-104 result (record 421), the whole-sheet view (lettering about 6 px tall) listed
    "409" at the spot where door 408 is and said "408 is not visible". The middle tile read
    "408" there correctly.
  - The tool's merged list of found things shows "408, seen in page, r2c2". It does not report
    that the whole-sheet view read that spot differently.
  - The final answer took 401-408 for the plan and kept its own assumed 401-409 for the schedule.
- **Check verdicts:** F4 failed, which is correct. The 409 false alarm was not penalised.

---

## 7. What the extra model calls are

- Every Sol run's call count is mostly vision calls made inside the page tools, recorded as
  model calls of the agent that used the tool. Each run has:
  - 5 start-up probe calls.
  - One call per whole-sheet view.
  - One per tile.
  - One per zoom.
  - One per contact-sheet reading.
- Example, Sol baseline long, 245 calls: 12 agent turns, 198 tiles, 22 whole-sheet views, 7
  zooms, 5 probes, 1 contact sheet.
- The agent turns are few but carry most of the input tokens, because each turn resends the
  whole conversation. In Sol baseline long they took 709k of 1,555k input tokens. On the text
  set they took about 400k of 600k.

---

## 8. Proposals (would hold on any model)

**P1 (scorer): a false-alarm check for the schedule tasks.**
- Fail an answer that reports a mismatch for a mark that is not planted: a door number, footing,
  FCU or panel name stated with an absence word, or listed under a mismatch heading, that is not
  in the planted list.
- The fixture already knows every real mark per sheet (`coordination_fixture._project`).
- Evidence: GPT-5.4 scanned (201, 202, 203, 205, 206) and GPT-5.4 long (409). Both passed the
  "204" or overall checks. The first should have failed outright.

**P2 (tool): thumbnail labels on the contact sheet should not be bare 0-based numbers.**
- The labels read "`3 drawing_sheet`". The page tools take both `page` (from 0) and `pdf_page`
  (from 1), and the vision call that reads the contact sheet has to guess which a bare "3" is.
- Sol's contact-sheet readings guessed right ("zero-based contact-sheet page labels"). GPT-5.4's
  guessed wrong ("PDF page 3"), and the agent read the wrong sheets.
- Suggested fix: print the label as the PDF page (or both, e.g. "p3 / PDF 4"), say so in the
  result's legend, and pass the legend into the contact-sheet vision prompt.
- Evidence: GPT-5.4 scanned records 31, 34-38. This changed a saved answer: five false alarms
  and "no second-floor plan found".

## 9. Noted, not proposed

- **The per-turn tool hash covers names only.** `tools_offered_sha` (`activity_log.py`,
  `_turn_facts`) reads `601349d305a5217f` for baseline, show_me and method alike, so a turn's
  record cannot show that an argument such as show_me was offered. The run-level
  `fingerprint.tools.sha` in run.json does show it. Hashing the arguments too in the per-turn
  value would let the activity record answer this on its own.
- **Direction-blind checks.** "Stated together" passes whichever way the mismatch is stated. No
  run here got a direction wrong.
- **The Word file is not scored.** Two Sol files carry wrong totals (baseline long "four", show_me
  long "eight ... six") that the chat answers do not. The checks would not catch a count anyway.
- **The page tool's merged "found" list hides disagreements.** When the whole-sheet view and a
  tile read the same spot differently (GPT-5.4 long, record 421: "409" vs "408"), the list keeps
  one reading and says it was "seen in page" as well. `found_note` only reports things found in
  tiles alone. This did not cause the 409 here (the 409 came from the prompt), but a model could
  pick up the wrong reading from it.
- **A search field that names pages it just found hits on.** `search_document` returns
  `pages_not_searchable_as_text` listing every drawing sheet, even when the hits come from those
  same sheets. Example: method long records 590-596, with hits on pages 3, 6, 15, 18, 19, 21, 30
  and 37. The field is meant as "a miss here is not proof of absence". No run was misled.
- **Text continuation misuse.** Sol baseline text set, record 171: one wasted call.
- **Single long vision waits.** method long A-102: 202 s, record 294. sheet_tools scanned S-501:
  204 s. Time only.
- **GPT-5.4 told the vision tool what to expect** ("range 401-409"), and the whole-sheet view
  obliged. This is model behaviour. It is not a reason to tune a prompt.
- **sheet_index gives nothing on a scan.** The scanned-set sheet_tools runs then guessed pages
  from thumbnails (A-201 taken for the door schedule). This cost time but never changed an
  answer.
- **Helpers repeat sheet_index.** Expected, since each helper starts fresh, and cheap.
- **The "pages read" information line** lists PDF page numbers only. Naming the sheets it missed
  (for GPT-5.4 long: "S-101, S-501") would make a miss like F4 readable at a glance. The F4 check
  already catches it.
