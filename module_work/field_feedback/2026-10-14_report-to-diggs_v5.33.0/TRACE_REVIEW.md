# Field session 2026-10-14: a report's explorations to DIGGS — the full read

## What ran

**The ask.** One turn on the geotech page (app 5.33.0, Funhouse, the Prompter
API key route, GPT-5.6 Sol, recursion limit 200): build a DIGGS file of the new
and historic explorations in Volume 1 of a geotechnical report, with an x-y
grid read off the investigation plan (Figure 2, PDF page 24) where no
coordinates are given. The report is a 242-page PDF: 2026 logs (vector), the
logs of three earlier investigations (2019 vector; 2024 and 1987 scanned),
test-pit photos and laboratory sheets. The lead's first read is
[FINDINGS.md](FINDINGS.md).

**What came back.** 51 minutes (3,065 s), 997 model calls, 349 tool calls,
then a rate-limit refusal ended the turn. One file: a DIGGS file for test pit
TP-1. No answer.

**The record.** `raw/conversation/activity.jsonl` (2,696 records; "#N" below is
the 0-based line number, "t" is seconds from the turn's start), plus
`messages.json`, `transcript.jsonl`, `trace.jsonl`, `meta.json`, and the PDF
and the DIGGS file in `raw/files/`. Everything that names the report, its
firms or its people, the extraction tables and the scripts are in the
git-ignored `raw/notes/` (`NOTES.md`, `grid_table.md`, `scripts/`).

**Method.**
- **Indexing.** A read-only script indexed every record. Helper records carry
  only `agent: general-purpose`; their `parent_run_id` points at graph nodes
  the log never records. So each helper was rebuilt by chaining its calls:
  model call with n messages, k tool calls, k results, next call with
  n + 1 + k messages.
  - The rebuilt chains end exactly when their `task` calls return.
  - Of the 916 one-message (vision) calls, 309 could be pinned to one helper.
    The other 607 ran while several helpers had vision tools in flight (§6).
- **What was read.** Every primary step and every helper step, their briefs
  and their answers.
- **What was checked.** Six explorations against the rendered PDF pages, and
  the grid against the vector symbols on page 24 (PyMuPDF).

The helpers, named here by what they were asked to do:

| | brief (abridged) | started | ended | returned |
|---|---|---|---|---|
| H1 | ALL investigation data, every log and lab page, and Figure 2 | t 278 | t 618 | an apology, 366 chars (#370) |
| H2 | a local metric grid from Figure 2 | t 278 | t 811 | a transform and 54 points, 6,558 chars (#394) |
| H3 | the 2026 borings (10) | t 823 | t 856 | **rate limit** (#570) |
| H4 | test pit TP-1 | t 823 | t 1,751 | wrote `TP-1_2026.diggs.xml` itself; 3,646 chars (#1326) |
| H5 | the 2019 borings (4) | t 823 | t 2,386 | JSON, 50,529 chars (#2044) |
| H6 | the 1987 borings (4) and two resistivity soundings | t 823 | t 2,074 | JSON, 10,918 chars (#1676) |
| H7 | the 2024 light-pole borings | t 823 | t 3,006 | `write_diggs` refused 17 rows; an apology, 268 chars (#2610) |
| H8 | the 2026 borings again ("omit n on refusal; do not write the file") | t 3,015 | t 3,055 | **rate limit** (#2690) |

Then the primary's own next call was refused (#2694), and the turn ended.

---

## In short

1. **The turn did the reading and lost it.**
   - What helpers had in hand when they stopped:
     - H1: 12 log grids;
     - H3 and H8: the 2026 log grids;
     - H7: 21 borings' worth of data.
   - What the primary got back from those four: two apologies and two error
     strings.
   - A helper returns one message. On failure or give-up, everything in its
     context goes with it.
   - Nothing reached the conversation either (TA-19).
   - What survives: one DIGGS file for a 1.5 m test pit, with no strata and a
     made-up location (§2).
2. **Where the data was finished, the reading was good.**
   - Six explorations were checked against the pages: 401 of 406 values agree.
   - Of the 5 disagreements, two matter: a ground elevation 1 m off on a 1987
     scan, and a moisture value.
   - The losses are elsewhere:
     - **Omissions the tool shape caused:** 191 moisture values, recovery
       lengths, drive increments.
     - **Scope:** 18 of the report's 2024 logs were left out, and no 2026
       boring was ever extracted.
     - **Orchestration and failure handling** (items 1, 4 and 5).
3. **The grid is mirrored, and its points are off.**
   - The helper wrote its own rotation from the north arrow. East points
     west, so every x has the wrong sign.
   - The points are labels read from vision prose in mixed frames. Against
     the PDF's own vector symbols they are a median of 4.7 m off; the helper
     claimed ±2 m.
   - The vector symbols give every centre exactly, one circle size per
     program.
   - The grid never reached a file: TP-1 was written at 0, 0, 0.
4. **The rate limit killed the agents' own calls; the vision calls survived
   theirs.**
   - 8 refusals:
     - 5 hit vision side calls. All were retried by the vision engine's own
       ladder and read.
     - 3 hit agent calls (helper H3, helper H8, the primary). Each ended its
       agent, since those calls get only the client's quick retries (about
       10–12 s).
   - **The last refusals came with one helper running.** The minute before
     them carried at least ~295 k input tokens of tool results, so a cap on
     concurrent helpers alone would not have saved the turn.
5. **Parallel helpers cost more than they saved.**
   - Five helpers shared 8 vision slots (saturated in 26 of the 51
     minutes).
   - They also shared one lock on the one open document. Raster `log_grid`
     calls held it for 14–20 s each, and other tools queued behind them for
     up to 10 minutes.
   - The 2026 log grids ran three times. The shared lab pages were read by up
     to three helpers.
   - When H3 failed at t 856, the primary could not react until the slowest
     sibling returned at t 3,006: 36 minutes later.
6. **A brief set the scope.**
   - The primary listed the plan's 2024 IDs "expected on Figure 2".
   - H7 read all 18 of the other 2024 logs page by page (about 180 side
     calls), then dropped them as "not the requested run".
   - The user asked for the report's data. Scope has to come from pages and
     sections, not from a list in a brief.
7. **The refusal rows show a tool description that invites the error.**
   - "`n` (ONLY if printed)" plus a printed N of ">50" gave `n: ">50"`, and
     the whole file was refused.
   - Three helpers handled refusal three ways: right, refused, and dropped.
   - The helper that was refused gave up 8 s later and returned none of its
     data (RI-26 fixed the message, not the loss).
8. **The record cannot answer the cost or the cause of the refusals.**
   - No usage at all: `usage` is null on every call, likely because streaming
     usage is off on the Prompter client (§6).
   - No 429 body or headers, no retry count, no engine named.
   - Helper events carry no helper id.
   - The 2019 answer is cut at 32,000 of 50,529 characters.
   - No queue times.
9. **This was a pipeline job done by a chat loop.**
   - `report_ingest` reads every log through work items made from the page
     labels:
     - resumable;
     - serial, with a 15–120 s rate-limit ladder;
     - a deterministic floor under every reader;
     - DIGGS gates.
   - It is not reachable from the page (OWN-11), and on this route its engine
     cannot run (RI-03). It also has no plan-location step (RI-24).
10. **The continue turn (Part 2) confirms TA-19 and finds two writer bugs.**
    - It rebuilt the inventory and the grid from nothing. It marked the 2019
      and 1987 logs "pending", although the first turn's primary had held
      both transcriptions.
    - Its grid is right: right-handed, median 0.3 m off the vector symbols,
      every ID on the same symbol as turn 1.
    - Its file (40 explorations, no SPT) says groundwater was found at the
      bottom of every hole that found none. The writer writes a
      `not_encountered` row that has a depth as a strike.
    - It also puts 33 explorations at elevation 0. Both pass the verdict
      "valid … reads back equal".
    - It called the `reviewer` for the first time on record, to check data
      against a report the reviewer cannot open.

---

## 1. The plan and the path

### 1.1 The primary, step by step

| t (s) | record | step | took |
|---|---|---|---|
| 0–7 | #1–#7 | `write_todos` (5 items), then `sharepoint_download_file` | — |
| 7–188 | #7–#8 | the 26.8 MB PDF downloads | **181.6 s** |
| 190–198 | #11–#12 | `open_document`: page kinds (text, form, scanned, figure, drawing) | 8 s |
| 203–267 | #14–#46 | in one step: `analyze_pdf_page` of page 24 (tiled, 10 side calls), `read_document` of pages 1–21 (paged: only 1–6 came back), 4 contact sheets (never looked at) | 64 s |
| 267–279 | #47–#48 | one model call, then three tool calls: `describe_method(write_diggs)`, `task` H1 (everything), `task` H2 (the grid) | 12 s |
| 279–811 | #50–#394 | **waits for both**; H1 gives up at t 618, H2 returns at t 811 | **532 s** |
| 811–823 | #395–#400 | new todo list (7 items); five `task` calls, one per vintage | 12 s |
| 823–3,006 | #401–#2611 | **waits for all five**; H3 fails at t 856, the last (H7) returns at t 3,006 | **2,183 s** |
| 3,006–3,015 | #2612–#2614 | one call: retry the 2026 borings (H8) | 8 s |
| 3,015–3,055 | #2614–#2692 | H8 runs 40 s, is refused | 40 s |
| 3,055–3,065 | #2693–#2694 | the primary's own call is refused; the turn ends | 10.6 s |

The primary's own model time: **56 s in 9 calls**. It never wrote a word of
text (every `model_end` carries tool calls only). It never passed the grid to
any helper: none of the five second-wave briefs mentions coordinates. The plan
was to merge everything at the end, and the end never came.

**What each brief got right and wrong.**
- **Right.** The section map (2026 logs 28–43, 2024 logs 45–96, 2019 97–106,
  1987 107–156, lab 220–242), the scale (1:1500, 0.5292 m per PDF point) and
  the north arrow's direction. All came from the primary's own page-24 look
  and the contents pages.
- **Wrong:**
  - **TP-1.** H4 was sent to Appendix B (pages 159–219, the photo log). The
    TP-1 log itself is on page 43, inside the 2026 log range, and again on
    page 158. H4 found it after 12 minutes.
  - **2024 scope.** H7's brief named the Figure-2 IDs. The other 18 logs were
    dropped (§2.3).
  - **H1's size.** H1 was asked for "ALL" data from 140+ pages. A
    general-purpose helper has no budget and one context, so it gave up.

### 1.2 Where the 51 minutes went

- **Critical path:**
  - download 182 s;
  - first look 90 s;
  - wave 1, 532 s (H2's grid);
  - wave 2, 2,183 s (H7's 2024 set);
  - the retry and the refusals, 59 s.
- **Model time, agents' own calls:** primary 56 s; helpers 735 s over 72
  calls.
- **Vision side calls:** 915 calls, 15,900 s summed, mean 17 s. At most 8 run
  at once per process (`GEOTECH_VISION_MAX_INFLIGHT`). The 8 slots were full
  (≥ 440 of a possible 480 call-seconds a minute) in 26 of the 51 minutes.
  So from minute 23 to minute 47 the helpers were mostly queuing for a slot.
- **Tool waits inside the helpers** (the time between a helper's tool calls
  and its next model call):

| helper | wall | its model calls | waiting on tools |
|---|---|---|---|
| H1 | 339 s | 42 s | 297 s |
| H2 | 532 s | 119 s | 413 s |
| H4 | 928 s | 77 s | 851 s |
| H5 | 1,560 s | 151 s | 1,409 s |
| H6 | 1,251 s | 102 s | 1,149 s |
| H7 | 2,183 s | 189 s | 1,994 s |

**The slowest single steps:**
1. **H7, t 880 → 1,551 (671 s): 21 `log_grid` calls on scanned 2024 logs**
   (#593–#925).
   - Every one came back with 0 rows; 13 said `needs_values`.
   - Each call holds the open document's lock for the whole raster pass:
     `planlens/tools/toolkit.py:1077`, `with entry.lock: grid = log_grid(...)`.
   - That pass takes 14–20 s for one scanned pair when timed alone (0.7 s for
     a vector pair).
   - All five helpers used the same document handle, so they queued on one
     lock:
     - the slowest `log_grid` calls finished after 597–605 s;
     - H4's `read_document` took 215 s and 257 s (#769, #840);
     - H5's `search_document` took 504 s (#931);
     - H5's four vector `log_grid` calls took ~223 s each (#819–#823).
2. **H7's page reads.** Three waves of 11–16 `analyze_pdf_page(tiles="3x3")`
   calls (10 side calls each): 347 s, 414 s and 263 s, then 206 s for the lab
   pages.
3. **H2's 14 `render_region` looks at Figure 2:** 84 s and 149 s steps.
4. **The download**, 182 s.
5. **One model call of 98.7 s**: H7 writing the 52,963-character
   `write_diggs` arguments (#2605–#2606).

**H7's context.** It reached about a million characters of tool results by
its 68th message. deepagents' summarization then compacted it (#2375, 53 s).
The 2024 data was written from that summary. The log keeps the summary only
to 8,000 characters (§6).

### 1.3 Work done more than once

- `log_grid` on the 2026 logs (pages 28–43): **three times** (H1, H3, H8).
  On the 2019 logs: twice (H1, H5).
- The lab pages 220–242: up to three helpers each read the same pages,
  looking for "their" IDs. 33 page looks were duplicates.
- Page 24: the primary (tiled), H1 (4×4 tiles), and H2 (4×4 tiles plus 14
  zooms). That is about 60 side calls on one figure.
- The vector `log_grid` results were paged (`next_offset`) on 38 of 40
  calls, after 33–61 rows of 55–234. **No helper ever asked for the rest**
  (no call passed `offset`). The helpers saw a median of about half of each
  log's grid: a seventh of the longest log, the top quarter of most 2026
  logs.

---

## 2. Correctness

### 2.1 The tally

Six explorations, four vintages, checked value by value against the rendered
pages. Pages are PDF pages (1-based). The values themselves are in
`raw/notes/NOTES.md`.

| exploration | vintage, page kind | source | checked | agree | wrong | left out (printed, not in output) | added (not printed) |
|---|---|---|---:|---:|---:|---|---|
| TP-1 (the saved file) | 2026, vector log + narrative | 43 (copy on 158), 14, 20 | 12 | 12 | 0 | all 3 strata | position 0, 0 and ground elevation 0 (the writer) |
| LB-1 | 2019, vector | 98–99 | 81 | 80 | 1 (a misspelling silently corrected) | 8 recovery lengths | — |
| boring 12 | 1987, scan | 133 | 61 | 58 | 3 (an elevation 1.00 m off; one moisture 43.4 for 45.4; one description garbled) | 1 moisture (flagged unsure); no layers | — |
| boring 20 | 1987, scan | 149 | 64 | 64 | 0 | 2 SPT rows (one unsure, one refusal row) | — |
| P5-4 | 2024, scan | 72 | 87 | 86 | 1 (a description cut short, USCS lost) | 12 moisture values | water at 0 m |
| P6-1 | 2024, scan | 74–75 | 101 | 101 | 0 | 14 moisture values | water at 0 m |
| **total** | | | **406** | **401** | **5** | 3 strata, 46 values | 4 values |

Two more classes are not in the counts:
- **Ditto marks.** 10 of the sampled 2024 layers, and **89 of the 165 layers
  in the whole 2024 set**, carry a description that is a bare ditto mark
  (`"`). It is faithful to the page and useless as data.
- **Drive increments.** The 1987 and 2024 forms drive 20 / 15 / 15 cm.
  - The tool takes no increment, and the writer assumes 0.1524 m per drive
    (`report_ingest/diggs_writer.py:2313-2323`).
  - So every drive record from those forms would be written with the wrong
    first increment. The N stays as printed.

**Reading quality.** Where a helper finished, its reading of scanned logs was
good:
- 0 wrong among 40 + 48 + 56 SPT values on three scanned logs;
- the three real reading errors were all on one 1987 scan.

The helpers also flagged their own uncertain values rather than guessing (H6
omitted two readings it saw two ways).

### 2.2 What the tool shape threw away

- **Moisture.**
  - The 2024 logs print a moisture content on every SPT row: 191 rows in the
    21 logs (a number, or a refusal code on a refusal row).
  - None reached the data. The tool's `samples` list is the only place for
    `water_content`, and the helper had been told "no sample IDs, do not
    invent sample objects".
  - The record model (`report_ingest/model.py`, `Sample`) accepts a sample
    with no label (`sample_id=""`). The tool description does not say so.
- **Recovery.**
  - The 2019 logs print recovery in centimetres.
  - The tool description offers only `recovery_percent`, so H5 left out all
    8 values on LB-1.
  - The model has `recovery` (a length) and `rqd_percent`, but the
    description lists neither (`funhouse_agent/adapters/subsurface_adapter.py:913`).
- **Drive increments.** The model has `SPT.increment` (one length per
  record). The description does not list it either.

### 2.3 Scope: what was never extracted

- **2024.**
  - Pages 54–71 hold 18 more 2024 logs, from the same study's other sectors.
    The report says that study had 39 borings.
  - H7 read every one of those pages with 3×3 tiles, then excluded all 18
    as "not the requested Figure 2 light-pole run" (its summary, #2376).
    Its brief had listed the Figure-2 IDs.
- **2026.**
  - **None of the 10 borings was extracted.** H1 gave up, H3 and H8 were
    refused.
  - Their deterministic floor existed and was good. `log_grid` on the vector
    2026 logs placed blows, recovery and samples correctly in the rows
    checked (LBA-1, page 30). It did merge thin strata: TP-1's three strata
    came back as one layer.
- **Laboratory data.** **No lab test was extracted for any exploration.**
  - TP-1 rightly had none: H4 found that the lab sheets it had suspected
    print LBA-1 and LBA-5.
  - H6 found no lab sheet printed with a 1987 boring's ID, and linked none
    rather than guess.
  - H7's arguments carry `lab_tests: []` after it read the lab pages.
  - Whether H5 returned any for 2019 cannot be told: the record cuts its
    answer at 32,000 characters.

### 2.4 The saved DIGGS file

`TP-1_2026.diggs.xml` (H4, `write_diggs` #1304–#1305) is schema-valid and
reads back equal, but:
- **No strata.**
  - The log describes three (topsoil over two fill units). Its ruled line
    under the topsoil sits at about 0.12 m on the depth scale.
  - The helper wrote `layers: []`, because the boundaries are "not printed
    numerically".
  - The descriptions are not in the remarks either.
- **Position `0 0 0`.**
  - The writer fills a missing x, y and elevation with 0.0
    (`report_ingest/diggs_writer.py:604-646`).
  - It reports nothing in `written.assumed`, and the read-back compares only
    what was given, so the verdict says "valid … reads back equal".
  - In a file with the 2019 borings at about 876 m, this pit would sit at
    elevation 0.
- **The helper's answer.** It opened with a `sandbox:/mnt/data/...` link,
  although the tool's own `output_note` says such a link will not reach the
  user.

### 2.5 Two vision misreadings worth knowing

- **H4's two contact-sheet reads** were asked to find the pages relevant to
  TP-1. Both found TP-1 where it is not.
  - #620 named pages 169–172 (0-based) as TP-1's site plan, photo log and
    field log. They are the geophysics report.
  - #620 also named pages 185–187 as "the TP-1 packet". They print LBA-1.
  - #620 called the real TP-1 photo pages (158–164) "not TP-1 records".
  - #812 put "TP-1" on seven lab sheets, 206–212, with an Atterberg sheet
    visibly "NP". Page 206 (PDF 207) is the 2026 lab's moisture sheet for
    **LBA-5**.
  - H4 spent most of its first 12 minutes reading those pages at full size.
    It then excluded them correctly.
  - The other 13 `analyze_image` calls on contact sheets mostly answered
    "too small to read". Asked to find one ID, the model found it
    everywhere.
- **H2's 14 zooms on Figure 2** were asked for "symbol center ... in PDF page
  points".
  - A side call sees only an image. The answers came back in at least four
    frames: crop pixels, "PDF pt crop-relative", page points, and a
    lower-left origin (#249–#391).
  - That is how the grid's points went wrong (§3).

---

## 3. The grid

**What the user asked.** An x-y grid from Figure 2 for explorations with no
coordinates. No log in the report prints coordinates.

**What H2 did.**
- **Scale.** It measured the stated scale with `measure`: 1:1500 stated,
  0.5292 m per point (#173). The scale bar agrees within 1.3 %.
- **North arrow.** It zoomed on the arrow and read 152.6° clockwise from the
  page's x axis. That is right: the arrow points to the sheet's lower left.
- **Points.** It read every plotted ID by tiles and 14 zooms, and returned a
  transform plus 54 points (#394).

**What is wrong.**
1. **The axes are mirrored.**
   - The transform is `x = 0.529 [0.460 (u−54) + 0.888 (v−54)]`,
     `y = 0.529 [−0.888 (u−54) + 0.460 (v−54)]`, and the helper calls x east.
   - Its east vector (0.460, 0.888) points to the sheet's lower right. With
     north to the lower left, that is **west**.
   - Any rotation of a page frame whose y runs down gives a mirror image.
     Getting a right-handed grid needs a reflection, which the helper's
     matrix lacks.
   - Every x has the wrong sign. Boring 2: the helper has x = +135 m; the
     right-handed value from its symbol is −136 m.
   - The y values and the distances between points are right. The map is
     the mirror image.
   - This is hand arithmetic in a model's prose: the case the prompt's
     "always use your tools for calculations" rule exists for, with no tool
     to call.
2. **The points are labels, read off prose.**
   - The PDF draws each program's symbol as a circle of its own size:
     12.6 pt × 10 (2026), 10.2 pt × 21 (2024), 9.5 pt (2019), 9.0 pt × 18
     (1987). So every centre is in the vector data exactly.
   - Against those centres, the helper's 53 matched points are:
     - **median 8.8 pt off (4.7 m)**;
     - 43 of 53 beyond its stated ±2 m;
     - 19 beyond 5.3 m;
     - the worst 13.8 m off.
   - Overlaid on the page (`raw/notes/img/p24_overlay_left.png`), most points
     sit on the label text beside the symbol, not on the symbol.
   - Its "±2 m" claim was not measured. It also leaves out the ±0.7° arrow
     reading, which it mentions separately.
3. **It never reached a file.**
   - The primary did not pass the grid to any helper.
   - The one file written has TP-1 at 0, 0, 0, although H2 had placed TP-1.

The task's own brief suggested the vector tools ("pdf_import or
drawing_ir"); H2 never called them.

**What it should be.** Two steps, both of them code:
- a tool that lists a plan sheet's plotted symbols by drawn type, with their
  exact centres and the label beside each (vision only where a label is
  drawn rather than text);
- a tool that turns page points into a right-handed local grid from a north
  vector and a scale, and states its own check (east is 90° clockwise from
  north).

The model's part is to choose the origin and say which program each symbol
type is. RI-24 already names "boring locations off the plan figure" as a
report-ingest step.

---

## 4. The refusal rows

**What the logs print.**
- **2024:** "50 – >50" for the drives and "> 50" for N, on 17 rows across 21
  logs.
- **2019:** "60/15"" (sic), N "Refusal".
- **1987:** "97 -- --", N 97.

**What the helpers did:**
- **H5 (2019):** `blows: ["60/15\""], refusal: true`, no `n`. This is right,
  and it is the format the writer was built for.
- **H7 (2024):** `blows: [50, "-", ">50"], n: ">50", refusal: true`. The tool
  description says "`n` (ONLY if printed)", and the N *was* printed, as text.
- **H6 (1987):** dropped the 6.50 m row ("refusal status unresolved").

**What `write_diggs` did.**
- It refused the whole call (#2608): 17 problems, all `spt[k].n: Input should
  be a valid integer`, and the hint "fix the named fields … and call
  write_diggs again".
- **H7 did not fix them.** One 8-second call later it answered: "the
  export correctly rejected 17 refusal rows … no invalid or partial file was
  written" (#2610). That was 268 characters.
- The 21 borings' data (52,963 characters of arguments) existed only in H7's
  context and went with it.

**What the primary did.**
- It learnt the fix. H8's brief says "for refusal omit integer n and retain
  printed notation in blows/refusal" (#2614).
- But it applied the fix only to the 2026 retry. It had no way to get the
  2024 data back, and it did not ask for the 2024 set again.

**Since then.** RI-26 (cc51fa7) makes the refusal message point to the
`refusal` field. That helps a model that retries.

**Still open:**
- **The writer could take the printed text.** It could accept a printed
  non-integer N (">50", "R", "50+", "Refusal") as a refusal: keep the text,
  set `refusal`, report it under `written.assumed`, and write the file.
  - This is "as printed" applied to the writer itself.
  - It does not depend on the model reading a hint, and it covers every
    form's notation, not this one.
- **The description could say it.** It could state the rule outright: N is
  an integer; a printed N that is not one goes in the drives with
  `refusal: true`.
- **The general fix is §8's A2.** A helper refused by a validation error
  must hand back the data it had.

---

## 5. The rate limit

### 5.1 Every refusal

| # | record | t | call | started | took | what was running |
|---|---|---|---|---|---|---|
| 1–5 | #568, #573, #575, #577, #579 | 856.45–857.02 | 5 vision side calls (helpers' page looks; the record cannot say whose) | 844.8–845.3 | 11.5–12.8 s | 5 helpers 33 s after they started; 8 vision slots full; 2–3 agent calls |
| 6 | #570 | 856.47 | **H3's own call** (21 messages) | 846.07 | 10.4 s | same → H3 raised, `task` error (#571) |
| 7 | #2690 | 3,054.81 | **H8's own call** (39 messages) | 3,043.37 | 11.4 s | H8 alone; no vision in flight → `task` error (#2691) |
| 8 | #2694 | 3,065.44 | **the primary's call** (29 messages) | 3,054.82 | 10.6 s | nothing else → turn ends (#2695) |

- **The 5 vision refusals cost nothing visible.** No tool result says an
  image was not read. The vision engine's own busy ladder
  (`funhouse_agent/deep/vision_engine.py:217-345`) retried them: 3 tries, 2 s
  then 4 s with jitter, honouring `retry-after`.
- **The agents' own calls have no such ladder.** Each refusal arrived
  10.4–12.8 s after its call started. That fits the client's own quick
  retries (on the Tiny Apps engine: `max_retries=4`, exponential from 0.5 s,
  about 7.5 s of backoff; `webapp/tinyapps_engine.py:133-152, 497-503`). It
  does not fit any wait that spans a one-minute window.
- **What the record cannot confirm.**
  - The error class, `OpenAIRateLimitError` with "Reason (from remote): 'Too
    Many Requests'", is not the stock OpenAI SDK's wording. So the record
    cannot say which client raised it.
  - It cannot say how many HTTP tries each call made (§6).

### 5.2 Requests or tokens?

| minute | calls started | refused |
|---|---|---|
| 37 | 48 (47 vision) | 0 |
| 40–42 | 38–43 a minute (vision) | 0 |
| 14 (t 840–900) | 18 | 6 |
| 50–51 | 8 | 2 |

The refusals cluster in the minutes with the heaviest **contexts**, not the
most **requests**:
- **The 60 s before refusal 6** held 27 agent calls carrying ≥ 637 k
  characters of tool results. 11 vision calls also started, with all 8
  slots busy.
- **The 60 s before refusal 7** held 8 agent calls carrying ≥ 1.18 M
  characters of tool results: about 295 k tokens, before adding each call's
  ~68 k characters of system prompt and tool schemas. No vision call ran.
  - One was H7's last (231 k characters). The 98.7 s call before it had
    generated the 53 k-character `write_diggs` arguments.
  - One was the primary's (139 k).
  - Three were H8's, 6 s apart, at 245 k, 274 k and 284 k characters. H8 had
    put 16 `log_grid` results of about 15 k characters each into its context
    in one step (#2625–#2656).

The pattern fits a **tokens-per-minute** limit on the deployment. With
`usage` null and no 429 headers kept, it cannot be proved from the record.

### 5.3 Would a concurrency cap on helpers have avoided it?

- **The first refusals: probably yes.** Five helpers had started within 12 ms
  of each other (#401–#405). With two at a time the burst would have been
  less than half the size.
- **The final refusals: no.** They came with one helper running. Its own
  contexts, and H7's last calls, filled the minute.
- **What would have carried the turn** is the combination:
  - **A ladder on the agents' own calls**, honouring `retry-after`, else
    15 / 30 / 60 s. That is the ladder `report_ingest`'s engine has used
    since 5.20.5, after 429s killed 10 of 38 label reviews.
  - **A pacer shared by every call on one key.** It would count estimated
    input tokens (characters ÷ 4 plus image tokens), not requests.
  - **Smaller contexts.** For example, `log_grid` results read as compact
    per-log records rather than 15 k-character row dumps.

---

## 6. The record

**What it cannot answer, and why:**

| question | why not | what would answer it |
|---|---|---|
| What did the turn cost? Which limit was hit? | `usage` is null on every one of the 989 `model_end` records; `turn_tokens` is 0 (`trace.jsonl`) | The Prompter client is a `ChatOpenAI` with a custom `base_url` and `http_client`. langchain-openai then leaves `stream_usage` off by default (`langchain_openai/chat_models/base.py:1217-1236`), and the app streams the graph. **Likely cause; verify:** set `stream_usage=True` in `webapp/tinyapps_engine.py:build_chat_model` (if the gateway honours `stream_options`), else read usage from a non-streamed response |
| Which helper made a call? | Every helper event says `general-purpose`. `parent_run_id` points at unrecorded graph nodes. 607 of 916 vision calls could not be attributed even by chaining | A `task_id` (the `task` tool call id) and the helper's label on every nested event, through a contextvar set when the sub-agent starts |
| What did the 429 say? How many tries? How long were the waits? | Only the exception text is kept; the client's retries are inside one call | Status, `retry-after`, any `x-ratelimit-*` headers and the attempt count on `model_error` (an httpx response hook can collect them); each app-level wait as an event |
| Which engine, which retry policy? | `meta.json` names the model only | The engine class, `max_retries`, the vision busy tries and slots, in `meta.json` and on `turn_start` |
| What did the primary see from H5? | Tool results are cut at 32,000 characters in the log (`activity_log.py`); H5 returned 50,529 | `task` results in full, or a sidecar file per long result |
| What did summarization keep of H7's data? | The summary's `text` is cut at 8,000 characters (#2376) | The full summary in a sidecar |
| Was a slow tool working or queuing? | Tool durations include the vision-slot wait and the document-lock wait (`log_grid` 600 s, `read_document` 257 s) | `queued_s` on `tool_end` and vision `model_start` (slot wait), and the lock wait in the document bridge |
| Which page or tile did a vision call read? | A side call is a bare `model_start` with `n_messages: 1` | The parent tool's run id, the page, the tile and the image size and detail on the vision call's events |
| Why did the primary choose five parallel helpers? | Its `model_end`s carry no text and no reasoning summary | A reasoning summary where the route returns one; nothing more to log otherwise |

---

## 7. Fit: was this a job for the chat agent?

**What the chat agent is good at here:**
- **Deciding what the user means:** which investigations count as "new and
  historic", what the grid's origin should be.
- **Reading single hard pages,** and it read them well (§2.1).
- **Saying what it could not do.**

**What it is bad at here:**
- **Doing 140 pages of the same thing in one context.** Contexts grow to a
  million characters, get summarized, and are lost when a call fails.
- **Splitting work by a model's brief.** The brief set the scope, and the
  shared pages were read three times.
- **Running it as one turn.** One refusal ended 51 minutes, and nothing
  persisted.

**What `report_ingest` does instead** (`module_work/harness_theory/report_ingest.md`):
- **Work items come from page labels, not from a brief.** Every log page in
  the report is read, including the 18 2024 logs this turn dropped.
- **Each item is saved as it finishes** (`items/item_N.json`). A rerun
  resumes.
- **Calls are serial,** with a 15–120 s rate-limit ladder.
- **A deterministic floor sits under every reader** (`log_grid`, the lab
  tables), with merge rules that never drop a floor value.
- **DIGGS with two gates.**
- **Honestly:**
  - Its measured accuracy is mixed:
    - logs 73 % blind;
    - lab 55 → 82 % blind;
    - DIGGS read-back differed on 9 of 12 files at 5.31.0, fixed since in
      `f61505d`.
  - A report this size costs it 60–900 calls and 10 minutes to 3 hours,
    serially.
  - It may treat the appended earlier investigations as bound documents,
    each its own record and DIGGS file (triage decides; it called 19 of the
    38 corpus reports `multi_document`). This user wanted one file, so a
    "merge a record and its bound children into one DIGGS" step is needed.
  - It has no plan-location step (RI-24).

**What blocks it:**
- It is off on the page (OWN-11, Later).
- Its engine needs a Funhouse SDK prompter, so it cannot run on the Prompter
  key route this session used (RI-03).

**Verdict.**
- **The shape that fits:** the chat agent plans and asks the user; a pipeline
  reads the pages; code places the points; the chat agent assembles and
  explains.
- **The chat loop is the wrong place** for the 140-page middle. Until the
  pipeline is reachable, the general fixes below (A1–A5) are what make the
  chat loop survive a job this size.

---

## Part 2. The continue turn

**Source.**
- The owner then typed "Are you able to continue?". This part is read from
  the app's progress list for that second turn:
  `raw/conversation/continue_turn_details.txt` (114 lines; long tool results
  cut with "[+N chars]").
- It was taken at 25 min 52 s, while the turn was still running.
- "Line N" below is a line of that file.
- There is no activity log for this turn yet. So what a helper did inside its
  `task`, and the reviewer's answer, are not in it.

### P2.1 It started again from nothing (TA-19, confirmed)

**The step list:**
1. A new five-item todo list beginning "Inventory Volume 1 investigation
   datasets".
2. `open_document`.
3. `read_text_file` of `TP-1_2026.diggs.xml`, the one file the first turn
   left; the per-turn files note pointed at it.
4. `parse_diggs` on it. It reads back x 0.0, y 0.0, elevation 0.0.
5. A `task` to redo the inventory.
6. A new look at page 24, including a `render_region(..., reuse=True)` "to
   recover the prior zoom reading". Whether a kept reading was reused is not
   shown.
7. Two more `task` helpers: the 2026 logs and the grid.

**What it lost.**
- The first turn's primary *had* the 2019 transcription (H5, 50,529
  characters) and the 1987 one (H6, 10,918 characters) in its context.
- A turn that ends on an error saves only an error stub. So the second turn
  started without them, and wrote all four 2019 borings and all four 1987
  borings as "detailed log transcription pending".
- It also rebuilt the grid.
- Persisting the first turn's tool results (TA-19), or helpers writing their
  data to files (TA-21), would have handed it eight transcribed borings and a
  grid.

### P2.2 The second file: `..._DIGGS_development_v1.xml`

**What was written.** `write_diggs` (progress list line 109) wrote 40
investigations, 39 layers, 0 SPT, 0 samples, 8 water rows and 1 test. The
verdict was "valid DIGGS 2.6: it passed the schema and reads back equal";
`assumed: []`.

| group | in the file |
|---|---|
| 2026 borings (EGB-2, LBA-1–LBA-6) | header, layers, water; **SPT and samples left out** ("SPT held for depth QA") |
| EB-1, EB-3, EB-4 | location and a total depth of 4 m only |
| TP-1 | location, date, remarks, water; **no strata again** |
| 2019 (LB-1–LB-4), 1987 (12, 13, 14, 20), 2024 (21 P-series) | **location only** ("transcription pending") |

**Checked against the pages.** The 2026 headers agree: 14 of 14 values
(ground elevation and completion depth on 7 logs, PDF 28–41). The defects are
elsewhere:

1. **Groundwater "found" at the bottom of every hole that found none.** This
   is a writer bug, reproduced.
   - **What the agent passed.** All 8 water rows, for the seven 2026 borings
     and TP-1, are `when: not_encountered` with `depth` equal to the total
     depth (e.g. LBA-1: 12 m).
     - The logs print NE for first, completion and 24-hour readings. The
       report says groundwater was not encountered in any boring.
     - Passing the depth is the agent's reading of "not encountered to
       12 m". Nothing in the tool description says to leave the depth out.
   - **What the writer does.** `water_strikes` treats any row with a depth as
     a reading, whatever its `when`
     (`report_ingest/diggs_writer.py:745-760`: `readings = [w for w in
     inv.water if _si(w.depth) is not None]`). Only a row with no depth
     becomes `<notEncountered>true</notEncountered>`.
     - So each boring got an `initialWaterStrikeReading` at its bottom.
     - The pit got a "Water level" `Test` at 1.5 m (`pit_water`,
       `diggs_writer.py:990-1011`). That is the file's "1 test".
   - **What reading it back gives.** `parse_diggs` turns these into
     `gwl_depth_m` = total depth for all eight (line 113). The 2.6 reader
     takes the initial strike reading (`subsurface_characterization/diggs26.py:742-770`).
   - **Why nothing caught it.** The round-trip gate compares the depth and
     not `when`, so it passed.
   - **Reproduced** with `_run_write_diggs` on four synthetic holes:
     - not encountered with a depth: a strike at that depth, `gwl` 12.0;
     - not encountered without a depth: `notEncountered`, `gwl` None;
     - a pit not encountered at 1.5 m: `gwl` 1.5.
     - The verdict was "valid … reads back equal" and `assumed: []` every
       time.
   - **What DIGGS can say.** It can say "not encountered" for a borehole:
     `WaterStrike` is a choice between `notEncountered` and the readings
     (2.6 `Kernel.xsd:8836-8844`). A `TrialPit` has no `waterStrike` at all.
     The depth explored is already the hole's total depth.
   - **The fix.**
     - **Writer:** a `not_encountered` row is never a reading. It is
       `notEncountered` for a borehole, and the existing "kept in the record"
       note for a pit.
     - **Gate:** compares `when`.
     - **Description:** says to leave the depth out when water was not
       encountered.
     - **Reader:** has no field for "not encountered" (`gwl_depth_m` None
       either way). A `groundwater: not encountered` flag would let plots
       and summaries say it.
2. **Elevation 0 and position 0, 0 where none was given.** This is a writer
   default, read back as data.
   - 33 of the 40 explorations have no elevation: EB-1/3/4, TP-1, the
     2019, 1987 and 2024 sets. Each is written at elevation 0, about 880 m
     below its neighbours. `parse_diggs` reports `elevation_m: 0.0` (EB-1,
     line 113).
   - LB-1's log prints its elevation. The second turn did not have
     it.
   - 1987-B20 has no x, y. It was written at 0, 0, which in this file is
     the grid's own origin: a real-looking point at the corner of the
     drawing frame.
   - **Why the writer must write something.** DIGGS 2.6 requires a
     `referencePoint` on every sampling feature (`Kernel.xsd:1215`,
     `minOccurs="1"`).
   - **Why the reader cannot tell.** The writer fills 0.0
     (`diggs_writer.py:604-646`). The reader also defaults to 0.0 when no
     position is found (`subsurface_characterization/diggs_parser.py:264-277`),
     and the site model's `x`, `y` and `elevation_m` are plain floats
     defaulting to 0.0 (`site_model.py:217-219`).
   - **The fix.** The writer marks a placeholder position as unknown (a
     named property, or an `srsName` that says so) and lists it in
     `written.assumed`. The reader returns None for a marked or missing
     position; plots skip unknown points. Same item as RI-27 (a); note that
     "leave it out" is not open for `referencePoint`.
3. **Depths that are not measured.**
   - EB-1/3/4 carry `total_depth` 4 m. That comes from the legend of a plan
     titled as the *proposed* investigation ("EB; 4m"), not from a log; the
     report has no log for them.
   - It is written as `totalMeasuredDepth`.
4. **A layer below the bottom of the hole passes silently.**
   - LBA-2's header prints 20.1 m and its last page "End of boring at
     21.1 m". The agent kept both and said so in the remarks.
   - The layers run to 21.1 m against a total depth of 20.1 m. The writer's
     cross-checks found 0 entries.
   - A layer, sample or SPT deeper than the total depth is a cheap, general
     check.
5. **Renamed IDs.** The 1987 borings are written as `1987-B12` and so on;
   the log prints `12`. A program prefix helps uniqueness, but the printed ID
   should stay findable (as the name, with the prefix in the id).
6. **No SPT, and the verdict is still "valid".** The agent said why in each
   remark, which is honest. The file the owner has holds no blow count at
   all.

### P2.3 The grid, second attempt: right

**How it is defined.** The origin is the lower-left corner of Figure 2's
drawing frame, (34.8, 757.5) pt. +X runs to page-right and +Y to page-up, so
the grid is right-handed and page-aligned, not north-up. The scale is
1:1500, 0.52917 m per point. It is built from "PDF symbol centres".

**Checked against the vector symbols** (each point's page position recovered
from its x, y):
- **Accuracy.** 38 points matched a symbol of their program's size: median
  **0.6 pt (0.3 m)** off, one over 3 pt (P9-4, 5.1 pt = 2.7 m).
- **TP-1's diamond:** 0.7 pt off.
- **Agreement with turn 1.** For all 38, the symbol each ID was given is the
  same symbol turn 1's points lie nearest. Two independent readings agree on
  every label-to-symbol pairing.
- **By eye.** Twelve were checked on a render, all on the right symbol:
  LBA-1, P9-2, 12, TP-1, 13, LB-1, LB-2, LB-3, P9-3, P9-4, P8-1 and P8-2
  (`raw/notes/img/turn2_check.png`).
- **1987-B20.** Left unplaced, rightly: boring 20 is not plotted.

**Two things to say about it:**
- The coordinate-system text does not give the north direction (153°
  clockwise from +X on the page). A reader could take +Y for north.
- How the helper got symbol centres this exact is not in the progress list.
  The turn-1 helper read labels from vision prose. Unless the method is a
  tool, the next run may do either (PL-25).

### P2.4 The reviewer was called, for a job it cannot do

**The call.** After writing and re-reading the file, the primary delegated
to `reviewer` (line 112): "QA/QC the DIGGS contents against the report". It
is the first use on record; GM-43 counted 0 of 106 in the evaluation.

**What the reviewer is:**
- **Description:** it checks "methodology, parameter ranges, required
  factors of safety, and any missing code provisions against the published
  references" (`funhouse_agent/deep/agent.py:273-279`).
- **Tools:** reference lookups only (`build_reviewer_subagent`,
  `deep/agent.py:587-625`). It cannot open the report, the page images or the
  DIGGS file.
- **So:** it can only judge what the primary pasted into its brief. Its
  answer was not in the progress list.

**What data QA needs.** Checking transcribed values against pages takes a
helper with the document tools, or a deterministic comparison with the
`log_grid` floor, which is report ingest's merge rule.

---

## 8. Lessons and recommended actions

Each action is general (no rule fitted to this report). Tiers are suggestions
for the lead; "extends" names the BACKLOG item it grows, where one exists.

**Where they are filed** (FINDINGS.md, BACKLOG):

| action | BACKLOG |
|---|---|
| A1 | TA-20 |
| A2 | TA-21 |
| A3 | TA-22 |
| A4 | EV-14 |
| A5 | RI-27 |
| A6 | RI-28 |
| A7 | PL-24 |
| A8 | PL-25 |
| A9 | VIS-24 |
| A10 | VIS-25 |
| A11 | EV-15 |
| A12 | RI-29 |
| A13, A14 | TA-23 |

Part 2's actions (B1–B5) follow the first table.

| # | what | why (evidence) | where | tier | extends |
|---|---|---|---|---|---|
| A1 | **A busy ladder on the agents' own model calls** (the primary and every helper): on a 429 or overload, wait per `retry-after`, else 15 / 30 / 60 s, up to ~3 min, then raise; log each wait | 3 of 8 refusals hit agent calls and each ended its agent (#570, #2690, #2694); the 5 vision refusals were retried by the vision ladder and survived | a wrapper around the chat model that `build_deep_agent` gives the primary and the sub-agents (`funhouse_agent/deep/agent.py`), shared by `webapp/tinyapps_engine.py` and `funhouse_agent/deep/databricks_bridge.py`; reuse `vision_engine.busy_kind` / `_busy_wait_s` | Now | TA-12, TA-04 |
| A2 | **A helper never loses its work.** When a sub-agent raises or gives up, its tool results so far go to a file in the conversation's scratch folder, and the `task` result names the file and lists what is in it. A helper's brief asks for partial data over an apology | H1 had 413 k chars of results and returned 366 chars (#370); H7 had 21 borings and returned 268 (#2610); H3 and H8 died holding 147 k and 282 k | `funhouse_agent/deep/tool_guards.py` (`ToolErrorGuard` for `task`), sub-agent middleware in `deep/agent.py`, `funhouse_agent/error_text.py:91-105` ("nothing it was doing was finished") | Now | TA-19 |
| A3 | **Bound the helpers:** at most N `task` helpers at once per conversation (queue the rest); a per-key pacer over agent and vision calls counting estimated tokens; a model-call budget with a graceful last call on `general-purpose`, as `references` (8) and `calc` (16) have | Five helpers started within 12 ms (#401–#405), refusals followed 33 s later; H7 ran 12 agent calls and ~500 side calls with no budget; the slot cap, not the helpers, set the pace | `deep/agent.py` (the `task` tool, sub-agent middleware), `deep/limits.py`, `deep/vision_engine.py` (`call_slot`) | Now | GM-45 |
| A4 | **The record** (§6): usage on the Prompter route (`stream_usage`); `task_id` on every nested event; 429 status, headers and attempts; engine and retry policy in `meta.json`; `queued_s` for slot and lock waits; `task` results and summaries in full or as sidecars; vision events tied to their tool, page and tile | every row of §6's table | `webapp/activity_log.py`, `webapp/tinyapps_engine.py`, `deep/vision_engine.py`, `funhouse_agent/document_tools.py` | Now | RI-21 (cost records), new |
| A5 | **`write_diggs` never invents a location and keeps printed refusals.** (a) A missing x, y or elevation is named in `written.assumed` and in the verdict, not written silently as 0. (b) A printed non-integer N (">50", "R", "Refusal") becomes a refusal record with its text kept, reported under `assumed`, instead of refusing the whole file | TP-1 at `0 0 0`, `assumed: []` (#1305); 17 rows refused the whole 2024 set (#2608); three helpers, three refusal conventions | `report_ingest/diggs_writer.py:604-646`; `funhouse_agent/adapters/subsurface_adapter.py:423-` (validation) and `report_ingest/model.py` `SPT.n` | Now | RI-26 |
| A6 | **The `write_diggs` description matches the model it feeds:** list `recovery {value, unit}`, `rqd_percent` and `increment` (and allow a different length per drive, e.g. 20/15/15 cm); say a sample needs no printed label, so a moisture or index value printed on a row is a sample value; a ditto mark means "as the row above", so give the text it repeats, and have the writer flag a bare ditto | 8 recovery lengths dropped on LB-1; 191 moisture values dropped in 2024; 89 of 165 2024 layers are a bare `"`; the writer assumes 0.1524 m drives (`diggs_writer.py:2313-2323`) for forms that drive 20/15/15 cm | `funhouse_agent/adapters/subsurface_adapter.py:913`; `report_ingest/model.py` (`SPT.increment`, one length today); `diggs_writer._drive_sets` | Next | new RI |
| A7 | **`log_grid` for agents.** (a) A paged result says how much is left (e.g. "43 of 127 rows; call again with offset=43"), or the agent gets a compact per-log record by default. (b) The raster leg's pixel work runs outside the document lock, and a page with no text says `needs_values` quickly | 38 of 40 vector results were paged and never followed; 25 scanned calls returned 0 rows (17 `needs_values`, none answered) and held the lock 14–20 s each, queuing other tools up to 605 s | `planlens/tools/toolkit.py:1066-1100`; `planlens/document/loggrid.py` (the raster leg) | Next | PL-04, PL-08, VIS-08 |
| A8 | **Plan locations as tools, not hand maths:** symbols on a plan sheet by drawn type with exact centres and the adjacent label; page points → a right-handed local grid from a north vector and a scale, with the handedness check printed | the grid is mirrored by a hand-written rotation; points are labels read from prose in four frames, a median of 4.7 m off the vector centres, which the PDF holds exactly by symbol size | planlens (`ir` construct finders / `get_entities`), `funhouse_agent/measure_tool.py`; the report-ingest step of RI-24 | Next | RI-24, PL-13 |
| A9 | **A side call is never asked for page coordinates.** The vision tools rewrite or refuse "give the centre in PDF points" (the side call sees only an image) and ask for pixel boxes, which they convert | 14 zooms on Figure 2 answered in four different frames (#249–#391) | `funhouse_agent/vision_tools.py:2600` (`_vision_prompt`), the `render_region` / `analyze_pdf_page` descriptions | Next | VIS-02 |
| A10 | **Contact sheets are for layout, not for reading IDs:** a contact sheet's result, and `analyze_image` on one, say that IDs need a page look | both contact-sheet reads asked to find TP-1 put it on pages that are not TP-1's: the geophysics report, the LBA-1 packet and seven LBA-5 lab sheets (#620, #812), and called the real TP-1 photos "not TP-1"; most other reads said "too small"; 15 calls mostly spent for nothing | `funhouse_agent/document_tools.py` (`render_page_thumbnails` note), `vision_tools.analyze_image` | Next | VIS-12 |
| A11 | **Suite tasks from this session** (tester feedback becomes a task, not a prompt rule): (1) DIGGS for every log in a section where some explorations are not on the plan; (2) refusal rows in three notations written as refusals; (3) a local grid from a plan with a rotated north arrow, scored on handedness and centre error; (4) lab results linked only by printed IDs; (5) a long extraction that meets a 429 mid-turn | §2.3, §4, §3, §2.5, §5 | `funhouse_agent/review_eval/` or the geotech eval (`funhouse_agent/deep/eval_harness.py`) | Next | EV (new) |
| A12 | **Report → DIGGS through the pipeline** once it is reachable: the chat agent plans and asks; `report_ingest` reads every page (items saved, resumable, serial, floors, gates); a merge of a record and its bound children into one DIGGS file on request; A8 for locations | §7 | `report_ingest/subagent.py`, `report_ingest/writers.py` | Later [owner] | OWN-11, RI-03, RI-24 |
| A13 | **The primary is not blocked by its slowest helper:** with A3's queue, a failed helper frees its place at once, and the primary hears of the failure when it happens (or helpers run as background jobs the primary checks) | H3 failed at t 856; the primary could react only at t 3,006, 36 minutes later | `deep/agent.py` (the `task` tool) | Later | GM-45 |
| A14 | **Divide document work by pages, not by groups,** so shared pages are read once: one pass over the lab appendix, then a link by printed ID | the lab pages were read by up to three helpers (33 duplicate looks); the 2026 grids ran three times | covered by A12; meanwhile a note in the `task` description | Later | GM-45 |

**From Part 2 (the continue turn):**

| # | what | why (evidence) | where | tier | extends |
|---|---|---|---|---|---|
| B1 | **"Not encountered" is never written as a water level.** Writer: a `not_encountered` row is `notEncountered` on a borehole whatever its depth, and the existing note on a pit. Round-trip gate: compares `when` as well as depth. Description: leave the depth out when no water was found. Reader: a "groundwater not encountered" flag beside `gwl_depth_m` | all 8 water rows of the second file read back as groundwater at the bottom of the hole (`gwl_depth_m` = total depth), against logs that print NE; the verdict said "reads back equal"; reproduced on synthetic holes | `report_ingest/diggs_writer.py:738-763` (`water_strikes`), `:974-1011` (`pit_water`), the round-trip gate's `_compare_water` (`diggs_writer.py:2841-2856`); `funhouse_agent/adapters/subsurface_adapter.py:913`; `subsurface_characterization/diggs26.py:742-770`, `site_model.py` | Now | RI-27 |
| B2 | **An unknown position stays unknown end to end.** DIGGS requires `referencePoint`, so the writer marks a placeholder as unknown and says so in `written.assumed` and the verdict. The reader returns None for a marked or missing x, y or elevation (`Investigation.x/y/elevation_m` become Optional), and plots skip such points | 33 of 40 explorations at elevation 0 and one at the grid's origin in the second file; TP-1 at 0, 0, 0 in the first; `parse_diggs` reports them as 0.0 data | `diggs_writer.py:604-646`; `subsurface_characterization/diggs_parser.py:264-277`; `site_model.py:217-219`; the plan and section plots | Now | RI-27 |
| B3 | **`write_diggs` cross-checks depth against depth:** a layer, sample, SPT or water reading below the total depth is an entry. A total depth that comes from a legend or a plan rather than a log is flagged by the description as not a measured depth | LBA-2's layers run to 21.1 m against a total depth of 20.1 m with 0 cross-check entries; EB-1/3/4 got the proposed plan's 4 m as `totalMeasuredDepth` | the cross-checks in `subsurface_adapter._run_write_diggs` / `report_ingest/reconciler.py`; the description at `subsurface_adapter.py:913` | Next | RI-28 |
| B4 | **Keep the printed ID findable:** when the agent prefixes an ID for uniqueness (`1987-B12`), the printed `12` stays as the name. Also, the coordinate-system text of a page-aligned grid states where north is | the second file renamed every 1987 boring; its grid says "+Y page-up" with no north direction (north is 153° from +X) | `write_diggs` description; PL-25's transform tool writes the north direction into the CRS text | Later | RI-28, PL-25 |
| B5 | **Data QA needs eyes on the page.** The `reviewer` was invoked for the first time on record, to check DIGGS contents against the report, but it has reference lookups only. Either route "check these values against the page" to a document-reading helper, or compare against the deterministic `log_grid` floor in code, and let the suite say which | Part 2 §P2.4; GM-43 (0/106 before) | `funhouse_agent/deep/agent.py:273-279, 587-625`; report ingest's floor merge (`report_ingest/floor.py`) | Later | GM-43 |

**What is not an action.**
- **The download.** It took 182 s for 26.8 MB. It is the host's speed, and
  the turn waited on it only once.
- **The reading quality of the scanned logs.** It is not where this turn
  failed (§2.1).

---

## Part 3. The retry turn (turn 3), and turn 2 from its record

**Source.**
- The app's export after turn 3: `raw/run3/conversation/` and `raw/run3/files/`.
  Its `activity.jsonl` holds all three turns in 3,865 records: turn 1 is
  #0–#2695 (as in Part 1), turn 2 #2696–#2996, turn 3 #2997–#3864. "#N" is
  the 0-based line.
- Helpers were rebuilt by the same chaining as Part 1. Every chain ends
  exactly when its `task` returns.
- Checked against the PDF pages, the vector geometry of the logs and of the
  investigation plan (PyMuPDF), and Part 1's truth tables.
- Private tables, scripts and page crops: `raw/notes/run3/` (`NOTES_run3.md`).

**The owner's three turns.**
1. Turn 1: 51 min, rate limit (Parts 1 and 2).
2. Turn 2, "Are you able to continue?": hung on one `reviewer` model request
   (TA-26, TA-25, fixed).
3. Turn 3, "Can you retry?", with "Consult references" switched off: 1,036 s
   (17 min), no errors, a DIGGS file, a plan view and an SPT profile.

---

### In short

1. **The 54 SPT records for LBA-1 to LBA-6 were in hand and right, and were
   held back** (P3.2).
   - The 2026 logs print no sample depth as a number. Each sample is a box
     drawn against the log's depth scale, exact in the PDF (1.05–1.50 m,
     2.55–3.00 m, every 1.5 m).
   - `log_grid` read the scale right and placed every row in depth. But it
     reports no sample intervals. And 7 of its 8 results stopped after the
     first 2.6–4.4 m of each log, and nobody asked for the rest.
   - The helper took layer contacts off that same scale. It refused sample
     depths off it as "not printed". It kept EGB-2's only because a vision
     call happened to state them.
   - Its 60 SPT records were 59/60 right for blows and 60/60 for N. Turn 2's
     helper made the same call.
   - Verdict: not a tool failure. A tool gap plus a description gap, which
     turned a sound "never guess" rule into a wrong one.
2. **The file is accurate where it has data: 250 of 256 checked values
   right** (P3.3). Of the 6 wrong, one matters most: TP-1 sits 141 m away,
   on another exploration's coordinates.
3. **What it leaves out is larger than what it holds:**
   - 54 SPT records;
   - every recovery value;
   - all four 1987 borings (their helper had read the SPT tables and gave
     up);
   - all the 2024 borings;
   - 16 printed layer contacts.
4. **The grid is right-handed and page-aligned, median 0.94 m off the
   symbols** (P3.4). Three of 15 points are off their symbol: one on its
   label, one beside it, one copied from the wrong row. The plan view calls the page axes
   Easting and Northing (GM-76). Here +y points about 116° from north.
5. **11 fields read back as zero** (P3.5). RI-27 on master turns 7 of them
   into "unknown". The 4 total-depth zeros remain (RI-30).
6. **The turn spent its time queuing and retyping** (P3.6):
   - 4 helpers at once on 8 vision slots, full in 8 of 17 minutes;
   - 25 pages looked at again that earlier turns had already read;
   - the primary retyped 29,000 characters of helper output into
     `write_diggs`. Several of the file's faults entered there.

---

### P3.1 What ran

**Turn 2, now from its record** (Part 2 read only the progress list):

| helper | brief | t (s) | agent calls | returned |
|---|---|---|---|---|
| inventory | programs, IDs, pages | 9–145 | 6 | an inventory, 8,128 chars |
| 2026 logs | PDF 28–43 | 203–388 | 7 | records with no sample depths, 29,498 chars |
| grid | Figure 2 | 203–527 | 7 | the right-handed grid (Part 2: median 0.3 m) |

- The primary wrote its file at t 573. It called the `reviewer` at t 577.6.
  After the reviewer's one `model_start`, the record holds nothing more
  from that turn but the end of a parse started at the same moment (TA-26).
- Turn 2 made 74 model calls before the hang: 8 primary, 20 helper,
  45 vision and 1 reviewer.
- **The 2026 helper paged.** It called `log_grid` 15 times: 8 first pages,
  then 7 second pages. It still reached only 5.0–8.6 m of 12–21 m logs. Like
  turn 3's, it gave no sample depths, and the primary wrote "SPT held for
  depth QA".
- **One reuse worked and one missed.**
  - The primary's `analyze_pdf_page(reuse=true)` on page 24 returned a kept
    reading in 0.04 s.
  - Its `render_region(reuse=true)` passed a box that matched no kept zoom
    (a letter-size box on a 17 × 11 in sheet). It looked again (38.5 s).

**Turn 3**: 13 primary calls, 34 helper calls, 168 vision calls. All four
helpers started within 14 ms of each other (t 54.18–54.19).

| helper | brief | t (s) | agent calls | tool results (chars) | returned |
|---|---|---|---|---|---|
| G | the Figure 2 grid | 54–931 | 16 | 197,675 | a coordinate table, 5,350 chars (#3829) |
| 2026 | the 2026 logs, PDF 28–43 | 54–605 | 7 | 226,695 | JSON, 30,713 chars (#3474) |
| 2019 | the 2019 logs, PDF 98–106 | 54–257 | 3 | 67,005 | JSON, 17,800 chars (#3258) |
| 1987 | the 1987 logs, PDF 108–156 | 54–827 | 8 | 284,026 | **an apology, 162 chars** (#3636) |

**The primary**, step by step:
- t 0–54: it reread page 24's text (12 s) and three searches.
  `document_page_map` refused one result for being 159 characters over its
  14,500 limit, and the primary did not retry.
- t 54: the four helpers. No helper was sent for the 2024 light-pole logs.
- t 931–999: one model call of 63.9 s that typed 29,293 characters of
  `write_diggs` arguments.
- Then `parse_diggs`, the two plots and the answer at t 1,036.
- The primary's own model time was 136 s.

---

### P3.2 Why the 2026 SPT records were withheld (question 1)

**Were the depths printed?**
- **No, not as numbers.**
  - On the 2026 logs a sample is a box in the sample columns, drawn against
    the depth scale.
  - The only depths printed as numbers are the scale ticks every metre, an
    elevation beside each stratum change, and a few remarks.
- **They are exact in the drawing.**
  - The box rules on these vector pages put every SPT at 1.05–1.50,
    2.55–3.00, 4.05–4.50, 5.55–6.00, 7.05–7.50 m and so on, every 1.5 m.
  - That is ±0.01 m against the page's own ruler (`truth_spt2026.json`).
  - Refusals are shorter: EGB-2's S-5 box ends at 10.30 m.
- **The 2019 logs print each interval** in their remarks ("SS 0 to 0.5 m").
  That is why the 2019 SPT records reached the file and the 2026 ones did not.

**What the tools returned.**

| call | by | returned |
|---|---|---|
| `log_grid` ×8 | 2026 helper, PDF 28–43 | ruler at confidence 0.98; every row placed in depth (LBA-1's S-1 label at 1.20–1.35 m, its three blows at 1.12, 1.24, 1.37 m); layers with contacts off the drawn stratum lines. **No sample intervals.** 7 of 8 results stopped at 33–43 of 117–234 rows, i.e. at 2.6–4.4 m; no call passed `offset` |
| `log_grid` ×1 | 2019 helper, nine pages at once | 20 of 544 rows (to 0.68 m) |
| `log_grid` ×4 | 1987 helper, scans | 0 rows, `needs_values`; 415–417 s each |
| `measure` ×1 | grid helper, the plan sheet | the stated 1:1500 scale; nothing to do with the logs |

The vision page reads (16 by the 2026 helper) gave blows, N and recoveries.
They stated intervals only on EGB-2's two pages, where the side-call model
read the ruler itself ("interval 7.05–7.50 m"). They stated none on the LBA
pages.

**What the helper did.**
- It wrote: "Most SPT interval depths are graphical rather than numerically
  printed. Except for EGB-2 intervals verified directly, they were not
  inferred; SPT records are keyed to sample ID."
- The primary then left them out of the file: "depth intervals require
  additional graphical digitization".

**Was it sound?**
- **The values were right.**
  - Checked against the pages, the helper's 60 SPT records were 59/60 right
    for the blows and 60/60 for N or refusal.
  - The one slip is EGB-2's S-1: two blows for a printed three.
  - 54 of the 60 were withheld.
- **The rule was applied unevenly.**
  - The same helper took its layer contacts off the same drawn scale
    (`log_grid`'s stratum lines: LBA-1 at 0.75 m and 5.55 m).
  - It took EGB-2's sample depths from a vision call reading that scale.
  - That is the less exact route, and it got one wrong. EGB-2's S-5 was
    written 10.05–10.50 m; the drawing gives 10.05–10.30 m, 32 blows plus
    50 for 5 cm.
  - So whether a boring's SPT reached the file depended on whether a side
    call happened to volunteer a depth.
- **It is systematic.** Turn 2's helper, a separate run, held the same records
  back for the same reason.
- **The cause.**
  - `write_diggs` says "numbers AS PRINTED on the log (never inferred)", and
    the brief added "do not infer". Neither says what a depth measured off
    the log's own scale is.
  - `log_grid`, the tool built to measure that scale, does not report a
    sample's interval. Its paging hid the deeper half of every log.
  - Not a tool failure: the ruler and the row depths were right. A tool gap
    and a description gap.

---

### P3.3 The delivered file, checked against the pages

`SAL_DI1_Volume_I_initial.diggs.xml`: 16 explorations, 47 layers,
40 samples, 38 SPT records. Verdict "valid DIGGS 2.6 … reads back equal",
0 cross-check entries, `assumed: []` (#3837).

**Contents by program:**
- **2026:** EGB-2 with SPT; LBA-1 to LBA-6 with layers and no samples; TP-1.
- **2019:** LB-1 to LB-4 with layers, samples and SPT.
- **1987:** four placeholders (borings 12, 13, 14, 20).

**The tally.** All the file's positions, elevations, depths, dates, layer
contacts and test values were checked, not a sample. Descriptions and
drilling details were not counted.

| class | checked | right | wrong |
|---|---:|---:|---:|
| positions (x, y), against the vector symbols | 15 | 12 | 3: TP-1 141 m off; LB-2 4.5 m (on its label); boring 14 3.9 m (beside its symbol) |
| ground elevations (2026 ×7, 2019 ×4) | 11 | 11 | 0 |
| total depths (the "End of boring" line where the header rounds it) | 12 | 12 | 0 |
| start and finish dates | 24 | 24 | 0 |
| internal layer contacts, against the printed elevation marks (±0.1 m) | 36 | 36 | 0 |
| SPT: top, bottom, blows, N or refusal (38 records) | 152 | 149 | 3 (below) |
| LB-3 core runs: top, bottom, recovery % | 6 | 6 | 0 |
| **total** | **256** | **250** | **6** |

**The three SPT faults:**
1. EGB-2 S-1: blows `[3, 4]`. The log prints 3 / 3 / 4, N 7. The written N is
   right.
2. EGB-2 S-5: bottom 10.50 m. Drawn 10.30 m.
3. LB-1 S-8, a refusal printed `60/15"` beside the remark "SS 3.5 to 3.65 m".
   - The helper kept the printed text. The primary rewrote it `60/15in`.
   - The writer turned that into a 0.381 m drive inside a 0.15 m interval.

**Printed but not in the file:**
- **54 SPT records** of LBA-1 to LBA-6 (P3.2).
- **Every recovery value.** The tool offers only `recovery_percent` (RI-28).
  So are all the 2026 disturbed samples, which print a non-unique "D".
- **From the 2019 logs:**
  - LB-3's grab sample (0–0.4 m, drawn);
  - two strength readings in LB-3's remarks;
  - nine DCP intervals (the answer says the writer has no place for them).
- **16 printed layer contacts**, merged into a neighbour.
  - 5 are a topsoil or asphalt layer.
  - 14 of the 47 written layers span one.
  - The written descriptions are the primary's short summaries, not the
    printed text.
- **"Not encountered" groundwater** on all 12 logs, in remarks only. DIGGS can
  say it (`notEncountered`), and the writer on master writes it (RI-27).
- **TP-1's three strata**, for the third time.
- **The 1987 logs.**
  - The 1987 helper's 3×3 tiles read the SPT tables.
  - Boring 12's tile matches turn 1's transcription row for row. Part 1
    checked that transcription: 58 of 61 values right on boring 12.
  - The helper still returned only "I'm sorry, but I couldn't complete a
    sufficiently reliable structured extraction …" (TA-21, fixed since).
- **The 2024 logs**: no helper was sent. The answer lists them as remaining
  work.

**Written but not printed:**
- **11 zero fields** (P3.5).
- **Four page lists for the 1987 placeholders.**
  - They are even splits of the 1987 section (boring 12: PDF 108–119).
  - Boring 12's log is on PDF 133–134. The helper's own `log_grid` calls
    named the right pages, but its answer was an apology, so the primary
    made the lists up.
  - The answer calls this "source traceability".
- **The drive length.** All 108 drive sets are written as 0.1524 m on forms
  headed "BL/15cm" (the writer's default, RI-28).
- **The hammer type field** holds a sentence about where the value came from.
  The logs leave it blank.

**Changed silently:** two misspellings on the logs were corrected (an
equipment entry and a firm name); `60/15"` became `60/15in`.

**Where each fault came from:**
- **The primary's retyping:** `60/15in`, the shortened descriptions, the
  dropped recoveries, the made-up page lists.
- **A helper's slip:** TP-1's position.
- **A vision read:** EGB-2's S-5 bottom.

---

### P3.4 The grid

**How it is defined.**
- The origin is the outer frame's lower-left corner, displayed (34.92,
  757.08) pt. The vector frame lines confirm it.
- +x runs page-right and +y page-up, at 0.5291667 m per point (1:1500).
- It is **right-handed, not mirrored**, and page-aligned: +x points about
  206° from north, +y about 116°.
- The coordinate-system text says "not georeferenced" but not where north is
  (RI-28).

**Checked against the vector symbol centres** (15 placed points):
- Median **0.94 m** off. 8 within 1 m; 12 on their symbol.
- **Off their symbol:**
  - LB-2, 4.5 m: its vision box was on the label.
  - Boring 14, 3.9 m: beside the symbol.
  - **TP-1, 141 m.**
- The answer claimed ±0.5 to ±0.8 m. Only 6 of 15 meet their own stated
  figure.

**How the points were made.**
- `render_region` returns boxes on a 0–999 grid over its view, not page
  points. The helper converted those boxes to page points in its head (most
  within 0.3 pt) and multiplied them out with 94 `calculate` calls.
  - Two of the calls had a `**` typo; both were redone.
  - The 2026 boxes sat on the symbols. The boxes for LB-2 and the 1987
    borings sat partly on their labels.
- **TP-1:**
  - The helper's own calculation (t 864) had TP-1 within 1.4 m of its
    diamond.
  - Its final table gave TP-1 EB-1's numbers. Its evidence column reads
    "locally overlaps EB-1 vicinity".
  - The primary saw two explorations at one point and told the owner they
    "appear in the same immediate vicinity on Figure 2" and need a field
    check. In fact they are 141 m apart.
- **Boring 20** was left unplaced, rightly. The writer then put it at 0, 0
  (P3.5).

**Three turns, three grids.**
- Turn 1: mirrored, median 4.7 m (Part 1).
- Turn 2: right-handed, median 0.3 m (Part 2).
- Turn 3: right-handed, median 0.94 m, one point 141 m off.
- Turn 2's file, with the better grid, sat in the working folder. Every
  turn-3 helper's `list_files` showed it. But the turn-3 files note named
  only turn 1's files, and nobody opened it (P3.6).

---

### P3.5 Zeros (question 3)

**Where a missing value became zero:**

| where | exploration | field | written / read | why |
|---|---|---|---|---|
| file (`gml:pos`) | TP-1 | ground elevation | 0 (centre line 0 to −1.5) | the log prints no elevation |
| file | borings 12, 13, 14 | ground elevation | 0; centre line 0 to 0 | not transcribed (the 1987 logs print one) |
| file | boring 20 | x, y, ground elevation | 0, 0, 0 | not plotted on Figure 2 |
| read-back (`parse_diggs`, #3845) | TP-1, borings 12, 13, 14, 20 | `elevation_m` | 0.0 (5 fields) | as above |
| read-back | borings 12, 13, 14, 20 | `total_depth_m` | 0.0 (4 fields) | no depth given |
| read-back | boring 20 | `x`, `y` | 0.0, 0.0 (2 fields) | as above |
| plan view | boring 20 | the marker | drawn and labelled at the grid's origin | from the read-back |
| plan view | the same five | hover text | "Elev: 0.0m" (5), "Depth: 0.0m" (4) | from the read-back |

- **11 fields read back as zero.** The answer said so, plainly: "these zeros
  should not be interpreted as actual elevations, depths, or coordinates".
- **On master, RI-27:**
  - The writer marks an unknown x, y or elevation as NaN. The reader returns
    None, and plots skip such points and list them.
  - That covers 7 of the 11 fields and boring 20's marker.
- **Still open, RI-30:** `total_depth_m` is still 0.0 when no depth is given.
  That is the other 4 fields and the "Depth: 0.0m" hovers.
- **One more gap:** for borings 12–14 the elevation is printed but was not
  transcribed. "Not transcribed" is not "unknown", and the placeholder's
  remark should be the only place that says which.

---

### P3.6 How the turn used its budget (question 4)

**Model calls.**
- The primary made 13. The helpers: G 16, 2026 7, 2019 3, 1987 8.
- 168 vision side calls on top. Only 15 can be pinned to a helper; the other
  153 ran while several helpers had looks in flight (EV-14 adds the
  `task_id`).
- `turn_tokens` is 0 again: no usage was recorded (EV-14).
- No refusals and no retries.

**Where the 17 minutes went.**
- **Critical path:**
  - first look 54 s;
  - the grid helper 877 s;
  - the write step 68 s (one 64 s model call);
  - parse, plots and answer 37 s.
- **The vision slots** (8 per process) were full in 8 of the 17 minutes:
  minutes 2–6 and 10–12.
  - The 2026 helper's 16 whole-page looks at vector logs took 97–239 s
    each.
  - The grid helper's 9 zooms took 51–232 s each, 1,230 s in all.
  - Most of that is queuing for a slot.
- **The 1987 helper's four `log_grid` calls**, on scanned pages:
  - 415–417 s each, all ending within 1.1 s of each other (t 568.8–569.9);
  - that pattern points at a wait, not work;
  - Part 1 timed one scanned pair at 14–20 s alone;
  - the record cannot say what they waited on. EV-14 now times lock and
    slot waits.

**Tool results.**
- 882,000 characters into the helpers' and primary's contexts:
  - the 1987 helper 284 k;
  - the 2026 helper 227 k;
  - the grid helper 198 k, with one `drawing_ir` geometry result of 43 k;
  - the primary 106 k;
  - the 2019 helper 67 k.
- The 1987 helper's 284 k ended as 162 characters.
- **The same data was typed twice.**
  - The 2026 helper spent 85 s typing a 30.7 k JSON answer.
  - The primary spent 64 s retyping it, with the 2019 answer, as 29.3 k of
    `write_diggs` arguments.
  - The retyping is where four kinds of fault entered (P3.3).

**Repeats.**
- Turn 3 looked again at **25 pages that turns 1 and 2 had already looked
  at**: page 24, the TP-1 log, the nine 2019 log pages and 14 of the 1987
  pages.
  - The turn-start note listed 80 kept readings, but only the primary sees
    that note. No helper passed `reuse=true`.
- **Turn 2's file was missed.**
  - The note named the PDF and turn 1's TP-1 file, not the file turn 2
    wrote. Turn 2 was interrupted, so its files never reached the note.
  - So turn 3 built a third grid while a better one sat in the folder.

**What the fixes on master would change here:**

| fix | effect on this session |
|---|---|
| TA-19 (an errored turn keeps its work) | Turn 1's transcriptions would have reached turns 2 and 3: the 2019 and 1987 JSON (Part 1: 401/406 right) and the 21 light-pole borings. The 1987 placeholders and the missing 2024 set are exactly what turn 1 had. Whether a turn killed mid-turn (turn 2) gets its files into the next turn's note is worth a test |
| TA-21 (a stopped helper returns a file) | The 1987 helper read 284 k characters, including the SPT tables, and gave up. It would now leave `kept/helper_….md` with every result and a summary, and the primary would get the path. Its prompt now asks for data over an apology |
| TA-22 (two helpers at once, pacer, 30-call budget) | Four helpers would run two at a time: probably a longer turn, shorter vision queues. No 429 came, so the cap would have saved nothing here. The grid helper's 16 calls fit the budget |
| TA-20 (busy ladder), TA-26 (time limit), TA-25 (reviewer) | Turn 3: nothing (no refusal; references off). Turn 2: the reviewer request would have timed out and been retried, and the reviewer's description now says it cannot check data against a report |
| EV-14 (the record) | Token use; a `task_id` on all 168 vision calls; queue times that would explain the 415 s `log_grid` calls and the 97–239 s looks |
| RI-27 (writer and reader) | 7 of the 11 zeros become "unknown"; boring 20 is not drawn at the origin; water rows passed as not encountered would be written as such (this turn passed none) |

---

### P3.7 The plan view and the SPT profile (question 5)

**Plan view** (`SAL_DI1_initial_plan.html`, a Plotly scatter with equal axes):
- **The axes are labelled "X (Easting)" and "Y (Northing)"** (GM-76).
  - On this grid +x points about 206° from north and +y about 116°.
  - The agent's own answer says they are not eastings and northings.
  - The plot has no north arrow.
- **Boring 20** is drawn and labelled at (0, 0), the frame corner, a place
  where no boring is (P3.5).
- **TP-1** is drawn on EB-1's spot, 141 m from its symbol.
- **The rest:** 12 of the 15 real points sit on their symbols. The plot is
  page-aligned, so it can be laid against Figure 2 by eye.
- **That comparison is the check that would have caught TP-1 at once.**
  GM-76's option of ringing each point on the report's own plan sheet (the
  grid's inverse transform) does it without the eye.

**SPT profile** (`SAL_DI1_initial_SPT.html`):
- 35 points: EGB-2 4, LB-1 7, LB-2 4, LB-3 14, LB-4 6. Every plotted N
  matches the logs.
- The title says "2019 plus EGB-2", which is honest about the 54 missing
  LBA values.
- **The three refusals are not drawn**: EGB-2 S-5 and S-6, LB-1 S-8. The
  densest material is missing from the plot, with no mark.
- **N is drawn at the top of each drive** (LB-1's first at 0.0 m), and the
  plot does not say so.
- **Two hammer types share one axis without a note**: a donut hammer in 2019,
  a safety hammer in 2026. The axis reads `N_spt`, not "field N,
  uncorrected".

---

### P3.8 The answer against the record

**True and useful:**
- the validation counts;
- the zero warning;
- that the coordinates are a drawing grid;
- that boring 20 is not plotted;
- the termination depths used over rounded headers;
- the EGB-2 / EBG-2 spelling note;
- the remaining-work list.

**Not borne out:**
- **EGB-2's sample depths** "could be read directly with adequate
  confidence". They were read off the drawn scale, as the LBA depths could
  have been.
- **The LBA depths** "must be graphically digitized". The PDF holds them
  exactly, and the helper had every value but the depth.
- **The 1987 logs** "not sufficiently reliable for bulk automated
  transcription". The tiles read them, and turn 1's transcription of the same
  logs checked 58/61 and 64/64.
- **"Source traceability"** for the 1987 placeholders. The page lists were
  made up.
- **"±0.5 to ±0.8 m"**. The median is 0.94 m and three points are 3.9–141 m
  off.
- **TP-1 and EB-1** "in the same immediate vicinity". They are 141 m apart.

---

### P3.9 Lessons and recommended actions

**Lessons:**
- **"As printed, never inferred" needs a definition of printed for drawn
  data.** On a vector log the scale is data. A depth read off it with
  geometry is a measurement, and refusing it lost 54 correct records.
- **A tool that measures must return what the writer needs.** `log_grid`
  measured every row and still left the agent to turn rows into sample
  intervals.
- **Data should travel as files, not as retyped text.** Each retyping step
  added faults.
- **Checks that compare a record with itself catch what "reads back equal"
  cannot:** a drive longer than its interval, two explorations at one point,
  a page list nobody read.
- **Earlier turns' readings must reach the helpers**, not just the primary.

Each action below is general (no rule fitted to this report). Tiers are
suggestions for the lead.

| # | what | why (evidence) | where | tier | extends |
|---|---|---|---|---|---|
| C1 | **`log_grid` returns sample intervals**: each sample's top and bottom off the drawn sample box and the log's own ruler, with the box and a confidence, and the drives inside it. Plus a compact per-log record by default, so paging cannot hide the lower half of a log | 54 SPT records withheld with blows and N in hand (54/54 right); the boxes are exact to ±0.01 m; 7 of 8 results stopped at 2.6–4.4 m and were never paged on (turns 2 and 3) | planlens `document/loggrid.py` and the toolkit spec | Next | PL-24, PL-04 |
| C2 | **Say what "as printed" means for drawn depths**: in the `write_diggs` description and the general-purpose helper prompt, a depth measured off the log's own depth scale (by `log_grid`, `measure` or a zoom against the ruler) is read off the page, not inferred. Record how it was read. A record with values in hand is written, or listed in `written.left_out` with its reason, never silently held back | the same call in two separate turns; layer contacts taken off the scale, sample boxes refused; EGB-2 kept only because a side call volunteered depths, and one of those was wrong | `funhouse_agent/adapters/subsurface_adapter.py` (description), the general-purpose prompt in `funhouse_agent/deep/agent.py` | Next | RI-28 |
| C3 | **Writer cross-checks inside an SPT**: drive lengths add up to the interval; a record that is not a refusal has as many drives as the form's increments; a drive length in a unit other than the log's depth unit is named in `assumed` | `60/15in` became a 0.381 m drive in a 0.15 m interval; EGB-2 S-5 had 0.20 m of drive in a 0.45 m interval; EGB-2 S-1 had two drives for a printed three; all passed "reads back equal" | `report_ingest/diggs_writer.py` (`_drive_sets`), the cross-checks in `_run_write_diggs` | Next | RI-28 |
| C4 | **Helpers hand records over as a file** in `write_diggs`'s own schema, and `write_diggs` takes that file, so the primary assembles by name instead of retyping | 30.7 k typed by the helper, then 29.3 k retyped by the primary (85 s and 64 s of generation); the retyping shortened descriptions, dropped recoveries, turned `60/15"` into `60/15in` and made up the 1987 page lists | `write_diggs` (a file input), the helper prompt | Next | new (with TA-21, GM-45) |
| C5 | **Cited pages are checked against pages read**: `write_diggs` in the app compares each exploration's `pages` with the pages this conversation read and lists any never read. The description says pages are where the values came from, and a placeholder has none | four 1987 page lists were even splits of the section; the logs are on other pages, which the helper's own calls named | `subsurface_adapter._run_write_diggs`, the reads record (`reads.json`) | Next | new |
| C6 | **Plan locations: checks and page points.** The plan-location tool flags two explorations at one point and any point outside the symbol it names. `render_region` returns page-point boxes beside its 0–999 boxes, so no agent converts by hand | TP-1 written with EB-1's coordinates (141 m) although the helper's own calculation was right; 3 of 15 points on labels or beside their symbols; boxes converted to page points in the model's head, then 94 `calculate` calls for one linear transform; three turns, three grids (4.7, 0.3, 0.94 m median) | planlens (PL-25's tool), `funhouse_agent/vision_tools.py` (`render_region` result) | Next | PL-25, VIS-24 |
| C7 | **Plots say what they show**: the plan view takes its axis labels from the stated coordinate system and draws a north arrow when one is stated; the SPT profile marks refusals at the axis edge and says it plots field N at the top of each drive | "X (Easting)" for an axis pointing 206° from north; the three refusals absent; N drawn at drive tops without a word | `subsurface_characterization/plots_plan.py` and the depth-plot module | Next | GM-76 |
| C8 | **The last silent zero**: total depth None when not given, and the plots' hover text with it. A placeholder for a log that prints an elevation says "not transcribed", not "unknown" | 4 of this file's 11 zero fields and the "Depth: 0.0m" hovers; borings 12–14 print elevations | `subsurface_characterization/site_model.py`, `diggs_parser.py` | Next | RI-30 |
| C9 | **Earlier turns' work reaches the helpers**: the files note lists every file in the working folder, including an interrupted turn's, and the vision tools reuse a kept reading of the same page and view without being asked, saying so | turn 3 rebuilt a grid the folder already held, a better one, from turn 2; 25 pages looked at again that earlier turns had read; 80 kept readings went to the primary only | `webapp/core.py` (the per-turn files note), `funhouse_agent/vision_tools.py` (kept readings) | Next | TA-19 |
| C10 | **Suite tasks and one live check**: SPT on a vector log whose sample depths are drawn, not printed; a refusal printed in inches on a metric log; two explorations near each other on a plan; a scanned log legible only in tiles, where the helper must hand back data. Live: re-run this retry on master and confirm the 1987 helper's kept file reaches the primary | P3.2, P3.3, P3.4; the 1987 helper's 284 k → 162 characters | `funhouse_agent/review_eval/` or the geotech eval; `module_work/LIVE_TEST_QUEUE.md` | Next | EV-15, TA-04 |

**What is not an action.**
- **Leaving boring 20 unplaced.** It is right; only the writer's 0, 0 was
  wrong, and RI-27 fixed that.
- **Holding back the 1987 values** would have been sound if they had been
  illegible. They were not: the fault was returning nothing (TA-21, done).
- **The reading quality of the vector logs.** 59/60 and 60/60 for the 2026
  SPT and 127/128 for the 2019 SPT; it is not where this turn lost data.
