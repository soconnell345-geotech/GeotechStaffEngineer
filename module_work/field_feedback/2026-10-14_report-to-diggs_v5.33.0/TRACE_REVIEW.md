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
| boring 12 | 1987, scan | 133 | 61 | 58 | 3 (elevation 871.12 for 870.12; one moisture 43.4 for 45.4; one description garbled) | 1 moisture (flagged unsure); no layers | — |
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

## 8. Lessons and recommended actions

Each action is general (no rule fitted to this report). Tiers are suggestions
for the lead; "extends" names the BACKLOG item it grows, where one exists.

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

**What is not an action.**
- **The download.** It took 182 s for 26.8 MB. It is the host's speed, and
  the turn waited on it only once.
- **The reading quality of the scanned logs.** It is not where this turn
  failed (§2.1).
