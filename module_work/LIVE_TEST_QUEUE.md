# Live checks for 5.32.0 — what to run on Funhouse, and why

**Why this list exists.** Several 5.32.0 changes were only tested with
stand-in models here, or on Foundry with a different model. These checks
confirm them with the real model on Funhouse. Each one is small. You said
(2026-10-04): "For small tests moving forward, just start keeping a list of
tests and we'll do it in funhouse in the future." — so small checks go here
instead of into a new Foundry round.

**Order.** Do the setup, then check 1 (the most important), check 2 (ten
seconds, free), then check 3 when you have half an hour. Tick each one off in
"Done" at the bottom with the date and what you saw.

---

## Setup (once)

In a Funhouse notebook, as for any new release:

```python
%pip install "geotech-staff-engineer==5.32.0"
dbutils.library.restartPython()
```

Then, in a new cell, confirm the versions:

```python
import importlib.metadata as md
print(md.version("geotech-staff-engineer"), md.version("planlens"))
# expect: 5.32.0 0.11.0
```

If pip says it cannot find 5.32.0, the Nexus mirror has not caught up yet (it
usually lags PyPI by a day or two). Either wait, or upload the two wheel files
from `foundry_handoff/` to the cluster and install them directly, planlens
first:

```python
%pip install /path/to/planlens-0.11.0-py3-none-any.whl /path/to/geotech_staff_engineer-5.32.0-py3-none-any.whl
dbutils.library.restartPython()
```

You also need your usual `fh_prompter` (and `fh_sp_client` for the SharePoint
copy) set up in the notebook, as in earlier runs.

---

## Check 1 — Do the red circles land on the tags? (most important)

**What changed.** When the app draws circles or boxes on a PDF, it now looks at
each mark afterwards and asks the model two questions: *is the thing inside
the circle?* and, new in 5.32, *is the circle drawn closely around it?* The
second question was added because, in one Foundry run, the agent "fixed"
circles that were in the wrong place by making them huge (70–125 points
across around 10-point tags) until the old check accepted them.

**What could go wrong.** No real model has answered the new question yet. If
the model judges "closely" too strictly, good circles get rejected and the
agent keeps redrawing — slow and costly — or leaves correct circles out.

**Option A — quick look in the app (no notebook).** Open the Document Review
page, upload a drawing sheet that has small tags on it, and paste:

> Circle every GCE penetration callout on this sheet in red (not the legend
> rows) and give me the marked-up PDF.

(Change "GCE penetration callout" to a tag that is really on your sheet.)
Then open the PDF it gives you. **Pass:** each circle sits snugly round one
tag, and the turn details show it wrote the file only once or twice. **Fail:**
circles missing for tags you can see, circles far bigger than the tags, or the
file written over and over.

If you have no suitable sheet handy, this makes the same synthetic sheet the
test suite uses (download it from the cluster, then upload it in the app):

```python
from planlens.testing.tag_fixtures import build_synthetic_tag_set
open("/tmp/tag_set.pdf", "wb").write(build_synthetic_tag_set().pdf)
# 7 "GCE" callouts with leaders on page 1, plus look-alikes and a legend
```

**Option B — the measured way (notebook, about 20–30 minutes, roughly
$5–10).** This runs the two markup tasks from the test suite three times each
and scores where the circles landed:

```python
from funhouse_agent.review_eval import score_review_suite

DOCS = "/Volumes/.../review_eval_docs"   # <- the folder you used for the suite run on Sept 30

res = score_review_suite(
    prompter=fh_prompter, model_name="funhouse-gpt-high",
    docs_dir=DOCS, out_dir="/tmp/check_532_markups",
    ids=["produce-circle-tags", "produce-markup"],
    arms=("baseline", ("baseline_r2", {}), ("baseline_r3", {})),   # = 3 repeats
    sharepoint=fh_sp_client)                                         # keeps a copy

print(res["results_md"])
for arm, runs in res["results"].items():
    for task, r in runs.items():
        print(f"{arm:12} {task:20}",
              "PASSED" if r["score"]["passed"] else "FAILED",
              "| times it wrote the marked PDF:",
              r.get("tool_counts", {}).get("annotate_document", 0),
              "| minutes:", round((r.get("seconds") or 0) / 60, 1))
```

**Pass:** all 6 lines say PASSED, and none wrote the marked PDF more than 3
times. **Note the minutes** too: 5.32 also limits the app to 8 image calls at
a time (it crashed Foundry when many ran at once); if any single run takes
more than about 15 minutes, tell me. **If anything fails:** send me the
printed table. The run's details are saved under
`GeotechStaffEngineer/review_eval/check_532_markups` on SharePoint, and I can
read them from there.

---

## Check 2 — Is the "find every copy of a tag" tool still available? (10 seconds, free)

**What changed.** The `find_like` tool (it finds every copy of a tag across a
drawing set) uses an image library called OpenCV. On Foundry, simply loading
OpenCV crashed the whole program, because of a government security setting
(FIPS mode). 5.32 now tries loading OpenCV in a throwaway process first, and
hides `find_like` if that process dies, so the app survives. On Funhouse
OpenCV should load fine, so the tool should still be there. This check
confirms the new safety test does not hide it by mistake.

```python
from planlens.opencv import available
print(available())            # expect: (True, '')

from funhouse_agent.deep.tools import _find_like_available
print(_find_like_available()) # expect: True
```

**Pass:** `(True, '')` and `True`. The first line may take a second or two
(that is the throwaway process). **Fail:** `(False, '...')` — send me the text
in the quotes.

The same question matters for Tiny Apps: it is worth asking CfA whether their
App Service runs in FIPS mode. If it does, `find_like` will hide itself there
and the rest of the app is unaffected.

---

## Check 3 — Report ingest: narrative rules, lab links, boring logs (about 30 minutes, roughly $5)

**What changed.** The Foundry run of 2026-10-02 found three problems, now
fixed but not yet re-measured:

1. **Narrative questions answered the opposite way from your answer keys.**
   For example, `siteResponseMention` was answered "yes" on 7 of 8 reports
   because the reports mention Site Class D and site coefficients; your keys
   say "no", meaning no site-specific site-response analysis was done. I
   rewrote the reading rules from your keys (they are still marked DRAFT for
   you to confirm, in `report_ingest/narrative_glossary.py`).
2. **Lab sheets linked to the wrong boring.** Part of it was a scoring fault
   ("LB-2" did not match "LB2"), now fixed. Whatever remains will show in this
   run's saved files, which now record what the model wrote.
3. **Boring logs:** recovery printed as a length (cm) was never counted.
   Fixed.

This rerun measures all three with real GPT-5.4. Use the same folders as your
September report-ingest runs (change the paths below if yours differ), and a
NEW output folder:

```python
from report_ingest.cluster_scoring import score_on_cluster

score_on_cluster(
    reports_dir = "/Volumes/main/geotech/reports",            # <- your reports folder
    manifest    = "/Volumes/main/geotech/wp1b/MANIFEST.md",   # <- the manifest
    truth_dir   = "/Volumes/main/geotech/wp1b/truth",         # <- the folder holding logs/, lab/, narrative/
    di_dir      = "/Volumes/main/geotech/report_di",          # <- the DI results
    out_dir     = "/tmp/check_532_stage_d",                    # NEW folder
    stages      = ("logs", "lab", "narrative"),
    prompter    = fh_prompter,
    model       = "funhouse-gpt-high",
    sharepoint  = fh_sp_client,                                # keeps a copy
)
```

It prints where `RESULTS.md` was written. **What to look for** (the Foundry
numbers to beat are in brackets):

- **Narrative section**, "Per question" table: `siteResponseMention` right on
  most of the 8 [was 0 of 8]; `soilCorrosion` better [was 2 of 6]; overall
  recall at least 75 % [75 %].
- **Lab section**, the "open" table: `link` well above 55 % [55 %; the blind
  set was 100 %].
- **Logs section**, the "blind" table: `recovery` above 53 % [53 %].

**Either way, send me the `RESULTS.md`** (or tell me the SharePoint folder,
`GeotechStaffEngineer/report_ingest/check_532_stage_d`). Unlike the Foundry
files, this run's per-item files record what the model actually wrote, so I
can see exactly what is still wrong.

---

## Check 4 — How far off are the vision model's positions? (about 15 minutes)

**Why.** The circles missed because the vision model reports positions on a
whole page as a shrunken copy of the truth (the location review,
`module_work/harness_theory/locating_things_on_a_page.md`), and because
Funhouse appears to shrink any image wider than 2,048 px before the model
sees it. This cell measures both directly, without the agent: it sends the
synthetic tag sheet to GPT-5.4 at several sizes and zooms and compares the
positions it reports with the true ones.

**How.** Paste `location_measurement_cell_ready.py` (sent in chat; the same
code is in §5 of the location document) into one notebook cell with
`fh_prompter` set up. About 17 model calls. **Send me the printed output**
(it also saves a JSON file in the notebook's folder). The results decide how
the next release sizes its whole-page looks and its zooms.

## Re-run of check 1 on the 5.32.1rc2 test wheel (after the 2026-10-07 fix)

Upload `geotech_staff_engineer-5.32.1rc2-py3-none-any.whl` (sent in chat;
not published; it replaces rc1 — rc2 also keeps the model's own words in
each run's `activity.jsonl`, so the re-run can be reviewed in full) to the
cluster, then:

```python
%pip install /path/to/geotech_staff_engineer-5.32.1rc2-py3-none-any.whl
dbutils.library.restartPython()
```

**After any check:** send the run folder (the zip, or the SharePoint folder
name). Every run's full record is read, not just the scores (CLAUDE.md,
"REVIEW EVERY MODEL RUN IN FULL").

Run the Option B cell again with a NEW `out_dir`,
`"/tmp/check_532_markups_rc1"`. `sharepoint=fh_sp_client` now works directly
(the copy fix is in the wheel), and the notebook cell for viewing the PDFs
works as before with the new folder name. **Pass:** most of the 3
`produce-circle-tags` runs pass, and none withholds a file whose circles you
can see sit on the tags.

## Done

- **2026-10-07, check 4 (location measurement), Funhouse GPT-5.4.** Funhouse
  caps images at 2,048 px (confirmed by token counts). Whole-page positions
  on the 0-999 grid were 57–91 pt off with a scale that changed between
  identical calls; the SAME model's pixel positions, converted with the true
  image size, were within ~1–6 pt; zooms 0.2–4 pt. Next build: ask for pixel
  positions on images ≤ 2,048 px and convert in code. Details:
  `module_work/harness_theory/locating_things_on_a_page.md` §5.1.

- **2026-10-07, check 2 on 5.32.0 (Funhouse): the guard WORKED, and Funhouse
  is a FIPS host.** `available()` returned `(False, 'OpenCV cannot load on
  this host (a test import exited with code -6: crypto/fips/fips.c:154:
  OpenSSL internal error: FATAL FIPS SELFTEST FAILURE)')`; `find_like` is
  hidden. So OpenCV — and with it `find_like`, the raster-drawing leg and OCR
  — has never been usable on Funhouse (before 5.32.0 the first `find_like`
  would have killed the process), and almost certainly not on Tiny Apps.
  Follow-up on the to-do list: `find_like` without OpenCV.

- **2026-10-07, check 1 Option B on 5.32.0 (Funhouse, GPT-5.4): FAILED, cause
  found and fixed.** `produce-markup` passed 3/3. `produce-circle-tags` failed
  3/3 (0, 1 and 2 of 7 tags marked). Two causes:
  1. The new "drawn closely round it?" question was too strict on GPT-5.4. A
     19 × 14 pt ring centred on a 10 × 4 pt tag was called "drawn far wider",
     so the agent (correctly, by its rules) withheld good work. Fixed in
     cc697e6: size is now measured (the look returns the thing's box; a mark
     is too wide only above 30× its area).
  2. Circles placed from a whole-page look were 40–80 pt off. The check
     rightly rejected those. The run that zoomed first placed its circles
     within 1–13 pt.

  Also found: the SharePoint copy silently copied nothing (fixed in 4334fce;
  the workaround `Mirror(sharepoint=fh_sp_client.file_manager, ...)` uploaded
  all 24 files). Two answers wrote "tool page 0" beside "PDF page 1", which
  the suite flags.
