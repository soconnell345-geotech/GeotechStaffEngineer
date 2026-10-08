# Live checks — what to run on Funhouse, and why

**Why this list exists.** Some changes can only be confirmed with the real
model on Funhouse: offline tests use stand-in models, and Foundry runs a
different model. Each check here is small. You said (2026-10-04): "For small
tests moving forward, just start keeping a list of tests and we'll do it in
funhouse in the future." So small checks go here instead of into a new
Foundry round. Tick each one off in "Done" at the bottom with the date and
what you saw.

**After any check:** send the run folder (the zip, or the SharePoint folder
name). Every run's full record is read, not just the scores (CLAUDE.md,
"REVIEW EVERY MODEL RUN IN FULL").

**Moved to Foundry, and DONE there (2026-10-07).** Your Funhouse tokens ran
out for the month, so checks 6 and 3 ran on Foundry as brief 4 (see Done).
Their instructions below stay as the Funhouse version, in case either needs
repeating there.

**Still to run on Funhouse: check 5 only, after 5.32.1 is released.** It
tests the Funhouse app's own SharePoint and sign-in connections, which
Foundry does not have. It tests connections rather than the model, so run it
on the cheaper model (`funhouse-gpt-medium` in the sidebar) with a short PDF
of 2–5 pages: about 50–150 thousand tokens in all.

**Token estimates from now on.** Every check states its tokens, not just its
minutes; "small" was wrong before. One run of the circle task reads 160–360
thousand input tokens, and check 6 runs it six times.

Checks 1, 2 and 4 are done (see "Done").

---

## Setup (once per release)

5.32.1 and planlens 0.12.0 are on PyPI (2026-10-08); install them when
they reach the Nexus mirror, a day or two after PyPI. Until then, use the
wheels in `foundry_handoff/`.
In a Funhouse notebook:

```python
%pip install "geotech-staff-engineer==5.32.1"
dbutils.library.restartPython()
```

Then, in a new cell, confirm the versions:

```python
import importlib.metadata as md
print(md.version("geotech-staff-engineer"), md.version("planlens"))
# expect: 5.32.1 0.12.0
```

If pip says it cannot find 5.32.1, the Nexus mirror has not caught up yet (it
usually lags PyPI by a day or two). Either wait, or upload the two wheel
files from `foundry_handoff/` to the cluster and install them directly,
planlens first:

```python
%pip install /path/to/planlens-0.12.0-py3-none-any.whl /path/to/geotech_staff_engineer-5.32.1-py3-none-any.whl
dbutils.library.restartPython()
```

You also need your usual `fh_prompter` (and `fh_sp_client` for the SharePoint
copy) set up in the notebook, as in earlier runs.

---

## Check 6 — Do positions now come back right, and do the circles land? (after the location fix; about 45 minutes)

**Why.** Check 4 showed GPT-5.4's positions on the 0-999 grid were 57-91 pt
off a whole sheet, while its positions in PIXELS were within a few points,
and that Funhouse shrinks every image to 2,048 px. The location fix
(`module_work/harness_theory/locating_things_on_a_page.md`, "What changed")
makes five changes:
- the app asks for pixel positions and converts them itself, with the size
  it sent;
- no image is sent larger than 2,048 px, so small lettering is tiled
  automatically again;
- `tiles="6x6"` is no longer silently ignored;
- a zoom on a reported position is wide enough to hold the thing;
- planlens refuses a mark placed from a whole-page look (zoom first), and a
  comment by quote on a CAD notes column lands on the quoted line.

**Setup.** As in "Setup" above (5.32.1 with planlens 0.12.0).

**Step 1: the measurement, without the agent (about 23 model calls, ~15
minutes).** Copy the cell in §5.2 of the location document into one notebook
cell with `fh_prompter` set up, and run it. **Pass:**
- the profile line ends "the host delivers at most 2048 px";
- `page-as-shipped` is within a few points on all three repeats, with
  y scale near 1.0;
- the tool line shows `tiles: 9`;
- both zoom rows say "window holds the tag".

**Send me the printed output** (it also saves a JSON file in the notebook's
folder).

**Step 2: the circle task again (about 20-30 minutes, roughly $5-10).** This is the
measured circle task from check 1, in a NEW folder:

```python
from funhouse_agent.review_eval import score_review_suite
DOCS = "/Volumes/.../review_eval_docs"   # <- the same folder as check 1

res = score_review_suite(
    prompter=fh_prompter, model_name="funhouse-gpt-high",
    docs_dir=DOCS, out_dir="/tmp/check_locations_fix",
    ids=["produce-circle-tags", "produce-markup"],
    arms=("baseline", ("baseline_r2", {}), ("baseline_r3", {})),
    sharepoint=fh_sp_client)
print(res["results_md"])
```

**Pass:**
- `produce-circle-tags` passes on at least 2 of the 3 runs;
- in each marked PDF the circles sit on the tags;
- the agent zoomed before it circled. If it tried to circle from a
  whole-page look, the tool refused with "Zoom on the thing first", which
  is fine, as long as it then zoomed.

Also open the three `produce-markup` PDFs: the comment's arrow should point
at the line "RAMP SLOPE CANNOT EXCEED 8.33% MAX." (the bottom line of
note 4), not beside note 3.

**Send:** the printed table and the SharePoint folder name
(`GeotechStaffEngineer/review_eval/check_locations_fix`). Every run is read
in full.

## Check 5 — SharePoint links, remembered files, past conversations (after 5.32.1; in the app, about 20 minutes)

**Why.** These come from your geotech-page session of 2026-10-06.
- **The pasted link was fine.** In one turn every SharePoint call failed for
  a few seconds, almost certainly an expired sign-in, and the tool reported
  that as "not found".
- **The agent "forgot" the file.** It cannot see earlier turns' tool results,
  only its own answers.
- **The geotech page listed the review page's past conversations.**

All three are fixed in 5.32.1. This check confirms the fixes live.

**How.** Start the app with SharePoint set up as usual. Then, on the
**geotech page**:

1. Paste a SharePoint "copy link" to a PDF and ask for a short summary.
   **Pass:** it downloads and summarises.
2. Leave the app open for more than an hour (the sign-in renewal now checks
   every minute), then ask a follow-up about the same PDF. **Pass:**
   - it answers from the file it already has, without "I no longer have
     it";
   - "Turn details" (now on by default) shows the turn started with a note
     listing that file.
3. If a SharePoint call does fail, the message should now say SharePoint
   refused or is not answering, not "file not found".
4. Open "Find a past conversation" on the geotech page, then on the
   Document Review page. **Pass:** each lists its own conversations.

**Send:** the conversation's SharePoint folder name. `activity.jsonl` now
records the note and any SharePoint error code.

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

## Check 7 — Whole 32-px patches: does GPT-5.4's small y stretch go? (next release; about 30 minutes, ~130 thousand input tokens)

**Why.** In brief 4 (part A) GPT-5.4's pixel boxes came back stretched in y
by 1.4–1.7 % and not at all in x. Each stretch matched the image's height
rounded UP to whole 32 px patches (1344 / 1325 = 1.014), while the width
(2,048) already was a whole number of patches. If GPT-5.4 works in a frame
of whole patches, an image that already is one should remove the stretch:
up to ~10 pt at the bottom of an 11 × 17 sheet, under 4 pt in a zoom. This
is a **hypothesis to test, not a fix**: the next release carries a switch,
`GEOTECH_VISION_PATCH_ALIGN`, OFF by default, that pads every image with at
most 31 px of white paper on the right and bottom so both sides are whole
patches (the view is widened to match, so conversions stay exact).
TRACE_REVIEW §7 item K; code `funhouse_agent/vision_view.align_to_patches`.

**Where.** GPT-5.4 (`funhouse-gpt-high` on Funhouse, or GPT-5.4 on Foundry).
Sol showed no stretch, so it needs no run.

**Setup.** The release (or test wheel) that carries the switch.

**Step 1: as shipped (switch off).** Run the §5.2 cell of
`module_work/harness_theory/locating_things_on_a_page.md` unchanged. Note the
`y scale` of the three `page-as-shipped` rows and of the `tool-tile` rows,
and T7's error (the lowest tag).

**Step 2: the switch on.** In a new cell run

```python
import os
os.environ["GEOTECH_VISION_PATCH_ALIGN"] = "1"
```

then run the §5.2 cell again, unchanged (it does not clear this setting),
and afterwards `os.environ.pop("GEOTECH_VISION_PATCH_ALIGN")`.

**What to read.**
- `page-as-shipped` should now show both sides a multiple of 32: `sent
  (2048, 1344)` where step 1 showed `(2048, 1325)` (checked offline: the
  cell runs unchanged with the switch on).
- **Hypothesis holds:** its y scale is within about 0.003 of 1.000 on all
  three repeats (was 1.014–1.017), the tool-tile rows likewise (was up to
  1.011), and T7's error falls to a few points (was 8.7–11.5 pt).
- **Hypothesis fails:** the y scale stays near 1.015. Then the switch stays
  off and the stretch is not about patches.

**Send:** both printed outputs and both saved JSON files
(`location_remeasure_*.json`).

## Check 8 — Visual scales: label crops, `measure`, the chart's second reading, report ingest (next release; on Foundry as part of brief 5)

**Why.** W3's app side (`module_work/VISUAL_SCALES_DESIGN.md` §12) has only
been tested offline, with stand-in models.
- The agents now have `measure` and `log_grid`.
- A scan's label values are read by ONE vision call over numbered crops.
- `read_reference_figure` returns a value measured from the chart beside
  the vision estimate.
- Report ingest can vote with measured layer tops, check a grading plot
  against its table, and read a digitised sounding again. This one is
  behind `GEOTECH_INGEST_VISUAL_SCALES`, OFF.

Nothing here has met a real model. Steps 1-3 are small. Step 4 is a cluster
run and belongs in brief 5.

**Setup.** The release (or test wheel) that carries W3's app side, with its
planlens.

**Step 1: the model reads numbered label crops** (no agent; about 3 calls
per model, under 20 thousand input tokens). Run once with
`funhouse-gpt-high` (GPT-5.4) and, on Foundry, once with Sol.

```python
from planlens.testing.visual_scale_fixtures import (
    LogVariant, PlotVariant, build_log, build_plot, build_chart_family, _overlap)
from planlens.document.scalefinder import find_scales
from funhouse_agent import scale_labels
from funhouse_agent.deep.databricks_bridge import PrompterChatModel
from funhouse_agent.deep.vision_engine import LangChainVisionEngine
engine = LangChainVisionEngine(PrompterChatModel(prompter=fh_prompter,
                                                 model="funhouse-gpt-high"))
for fx in (build_log(LogVariant("lt_log", skew_deg=0.4, seed=91)),
           build_plot(PlotVariant("lt_grad", "grading", raster=True,
                                  skew_deg=0.3, seed=52)),
           build_chart_family("loglog", raster=True, seed=72)):
    doc = fx.open()
    items = [(0, b) for s in find_scales(doc, 0).needing_values()
             for b in s.label_boxes]
    texts, info = scale_labels.read_labels(doc, items, engine)
    want = [max(fx.labels, key=lambda lb: _overlap(b, lb.box)).text
            for _p, b in items]
    right = sum(t == w for t, w in zip(texts, want))
    print(fx.name, f"{right}/{len(items)} right,",
          f"{info['unreadable']} unread,", "wrong:",
          [(w, t) for t, w in zip(texts, want) if t not in (w, None)])
```

**Pass:**
- every label is right, or at most one is unread per sheet;
- no label is wrong. A wrong one is dropped by the fit if it is alone, and
  refused if there are two, but say which.

**Step 2: `measure` from a look, in the Document Review app** (one turn,
about 100-300 thousand tokens).
- Upload the scanned log fixture as a PDF: `open("lt_log.pdf","wb").write(build_log(LogVariant("lt_log", skew_deg=0.4, seed=91)).pdf)` in a notebook, then download it.
- Ask: "At what depth does the second layer start on this boring log, to
  the nearest centimetre?" The true value is 2.26 m.
- Look in `activity.jsonl` for:
  - whether the agent called `measure`, which it knows only from the tool's
    own description;
  - with what (`bbox`, or a zoom's `view` + `image_box`);
  - whether one label-reading call was made;
  - whether the answer is 2.26 m within the result's +/-.

  Not calling it is a finding, not a failure: it is what the suite tasks in
  brief 5 are for.

**Step 3: the chart's second reading** (on the geotech page, two
questions, about 50-100 thousand tokens).
- Ask for the Nordlund toe-resistance limit off GEC-12 Figure 7-15 at
  phi = 32 deg and at 42 deg.
- Ask for Kp off DM7.2 Figure 4-12 for phi' = 35 deg, theta = 10 deg,
  delta/phi = 0.66.
- In each `read_reference_figure` result, read `code_reading`:
  - did the vision call write `READ` lines with `px=` boxes;
  - did code measure (`status: measured`);
  - do the two agree.

  At 32 and 42 deg, design E8 expects code near 16 tsf and 296 tsf. That is
  the re-verification owner decision 8 asked for: it reads the chart, it
  does not settle it.

**Step 4: report ingest with the setting on and off** (cluster, brief 5).
- `score_on_cluster(stages=("logs", "lab", "ingest"), ...)` twice, each into
  its own `out_dir` (item files resume, so a shared folder would mix the
  two runs):
  - once as shipped;
  - once inside `from report_ingest.visual_scales import use_visual_scales`
    / `with use_visual_scales():`, or with
    `os.environ["GEOTECH_INGEST_VISUAL_SCALES"] = "1"`.
- Read:
  - log `layer_top` before and after, against the hand truth;
  - the label calls' cost;
  - the `plot_vs_table` entries against the lab truth;
  - the sounding counts (agree, disagree, several traces).
- "Off" should match 5.32.1, except where planlens' new ruler moves the
  depths of scans with Azure DI text (design §12, departure 6).

**Send:** the printed outputs, the conversation folders (steps 2-3) and the
two RESULTS.md (step 4). Every run is read in full.

## Done

- **2026-10-07, checks 6 and 3 on Foundry (brief 4, 5.32.1rc3; GPT-5.4 and Sol).**
  - **The location fix held:** whole-page pixel boxes were a median 3.5–19 pt off on GPT-5.4 and 1–1.5 pt on Sol. Every ring was placed from a zoom.
  - **The circle task** passed 2 of 3 on GPT-5.4 (0 of 3 on Funhouse before) and 3 of 3 on Sol.
  - **The full suite on Sol** passed 37 of 37.
  - **Report ingest** narrative recall went from 75 % to 83 %, and `siteResponseMention` from 0/8 to 8/8.
  - **Next:** the markup check cannot yet confirm review comments, a fix for the next release.
  - **Write-up:** `module_work/review_eval_results/2026-10-07_foundry_5.32.1rc3/TRACE_REVIEW.md`.

- **2026-10-07, check 1 re-run on the 5.32.1rc1 test wheel (Funhouse,
  GPT-5.4): still failing, and it located the real cause.**
  - `produce-markup` passed 3/3.
  - `produce-circle-tags` failed 3/3, with 0, 5 and 0 of the 7 tags marked.
    The run with 5 had all 5 of its circles on tags.
  - The measured size check worked. The third run's 8 circles were all
    rejected; the file it left was an unchecked draft, not circles the check
    accepted.
  - Circles were still placed from whole-page looks, tens of points off.

  That led to check 4 and the location fix, which check 6 re-measures. The
  rc2 re-run was dropped in favour of check 6. Details:
  `module_work/harness_theory/locating_things_on_a_page.md` §3.

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
