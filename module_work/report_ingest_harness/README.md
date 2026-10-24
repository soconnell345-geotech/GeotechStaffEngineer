# Report-ingest corpus harness (WP0)

Dev-only tooling for the report-ingest train (`module_work/REPORT_INGEST_PLAN.md`).
It opens the 38-report corpus by ID, attaches an Azure Document Intelligence
result only to the pages that need one, loads the hand-labelled page types, and
prints the per-work-package measurements. It is **not shipped**: the wheel's
`[tool.setuptools.packages.find]` include-list names every package that goes in
and `module_work` is not one of them, so this folder is source-tree tooling like
`module_work/drawing_ground_truth/doc_claims_check.py`. Run it from the repo
root with the app venv, which carries the editable `planlens` checkout:

```
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp0
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp0 --only R36 --no-append
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1_labels
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp2b_logs --grid-only
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp3_lab --tables-only
.venv/Scripts/python -m pytest module_work/report_ingest_harness/tests -q
```

`measure_wp2b_logs` and `measure_wp3_lab` are the reader scorecards, and each
runs its DETERMINISTIC half with no engine, no credential and no network:
`--grid-only` scores what `log_grid` alone recovered, `--tables-only` scores
what a page's own detected tables hold. Those are the honest baselines and are
worth re-running whenever planlens changes. With an engine they run the reader
as well and print both columns side by side; the numbers that count come from
`report_ingest.cluster_scoring.score_on_cluster`, because the app runs against
OpenAI models through Prompter and a score measured on any other model measures
a model that will never do the work.

`lab_truth_records.py` turns the 31 hand-truthed laboratory sheets into record
objects, and reads their keys through `report_ingest.lab_scoring` rather than a
second copy — the scorer ships in the wheel and runs on the cluster, this
converter is repo-only, and the two have to read a truth file the same way or
the DIGGS gate and the scorecard would be measuring different files.

`measure_wp1_labels` is WP1's scorecard: it runs planlens'
`document.roles.page_roles` over every report that has hand labels and scores
it against them — precision, recall, F1 and support per role, the confusion
matrix, each report's accuracy, and every miss on the five GATED roles
(`boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`, which
must reach 0.90 on both rates). It appends a round to the ledger each run,
with the gated numbers of every round so far, so the effect of a fix is
visible; `--note` says what changed. **The ledger's miss lines carry the
evidence planlens recorded, never the page's heading** — the largest type on
a log or a laboratory sheet is a firm's title block, and the ledger is
tracked. The same misses WITH headings go to `raw/checks/wp1_misses.txt`,
which is gitignored.

`measure_label_review <ingest stage folder>` (BACKLOG RI-15) reads what a
cluster ingest run left in each `<ID>/labels.json` and `review.json` — the
vote's labels and every change the label review made — and counts, against the
spreadsheet's hand labels, how many changes fixed a page, broke one or moved a
wrong label to another wrong one: per report, by the evidence the change named,
by whose label it took, by the vote's label and by the move, then again under
the rule that refuses a change naming no look at the page. It runs no model,
refuses any report in the blind set before reading it, and never reads a
change's reason. `--raw-dir` points it at the private corpus when it is run
from a worktree.

`run_claude_local` (2026-10-24) runs one reader stage -- `logs`, `lab`,
`calc`, `narrative` or `soundings` -- through `score_on_cluster` on Claude
over the Anthropic API, locally, against the same hand-truthed items the
Foundry runs read: a `ChatAnthropic` goes in as `chat_model` (RI-67), so the
calls take the `ChatModelEngine` path the Foundry Claude run takes. Every
call is metered by `module_work/live_smoke/spend.py` -- a named wave with a
cap on the one ledger, the meter on the model object, the next call refused
once the cap is spent -- and the run appends a block to `RESULTS.md` setting
the ledger beside the run files' own cost block, with the stop reasons,
failed requests, SDK retries, the pictures each call carried and how many
values carry a page and a box. `--sets open|blind` and `--only <item or
report>` pick items; `--env NAME=VALUE` sets a switch (every
`GEOTECH_INGEST_*` is cleared first); the out_dir defaults to the main
checkout's gitignored `module_work/field_feedback/2026-10-24_claude_local/raw/<arm>/`
and resumes. `check <out_dir>` reads any out_dir back for $0, a Foundry one
included. The offline test drives it on a made-up printout with a scripted
model.

**The corpus is private and this repo is public.** Every PDF, DI result, hand
label and derived cache lives under
`module_work/field_feedback/2026-09-16_report-ingest/raw/`, which `.gitignore`
excludes (`module_work/field_feedback/**/raw/`). Nothing tracked by git may name
a report, a project, a place, a firm or a person from it — reports are R01–R38
here and in everything the harness prints. `ReportInfo.private_name` holds the
source stem because the label sheets have to be matched to IDs, and it is the
one field `repr()` leaves out; the sheet-to-ID map is derived at run time and
cached into `raw/labels_map.json`. When the corpus is absent — CI, any other
machine — the loaders raise a clear error and their tests skip; the label-mapping
tests are synthetic and always run.

`corpus.open_report(rid, di=...)` has three modes, because planlens' `text_source`
**replaces** the PDF text layer on every page it covers rather than filling gaps:
`"none"` is the text layer alone, `"all"` hands every covered page to DI, and
`"auto"` (the default) attaches DI only to the pages `pages_needing_ocr` names —
image-only pages plus pages whose text layer is present but unreliable. On R28
that is 28 pages of 455 rather than all 455, at a sixth of the wall clock. The
module docstring carries the measurement.

## Label mapping

`labels.RAW_TO_LABEL` maps the 23 `page_type` strings the spreadsheet uses onto
the 18-label vocabulary in `labels.LABELS`. It is total over those 23 and an
unknown string raises `UnknownPageType`, so no page falls quietly out of a
score. Change a row here rather than in the spreadsheet.

| spreadsheet `page_type` | label |
|---|---|
| main report narrative | `narrative` |
| figure | `figure` |
| test location plan | `plan` |
| subsurface profile | `profile` |
| boring log | `boring_log` |
| test pit log | `test_pit_log` |
| cpt log | `cpt_log` |
| dcp log | `dcp_log` |
| lab testing | `lab_test` |
| subsurface drainage testing | `field_test` |
| calculation | `calculation` |
| appended report | `appended_report` |
| field photos | `photos` |
| appendix cover page | `divider` |
| figures or tables cover page | `divider` |
| figures or tables cover sheet | `divider` |
| figures or reports cover page | `divider` |
| cover page | `cover` |
| cover letter | `letter` |
| table of contents | `toc` |
| informational appendix content | `other` |
| other or unknown | `other` |
| other appendix table | `other` |

`field_test` and `letter` are additions to the plan's list: the hand labels
distinguish infiltration and percolation results from laboratory testing, and a
cover letter from a cover page, and a vocabulary that could not express the
difference would score both as misses.
