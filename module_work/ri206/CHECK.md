# RI-206 and DOC-16: check

Branch `claude/ri-206-resume-cache-tokens`, made from master `6f88a83`.

## (1) RI-206: a resumed run counted fewer tokens in than it paid

**Cause.** A reader's own `cost` block leaves out its cache writes. During the
run `_Spend.settle` tops this run's count up from the `_Tally`, which saw every
reply's usage in full. But the item file saved for a resume held the reader's
own short `cost`. On a resume `_Spend.reuse` read that file back. It already
summed all three input parts (`tokens_in`), but the file had no cache writes to
sum, so `document.resumed.report_input_tokens` (the record's and the summary's
whole-report figure) came out short. `reuse` itself was right. The saved file
was what was wrong.

**Fix** (`report_ingest/graph.py`): a new `_paid(blob, spend, mark)` returns
the blob to save with its cost raised to what the tally saw since the step
began. That covers calls, all three tokens-in parts, output tokens and
dollars. `_keep` saves through it, so every reader item (today's and model
first), the ranges, vision, review and identity steps are covered, and so is
triage's `triage.json`.
- A count is raised only where the tally saw MORE.
- A cost that left nothing out is saved exactly as before, unchanged. This
  keeps the model-first-off snapshot (`graph_off_snapshot.json`) byte for byte.
  A first version that always wrote every key changed `bound1/identity.json`'s
  digest and was reworked.
- The in-memory blob is never mutated, so this run's own count still comes from
  `settle` with no double count.

**Test:** `report_ingest/tests/test_ri206.py`, two tests with made-up usage (200
input, 3,000 cache write, 700 cache read, 100 output per call) on the synthetic
report.
- A resumed run's `report_input_tokens`, `saved_input_tokens`, the written
  record and the summary line all equal what the first run paid.
- Every item file carries each part.
- Both fail before the fix (12,600 against 54,600 paid: the cache writes were
  missing) and pass after.

## (2) DOC-16: `report_ingest/README.md` was out of date

Rewritten from the code (`graph.py`'s docstring and `model_first_stages`,
`model_first.py`, `model_first_pages.py`, `model_first_read.py`, `locate.py`,
`range_labels.py`, `checks.py`, `triage.settle_workflow`, `_bound_workflow`):
- **Intro:** "Calculation printouts are WP5" is gone. A new paragraph says model
  first is the default for every stage, how a stage is set, and that `off` is
  today's pipeline byte for byte.
- **Piece table:**
  - new rows for the range labeller, the model-first pages, schemas, readers,
    placement and checks, the calculation reader, the sounding reader, the
    calc and sounding scorers and the library;
  - today's readers are marked as the path for a stage set off;
  - the narrative row is no longer "one call".
- **Record section:** "`CalcEntry` is still WP5" now says calculations are
  `Calculation` on `ReportRecord.calculations`, and `calcs` is a load-only stub.
- **Graph section:**
  - the flow names both labelling paths;
  - a new "Model-first reading" paragraph covers the setting, the grouping, one
    call per group, halving, `checks="flag"`, placement and
    `items/mf-<key>.json`;
  - the narrative paragraph adds the one-call, text-first model-first
    narrative;
  - the workflow prose is now a table that matches the code: `partial` and
    `multi_document` run everything; `scanned` skips only today's text-only
    narrative reader, and model first sends those pages as pictures;
    `appendix_only` skips the narrative on both paths; a bound document's
    workflow comes from its own labels;
  - "calcs recorded as skipped" is replaced: they are read.
- **Bound section:**
  - "parent keeps appended_report" now says the parent keeps its own labels
    and hands the pages over;
  - "counts from the parent's own logs" now says `counts_from_logs` is off by
    default, and the parent's own logs count only when it is set;
  - a sentence on how a bound document is labelled under model-first labels.

The README's notebook cells were not touched; the two tests that check the
cells' keywords pass.

## Tests (Python 3.13, planlens 0.14.0, `pip install -e .`)

- `python -m pytest -q -n 4 report_ingest`: 4,373 passed, 8 skipped (pydiggs not installed), 0 failed; exit code 0. Run with the graph fix and the new test in place; the README edits came after, and the README tests (`test_cluster_scoring.py -k README`, 2 passed) were re-run on them
- `python -m pytest -q webapp/tests/test_public_text.py webapp/tests/test_page_fingerprints.py`:
  19 passed, 0 failed.
- Focused, before the full run (graph, graph model first, RI-125, RI-206):
  166 passed, 0 failed.

No model prompt text changed; `READ_PROMPT_VERSION` is untouched.

## Noted, not fixed

- Old item files saved before this fix still lack their cache writes; a resume
  over such a folder still under-counts. They cannot be repaired from what is
  on disk. Only a fresh read rewrites them.
- The module docstrings of `model_first.py`, `model_first_pages.py` and
  `model_first_read.py` still say "nothing in ingest calls this module yet" and
  "only under `=on`", which RI-186 made stale. Not a saved-output value.
- The README's long cluster-run sections still carry old "WP5" headings. These
  quote the RESULTS.md headings `cluster_scoring.py` still prints
  (`# WP5 on the cluster: ...`), so they match the code and were left alone.
