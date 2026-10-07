# Coverage in code, and the report-review checklist (W4, built 2026-10-08)

Plan of record: `module_work/SCALES_COVERAGE_CROSSCHECKS.md`, item W4.
Why: the 2026-10-06 field session
(`module_work/field_feedback/2026-10-06_geotech-report-session_v5.32.0/FINDINGS.md`,
P3 and §8). Asked for a report's subsurface data, the agent read 10 of 23
classification sheets, none of the chemistry, compaction or density sheets,
no lab results for one boring, and skipped the newest logs because it took
"boring log" to mean "scanned page". The prompt already said to look at every
page. Your direction: enforce coverage in code, generally, and draft a
checklist as one data file used two ways.

Everything here is behind switches that are **OFF by default**. Nothing
changes for users until the review suite has measured it and you say so.

## What was built

**1. An inventory of the document, in code.** When a document is first
touched, the app lists every page with what it is: exploration log,
laboratory sheet, field test, plan or figure, calculation, text, a page of a
report bound inside, photographs, other, and the cover, contents and tabs.
This comes from the document's own structure through planlens' page roles,
the same rules report ingest builds its work items from. Each page also
records whether it has a usable text layer.

**2. A ledger the app keeps, not the model.** After every tool call, by the
main agent or any helper, the app works out from the call itself which pages
were read:

- **Looked at:** a page or zoom tool (`analyze_pdf_page`, `render_region`).
- **Read as text:** a text tool (`read_document`, `read_pdf_text`) on a page
  that has a text layer. A scanned page "read" as text was not read, because
  there was nothing to read, and the ledger says so.
- **Not reads:** searches, thumbnails and `find_like`.

The agent can also mark pages **extracted** or **skipped**, with a reason.
When data are written with `write_diggs`, the pages its data cite are marked
extracted automatically. The ledger is kept in the conversation's folder as
`coverage.json`, beside `activity.jsonl`, so it outlives an agent rebuild and
goes to SharePoint with the rest of the conversation.

**3. Coverage as counts.** For example: "laboratory sheets 10 of 14 read (not
read: PDF pages 28-30); exploration logs 7 of 7 read". It also flags pages
marked extracted that no tool ever read.

**4. The gate.** When the agent tries to finish a turn that took data out of
a document, and data pages that nobody read remain, it is told **once**, with
the list. It may then read them, mark them skipped with a reason, or say in
its answer why not. It is also told to state coverage as counts. A turn
counts as taking data out of a document in three cases:

- **It said so.** The agent opened a coverage task with `document_coverage`.
  Every page but the cover, contents and tabs then counts.
- **It wrote data out.** The turn called `write_diggs` after reading the
  document. Every page but the cover, contents and tabs then counts.
- **It read data pages.** The turn read any log, lab or field-test page.
  Every log, lab and field-test page then counts. This is the field session's
  failure: some old logs and some sheets were read, and the rest of exactly
  those groups were missed.

The gate never loops. It speaks once per user turn, and an auto-continue
belongs to the same turn. It stands down rather than risk the turn's answer
when fewer than 12 graph steps are left, or on a lean agent's last budgeted
model call.

With the switch on, the page's step cap is raised to at least 80 (the
default is 50), to leave room for the reads the gate asks for. The note the
model was given is written into `activity.jsonl` as a `coverage_gate` event,
so a review of the run can see it.

**5. Two tools**, which the agent learns only from their own descriptions. No
prompt names them.

- `document_coverage` gives the inventory, marks pages, and returns the
  counts to state.
- `report_checklist` gives the checklist.

**6. The checklist, as data:** `funhouse_agent/report_review_checklist.json`,
marked **DRAFT**. It is your first draft from the plan, item for item. Each
item is either `code` or `judge`.

- **Code items** are run by `funhouse_agent/report_checklist.py`:
  - every log page and lab sheet was read (from the ledger);
  - explorations named in the text and on the plan against the logs (from
    the text layer; when a log is a scan, it asks for a look instead of
    guessing);
  - lab sheets against the logs;
  - every summary row has a sheet, and every sheet is on the summary;
  - summary values against the sheets, and units (the reconciler's
    cross-checks, which `write_diggs` now runs);
  - every value written cites a page, and every cited page was read.
- **Judge items** are handed to the agent to report against. So is a code
  item that cannot run in that conversation, for example before any data are
  written. Nothing on the list is dropped in silence.

With both switches on, a failed code check also joins the gate's note.

## The switches and where they act

| Switch | What it turns on |
|---|---|
| `GEOTECH_COVERAGE=1` | the ledger, `document_coverage`, the gate |
| `GEOTECH_REVIEW_CHECKLIST=1` | `report_checklist` (and the ledger it needs) |

Both act on both pages: the geotech page, and the Document Review page's
default agent and its lean agent, with their helpers. They do not act on the
looking-only `minimal` measuring stick.

## The suite

Three new tasks are in `funhouse_agent/review_eval/`. They run on a
synthetic 30-page report (`review_eval/report_fixture.py`) with the shape of
the field session's report:

- new logs as vector pages;
- older logs as scans;
- a lab appendix of a summary table and 13 sheets.

The summary table has one planted error, of the same kind as the field
report's: B-2 S-3 has its liquid and plastic limits swapped. Every name and
number is invented.

| Task | Page | Scored on |
|---|---|---|
| `report-extract-all` | Document Review | every log and lab page read; the answer states coverage as counts; all six borings named; values from the newest log, a scan and the last sheets |
| `report-extract-diggs` | geotech | every log and lab page read; coverage stated; a DIGGS file written; all six borings named |
| `report-summary-vs-sheets` | Document Review | names B-2 S-3 with both values; cites the sheet; every lab page read |

"Every page read" is worked out from the run's own `activity.jsonl`, not from
the ledger, so the arm that adds the ledger is not graded by it. Tasks can
now be asked on either page (`Task.page`), and RESULTS.md has a "By app page"
table.

**Arms:**

- `coverage` = `GEOTECH_COVERAGE=1`;
- `checklist` = coverage plus `GEOTECH_REVIEW_CHECKLIST=1`.

Both sit on each page's default agent, so they compare straight against
`baseline`. To run:

```python
score_review_suite(prompter=fh_prompter, arms=("baseline", "coverage", "checklist"), ...)
```

Use `ids=["report-"]` for just the three new tasks; run the whole suite to
see what the gate costs on the narrow questions.

## Found on the way, and fixed

The activity log lost records whenever tools ran in parallel. A model step
that asks for ten page looks runs ten tools at once, and their long lines
interleaved in `activity.jsonl`. A suite run had a page look with no
`tool_end` at all. Each line is now written whole, under a lock. A test fails
without the lock, every time.

This matters for "review every run in full". Earlier traces with parallel
page looks may be missing records, and `runner._activity` silently skipped
the broken lines.

## What is left for you

1. **Mark up the checklist:** `funhouse_agent/report_review_checklist.json`.
   Add, cut or reword items, and say which `judge` items code should learn
   next. Each item's `text` is free to edit; keep its `id`. The file says
   how.
2. **Run the suite** with the three arms above. A switch goes on by default
   only when the suite says it helps.
3. **Decide:**
   - **Narrow questions.** Should reading a few data pages hold a turn to
     every data page, as now? The note on a narrow question ("B-3's water
     level?") then asks the agent to read the other logs or say why not. That
     costs one model call. The suite's narrow tasks will show it.
   - **Text and plans.** Should a declared or written-out extraction also
     owe the text pages and the plans, as now? Or only the data pages?
   - **Step cap.** Is the raised step cap of 80 right, with the switch on?

## What the lead wires

Nothing for the reconciler. `write_diggs` already returns its
`cross_checks`, and the app records them from the call's result. If that
result's shape changes (`cross_checks.entries` with `kind`, `where`,
`detail`, `values`, `pages`), update:

- `coverage.CoverageLedger._record_output`;
- `report_checklist._cross_entries`.

A new code check is added with `report_checklist.register_check(name, fn)`
and named in the JSON.

## Known limits

- **What does not count as a read:**
  - pages the whole-report ingest tool reads;
  - `find_like`;
  - contact sheets.
- **Earlier turns.** Pages read in earlier turns of the conversation count as
  read, but only this turn's reads arm the gate.
- **Planlens roles.** The inventory is only as good as planlens' page roles.
  A tab with no text layer (the field report's dividers were markup only;
  not checked against that report) can leave the scans behind it without a
  "log" label; they then count as "other". A declared task or a
  `write_diggs` still holds them, and a data-only turn does not.
- **No live run yet.** Nothing has been run on a real model. The first check
  is the suite run on Funhouse or Foundry, with every run read in full.
