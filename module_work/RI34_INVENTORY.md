# RI-34 inventory: where the app asks a geotechnical report its questions

BACKLOG RI-34 (owner, 2026-10-20): "move all questions related to geotech
reports to report_ingest"; geotech reports are not fed to the Document Review
page, which becomes its own package for people who are not geotechnical
engineers (INF-26). This lists every place, at master ac94ba5, where the app
asks a geotechnical report its standard questions, makes a report extraction
or checks one, and where each goes. Written before anything moved.

How it was found: a grep of the app for the checklist, the switches and
report words, and a scan of the Document Review page's whole model-facing
surface (system prompts and every tool description, built offline on a fake
model) for the legacy, lean and minimal builds, with every review switch off
and on.

Destinations:

- **report_ingest**: the question or check moves into the `report_ingest`
  package (or is already there).
- **geotech page**: stays on the GeotechStaffEngineer page only (taken off
  the Document Review page where it was offered there).
- **generic (Document Review)**: a general document question or mechanism
  that also happens to run on reports; stays on the Document Review page.
  Where its wording named report things, the page gets neutral wording.

## Headline

| Destination | Count | Items |
|---|---|---|
| report_ingest (moves now) | 3 | 1, 2, 3 |
| report_ingest (already there) | 4 | 10, 11, 12, 21 |
| geotech page | 9 | 4, 5, 8, 9, 13, 20, 22, 23, 24 |
| generic, wording neutralized for Document Review | 8 | 6, 7, 14, 15, 16, 17, 18, 32 |
| generic, unchanged | 8 | 19, 25, 26, 27, 28, 29, 30, 31 |

The Document Review page loses two tools (`report_checklist`, which it had
only behind `GEOTECH_REVIEW_CHECKLIST`, and `log_grid`), one `plot_data`
argument (`reference_chart`, the soil-classification charts), two review_eval
tasks (now asked on the geotech page) and every report word in its prompt and
tool descriptions.

## The list

| # | What | Where (master ac94ba5) | Destination | What happens |
|---|---|---|---|---|
| 1 | The review checklist's data: "Geotechnical report review checklist", DRAFT, 18 items (7 code items over 6 checks, 11 judge) | `funhouse_agent/report_review_checklist.json:1-58` | report_ingest | Moves to `report_ingest/report_review_checklist.json`, packaged in the wheel. Status, version, items, ids and wording unchanged (OWN-31); `how_it_is_used` / `how_to_edit` say where the checks now read from, and (review, 2026-10-20) `consistency.summary_vs_sheets` gained `where: "lab_tests."`: the record holds every reader's QA entries, and without it a log value read two ways or an SPT whose drives do not add up failed "Summary-table values match the individual sheets". |
| 2 | The checklist's six code checks (`logs_and_lab_read`, `explorations_vs_logs`, `lab_ids_vs_logs`, `summary_and_sheets`, `cross_checks`, `values_cite_pages`) | `funhouse_agent/report_checklist.py:178-388` | report_ingest | Re-written in `report_ingest/checklist.py` over the record (`ReportRecord`: page labels, investigations, lab tests, narrative answers, QA entries, the reconciler's disagreements) instead of the coverage ledger, the PDF's text layer and `write_diggs` cross-checks. Same check names, so the JSON items run unchanged. A compatibility shim stays at `funhouse_agent/report_checklist.py`. |
| 3 | The `report_checklist` tool and its description | `funhouse_agent/deep/coverage_tools.py:115-125, 489-519` | report_ingest | `report_ingest.checklist.make_checklist_tool`: takes the record `report_ingest` wrote (`paths.record`, or its folder), not a document handle. |
| 4 | `GEOTECH_REVIEW_CHECKLIST` ("both pages") | `funhouse_agent/review_flags.py:78-83, 184, 283-285` | geotech page | Offers the tool on the geotech page only. OFF stays OFF; ON no longer reaches the Document Review page. |
| 5 | Failed checklist checks in the coverage gate's note | `funhouse_agent/deep/coverage_tools.py:314-329` | geotech page | Dropped from the gate: the gate is about pages read in a conversation; the checks now read a record, and report through their own tool. |
| 6 | The coverage ledger and gate (`GEOTECH_COVERAGE`): its page groups are report_ingest's page roles (logs, lab sheets, field tests) | `funhouse_agent/coverage.py:55-79` (groups), `1092-1136` (`armed`) | generic (Document Review) | The mechanism (every page any tool read, coverage as counts, a declared coverage task) stays on both pages. On the Document Review page the gate now arms only on a declared coverage task: no longer on a turn that read some log or lab pages (report-data arming). The report groups stay in `coverage.py` until INF-26 splits it. |
| 7 | `document_coverage`'s description ("every boring log and laboratory sheet of a report into a table, a DIGGS file") | `funhouse_agent/deep/coverage_tools.py:91-113` | generic (Document Review) | The geotech page keeps it; the Document Review page gets neutral wording (a schedule, an appendix of tables, every sheet of a set). |
| 8 | Coverage arming on `write_diggs` and its record of outputs | `funhouse_agent/coverage.py:794-880` | geotech page | Unchanged; `write_diggs` exists only there. The checklist no longer reads `ledger.outputs`. |
| 9 | `write_diggs` cross-checks | `funhouse_agent/adapters/subsurface_adapter.py:984, 1405` | geotech page | Unchanged. The checklist reads the same reconciler's QA entries in the record instead. |
| 10 | The narrative schemas (the owner's two standing query schemas) | `report_ingest/model.py:1812` (`GeneralFacts`), `:1919` (`NaturalHazardFacts`), `:2003` (`NarrativeFacts`) | report_ingest | Already there. |
| 11 | The narrative conventions (OWN-07; RI-33 adds two) | `report_ingest/narrative_glossary.py:72` | report_ingest | Already there. |
| 12 | The reconciler's checks (lab to ground, counts, summary table, units, pages) | `report_ingest/reconciler.py` | report_ingest | Already there; now what the checklist's code items read. |
| 13 | `log_grid` (a boring or test-pit log read as its grid) on the Document Review page | `funhouse_agent/deep/review_agent.py:75`; `funhouse_agent/deep/tools.py:1942`; `funhouse_agent/measure_tool.py:255` | geotech page | Taken off both review builds (legacy and lean) and the reading helper; the geotech page keeps it. |
| 14 | `measure`'s description: "a log's depth or elevation ruler", "layer boundaries in a log's description column" | `funhouse_agent/measure_tool.py:39-58` | generic (Document Review) | Tool stays; the Document Review page gets wording without the log examples. |
| 15 | `plot_data`'s description and `reference_chart` ("an SPT or CPT profile", "a plan view of borings", "boring-log convention"; the USCS / AASHTO plasticity charts) | `funhouse_agent/deep/tools.py:309-332` (from `profile_figure` METHOD_INFO) | generic (Document Review) | Tool stays; the Document Review page gets neutral wording and no `reference_chart`. |
| 16 | `search_document`'s description (planlens: "a boring or sheet id") | planlens `tools/specs.py` | generic (Document Review) | The page's own wording ("an item or sheet id", a phrase swap in `review_agent.REVIEW_REWRITES`); planlens' text is a backlog candidate. |
| 17 | `render_page_thumbnails`' description (planlens: "the plan, the logs, the tables") | planlens `tools/specs.py` | generic (Document Review) | The page's own wording; planlens backlog candidate. |
| 18 | The Document Review prompt's "a plotted log" (also in the reading helper's prompt) | `funhouse_agent/deep/prompt.py:373` | generic (Document Review) | "a chart". |
| 19 | The Document Review prompt sends analyses to the GeotechStaffEngineer page | `funhouse_agent/deep/prompt.py:479-494` | generic (Document Review) | Not a report question; what the page says after the split is INF-26's. |
| 20 | The geotech domain prompt's document section (a boring log is a picture) and `REPORT_INGEST_NUDGE` | `funhouse_agent/deep/prompt.py:117, 145, 199` | geotech page | Unchanged. |
| 21 | The `report_ingest` tool and sub-agent | `report_ingest/subagent.py:53`; `funhouse_agent/deep/tools.py:2208` | report_ingest | Unchanged; geotech page, behind `enable_report_ingest` (OFF). |
| 22 | review_eval `report-extract-all` (subsurface data into tables) | `funhouse_agent/review_eval/tasks.py:969` | geotech page | `page="geotech"`; checks unchanged. |
| 23 | review_eval `report-summary-vs-sheets` (the lab summary against the sheets) | `funhouse_agent/review_eval/tasks.py:1035` | geotech page | `page="geotech"`; checks unchanged. |
| 24 | review_eval `report-extract-diggs`, `scale-log-depth` | `funhouse_agent/review_eval/tasks.py:1009, 1069` | geotech page | Already `page="geotech"`. |
| 25 | review_eval `scale-plan-distance` (two labelled points on a re-plotted site plan) | `funhouse_agent/review_eval/tasks.py:1096` | generic (Document Review) | A measuring question on a drawing; the points happen to be borings. |
| 26 | review_eval `calc-wall-check`, `calc-bearing-consistency` | `funhouse_agent/review_eval/tasks.py:823, 841` | generic (Document Review) | Checking a calculation package against itself; any discipline's. |
| 27 | review_eval `ufc07-drilled-shaft-table` | `funhouse_agent/review_eval/tasks.py:706` | generic (Document Review) | Finding a table in a criteria manual. |
| 28 | The suite's synthetic report and its document entry | `funhouse_agent/review_eval/report_fixture.py`; `review_eval/documents.py:67` | generic (Document Review) | Stays in review_eval, which runs both pages; serves geotech-page tasks only now. Where it lives after the split is INF-26's. |
| 29 | Arms `coverage` and `checklist` | `funhouse_agent/review_flags.py:387-388` | generic (Document Review) | Unchanged; their report tasks are now asked on the geotech page. |
| 30 | The review digest counts planlens' page roles | `funhouse_agent/review_digest/build.py:634` | generic (Document Review) | Unchanged: it reports whatever roles a document has. |
| 31 | The Document Review page's orientation turn and upload note | `webapp/profiles.py:31-90`; `webapp/core.py:172` | generic (Document Review) | Unchanged; no report wording. |
| 32 | The figure helper's description in the `task` tool ("a plan of borings ..., a profile (SPT, CPT, su, water content against elevation)"); on both pages since GM-227, found after merging master | `funhouse_agent/deep/figure_helper.py:59-75` | generic (Document Review) | A phrase swap on the review page (`review_agent.REVIEW_HELPER_REWRITES`). The helper's own prompt, `figure_examples` and its `plot_data` keep their geotech examples: that is the INF-26 plan's D8 (b), its own item. |

## What the record cannot answer yet (the checklist's code items)

- `completeness.explorations_have_logs`: the narrative's named explorations
  (`boringDictionary`, `testPitDictionary`) and stated counts are in the
  record; names printed only on the location plan are not (boring locations
  off the plan figure are RI-24).
- `completeness.pages_read`: needs the record's per-page labels
  (`page_labels`, written since the label vote); a record written before
  them has only role counts, so the check says it cannot list pages.
- `traceability.values_cite_pages`: the old check also asked "pages cited
  that no tool read"; the ingest reads every page it cites by construction,
  so only the first half (values with no page) is checked.
- The eleven judge items stay judge items. Several could be read from the
  record (groundwater per boring, locations and datum, log descriptions
  against lab classification, N against the density description); which
  ones code should learn is the owner's call in OWN-31.
