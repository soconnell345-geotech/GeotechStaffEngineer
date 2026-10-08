# Reading reports faithfully: scales, coverage, cross-checks (plan of record, 2026-10-08)

## Why

The 2026-10-06 geotech-page session was checked number by number against the report
(`module_work/field_feedback/2026-10-06_geotech-report-session_v5.32.0/FINDINGS.md` §8).

- **Printed numbers were read right:** 48 of 48 SPT rows; lab sheets exact.
- **Positions read off drawings were eyeballed.** Layer depths came out 0.1–0.7 m off. This is the same failure as the misplaced circles (`module_work/harness_theory/locating_things_on_a_page.md`).
- **Coverage was lost.** Only 10 of 23 classification sheets were read. A whole boring's lab results and the newest borings' logs never reached the files.
- **A summary-table error in the report itself went unflagged.**
- **The "DIGGS" file was typed by hand.** No writer was reachable, and the schema check needs pydiggs, which the cluster does not have.

## Owner decisions (2026-10-08)

1. **One general workflow for visual scales, in both harnesses** (geotech page and Document Review). It merges with the location work.
2. **`write_diggs` runs report ingest's reconciler cross-checks.** No test-specific gates: the proposed Atterberg check was over-fitting to one case. The reconciler's summary-vs-sheet cross-check catches the session's error generally, since the sheet itself reads LL 32 / PL 20.
3. **Coverage is enforced in code,** and a report-review checklist is drafted for the owner to mark up (below).

**Release.** 5.32.1 is cut from a4ef417, the tree Foundry brief 4 measures, plus any fixes it needs. Everything here goes to the next release (5.33).

## W1: the DIGGS schema check without pydiggs (lead, now)

- **Ship the schema with the app.** Copy the DIGGS 2.6 XSD files into `subsurface_characterization`, unmodified and with their MPL-2.0 licence. Only the files `Diggs.xsd` actually imports go in, measured by walking its imports.
- **Validate with lxml,** which is already installed through python-docx.
- **Keep pydiggs as an alternative.** Where it is installed it stays usable.
- **Effect:** `validate_diggs_schema` and the writer's schema check then run on every host, with no network access.
- **Tests:**
  - the writer's own output passes;
  - the session's hand-typed file fails at its root;
  - the schema is in the wheel;
  - nothing is fetched from the network.

## W2: `write_diggs` runs the reconciler (lead, after W1)

- **Build a `ReportRecord`** from what `write_diggs` is given.
- **Run `report_ingest.reconciler.reconcile` on it.** It checks summary against sheets, links lab tests to borings and samples, and catches a sample deeper than its boring.
- **Return its QA entries** in the result, and name them in the verdict.
- **Nothing is changed or resolved.** Disagreements are reported, exactly as report ingest does.
- **Test:** synthetic data whose summary table disagrees with the matching sheet must come back as a disagreement.

## W3: visual scales, both harnesses (Opus subagent: survey and design first)

**Principle:** the model says WHAT a thing is and roughly where. Code finds the exact drawn thing and converts its position through a fitted scale.

**What a scale is:** a fitted map from page position to a quantity.
- **Kinds:** linear or log, per axis.
- **Built from:**
  - printed labels and ticks (text layer, or OCR on scans);
  - a stated scale;
  - a scale bar;
  - the scale Bluebeam or Acrobat store in a PDF.
- **Carries** its fit residual and a confidence.
- **One idea covers:**
  - logs (depth and elevation);
  - plots (both axes, including log axes such as grain size);
  - profiles (station and elevation);
  - plans (distance and coordinates).

**Pieces that exist:**

| Where | What |
|---|---|
| planlens `log_grid` | the depth ruler and stratum lines, on vector pages and on scans with OCR lines |
| report ingest's sounding reader | axis ranges as gates |
| `zoom_plot` | crops with the axes in view; the model reads values against the ticks |
| planlens `measure.Quantity` | refuses to turn page points into feet without a resolved scale |
| planlens, other | viewport scales; page-map scale notes; the cross-section importer's calibration |
| the app | pixel boxes converted to page points (cf43c90) |

**Step 1: a design document from the subagent.** It covers:
- what exists, with what each piece measures;
- the gaps;
- one primitive and one tool surface for both harnesses;
- how the agent is told about it, through tool descriptions and not prompt rules;
- fixtures and measurement;
- build order.

**Step 2:** the lead reviews the design and the build follows.

**Measurement:**
- synthetic fixtures with known depths, axes and scales;
- the session's scanned log sheets as a private local spot check (four sheets, contacts measured 2026-10-07).

## W4: coverage in code, and a report-review checklist

**(a) Coverage for extraction tasks:**
- **Inventory first, in code:** planlens page roles give the target pages.
- **A ledger:** each page is marked extracted, or skipped with a reason.
- **Coverage from the ledger:** the answer states it as counts ("10 of 23 lab sheets read").
- **Delivered as** a tool and its description. Report ingest already does the inventory structurally.

**(b) The checklist is one data file used two ways:**
- **Items code can check** run automatically.
- **Judgement items** are worked through by the agent and reported against.

This is also review-harness Phase 5: the review prompt rewritten around the review method (`module_work/REVIEW_HARNESS.md`).

### Report-review checklist: FIRST DRAFT, for the owner to mark up

In each item, [code] means a check code can run; [judge] means the agent works through it and reports.

**Completeness**
- [code] Every exploration named in the text, on the location plan or in a boring list has a log in the report, and every log names an exploration that is listed.
- [code] Every log page and every lab sheet was read, stated as counts.
- [code] Every sample on a lab summary table has a sheet, and every sheet is on the summary.
- [judge] Every campaign is identified and dated: earlier reports bound in, and the newest borings.
- [judge] Groundwater is recorded for every boring, or stated as not encountered.
- [judge] Locations, elevations and their datum are stated.

**Consistency**
- [code] Summary-table values match the individual sheets (the reconciler).
- [code] Lab sample IDs and depths match samples on the logs.
- [judge] Log descriptions agree with the lab classification of the same sample.
- [judge] Groundwater and stratigraphy stated in the text agree with the logs.
- [code] Units are stated and consistent.

**Currency**
- [judge] Which campaign governs, and whether superseded data are used in the recommendations.
- [judge] The code editions cited (seismic code, ASCE 7 edition) are current for the project.

**Traceability**
- [code] Every extracted value cites its page.
- [judge] Every recommendation traces to data in the report.

**Plausibility**
- [judge] Values are plausible for the soil described: N against the density description, moisture against classification.

**Recommendations**
- [judge] Design parameters derived from the data, with the method cited.
- [judge] Topics the data call for are addressed: bearing, settlement, seismic site class, liquefaction, and corrosion or chemistry exposure for concrete and steel.

## Order and status

| Item | State |
|---|---|
| W1 | DONE (499e1cf): bundled DIGGS 2.6 schema, checked with lxml on every host |
| W2 | DONE: `write_diggs` reconciles before writing and returns `cross_checks` |
| W3 | Steps 0–5 BUILT in planlens a3f2a1d (`find_scales`, `measure`, `log_grid` on scans). Harness at `module_work/scales_harness/`: GATE PASS, 99.93 % inside ±. Private spot check: 32/32 stratum lines, median 0.007 m. Steps 6–9 (app side) next |
| W4 | BUILT behind two switches, both OFF (`module_work/COVERAGE_AND_CHECKLIST.md`). The checklist content waits on the owner's markup. Measurement waits on a suite run. |

**W4 in brief.**
- **Switches:** `GEOTECH_COVERAGE` turns on the ledger, `document_coverage` and the gate; `GEOTECH_REVIEW_CHECKLIST` turns on `report_checklist`.
- **Suite arms:** `coverage` and `checklist`.
- **Three new suite tasks** run on a 30-page synthetic report: vector new logs, scanned old ones, and a lab appendix with one LL/PL swap planted.
- **Not yet measured:** run `arms=("baseline", "coverage", "checklist")` on Foundry or Funhouse and read every run in full before either switch goes on.
- **Found on the way and fixed:** the activity log lost records when tools ran in parallel. Each line is now written under a lock. Earlier traces with parallel page looks may be missing records.

**Corrections from the W3 design** (its §11). Read those, not the W3 section above, for what exists:
- `log_grid` finds no stratum lines on scans, even with OCR.
- Its ruler assumes each label is centred on its depth; on the scanned form labels sit about 0.1 m above it.
- Neither harness can reach it.
- The sounding reader's axes are ranges with no positions.
- Scale notes are never applied, and scale bars are not found at all.
- The free RapidOCR path loads OpenCV, so it cannot run on the government hosts.

Gate for every item: offline tests on fakes and pytest exit codes, then a live check (Foundry or Funhouse) with every run read in full.
