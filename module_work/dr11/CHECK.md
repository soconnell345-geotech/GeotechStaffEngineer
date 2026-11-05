# DR-11: first slice, what was built and checked

Branch `claude/dr-11-recompute-qc`. Plan: `module_work/dr11/PLAN.md`.

## Built

* **`report_ingest/recompute.py`**:
  * `recompute_record(record, tolerance=0.10)` takes a `ReportRecord`, its
    dict, or the path of `report.record.json` or its folder;
  * `recompute_calculation(calc)` does one calculation;
  * `table_markdown(table)` writes the table as Markdown;
  * `find_record(where)` finds a record file;
  * `make_recompute_tool()` builds the page tool.

  Each `Calculation`'s printed labels are matched to module inputs with a
  small label vocabulary per kind. Units go through the record's own
  `calc_si`. The module runs through `funhouse_agent.dispatch.call_agent`,
  the route the page's agent takes, so each row carries the module's
  `choices` basis and `defaults_used`. No equation, chart or default was
  added.
* **The tool `recompute_report_calcs(record="")`** is on the geotech page,
  offered with `report_ingest` (both behind `GEOTECH_REPORT_INGEST`, and
  never on the Document Review page). With `record` left empty it uses the
  one record in the working folder. Above the result cap it drops detail
  first, then keeps the flagged rows first.

## Kinds covered

| Kind | Recomputed by | Method choice |
|---|---|---|
| `shallow_foundation_bearing` | `bearing_capacity.bearing_capacity_analysis` | If the report names Vesic, Meyerhof or Terzaghi, that one is used. If it names Hansen, Vesic runs with Hansen's N-gamma. If it names none, the module default (Vesic) runs, and Meyerhof and Terzaghi run beside it with their values and differences. |
| `settlement`, elastic | `settlement.elastic_settlement` | The printed method if there is one. If not, it is inferred from the printed inputs (a modulus and no Cc means elastic), and the row says so. |
| `settlement`, one-layer Cc/Cr | `settlement.consolidation_settlement` | The printed method if there is one. If not, it is inferred (a Cc and no modulus means consolidation). No sigma_p printed means normally consolidated, and that is listed under `assumed`. |
| every other kind | none yet | a `not_recomputed` row with the reason (`NOT_RECOMPUTED`); slope stability is "held, not recomputed yet" |

**A missing input is named, never filled in.** That covers an input not
printed, an input printed twice with different values, a unit of the wrong
kind, a shape that is not printed and not settled by L against B, and an
allowable pressure printed with no factor of safety. Where the printout
names more than one settlement, nothing is compared, so the code never
picks one.

## Tests

`report_ingest/tests/test_recompute.py` has 39 tests. They run offline,
make no model calls and use made-up values. Each expected value comes from
the module itself. They cover:

* **bearing capacity:**
  * a claim that matches;
  * a claim that differs;
  * net allowable against gross;
  * US customary units;
  * a printed method;
  * a method not printed, where the claim matches the Terzaghi option;
  * a missing unit weight;
  * no strength and no shape;
  * the width printed twice;
  * an allowable with no factor of safety;
  * no claimed pressure;
* **elastic settlement:** a claim that matches (method inferred), a claim
  that differs (method printed), a missing Es, a strip with no Iw;
* **consolidation:**
  * a claim that matches (normally consolidated, assumed);
  * a printed sigma_p;
  * a missing Cr;
  * inputs that fit both methods;
  * Schmertmann;
  * two settlements printed;
* **kinds not recomputed:** one test per kind, plus a check that slope
  stability is held;
* **the record level:** counts, Markdown, the caller's tolerance, loading
  from the file and from its folder;
* **the tool:** finding the one record, refusing two, the size cap, and
  coming and going with `report_ingest` on the page.

Runs, each judged by pytest's exit code:

* `pytest report_ingest`: 4,620 passed, 8 skipped (pydiggs is not
  installed here). Exit 0.
* `report_ingest/tests/test_recompute.py`, `webapp/tests/test_page_fingerprints.py`,
  `webapp/tests/test_public_text.py` and
  `funhouse_agent/deep/tests/test_report_ingest_{page,offline}.py`: 85
  passed. Exit 0.
* `pytest funhouse_agent/deep/tests` (because `agent.py` changed): 1,171
  passed and 1 failed. The failure is
  `test_wave1_geotech.py::test_the_flags_match_what_the_code_demands`,
  which needs `liquepy`, and liquepy (with eqsig and geofound) does not
  build in this container. Nothing in this change touches it.

**Page fingerprints were not re-recorded.** The tool comes with the report
ingest switch, which is off in every pinned build, so the default geotech
page's tool list is unchanged and the fingerprint test passes as recorded.
No adapter or dispatch file changed.

## Left for later slices, by kind

* **settlement:** Schmertmann, Hough, consolidation over several layers, the
  combined analysis, and a stress increase worked out from q and B. All of
  these need the printout's layer table kept as rows, which the record's
  flat label list does not keep.
* **shallow_foundation_bearing:** two-layer cases (a second layer's labels),
  eccentric and inclined loads, sloping ground.
* **axial_pile / lateral_pile / pile_group:** these need a layered profile
  and the section.
* **retaining_wall:** earth-pressure coefficients first, then whole walls.
* **liquefaction, site_response:** these need a layer table, or a printed
  Vs30 or N-bar for the site class.
* **pavement:** AASHTO 1993 SN and slab D.
* **ground_improvement:** needs its own slice.
* **slope_stability:** held (GM-140).
* **seepage:** no module recomputes it from a printout's labels.
* **Excel sheet:** a "Recompute" sheet and `report.recompute.json`, once the
  owner says where results are read. Today the ingest writes Excel only for
  the question list.
* **Label vocabulary:** measure it on the saved corpus records. Only the
  owner's private records can show which printed labels the patterns miss.
  That is the first thing to run before the next slice.

## Release note (three lines)

* New: `recompute_report_calcs` on the geotech page (it comes with report
  ingest) and `report_ingest.recompute.recompute_record`. Together they
  re-run the bearing capacity and settlement calculations an ingested
  report claims, through the app's own modules.
* Each claimed result gets the recomputed value, the difference and a flag
  past 10 %. The row also gives the method used and why, and the inputs with
  their pages.
* An input the report did not print is named, never filled in. Other
  calculation kinds are listed with the reason they were not recomputed;
  slope stability is held.
