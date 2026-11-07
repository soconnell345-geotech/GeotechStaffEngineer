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
| `shallow_foundation_bearing` | `bearing_capacity.bearing_capacity_analysis` | If the report names Vesic, Meyerhof or Terzaghi, that one is used. If it names Hansen, Vesic runs with Hansen's N-gamma. If it names none, the module default (Vesic) runs, and Meyerhof and Terzaghi run beside it with their values and differences. With Df > 0 and no depth factor printed, the method also runs with its depth factors off. A printout that names Meyerhof, with Df/B > 1, also runs Meyerhof's own k = Df/B (`depth_factor_method 'linear'`, UFC 3-220-20 Table 5-6; GM-309), whether or not a depth factor is printed. |
| `settlement`, elastic | `settlement.elastic_settlement` | The printed method if there is one. If not, it is inferred from the printed inputs (a modulus and no Cc means elastic), and the row says so. |
| `settlement`, one-layer Cc/Cr | `settlement.consolidation_settlement` | The printed method if there is one. If not, it is inferred (a Cc and no modulus means consolidation). A printed OCR gives sigma_p = OCR x sigma_v0. No sigma_p or OCR printed means normally consolidated, listed under `assumed`, unless the printout calls the soil overconsolidated (then not recomputed). |
| every other kind | none yet | a `not_recomputed` row with the reason (`NOT_RECOMPUTED`); slope stability is "held, not recomputed yet" |

**A missing input is named, never filled in.** That covers an input not
printed, an input printed twice with different values, a unit of the wrong
kind, a shape that is not printed and not settled by L against B, and an
allowable pressure printed with no factor of safety. Where the printout
names more than one settlement, nothing is compared, so the code never
picks one.

## After the review (`REVIEW_DR-11`, ten findings)

The rule for every fix: never a guessed input, never a confident wrong flag.
Where the printed inputs do not let the module reproduce the report's own
method, the row is `not_recomputed` or `not_compared` with the reason, or
the module's own alternatives run beside the default and are shown. No
equation, factor or default was added; only options the modules already
have are used.

* **The flags.** `agrees` within 10 % of the module's default. `differs`
  only when the claim is more than 10 % outside the range of every option
  the module ran (default and alternatives). `within_options` when the
  claim misses the default but falls inside that range: the printout does
  not settle which option it used. `below` for an allowable pressure lower
  than q_ult/FS. Each row with alternatives carries `options_range_si` and
  a `flag_basis` naming the options run.
* **Options run beside the default** (finding 6): for bearing capacity with
  Df > 0 and no depth factor printed, `depth_factor_method="none"` with the
  module's own basis (GEC-6 Sec. 5.2.3.3), whether or not the method is
  printed; with no method printed, Meyerhof and Terzaghi as before. For
  elastic settlement with no Iw printed and no rigidity printed, GEC-6
  Table 5-12's rigid and average Iw (`rigidity="rigid"`, `point="average"`),
  each with its Iw.
* **An allowable below q_ult/FS is `below`** (finding 4), with a note:
  UFC 3-220-20 Sec. 5-2.4 defines the allowable as the highest pressure that
  meets both the factored bearing capacity and settlement (p. 278), prints
  Eq. 5-14 as an upper bound and lowers the allowable where settlement
  controls (step 3b, p. 279). Checked in the reference library's
  `dm7_2/text/chapter05.json` and on the printed pages.
* **Inputs the module cannot take as printed:**
  * an undrained strength (su, cu, undrained shear strength) is its own
    input: alone it runs as the cohesion with phi = 0, and the row says so;
    printed with a c' or a phi above 0 the row is `not_recomputed`, the
    short- and long-term cases being separate runs (finding 1);
  * a printed OCR gives sigma_p = OCR x sigma_v0 with the printed effective
    stress the module runs the layer at, said in `inputs_used`; a sigma_v0
    labelled as a total stress refuses the row; a printout that calls the
    soil overconsolidated and gives neither sigma_p nor OCR is not run as
    normally consolidated (finding 2);
  * gamma' (and gamma_sub, gamma_b, "prime") is not the total unit weight
    (finding 3);
  * a "net ultimate" claim is `not_compared`: the module's q_ult is the
    gross one (finding 5);
  * only "Iw", "influence factor" alone and "shape and rigidity factor" are
    GEC-6 Eq. 5-36's Iw; a strain influence factor Iz, a depth or embedment
    factor or another influence factor makes the elastic row
    `not_recomputed`; a compression or recompression ratio (CR, RR) is not
    taken as Cc or Cr, and the missing index names the ratio printed;
    C-alpha is never Cc (finding 7).
* **The size cap** drops `inputs_used` first, then the alternatives and
  assumptions of the rows not flagged `differs`, then their alternatives;
  a `differs` row keeps `assumed`. The Markdown table lists each row's
  assumptions in its method cell (finding 8).
* **The tolerance** is said to be this table's own QC threshold, not an
  engineering criterion, in the result (`tolerance_basis`), the Markdown
  heading and the tool description (finding 9).
* **The record path**: the description asks for the path report_ingest
  returned (`paths.record`); with `GEOTECH_REPORT_LIBRARY` set, an empty
  `record` that finds nothing asks for that path instead of a re-ingest
  (finding 10).

## Tests

`report_ingest/tests/test_recompute.py` has 55 tests (39 at the first
slice, 16 added with the review fixes). They run offline, make no model
calls and use made-up values. Each expected value comes from the module
itself. They cover:

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
  coming and going with `report_ingest` on the page;
* **the review fixes, one failing case each:** su with phi' not run
  together, su alone run at phi = 0; OCR to sigma_p, OCR with a total
  sigma_v0 refused, "overconsolidated" with no sigma_p refused; gamma'
  not the total unit weight; an allowable below q_ult/FS flagged
  "below" and one above flagged "differs" (1.3 x); a net ultimate not
  compared; depth factors off beside the default (Df/B = 1), and not run
  where a depth factor is printed; Table 5-12's rigid and average Iw
  beside the default; a strain influence factor Iz not taken as Iw; CR,
  RR and C-alpha not taken as Cc or Cr; the cap keeping `assumed` on
  "differs" rows; the tolerance basis in the table and the description;
  `paths.record` with a shared library. Each of these fails on the first
  slice's code (21 failures there) and passes on this one.

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

After the review fixes (merged with master, app venv, one run at a time,
each judged by pytest's exit code):

* `report_ingest/tests/test_recompute.py`: 55 passed. Exit 0.
* `webapp/tests/test_page_fingerprints.py`: 15 passed. Exit 0.
* `webapp/tests/test_public_text.py`: 4 passed. Exit 0.
* `funhouse_agent/deep/tests/test_report_ingest_page.py`: 5 passed. Exit 0.
* `funhouse_agent/deep/tests/test_report_ingest_offline.py`: 22 passed.
  Exit 0.

**Page fingerprints were not re-recorded.** The tool comes with the report
ingest switch, which is off in every pinned build, so the default geotech
page's tool list is unchanged and the fingerprint test passes as recorded.
No adapter or dispatch file changed.

## Left for later slices, by kind

Detail only: the open work is BACKLOG DR-11.

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
* Each claimed result gets the recomputed value, the difference and a flag:
  'differs' only past 10 % (this table's QC threshold) of every option the
  module ran, 'within_options' inside their range, 'below' for an allowable
  under q_ult/FS. The row also gives the method used and why, the options
  run, and the inputs with their pages.
* An input the report did not print is named, never filled in. Other
  calculation kinds are listed with the reason they were not recomputed;
  slope stability is held.
