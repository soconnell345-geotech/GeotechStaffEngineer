# `report_ingest` — a geotechnical report as organised, cited data

Part of the report-ingest train (`module_work/REPORT_INGEST_PLAN.md`). planlens
turns a PDF into pages with kinds, located text, an outline and a first draft of
what each page **is**; this package adds the geotechnical judgement and the
record. The narrative reader and the reconciler are later work packages.

| Piece | Module | Shape |
|---|---|---|
| 0b. Document triage | `triage.py` | one structured call |
| 0c. Label review | `label_review.py` | an agent loop with four tools |
| The record | `model.py` | pydantic; the product everything else exports |
| 2. Log reader | `log_reader.py` | one structured call per log, image alongside |
| 3. Lab reader | `lab_reader.py` | one call per sheet, `zoom_plot` when a curve is only plotted |
| 4. DIGGS writer | `diggs_writer.py` | deterministic, with two gates |
| Scoring one log | `log_scoring.py` | the grid alone, then the reader |
| Scoring one sheet | `lab_scoring.py` | the page's tables alone, then the reader |

## The record (`model.py`)

`ReportRecord` is what the ingest is **for**; the summary page, the library page
and the DIGGS file are three exports of it. Three rules hold it together.

**A number keeps the unit it was printed in.** A log that prints 21.5 ft is
`Quantity(21.5, "ft")`, and a `Quantity` cannot be built without a unit.
`to_si()` converts once, in the writer that needs SI, so a reviewer setting the
record beside the page sees the page's own numbers.

**Every value-bearing field carries `Provenance`** — the page, the box, and how
it was read: `text`, `di`, `ocr`, `grid`, `vision` or `derived`. A reviewer can
go to the page; a QA pass can ask which values rest on vision alone.

**What could not be read is recorded, not guessed.** An optional field stays
`None`, a `QAEntry` says what was skipped and why, and
`Investigation.units_known` is `False` when no depth unit was printed anywhere.

`NarrativeFacts` names the owner's two query schemas for WP4 and `CalcEntry` is
WP5. They are in the schema now so a consumer written against it keeps working
as the stubs fill in. `record_json_schema()` exports the whole thing.

**`LabTest.result` is a typed result per kind** since WP3, discriminated on
`kind`, and a result whose kind does not match its test is refused at
construction — `kind` is what every consumer dispatches on, so an Atterberg
result filed under a gradation would be read as a gradation by everything
downstream and the mistake would be invisible in the JSON.

| Result | For | Holds |
|---|---|---|
| `AtterbergResult` | atterberg | LL, PL, PI, shrinkage limit, `non_plastic`, the Casagrande flow curve, each plastic-limit trial |
| `GradationResult` | gradation | the percent-passing curve as `SievePoint`s, D10–D100, Cu, Cc, the cobble/gravel/sand/silt/clay/fines fractions |
| `ConsolidationResult` | swell_consolidation | `test_type` swell/collapse/oedometer, the pressure-strain (or void-ratio) curve with a stage per point, swell %, pc, Cc, Cr, cv, e0 |
| `StrengthResult` | triaxial, direct_shear, unconfined, unconfined_rock | `test_type`, a `StrengthSpecimen` per specimen (confining, peak deviator, strain at peak, pore pressure, stress ratio, densities, dimensions), the envelope c and φ, qu, su, the envelope or stress-strain points |
| `CompactionResult` | compaction | the Proctor points, maximum dry density, optimum water content, method, mould volume, layers and blows |
| `CBRResult` | cbr | CBR and the 0.1/0.2 in values, swell, soaking, surcharge, the penetration points |
| `MoistureDensityResult` | moisture_content, density, organic_content | water content and each determination, bulk and dry density, Gs, e, saturation, ash and loss on ignition |
| `ChemicalResult` | chemical | pH, resistivity (as received and minimum), sulfate, chloride, sulfide, redox, total salts, conductivity, temperature |
| `SummaryTableResult` | summary_table | one `SummaryRow` per specimen, each with its own boring, depth, index values, sieve columns and an `other` list for a column with no field |
| `OtherResult` | other, specific_gravity, permeability | `no_results` for a page that reports none, and whatever the sheet printed |

Every result field is a `Quantity` in the unit the sheet printed, or a plain
number where the value has none — a percentage, a pH, a blow count, Cu, Cc.
A chemical value takes a **string** as well, because these sheets print `<10`,
`Nil` and `trace` as often as figures: below the reporting limit is not the
number ten, and a record that stored ten would be *wrong* rather than
incomplete. `si_numbers()` walks any part of the record and returns the numbers
in it, in SI, which is how a check can ask "did these survive" without a second
copy of whatever mapping a writer uses.

## The log reader (`log_reader.py`)

`read_log(doc, item_pages, engine, budget=6)` turns one log — continuation
sheets included — into one `Investigation`.

**Geometry says where, the model says what.** `log_grid` has already found the
columns, fitted the depth ruler and placed every line of text at a (column,
depth) with its box. Those rows are the primary source. The page image at about
110 dpi is for resolving only what the rows leave ambiguous: sample symbols,
water symbols, refusal notation, stacked drives, which of two numbers is the N
value. Every value the picture settled comes back in `changes`, because rows can
be gone back to and a look cannot.

**The depth gate is the point.** A ruler is a linear map from y to depth, so it
says what depth sits at the top edge of the paper and what sits at the bottom;
everything printed on that sheet is between them and nothing else is. A depth
outside that window is refused into `unresolved` rather than accepted. A log
whose pages have no ruler carries no depths at all — the header fields still
come back, because the page still says whose log it is. Total depth is the one
exception and is not gated: on sheet 1 of 3 it names the bottom of the whole
hole, below that sheet's own paper.

**Nothing is computed.** `n` is recorded only where the log prints it; where it
prints only the drives, `n` stays `None`. Half the templates do one and half the
other, and adding the second and third drives of a four-drive rock core would
invent a number. A refusal stays the string the log printed.

**The budget** is one call per log, a second only when the reader itself says it
has pages left, and six at the most.

## The lab reader (`lab_reader.py`)

`read_lab_sheet(doc, item_pages, engine, budget=4, hint_kind=None)` turns one
laboratory sheet into typed `LabTest` records — several, when the sheet reports
several specimens or two different tests on one.

A boring log is a form with a depth ruler, and `log_grid` can say *where* every
number sits before any model sees it. A lab sheet is not that: every laboratory
prints its own form, half of them are a plot with a results box beside it, and
what says a number's meaning is the word printed next to it. So the geometry is
weaker, the reading is more of the work, and the rules are stricter.

**The sheet's own title says what the test is** — not the appendix tab, not what
the numbers look like. The kind vocabulary is seventeen kinds each with a
one-line definition written in no single laboratory's words, and each definition
is about *what was measured* rather than how the sheet looks.

**A tabulated value beats the plot, every time.** Most of these sheets print the
curve **and** the values it was drawn from. A curve is digitised only where the
values appear nowhere in text, and then the whole test is flagged
`curves_digitised` — as a whole, because a reviewer checks the sheet, not one
number. To digitise, the reader calls **`zoom_plot`**, its one tool: a crop of
the plot rendered at 300 dpi with the axes and their tick labels inside the box,
because a curve read without its own ticks in view is a guess.

**The link to the ground is what the sheet prints** — the boring identifier and
the depth, copied. This reader never matches a sample to a log; the reconciler
does that later and records a conflict when it cannot.

**A summary table is one test**, with a row per specimen, because the table is a
thing the report prints and splitting it would lose which values were printed
together. **A certificate that lists samples and reports nothing** is an
`OtherResult` with `no_results` — a page read and found empty and a page skipped
are different things.

**Four Python gates**, each on something that cannot be true of a real sheet: a
depth outside 0–300 m, a percentage outside 0–100, a liquid limit below the
plastic limit (all three limits go, because which of them is wrong cannot be
known from here), and a grading series in which *more* passes a smaller sieve
(the series goes whole — a partly reversed grading is worse than none, because
it looks like a reading). Each refusal costs that value and nothing else.

**The budget** is four model calls and most sheets cost **one**: every call asks
for the answer and offers the tool at the same time, so a tabulated sheet is
read and answered in a single call. On the last allowed call the tool is
withdrawn, so a reader that keeps zooming runs out of looking rather than out of
answering.

## The DIGGS writer (`diggs_writer.py`)

`write_diggs(record_or_investigations, project) -> str` produces DIGGS 2.6 XML.
`diggs_schema_gate(xml)` validates it against the schema pydiggs bundles;
`diggs_roundtrip_gate(xml, investigations)` reads it back with the app's own
`parse_diggs` and compares every value. Neither gate alone is enough: a file can
be XSD-valid and wrong, and a file the parser likes may not be DIGGS.

**The 2.6 schema is not the shape the app's older fixtures use**, and this was
checked before anything was written — the app's own fixture fails the schema on
its second line. In real DIGGS the test procedures (`DrivenPenetrationTest`,
`AtterbergLimitsTest`, `WaterContentTest`, `LabDensityTest`, `ParticleSizeTest`,
`UnconfinedCompressiveStrengthTest`, `PocketPenetrometerTest`) are in the
`.../2.6/geotechnical` namespace and carry no result; the value lives in
`Test/outcome/TestResult/results/ResultSet`, named by a `propertyClass` from the
DIGGS dictionary; a depth is a position along the hole's own linear reference
system; lithology is an `observation/LithologySystem` at the document root; and
there is no `WaterLevelObservation` or `MoistureContent` element at all — water
is the borehole's own `waterStrike`. So the writer emits real DIGGS and
`subsurface_characterization/diggs26.py` was added to read it, additively, after
the flat readers. Every existing fixture reads exactly as before.

The element map, record field to DIGGS:

| Record | DIGGS 2.6 |
|---|---|
| `Project` | `project/Project` (`gml:name`, `gml:identifier`) |
| `Investigation` (boring) | `samplingFeature/Borehole` |
| `Investigation` (test pit) | `samplingFeature/TrialPit` |
| `x`, `y`, `elevation` | `referencePoint/PointLocation/gml:pos` |
| `total_depth` | `totalMeasuredDepth` (uom m) |
| `date_started`, `date_finished` | `whenConstructed/TimeInterval` |
| `drilling.method`, `.equipment` | `constructionMethod/BoreholeConstructionMethod` |
| `fields` (anything else printed) | `otherSamplingFeatureProperty/Parameter` |
| `Sample` | `samplingActivity/SamplingActivity` + `sample/Sample` |
| `Sample.top`, `.bottom` | `samplingLocation/LinearExtent/gml:posList` |
| `Sample.recovery` | `totalSampleRecoveryLength` |
| `Sample.recovery_percent` | `otherSamplingActivityProperty/Parameter` — DIGGS has only a length |
| `Sample.rqd_percent` | `samplingActivityRQD` (uom %) |
| `Layer` | `observation/LithologySystem/lithologyObservation/LithologyObservation` |
| `Layer.uscs`, `.description` | `Lithology/classificationCode`, `/lithDescription` |
| `SPT.n` | `ResultSet` `propertyClass` **n_value** |
| `SPT.blows` (no printed N) | `ResultSet` `propertyClass` **blow_count** |
| `SPT.blows`, `.refusal` | `DrivenPenetrationTest/driveSet/DriveSet` (`blowCount`, `penetration`) |
| `drilling.hammer_type`, `.hammer_energy_ratio` | `hammerType`, `hammerEfficiency` |
| `WaterLevel` | `Borehole/waterStrike/WaterStrike` |
| `Sample.water_content` | **water_content_natural**, `WaterContentTest` |
| `Sample.dry_unit_weight` | **dry_density**, `LabDensityTest` |
| `Sample.liquid_limit`, `.plastic_limit`, `.plasticity_index` | **liquid_limit**, **plastic_limit**, **plasticity_index**, `AtterbergLimitsTest` |
| `Sample.fines_percent` | **percent_fines**, `ParticleSizeTest` |
| `Sample.qu` | **compressive_strength_unconfined**, `UnconfinedCompressiveStrengthTest` |
| `Sample.pocket_pen` | **compressive_strength_unconfined**, `PocketPenetrometerTest` |

The laboratory half of the map (WP3). A `LabTest` becomes one `measurement/Test`
for its scalars and one more for each curve it carries, both positioned in the
same hole at the same depth; a summary table becomes one Test per **row**,
because its rows sit in different holes.

| Record | DIGGS 2.6 |
|---|---|
| `LabTest.kind` | the procedure element (below) |
| `LabTest.standard` | `testProcedureMethod/Specification/standardReferenceNumber` |
| `LabTest.sample_id`, `.lab`, `.date`, `.pages` | `otherMeasurementProperty/Parameter` — named, not linked, because an xlink to a `Sample` this file may not hold is a link to nothing |
| `AtterbergResult` | **liquid_limit**, **plastic_limit**, **plasticity_index**, **shrinkage_limit**, **non_plastic**, `diggs_geo:AtterbergLimitsTest` |
| `AtterbergResult.flow_curve`, `.pl_trials` | a result set of many rows: **blow_count** + **water_content_natural** |
| `GradationResult` fractions | **percent_cobbles**, **percent_gravel**, **percent_sand**, **percent_silt**, **percent_fines**, `diggs_geo:ParticleSizeTest` |
| `GradationResult` D-values | **d10**, **d30**, **d50**, **d60**, **d85** (dictionary), *d90*, *d100* (ours), in **mm** |
| `GradationResult.cu`, `.cc` | **coef_uniformity**, **coef_curvature** |
| `GradationResult.percent_passing` | a result set of many rows: *particle_size* (mm) + *percent_passing* + *sieve_designation* |
| `ConsolidationResult` | **preconsolidation_pressure**, **compression_index**, **recompression_index**, **coef_consolidation_vertical**, *void_ratio*, *swell_percent*, *swell_pressure*, `diggs_geo:ConsolidationTest` (+ `consolidationTestType`, `swellingPressure`, `estimatedPreConsolidationStress`) |
| `ConsolidationResult.points` | a result set of many rows: *applied_pressure* + *axial_strain* and/or *void_ratio* + *load_stage* |
| `StrengthResult` | **compressive_strength_unconfined**, **shear_strength_undrained**, **cohesion_peak**, **friction_angle_peak**, **cohesion_residual**, **friction_angle_residual** |
| `StrengthResult` procedure | `TriaxialTest` (**the DIGGS namespace**), `diggs_geo:DirectShearTest`, `diggs_geo:UnconfinedCompressiveStrengthTest` |
| `StrengthResult.specimens` | a result set of many rows, one per specimen |
| `StrengthResult.points` | a result set of many rows, told apart by the x unit: an envelope when x is a pressure, a stress-strain curve when x is a percentage |
| `CompactionResult` | **dry_density_max**, **water_content_optimum**, `diggs_geo:LabCompactionTest` (+ `mouldVolume`, `numberOfLayers`, `blowsPerLayer`); the points as a many-row result set |
| `CBRResult` | **cbr_0.1**, **cbr_0.2** in `diggs_geo:LabCBRTest/trial`, plus *cbr*, *swell_percent*, *surcharge* |
| `MoistureDensityResult` | **water_content_natural**, **bulk_density**, **dry_density**, **specific_gravity_solids**, **degree_of_saturation**, **LOI**, *ash_content*; `WaterContentTest`, `LabDensityTest` or `LossOnIgnitionTest` by kind |
| `ChemicalResult` | **pH**, **resistivity**, **sulfate_content**, **chloride_content**, **redox_potential**, **conductivity**, **temperature**, *sulfide_content*, *total_salts*, `diggs_geo:LabChemicalTest` |
| `SummaryRow` | one Test per row, values under the dictionary terms above, with no procedure element — a row is one sample's results gathered out of several tests, and naming one procedure for it would say the laboratory ran a test it did not |
| `OtherResult` | `diggs_geo:SpecificGravityTest`, `diggs_geo:LabPermeabilityTest`, or a Test with no procedure |

**Bold** is a term of the DIGGS property dictionary pydiggs publishes;
*italic* is one of ours, written under a codespace that says so, because the
sheet printed the value and dropping it would be worse than naming it plainly.

Four facts about 2.6 that the code exists to get right:

- **`TriaxialTest` is in the DIGGS namespace, not the geotechnical one.**
  Twelve of the thirteen laboratory procedures are in `diggs_geo`; the schema
  declares this one in `TestProceduresAll.xsd`. A file that puts it in the other
  namespace validates against nothing and reads as empty.
- **A `Test` has exactly one `outcome`**, so a gradation with both derived
  fractions and a grading curve is two Tests, not one Test with two result sets.
- **A curve is a result set of many rows**, whatever kind of curve it is. 2.6 has
  native homes for several, and every one demands a value these sheets do not
  print — a `Grading` requires a particle size and half the forms label their
  sieves by number alone, a consolidation increment requires a final axial
  deformation, a triaxial shear stage a cell pressure. Filling a required
  sibling with a made-up number to reach a nicer element is the one thing this
  writer will not do.
- **A result is positioned.** `TestResult/location` is required and is a
  position along a hole's own linear reference system, so a lab test that names
  no boring, or names one and no depth, **cannot be written**: there is nowhere
  in the file for it to be. Those are named in `DiggsWriteNotes.skipped` and
  stay in the record, which does not require a value to have a place. A test
  that names a boring no log in the record describes gets a minimal `Borehole`
  of its own, recorded in `DiggsWriteNotes.synthesised`.

A particle size is written in **millimetres** and says so in its `uom`: in
metres a No. 200 sieve is 7.5e-05, which the file's four decimals would round
to 0.0001. A friction angle is written `dega`, which is the schema's own code;
`deg` is not in its list and fails the whole file.

`subsurface_characterization/diggs26.py` gained the reading side: a result set
of more than one row is a curve and is **not** flattened into measurements at
one depth, and `parse_diggs26_result_sets(content=...)` returns every result set
in a file as the table it is — columns, units, rows, dictionary terms and ours,
numbers and the values a laboratory printed as words. That is what the
round-trip gate compares against for anything a `SiteModel` has no shape for.

SI once, here, with a `uom` on every measure. A coordinate gets nine decimals,
because a latitude to four is eleven metres. A unit not in the record's
conversion table is **not** written with a guessed one: it is left out and named
in `DiggsWriteNotes.skipped`, which the round-trip gate then does not look for.
A layer whose base the sheet never printed is written as a contact and named the
same way — DIGGS can carry it, the app's `LithologyInterval` holds intervals
only.

**The gate that matters is deterministic and needs no model.** All fifteen
hand-truthed logs and all **thirty-one** hand-truthed laboratory sheets convert
to records, write, validate and round-trip:
`module_work/report_ingest_harness/tests/test_diggs_truth.py` and
`test_lab_diggs_truth.py` (skipped where the gitignored corpus is not).
`report_ingest/tests/test_diggs_writer.py` and `test_diggs_lab.py` walk the same
path on synthetic records in CI.

**What the 31-sheet gate found, and the tolerances it needed.** It passes
31/31 at the tolerances the writer already published — a depth to 5 mm, a
percentage to 0.05, a stress to 0.05 kPa — with **no tolerance loosened for
the laboratory half**. The lab comparison is tighter than that: every number in
the record has to come back to within 5e-4, or six significant figures for a
value below a thousandth. Getting there took three fixes rather than three
tolerances: writing a particle size in millimetres (in metres a No. 200 sieve
rounds to 0.0001 at the file's four decimals), writing small numbers to six
significant figures instead of four decimals, and using the schema's own `dega`
for an angle. Two of the thirty-one sheets write **no** DIGGS at all, because
neither prints a depth for the specimen; the gate asserts which two they are, so
a third appearing is a failure rather than a quietly smaller number. The lab
half of the round trip walks the record's own models rather than a second copy
of the writer's property table, so it cannot agree with the writer by sharing
its mistakes: it asks whether the numbers survived, not whether they were filed
under the names the writer chose.

**The numbers come from the cluster.** This app runs in Funhouse against
OpenAI models through the Prompter API, so a score measured on any other model
measures a model that will never do the work. `PrompterEngine` is the engine
that counts and `cluster_scoring.score_on_cluster` is how a run is made.
`ClaudeEngine` stays in the package as a development engine — it is what the
passes were built and debugged against — but its numbers are a checkpoint, not
a result.

## Why the two label passes exist

The rules label a page from what that page prints about itself. Measured on the
corpus, they reach 0.91 accuracy across the fourteen reports they were built on
and 0.79 on the next report anyone opens. The gap is structural, not a missing
rule: a page only makes sense in the context of the rest of the report. An
appendix tab that says "LABORATORY TEST DATA" is right about most of its
appendix and wrong about the summary table, the legend sheet and the stray
calculation bound into it, and no amount of per-page cleverness fixes that.

So both passes are given everything the report says about **itself** before any
of it is read in detail.

## What each pass reads and returns

### `triage(doc, roles, outline, engine=...) -> DocumentProfile`

One call. It is given:

- the per-page ledger, one line a page (`planlens.document.roles.page_ledger`);
- the outline the document prints: contents, lists of figures, tables and
  appendices, every divider's text, figure captions, section headings;
- the front matter verbatim, through the end of the contents;
- the first contact sheet as a picture;
- **the facts Python already counted** — the scanned fraction, the
  unreliable-text fraction, where the printed page numbering restarts, the page
  kinds and the rule labels.

That last point is the rule of the module. A model asked to count 455 ledger
lines will be nearly right, which is the worst thing a scorecard field can be.
Those fields are measured by `document_facts()` and copied into the profile;
`TriageFindings` — the schema the model is held to — does not contain them.

It returns a `DocumentProfile`: `document_type` (the owner's enumeration),
`bound_together`, `has_narrative` / `has_logs` / `has_lab` / `has_calcs`,
`languages`, `toc_agreement`, `anomalies`, `rationale`, the measured fractions,
and a `workflow` in `standard`, `appendix_only`, `partial`, `multi_document`,
`scanned`, `needs_person`. The workflow chooses the readers that follow, so an
odd file is flagged rather than forced down the standard path.

### `review_labels(doc, roles, outline, profile, budget, engine=...) -> Review`

An agent loop. Its tools are `read_page(page)` (the toolkit's text for one
page), `render_page(page)` (the page as a picture at 80 dpi),
`contact_sheet(start_page)` (48 thumbnails, each labelled with its page index
and kind) and `outline()`.

Its brief carries the label vocabulary with one-line definitions, the ledger,
the outline, the triage profile, and **the rules' own weak spots as the list to
check first** — every page whose label came from an appendix tab rather than the
page, every page the tab left ambiguous, every page guessed from its shape,
every page with no readable text, and every page whose label contradicts both
its neighbours. A document with no appendix tab anywhere is said so in the first
line, because then nothing inherited and every label is weak.

**Budget:** `max(60, 0.25 × pages)` tool calls. When it is gone, further calls
come back as errors telling the model to report what it has, and
`Review.stopped_on_budget` records it. `max_model_calls` (default 60) stops the
loop itself.

It returns a `Review`. The model supplies `changes` (page, from, to, a reason of
25 words or fewer, and which tool showed it), `structure` (section or appendix
title → the page range it actually occupies) and `unresolved` (pages it could
not settle, with why).

`final_labels` is **not** asked of the model: it is the rules' labels with the
accepted changes applied, in Python. A model re-emitting 455 page-to-label pairs
can drop a page or shift a number silently, and the scorecard would then be
scoring the slip rather than the judgement. Four kinds of change are refused and
recorded in `rejected_changes` instead of applied: a page that does not exist, a
label outside the vocabulary, a second change to a page already changed, and a
change to the label the page already has. A change's `from` is always overwritten
with what the rules really said, so a model that misremembers is visible.

## Running it on the cluster

This is the run that produces the real numbers, and the owner makes it.

**The reports stay where they already are.** They do not have to be renamed,
copied or re-uploaded: point `reports_dir` at the folder that holds them under
the names their authors gave them, and the manifest's source-file column is
what turns each file name into an ID. Three small private files travel with
the run and can sit anywhere the cluster can read:

| File | What it is | Without it |
|---|---|---|
| `MANIFEST.md` | the corpus manifest; its `file` column names each report | IDs cannot be resolved at all, unless the PDFs are already named `R01.pdf` … |
| `trial_pages_working_r2.xlsx` | the hand page labels | the in-sample set runs and produces profiles, but scores nothing |
| `oos_labels.json` | the lead's out-of-sample labels | the two out-of-sample sets run and score nothing |

The Azure Document Intelligence results are read in **either** form —
`<ID>.json.gz` or the uncompressed `DI_data_<original stem>.json` that
Funhouse wrote — from whatever folder `di_dir` names. Without them the scanned
pages are read from their own text layer alone.

```python
# 1. From PyPI through Nexus. planlens 0.5.0 arrives with it.
%pip install "geotech-staff-engineer==5.19.0"
dbutils.library.restartPython()
```

```python
# 2. One cell. fh_prompter is the object you already have.
from report_ingest.cluster_scoring import score_on_cluster

results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",          # the PDFs, under their own names
    manifest     = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    labels_xlsx  = "/Volumes/<your volume>/wp1b/trial_pages_working_r2.xlsx",
    oos_labels   = "/Volumes/<your volume>/wp1b/oos_labels.json",
    di_dir       = "/Volumes/<your volume>/report_di",
    out_dir      = "/tmp/report_ingest_wp1b",                 # /tmp or a Volume, never /Workspace
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",     # the label review, on the tier the app runs on
    triage_model = "funhouse-gpt-medium",   # one call over a ledger; a cheaper tier does
    sets         = ("insample", "oos_open", "oos_blind"),
    stages       = ("labels",),             # or ("labels", "logs", "lab")
    max_reports  = 2,                       # drop this line after the first run
)
```

### Scoring the log reader as well (`stages=("labels", "logs")`)

The `logs` stage runs `log_grid` and then `read_log` over each hand-truthed
log, and scores the result twice: **before**, from the grid's cells alone, and
**after**, from the record the reader built. The grid runs once and is handed
to the reader, so the difference between the two columns is the model and
nothing else. It needs one more private file — the folder of hand-truthed logs
— and an `OPEN.txt` beside them names the reports the rules were allowed to be
tuned on, so the blind figure stays blind.

```python
results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",
    manifest     = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    di_dir       = "/Volumes/<your volume>/report_di",
    truth_dir    = "/Volumes/<your volume>/wp2b/truth/logs",   # <ID>_p<page>.json + OPEN.txt
    out_dir      = "/tmp/report_ingest_wp2b",
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",
    stages       = ("logs",),       # or ("labels", "logs") for both in one run
    log_budget   = 6,               # model calls per log; the reader's ceiling is six
    max_reports  = 2,               # drop this line after the first run
)
```

Metrics, all against the hand truth: **N values exact**; sample depths,
index values and water levels within 0.15 m; layer tops within 0.3 m; USCS
symbols matched; recovery and RQD where the log prints them; header fields
recovered. Depths are compared in metres whatever the log prints, and the
matching rules are the WP2a ones — a sample is an interval, a blow record
counts when its drives stand at that depth, and an N the log never printed
counts as found when the drives that define it are there, because neither the
grid nor the reader does arithmetic by design.

### Scoring the lab reader as well (`stages=("labels", "logs", "lab")`)

The `lab` stage scores each hand-truthed laboratory sheet twice: **before**,
over the numbers in the page's own detected tables, and **after**, over the
records `read_lab_sheet` built from the same open document.

```python
results = score_on_cluster(
    reports_dir   = "/Volumes/<your volume>/reports",
    manifest      = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    di_dir        = "/Volumes/<your volume>/report_di",
    truth_dir     = "/Volumes/<your volume>/wp2b/truth/logs",
    lab_truth_dir = "/Volumes/<your volume>/wp3/truth/lab",  # <kind>__<ID>_p<page>.json + OPEN.txt
    out_dir       = "/tmp/report_ingest_wp3",
    prompter      = fh_prompter,
    model         = "funhouse-gpt-high",
    stages        = ("labels", "logs", "lab"),
    lab_budget    = 4,              # model calls per sheet; the reader's ceiling is four
    max_reports   = 2,              # drop this line after the first run
)
```

Five metrics: **kind** (the test, from the sheet's own title), **link** (the
boring identifier *and* the depth printed on the sheet, within 0.15 m compared
in metres), **index** (every scalar the sheet printed, exact), **series** (a
grading's percent-passing values within 1 %), **curve** (a plotted curve's
points within the tolerance that sheet's own truth file states).

**`kind` and `link` have no before column at all.** A table is a grid of
numbers: it does not know that 31 is a liquid limit, that the sheet is a direct
shear test, or that the specimen came from B-1 at 2.5 ft. Crediting it with any
of those would score the reader's job against the extractor's output, so the
baseline is asked only "is this number on the page", which is the most a table
can answer. Read the metrics rather than the OVERALL rows.

**The tables-alone baseline, measured here 2026-09-17** with no model, no
credential and no network (`measure_wp3_lab.py --tables-only`, and in the
ledger): **68 %** of the numbers are somewhere in a detected table (452/667) —
83 % on the open sheets and 55 % on the blind ones, which are the scanned,
optically-read and rotated pages where no table is detected at all. Two kinds
score **zero**: every triaxial (0/64, all three of them) and the rock core
(0/5), whose pages either have no usable text layer or print their values as
nested stages no table detector groups. A grading's percent-passing series is
the one thing a table does well (88 %), and a curve is what it does worst
(35 %). That gap, plus the two metrics a table cannot answer at all, is what
the reader is for.

`RESULTS.md` then carries a second half: before and after per metric for the
open set, the blind set and all logs; a per-log line with model calls, what was
left unresolved and how many values came off the picture rather than the rows;
and the cost per log. `logs/<log id>.json` holds the per-log detail and, like
the label runs, makes the stage restartable — a log that already has one is
skipped.

Locally, `module_work/report_ingest_harness/measure_wp2b_logs.py` does the same
scoring against the development engine, and `--grid-only` prints the before
column with no engine, no key and no network at all.

Bring back **`/tmp/report_ingest_wp1b/RESULTS.md`**. That is the whole report,
and it carries IDs, labels, counts and rates only. The per-report runs and the
triage profiles stay on the cluster in `runs/` and `triage/` unless you move
them deliberately, because a change's reason and a triage rationale can name a
firm, a project or a person.

Notes that matter:

- **`out_dir` must be `/tmp` or a Volume.** A `/Workspace` path is refused up
  front rather than discovered at the end, because those writes are
  non-durable and permission-blocked here (`docs/DATABRICKS_INSTALL.md`).
- **It resumes.** Each report writes its run file as it finishes and a later
  call skips any report that already has one. A detached notebook costs the
  reports that had not finished, not the ones that had. Pass `redo=True` to
  start over, or delete one file to redo one report.
- **Start small.** `max_reports=2` on the first run proves the path end to end
  for the price of two reports.
- **A report the folder does not have is named and skipped**, once, before
  anything runs, rather than failing one report at a time deep in the set.
- **No credential is read.** Authentication is whatever `fh_prompter` was built
  with. Nothing here touches an environment variable or a secret scope.
- **Tokens, not dollars.** Funhouse publishes no per-token price for a
  capability tier, so the run reports tokens and you read the spend from
  Funhouse's own budget endpoint for the same window.

## What the Prompter engine can and cannot do

| | Prompter (OpenAI through Funhouse) | Claude API (development) |
|---|---|---|
| Multi-turn tool loop | yes, through `prompter.client` | yes |
| Images | yes, as `image_url` data URIs | yes |
| Structured output | yes, `response_format` with a strict JSON schema | yes |
| Image inside a tool result | **no** — see below | yes |
| Explicit prompt caching | no; the provider caches what it caches | yes, and it is asked for |
| Budget guard on every call | only on single-shot calls | not applicable |

Two of those need explaining because they changed the code.

**An image cannot go in a tool result.** OpenAI's `role="tool"` message takes a
string and nothing else, and the label review's `render_page` and
`contact_sheet` answer with a picture. So an image-bearing result is sent as a
tool message saying the picture follows, and the picture rides in a user
message immediately after the tool messages. The model sees both and the
ordering rules are kept. On the Claude API the picture simply goes in the tool
result, so the same review does slightly different work on the two engines —
worth remembering when comparing their numbers.

**The tool loop bypasses the SDK's budget guard.** Prompter's own `chat()`
sends exactly one system and one user message, which cannot express an
assistant turn with tool calls, so the loop drives `prompter.client` directly
exactly as the app's `NativeToolEngine` does. Single-shot calls with no tools
still go through `chat()`, so triage keeps the guard and the SDK's logging and
only the review is outside it. On a backend where `prompter.client` is `None`
(Grok) there is no loop at all, and the engine says so rather than failing
deep in a run.

## The engine, and where it runs

Neither pass imports an SDK. Both take an `Engine` — one method:

```python
complete(messages, system=..., tools=..., images=..., output_format=...) -> Reply
```

Messages are neutral blocks (`text`, `image`, `tool_use`, `tool_result`,
`opaque`), and the engine translates to and from its provider's shape in both
directions. `opaque` is how a pass carries a thinking block back to the model
unchanged without ever reading it. So the same passes run on the Claude API here
and on the cluster's Prompter later, behind the app's own engine abstraction.

`ClaudeEngine` is the development implementation. **The credential comes from
the environment** — `ANTHROPIC_API_KEY`, or an `ant auth login` profile. It is
never an argument, never logged and never printed; only its presence is checked,
and a missing one fails at construction rather than deep inside a run. The
prompt prefix is cached by default (`auto_cache=True`), which on the checkpoint
turned roughly 1.4 M resent tokens into cache reads at a tenth the price.

`CostMeter` records calls, input and output tokens, cache reads and writes, wall
clock and dollars at the list prices in `MODEL_PRICES` (checked 2026-09-16).
Prices are pinned in the source rather than fetched: a scorecard has to be able
to say what a run cost months later.

## Running the scorecard

```
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set checkpoint --note "what changed this round"
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set insample --reuse --changes
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set oos_blind --no-append
```

The log and lab readers have their own scripts, each of which runs its
deterministic half with no engine at all:

```
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp2b_logs     --grid-only
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp3_lab     --tables-only --append --note "what changed this round"
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp3_lab     --kind gradation --detail
```

Sets are `insample` (the reports with spreadsheet labels), `oos_open` (the
lead's out-of-sample labels on ten reports), `oos_blind` (the other fourteen)
and `checkpoint` (six reports chosen to span short, long, scanned and old).
Running the first three covers all 38 corpus reports exactly once, which is also
how every report gets a triage profile. `--reuse` re-scores the saved runs
instead of calling the model, so tuning the scorecard is free.

Each run writes the full private detail to `raw/checks/wp1b/` and each report's
profile to `raw/checks/triage/`, both gitignored. The ledger gets IDs, labels,
counts and rates only, never a heading, a change's reason or a triage
rationale.

**Every change is graded** against the hand label as `fixed`, `broke`,
`still_wrong`, `disputed` or `unscored`. A review that raises accuracy while
breaking three correct labels has not earned the raise, and a total alone would
hide it.

**The blind set is protected and reported honestly.** `oos_blind` prints one
summary, writes no miss list and no change list, and refuses `--changes`. Two of
its fourteen reports were named in the cost checkpoint and their changes were
read, so the set is reported twice: over all fourteen, and over the twelve
nobody has opened. The second figure is the honest one and the output says so.

**Disputed hand labels.** When the review contradicts a hand label and the lead,
looking at the page, judges the hand label the doubtful one, the entry goes in
`DISPUTED`. The spreadsheet is never edited: a hand label records what a person
decided, and rewriting it to suit a model would destroy the only independent
thing in the measurement. A confirmed dispute drops the page from the before
**and** the after score, so it counts as neither a rule hit nor a review miss,
and the number of dropped pages is printed. It only drops while the review still
says what it said when the dispute was raised; a later run that moves the page
somewhere else is a new answer and is scored.

**Three guards, each from a run that went wrong.** A run in which every report
fails no longer overwrites the last good run's detail file. Every saved run
records a hash of the prompts that made it, and scoring a set whose saved runs
disagree refuses to stand behind the totals — half a set on new prompts and half
on old is a set that never existed. And `--max-report-dollars` (default $5) and
`--max-total-dollars` (default $45) stop the set rather than spend past them,
naming the report they stopped before.

## Tests

```
.venv/Scripts/python -m pytest report_ingest/tests -q
.venv/Scripts/python -m pytest module_work/report_ingest_harness/tests -q
```

Offline and free: no credential, no network, no corpus. `FakeEngine` replays a
script of turns in the engine's place, so the whole of both label passes and the
log reader runs for real — the prompt each builds, the tool loop, the budget
stop, the rules for applying a change. Running past the end of a script raises,
because a pass that makes one more call than the test expected is the bug the
test exists to catch.

The log reader's tests run over real `log_grid` output on planlens' synthetic
log fixtures, and the lab reader's over real located text and real detected
tables on four synthetic sheets built here (`tests/lab_fixtures.py`: a
plasticity chart with its limits in a box, a grading curve with its values
tabulated beneath it, a summary table of four specimens, a laboratory
certificate that reports nothing). So the brief that is asserted on is the brief
a model would actually be sent, and the refusals are the refusals that would
actually happen: a depth past the ruler, a log with no scale, a provenance
naming a page outside this sheet, a liquid limit below the plastic limit, a
grading running the wrong way.

The DIGGS writer is tested against the bundled 2.6 schema and through
`parse_diggs` on a synthetic investigation carrying one of everything, and on
synthetic lab records carrying one of every kind — including the namespace trap,
the millimetre rule, the `dega` code, and the two reasons a lab test cannot be
written at all. The two scorers are tested on synthetic truth where what is
checked is the matching itself, and — the important one — that the scorer does
**not** ask for a date, a description, a link field twice, or a value in the
wrong unit, because a scorer that over-asks turns a good reader into a bad
number.

The harness suite adds the scorecard's own arithmetic, the disputed-label rule,
the prompt fingerprint, the deterministic fifteen-log DIGGS gate, the
thirty-one-sheet lab gate, and the floor under the lab scorecard: a flawless
reading of all thirty-one sheets must score 100 %, or the scorer is what is
wrong. Its corpus-dependent tests skip cleanly on a machine without the private
data.

## What the measurements have taught the prompts

Kept here because each line is a prompt rule that exists for a measured reason,
and a later editor who does not know why will delete it.

- **The four exploration labels say they may be PLOTTED, not just tabulated,
  and that each belongs to one named exploration.** On the cost checkpoint,
  fifteen of the review's nineteen wrong changes were an exploration's own
  results called `figure`, or one kind of sounding called another. The rules had
  been wrong about those pages and the review was right to move them; it moved
  them next door. A page showing depth against blow count looks like a graph,
  and nothing had said that a graph of one exploration's results is that
  exploration's log.
- **`figure` says it is the report's own numbered figure series and never an
  exploration's results**, for the same reason, from the other side.
- **The review is told to choose among the four logs on what is MEASURED, not on
  how it is drawn**: blows per increment of penetration is a DCP, continuous tip
  and sleeve and pore pressure is a CPT, a logged pit or trench face is a test
  pit, a drilled hole with driven samples and blow counts is a boring.
- **The brief leads with the rules' own weak spots.** The out-of-sample study
  sorted every rule miss into four classes and three are answerable only by
  opening the page, so those pages are named first and the rest of the document
  follows.
- **The deterministic facts are given, never asked.** A model asked to count 455
  ledger lines will be nearly right, which is the worst thing a scorecard field
  can be.

## How this package ships

It ships in the wheel as of **5.19.0** (2026-09-17) as a library: no tool of
the app calls it yet, and nothing in the chat surface changed.

- **`report_ingest*` is in `[tool.setuptools.packages.find]`.** Until that line
  landed, a wheel built from this repo did **not** contain this package, and
  the failure would have appeared on the cluster as an import error rather
  than at build time. Check it after any edit to that list:
  `python -m build --wheel` and look for `report_ingest/` in the wheel.
- **`report_ingest` is in pytest's `testpaths`**, so this suite runs in the
  release gate. The WP0/WP1 harness under `module_work/` is dev-only and is
  deliberately not.
- **`planlens>=0.5`** — the page roles, the printed outline and the per-page
  ledger these passes read are 0.5.0's. planlens 0.5.0 declares exactly the
  dependencies 0.4.0 did, so the pin adds no new package to the cluster.
- `pydantic` and `openpyxl`, both already there: pydantic through the agent
  stack, openpyxl as a hard requirement of `python-ags4`, which this app
  depends on directly. The hand-label reader imports openpyxl lazily anyway,
  so a run that never touches the spreadsheet never needs it.
- `anthropic` is **optional**, and on the cluster not installed at all.
  Nothing outside `ClaudeEngine` imports it, `report_ingest/__init__.py` reaches
  every entry point through a lazy import, and a test spawns a fresh
  interpreter to prove that importing the package pulls in neither `anthropic`
  nor `planlens`.

## How the corpus is read

`corpus.Corpus` takes directories as arguments — the PDFs, the Azure results,
the spreadsheet — because the same corpus is a gitignored folder in this repo
during development and a Volume on the cluster during a real run. The WP0
harness (`module_work/report_ingest_harness/`) is now that loader bound to the
repo's own paths, so there is one implementation and the harness's own tests
check it.

It reads **two layouts**, because the repo's copy was renamed and the
cluster's was not. `R01.pdf` … is one; the reports under their original names
plus a manifest is the other, and an ID is resolved through the manifest's
source-file column — the path as written, then without its leading folder
(`Reports_PDF/`), then on the base name case-insensitively, then on a
punctuation- and accent-insensitive form of the stem. DI results are read as
`<ID>.json.gz` or as `DI_data_<original stem>.json`, gzip preferred where both
exist and the file sniffed rather than trusted from its suffix. Both
resolutions and both DI forms are pinned by offline tests on synthetic files
(`tests/test_corpus_paths.py`), because the cluster run is the one place this
code has to work first time. `scoring.py` holds the rates, the sets and the change verdicts for the
same reason: two copies of that arithmetic would drift, and the second copy's
numbers would be the ones nobody checked.
