# `report_ingest` — a geotechnical report as organised, cited data

Part of the report-ingest train (`module_work/REPORT_INGEST_PLAN.md`). planlens
turns a PDF into pages with kinds, located text, an outline and a first draft of
what each page **is**; this package adds the geotechnical judgement and the
record. WP4 closed the loop: a PDF now goes in one end and a record, a summary,
a library page and a DIGGS file come out the other, with the whole thing
available to the app as one sub-agent. Calculation printouts are WP5.

| Piece | Module | Shape |
|---|---|---|
| 0b. Document triage | `triage.py` | one structured call |
| 0c. Label review | `label_review.py` | an agent loop with four tools |
| The vision experiment | `vision_labels.py` | the page as a picture: one call a page, one a sheet, or one a window of the whole report |
| The record | `model.py` | pydantic; the product everything else exports |
| 2. Log reader | `log_reader.py` | one structured call per log, image alongside |
| 3. Lab reader | `lab_reader.py` | one call per sheet, `zoom_plot` when a curve is only plotted |
| 3. Narrative reader | `narrative_reader.py` | one call for the owner's two schemas, cited |
| 4. Reconciler | `reconciler.py` | pure Python; records disagreements, never settles them |
| 5. Writers | `writers.py` | the record, the summary, the library page, DIGGS, the index |
| The DIGGS writer | `diggs_writer.py` | deterministic, with two gates |
| The whole ingest | `graph.py` | a deterministic loop, resumable per item |
| A folder of reports | `run_folder.py` | the same graph headless, into one library |
| The app's sub-agent | `subagent.py` | a `CompiledSubAgent` and one primary tool |
| Scoring one log | `log_scoring.py` | the grid alone, then the reader |
| Scoring one sheet | `lab_scoring.py` | the page's tables alone, then the reader |
| Scoring the narrative | `narrative_scoring.py` | recall, precision and the flattering one |

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

**The owner's two query schemas are real fields** since WP4 (schema 4.0):
`general` and `natural_hazards` carry them field name for field name, camel
case and all, because an answer is comparable with an answer given three years
ago or it is worth nothing. Everything is optional and **None means the report
did not say** — never an empty string, never "not stated". The typed twins sit
BESIDE the strings rather than replacing them: `bearingCapacityValues` beside
`bearingCapacity`, `strataList` beside `strata`, a `Mention` beside each
yes/no question, and the normalised site class, ASCE edition and ISO date
beside the printed ones. `NarrativeFacts` is what sits around the answers —
what the narrative STATED, what Python COUNTED, what the appendix turned out to
hold, and the caller's own questions. `CalcEntry` is still WP5.
`record_json_schema()` exports the whole thing.

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

## The narrative reader (`narrative_reader.py`)

`read_narrative(doc, narrative_pages, engine, budget=8)` answers the owner's
two query schemas off the narrative's own prose and returns them with a
citation on every answer.

**The names are the owner's, verbatim**, camel case and all, because an answer
is comparable with an answer given three years ago or it is worth nothing.
`general` is the twenty-five-question list (what the document is, who wrote it,
the counts, the strata, the foundations, the bearing pressures) and
`natural_hazards` is the twelve (liquefaction, ASCE 7 edition, the hazards,
the seismic code, the site class, the date).

Four rules the prompt is built around, each of them a way of not inventing:

- **Answer only from the text in front of you.** Not from what a firm of that
  name usually recommends, not from what a site of that description usually is.
- **Null is an answer and the commonest one.** A field the report does not
  address stays null. It never becomes an empty string or "not stated", because
  a scorer has to tell a report that did not say from a reader that did not
  read — and the builder turns "N/A", "unknown" and their friends back into
  null on the way in.
- **The enumerations are the owner's words, and only two lists are closed.**
  `documentType` and `outsideProject` are the owner's own lists and an answer
  outside one is REFUSED into `unresolved`. The rest are the values seen in the
  hand answers so far: a spelling variant folds onto one of them and anything
  else is kept in the report's own words. That split is not fastidiousness. The
  first draft of this package guessed those vocabularies (government owned /
  leased, planning / feasibility / design, high / moderate / low) and every
  guess was wrong against the first hand answers that arrived, so the schema
  refused the true answer, the reader stored nothing, and the scorer marked
  seven fields wrong on all eight reports for a reading that was correct.
- **A public report about somebody else's project names no post.** `postName`
  and `propertyType` stay null and `outsideProject` becomes `"yes"`. Inferring
  a post from a city name is exactly the guess that makes a library of answers
  untrustworthy.

**What Python does rather than ask.** The table and figure counts are counted
off the captions in the main body — a line that LEADS with "Figure 3" and goes
on to name the thing, counted by label so a caption repeated on two pages
counts once — and handed to the model as facts. The model is asked anyway and
a disagreement is recorded, but the number stored is the counted one. The site
class is reduced to its letter here, the ASCE 7 edition to its year, and the
report date to ISO; each is a small deterministic job, and each refusal (a
"site class" that is not a letter, a date that will not parse) is an
`unresolved` line rather than a stored guess.

**The budget is eight calls and a normal report costs one.** The narrative goes
in whole when it fits. A narrative too long for one call is sent in page chunks
— never splitting a page — one call each, and merged in Python: first answer
wins, lists are unioned, and every disagreement between two chunks is recorded
rather than settled. Only a multi-part read spends a final call, and only on
the four prose summaries, which are the one thing a merge cannot do.

**The caller's own questions** ride along: pass `questions=[...]` and each is
answered in the same call under the same rules, with an empty answer where the
narrative does not say. They come back in `record.narrative.extra_answers` and
in the summary page.

## The reconciler (`reconciler.py`)

`reconcile(record, labels=…, items=…, no_text_pages=…, di_pages=…,
reader_unresolved=…)` is the only pass that sees every reading at once, and its
job is the one nothing else can do: put them beside each other and say where
they disagree.

**It never resolves a disagreement.** A summary table that says 31 and a sheet
that says 29 for the same specimen are both recorded, as a `conflict` QA entry
carrying both values and both pages. Picking one would destroy the only
evidence a reviewer has that there is something to look at. The same holds for
a narrative that says four borings over an appendix that carries five: the
count is not corrected, the mismatch is recorded.

| Check | What it does |
|---|---|
| lab → ground | Finds the investigation the sheet NAMES and the sample at its depth, within 0.15 m and unit-aware, and writes the link into `linked_investigation_id` / `linked_sample_id`. The printed values are never touched. A sheet naming a hole this report does not carry is a QA entry, not a link. |
| counts | `boringCount` / `testPitCount` / `cptCount` against the investigations read, and `boringDictionary` / `testPitDictionary` against the identifiers found, both ways: named-but-missing and found-but-unnamed. |
| the summary table | Every value a row and a sheet both carry, compared in SI; only the disagreements are recorded. A value only one side carries is not a conflict, it is a column. |
| units | Every quantity whose printed unit has no conversion, named once with its fields: stored as printed and absent from the SI view. |
| pages | A page whose label says boring log that ended in no work item; a page with no reliable text and no Azure Document Intelligence result. |
| the readers | Every `unresolved` line they returned, carried through as a QA entry. |

`si_view(record)` is the derived all-SI view, walked off the models themselves
so it cannot drift from the record. Pass an `engine` and ONE call is spent
asking for a sentence about each conflict — which value looks like the
misreading, what would settle it. The comment is appended; the values and the
verdict are untouched.

## The writers (`writers.py`)

`write_outputs(record, out_dir, source=…, db_path=…)` writes five things, in
this order, because the DIGGS verdicts are QA entries and the record has to
carry them:

| File | What it is |
|---|---|
| `report.diggs.xml` | DIGGS 2.6 through the existing writer, with BOTH gates run — the bundled XSD and a round trip through the app's own readers, value by value. Each verdict goes into the record's QA, because a file that failed its gate and a file nothing checked must not look the same. |
| `report.record.json` | The record. Everything else is derived from it. |
| `report.summary.md` | The two schemas answered, each with its pages; the bearing values and the profile as tables; the caller's own questions; what was extracted; the QA list. |
| `report.page.md` | The owner's WikiLLM page: front matter (`id`, `report_id`, `title`, `authors`, `year`, `source`, `doc_type`, `tier`, `disciplines`, `topics`, `methods`, `materials`, `standards_referenced`, `confidence`, `status`, `n_pages`, `original_path`), then the summary, the key takeaways, the key parameters, the questions answered, and the record's sections as tables. |
| `reports.db` | One SQLite row per report, keyed by a stable hash of the source file, so a folder of reports becomes searchable and re-ingesting one updates its row instead of adding a second. |

**Every tag is earned by something in the record** — an investigation kind, a
test kind, a USCS symbol, an answered hazard question — and nothing is tagged
because reports usually have it. `doc_type` is the owner's own `documentType`
answer rather than the library's document vocabulary, because that is the
answer this pipeline produces and it is the one that has to stay comparable.
`status` and `confidence` follow what went wrong: a page with no text and no DI
makes the record `needs_ocr`; a conflict or a count mismatch makes it `medium`;
a narrative that barely answered makes it `low`.

## The graph (`graph.py`) and the folder run (`run_folder.py`)

`ingest_report(source, engine, out_dir=…, budgets=…, questions=…,
di_result=…)` is the whole ingest: open (with DI when given) → roles, outline
and ledger → triage → label review → work items from the REVIEWED labels →
one reader per item → reconcile → write.

A deterministic loop, not a planning agent. It is budgetable (`Budgets` is a
ceiling per pass, and a 150-page report is about a hundred calls), testable
offline (every reader takes an engine), restartable (each item writes
`items/<id>.json` as it finishes and a second run picks up where the first
stopped), and it cannot forget an appendix.

**The workflow triage chose decides what runs.** `appendix_only` skips the
narrative reader — a reader pointed at an appendix would answer the owner's
questions off a boring log. `needs_person` stops after triage with a QA entry.
A scanned narrative with no DI result is skipped with a QA entry rather than
run against nothing, because the narrative reader reads TEXT; the log and
laboratory readers look at the page image and work regardless. Calculation
printouts and appended reports are recorded as QA entries saying they were not
read, so a reviewer knows the pages exist and were skipped on purpose.

`run_folder(folder, engine_factory, out_dir=…)` drives the same graph headless
over a folder into one `reports.db`, with an `INDEX.md` of what it came to. It
resumes, one bad report cannot stop it, and two files with the same bytes are
one document sharing one library row — which it says out loud, so a folder of
300 files that makes 297 rows is not a mystery.

## The sub-agent, and how it is gated (`subagent.py`)

The primary agent must never try to read a 400-page report. Its document tools
are built for looking one thing up in a document a person is discussing;
pointed at a whole report they cost a fortune, fill the conversation with pages
of text and still miss the appendix. So the ingest is one delegation and one
tool:

```python
agent = build_deep_agent(model, enable_report_ingest=True)   # default OFF
```

That adds the `report_ingest(source, questions="")` primary tool and a
`CompiledSubAgent` — a one-node LangGraph that runs `ingest_report` and returns
a compact `structured_response`: counts, the paths, the first lines of the
summary, the QA count, the workflow and the caller's answers. Never the record,
which runs to megabytes and would be re-sent on every following turn.

**Both are feature-detected**, not trusted from the pin: the tool is built only
when the INSTALLED planlens publishes `document_roles` and `log_grid`, because
the cluster installs from PyPI and has resolved older than the pin before. On
an older planlens the tool is never advertised and the prompt line that tells
the primary to delegate is never added.

**The engine is the app's own.** `report_ingest.engine.engine_for` digs the
live Prompter out of whatever the host built the agent with; when there is
none, the tool says so rather than reading a report on a model nobody asked
for.

**About the middleware on the spec.** deepagents uses a compiled sub-agent's
runnable AS PROVIDED — for a spec carrying a `runnable` it reads the name, the
description and the runnable and nothing else — so the `ScratchFilesystemGuard`
and `ModelCallBudgetMiddleware` on this spec reach no model and enforce nothing
by themselves. They are there because every other sub-agent in this build
carries the same two and a reader comparing the specs should not have to wonder
which was forgotten. What actually bounds this graph is Python: `Budgets`,
applied per reader, and a graph with no filesystem tool on any model for a
guard to intercept.

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
what turns each file name into an ID.

**One folder of small private files travels with the run**, and the whole of
it is what gets uploaded. The hand truth is private and does not ship in the
wheel, so it is put on a Volume before a run; three separate Volume paths that
must each be right is three chances for one to be stale while the run starts
anyway and scores against it. Upload this, and nothing else:

```
<your volume>/report_ingest/
    MANIFEST.md                      the corpus manifest; its `file` column names each report
    trial_pages_working_r2.xlsx      the hand page labels
    oos_labels.json                  the lead's out-of-sample labels
    truth/
        logs/       <ID>_p<page>.json …, OPEN.txt, BLIND2.txt
        lab/        <kind>__<ID>_p<page>.json …, OPEN.txt
        narrative/  <ID>.json …
```

| File | Without it |
|---|---|
| `MANIFEST.md` | IDs cannot be resolved at all, unless the PDFs are already named `R01.pdf` … |
| `trial_pages_working_r2.xlsx` | the in-sample set runs and produces profiles, but scores nothing |
| `oos_labels.json` | the two out-of-sample sets run and score nothing |
| `truth/logs/`, `truth/lab/`, `truth/narrative/` | that stage refuses to start rather than running unscored |

`truth/` is what `truth_dir` points at: each stage takes its own subfolder out
of it, and the subfolders are named after the stages. `OPEN.txt` beside a
stage's truth files names the reports whose pages the prompts were allowed to
be tuned against, so the blind figure stays blind; without one the built-in
open set is used. (`lab_truth_dir` and `narrative_truth_dir` still override
the root, for truth that is not in one place.)

The Azure Document Intelligence results are read in **either** form —
`<ID>.json.gz` or the uncompressed `DI_data_<original stem>.json` that
Funhouse wrote — from whatever folder `di_dir` names. Without them the scanned
pages are read from their own text layer alone.

```python
# 1. From PyPI through Nexus. planlens 0.6.0 arrives with it.
%pip install "geotech-staff-engineer==5.22.1"
dbutils.library.restartPython()
```

```python
# 2. The restart wiped the fh_* objects; the Funhouse setup notebook puts them
#    back. Use whatever path your own notebooks already use for it.
%run /Workspace/funhouse-sdk/setup_python
```

```python
# 3. One cell, all six stages. fh_prompter is the object setup just made,
#    and fh_sp_client is the SharePoint client the mirror writes through.
from report_ingest.cluster_scoring import score_on_cluster

results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",          # the PDFs, under their own names
    manifest     = "/Volumes/<your volume>/report_ingest/MANIFEST.md",
    labels_xlsx  = "/Volumes/<your volume>/report_ingest/trial_pages_working_r2.xlsx",
    oos_labels   = "/Volumes/<your volume>/report_ingest/oos_labels.json",
    truth_dir    = "/Volumes/<your volume>/report_ingest/truth",   # holds logs/ lab/ narrative/
    di_dir       = "/Volumes/<your volume>/report_di",
    out_dir      = "/tmp/report_ingest_522",                  # /tmp or a Volume, never /Workspace
    sharepoint   = fh_sp_client,            # the durable copy; see "Where the output goes"
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",     # every reader, on the tier the app runs on
    triage_model = "funhouse-gpt-medium",   # one call over a ledger; a cheaper tier does
    sets         = ("insample", "oos_open", "oos_blind"),
    stages       = ("labels", "logs", "lab", "narrative", "vision_labels", "vote"),
    vision_model = "funhouse-gpt-low",      # the experiment: GPT-4.1, the cheap tier
    vision_mode  = "sheet",                 # six pages a call; or "page" / "document"
    vision_fallback = True,                 # one page-mode call for a page nothing answered for
    max_reports  = 2,                       # drop this line after the first run
)
```

**If the tier's model refuses a parameter** — a reasoning-class deployment
wants `max_completion_tokens` rather than `max_tokens` and takes no
`temperature` — the engine resends the call with that parameter renamed or
dropped, once per refused parameter, and keeps the lesson for the rest of the
run (`engine.adaptations` says what it learned). The SDK's `chat()` helper
swallows such a refusal and returns None; that case is retried on the raw
client. This is what stopped the first run on 5.20.0 and is fixed in 5.20.1.

Bring back **`/tmp/report_ingest_522/RESULTS.md`**. Run it with
`max_reports=2` first. It caps every stage at two — two reports reviewed, two
logs, two laboratory sheets, two narratives — which proves the paths, the
prompter and all three truth folders for a few minutes of calls, and it
resumes, so the full run afterwards does not redo them.

### Where the output goes, and why it is mirrored

**`/tmp` does not survive a cluster restart.** The first full run put 38 label
reviews — about $17 of model calls — into `/tmp/report_ingest_520`, and a
restart wiped them. That is the failure this section exists to prevent, and it
is the same one the app already solved for conversations: it has mirrored every
conversation to SharePoint after every turn since 5.10.0.

So pass **`sharepoint=fh_sp_client`**, or **`durable_dir="/Volumes/…"`**, or
both. Then:

- the run's folder is **`<sharepoint_folder>/<the out_dir's own name>`** —
  `out_dir="/tmp/report_ingest_522"` mirrors to
  `GeotechStaffEngineer/report_ingest/report_ingest_522`, with the same layout
  inside (`runs/`, `triage/`, `vision/`, `vote/`, `RESULTS.md`, `results.json`);
- **every run file is copied as soon as it is written**, by every stage, so a
  restart mid-run costs the report in flight and nothing else;
- **at the START, anything the mirror holds that `out_dir` does not is copied
  back**, so a wiped `/tmp` resumes from what was already paid for rather than
  paying again. The first line the run prints says where the mirror points and
  how many files came back;
- the copy is **incremental** (a local `mirror_manifest.json` of each file's
  size and modification time), so the mirror after report 38 sends one file,
  not 38;
- **a mirror failure never stops the run.** It prints one warning line — once
  per distinct error, not once per report — and the run carries on with its
  output in `out_dir` alone.

| parameter | what to pass |
|---|---|
| `sharepoint` | the live `fh_sp_client`, its `.file_manager`, or the app's `SharePointStore`. All three are accepted, because remembering which one this argument wants is how a run ends up mirroring nowhere. |
| `sharepoint_folder` | the folder the run folders sit under. Default `GeotechStaffEngineer/report_ingest`, which is where the 2026-09-20 sheet-mode run was copied by hand. |
| `durable_dir` | your workspace folder under `geotech_app/`, which persists; never `/tmp` and never DBFS (the owner's rule, 2026-09-20). The run's name is appended and the SharePoint prefix is not, so a run lands at `<durable_dir>/report_ingest_522/`. |

`out_dir` still refuses `/Workspace`. `durable_dir` does not: whether a path
is durable on this cluster is your finding, not this package's, and the one
place a guess would be expensive is the one it is guessing about.

**The mirror sends the WHOLE `out_dir`**, which is the point — and that means
`runs/` and `triage/`, whose reasons and rationales can name a firm, a project
or a person, and the derived `labels_map.json`, which holds the label
spreadsheet's own sheet names. The remote folder is exactly as private as
`out_dir` is, so put it where the reports themselves already live. Only
`RESULTS.md` is written to be carried anywhere.

The six stages are described one at a time below, each with the cell that
runs it alone.


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
    truth_dir    = "/Volumes/<your volume>/report_ingest/truth",   # its logs/ holds <ID>_p<page>.json + OPEN.txt
    out_dir      = "/tmp/report_ingest_logs",
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
    truth_dir     = "/Volumes/<your volume>/report_ingest/truth",  # its lab/ holds <kind>__<ID>_p<page>.json + OPEN.txt
    out_dir       = "/tmp/report_ingest_lab",
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

### Scoring the narrative reader as well (`stages=("labels", "logs", "lab", "narrative")`)

The `narrative` stage runs `read_narrative` over every report that has a hand
answer in `narrative_truth_dir` (`<ID>.json`, the owner's two schemas answered
by hand with **null** where the report does not say) and scores it field by
field. It reads the truth files that are PRESENT and skips every report without
one, because the hand answers arrive a few reports at a time.

A truth file carries two things beyond the answers. **`_alternates`** is the
hand's fairness valve: other answers it will accept for this report, keyed by
field and always a list, because a report can say a thing in more than one
defensible way and the hand is one reading of it rather than the only one. A
`null` among them means "not stated is acceptable too"; a list among them is a
whole alternative list answer. **`_skip`** names the questions this report does
not settle, and they are excluded from every count rather than scored as
misses. Neither ever reaches the reader.

```python
results = score_on_cluster(
    reports_dir         = "/Volumes/<your volume>/reports",
    manifest            = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    di_dir              = "/Volumes/<your volume>/report_di",
    truth_dir           = "/Volumes/<your volume>/report_ingest/truth",  # its narrative/ holds <ID>.json
    out_dir             = "/tmp/report_ingest_narrative",
    prompter            = fh_prompter,
    model               = "funhouse-gpt-high",
    stages              = ("labels", "logs", "lab", "narrative"),
    narrative_budget    = 8,        # model calls per report; most cost one
    max_reports         = 2,        # drop this line after the first run
)
```

**Three numbers, and the third is there to be distrusted.** Most reports answer
most of the general list and only part of the hazards list, so a reader that
says nothing agrees with the hand on a great many fields.

| | |
|---|---|
| `recall` | of the questions the report DOES answer, how many came back right. The one that says whether the reader reads. |
| `precision` | of the answers the reader GAVE, how many were right. The one that catches a reader inventing plausible facts. |
| `agreement` | every field, including the ones both sides left null. Printed because it is what a naive scorer would print, and it flatters. |

How each kind of field is judged: **enumerations and counts exact**; **strings**
by a normalised token match (rapidfuzz partial ratio ≥ 85, or a shared proper
noun — "Soil & Rock Consulting Engineers" and "Soil and Rock Consulting
Engineers, Inc." are one firm, and `siteClass` and `reportDate` also pass on
their normalised forms); **lists** by set overlap, with the items' own precision
and recall pooled across every list field and the field counting as right at a
Jaccard of 0.6 — and two kinds of list, since `boringDictionary` and
`testPitDictionary` hold identifiers and are matched exactly (B-1 and B-12 are
two holes) while the prose lists are matched item by item by the string rule;
**the four verdict questions on their verdict alone**, because two readers who
both find the geophysical survey will not word the reason the same way; and
**the four prose summaries on presence and word limit ONLY**
(≤ 100 / 100 / 200 / 100 words), because whether a summary is a good summary is
a person's call and a scorer that pretended otherwise would be scoring its own
opinion. Null against anything is a miss either way; null against null is
agreement and earns no recall.

`RESULTS.md` carries the three numbers for the open, blind and whole sets, the
recall broken out per kind, a **per-question** table (which questions are hard
is what a prompt change is aimed at), a per-report line and the cost.
`narrative/<ID>.json` holds the per-report detail and makes the stage
restartable.

Locally, `module_work/report_ingest_harness/measure_wp4_narrative.py` does the
same scoring against the development engine (`--fields` for the per-question
table, `--detail` for every miss on the open reports).

### Labelling a page by looking at it (`stages=("vision_labels",)`)

An experiment, and it is the one stage here that is not a reader. The rules
label a page from what that page prints about itself; the review corrects them
with the whole report in view. **Both read text.** This stage asks the third
question: hand the page to a cheap model as a PICTURE, with the eighteen-label
vocabulary and nothing else, and see what it says. It runs on the same reports
as the `labels` stage, scores against the same hand labels with the same
scorer, and writes one table with the three answers side by side.

```python
results = score_on_cluster(
    reports_dir            = "/Volumes/<your volume>/reports",
    manifest               = "/Volumes/<your volume>/report_ingest/MANIFEST.md",
    labels_xlsx            = "/Volumes/<your volume>/report_ingest/trial_pages_working_r2.xlsx",
    oos_labels             = "/Volumes/<your volume>/report_ingest/oos_labels.json",
    di_dir                 = "/Volumes/<your volume>/report_di",
    out_dir                = "/tmp/report_ingest_document",   # one out_dir per mode
    prompter               = fh_prompter,
    stages                 = ("vision_labels",),   # or beside ("labels", ...)
    vision_model           = "funhouse-gpt-low",   # GPT-4.1; the cheapest tier on purpose
    vision_mode            = "document",           # or "page" / "sheet"
    vision_dpi             = 100,                  # the default; see below
    vision_detail          = None,                 # or "low": ~85 tokens an image
    vision_window          = 36,                   # full-size pages in one call
    vision_images_per_call = 50,                   # the endpoint's own cap; measured
    vision_fallback        = True,                 # one page-mode call per unresolved page
    vision_outline_context = False,                # True = it also sees the contents list
    max_reports            = 2,                    # drop this line after the first run
)
```

Give each mode its **own `out_dir`**. A run file is keyed by report, not by
mode, so a second mode written into the first one's folder would be skipped as
already done rather than run.

**Why the cheapest tier.** The question is not whether a good model can label
a page — it is whether the cheapest one can, looking, do what the rules and the
review do by reading. Scoring it on `funhouse-gpt-high` would answer a
different question and cost more to answer it.

**The three modes are the trade being priced.**

| mode | what the model sees | calls / 100 pp | input tokens / 100 pp | ~$ / 100 pp | use it when |
|---|---|---|---|---|---|
| `page` | one page, rendered whole, and nothing else | 100 | ~250,000 | $0.11 | you want the page's own fine print read and no neighbour to lean on |
| `sheet` | six thumbnails a call, each with its page number | 17 | ~43,000 | $0.02 | you want the cheapest sweep that still scores like page mode |
| `document` | a window of 36 full-size pages, stamped, with contact sheets of the WHOLE report beside them | 3 | ~170,000 (A4), or ~16,000 at `vision_detail="low"` | $0.07, or $0.01 low | the page cannot be read without knowing which appendix it is in |

**What the cluster has actually measured** (2026-09-20, `gpt-4.1-mini`, on
5.21.1 — the first run in which no vision call saw a page's rule-derived kind):

| mode | reports | pages | strict accuracy | calls | ~$ a report |
|---|---|---|---|---|---|
| `page` (2026-09-18) | 2 | 307 | 0.928 | 307 | $0.25 |
| `sheet` | 2 | 307 | **0.932** | 26 | **$0.04** |
| `document` | 1 (text pages) | 151 | 0.927 | 5 windows | — |

**Sheet mode is the one to run.** With number-only contact sheets it scores
what page mode scores — 0.932 against 0.928, and 0.947 / 0.917 on the two
reports read separately — for a sixth of the calls and a quarter of the money.
The 0.557 that sheet mode scored on 2026-09-18 measured planlens' `<index>
<kind>` captions, not the model; those numbers are void and the fix shipped in
5.21.1.

**Document mode's own numbers, and the two things that bit it.** On a 151-page
report of text pages it scored 0.927 in five windows — level with the other
two — but it cost 254,724 input tokens, about 51,000 a call. The pages are A4,
and an A4 page at 100 dpi scales to 768 x 1086, which is **six** 512 px tiles
rather than a letter page's four. That is the arithmetic, not a defect. The
two defects were real, and 5.21.2 fixes both:

- **Three pages came back unresolved** because the model skipped them and the
  window that overlaps each one skipped them too. The overlap is a second
  chance, not a guarantee.
- **Every window of a 156-page SCANNED report failed** with `APIStatusError:
  The page was not displayed because the request entity is too large`. That is
  the gateway's limit on the REQUEST BODY, which the 50-image probe — done
  with tiny images — never came near.

**Every page picture now travels as JPEG** at quality 80
(`vision_labels.VISION_JPEG_QUALITY`), at the render's own pixel size. The
provider scales and tiles the pixels, so **the token count does not change**;
only the bytes on the wire do. On a scan-like page that is a factor of about
three (780 KB of PNG against 250 KB of JPEG). On a crisp vector text page PNG
is already good and JPEG runs between 0.84 and 1.20 of it — the rule is
unconditional anyway, because the pages that refuse a request are the scanned
ones and a per-page choice would make one report's windows a different size
from another's for reasons no run file records.

**And a window that is still refused splits itself.** When a call fails with
`too large` / `request entity` / `413`, the pass halves that window, puts both
halves back at the front of the queue, and **re-cuts every window still
queued** to the new size, so one refusal is paid for once rather than at every
window after it. A refused request bills nothing, so a split costs time and no
money. A window already at or under `DOCUMENT_MIN_SPLIT` (4 pages) that is
still refused raises instead: at four pages the body is small and the refusal
means something else. The `split` column in the per-report table says how often
it happened, and a dash means never.

**`vision_fallback=True` gives every page still unresolved ONE page-mode
call.** It runs after the sheet or document pass, spends the same budget as
everything else, and never touches a page something already answered for. A
page goes unresolved because the REPLY left it out, not because the page could
not be read, and asking about that one page alone is the mode that cannot skip
it by construction. `vision_fallback=False` leaves those pages unresolved,
which is what every run before 5.21.2 did.

Only the `page` and `sheet` rows of the cost table above are measured end to
end; the `document` row is that same measured per-call overhead put through the
provider's own tile arithmetic — 85 tokens plus 170 a 512 px tile, six tiles
for an A4 page and six for a 48-up contact sheet — and the dollars are those
tokens at the owner's `gpt-4.1-mini-2025-04-14` rate. Treat them as the
arithmetic they are until a run replaces them.

**`document` exists because of what `page` mode got wrong.** In page mode the
cheap model matched rules-plus-review overall and beat both on narrative and
figure recall, but lost on `plan` and `lab_test` — the two labels a human
settles by knowing which appendix the page sits in, which is exactly what a
page seen alone cannot say. So document mode hands the model the report's own
shape along with the pages: a strip of contact sheets covering the whole
report for orientation, then the window of full-size pages it is actually
answering for, then, from the second window on, the labels already decided as
runs (`61-118: boring_log`), which is what an appendix looks like written down.

**The 50-image cap is the provider's, and it is measured — and it is not the
only ceiling.** On the owner's cluster on 2026-09-18 a request with 50 images
went through and a 51st came back `Too many images in request: 51, maximum
allowed: 50`. The gateway ALSO limits the request body, which a report of
scanned pages reaches at far fewer than 50 images; that is what the JPEG
encoding and the self-splitting window above are for. That cap, not the
context window, is why document mode slides a window rather than sending the
whole report in one call — GPT-4.1 on `funhouse-gpt-low` has a 1M-token
window, which would hold a 400-page report at ~770 tokens a page with room to
spare, and the image count would still refuse it. So one call carries the
strip and the window and their sum never exceeds `vision_images_per_call`:

| report | contact sheets | strip sent | window | images a call | calls |
|---|---|---|---|---|---|
| 151 pp | 4 | 4 | 36 | 40 | 5 |
| 426 pp | 9 | 9 | 36 | 45 | 13 |
| 729 pp | 16 | **12** | 36 | 48 | 22 |

The strip is capped first — at twelve sheets, at what the document has, and at
whatever leaves the window twelve pages — and the window takes what is left.
So the 729-page report is the one that gives up seeing all of itself, and it
gives up the far end: the twelve sheets NEAREST the window are the ones sent,
because the divider that opens this appendix is a few pages back, not 500.
Consecutive windows overlap by three pages, so a page the model skipped in one
window can still be answered by the next; where both answer, the LATER answer
wins, because it was made with more of the report already decided.

**Every full-size page is stamped `p. N` in a box at its top-left corner**, in
the 0-based index this pass counts in. A geotechnical report restarts its
printed numbering in every appendix — three pages numbered "1" is normal — so a
model told to use the number printed on the page would answer for the wrong
page. The stamp is drawn ON the render rather than added as a margin, so the
page size, and therefore the token count, is the render's.

**`vision_detail="low"`** tells the endpoint to look at one 512 px tile of
every image and charge about 85 tokens for it, whatever the image is. On the
measured document run that is the difference between ~51,000 input tokens a
call and about 3,400. What it costs in accuracy has not been measured; it is
the next thing to run.

**The dpi is 100 because that is what the model keeps.** A 4.1-class vision
stack scales an image to its own working size before it looks at it or charges
for it — the short side lands around 768 px — and then prices what is left in
512 px tiles. The arithmetic is one-sided:

| render | what the model sees | tiles | tokens |
|---|---|---|---|
| letter at 72 dpi | 612 x 792 (not scaled: under the ceiling) | 4 | ~765 |
| letter at 100 dpi | 768 x 994 | 4 | ~765 |
| letter at 200 dpi | 768 x 994 | 4 | ~765 |

So 72 dpi is not the cheap option, it is the same price with a quarter of the
short side thrown away; 200 dpi is the same price again for four times the
bytes on the wire and not one pixel the model keeps. Everything from about
90 dpi up lands on that identical 768 px short side, and 100 leaves room for a
page that is not letter-sized (A4 at 100 dpi is 827 x 1169, which scales to
768 x 1086 and costs six tiles because it is taller, not because of the dpi).
It is a parameter because the ceiling is the provider's and it moves.

**`vision_outline_context=True`** prepends what the document prints about
itself — the contents list, the lists of figures, tables and appendices, the
dividers — to every call, so a run can put pure vision beside vision that knows
which appendix it is standing in. It rides on EVERY call — a hundred of them
in page mode — so it is cut at 6,000 characters. In document mode it is
partly redundant with the contact-sheet strip, which shows the same dividers
as pictures; running it both ways is the way to find out which the model
actually uses.

**How to read the table.** `RESULTS.md` gains a WP5 section whose rows are the
labels and whose columns come in pairs — precision and recall for `rules`
(planlens' per-page rules), for `+review` (those rules with the label review's
accepted changes applied) and for `vision` — over one set of hand labels and
one scorer. A column appears only where its run exists: a report with no
`runs/<ID>.json` beside it prints two columns rather than three, and the header
says how many reports the `+review` column covers when it is not all of them.
**A page the vision pass left unresolved is scored as `other`** — a non-answer
is scored, not excused — and the per-report line carries how many there were.
The blind set prints a summary and nothing else, because a blind figure read
report by report stops being blind.

What Python refuses rather than passes on: a label outside the vocabulary
becomes `other` with a QA note, a confidence outside 0 to 1 is clipped, a page
the reply left off its own sheet or window is `unresolved` and never guessed
from its neighbours, a page the reply invented is dropped with a note, and the
budget caps model calls so every page past it is `unresolved` too. In document
mode a page left out of one window is still `unresolved` only if the window
that overlaps it leaves it out too — the overlap is a second chance, not a
licence to guess — and with `vision_fallback` on, only if one page-mode call
about that page alone cannot settle it either.

`vision/<ID>.json` holds the per-report detail and makes the stage restartable.
It stays on the cluster with `runs/` and `triage/`: a vision REASON says what
the model saw on a page and can therefore quote a title block.

Locally, `module_work/report_ingest_harness/measure_wp5_vision.py` does the
same scoring against the development engine — `--mode document`, `--detail
low`, `--window`, `--overlap`, `--images-per-call` and `--no-fallback` are all
there — and `--reuse` re-scores the saved runs with no model, no key and no
network at all.

### Making the disagreement itself the answer (`stages=("vote",)`)

The cheapest stage here, because it calls **no model at all**. Everything it
reads is already on disk: the rules' labels and their confidences, the vision
labels and theirs, and the review's labels from the label runs where a report
has one. So it costs nothing, and it can be re-run after every change to the
arithmetic.

Run files written by 5.22.0 and later carry `rules_confidence` — planlens' own
per-page confidence — beside the labels, and those are used as they stand. A
run file written earlier has the labels and no confidence, so planlens is
asked again for that report; the rules are deterministic, so the answer is the
one that produced the saved labels. Where the PDF is not to hand the saved
labels stand with no confidence, and the `confidence` policy simply has
nothing to weigh on that report.

```python
results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",
    manifest     = "/Volumes/<your volume>/report_ingest/MANIFEST.md",
    labels_xlsx  = "/Volumes/<your volume>/report_ingest/trial_pages_working_r2.xlsx",
    oos_labels   = "/Volumes/<your volume>/report_ingest/oos_labels.json",
    di_dir       = "/Volumes/<your volume>/report_di",
    out_dir      = "/tmp/report_ingest_522",
    sharepoint   = fh_sp_client,
    prompter     = fh_prompter,             # required, and never called by this stage
    stages       = ("vote",),
    sets         = ("insample", "oos_open", "oos_blind"),
    vision_dirs  = ["/tmp/report_ingest_521_sheet/vision"],   # default: this run's own vision/
    review_dir   = "/tmp/report_ingest_520/runs",             # default: this run's own runs/
)
```

**Why.** The corpus run of 2026-09-20 did not say one voter is better. It said
they are **complementary by label class**: the rules own the structural labels
— `appended_report` (494 in-sample pages, which vision never once emitted),
`other` (216, the same), `calculation` (1,009, of which vision missed 41 %) —
and vision owns the visual ones, beating the rules outright on `plan`,
`profile`, `photos`, `cover`, `toc` and `figure` recall. Overall: rules 0.908
in sample and 0.767 honest blind, rules plus the review 0.916 / 0.850, vision
in sheet mode at $0.05 a report 0.655 / 0.867. So the question is not which to
keep. It is what a page they split on is worth looking at.

`RESULTS.md` gains a `# Vote` section in five parts:

1. **Agreement, and what it is worth.** How often the two voters agree, over
   every page they both covered — the number a production run can compute for
   itself, with no hand labels — and then, over the hand-labelled pages, how
   right they are when they agree against when they do not. That is the
   confidence claim. A page the vision pass left unresolved counts as `other`
   and therefore as a disagreement, which is the honest reading: a page nothing
   could answer for is exactly the page that wants a second look.
2. **Per-label trust**, learned on the **in-sample reports only**. For each
   class the RULES put a page in, which voter was right more often on the pages
   they split on. Keyed by the rules' label because that is what a production
   run has before it knows the answer. A tie goes to vision, and so does a
   class the in-sample pages never split on, so `believe rules` always means
   the rules were strictly better on pages somebody has checked. The
   out-of-sample sets are SCORED with this table and never learned on; the
   blind set never is under any circumstance, and the file says so.
3. **Combined labels under three policies**, each scored by the same scorer
   against the same hand labels, beside `rules`, `vision` and `+review`:
   - **`trust`** — on a disagreement, believe whoever the in-sample table
     favours for the rules' label class; vision where it says nothing.
   - **`structural`** — vision everywhere except `appended_report`, `other`,
     `calculation` and `lab_test`, which the rules own because they are decided
     by where a page SITS in the document rather than by what it looks like.
     It learns nothing, which makes it the policy to beat.
   - **`confidence`** — whoever said it more confidently (planlens' own rule
     confidence against the vision pass's), ties to the rules.

   Read the out-of-sample rows: `trust` is scored in sample with a table
   learned on those very pages, so its in-sample figure is a ceiling rather
   than a result. `structural` and `confidence` are honest everywhere.
4. **The disagreement set.** How many pages per set would go to a targeted
   review, what fraction that is, and the accuracy that review would need **on
   those pages alone** for the whole set to clear the 0.98 gate. `>1.000` means
   the gate is out of reach even with a perfect review, because the pages the
   two voters agree on already carry more error than the gate allows — which is
   a result about the agreed pages, not about the review.
5. **Per report**, for the sets that are not blind.

`vote/<ID>.json` holds every page the voters split on: the page number, what
each voter said, its confidence, the hand label where there is one, and the
label each policy chose. Page numbers and labels only, so unlike `runs/`,
`triage/` and `vision/` these files carry nothing that could name anyone — but
they stay on the cluster with the rest all the same.

**Give it the vision runs you mean.** `vision_dirs` takes one or more folders
and the first that holds a report wins, so the vote can be run over an older
mode's output without re-running it; the default is this run's own
`out_dir/vision`. The stage refuses to start if it finds no vision run at all,
and says which folders it looked in. `review_dir` is the same idea for the
label runs, and the review is simply absent as a voter where it is not there.

Locally, `module_work/report_ingest_harness/measure_wp6_vote.py` is the twin:
it reads the WP5 and WP1b runs the development engine left behind and prints
the same tables, with `--vision-dir` and `--append`.

Bring back **`RESULTS.md`** from whatever `out_dir` was. That is the whole report,
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
- **It is mirrored somewhere durable**, if you pass `sharepoint=` or
  `durable_dir=`. Every run file is copied as it is written and a wiped
  `out_dir` is refilled from the mirror at the start. See "Where the output
  goes" above; a mirror failure warns and never stops the run.
- **Start small.** `max_reports=2` on the first run proves the path end to end
  for the price of two reports.
- **A report the folder does not have is named and skipped**, once, before
  anything runs, rather than failing one report at a time deep in the set.
- **No credential is read.** Authentication is whatever `fh_prompter` was built
  with. Nothing here touches an environment variable or a secret scope.
- **Dollars, at your own rates.** A call is priced by the DEPLOYMENT that
  answered it, never by the tier that was asked for: a tier is an alias and the
  model behind it changes without notice, so a number priced by the tier would
  be priced by a name that means something different next month. The four
  deployments on the Funhouse budget page as of 2026-09-18 are in
  `engine.PROMPTER_PRICES` and every `## Cost` block prints dollars beside its
  tokens. A call served by a deployment with no rate on file adds its tokens
  and no dollars, so read a total as a FLOOR and check Funhouse's own budget
  endpoint for the same window. A run where nothing was priced still reports
  tokens alone, exactly as it did before.

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
script of turns in the engine's place, so the whole of every pass — both label
passes, all three readers and the end-to-end graph — runs for real — the prompt each builds, the tool loop, the budget
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

The narrative reader, the reconciler, the writers and the whole graph are
tested the same way. The graph's test runs the synthetic report end to end on a
scripted engine — every item to its own reader, every output written, both
DIGGS gates green — and then proves that a second run over the same folder
makes ZERO model calls, that `needs_person` stops after triage, that
`appendix_only` never reaches the narrative reader, that a relabelled page
changes which items are read, and that one reader raising does not take the
report down with it. The writers' tests read the library row back out of
SQLite; the reconciler's pin the rule that matters most, which is that both
values survive a conflict, in the record, with their pages.

The harness suite adds the scorecard's own arithmetic, the disputed-label rule,
the prompt fingerprint, the deterministic fifteen-log DIGGS gate, the
thirty-one-sheet lab gate, the floor under the lab scorecard (a flawless
reading of all thirty-one sheets must score 100 %, or the scorer is what is
wrong), and the WP4 table's own arithmetic — including the one that matters:
a reader which answers nothing prints a recall of zero rather than an accuracy
of ninety. Its corpus-dependent tests skip cleanly on a machine without the
private data.

The mirror is tested against a fake file manager holding its remote tree in a
dict — uploads land in it, `ls` and `download_file` read it back — so the whole
of both directions runs for real: what is skipped because its stamp has not
changed, what is retried after a failure, what a restore brings back and what
it leaves alone, and the manifest that keeps one backend from being credited
with the other's uploads. The vote's arithmetic is tested on synthetic run
files small enough to check on paper: the agreement fraction, the trust table
learned in sample and applied out of it, the three policies' chosen labels page
by page, and the accuracy a review of the disagreements would need.

The app-side wiring has its own offline test,
`funhouse_agent/deep/tests/test_report_ingest_offline.py`: that the tool and
the sub-agent appear only when asked for AND the installed planlens can serve
them, that the spec is built like every other sub-agent spec, and that a run
through the tool comes back as the compact result rather than the record.

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

It shipped in the wheel as of **5.19.0** (2026-09-17) as a library, and
**5.20.0** adds the rest of it: the record, the three readers, the reconciler,
the writers, the graph and the sub-agent. Since WP4 the app CAN call it —
`build_deep_agent(enable_report_ingest=True)` adds one primary tool and one
sub-agent — but the flag is **OFF by default**, so nothing on the chat surface
changes until a release turns it on. It stays off until the owner's four-stage
cluster run has measured the readers on the tier that will do the work.

- **`report_ingest*` is in `[tool.setuptools.packages.find]`.** Until that line
  landed, a wheel built from this repo did **not** contain this package, and
  the failure would have appeared on the cluster as an import error rather
  than at build time. Check it after any edit to that list:
  `python -m build --wheel` and look for `report_ingest/` in the wheel.
- **`report_ingest` is in pytest's `testpaths`**, so this suite runs in the
  release gate. The WP0/WP1 harness under `module_work/` is dev-only and is
  deliberately not.
- **`planlens>=0.6`** (5.20.0) — the page roles, the printed outline and the
  per-page ledger are 0.5.0's; `log_grid`, which the log reader is built on,
  and the text extraction that returns an overprinted line ONCE are 0.6.0's.
  That second one is not a nicety: a depth scale drawn twice reads "5, 5, 10,
  10", which holds no strictly rising run of three, so the ruler was refused
  and the sheet came back with no depths at all. planlens 0.6.0 declares
  exactly the dependencies 0.4.0 did, so the pin adds no new package to the
  cluster.
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
