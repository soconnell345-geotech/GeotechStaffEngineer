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
| The floors | `floor.py`, `log_floor.py`, `lab_floor.py` | the grid's and the tables' own record before any call, and the merge: add, correct with evidence, never drop |
| 3. Narrative reader | `narrative_reader.py` | one call for the owner's two schemas, cited |
| Reports bound inside reports | `bound.py` | where a bound document begins and ends, and the one call that says what it IS |
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

**Every value-bearing field carries `Provenance`** — the page, the box, how it
was read, and a confidence from 0 to 1. Since 5.23.0 the method names the
VOTER: `grid` and `tables` are the deterministic first voter (the log grid's
geometry, the page's detected tables), `model` and `model_from_picture` the
second (a model reading the rows or the text, or looking at the picture),
`reconciled` a value both gave within the scorer's tolerance; `text`, `di`,
`ocr`, `vision` and `derived` are what they were. Where the two voters split,
the slot holds one voter's value and the other's stands beside it in
`prov.alternatives` (an `Alternative`: field, value, unit, method,
confidence), and a `QAEntry` of kind `disagreement` sends a reviewer to look.
A reviewer can go to the page; a QA pass can ask which values rest on the
model alone, and which the two voters disagreed on.

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

**The floor.** The grid is the first voter and what it placed is the floor. Before
any call, `log_floor.seed_from_grid` turns the rows into an `Investigation` —
samples with their depths, blow records and N values, recoveries and index
values, the layers the grid bound to depths with any USCS symbol the description
prints, the header fields, a water level off the groundwater field — each with
the grid's own confidence, and the model is shown it as THE STARTING RECORD. Its
answer is merged back onto the floor (`log_floor.merge_investigations`) under one
rule. It may **add** what the floor lacks: a sampler type from the symbol, a USCS
symbol the log prints, a water level and its timing, a refusal, a layer base. It
may **correct** a floor value only with evidence — the box of the row it read and
a note saying what the page prints there, or a note on a value read off the
picture — in which case the model's value takes the slot and the floor's stands
beside it; a contradiction with no box and no note leaves the floor's value in
the slot with the model's beside it. It may **never drop** one: a value the
model leaves out is kept from the floor with a QA note. A difference inside the
scorer's own tolerance (0.15 m on a depth, 0.30 m on a layer top, exact on an N)
is not a disagreement — the model's printed number takes the slot as
`reconciled`. Every other split is a `disagreement` in the QA section, both
values, both confidences, and which one the record carries. The first full
cluster run is why: the reader re-emitted the record from its own answer and
seven of ten blind logs lost values the grid already had (recovery 15/15 →
3/15, index 7/16 → 0/16).

**The budget** is one call per log, a second only when the reader itself says it
has pages left, ONE more on the reader's own unsettled list when the budget
allows — the rows in question again, magnified through the page's ruler to the
depth the item names — and six at the most.

## The log-template recogniser (`log_templates.py`)

A firm prints its logs on one form, and the form says so: the gINT report name
in the footer, the firm's own field labels in the title block, its own column
headings. `recognise(doc, page, grid=None)` compares a page's located text
against a list of FINGERPRINTS with rapidfuzz and returns the best one above a
threshold — the family, a confidence 0 to 1, the margin over the best
fingerprint of a *different* family, and the list of what matched. No model
call, no render; the cost is milliseconds.

It is worth having twice over.

- **As a voter.** `ledger_note(match)` is the line
  `template <family> (0.92)`, which goes beside the rules' label and the
  vision label on the page's row — strong evidence that the page is a boring
  log of that firm, and evidence no picture of the page and no page-role rule
  carries. `annotate_ledger` puts it on planlens' page ledger for the label
  review.
- **As a key to the grid.** `log_grid` names a column by classifying its
  printed header, and a form that prints `DATA` over a stacked sample id,
  sampler code, blow record and recovery gives that vocabulary nothing to
  classify. A fingerprint's `column_map` says what that column carries;
  `seed_from_grid(grid, pages, template=match)` puts those names on it and the
  floor reads the values. The method stays `grid` — the value was still placed
  by geometry — and the note names the template.

**The fingerprints are DATA and they are not in this repository.** This repo is
public and a fingerprint names a firm. The file travels with the private truth
folder as `<truth_dir>/../templates.json`, or is passed as `templates_path=`;
`score_on_cluster(..., templates_path=…)` puts it in force for the whole log
stage, and with no such file anywhere every entry point here is a no-op and
nothing downstream changes. `templates.json.EXAMPLE` ships beside this module
with three INVENTED firms and documents the shape:

```json
{
  "name": "Northgate Soils gINT 2014",
  "family": "Northgate Soils",
  "years": "2012-2018",
  "title_phrases": ["RECORD OF TEST BORING", "Northgate Representative:"],
  "column_headers": ["DEPTH (ft)", "SAMPLING DATA", "LABORATORY"],
  "footer_phrases": ["NORTHGATE STANDARD LOG 2014_03_11.GDT"],
  "layout": {"columns": 8, "ruler_side": "left", "units": "ft"},
  "column_map": {
    "sample_id": "SAMPLING DATA", "sample_type": "SAMPLING DATA",
    "blows": "SAMPLING DATA", "recovery": "SAMPLING DATA",
    "index": "LABORATORY"
  }
}
```

Several fingerprints share one `family`, which is how a form that drifted over
the years is described: one fingerprint per era, one family for all of them,
and the family is what the voter is told. The footer group is worth half the
score, the title block three tenths and the column headings two tenths, over
the groups a fingerprint actually declares — so give every fingerprint a footer
phrase or at least three title phrases, or it will claim pages it should not.

Measured locally, 2026-09-20 (no model, no network): on the fifteen
hand-truthed logs the recogniser is **100 % recall and 100 % precision on both
families** and claims none of the five logs printed on forms no fingerprint
describes; on thirty pages the hand says are not logs it claims **none**. The
column map is worth **+13 values** on the ten logs it claimed (the floor
scoring 220/311 without it and 233/311 with it), all of it in `recovery` and
`index` — the two columns a stacked form gives the general vocabulary no way to
name.

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

**The floor.** The page's detected tables and its title are the first voter.
Before any call, `lab_floor.floor_from_tables` reads them into typed `LabTest`
records: the kind the title names (seventeen title patterns, in three
languages), the boring, sample and depth printed on the sheet, every labelled
value in a property/value table (`Liquid limit | 31`), every row of a grading
series (a SIEVE or SIZE column beside a PERCENT FINER column) and every row of a
summary table (a BORING or DEPTH column and more than one row) — each with the
table's own confidence — and the model is shown them as THE STARTING RECORD. Its
answer is merged back (`lab_floor.merge_lab_tests`) under the same rule as the
log's: a kind, a link, a value or a specimen the model **adds** is accepted; a
value it **corrects** with a box and a note replaces the table's, with the
table's kept beside it; a contradiction without evidence leaves the table's
value in the slot with the model's beside it; a value it **omits** is kept, and
a whole test it omits stands in the record as its own test, from the tables.
An index value agrees to 0.01, a grading point to one percentage point. Every
split is a `disagreement` in QA. Six sheets of the first cluster run came back
below the tables alone (a gradation 28/28 → 11/30, a chemical 18/18 → 14/22);
none can now.

## The calculation reader (`calc_reader.py`)

`read_calculation(doc, item_pages, engine, budget=2)` turns one calculation
printout into one typed `Calculation` — what it works out, what printed it,
what it was given and what it concluded.

**This was the largest unread class of page in the corpus.** A quarter of the
4,300 hand-labelled pages are calculations — 1,009 of them, over eight of the
fourteen labelled reports — and until this train every one was a QA entry
saying the pages existed and had been skipped. They are also the pages a
reviewer most wants: a boring log says what the ground *is*, a laboratory
sheet says what a specimen *did*, and a calculation says what the engineer
**assumed**, **worked out** and **concluded**.

**A calculation page is not a lab sheet.** It is one of three things and they
look nothing alike: a **program printout** (a banner, then fixed-width columns
of echoed input and computed output, often for several load cases and often a
dozen pages); a **spreadsheet printed to PDF** (a title block, a project
block, a table of inputs with the unit in the column *heading* and the number
in the cell beside it, and a total at the bottom); or a **hand calculation**
on a printed form. So there is no title vocabulary to classify from and no
single table to read. What all three have is **labelled numbers**, and that
pair — the printed label and the value — is the whole of what this record
holds.

**The kind is what it WORKS OUT**, from `model.CALC_KINDS` (thirteen kinds,
each with a one-line definition written in no program's words): a settlement
worked on a spreadsheet and one worked by a commercial program are both
`settlement`. A printout that is none of them is `other`, which is an answer.

**An input is what it was given; a result is what it worked out.** Where the
page does not make the line clear the value is an **input** — claiming
something was concluded when it was assumed is the worse error. The label
keeps the page's own words, parentheses and all, because a reviewer goes
looking for what is on the page.

**The floor.** Before any call, `floor_from_pages` reads every (label, value,
unit) a pattern can find: a `label | value` table row, a one-row table under
its own headings, the **last filled row** of a ruled data table (which is
where a cumulative settlement and a governing case live), a label span and a
value span on the same printed *line* (which is how a spreadsheet sets them,
and why the floor groups spans into bands first), a single span reading
`Pile-head deflection = 0.025 meters` or `Design ESALs ...... 137,774`, and
planlens' own `quantities` pass for a value stated in prose, labelled by the
words in front of it. The program name comes off a running banner only where
a small **generic** list of commercial program-name patterns matches, and is
`None` otherwise — which, on a spreadsheet, is the right answer. That is the
STARTING RECORD the model is shown, and its answer is merged back under the
same rule as the other two readers: **add, correct only with evidence, never
drop.** A floor value the reader did not return is kept as an *input* with a
note (a pattern cannot tell an input from a result, and the smaller claim is
the honest one); every split is a `disagreement` in QA.

**Three Python gates.** A **result whose number is on none of the pages** to
the precision it was reported at keeps its place in the record — it may be
right and the text layer wrong — but drops to **confidence 0.3** and is
listed. A **kind outside the vocabulary** becomes `other` with a note (a
handful of trade spellings are understood first: `bearing_capacity`,
`drilled_shaft`, `p_y`, `earth_pressure`, `site_class`). A **unit the record
cannot convert** keeps the value as printed — that is the record's own rule —
and says on the unsettled list that nothing downstream can convert it.

**The budget** is two model calls and most printouts cost **one**. A run
longer than `MAX_CALC_PAGES` (12) is shown a window of that many pages with
the **first and the last always in it** — the first carries the banner and
the inputs, the last carries the answer — and the second call is spent only
on the rest of the run or on the reader's own unsettled list. Its one tool is
**`zoom_plot`**, for a result printed *on* a drawing: a slope section with its
factor of safety written beside the critical surface, which is what a
stability program prints and what no text pattern will ever find.

**DIGGS ignores them, deliberately.** DIGGS 2.6 is an interchange format for
what was *observed* in the ground and has no concept of a calculation: no
element for a method, no home for a chosen thickness. So `calculations`
reaches a reader through the record, the summary page's **Calculations**
section and the library page, all three carrying the page each value was
printed on. A writer that squeezed a design calculation into an observation
element would produce a file that validates and lies.

**The floor alone, measured 2026-09-21** over ten hand-truthed runs (44 pages,
six reports, seven kinds, no model at all): program 80 % (8/10), inputs 73 %
(64/88), results 48 % (36/75). **Every one of those runs is in sample** — they
were read while this reader's prompt was written — so a blind calculation
truth set is owed (`FUTURE_IDEAS.md`).

## The pit, the cone sounding and the dynamic probe (`sounding_reader.py`)

Three kinds `InvestigationKind` could always NAME and the record could not
HOLD. A `test_pit` came back as a hole with strata in it; a `cpt` and a `dcp`
came back as a hole with nothing in it at all, because a depth SERIES of three
channels has nowhere to go among layers, samples and driven records. The
corpus has **251 test pit pages, 54 DCP pages and 23 CPT pages**, and every one
of them was read as a boring or not at all.

### A test pit is a log, and the log reader reads it

A pit form IS a form - a depth ruler, a description column, samples - so
`log_reader.read_log` takes it, `log_floor.seed_from_grid` seeds it, and the
merge rules are the ones every reader shares. What the train added is the one
thing a pit has that a hole does not: `Investigation.pit`, a `PitDimensions`
of a length, a width and the pit's own depth.

**No test pit in the corpus prints a labelled plan dimension.** Fourteen pit
logs across four reports were checked and not one carries a length or a width
as a field. Every one of them names the machine and its BUCKET - *"... with a
55 cm bucket"*, *"... Rubber Tire Backhoe 90 cm Bucket"*, *"... w/ 1 m wide
bucket"* - and a trench dug with a 90 cm bucket is 90 cm wide. So
`log_floor.bucket_width` reads it out of the equipment field, records it as the
pit's WIDTH at a lower confidence than a labelled dimension would carry, and
says in the provenance note exactly where it came from. The prompt tells the
model the same rule, and tells it that a bucket mentioned in a *remark*
("backfilled with the bucket") says nothing about how wide the pit was.

A printed form sets a label and its value as two separate text runs on one
baseline, so nothing splits on a colon; `log_floor.header_pairs` reads both
shapes, which is what made the bucket visible at all.

**A pit photographed with a sketch** - a picture of the excavated face with the
contact depths written beside it and no ruler for a ruler-fitter to find - is
the one log that is read from the picture first. Its floor is the header fields
alone. The depth gate normally REFUSES every depth on a log with no scale, and
that stays true for a borehole; on a PIT with no ruler a depth is accepted, at
a reduced confidence and listed as a change, because the numbers printed on the
sketch are the log and refusing them would refuse every depth such a pit ever
states.

### A sounding is not a log

`read_sounding(doc, item_pages, engine, kind="cpt"|"dcp", budget=2)` returns an
`Investigation` carrying a `CPTData` or a `DCPData`: a depth series of `qc`,
`fs` and `u2` in the units printed, or blows over the printed increment with
any printed index column, plus the cone or the hammer that made it.

**Two shapes of sheet, read differently, and the scorecard prints the split.**

* **TABULATED.** The sheet prints the series as a table and THE TABLE IS THE
  RECORD, read deterministically. planlens' detected tables first; then, where
  a sheet rules no table at all - which the corpus's tabulated soundings mostly
  do not - the text lines banded by their own geometry. That band reader had to
  learn three things the real sheets do: a header can be **staggered over five
  printed lines** (*"Nbre de | Resist. | Contrainte dyn."*, the plot's title,
  *"Profondeur"* alone, *"coups | dynamique | Admissible"*, *"[m] |
  (daN/cm2)"*), so a window of lines is clustered by x-overlap into columns; a
  column with no role is still READ, so its numbers cannot snap into a named
  neighbour; and a long table's depth cell drifts onto its own printed line, so
  the series ends after three depth-less lines rather than the first. One model
  call then goes on the HEADER, and the brief tells it **not to re-type** two
  hundred rows it would get wrong.
* **PLOTTED.** The floor is the header fields and THE AXIS RANGES read off the
  plot's own text - the axis titles with their units and the tick labels. Those
  ranges are what makes a digitised reading checkable: a qc of 42 MPa on an
  axis that runs to 20 is refused here rather than argued about later. The
  traces are digitised through the picture with `zoom_plot` at a fixed depth
  step, one point per depth with every trace's value at it, and a point where
  one trace CROSSES another says so and carries a lower confidence.

Reading the axes took three passes to get right, and each failure is worth
knowing: a tick label belongs to ONE axis (a cone sheet prints tip resistance
and friction ratio along the same printed line, one rising and one falling, and
the first version gave both scales to whichever title it met first); a scale is
EVENLY SPACED, linearly or in decades, which is what tells a real axis from the
three rising numbers inside the little cone symbol every Dutch sounding prints
in its corner; and a depth axis is always LINEAR, which is what stops a
logarithmic resistance scale being read as one.

**The vertical axis is not always a depth.** Many Continental sheets plot
against an ELEVATION on a datum with the numbers going up the page;
`CPTData.vertical_axis` says which, and nothing converts one to the other.

### What Python refuses

A depth outside the sheet's own depth axis. A channel value outside its printed
range by more than one tick. A negative tip resistance. A series with no depth
unit. Each refusal is recorded with the range it fell outside, because a
reviewer needs to know the reader tried.

### DIGGS

`diggs_geo:StaticConePenetrationTest` for a cone sounding, whose dictionary
entry names `tip_resistance`, `sleeve_friction` and `pore_pressure_u2` as the
properties that occur under it. A sounding of four hundred readings is ONE Test
with four hundred ROWS - a ResultSet is a table - so the depth of each reading
is a COLUMN of the set and the test's own `location` is the linear extent the
sounding ran over.

**DIGGS 2.6 has no `DynamicConePenetrometerTest`**; the name is nowhere in the
published schema. What it has is `diggs_geo:DynamicProbeTest`, whose own
documentation is *"all methods that involve driving a rod by impact hammer"*
and whose elements are exactly what a DCP record needs - a required
`penetrationTestType`, then `hammerMass`, `hammerDropHeight`,
`totalPenetration`. So that is what a dynamic probe is written as.
`diggs26.parse_diggs26_soundings` reads both back, row by row, at each row's
own depth, and the round-trip gate checks **every reading**, not a count and
not a spot check.

### The hand truth and what it produced

**Thirteen sheets over six reports**: five test pits (five different firms'
forms, one of them a photograph, one under a DRAFT watermark the grid reads
nothing off), four cone soundings (two tabulated, one the SAME sounding
plotted, one a negative case) and four dynamic probes (three tabulated, one
four-soundings-to-a-page).

**The floor alone, measured 2026-09-21** (no model, no network): **73 %
overall (257/353)**, which splits into **96 % on the four TABULATED sheets
(217/226)** - 100 % of every series metric, the misses being header fields a
pattern cannot read - and **31 % on the nine PLOTTED ones (40/127)**, where the
floor holds no points at all by construction and the whole of the reader's job
is the digitising. Averaging the two would hide the only number anyone wants.
**Every sheet is in sample**, so a blind set is owed (`FUTURE_IDEAS.md`).

One finding is recorded in the truth's own README and is worth repeating here:
**a plotted cone sounding cannot be hand-truthed as a digitised series on this
corpus.** The traces oscillate faster than the sheet's own printed grid - tip
resistance goes from 0.24 to 9.85 MPa across a fifth of the 1 m grid spacing -
so at most whole-metre levels there is no single value of the trace to read.
That is why `default_step` takes the FINER of the printed grid and
`CPT_STEP_M`, never the coarser.

### Scoring (`sounding_scoring.py`), and the stage

A test pit is scored by the log scorer's own metrics - which it CALLS rather
than copies, so the two cannot drift - plus `dimensions`. A sounding is scored
per depth step: a truth point is found when the reader has a reading within
0.05 m of that depth and its value is within **max(5 %, one axis tick)** for
tip resistance, sleeve friction and the printed index, **10 %** for pore
pressure, and **exactly** for a blow count, which is a count. Cluster stage
**`"soundings"`** with `sounding_budget=2` and a `soundings/` truth subdir; the
local twin is `measure_wp6_soundings.py`; the `ingest` stage's record scoring
includes them.

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

### The accuracy levers (5.24.0)

The first full cluster run put this reader at 64 % recall and 74 % precision
over eight reports, and the per-question table said most of the loss was not
comprehension. Five levers follow from it, all of them default-on and each of
them switchable.

**1. The front matter always goes in.** `reading_pages(doc, narrative_pages)`
is the input set: every page labelled `cover`, `letter`, `toc` or `narrative`,
plus the first `front_pages=50` pages of the report whatever they were
labelled, deduplicated and in page order, capped by a token budget the FRONT
wins. `primeContractor`, `primeAe`, `postName` and `projectNumber` are answered
on the transmittal letter and the cover, and neither is a "narrative" page: on
the eight-report run `postName` was missed on three of the four reports that
have one. A page in the set with less than 120 characters of text — a scanned
letter — is sent as a PICTURE instead, up to eight of them.

**2. A glossary and conventions block in the prompt**, held as DATA in
`narrative_glossary.py` so the owner edits a file rather than a prompt string.
Six rules, every one of them marked **DRAFT and awaiting the owner's
confirmation**:

1. A count of explorations the report says it did NOT do is `0`, not null.
2. `null` means THE REPORT DOES NOT SAY — never zero, never not-applicable.
3. The four "mention" questions are about whether the report DISCUSSES the
   topic anywhere, not whether the work was done or the finding was positive.
4. `postName` is the CITY of the diplomatic post and nothing else.
5. `primeAe` is the architect-engineer of record, not the geotechnical firm
   and not the construction contractor.
6. `earthHazardsExposed` uses the listed phrases and no others, and NEVER
   includes seismic shaking.

Each carries the evidence it came from, so the owner can argue with the
reading rather than only with the rule. `REPORT_INGEST_PLAN` §7 — the owner's
standing review of the two query schemas — now points here as the place every
ruling goes.

**3. Per-question retrieval over the whole report.** Seven questions are as
often answered in an appendix table as in the prose (`siteClass`,
`asceSevenVersion`, `seismicCodeUsed`, `soilCorrosion`, `bearingCapacity`,
`liquefactionPotential`, `reportDate`). `retrieval_passages(doc)` searches
EVERY page for each one's key phrases — exact first, planlens' fuzzy search
where exact found nothing, because a fuzzy pass costs a full scan per phrase —
and the passages travel with the brief carrying their page numbers, which are
then citable.

**4. Deterministic answers where the data exists.** Pass
`investigations=record.investigations` and `boringCount`, `testPitCount`,
`cptCount`, `boringDictionary` and `testPitDictionary` come from the logs the
labeller found rather than from the prose; the narrative's own answer stays in
`stated_counts` and any difference is an `unresolved` row the reconciler raises
again. A ZERO does not overrule a stated number — no cone logs were read is not
the same as no cones were pushed — but it is stored where the narrative said
nothing at all, which is convention 1 above.

**5. A quote gate.** Every citation names a page and repeats that page's own
words, and both halves are checkable. An answer none of whose quotes can be
found on the page it cites (fuzzy, ≥ 85) drops to confidence **0.3** and is
listed in `unresolved`; it is not deleted, because a right answer with a bad
quote is still a right answer. An answer with no citation at all keeps its
confidence and is flagged separately. `result.confidence` is the map.

**6. And the scorer reports both views.** `narrative_scoring` gains a LENIENT
view of the four long free-text fields (`recommendedFoundations`,
`structureList`, `bearingCapacity`, `strata`): an item matches at a partial
ratio of 70, or on a number and its unit in common. Strict and lenient print
side by side and neither is "the" score — the gap between them says whether the
reader found the thing and worded it differently or did not find it. The
per-question table gains a **why** column naming the dominant kind of miss, and
`convention` there means a house rule rather than a reading failure.

The levers are **unmeasured until the next cluster run**: the numbers above are
the 5.23.0 baseline, and the development engine is a checkpoint rather than a
result. `measure_wp4_narrative.py` gains `--front-pages` and `--lenient`.

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

A report BOUND INSIDE another one gets the same five outputs in its own
folder (`bound/<id>/`) and a row of its own in the same library, with `parent`
set to the key of the report it came out of. Its DIGGS file holds ITS
explorations and the parent's holds the parent's: DIGGS 2.6 has no way to say
"this sampling feature belongs to another report bound into this one", and a
file that quietly merged the two would be wrong in the one way the whole
exercise exists to prevent. See "Reports bound inside reports" below.

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
di_result=…, label_policy=…, review_mode=…, vision_engine=…)` is the whole
ingest: open (with DI when given) → roles, outline and ledger → triage → the
page-label VOTE (the rules, one vision pass on a cheap tier, the printed form)
→ the label review over the pages the voters split on → work items from the
settled labels → the reports BOUND inside this one → one reader per item →
reconcile → each bound document round the same loop → write.

See "The page labels are a vote" below for `label_policy` and `review_mode`.
`vision_engine` should be an engine on a CHEAP tier, since that voter looks at
every page; without one it runs on the same engine as everything else, which
works and costs more than it needs to. `label_policy="rules"` calls no vision
pass at all and reproduces what this graph did before the vote.

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
printouts are recorded as a QA entry saying they were not read, so a reviewer
knows the pages exist and were skipped on purpose; a report bound inside this
one is read as its OWN record (`ingest_bound=True`, the default) and listed on
`bound_documents`.

`run_folder(folder, engine_factory, out_dir=…, vision_engine_factory=…,
label_policy=…, review_mode=…)` drives the same graph headless
over a folder into one `reports.db`, with an `INDEX.md` of what it came to. It
resumes, one bad report cannot stop it, and two files with the same bytes are
one document sharing one library row — which it says out loud, so a folder of
300 files that makes 297 rows is not a mystery. It can also make MORE rows
than files, because a report bound inside a report is a row of its own; the
index counts them in a `Bound in` column.

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

## The report library (`library.py`, `library_agent.py`)

The ingest reads ONE report. The library answers questions ACROSS all of them:
which reports belong to a post, what each one recommended, where a value is
printed, where two readings disagree. It reads records that were already
written — no PDF is opened, no reader runs — which is what makes every answer
citable and what makes the whole thing cost nothing but the model that writes
the sentence.

### The folder is the library

```
<root>/reports.db                  the index (writers.upsert_report)
<root>/<ID>/report.record.json     the record
<root>/<ID>/report.page.md         the WikiLLM page
<root>/<ID>/report.summary.md      the summary
<root>/<ID>/bound/<child>/...      a report bound inside that one
```

That is exactly what `run_folder` writes, and what one `ingest_report` writes
for a single report — so a single ingest's own output folder is a library of
one, and the report just read is askable in the same session.

**The index is DERIVED and the records are the truth.** A folder restored from
SharePoint with no `reports.db` beside it, or one whose records were rewritten
after the database was built, rebuilds on the next question — both the
`reports` rows the writers own and the full-text index the library adds. The
check is mtimes: the newest record, page or summary under the root against a
stamp in the database's `meta` table. A rebuild reads the key off each
`report.page.md`'s front matter, so a report keeps the identity the ingest gave
it from its source file's own bytes rather than acquiring a second one. Nothing
is ever written back into a record.

A REPORT BOUND INSIDE ANOTHER is a report of the library in its own right, with
its own row, its own id and `parent` naming the one it came out of. Its borings
are its borings; the parent's counts stay the parent's.

### The query layer (`library.py`)

`Library(root)` and ten functions, each returning plain data with the report id
and the PDF pages behind every row:

| Query | What it answers |
|---|---|
| `list_reports(post, property_type, phase, firm, date_from, date_to, document_type, has_kind)` | the reports matching every filter given; `has_kind` takes an exploration kind, a laboratory test kind, `calculations` or `bound` |
| `find(text, k, section, report_id)` | FTS5 over the records, the library pages and the summaries, with a rapidfuzz fallback on the field values; a snippet and the pages per hit |
| `where_is(text)` | the same search shaped as places: one row per report and page |
| `facts(report_id, fields)` | any of the 37 narrative fields with its pages and the quote it came off, plus the counts and the QA summary; a field the report did not answer is NAMED rather than returned empty |
| `compare(field, report_ids)` | one field across several reports as a table, with the reports that did not answer it named |
| `explorations(report_id, kind)` | the holes, pits and soundings with depths, layers, samples, driven records and water |
| `lab_summary(report_id, kind)` | how many of each test kind, and each test with its link, depth, standard and values |
| `calculations(report_id)` | each printout: the program, the method, what it was for, what it concluded |
| `disagreements(report_id)` | the six QA kinds a PERSON should look at — a `note` and a `skipped` are not among them |
| `library_stats()` | what the library holds, by document type, status, confidence, firm, post and kind |

**Search is FTS5 the way the reference layer does it**: a contentless external
-content table over chunks, `porter unicode61`, BM25 with the chunk's subject
weighted over its text. The chunks come from the RECORD, because the record is
what carries pages — a hit that cannot name the page it came off is not a
citation — and from the written page and summary block by block, so anything
those print is findable too. The **fuzzy fallback runs on the field VALUES**
rather than on whole chunks: a twenty-character query against four hundred
characters of text scores as a mismatch however close the firm's name inside it
is. It only runs when the full-text query comes back thin, and never on a query
under four characters.

Every answer stops at `max_rows` (40) and says `truncated` when it did.

### The sub-agent (`library_agent.py`)

```python
agent = build_deep_agent(model, enable_report_library=True,
                         library_root="…/report_ingest_out")   # default OFF
```

That adds the `report_library(question)` primary tool and a `CompiledSubAgent`
named `report_library` — the query layer as ten JSON-Schema tool specs bound to
the app's OWN chat model (unlike the ingest, which runs readers on its own
engine). **Feature-detected on the FOLDER**, not on a package version: a
deployment either has reports read into it or does not, and a library agent over
an empty folder would only teach the primary to ask it things nothing can
answer.

Its system prompt says the one rule — every fact comes from a tool result in
that conversation, never from what a geotechnical report usually says — plus
cite `(report id, page)` after every fact, say plainly when the library holds no
answer, and end with a `Gap:` line per thing left unsettled.

**The ceilings are Python**, for the same reason as the ingest's: deepagents
uses a CompiledSubAgent's runnable as provided and reads no middleware for one.
So the graph counts queries itself — `MAX_TOOL_CALLS = 8` per answer, refused
with a message telling the model to answer from what it has — and every result
is capped at 25 rows and 4,000 characters before it reaches the model.

**The structured response**: `answer`, `citations[]`, `reports_consulted[]`,
`gaps[]`, `queries`, `error`. The citations are CHECKED rather than copied: a
`(report, page)` the answer claims and no query returned is left out of
`citations` and named in `gaps`, so an invented page is visible instead of being
either silently dropped or silently kept. A query that came back empty and a
query that failed are gaps too.

**The ingest's own result now carries `library_root`** — the folder its database
sits in — so a freshly ingested report is queryable with `report_library` in the
same session rather than after a restart.

### Measuring it

The deterministic half runs locally with no model and no network:

```
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp7_library
```

Twenty hand-written questions over a synthetic library of six records (seven
rows — one carries a bound report), each with the report ids and pages a
correct answer must carry. It prints two columns: **the chosen query**, which
is what the sub-agent's tool call returns once the model has picked the right
query, and **search alone**, which puts the question's own prose into `find()`
and nothing else — the floor a model gets when it reaches for search instead.
On 2026-09-21 the chosen query scores **reports 0.949 / 1.000** and **pages
0.939 / 1.000** (precision / recall), and search alone **reports 0.647 /
0.943**, **pages 0.417 / 0.323**. The two remaining false positives are honest
ones: a second report genuinely prints "spread footings", and a second one
genuinely discusses seismic hazard.

**The model half is measured on the cluster**, because the model that will do
the work is the app's. It needs no PDF and no truth folder of pages — just a
`library_questions.json` beside the library (see
`module_work/report_ingest_harness/library_questions.EXAMPLE.json` for the
shape) and the library folder itself:

```python
# %pip install "geotech-staff-engineer==<the release with the library>"
import json
from funhouse_agent.deep.databricks_bridge import PrompterChatModel
from report_ingest.library_agent import answer_question

LIBRARY   = "/Workspace/.../report_ingest_out"        # what run_folder wrote
QUESTIONS = "/Workspace/.../library_questions.json"   # the private truth

model = PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")
asked = json.load(open(QUESTIONS, encoding="utf-8"))["questions"]

rows, hit, missed, extra, said_no = [], 0, 0, 0, 0
for item in asked:
    answer = answer_question(item["question"], model=model,
                             library_root=LIBRARY)
    cited = {c.report for c in answer.citations}
    want = set(item.get("reports") or ())
    hit += len(cited & want); missed += len(want - cited)
    extra += len(cited - want)
    if item.get("must_say_no"):
        # The unanswerable ones: the right answer cites nothing and says so.
        said_no += int(not cited and bool(answer.gaps))
    rows.append({"id": item["id"], "cited": sorted(cited),
                 "want": sorted(want), "queries": answer.queries,
                 "gaps": answer.gaps, "answer": answer.answer})

print(f"cited report ids: precision {hit / max(hit + extra, 1):.3f}, "
      f"recall {hit / max(hit + missed, 1):.3f}")
print(f"unanswerable questions answered as unanswerable: {said_no}")
for row in rows:
    print(row["id"], row["cited"], "want", row["want"],
          f"({row['queries']} quer(ies))")
```

Score it on the CITED report ids rather than on the prose: what the answer
cited is what a reader can check, and an answer that is right about the ground
and cites the wrong report is wrong in the way that matters. Print the `gaps`
beside them — a question the library genuinely cannot answer should come back
as a gap, and an answer with no gaps to a question with no answer is the
failure this design exists to prevent.

## The page labels are a vote (`label_vote.py`)

Everything downstream hangs on what each page IS, and until this train one
voter decided it — planlens' rules — while an agent loop costing about $0.45 a
report checked every page afterwards. The corpus run of 2026-09-20 (ledger runs
7 and 10) says both halves of that were wrong:

| | in sample | honest blind | about |
|---|---|---|---|
| rules | 0.908 | 0.767 | free |
| rules + the label review | 0.916 | 0.850 | $0.45 a report |
| vision, sheet mode, cheap tier | 0.655 | 0.867 | $0.05 a report |

The two cheap voters are **complementary by label class** rather than one being
better. The rules own `appended_report` (494 in-sample pages the vision pass
never once emits), `other` (216, the same) and `calculation` (1,009, of which
vision misses 41 %); vision owns `plan`, `profile`, `photos`, `cover`, `toc`
and `figure` recall outright. And the review breaks nearly as many labels as it
fixes on the reports its prompt was tuned against, while earning its money on
the ones it has never seen.

So: **three cheap voters label every page, and the expensive look goes where
they split.** This is the owner's own standing direction of 2026-09-18 — *"if
multiple methods say different things, it could trigger an extra review …
would be good to have confidence values associated with the classifications and
data extractions."*

**The voters.**

1. **The rules** — planlens' `page_roles`, with its own per-page confidence.
   Free, deterministic, and it labels every page.
2. **Vision** — one pass over the pages as PICTURES on the cheap tier
   (`vision_labels.classify_pages_by_vision`, `sheet` mode by default: one
   call per contact sheet of six pages), with its own per-page confidence.
3. **The printed form** — `log_templates.recognise_pages` where a private
   fingerprint file is in force, which is not the default state. It recognises
   the FORM, not the label: a fingerprint says "this came off that firm's
   boring-log template" and never which of the four exploration labels the page
   should carry. So it votes for the CLASS, and the whole of what that buys is
   one rule — where a fingerprint claims a page and the policy's own answer is
   not one of the four log labels, a log label either of the other two voters
   gave wins; where neither offered one, the answer stands and the page counts
   as a disagreement.

**The policies** (`label_policy`, combined by `label_vote.combine`):

| policy | what it does |
|---|---|
| `structural` (default) | vision everywhere except `appended_report`, `other`, `calculation` and `lab_test`, which the rules own. It learns nothing — it is the ledger's reading written down — which is what makes it the one to beat. |
| `confidence` | whichever voter said so more confidently, ties to the rules. |
| `trust` | whichever voter a per-label trust table favours for the class the RULES put the page in. Needs a table learned in sample, which `stages=("vote",)` writes to `vote/trust_table.json`; **without one it falls back to `structural` and prints a note.** |
| `rules` | the rules' label, always. No vision pass is called at all. This reproduces exactly what the pipeline did before the vote. |

**What the review is given** (`review_mode`):

| mode | what it sees |
|---|---|
| `disagreements` (default) | ONLY the pages the voters split on, plus two either side for context, and the outline as before. Its tool-call budget is `max(20, 0.5 × split pages)` instead of `max(60, 0.25 × pages)`. Its tools are NOT narrowed — an agent that follows a hunch two pages further is allowed to, and a change it makes on a page the voters agreed on is applied and flagged in QA. |
| `all` | every page, which is what every run before this train did. |
| `none` | no review; the vote's labels stand. |

**What the record carries.** `ReportRecord.page_labels` is one `PageLabel` per
page: the label the work items were built from, its confidence (the highest
among the voters that gave that label), `agreed`, the policy, whether it was
settled by the `vote` or by the `review`, and every voter's own label and
confidence — the template voter's entry carrying its `family` and an empty
label, because that is what it actually claimed. Every split the review did not
settle is a `QAEntry(kind="label_disagreement")` naming both voters and both
confidences. The graph also writes `labels.json` beside `vision.json` and
`review.json` in the report's folder: the policy, the mode, the split pages,
what the review changed, and what each pass cost.

## Reports bound inside reports (`bound.py`)

A geotechnical report is very often not one report. An earlier firm's whole
investigation is reproduced as an appendix; a bridging report is bound into the
design-build report that answers it; a data report sits inside a design report.
The corpus measurement of 2026-09-20 (ledger run 7) has triage calling **19 of
38 reports `multi_document`, with one to four bound documents each**, and the
hand narrative notes for two of them say the same thing in the owner's own
words: the borings and test pits behind that tab belong to the EARLIER
investigation, not to the report they are bound into.

Before this train those pages were listed in a QA entry and never read. The
alternative that was never on the table is reading them into the SAME record:
an earlier firm's B-1 and this report's B-1 in one list of investigations, a
2009 water level beside a 2026 one, and a boring count that is the sum of two
investigations and the truth about neither.

**So a bound document becomes its own record.** The parent keeps those pages
labelled `appended_report` and LISTS the child; the child is built from the
same pages by the same pipeline — its own identity, its own labels, its own
work items, its own readers, its own reconcile, its own exports — and nothing
is attributed across the boundary.

**Where the page range comes from.** Two sources, and they are not the same
kind of evidence:

| source | what it is |
|---|---|
| triage's `bound_together` | a model that has read the whole ledger, the outline and the front matter saying "pages 112-184 are their own document". It sees the CONTENTS LIST, so it catches a bound report the pages themselves are shy about. |
| the record's `appended_report` labels | planlens' rules label a page `appended_report` when it sits inside a document the section builder found nested in this one. Structural, cheap, and the label class the rules demonstrably own — 494 in-sample pages a vision pass never once emits. |

They usually agree. Where they do not, the record takes the **union** and says
so in a `QAEntry(where="bound.extent")` naming both: a page wrongly included in
the child is recoverable, and a page wrongly left in the parent is an earlier
investigation's boring attributed to this report, which is the error the whole
exercise is about. Two claims that overlap or merely touch are ONE document. A
run shorter than `MIN_BOUND_PAGES` (4) is not a document — a reproduced log
sheet, a two-page letter from a previous consultant — and stays in the parent
with a `bound.short_run` note.

**What each document costs.** One extra structured call per bound document:
`identify_bound` reads its first four pages and answers the title, the firm,
the date, the document type and how it is bound in, all of it printed on those
pages or empty. It is cached in the child's folder, so a resumed run does not
pay it twice, and a call that fails leaves the identity empty with a QA entry
rather than losing the child. The vision voter and the log-template recogniser
are NOT re-run: they already answered about these pages, and a page's picture
does not change when you stop calling it an appendix. The rules ARE re-run,
over the child's pages rebased to zero — which is what makes its cover a cover
and its boring log a boring log again instead of four pages of
`appended_report`. The label review does not run over a bound document; a
split the vote leaves is a `label_disagreement` entry in the child's QA.

**What the parent and the child carry.**

```
ReportRecord.bound_documents: [BoundReport]   # on the parent
  bound_id report_id title firm date kind document_type
  pages first_page last_page n_pages said_by counts folder record_path read

ReportRecord.parent: ParentReport | None      # on the child
  report_id bound_id pages first_page last_page n_pages record_path
```

**Page numbers are the parent file's throughout.** There is one PDF, and a
reviewer sent to check a value opens it at the page the record names — so a
child record's provenance, item pages and page labels are all 0-based indexes
into that same file, and the child's own extent is recorded once, on `parent`.

**The parent's narrative reader is told.** It is given the ranges with each
document's title, firm and date, and told what to do with them: do not count
their borings as this report's, do not put their identifiers in this report's
`boringDictionary`, and each of them IS a previous investigation and counts
towards `previousInvestigationCount`. The counts and dictionaries themselves
still come from the parent's OWN logs (the deterministic path). The child's
narrative reader gets the mirror of that: a `window` that keeps every page it
is shown and every passage it is searched inside the bound document, so it
never answers off the cover, the letter or the recommendations of the report it
is bound inside.

**Exports.** The child gets its own `report.record.json`, `report.summary.md`,
`report.page.md` and `report.diggs.xml` under `out_dir/bound/<id>/`, and a row
of its own in the SAME `reports.db` with `parent` set to the parent's key. The
parent's summary page gains a "Reports bound inside this one" section — title,
what it is, pages, what it holds by count, and the child's folder — with the
sentence that matters: what they hold is in none of the counts above. The
parent's DIGGS file does NOT carry the child's data.

`ingest_bound=False` on `ingest_report`, `run_folder`, `run_ingest` and
`build_report_ingest_subagent` turns the whole thing off and lists the pages in
a QA entry as before.

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
        calc/       <kind>__<ID>_p<first page>.json …, OPEN.txt
        narrative/  <ID>.json …
```

| File | Without it |
|---|---|
| `MANIFEST.md` | IDs cannot be resolved at all, unless the PDFs are already named `R01.pdf` … |
| `trial_pages_working_r2.xlsx` | the in-sample set runs and produces profiles, but scores nothing |
| `oos_labels.json` | the two out-of-sample sets run and score nothing |
| `truth/logs/`, `truth/lab/`, `truth/calc/`, `truth/narrative/` | that stage refuses to start rather than running unscored |

`truth/` is what `truth_dir` points at: each stage takes its own subfolder out
of it, and the subfolders are named after the stages. `OPEN.txt` beside a
stage's truth files names the reports whose pages the prompts were allowed to
be tuned against, so the blind figure stays blind; without one the built-in
open set is used — except `calc/`, which has **no** built-in open set,
because every hand-truthed calculation is in sample and the scorecard says so
rather than reporting a blind figure that does not exist. (`lab_truth_dir`,
`calc_truth_dir` and `narrative_truth_dir` still override the root, for truth
that is not in one place.)

The Azure Document Intelligence results are read in **either** form —
`<ID>.json.gz` or the uncompressed `DI_data_<original stem>.json` that
Funhouse wrote — from whatever folder `di_dir` names. Without them the scanned
pages are read from their own text layer alone.

```python
# 1. From PyPI through Nexus. planlens 0.6.0 arrives with it.
%pip install "geotech-staff-engineer==5.24.0"
dbutils.library.restartPython()
```

```python
# 2. The restart wiped the fh_* objects; the Funhouse setup notebook puts them
#    back. Use whatever path your own notebooks already use for it.
%run /Workspace/funhouse-sdk/setup_python
```

```python
# 3. One cell, all nine stages. fh_prompter is the object setup just made,
#    and fh_sp_client is the SharePoint client the mirror writes through.
from report_ingest.cluster_scoring import score_on_cluster

results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",          # the PDFs, under their own names
    manifest     = "/Volumes/<your volume>/report_ingest/MANIFEST.md",
    labels_xlsx  = "/Volumes/<your volume>/report_ingest/trial_pages_working_r2.xlsx",
    oos_labels   = "/Volumes/<your volume>/report_ingest/oos_labels.json",
    truth_dir    = "/Volumes/<your volume>/report_ingest/truth",   # holds logs/ lab/ calc/ soundings/ narrative/
    di_dir       = "/Volumes/<your volume>/report_di",
    out_dir      = "/tmp/report_ingest_522",                  # the working folder; the durable copy is below
    sharepoint   = fh_sp_client,            # the durable copy; see "Where the output goes"
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",     # every reader, on the tier the app runs on
    triage_model = "funhouse-gpt-medium",   # one call over a ledger; a cheaper tier does
    sets         = ("insample", "oos_open", "oos_blind"),
    stages       = ("labels", "logs", "lab", "calc", "soundings",
                    "narrative", "vision_labels", "vote", "ingest"),
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
| `durable_dir` | your workspace folder under `geotech_app/`, which persists; never `/tmp` (the owner's rule, 2026-09-20). The run's name is appended and the SharePoint prefix is not, so a run lands at `<durable_dir>/report_ingest_522/`. A `durable_dir` whose own name ALREADY is the run's name is not nested inside itself: `.../results` and `.../results/report_ingest_522` both put the run at `.../results/report_ingest_522`. |

`out_dir` refuses nothing. A workspace path is allowed — that folder is one of
the two places that keep things here — and gets a printed note saying that a
local `out_dir` with `durable_dir=` pointing at the workspace folder writes the
same files and syncs them in one pass, which is the shape that works when a run
writes one small file per report per stage.

**The mirror sends the WHOLE `out_dir`**, which is the point — and that means
`runs/` and `triage/`, whose reasons and rationales can name a firm, a project
or a person, and the derived `labels_map.json`, which holds the label
spreadsheet's own sheet names. The remote folder is exactly as private as
`out_dir` is, so put it where the reports themselves already live. Only
`RESULTS.md` is written to be carried anywhere.

The seven stages are described one at a time below, each with the cell that
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

### Scoring the calculation reader as well (`stages=("calc",)`)

The `calc` stage scores each hand-truthed calculation twice: **before**, over
the FLOOR alone — every (label, value) a pattern reads off the pages' tables
and lines, with no model at all — and **after**, over the `Calculation`
`read_calculation` built from the same open document.

```python
results = score_on_cluster(
    reports_dir   = "/Volumes/<your volume>/reports",
    manifest      = "/Volumes/<your volume>/wp1b/MANIFEST.md",
    di_dir        = "/Volumes/<your volume>/report_di",
    truth_dir     = "/Volumes/<your volume>/report_ingest/truth",  # its calc/ holds <kind>__<ID>_p<first page>.json
    out_dir       = "/tmp/report_ingest_calc",
    prompter      = fh_prompter,
    model         = "funhouse-gpt-high",
    stages        = ("calc",),
    calc_budget   = 2,              # model calls per printout; the reader's ceiling is two
    max_reports   = 2,              # drop this line after the first run
)
```

Six metrics: **kind** (what it works out), **program** (the program and its
version as printed, fuzzy at 85), **method**, **subject** (both fuzzy at 80),
**inputs** and **results** (each value matched by its printed LABEL at 80 and
its VALUE within 2 % or the last printed digit, whichever is looser, compared
in SI wherever both units convert).

**`kind`, `method` and `subject` have no before column.** A pattern over a
page cannot say that a printout is a settlement calculation, that the method
is Schmertmann's or that the sheet is for the north wing's mat. `program`
*does* have one, because a banner is a pattern. And because the floor cannot
tell an input from a result either, it files every value it finds under
`inputs` and the scorer searches **both** lists for every truth value,
counting separately (the `misp` column) how many landed in the wrong one — a
cross-list find counts as found, since the number and its label were
recovered.

**A truth file whose `program` is `null` scores the reader for saying
nothing.** A spreadsheet names no program and naming one is a wrong answer,
not a blank.

**The floor alone, measured here 2026-09-21** with no model, no credential and
no network (`measure_wp5_calc.py --floor-only`, and in the ledger): over ten
runs, 44 pages, six reports and seven kinds — program **80 %** (8/10), inputs
**73 %** (64/88), results **48 %** (36/75). The floor does best on a
spreadsheet whose inputs are a ruled table (a settlement run at 94 %) and
worst where the answer is drawn rather than printed (a slope-stability run at
12 %, its factors of safety in boxes on the sections). That gap, plus the
three metrics a pattern cannot answer at all, is what the reader is for.

**THERE IS NO BLIND SET.** All ten runs were read while the reader's prompt
was written, so every number this stage prints is in sample and none of it is
evidence about an unseen report; `RESULTS.md` says so in place of the open/
blind line. A blind calculation truth set is owed (`FUTURE_IDEAS.md`).

`RESULTS.md` carries `# WP5 on the cluster: the calculation reader`: before
and after per metric, a per-kind table, and a per-run line with the floor's
size, the model calls, the zooms, what was left unresolved, the misplaced
count and the four merge counts (split / kept / added / reconciled).
`calc/<calc id>.json` holds the per-run detail and makes the stage restartable
— a calculation that already has one is skipped.

Locally, `module_work/report_ingest_harness/measure_wp5_calc.py` does the same
scoring against the development engine, and `--floor-only` prints the before
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

### Ingesting whole reports and scoring the record (`stages=("ingest",)`)

The stage that makes what the app would make. For each report in the chosen
sets it runs `graph.ingest_report` end to end — triage, the page-label VOTE,
the label review over the pages the voters split on, the work items, the three
readers on their floors, the reconciler, the writers —
into `out_dir/ingest/<ID>/`: `report.record.json`, `report.summary.md`,
`report.page.md`, `report.diggs.xml` with both gates run, `qa.json` (the
record's QA list alone), `run.json` (the counts), and the per-item files under
`items/` the graph resumes from; `ingest/reports.db` is the library index over
every report ingested. Every file is mirrored like every other run file.

**It does not pay for the review twice.** Where a `labels` run has left
`runs/<ID>.json` in this `out_dir` (or in `review_dir`), its triage profile and
its final labels are copied into the report's ingest folder before the graph
starts, and the graph resumes from them; a report that went through the
`labels` stage costs only its readers here. It resumes per report (a `run.json`
on disk is skipped) and per work item.

```python
HOME = "/Workspace/Users/<you>/geotech_app/report_ingest"     # where things are KEPT here

results = score_on_cluster(
    reports_dir  = "/Volumes/<your volume>/reports",
    manifest     = "/Volumes/<your volume>/report_ingest/MANIFEST.md",
    labels_xlsx  = "/Volumes/<your volume>/report_ingest/trial_pages_working_r2.xlsx",
    oos_labels   = "/Volumes/<your volume>/report_ingest/oos_labels.json",
    truth_dir    = "/Volumes/<your volume>/report_ingest/truth",   # optional: scores the record where truth exists
    di_dir       = "/Volumes/<your volume>/report_di",
    out_dir      = "/tmp/report_ingest_523",
    durable_dir  = HOME + "/results/report_ingest_523",       # the workspace folder: kept
    sharepoint   = fh_sp_client,                              # and/or SharePoint
    prompter     = fh_prompter,
    model        = "funhouse-gpt-high",
    stages       = ("ingest",),
    sets         = ("insample", "oos_open", "oos_blind"),
    review_dir   = HOME + "/results/520_labels",    # a saved label run to reuse; default is this run's own runs/
    label_policy = "structural",                    # how the page voters are combined
    review_mode  = "disagreements",                 # which pages the review is shown
    log_budget   = 6, lab_budget = 4, narrative_budget = 8,
    max_reports  = 2,                               # drop this line after the first run
)
```

`label_policy` and `review_mode` are the two parameters worth knowing, and
they are the production graph's own — see "The page labels are a vote" below.
The defaults are the ones above; `label_policy="rules"` with
`review_mode="all"` reproduces exactly what the pipeline did before the vote,
so the two can be run against each other on the same reports.

`RESULTS.md` gains `# Ingest: the record and its exports`, and inside it
**`## The page labels: the vote, and what went to the review`**: per report the
pages voted on, how many the voters SPLIT on and what fraction that is, how
many the review touched, how many of those were pages the voters had agreed on,
how many splits are still unsettled, and — where the report has hand labels —
the final labels' strict accuracy beside the rules' alone, by the same scorer
the `labels` and `vision_labels` stages use. The split fraction is the number
a production run can compute with no hand labels at all. Then, per report — pages,
the workflow triage chose, investigations by kind, samples, driven records, lab
tests by kind, narrative fields answered / null (37 asked), QA entries as
`disagreement / partial / out_of_range` (the two voters splitting; what a reader
could not settle; what Python refused), DIGGS as `written / schema / read back`
(`valid` against the bundled 2.6 schema when pydiggs is installed, `not checked
here` when it is not, `equal` when the app's own parser reads the file back to
the record's values), calls, tokens, dollars, seconds — then the totals, then
**the record scored against the hand truth** wherever `truth_dir` holds any for
that report: each log truth against the investigations read off its pages, each
sheet truth against the tests read off its page, the hand answers against the
record's two schemas, with the SAME scorers the reader stages use. That last
table is the whole-pipeline score.

Bring back **`RESULTS.md`** from whatever `out_dir` was. That is the whole report,
and it carries IDs, labels, counts and rates only. The per-report runs and the
triage profiles stay on the cluster in `runs/` and `triage/` unless you move
them deliberately, because a change's reason and a triage rationale can name a
firm, a project or a person.

Notes that matter:

- **`out_dir` is the working folder and nothing is refused.** Local disk is
  the sensible place for it, because a run writes one small file per report
  per stage. A workspace path is ALLOWED — the workspace folder is one of the
  two places that keep things here — and gets a printed note saying that a
  local `out_dir` with `durable_dir=` pointing at the workspace folder writes
  the same files and syncs them in one pass.
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

The sounding readers' tests run over five synthetic sheets built here
(`tests/sounding_fixtures.py`: a pit form with a ruler and a bucket in its
header, a cone sounding printed as an unruled column table, the same kind of
sounding drawn as traces with nothing but its axes in text, a dynamic probe
whose column header runs over three printed lines, and a four-page calculation
run carrying two program banners for the item split).

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

The floors are tested the same way, on the same fixtures: what the grid's seed
and the tables' floor carry before any call; that a reply which drops every
seeded value scores no lower than the grid or the tables alone (and the model's
answer, scored alone, lower); that a contradiction keeps both values with one
disagreement and the floor's in the slot; that a contradiction with a box and a
note overrules the floor; that a difference inside the scorer's tolerance is
reconciled, not disputed; that a symbol, a timing, a kind, a link the model
adds is added; that the follow-up on the unsettled list carries a magnified
band and is not made when the budget is spent. The `ingest` stage runs the
synthetic report through the whole graph with a fake engine in the Prompter's
place — every export written, both DIGGS gates recorded, a saved label run
reused so no review turn is spent, a second call free, the record scored where
hand truth exists, the run files mirrored.

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
**5.23.0** puts the floor under the log and lab readers and adds the `ingest`
stage, so the record the app would produce — and its DIGGS file — can be made
and scored on the cluster before the flag is turned on. **5.24.0** adds the
log-template recogniser and the narrative reader's first round of accuracy
levers, and changes no dependency: rapidfuzz, which both use, already arrives
with planlens.

- **`report_ingest*` is in `[tool.setuptools.packages.find]`.** Until that line
  landed, a wheel built from this repo did **not** contain this package, and
  the failure would have appeared on the cluster as an import error rather
  than at build time. Check it after any edit to that list:
  `python -m build --wheel` and look for `report_ingest/` in the wheel.
- **`templates.json.EXAMPLE` is in `[tool.setuptools.package-data]`** under
  `report_ingest`, so the example fingerprint file ships in the wheel. The
  REAL fingerprints are never in this repository.
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
