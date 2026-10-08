# Visual scales: one way to measure positions on a page (W3 design)

**BUILD STATUS (2026-10-08).**
- **Steps 0–5 BUILT in planlens main a3f2a1d** (unreleased):
  - `planlens/document/scales.py`, `raster.py`, `scalefinder.py` and `measuring.py`;
  - `log_grid`'s raster leg;
  - the `measure` tool spec;
  - 48 fixtures and 134 tests.
- **Harness (`module_work/scales_harness/`):** GATE PASS, with 99.93 % of 4,404 readings holding the truth inside their ± and no wrong snap returned without its alternatives.
- **Private spot check:** 32/32 stratum lines on the ten scanned log sheets (median 0.007 m); 0 wrong reads across 835 grading-sheet readings.
- **Departures from this design:**
  - ± is a 95 % half-width;
  - windows of 25 pt or more, or small marks from views over 300 pt, never snap, but list candidates;
  - a vector ruler is re-tied to its ticks only when it is off by ≥ 0.75 pt;
  - truth for six sheets comes from re-running the lead's frame method.
- **Left over:** HANDOFF 8a(viii) (rectangle items read as x, y, w, h in two older functions; speed 1–16 s per dense scan; a silent `values` length mismatch).
- **Steps 6–9 (app side)** are next, after the brief 4 fix list.

**What this is.** The design asked for in `module_work/SCALES_COVERAGE_CROSSCHECKS.md`, item W3. The owner's direction (2026-10-08):

- the depth fix belongs inside one general workflow for visual scales;
- that workflow merges with the location work (`module_work/harness_theory/locating_things_on_a_page.md`);
- it applies to both harnesses, the geotech page and Document Review, and report ingest uses the same piece.

The principle is report ingest's own: **geometry says where, the model says what.** A model reads printed text and says what a thing is. Code finds the drawn thing and converts its position through a fitted scale.

**How it was made.** Read against GeotechStaffEngineer `master` at `b2f73c3` and planlens `main` at `08a1d53`. Experiments were run locally on:

- the private field-session report (counts and error sizes only below);
- public reference PDFs;
- planlens' synthetic fixtures.

Nothing was changed in either repository except this file. No model was called. The experiment scripts are throwaway code in the session scratchpad and are not committed. Step 0 of the build turns them into fixtures and a harness.

---

## 0. In short

- **The failure is a position read by eye.** In the 2026-10-06 session the agent read all 48 SPT rows right. It placed layer boundaries by eye against the depth scale.
  - Re-measured here against the drawn lines on all ten scanned log sheets: the agent's 31 boundaries are a median **0.20 m** off.
  - 29 of the 31 are more than 0.1 m off, 7 are more than 0.3 m off, and the largest is 1.1 m.
  - Measured in code from the same sheets, the boundaries carry about **±0.02 m**.
- **Code can measure these pages with numpy and PyMuPDF alone.** No OCR and no OpenCV are needed. On all ten scanned log sheets, at 0.4–0.8 s a page:
  - the page's skew;
  - the 12 column rules and the depth frame;
  - all ten depth labels;
  - every stratum line.

  On all twelve scanned grading sheets, code found the linear axis and the **log axis**, and identified the log axis's decades from the pattern of its gridlines alone. On one sheet the plotted points match the sheet's own printed table to within 0.25 percentage points and 2 % of sieve size.
- **What the model is still needed for:** the label VALUES ("1.0 … 10.0", "0.001"), which it reads reliably, and naming which thing to measure. Positions come from code.
- **One primitive:** a **Scale**, a fitted map from page position to a quantity. It has:
  - linear or log axes;
  - anchors with provenance;
  - a residual, an uncertainty and a confidence.

  The four scale shapes in the code today become sources of it: log_grid's `Ruler`, the sounding reader's `PlotAxis`, the PDF's stored `Viewport`, and the cross-section importer's scale factor.
- **One measuring step.**
  1. The model names the thing and gives a rough box (a pixel box from a look, converted in code as since `cf43c90`).
  2. Code snaps to the drawn line, edge, marker or curve near that box.
  3. Code converts through the scale and returns the value with its ± and what it snapped to.
- **One new agent tool, `measure`, on both pages.** Called with no location, it lists the page's scales. The agent learns it from its own description only, with no prompt rule. Report ingest calls the same planlens functions directly: the log floor, a lab plot-against-table check, and digitised sounding traces.
- **Three findings correct the plan of record.**
  1. `log_grid` finds **no stratum lines on any scan, even with OCR**. It reads rules only from vector drawings.
  2. Its ruler assumes each depth label is **centred on its depth**. On the session's scanned form the labels sit about 0.1 m above it; on the vector form they are centred.
  3. `log_grid` is **not on either agent's tool list**.

  Details are in §11.

---

## 1. What exists today

"Vector" is a PDF page drawn as lines and text. "Scan + OCR" is an image page with Azure Document Intelligence (DI) lines. "Scan, no text" is an image page alone. On Funhouse, Foundry and probably Tiny Apps the only OCR is DI, which is paid and optional. planlens' free RapidOCR leg imports OpenCV, so it cannot load on those hosts (§11).

| Piece | Where | What it actually measures | Vector | Scan + OCR | Scan, no text | Who can reach it |
|---|---|---|---|---|---|---|
| **Depth / elevation ruler** | planlens `document/loggrid.py`: `Ruler`, `_fit_ruler`, `_ruler_candidates`, `_best_rising_run` | A least-squares line `depth = intercept + slope·y` through the **y centres of printed number labels** in one column. Gates: ≥3 ticks, longest monotone run ≥60 % of the band, residual ≤0.2 of a step. Elevation rulers only where a header says elevation. | yes | yes (labels from DI/OCR lines) | **no**: "no text to place" | report ingest only; planlens toolkit and MCP. **Not on either agent** (`document_tools.DOCUMENT_TOOL_NAMES` leaves it out). |
| **Stratum lines / layers** | `loggrid.py` `page_rules`, `_page_boundaries`, `_build_layers` | Horizontal **vector** rules crossing ≥55 % of the description column, with underlines excluded. Also printed contact depths in a margin, and USCS-symbol changes once calibrated. Each line's y goes through the ruler. | yes | **only printed ticks and USCS changes**: a scan has no vector rules | no | as above |
| **Depth gate** | report_ingest `log_reader._depth_window`, `_Builder._depth` | Refuses a model depth outside the ruler's page window. With no ruler, refuses **every** boring depth; a test pit keeps printed depths. | yes | yes | refuses all | report ingest |
| **Log floor** | report_ingest `log_floor.seed_from_grid`, `merge_investigations` | Turns grid cells and layers into the starting record. The model may add, may correct only with evidence, and may never drop. | yes | partial (no stratum lines) | nothing | report ingest |
| **Plot axis ranges** | report_ingest `sounding_reader.PlotAxis`, `axes_from_text` | The tick VALUES printed beside an axis title (linear or decade-regular), as a **range gate** only. It has **no positions** and cannot convert a position to a value. | yes | yes | no | report ingest |
| **`zoom_plot`** (three copies) | `sounding_reader._Tools.zoom_plot`, `lab_reader`, `calc_reader` | A 300 dpi crop. **The model reads values against the ticks by eye.** | by eye | by eye | by eye | report ingest |
| **Quantity** | planlens `ir/measure.py` | Value + unit + confidence (min-composed) + **relative** uncertainty. Page points refuse to become feet without `.scaled(...)`. | n/a | n/a | n/a | planlens internals |
| **Stored PDF scale** | planlens `document/scale.py`: `page_viewports`, `viewport_at`, `read_markup_measure`, `to_quantity` | The `/VP` viewport `/Measure` (`/C` × points → units) and measurement markups. It detects the untouched 1:1 identity and refuses it. | yes (rare: 0 calibrated pages in a 162-PDF corpus) | n/a | n/a | page map (`scale` field); review page's `drawing_dimensions` behind `GEOTECH_REVIEW_GEOMETRY` (off) |
| **Scale notes** | planlens `pdf/scale.parse_scale_annotations`; page map `PageSummary.scales` | Parses "1:N", '1" = 20 ft', "1 cm = 2 m" from text, **only on pages classed `drawing_sheet`**. Proposals only, never applied; they assume true plot size. | yes | yes | no | page map row; geotech `pdf_import.propose_scale` |
| **Two-point calibration** | planlens `pdf/scale.calibrate_scale`; `ir.from_pdf_vector(calibration=)` | One isotropic m-per-point factor from two points and a known distance. **No origin or datum, no separate x/y scale** (a profile with vertical exaggeration cannot be expressed). | yes | n/a | n/a | geotech `pdf_import`, `drawing_ir` |
| **Dimension finder** | planlens `ir/queries.find_dimensions` | Dimension constructs and their drawn length in points. A graphic scale bar is treated as a **decoy** and capped. | yes | no | no | geotech `drawing_ir`; review `drawing_dimensions` (switch off) |
| **Point-pattern spacing** | planlens `ir/spatial.point_pattern_stats` | Spacing conventions over given points. A physical unit only with a caller-supplied `units_per_input`. | n/a | n/a | n/a | not on any agent |
| **Pixel boxes → page points** | app `funhouse_agent/vision_view.py`: `PIXEL_INSTRUCTION`, `boxes_to_grid`, `px_box_to_page`, `location_error`, `precision_note`, `zoom_pad`; planlens `budget.image_box_to_page` | Converts a vision model's px box to page points exactly. Measured: 1–6 pt off on a whole sheet at 2,048 px, about 1 pt in zooms. The old 0–999 grid was 57–91 pt off. | yes | yes | yes | both agents (looks, zooms, marks) |
| **Chart read-off** | app `vision_tools._dispatch_read_reference_figure_at_budget` | One vision side call reads a value off a reference chart **by eye**. The result says "accurate to a few percent on linear axes, looser on log axes". | by eye | by eye | by eye | geotech agent (and its `references` sub-agent) |
| **Numpy matcher (precedent)** | planlens `document/findlike.py` (`_ink`, `_peaks_numpy`) | Template matching over a PyMuPDF grey render with numpy only. It already runs where OpenCV cannot. | yes | yes | yes | both agents (`find_like`) |

---

## 2. What the experiments showed

All runs were local: numpy and PyMuPDF only, 200 dpi grey renders in the displayed page frame. Private pages are reported as counts and error sizes.

**E1. `log_grid` on the session's log pages.**
- On the 4 vector log sheets it found a ruler on 4 of 4 (residual 0.000, confidence 0.98) and 3–4 layers each.
- On the 12 scanned pages it found nothing: no columns, no ruler, no layers, with the warning "no text to place".

**E2. The scanned log sheets measured from pixels (no OCR).** All 10 log sheets:

| What | Result |
|---|---|
| Skew | −0.52° to +0.40°, measured from the long rules |
| Column rules | 12 found per sheet |
| Depth frame | found: the rule under the header and the foot rule |
| Depth labels | 10 of 10 per sheet, found as ink blobs in the depth column |
| Label spacing | 46.13–46.21 pt, sd 0.13–0.31 pt |
| Straight line through the labels | max residual 0.20–0.54 pt, about 0.012 m |
| Stratum lines | 2–6 per sheet, 32 in all |
| Fit of each line's own y along it | under 1 px (0.36 pt) |

- The deskew is a two-pass shear, a few lines of numpy.
- The two trial-pit sheets in the same range use a different form: contacts are printed as text, and the depth column is not on the left. The experiment's "depth column is the leftmost band" shortcut failed there. **The design finds the depth column from the labels themselves, not from where it usually sits.**

**E3. Where a depth label sits relative to its depth.**
- **Scanned form:** on all 10 sheets the label CENTRES sit **4.4–4.9 pt above** the depth they mark. The marked depths are the frame lines at 0 and 10 m, and the line through the labels extended. That is about **0.10 m** at the sheet's 46 pt/m. The label BOTTOMS sit 1.3–1.8 pt above (about 0.03 m).
- **Vector form:** the label centres sit on the drawn tick rules to within **0.18 pt**.
- So `log_grid`'s centred-label rule is right for one form and 0.1 m wrong for the other. A fit's residual cannot see a constant offset.

**E4. The agent against code, same pages.**
- The session's 31 layer boundaries (five borings, ten sheets) were matched to the nearest code-measured line:
  - median 0.20 m;
  - 29 of 31 more than 0.1 m off;
  - 7 more than 0.3 m off;
  - largest 1.1 m.
- Code found 32 lines: the agent merged layers away.
- Code uncertainty is about ±0.02 m: frame-anchored, line located to 0.4 pt, label line ≤0.5 pt.

**E5. Snapping from a rough box.** A simulated rough box was dropped around each true stratum line and snapped to the nearest line crossing the description column.

| Rough-box error | Window | Correct line | Window held more than one line |
|---|---|---|---|
| sd 1–6 pt (pixel boxes, as measured) | ±12 pt | **99.9–100 %** | 0 % |
| sd 30–70 pt (old 0–999 whole-page boxes) | ±79 pt | 51–83 % | **60–64 %** |

Snapping finishes a good box; it cannot rescue a bad one. It must say when the window holds more than one candidate.

**E6. The scanned grading sheets (log x-axis, no text).** On 12 of 12 sheets:
- 11 horizontal gridlines at regular spacing;
- 46 vertical gridlines;
- the **log-decade pattern** found from geometry alone: the gaps between minor lines sit in the proportion of log10(k+1) − log10(k), with pattern error ≤0.002 of a decade;
- the minor gridlines within 0.27 pt of the fitted log axis;
- 10–15 plotted markers each, found as compact ink blobs.

On one sheet checked against its own printed table:
- 14 of 14 points matched;
- particle size within 2.1 %;
- percent passing within 0.25 points (median 0.11).

A reader needs to supply only one decade label and the 0/100 ends.

A trap found on the way: one gridline was broken by the axis title, a length filter dropped it, and numbering 0–100 by index put every value about 3 points low with a 4.6 % residual. **Values go to gridlines by fitted spacing, never by count, and the residual exposes the mistake.**

**E7. A vector plan sheet with a stated ratio and a graphic bar.**
- The bar's ticks were found under its three labels, with the labels centred on the ticks to 0.5 pt.
- The bar's scale and the stated ratio at true plot size agree to **0.01 %**.
- The bar's middle tick reads 4.998 against its printed 5.

**E8. A public design chart: GEC-12 Figure 7-15, an embedded image.**
- 17 horizontal and 16 vertical gridlines found.
- The curve was read at ten friction angles by darkness: the curve is black, the grid grey.
- **Against the repository's own hand digitisation** (`geotech_references/gec_12/figures.py`, 2° steps), the code read is:
  - the same at 36°;
  - 19 % lower at 32° and 10 % lower at 34°;
  - 3–6 % higher at 38–43°.
- Zooms at 32° and 42° show the code read is right: about 16 against 20 tsf, and about 296 against 280 tsf.
- **This is outside W3. It should be verified before anything changes, because `axial_pile` uses that table.** It shows what a code read is for.

**E9. How reference charts are stored.**
- Of 364 catalogued figure pages sampled across 32 references, **260 (71 %) carry the figure as an embedded image**, and 200 of those have no number in any text layer.
- 26 (7 %) are vector.
- Chart reading has to work on raster images without OCR, by the same route as a scanned log.

---

## 3. The gaps

**Scanned logs without OCR.**
- Nothing is placed (E1). Report ingest then refuses every boring depth.
- The geotech agent eyeballs (E4).
- Document Review has no route at all.

**Scanned logs with OCR or DI.**
- A ruler is found, but on the centred-label assumption (E3).
- Columns come from header text only.
- **No stratum line is ever found,** because pixels are never looked at.

**Vector logs.** Drawn tick marks and frame lines are not used to anchor the labels (E3).

**Plotted curves** (grading, consolidation, compaction, flow curve, CPT and DCP traces):
- three `zoom_plot` copies read by eye;
- `PlotAxis` has values but no positions;
- vector traces, which are exact polylines in the PDF, are never read as geometry;
- no log axis is fitted.

The sounding floor on plotted sheets is 31 % (40 of 127). FUTURE_IDEAS records that plotted CPT traces swing faster than the printed grid, so readings at fixed 0.5 m steps by eye are noise. A trace read from its geometry gives the whole curve.

**Profiles and cross-sections.**
- No station or elevation axis fit, and no "12+50" station parsing.
- No separate x and y scales, so a profile with vertical exaggeration cannot be expressed.
- The importer takes one isotropic factor with no origin or datum.

**Plans.**
- Stated notes are parsed only on `drawing_sheet` pages and never applied.
- **Graphic scale bars are not detected at all**; they are rejected as dimension decoys.
- The stored `/VP` scale is used only by a switched-off review tool.
- Several scale sources on one sheet are never reconciled. This matters when a sheet was re-plotted at another size, which invalidates its stated ratio but not its bar.
- No grid-coordinate frame (northing and easting ticks).
- Spacing maths takes a factor the caller must supply.

**Design charts in references.**
- Vision only.
- No code check of the axis or the curve.
- 71 % of the pages are images (E9).

**Shared shape.**
- Four scale representations with different meanings and no common uncertainty.
- `Quantity` carries only a RELATIVE uncertainty, which cannot express "0.3 m ± 0.02 m".

**Surfaces.**
- `log_grid`, stored scales and dimension lengths are on neither agent; Document Review's geometry tools sit behind a switch that is off.
- No tool returns a measured position.

---

## 4. The primitive: a Scale

### 4.1 What a scale is

A **Scale** is a fitted map from a position on the page to a quantity, valid over a stated region of the page.

| Field | Meaning |
|---|---|
| `id` | Stable within a document: `p26.depth`, `p31.plot1.x`, `p18.plan` |
| `page`, `extent` | The page, and the box the scale governs: a log column band, a plot's frame, a viewport's box, the whole sheet |
| `quantity`, `unit` | What it measures (depth, elevation, station, offset, distance, easting, northing, percent finer, particle size, N, qc …) and the unit **as printed**. `unit` is `None` when the page does not state it; it is never guessed |
| `axis` | A direction in the displayed page frame. Normally page x or y, turned by the page's measured skew on a scan, so "down the column" means down the column as drawn |
| `transform` | `linear` or `log10`, per axis. Room is left for others (probability paper) without building them |
| `a`, `b` | `value = a + b·s` (linear) or `log10(value) = a + b·s` (log), where `s` is the position along `axis` in points |
| `anchors` | The evidence. Each anchor: a position, a value, its kind, its source, its box, its weight (below) |
| `residual` | Largest and rms distance of the anchors from the fit, in **points and in value units** |
| `anchor_rule` | How label positions were tied to the drawing (§4.3), and its own uncertainty in points |
| `plus_minus_pt` | The position uncertainty of anything read through this scale, before the snap |
| `confidence` | 0–1, from the evidence ladder (§4.4) |
| `provenance` | Method names, sources, warnings, dropped anchors |
| `needs_values` | True when positions were found but the label values have not been read yet (§4.5) |

A **Frame** pairs two Scales over one region.
- **Plot or profile:** independent x and y, each linear or log.
- **Plan:** one isotropic distance scale, optionally with a coordinate origin and rotation when the sheet prints grid coordinates.
- **Log:** a Frame with one axis.

### 4.2 Anchors: where a scale's evidence comes from

| Anchor kind | Position from | Value from |
|---|---|---|
| `tick_label` | the label's box, snapped to its tick (§4.3) | text layer · DI or OCR line · vision reading of the label's crop |
| `tick_mark` / `gridline` | a drawn short rule or gridline (vector path, or found in pixels) | the label it carries, or its place in a regular run |
| `frame_line` | the body frame of a log or a plot's border | the value the label run predicts there (0 at a log's head, 10.0 at its foot) |
| `bar_tick` | a graphic scale bar's tick (vector or pixels) | the bar's labels: 0, 5, 15 … |
| `stated_ratio` | none (a distance scale has no positional anchor) | a parsed note: "1:N", '1" = 20 ft'. Assumes true plot size |
| `stored` | the viewport box | the PDF's `/Measure` (exact, unless the identity trap) |
| `two_point` | two points the user or model names | a known distance between them |
| `log_pattern` | the minor-gridline spacings of a decade | the decade's position; its VALUE needs one label |

### 4.3 The fit, and the anchor rule

1. **Collect** candidate anchors per axis.
2. **Keep** the longest run whose positions step evenly. Positions step evenly for a linear axis, or in the log10 pattern for a log axis.

   This is `loggrid._longest_monotone` and `_regular` generalised to positions. Values are assigned by **fitted spacing, never by count** (E6).
3. **Fit** by weighted least squares.
   - Drop at most one anchor that breaks a regular run, and say which.
   - Refuse with fewer than 3 positional anchors.
   - Stated, stored and two-point scales are exact by construction and carry their own uncertainty instead.
   - The gates are `loggrid`'s, generalised: residual ≤0.2 of a step, and a monotone run ≥60 % of the band.
4. **The anchor rule** says how a label's box becomes a position, in this order:
   - a drawn tick or gridline within half a label height of the label, which beats the box;
   - frame lines that the label run predicts as round values. These fix the offset. In E3 they reveal "labels sit on their depth" (centre-to-frame offset 4.4–4.9 pt, bottom-to-frame 1.3–1.8 pt);
   - otherwise the label centre, with **half a label height added to the uncertainty** and a warning that the label's alignment could not be checked.
5. **Skew.** On a scan the page's skew is measured from its long rules (median slope) and every position is taken along the deskewed axis.
6. **Scan stretch.** A sheet fed unevenly through a scanner is not quite linear. If the residual exceeds the gate but a two-piece fit passes, the scale says so. That is not built first; on the session's sheets the label spacing varied by sd 0.13–0.31 pt.

### 4.4 Uncertainty and confidence

**Position uncertainty** (points, one standard deviation) combines in quadrature:
- the fit's rms residual;
- the anchor rule's term: 0 when snapped to ticks or frames, half a label height when assumed centred;
- the snapped thing's own term: half the line thickness, or half a pixel at the render resolution;
- for a raster, the render's pixel size.

It is converted to value units through the slope. On a log axis it is converted multiplicatively: `value·ln10·|b|·σ`.

`Quantity` gains an optional **absolute** `plus_minus`, or a small `Reading` type converts to it. A relative uncertainty cannot express "depth 0.30 m ± 0.02 m".

**Confidence ladder.** These are starting values, measured and tuned on the fixtures; a reading's confidence is the MIN of its scale's and its snap's.

| Evidence | Confidence |
|---|---|
| Stored calibrated viewport | 0.98 |
| Vector ticks and text-layer labels, tick-snapped | 0.95 |
| Pixel positions and DI/OCR or text labels, frame- or tick-anchored | 0.90 |
| Pixel positions and **vision-read** labels, regular run, frame-anchored | 0.85 |
| Any of the above with the anchor rule unresolved | −0.10 |
| Stated ratio alone (plot size assumed) | 0.70; 0.90 when a bar or a dimension agrees within 2 % |
| Labels located by a model's boxes in a zoom, snapped to ink blobs | 0.60 |
| Positions read by eye, unsnapped | not a scale reading; reported as such |

### 4.5 Label values on a scan with no text

planlens never calls a model, and that stays. So:

1. The finder returns the scale with `needs_values: true` and the label crops: their boxes, in run order.
2. **In the app**, the `measure` tool makes ONE vision side call. It sends a numbered contact sheet of the crops (`find_like`'s `like_sheets` already draws such sheets with numpy and PyMuPDF) and asks "read each numbered label". The positions stay code's.
3. **In report ingest**, the log reader's existing first call also carries the numbered crops and returns their values in a structured field, so no extra call is spent.
4. **On the MCP server or any model-less caller**, the caller passes `values=[…]` back.
5. Then the fit runs. A misread value breaks the regular run and is dropped or refused (§4.3). The values must also rise or fall one way and step evenly.

### 4.6 How scales are found, page type by page type

| Page type | Vector (text layer) | Scan + DI/OCR | Scan, no text |
|---|---|---|---|
| **Log** (depth/elevation) | today's `log_grid` ruler, plus snapping labels to drawn ticks and frame lines | DI labels; frame lines and stratum lines **from pixels** | label blobs and frame from pixels; values by §4.5 |
| **Plot** (grading, consolidation, CPT …) | text tick labels; gridlines and ticks from paths; **traces read as paths** | DI labels; gridlines, log pattern and markers from pixels | gridlines and log pattern from pixels; one decade and the end labels by §4.5 |
| **Profile / cross-section** | labelled station and elevation gridlines, separate x and y, "12+50" parsed; the ground line as a path | as plot | as plot |
| **Plan** | `/VP` and markups; note, bar (paths) and dimensions reconciled | note from DI; bar from pixels | bar from pixels, labels by §4.5; note read by vision as text |
| **Design chart** | as plot | as plot | as plot (71 % of reference figures, E9) |

**Reconciling several sources on one page** (plans especially):
- They are compared, not chosen silently.
- Agreement within 2 % raises confidence.
- On disagreement, a bar or a stored scale beats a stated ratio. Re-plotting changes paper size, and a bar shrinks with the paper.
- The result names both and the size of the disagreement.

---

## 5. Measuring a thing

### 5.1 The steps

1. **The model names the thing and gives a rough box.** For example "the line under the gravel layer", "the water-level triangle", "the 0.4 curve where φ = 35°", or "the centre of boring symbol B-3".
   - The box is a `view` + `image_box` from a look (converted from pixels in code as now), or a `bbox` in points.
   - From a whole sheet a pixel box is 1–6 pt off; from a zoom, about 1 pt.
   - Without a box the tool can still enumerate (`kind="lines"`).
2. **Code builds the search window:** the box padded by the source view's location error. This is `vision_view.location_error`, the same number `render_region` already pads by.
3. **Code snaps** to the drawn thing of the asked kind inside the window (§5.2).
   - The window is restricted to the column or plot region the box is in. A blow-count row rule is never taken for a stratum line.
   - **One candidate:** take it.
   - **More than one:** return them all with their distances from the box, and choose none (E5).
   - **None:** return the box's own position converted, marked **unsnapped**, with the view's location error as its ±.
4. **Code converts** through the scale whose extent holds the snapped position, or the scale named in the call. The result carries:
   - value ± uncertainty, in the printed unit;
   - confidence;
   - the scale's id and how it was found;
   - what the thing snapped to (kind, box, how far it moved from the given box);
   - warnings.

### 5.2 Snap kinds

| `kind` | Vector | Raster | Typical use |
|---|---|---|---|
| `line` | nearest drawn rule (`page_rules` + `join_segments`) crossing ≥55 % of the region | nearest found rule (line finder of E2) | a stratum line, a water-level line, a gridline |
| `lines` | every rule in the box | every rule in the box | all layer boundaries in a description column; all gridlines |
| `point` | centre of the path cluster or symbol inside the window | centroid of the compact ink blob; optionally the `find_like` matcher seeded by the box | a boring symbol, a plotted marker, a sample symbol |
| `edge` | top or bottom of the region's paths | top or bottom of the ink | a triangle's apex, a hatch band's top |
| `curve` + `at` | the path crossing the axis value `at`, nearest the box | the ink run crossing `at`, nearest the box; darkness or colour separates the curve from the grid | a chart read-off, a trace value, a ground line at a station |
| `distance` + `to` | two `point` snaps; the length through an isotropic scale | the same | boring spacing, a dimension check |
| `text` | the text-layer line's box (exact) | the DI line's box | a printed contact depth (its value is already printed) |

### 5.3 Refusals and honest output

- **No scale on the page:** the snapped position comes back in points with `scale_known: false`. This keeps `ir/measure.py`'s rule: points never become metres without a resolved scale.
- **A position outside the scale's extent:** refused, with the extent named.
- **A whole-page `view` wider than 300 pt with a small box:** the snap still runs but says the window is large and lists every candidate. This is the marks rule of `cf43c90`, applied to measuring.
- **The page is not drawn to scale** (a schematic pit sketch, a summary column): the fit fails its gate and no position becomes a value. The result says to use the depths the page prints.

### 5.4 Two voters where vision also answers

Where a vision answer exists for the same quantity, it is kept beside the code's reading. This applies to a chart read-off, a model's layer depths in report ingest, and a sounding the model digitised.

- They agree within the combined uncertainty: the code's value is given, with "vision agrees".
- They disagree: both are given, with the gap.

This is the report-ingest floor rule — never drop, record disagreements — applied to positions.

---

## 6. One tool surface

### 6.1 The tool

There is ONE new tool, **`measure`**, served by planlens' tool layer (`planlens/tools/specs.py`, so the MCP server gets it too). The app bridges it the way it bridges the document tools.

- **No `where`:** it lists the page's scales (inventory). Each scale comes with its kind, unit, extent, residual, confidence and how it was found, plus any scale waiting for label values.
- **With `where` and `kind`:** it measures (§5).

There is no separate "find scales" tool, no log-specific tool and no plot-specific tool.

**How it fits the existing look tools:**
- the look tools (`analyze_pdf_page`, `render_region`) say WHAT and roughly where;
- `measure` says exactly where and how much;
- it takes the same `view` + `image_box` pair as `render_region` and `annotate_document`, so the agent learns no new location language.

Nothing is added to either system prompt. The agent learns about `measure` from its own description only (owner's rule, memory `feedback-no-overfitting-to-one-example`).

### 6.2 Draft tool descriptions

**App** (`funhouse_agent/deep/tools.py`, about 1,000 characters):

> Measure a position on a page through the page's own scale: a log's depth or elevation ruler, a plot's axes (linear or log), a profile's stations and elevations, a plan's scale bar, stated scale or the scale stored in the PDF. A position read off an image by eye is approximate; this finds the drawn thing itself (a line, a symbol, a curve) and converts its position, returning the value ± its uncertainty, the scale it used and what it snapped to. Call it with only source and page to list the scales on the page. To measure, give `where` — a `bbox` in PDF points, or a look's `view` + the thing's `image_box` (from a zoom, so the box is close) — and `kind`: `line` (a drawn boundary), `lines` (every line in the box, e.g. all layer boundaries in a log's description column), `point` (a symbol or marker), `edge`, `curve` with `at` (a curve's value at an axis value), or `distance` with `to`. When two things fit the box it lists both instead of choosing; with no scale on the page it returns points and says so.

**planlens** (the same text, without the look-tool sentence). Plus: "On a scan with no text layer the label values may be needed: the result then carries `needs_values` and the label boxes, and the call is repeated with `values`."

**Parameters:**
- `source`, `page` (0-based);
- `where`: `{bbox}` or `{view, image_box}`;
- `kind`;
- `at`: `{axis: value}` for `curve`;
- `to`: a second `where` for `distance`;
- `scale`: an id from the inventory, optional;
- `values`: planlens only.

### 6.3 What a result looks like (synthetic numbers)

```json
{"page": 4, "kind": "line",
 "value": {"depth": 3.40, "plus_minus": 0.02, "unit": "m", "confidence": 0.85},
 "snapped_to": {"kind": "line", "bbox": [80.0, 400.0, 250.0, 401.0],
                "moved_pt": 2.0, "alternatives": []},
 "scale": {"id": "p4.depth", "transform": "linear", "per_point": 0.02,
           "anchors": 10, "residual_pt": 0.4,
           "anchor_rule": "labels sit on their depth (frame lines at 0 and 10 agree to 1.5 pt)",
           "values_from": "vision read of 10 numbered label crops",
           "positions_from": "pixels (scan, skew -0.5 deg corrected)"},
 "warnings": []}
```

### 6.4 Where it goes, harness by harness

**Geotech page:**
- `measure` joins `make_vision_tools`.
- `read_reference_figure` gains code's reading as a second voter (§5.4):
  1. It finds the chart's Frame on the figure page.
  2. Where several charts share a page, it takes the one nearest the catalogued caption or the side call's box of the chart.
  3. The side call is asked for the chart's axis labels, the input value, and a pixel box where the requested curve crosses it.
  4. Code snaps and converts.
  5. For a parameter between two labelled curves, it reads both and interpolates, reporting both readings.
- The tool's own description says it returns a measured value beside the vision estimate. Its `_READ_OFF_NOTE` changes to say which value rests on what.

**Document Review:**
- `measure` on the legacy build and the lean build (`deep/review_agent.py`). The minimal build is an owner decision (§10).
- Uses: a plan distance, a callout's position on a profile, a log's layer depths, a plotted value.

**Both:** the existing `precision` line on look results is unchanged. Whether it may say "a box is not a measurement" in general words is §10.

### 6.5 Report ingest uses the same functions (no tool)

| Reader | Change |
|---|---|
| **Log reader / floor** | `log_grid` gains a raster leg: ruler from pixel label blobs (values from DI/OCR text, else from the reader's first call, §4.5), stratum lines from pixels, anchor rule from ticks and frames. `seed_from_grid` then puts geometry-measured layer tops into the floor, so the existing merge rule keeps them unless the model brings evidence. The depth gate works on scans. Disagreements stay QA entries. |
| **Lab reader** | A grading, compaction or flow-curve plot is measured and set against the floor's tabulated values: a new QA kind, plot against table. It is a cross-check, not a reading. The table stays the record (E6). `zoom_plot`'s result also carries code-read points. |
| **Sounding reader** | `PlotAxis` becomes a Frame with positions. A plotted trace is digitised by code: exactly from paths on a vector sheet, by column scan on a raster. The model's job becomes saying which trace is which channel and settling crossings. The four gates are unchanged. The floor gains a series where it had only ranges. |
| **Calc reader** | `zoom_plot` results carry code readings where a frame fits. |

The three `zoom_plot` copies become one helper over the primitive.

---

## 7. Fixtures and measurement

Every fixture is synthetic, public and committed in `planlens/testing/`, with truth computed from its inputs, never through the code under test (the `scale_fixtures` convention).

| Fixture | Built from | Variants | Truth |
|---|---|---|---|
| **Scanned-looking log** | `loggrid_fixtures` forms, rendered through `show_pdf_page(rotate=θ)` at a small angle, rasterised at 150–300 dpi, noise, blur and JPEG 60–80 added, embedded as the one image of an image-only page (also on a `/Rotate 270` page, as the real scans are) | label anchored centre / baseline / top; tick marks drawn or not; frame-labelled or not; ruler left or inside; skew −1° to +1°; dashed contact lines; a hatched legend column; a continuation sheet starting at 10 | every contact, label value, frame depth |
| **Plot** | drawn with PyMuPDF | grading (log x, 0–100 y, gridlines, markers + curve); linear with ticks only; reversed y (depth down); log y; two charts on one page; vector and rasterised; a gridline broken by a title (the E6 trap) | every marker, curve function |
| **Plan** | extends `scale_fixtures` | stated note; graphic bar (blocks, and ticks only); stored `/VP`; northing/easting grid ticks; **re-plotted at 50 %** (note wrong, bar right); symbols at known ground coordinates | distances, coordinates, which source must win |
| **Profile** | drawn | station labels "0+00…5+00", elevation gridlines, 5:1 vertical exaggeration, ground polyline, a boring stick; vector and rasterised | ground elevation at stations; contact elevations |
| **Chart family** | drawn from known functions | curves labelled by a parameter on semi-log and log-log axes; raster | read-offs at given inputs, interpolated |

**The harness** (`module_work/scales_harness/`, no model, no network):
- For every fixture and variant, it reports:
  - scale found or not;
  - the residual;
  - the measured value's error against truth;
  - whether the truth lies inside the reported ± (calibration of the uncertainty).
- Rough boxes are drawn from the measured error distributions:
  - pixel boxes from a whole sheet, 1–6 pt;
  - zooms, about 1 pt;
  - old grid boxes, 57–91 pt, for the ambiguity check.

  It reports the snap success and how often the result admitted ambiguity (E5).
- **The gate before shipping:**
  - on the fixtures, ≥95 % of readings carry truth inside their ±;
  - no reading with a wrong snap is returned without its alternatives.

**Free truth from real public documents:**
- Vector lab sheets that print both a table and a plot: the table is the truth for the plot.
- Public charts with a printed equation.

**Private spot check** (local, counts and error sizes only):
- A script that takes the private PDF path and a truth-file path as arguments and contains no private content.
- On the session's report:
  - the ten scanned log sheets: stratum depths against the contacts measured 2026-10-07 (§11.7: that truth should be saved as `raw/scales_truth.json`);
  - the four vector log sheets: ruler against drawn ticks (E3);
  - the twelve scanned grading sheets: plot against the printed table, one sheet's table truth a minute by hand.
- Then **the corpus** on the cluster (report ingest's 38 private reports): how many scanned log pages get a raster ruler and stratum lines, counts only. This guards against fitting to one form (owner's rule).

**Live checks** go on `module_work/LIVE_TEST_QUEUE.md` (owner's rule for small confirmations):
1. Vision reads numbered label crops on Funhouse and Foundry.
2. An agent turn on a synthetic report with scanned logs uses `measure` from its description alone.
3. A Document Review suite task with a plan-distance question and a log-depth question. These are suite TASKS, not prompt rules.

Every run is read in full (CLAUDE.md).

---

## 8. Build order

Small steps. Each lands with tests and a harness run before the next starts. **P** = planlens, **A** = app.

| Step | Where | What | Measured by |
|---|---|---|---|
| 0 | P (+ A harness) | The fixtures of §7 and the no-model harness. The scratch experiments of §2 become its first checks. | fixtures build; truth round-trips |
| 1 | P | `planlens/document/scales.py`: Scale, Frame, Anchor, Reading; linear and log fits, regular runs, residual, uncertainty, confidence; adapters from `Ruler`, `Viewport`, stated notes and two-point. Data only; no behaviour change anywhere. | unit tests on numbers |
| 2 | P | `planlens/document/raster.py`, numpy + PyMuPDF only: `_ink` moved out of `findlike`, skew, deskew transform, line finder, blobs, compact markers, curve crossing. Megapixel cap as `findlike`'s; regions for large sheets. | fixture lines found ±0.5 px; the 10 private sheets (E2 counts) |
| 3 | P | `find_scales(doc, page)`: log rulers (vector and raster, anchor rule), plot frames (gridline runs, log pattern, text ticks), plan scales (notes on any page kind near a bar, **scale-bar finder** in vector and raster, `/VP`, reconciliation), profile grids. `needs_values` where labels have no text. | harness: scale found, residual, anchor rule on all variants |
| 4 | P | `log_grid` raster leg: stratum lines from pixels on scans (with or without DI); ruler from step 3; layers with ±. Vector logs also snap labels to ticks. | log fixtures; private 10 + 4 sheets; then corpus counts on the cluster |
| 5 | P | `measure(doc, page, where, kind, …)`: snap + convert + refusals; the `measure` spec in `planlens.tools` and MCP; `Quantity` absolute ±. | harness snap success and ± calibration (§7 gate) |
| 6 | A | The `measure` tool on the geotech page and on the Document Review legacy and lean builds: `view` + `image_box` conversion via `vision_view`, the label-reading side call over a numbered crop sheet, feature-detected on the installed planlens (`document_tools.has_tool`). | offline tests with fake engines; tool-schema size check |
| 7 | A | `read_reference_figure` second voter. | public-chart fixtures; public charts with printed equations |
| 8 | A | Report ingest: log floor on scans, lab plot-against-table QA, sounding traces from code, one `zoom_plot` helper. | `score_on_cluster` before and after on hand truth; scorers unchanged |
| 9 | A | Live checks from §7, each run read in full; release. | LIVE_TEST_QUEUE |

**planlens versus the app.**
- **planlens** holds everything that needs no model: the primitive, raster analysis, finders, snap, measure, the tool spec and the fixtures.
- **The app** holds everything that needs a model or the agents: the label-reading side call, the tool bridge, `read_reference_figure`, report ingest, live checks.

This is the split `find_like` already uses.

**Releases.**
- planlens publishes first.
- The app pin rises.
- Steps 1–5 can ship as one planlens minor release with no app change.
- Steps 6–8 ship in one app release, or two if report ingest's numbers need their own cluster run.

---

## 9. Risks

- **Over-fitting to one form.**
  - The session's scanned form is one template.
  - The finders must not assume a column's place, a label's alignment or a line's length beyond what each page shows. E2 already tripped on the pit form.
  - Measure on the varied fixtures and on the corpus before shipping.
- **Hatch patterns, legends and grids.**
  - A legend column's hatching or a table's row rules can look like stratum lines.
  - Snapping is restricted to the column or region the box is in, and `lines` reports lines with their extents so a short hatch run is visible as one.
- **Dashed or faint contacts.**
  - Gradational or inferred contacts are often dashed.
  - The line finder closes small gaps. A dashed line should come back labelled `dashed`, because that is information, not noise.
- **Mis-assigned labels.** A dropped or extra gridline shifts every value (E6). Regular-run assignment and the residual gate catch it, and the tests pin the trap.
- **A misread label value.** Caught by the regular-step and monotone tests. Refused rather than fitted around when two disagree.
- **False precision.**
  - A code value looks exact.
  - Every reading carries its ± and its provenance.
  - Vision-read labels, an unresolved anchor rule and unsnapped positions all lower the confidence and say so.
  - Reported precision follows the uncertainty, not the float.
- **Pages not drawn to scale.** Pit sketches, summary columns and schematic profiles. The gate refuses them, and the result points to the printed depths.
- **Large sheets.** A D-size sheet at 200 dpi is about 30 MP. Use `findlike`'s cap and analyse regions, not whole sheets, when the box is known.
- **Hosts.**
  - Only numpy and PyMuPDF.
  - scipy is not a planlens dependency and is not needed.
  - PIL is not needed either: PyMuPDF renders straight to numpy, as `findlike` does.
- **Agent misuse.**
  - Passing a whole-sheet box returns a list of alternatives rather than a confident wrong value.
  - The review suite's tasks measure whether agents use the tool well; prompts are not changed to force it.
- **Cost.** One small side call per scale needing values. Report ingest adds no call (§4.5).

---

## 10. For the owner to decide

**DECIDED 2026-10-08: the owner accepted the lead's recommendations on all nine.**
1. No `measure` on the minimal build.
2. `log_grid` becomes an agent tool on both pages, measured on the suite.
3. `measure` makes its own small vision call to read label values on a scan.
4. On a disagreement, code's value is given with the vision value beside it. A gap over about 3× code's uncertainty is flagged.
5. Values are given with their ±, never finer than the source's printed resolution.
6. In report ingest, geometry layer tops enter as a voter first; they become the default only after a cluster run shows they help.
7. The precision line may say, in general words, that a box read off an image is not a measurement.
8. Re-verify GEC-12 Fig 7-15 (HANDOFF 8a(v)).
9. Live checks use the synthetic fixtures plus public sets already in use, unless the owner names others.

The original questions follow.

1. **The tool's name and reach.** `measure` on the geotech page and the Document Review legacy and lean builds. Should it also go on the **minimal** build, which keeps its tool list deliberately short?
2. **Whether `log_grid` itself becomes an agent tool on both pages.** `measure` with `kind="lines"` covers layer depths without it. `log_grid` adds the columns, rows and header fields; in the session it would have given the vector 2026 logs' rows exactly.
3. **The label-reading side call.** Should `measure` make one small vision call on its own when a scan has no text (recommended)? Or should the agent always be asked to supply the values?
4. **When code and vision disagree** on a chart read-off or a layer depth: give the code's value with the vision value beside it (recommended), and at what gap to call it a disagreement.
5. **Reported precision.** Round to the uncertainty (e.g. 3.40 ± 0.02 m), or to the log's own printed resolution (e.g. 0.05 m)?
6. **Report ingest defaults.** Do geometry-measured layer tops enter the log floor by default? This changes the reader's numbers and needs a cluster run. Is code digitising of plotted soundings on by default, or a voter first?
7. **The precision line on look results.** May it say, in general words that name no tool, "a box read off an image is not a measurement"?
8. **E8, outside W3.** GEC-12 Figure 7-15's hand digitisation differs from a code read by −19 % to +6 %. Re-verify and, if confirmed, correct it, since `axial_pile` uses it?
9. **Public test documents for the live checks:** public boring logs and agency charts the owner is content to use.

---

## 11. Corrections to the plan of record

Against `SCALES_COVERAGE_CROSSCHECKS.md` §W3, "Pieces that exist":

1. **"`log_grid`: the depth ruler and stratum lines, on vector pages and on scans with OCR lines."**
   - On scans, even with OCR, `log_grid` finds the ruler from the OCR'd labels, but **no stratum lines**. It reads rules only from vector drawings (`page_rules` → `get_cdrawings`), and a scan has none.
   - Layers then come only from printed contact ticks and USCS-symbol changes.
   - Its columns come from header text alone.
   - On scans with no text it finds nothing at all (E1).
2. **The ruler assumes centred labels.** It fits the y centres of the labels. That is right on the vector form (0.18 pt) and about 0.1 m off on the scanned form (E3), and the fit's residual cannot show it.
3. **`log_grid` is not reachable by either harness.** It is used by report ingest and offered by planlens' own toolkit and MCP. The app's document tool list leaves it out. Neither do the stored scales or dimension lengths reach the geotech page; on the review page they sit behind a switch that is off.
4. **"Sounding reader: axis ranges as gates."** Correct, but the ranges are tick VALUES with no positions. They are gates, not a scale, and cannot convert a position.
5. **"Viewport scales; page-map scale notes; the cross-section importer's calibration."**
   - Scale notes are parsed only on `drawing_sheet` pages and never applied.
   - The importer's calibration is one isotropic factor with no origin or datum, so it cannot express a profile's station and elevation, let alone vertical exaggeration.
   - Graphic scale bars are not found at all; the dimension finder rejects them as decoys.
6. **"Built from printed labels and ticks (text layer, or OCR on scans)."** The no-OCR path is the common one:
   - on the government hosts the only OCR is DI, which is optional and paid;
   - planlens' free RapidOCR leg imports OpenCV and cannot load there (`rapidocr_onnxruntime` imports `cv2`; planlens' `ocr.py` loads OpenCV through the guarded child-process probe, so it fails cleanly);
   - 71 % of the reference figures are images without numbers in any text layer (E9).

   The plan should add three sources of evidence:
   - label positions from pixels with values read by vision;
   - frame lines and gridlines as anchors;
   - the log-gridline pattern, which finds a log axis's decades from geometry alone.

   This also qualifies the standing memory "prefer the free RapidOCR path": it is not available where the app runs.
7. **"The session's scanned log sheets … (four sheets, contacts measured 2026-10-07)."** The measured contacts are not saved in the repository or in `raw/`. This design re-measured all ten scanned sheets (E2, E4). The lead's four-sheet truth should be saved as `raw/scales_truth.json` so the spot check has a fixed reference.

---

## Appendix: how the experiments were done

- **Rendering.** PyMuPDF `get_pixmap` in grey at 200 dpi, in the displayed frame (rotation applied). Ink was an Otsu threshold.
- **Lines.**
  1. Row runs of ink, with gaps up to 4 px closed and a ±1 row band for skew.
  2. Runs grouped across rows.
  3. Each line's y as a straight fit through per-column ink centroids.
  4. Skew as the median slope of the long rules.
  5. The image deskewed by two shears.
  6. Collinear pieces joined.
  7. Column rules from the column ink fraction over the body.
- **Labels.** Row clusters of ink inside the depth column band, under 12 pt tall.
- **Markers.** Pixels whose 5 × 5 neighbourhood is all ink (lines vanish, filled squares remain), clustered.
- **Log axis.** The 9 spacings of a decade compared with log10(k+1) − log10(k). Decades walked both ways.
- **Plan bar.** Vertical path segments under the bar's numeric labels.
- **Chart curve.** Pixels darker than the grey gridlines, in the column at the asked axis value.
- **Snap simulation.** 200 normal draws per true stratum line, nearest candidate crossing the description column.
- **Cost.** All of it ran in 0.4–0.8 s a page on the owner's laptop.
