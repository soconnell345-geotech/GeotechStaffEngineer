# DR-11: recompute a report's calculations (plan)

BACKLOG DR-11: from an ingested report, take each calculation the report
claims (its inputs and its result), re-run it through the app's own modules,
and return a claimed-against-recomputed table.

## What the records hold

Report ingest (model first by default) writes one `Calculation` per
calculation printout into `ReportRecord.calculations`
(`report_ingest/model.py`; read by `report_ingest/calc_reader.py` and, model
first, `report_ingest/model_first_read.py` stage `calc` with the same
`CalcReading` schema). Each holds:

* `kind`, one of `CALC_KINDS` (13 values, below);
* `program`, `method`, `subject`, `summary` as printed;
* `inputs` and `results`: lists of `NamedQuantity` -- the label AS PRINTED,
  then either a `Quantity` (number and printed unit, with its page) or the
  printed words (`"OK"`, `"Site Class D"`);
* `pages` (0-based PDF pages) and `prov`.

So the claimed inputs are free labels, not module parameters. The recompute
needs a label vocabulary per kind (which printed labels name which module
input) and a unit change through the record's own conversion
(`report_ingest.model.calc_si`, which already reads ft, in, psf, ksf, tsf,
pcf, ksi, kips, kips/ft, deg, g as gravity, ...). Nothing new is invented:
the recompute calls the module through the agent's own route
(`funhouse_agent.dispatch.call_agent`), so the result carries the module's
`choices` and `defaults_used` exactly as a tool call on the page does.

## Calc kinds and what recomputes them

| Kind | Module and method | First slice? |
|---|---|---|
| `shallow_foundation_bearing` | `bearing_capacity.bearing_capacity_analysis` | **yes** |
| `settlement` | `settlement.elastic_settlement` (elastic), `settlement.consolidation_settlement` (one layer, Cc/Cr) | **yes** (elastic and one-layer consolidation) |
| `settlement` (Schmertmann, Hough, several layers, combined) | `settlement.schmertmann_settlement`, `hough_settlement`, `consolidation_settlement` with a layer list, `combined_settlement_analysis` | later: the layer table has to come off the printout as rows, which the record's flat label list does not keep |
| `axial_pile` | `axial_pile` (driven), `drilled_shaft` | later: a layered profile and a pile section |
| `lateral_pile` | `lateral_pile` (COM624P p-y) | later: a layered p-y profile and a section |
| `pile_group` | `pile_group` | later: the group layout and each pile's stiffness |
| `retaining_wall` | `retaining_walls` (cantilever, MSE), `sheet_pile`, `soe`, `seismic_geotech` (M-O) | later: the earth-pressure coefficients (Rankine/Coulomb Ka, Kp) are the first candidate, then whole walls |
| `liquefaction` | `liquefaction` tool (B&I 2014 / NCEER 2001) | later: a layer table of N or qc |
| `site_response` | `seismic_geotech` site class; `pystrata_agent` | later: site class from a printed Vs30 or N-bar is the first candidate |
| `pavement` | `pavement_design` (AASHTO 1993, UFC 3-250-01) | later: SN and slab D map well from labels |
| `ground_improvement` | `ground_improvement` (GEC-13) | later |
| `seepage` | none in a form a printout maps onto (fem2d seepage needs a full mesh geometry) | none in the app |
| `slope_stability` | `slope_stability` | **held**: no slope_stability work until the owner's integral review (GM-140); rows say "not recomputed yet" |
| `other` | nothing: the reader found no kind | none |

The first slice takes bearing capacity and settlement: they have an exact
module match, their claimed inputs are single labelled values, and they are
two of the commonest kinds in the calc truth (the plan's WP5 notes:
settlement, pile capacity, seismic, slope, pavement, retaining walls). Pile
capacity is the obvious next slice once the profile can be read as rows.

## Inputs: direct, unit change, often missing

**Bearing capacity** (`bearing_capacity_analysis`, SI inputs):

| Module input | Printed labels (lower case, as matched) | Unit | Often missing |
|---|---|---|---|
| `width` | width, B | length -> m | rarely |
| `length` | length, L | length -> m | yes (a strip prints none) |
| `depth` | embedment, footing depth, Df | length -> m | sometimes |
| `friction_angle` | friction angle, phi | deg (a blank unit read as degrees, and said so) | for clay |
| `cohesion` | cohesion, c, undrained shear strength, su, cu | stress -> kPa | for sand |
| `unit_weight` | unit weight, gamma (not effective, buoyant, water) | -> kN/m3 | sometimes |
| `gwt_depth` | water table, groundwater, GWT | length -> m | often |
| `factor_of_safety` | factor of safety, FS, FOS | none | sometimes |
| shape | the words strip / continuous / square / circular / rectangular anywhere in the labels, text values, subject, method or summary; else L = B (square) or L > B (rectangular) | - | often |

Claimed results: `ultimate` -> `q_ultimate_kPa`; `allowable` with `net` ->
`q_allowable_net_kPa`; `allowable` without `net` -> `q_allowable_kPa` (the
gross allowable, UFC 3-220-20 Eq. 5-13, which the row says).

**Settlement, elastic** (`elastic_settlement`): `q_net` (applied / net /
bearing pressure, q), `B` (width), `Es` (modulus, Es, Young's), `nu`
(Poisson), `Iw` (influence factor), `L`, shape, rigidity (rigid / flexible
words). **One-layer consolidation** (`consolidation_settlement`):
`thickness`, `depth_to_center`, `e0` (void ratio), `Cc` (compression index),
`Cr` (recompression / swelling index), `sigma_v0` (initial / existing
effective stress, overburden), `sigma_p` (preconsolidation, maximum past),
`delta_sigma` (stress increase, added stress). Claimed result: the one
printed settlement (a label with "settlement"), preferring the one that
names the method (immediate / elastic, consolidation / primary); more than
one left after that is not compared, and the row names them.

## A missing input is reported, never guessed

Each recipe lists the inputs that must come off the printout (bearing:
width, depth, unit weight, shape, and friction angle or cohesion; elastic:
q, B, Es, and shape or Iw; consolidation: all eight but sigma_p). One that is
not printed, or printed twice with different values, makes the row
`not_recomputed` with `inputs_missing` naming it. Nothing is filled in.

An optional input the module itself documents a case for (bearing: no
water table, Vesic depth factors; elastic: nu = 0.3, GEC-6 Table 5-12's Iw;
consolidation: no sigma_p = normally consolidated) is left to the module,
and the row carries the module's own `defaults_used` under `assumed`, so the
reviewer sees what the report did not print. An allowable pressure with no
factor of safety printed is not compared (the module's 3.0 would be a
guess); the ultimate still is.

## The method choice

The printed `method` and `program` are read for a method the module offers
(bearing: Vesic, Meyerhof, Terzaghi; Hansen is the N-gamma option; elastic
vs consolidation for settlement). A named method is used and said
("printed: Meyerhof"). Where none is printed:

* bearing: the module's own default (Vesic) is the recomputed value, and
  the module's other cited options (Meyerhof, Terzaghi) are run too and
  listed in `alternatives` with their values, so a claim that matches one of
  them shows it;
* settlement: the method follows from the inputs printed (a modulus and no
  compression index: elastic; a compression index and no modulus:
  consolidation), and the row says so; both or neither: not recomputed.

A method the report names that the first slice does not build (Schmertmann,
Hough, a two-layer bearing case) is `not_recomputed` with the reason.

## The table, one row per claimed result

| Field | Holds |
|---|---|
| `calc_index`, `kind`, `subject`, `pages` | which calculation (pages 1-based, as a reader cites them) |
| `quantity`, `claimed` | the printed label; the value and unit as printed, with its page |
| `claimed_si`, `recomputed_si`, `unit_si` | both in SI |
| `difference_pct` | (recomputed - claimed) / claimed x 100 |
| `flag` | `differs` when the difference is over the tolerance, else `agrees` |
| `module`, `module_method` | what ran |
| `method_used`, `method_basis` | the method, and printed / module default / inferred from the inputs, with the module's own basis text |
| `alternatives` | the other cited options and what each gives |
| `inputs_used` | each input with its printed label, value, unit and page |
| `inputs_missing` | what the printout did not give (not recomputed) |
| `assumed` | the module's `defaults_used` for the run |
| `status`, `reason` | `recomputed`, `not_compared` or `not_recomputed`, and why |

Tolerance: 10 % by default (`TOLERANCE`, a QC threshold of this table, not
an engineering value; a printout's chart-read factors and rounding differ by
a few per cent), and a caller may pass another.

## Where it appears

1. **Package function**: `report_ingest.recompute.recompute_record(record,
   tolerance=...)` (a `ReportRecord`, its dict, or the path of
   `report.record.json` or its folder) -> `{"rows": [...], "counts": ...,
   "tolerance": ...}`; `recompute_calculation(calc)` for one; `table_markdown`
   for the summary.
2. **Tool on the geotech page**: `recompute_report_calcs(record)`, offered
   with the `report_ingest` tool (behind `GEOTECH_REPORT_INGEST`, as the
   ingest is; never on the Document Review page). The page's default build
   is unchanged, so the page fingerprints do not move.
3. **Later**: a "Recompute" sheet in the ingest's Excel output. Today the
   ingest writes Excel only for the owner's question list
   (`report.questions.xlsx`, when `GSE_REPORT_QUESTIONS` names one); once the
   owner says that is where results are read, the table goes there as a
   second sheet, and a `report.recompute.json` beside the record.
