# GM-237: driven piles, the alpha method's loose ends from GM-229

Page numbers are the printed pages of the manuals in `geotech-references/docs/`:
GEC-12 Vol I (FHWA-NHI-16-009), NHI-06-089 (Soils and Foundations Vol II),
UFC 3-220-20 (2025), GEC-10 (FHWA-NHI-18-024).

## 1. A clay layer's site-specific adhesion (built)

**What the print says**

- GEC-12 Eq. 7-10 (p. 245): fs = Ca = alpha su.
- GEC-12 p. 246: "All users should confirm the applicability of a selected
  design chart in a given soil condition with local correlations between
  static resistance calculations and static load tests results."
- NHI-06-089 Example 9-3 (pp. 9-58/9-59): "Site specific tests have indicated
  that the adhesion may be assumed equal to cohesion"; the solution takes
  ca1 = c1 and ca2 = c2 and reads no chart.

**What changed**

- `AxialSoilLayer` takes `alpha` (Ca = alpha su) or `adhesion` (Ca in kPa),
  with `adhesion_basis`, where the value came from. The value is used in place
  of the chart's (Fig. 7-17, Fig. 7-18, the smaller of the two, and Eq. 7-90's
  Fig. 7-17 Ca) for that layer; the shrink-swell zone (p. 247) still applies.
- Refused: a value without its basis, a basis without a value, both `alpha`
  and `adhesion`, a value of 0 or less, either on a cohesionless layer, and
  either on a `method="beta"` run, which would not use it.
- `choices` gains "site-specific adhesion (alpha method)" (how: given), naming
  each layer, its alpha and its basis, citing Eq. 7-10, p. 246 and Example 9-3.
  Where every clay layer gives its own value, the chart choices (the adhesion
  chart, Fig. 7-17's curves, D for Fig. 7-17, the Fig. 7-18 stratigraphy
  case) are left out, because no chart was read, and `defaults_used` no
  longer lists `adhesion_chart`. Where only some do, the Fig. 7-18
  stratigraphy case lists only the layers that read a chart.
- A layer with its own value needs no highly overconsolidated line, even where
  its su is past the chart.
- The calc package's method step says "alpha = 1.00 given (basis)" for such a
  layer, and names only the charts the run read.
- Adapter: the layer keys `alpha`, `adhesion`, `adhesion_basis`,
  `highly_overconsolidated` reach the module; the layers description says
  what they do.

## 2. A clay layer's own "highly overconsolidated" (a flag built; OCR not built)

**What the print says**

- GEC-12 p. 247: "In highly overconsolidated clays, undrained shear strengths
  may exceed the upper limits of Figures 7-17 and 7-18. In these cases, it is
  recommended that adhesion factor, alpha, be calculated according to API
  Recommended Practice 2A (1993)." Step 1 (p. 250) says the same.
- GEC-12 gives no test, in su / sigma'vo, OCR or anything else, for "highly
  overconsolidated".

**The flag (built).** The condition GEC-12 prints is the clay being highly
overconsolidated, so a per-layer true/false is that condition as the engineer
judges it. `AxialSoilLayer.highly_overconsolidated`: true takes API alpha
where su is past the chart, false holds Ca at the curve's end, and it wins over
`highly_overconsolidated_su_ratio` for that layer. A run whose every layer past
the chart gives the flag needs no line. It is refused on a cohesionless layer,
on a beta run, and with `adhesion_beyond_chart` "chart_end" or "api", where it
would change nothing. `choices` ("highly overconsolidated clay") states each
run of layers and whether the call was the line's or the layer's; the method
step says "(given)".

**OCR (not built).** No source in the library defines how an OCR decides
GEC-12's rule:

- GEC-12 uses no OCR in the alpha method or the API method (Eqs. 7-13 to
  7-15, p. 251, are in su / sigma'vo).
- UFC 3-220-20 Sec. 6-5.4.2 (pp. 419-420) restates API's alpha in OCR
  (Eqs. 6-18/6-19, alpha capped at 1.0 below OCR 1.2). That is a different
  formula for alpha (footnote 36: derived through DM 7.1 Eq. 8-20 with m =
  0.8), not a test for when a clay is highly overconsolidated.
- The OCR numbers in the library come from other uses: NHI-06-089 Ch. 10
  (Eq. 10-3, heavily overconsolidated OCR > 4, for K0) and GEC-10 p. 10-7
  (very heavily overconsolidated OCR > 8, drained against undrained drilled
  shafts).

An OCR input would need either a threshold or UFC's alpha formula placed into
GEC-12's procedure, and the print supports neither. So there is no OCR input.

## 3. Alpha 1.07 for the firm lower clay of Example 9-3 (checked; Fig. 7-18 fixed)

**What the module picks.** By default (`adhesion_chart` "fig_7_17") the lower
clay (su 1.1 ksf) is read on Fig. 7-17's concrete, timber and corrugated
curves at D = 42b, D measured from the ground surface to the toe (the layer
runs past the toe). That puts it on the D = 40b curve, Ca of about 1.18 ksf,
alpha 1.074.

**Against the print.** Fig. 7-17 (p. 248) has no stratigraphy case: its curves
are by pile type and D/b, and its legend reads "D = Distance from Ground
Surface to Bottom of Clay Layer". P. 246 says D is "the minimum value of the
length from the ground surface to the bottom of the clay layer, or the length
from the ground surface to the pile toe". The figure's D = 40b solid curve
starts at su 1.0 ksf with Ca 1.1 ksf, which is above Ca = su, and the alpha
method's steps set no limit on Ca (GM-229, OWN-63). **So the Fig. 7-17 reading,
and its 1.07, agree with the print.** Nothing changed there.

**Soft clay over firmer clay is a Fig. 7-18 case, and there the module
disagreed with the print.** P. 246 says the middle graph of Fig. 7-18 "should
be used ... when piles are driven through a soft clay layer overlying a stiff
clay layer", and p. 247 says the bottom graph is "for piles driven in stiff
clays without any different overlying strata". With `adhesion_chart`
"fig_7_18" or "smaller", the module treated stacked clay layers as one stratum
from the ground surface and put a clay lying under a softer clay on the
bottom graph. That case excludes an overlying clay of different strength.
Whether the middle graph applies depends on the clay above being soft and the
one below stiff, and GEC-12 prints no su that separates the two.

**Fix.** Where a clay layer of the profile's stratum lies directly under a clay
of lower su and gives no `fig_7_18_case`, a run that reads Fig. 7-18 for it
compares the profile's graph with the middle graph (D from the layer's own
top). Where the two give different Ca, the run is refused (`JudgmentNeeded`),
naming each such layer, the depth its clay stratum starts at, and both values
with the chart each came from (under "smaller" either may be Fig. 7-17's).
The message asks for the layer's `fig_7_18_case`: 'soft_clay_over_stiff_clay'
(D from the layer's top) or 'none' (Fig. 7-17). Where the clay above is not a
different stratum, it asks for the profile's case ('stiff_clay', or
'sand_over_stiff_clay' under a sand) on the TOP clay layer of the stratum, so
D runs from the stratum's top; a case given on the lower layer starts its
stratum there (the GM-144 rule), so 'stiff_clay' there would read the bottom
graph with D from that layer's top. The last option is `adhesion_chart`
"fig_7_17". This follows the GM-229 rule: a judgment the manual leaves is
asked for only where it changes the answer. A firmer clay over a softer one
keeps the profile's case. A given case wins, as before. Under Eq. 7-90 the
shaft is on Fig. 7-17, so nothing is asked.

Example (made-up values: 0.4 m closed pipe, L = 20 m, clay su 30 kPa over
su 90 kPa, `adhesion_chart` "fig_7_18"): refused with Ca 84.7 kPa by the
bottom graph and 60.9 kPa by the middle graph. With 'stiff_clay' on the top
clay layer, the lower clay reads the bottom graph at D = 50b, alpha 0.941,
Q_ult 1674.5 kN, the same as before GM-237; with 'stiff_clay' on the lower
layer it would read D = 30b, alpha 0.872, Q_ult 1580.0 kN.

**Noted, not fixed (it would change saved answers):** p. 246 recommends
Fig. 7-17 "generally ... unless one of the special soil stratigraphy cases
identified in Figure 7-18 is present at a site". The default stays Fig. 7-17
even where the profile shows sand over clay or soft clay over stiff clay; the
`adhesion_chart` choice quotes p. 246. Switching the default in those profiles
is the owner's call.

## 4. API RP 2A

API RP 2A is not in the library. No API limit or definition was added. API
alpha stays as GEC-12 prints it (Eqs. 7-13 to 7-15, alpha at most 1.0 per
p. 252, on the API route only).

## 5. WE-SF-2 (NHI-06-089 Example 9-3), like for like

With each clay layer given the example's alpha = 1.0 from its site tests:

| | Printed (p. 9-59) | Module | |
|---|---|---|---|
| Upper clay, 10 ft | 10 tons = 89.0 kN | 89.0 kN | 0.0 % |
| Lower clay, 32 ft | 2.2 tons/ft x 32 ft = 70.4 tons = 626.3 kN | 626.3 kN | 0.0 % |
| Shaft | 80 tons (rounded) = 711.7 kN; 80.4 tons = 715.2 kN unrounded | 715.3 kN | +0.5 % against 80 tons; +0.01 % against 80.4 tons |
| Toe | neglected | 44.0 kN (9 su At, reported separately) | |

The other readings of the same pile, for the record:

| Adhesion basis | Lower clay alpha | Shaft |
|---|---|---|
| site alpha 1.0 (the example's) | 1.000 | 715.3 kN |
| Fig. 7-17 as drawn (default) | 1.074 | 761.9 kN (+7.1 %) |
| Fig. 7-18 middle graph, `fig_7_18_case` given | 0.732 | 546.1 kN (-23.3 %) |
| `adhesion_chart` "fig_7_18" or "smaller", no case | refused: asks for the lower clay's case | |

`funhouse_agent/worked_examples.json` WE-SF-2 now runs with the layers'
`alpha` = 1.0 and `adhesion_basis`. Its computed result, comparison,
reconciliation and report notes are rewritten to match.

## 6. Saved answers that change

- **Default runs** (`adhesion_chart` "fig_7_17", no new layer inputs): none.
- **WE-SF-2:** shaft 761.9 -> 715.3 kN, Q_ultimate 806.0 -> 759.3 kN,
  Q_allowable 403.0 -> 379.7 kN. The pile and the method are unchanged; the
  example's own site-specific basis is now given.
- **Runs with `adhesion_chart` "fig_7_18" or "smaller"** that have a clay
  directly under a clay of lower su and no `fig_7_18_case`: refused where the
  middle graph would give another Ca (before: the bottom or top graph, D from
  the top of the stacked clays). One existing test (`test_gm144.py`,
  bottom-graph D) was changed to a firmer clay over a softer one, so it still
  checks D from the ground surface.

## 7. Tests

- `axial_pile/tests/test_gm237.py` (new): site alpha and Ca in place of the
  chart, choices and the method step, no line needed, the shrink-swell zone,
  the refusals, the layer flag (API alpha or the curve's end, and winning over
  the line), Fig. 7-17's D from the ground surface for soft over firm, the
  Fig. 7-18 refusal and a given case. After review: the refusal names the
  stratum's top and says to give the profile's case on its top clay layer
  (which reproduces the profile's D), it names the chart each Ca came from
  under "smaller", and a layer with its own alpha is not listed under a
  Fig. 7-18 case (no such choice where every layer gives its own).
- `funhouse_agent/tests/test_axial_pile_gm237.py` (new): the same through
  `axial_pile_capacity`.
- Run: `pytest axial_pile`, `funhouse_agent/tests/test_axial_pile_gm237.py`,
  `funhouse_agent/tests/test_defaults_used.py -k axial`,
  `funhouse_agent/tests/test_worked_examples.py` (WE-SF-2),
  `webapp/tests/test_page_fingerprints.py` and
  `webapp/tests/test_public_text.py`.

## Release note

- Driven piles: a clay layer takes a site-specific `alpha` or `adhesion`, with
  its `adhesion_basis`, in place of the alpha method's chart. NHI-06-089
  Example 9-3 (WE-SF-2) now matches like for like: shaft 715.3 kN against the
  printed 80.4 tons.
- A clay layer's `highly_overconsolidated` (true or false) decides API alpha
  past the chart for that layer, in place of the su / sigma'vo line. There is
  no OCR input: no manual in the library defines one for this rule.
- With `adhesion_chart` "fig_7_18" or "smaller", a clay under a softer clay is
  no longer read on the bottom graph unasked: where the middle graph would
  give another Ca, the run asks for the case, either on that layer
  ('soft_clay_over_stiff_clay' or 'none') or the profile's case on the
  stratum's top clay layer. The default Fig. 7-17 answers are unchanged.
