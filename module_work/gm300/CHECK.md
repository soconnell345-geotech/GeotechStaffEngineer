# wave_equation: drivability quakes and damping (GM-300), embedded length and shaft distribution (GM-301)

BACKLOG GM-300 and GM-301, 2026-10-31, branch `wip/gm300` from master
001526b. Both were filed by GM-294 (`module_work/gm294/CHECK.md`, "Noted,
not fixed"). Read from GEC-12 Vol 2 (FHWA-NHI-16-010, 2016) in
`geotech-references/docs/`, each page rendered and read; page numbers are
the printed ones.

## GM-300: the drivability tool's quakes and damping (fixed)

- **Before.** `drivability_study` took `quake_side`, `quake_toe`,
  `damping_side` and `damping_toe`, but the `drivability` tool passed none,
  so it always ran 2.5 mm / 2.5 mm / 0.16 / 0.49 s/m. A closed-end pipe's
  D/120 to D/60 toe quake (GEC-12 Vol 2 p. 211; D/60 in medium or dense
  sand, Examples 1, 3, 6) could not reach it.
- **Now.** The tool takes the four inputs with the same descriptions and
  defaults as `single_blow` and `bearing_graph` (`_SOIL_DOC`), and its
  returns list `choices`: the quake-weights choice the module already
  produced (`quake_weighting_choice`, the same text as the other tools).
  A made-up 0.32 m pipe at D/60 = 5.3 mm: drivability's blow count at 12 m
  equals the single blow's, and the choice is the same.
- Left out, each is in `defaults_used` with its value (new rows in the
  tool's output; the numbers do not move).

## GM-301: the embedded length (added)

- **Print.** GEC-12 Vol 2 Sec. 12.7, p. 224 (GRLWEAP's pile input):
  "Length is the total pile length in the leads in feet. For example, if
  plans require a pile of 50 feet in length but the contractor is driving
  60 long piles, the proper analysis length would be the full 60 feet";
  "Penetration refers to the analyzed pile toe penetration below grade in
  feet for Bearing Graph or Inspector's Chart and final penetration for
  Drivability Analyses." P. 227: the final penetration depth "cannot be
  greater than the pile length". Example 1 (p. 162, Fig. 12-3 p. 163): a
  65 ft pile driven 62 ft. Example 6 (p. 182): "The pile has a furnished
  length of 66 feet and an embedded length of 52.5 feet."
- **Module before.** The shaft resistance spread over the whole pile, the
  pile always fully embedded.
- **Now.** `SoilSetup.embedded_length` and the `embedded_length` input of
  `single_blow` and `bearing_graph` (m, more than 0, at most the pile
  length; refused above it, quoting p. 227). The shaft resistance acts
  only below grade; the toe resistance stays at the toe. Left out: the
  pile length (`defaults_used` says so), the whole pile as before.

## GM-301: what embedment means in the drivability study (added)

- **Print.** P. 224 above: in a drivability analysis the pile is the total
  length in the leads and the penetration varies; p. 236, the Depths form:
  "Depth to be analyzed in feet is a required input for at least two
  different depth values. The pile Temporary Length in feet may be less
  than or equal to the length value given as the final length input."
- **Module before.** At each depth a pile as long as the depth: no
  stick-up, so the pile analysed grows with the depth.
- **Now.** `pile_length` (new, drivability only): that pile at every
  depth, embedded to the depth; no depth may pass it (p. 227). Left out:
  the deepest depth analysed (the review fix below; the build kept the
  pile as long as each depth). A per-depth temporary length (p. 236) is
  not offered. The result gives `pile_length_m`.
- Made-up study, Vulcan 010, 0.0098 m2 pipe, 6 / 12 / 18 m at 600 / 1,100 /
  1,700 kN: 39.1 / 76.8 / 243.6 blows/m as long as each depth; 40.1 / 83.5
  / 315.1 with a 21 m pile in the leads (more pile to drive at the shallow
  depths); 39.9 / 83.4 / 337.4 triangular.

## GM-301: the shaft distribution (added; only what is printed)

- **Print.** Uniform: Example 5 (p. 179), "the nominal resistance is
  distributed as 92% uniform shaft resistance and 8% toe resistance".
  Triangular: Example 1 (p. 162), "the resistance on the shaft ... would be
  52% ... and that it would be triangularly distributed", Fig. 12-3 drawing
  it from none at grade to its largest at the toe; Example 6 (p. 182),
  "The shaft resistance will be distributed triangularly along the
  embedded portion of the pile shaft"; Ch. 9, p. 33, "a pile with an
  assumed linearly increasing soil resistance distribution (triangular)".
  Example 4 (p. 174, Fig. 12-9) takes "a variable shaft resistance
  distribution" from a static analysis.
- **Now.** `shaft_distribution` on `single_blow`, `bearing_graph` and
  `drivability`: "uniform" (default, as before) or "triangular". Each
  segment takes the distribution's integral over its part below grade,
  over the integral over the embedded length L_e (`SoilSetup.shaft_shares`):
  uniform, its length below grade over L_e; triangular, ((b − t)² − (max(a,
  t) − t)²) / L_e² for a segment from a to b below the pile head, grade at
  t. Hand-checked on a 10 m pile of 1 m segments, grade 3.5 m down. The
  layer-by-layer distribution of Example 4 is not offered (it needs a
  static analysis per layer, which the tools do not take).
- Fig. 12-14 (Example 6) draws the triangle's apex about 13 ft below grade,
  where the text says "along the embedded portion"; the module follows the
  text (none at grade). Noted only.
- The calc package's input table now prints the distribution and the
  embedded length ("20.00 (the whole pile)" when left out), the
  resistance-distribution step names the distribution and its page, and the
  drivability step says which pile was analysed.

## GEC-12 Example 6 with its printed embedment

14 in x 0.375 in closed end pipe, 66 ft, embedded 52.5 ft, 400 kips, 30 %
shaft triangular along the embedded portion, shaft quake 0.10 in, J_side
0.05 s/ft, J_toe 0.15 s/ft; Table 12-2's hammer cushions as the one cushion
and helmets (p. 183). Blows/ft (pile-top compression, ksi):

| Hammer, toe quake | GRLWEAP (p. 182) | all embedded, uniform (before) | 52.5 ft, uniform | all embedded, triangular | 52.5 ft, triangular (printed) |
|---|---|---|---|---|---|
| Vulcan 014, 0.40 in | 94 | 92.6 (28.8) | 89.5 (30.0) | 88.8 (29.5) | **87.5** (30.1) |
| Vulcan 014, 0.12 in | 56 | 51.7 (32.2) | 51.2 (33.0) | 51.4 (33.1) | **51.3** (33.8) |
| ICE 42-S, 0.40 in | 209 | 103.3 (45.0) | 103.2 (44.6) | 104.7 (44.2) | **105.6** (44.9) |
| ICE 42-S, 0.12 in | 99 | 50.3 (52.4) | 48.8 (52.0) | 48.7 (52.7) | **47.5** (53.1) |

With the printed embedment the Vulcan 014's blow counts move down a
little: at the 0.40 in toe quake from 2 % under GRLWEAP to 7 % under
(92.6 to 87.5 against 94); at 0.12 in hardly at all (51.7 to 51.3 against
56, 8 % under). The embedment does not close the gap; what remains is the
module's other simplifications (one cushion with the helmet lumped, the
quake weights; GM-294). The ICE 42-S stays about half of
GRLWEAP, the diesel cycle the module does not model (GM-29; its results
warn). The pile-top stress stays near Fig. 12-15's about 29 ksi (30.1).
Pinned in `wave_equation/tests/test_gm300_embedment.py`.

Example 1 (Delmag D12-42, a diesel, p. 162) is not reproduced: the module
runs a diesel as a drop ram at its rated energy, so it would test the
diesel cycle, not the distribution.

## Unset, nothing moves

(As built. The review fix below moves a drivability study with
`pile_length` left out above its deepest depth.)

- **Recorded blows.** A pytest plugin recorded every `simulate_blow` call
  (hammer, cushion, pile, soil in; set, stresses, forces, toe displacement,
  steps out) and every wave tool call through `dispatch.call_agent`, on
  master 001526b and on the branch, over `wave_equation/tests`,
  `test_gm205_bearing_graph`, `test_steps_from_a_minimal_analysis`, V-001
  to V-005, and the wave-related tests of the adapter, package, parity and
  engineering-check suites (158 tests both times). 221 blows identical to
  the last digit; of 8 tool calls, 5 identical apart from new
  `defaults_used` rows, and 3 deliberate refusals whose list of valid
  parameters now names the new inputs.
- **Made-up tool calls.** Nine calls (three single blows, three bearing
  graphs including smith_viscous and a Delmag, three drivability studies
  including skin fractions per depth, the 240 blows/ft limit and a 0.8 m
  segment) and two calc packages, master against the branch: every number
  and both packages' figures identical. New on every run: `defaults_used`
  lists `embedded_length` (the pile length) and `shaft_distribution`
  ("uniform") for single blow and bearing graph, and the four quakes and
  damping factors, `pile_length` ("as long as each depth") and
  `shaft_distribution` for drivability; the package prints the two new
  input rows and the assumed-input rows.
- An `embedded_length` equal to the pile length takes the unchanged path,
  so giving the default back moves nothing (the defaults guard checks it).

## Tool text

- `drivability`: `quake_side`, `quake_toe`, `damping_side`, `damping_toe`
  (as the other tools), `pile_length`, `shaft_distribution`; returns
  `choices`, `pile_length_m`, `shaft_distribution`.
- `single_blow`, `bearing_graph` (and the wave equation package):
  `embedded_length`, `shaft_distribution`, each with its pages.

## Checks run

Each in the foreground, by exit code, with at least 3 GB free:

- `wave_equation/tests` (18 new), `test_gm205_bearing_graph`,
  `test_steps_from_a_minimal_analysis`, `test_published_v001_v005`,
  `test_choices_stated_defaults`, `test_citations`: 157 passed.
- `test_defaults_used`, `test_adapter_ergonomics`, `test_top_level_params`,
  `test_gm179_unused_inputs`, `test_calc_package_adapter`,
  `test_calc_package_tool_parity`, `test_calc_package_figure_layout`,
  `test_calc_package_assumed_inputs`, `test_anchor_gm185`, `-k` wave,
  bearing_graph, drivab, blow, hammer: 38 passed.
- `webapp/tests/test_page_fingerprints.py`,
  `geotech_common/tests/test_engineering_checks.py`: 108 passed.
- `test_public_text` on every changed file: clean.

## Owner questions

None. Both distributions and the embedment are printed; the default
(uniform over the whole pile) is unchanged and stated.

## Noted, not fixed

- Example 4's layer-by-layer shaft distribution from a static analysis,
  and GRLWEAP's drivability input of unit shaft and toe resistance per
  depth (p. 233), are not offered; the drivability tool keeps R and the
  skin fraction per depth.
- A temporary pile length per depth (before splicing, p. 236) is not
  offered.
- Fig. 12-14 draws Example 6's triangle from about 13 ft; the module
  follows the text.

## Review fix (gm285-review): drivability's left-out pile length

- **Print.** GEC-12 Vol 2 p. 224: "Length is the total pile length in the
  leads", and the penetration is the "final penetration for Drivability
  Analyses"; p. 236: each depth is analysed with that final length, a
  shorter "Temporary Length" only "prior to splicing"; p. 227: the
  penetration "cannot be greater than the pile length". No page prints a
  pile as long as each depth.
- **Build.** With `pile_length` left out, each depth still ran a pile as
  long as the depth (no stick-up). A long pile at shallow penetration, the
  case that reads high tension, was never analysed.
- **Now.** Left out, the pile is the deepest depth analysed, the shortest
  pile that reaches it (p. 227), the same at every depth and embedded to
  each depth (pp. 224, 236). The result's `choices` state it (how
  "default", the basis, and a longer pile in the leads as the
  alternative, p. 224's 60 ft pile for a 50 ft plan length; a given
  `pile_length` gets the same entry, how "given", so giving the default
  back moves nothing);
  `defaults_used` gives the length taken; `pile_length_m` is always
  returned; the summary and the calc package's Drivability Study step say
  "pile_length left out, the deepest depth analysed". The old per-depth
  length is not offered as an alternative: no page prints it (p. 236's
  temporary length is a section before splicing).
- The deepest depth is unchanged (the pile is fully embedded there); the
  shallower depths move. Made-up study, Delmag D30-32, 0.0155 m2 pile,
  skin fractions 0.8 / 0.7 / 0.6 / 0.6 / 0.5:

| Depth (m), R_ult (kN) | Blows/m before -> after | Compression (MPa) | Tension (MPa) |
|---|---|---|---|
| 5, 300 | 8.5 -> 8.7 | 211.6 -> 222.2 | **43.4 -> 150.0** |
| 10, 700 | 21.1 -> 20.7 | 231.1 -> 225.1 | 11.2 -> 68.6 |
| 15, 1,100 | 33.9 -> 34.6 | 230.2 -> 224.1 | 1.9 -> 1.7 |
| 20, 1,500 | 44.5 -> 45.1 | 230.3 -> 224.0 | 0 -> 0 |
| 25, 2,200 | 79.6 -> 79.6 | 230.2 -> 230.2 | 0 -> 0 |

  Before: the build (41ff055) with `pile_length` left out; after: the
  25 m pile at every depth.
- **Replay** of eight made-up tool calls against 41ff055: single blows and
  bearing graphs identical; the three drivability studies move only above
  their deepest depth (a Vulcan 010 at 4 / 9 / 14 / 19 m: 13.0 / 39.4 /
  69.0 -> 13.6 / 41.7 / 70.3 blows/m, 19 m unchanged) and gain the choice,
  `pile_length_m` and the length in `defaults_used`.
- **Expectations changed** (all in `wave_equation/tests/test_gm300_embedment.py`,
  the build's own): `test_drivability_at_the_pile_length` (`pile_length_m`
  is now returned when left out; the choice is listed) and
  `test_drivability_takes_the_quakes_and_damping` (`defaults_used`
  `pile_length` is 12.0, the deepest depth, not the text). New:
  `TestDrivabilityLeftOutPile`. No other test, validation example or calc
  package pinned the old default: V-003 analyses one depth (the pile is
  the depth either way), and `test_wave_equation.py`'s drivability tests
  check counts, order and refusal only.
