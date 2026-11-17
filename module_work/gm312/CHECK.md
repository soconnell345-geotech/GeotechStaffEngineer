# GM-312: the cantilever wall's structural design

Branch `claude/gm-312-wall-structural-design`. Manual: USACE EM 1110-2-2104, "Strength Design for Reinforced Concrete Hydraulic Structures", 1 November 2023, read from `geotech-references/docs/EM_1110-2-2104_2025.pdf` with PyMuPDF (printed page p is `doc[p + 5]`). Every page cited below and in the code was read from the PDF, not from the library's text JSON.

## Contract changes

**One, from the review (2026-11-13, finding C6): the number of bar sets.** When the base slab's top and bottom covers differ (the usual case: 75 mm top by Table 2-1, 100 mm bottom against the foundation), the base's temperature and shrinkage bars along the wall are TWO sets: `F` (`face: "top"`, the top cover) and `G` (`face: "bottom"`, the bottom cover), same size and spacing. When the covers are equal there is one set `F` with `face: "both"`, as before. So `bars` holds six or seven entries, marks A to F or A to G: **the drawing code must not assume six marks**; read each set's `face` and `cover_mm`. The keys of each bar and of the description are unchanged; the contract test pins both shapes.

Otherwise the `reinforcement` description has exactly the keys of the brief, in its order, and a test pins them (`test_the_reinforcement_description_keeps_its_contract`). Four things GM-313 should know about the values:

- `geometry.key_width` and `geometry.key_distance_from_toe` are always `null` from this module, with or without a key. They are the calc package's own figure inputs, not inputs of the wall tool. The join can fill them from the package's `_shear_key_place`.
- A wall with no toe (or no heel) still gets marks D and E. The missing one is then a temperature and shrinkage set with `element: "base"`, not a flexure set.
- `bars[0].end.ldh_mm` is `null` when f'c is 6,000 psi (41.4 MPa) or more: the manual prints the hook formula's concrete factor for lower strengths only.
- `spacing_mm` and `cover_mm` are the metric values the manual prints beside its inch values (300 for 12 in., 150 for 6 in., 100 for 4 in., 75 for 3 in.). A spacing or cover the engineer gives is passed through in mm, and the callout shows its inch value to two decimals; since the review (C3), a given 300, 150, 100 or 75 mm is taken as 12, 6, 4 or 3 in. and reads "(12 in.)" and so on.

## What the print says

Each row was confirmed on the page. Where the brief's list and the page differ, it is said.

| Need | Page | What it prints |
|---|---|---|
| Load cases for an earth retaining wall | Table E-2, p. 93; notes pp. 93-94 | Case 1 normal operating (usual): serviceability (1.0 or 2.2)(D + EH + EV + Hs); strength (1.2 or 0.9) D + (1.5 or 0.5) EH + (1.35 or 1.0) EV + 1.0 Hs. Case 2 adds a temporary surcharge (unusual): (1.0 or 1.6)(...+ ES); strength + 1.6 ES. Cases 3A, 3B, 4, 5 as the brief says. |
| Permanent load factors | para. 3-3d, p. 22 | D 1.2 or 0.9; EV 1.35 or 1.0; "EH, active pressure, 1.5, passive pressure: 0.5"; "Maximum and minimum load factors must be applied to provide the greatest effect". So the 0.5 in Table E-2's "(1.5 or 0.5) EH" is the passive pressure's factor, not a minimum on the active thrust. |
| Usual and unusual by serviceability, extreme by strength | para. 3-1e, p. 13 | As the brief says. |
| Single load factors | para. 3-4a(2), Table 3-4, p. 26 | 2.2 usual, 1.6 unusual, flexure and shear; "apply only to design using reinforcing steel with yield strength of 60,000 psi (414 MPa)". |
| Service stress limits | Table 3-3, p. 26 | 25 ksi (170 MPa) usual, 35 ksi (240 MPa) unusual. |
| Unfactored analysis times the single factor | para. 3-1f, pp. 13-14 | As the brief says; for strength the analysis "must be performed using factored loads". |
| Steel ratio | para. 3-6 and 3-4b(2), p. 27; rho_b pp. 53, 75 | 0.50 rho_b must; 0.25 rho_b recommended. rho_b = 0.85 beta1 (f'c/fy) [87,000/(87,000 + fy)]. |
| Design yield | para. 3-5, p. 27 | "normally ... 60,000 psi (414 MPa)"; at most 80,000 psi (552 MPa). |
| Minimum wall thickness | para. 3-7, p. 27 | Over 10 ft (3 m) high: 12 in. (30.5 cm); never under 8 in. (20 cm). |
| Steel for a moment; capacity | Eq. D-8, D-9, p. 72; Eq. B-49, B-50, p. 51 | As the library's `design_singly_reinforced`. |
| Effective depth | para. 2-3, p. 4; pp. 75, 84 | The secondary steel's width comes off d "unless the plans and specifications specify that the primary reinforcement is to be on the outside"; both examples take d = h - cover - half a bar. |
| Shear | Eq. 5-1, para. 5-2, p. 35 | Vc = [2 sqrt(f'c) + Nu/(6 Ag)] b d lbs; SI form 0.17 sqrt(f'c). No Vc/2 limit. |
| Shear sections | para. 5-3, pp. 35-36; Fig. 5-1 | Stem at its base; toe at d from the front face of the stem; heel at the back face. |
| Cover | Table 2-1, p. 5 | 4 in. (10 cm) against the foundation; 4 in. at 24 in. (61 cm) or more; 3 in. (7.5 cm) over 12 in. (30 cm); 12 in. or less "According to ACI 318-19". Note: never under 1.5 x aggregate or 2.5 x the bar diameter. |
| Bar spacing | para. 2-6, pp. 4-5 | Clear distance at least 1.5 bar diameters and 1.5 x aggregate; at most 12 in. (30 cm) centre to centre; crack control per para. 2-6b(1) to (4). |
| Temperature and shrinkage steel | para. 2-9, Table 2-3, pp. 6-8 | 0.003 of the gross area, half in each face; No. 4 at 12 in. at least; No. 9 at 12 in. at most; Table 2-3 by joint spacing (0.003 / 0.004 / 0.005); the transverse direction by para. 2-9a (para. 2-9d); primary steel never less (para. 2-9g). |
| Splices | para. 2-8b, p. 6 | None above No. 11. |
| Stem-to-base joint | para. 2-10a, Fig. 2-1, p. 9; para. 5-3, p. 36 | Hooks turned for full moment transfer; both faces' vertical steel developed into the base. |
| Hook development length | D-5 Step 6, pp. 86-87 | ldh = [fy psi_e psi_c psi_r psi_o / (55 lambda sqrt(f'c))] db^1.5; psi_c = f'c/15,000 + 0.6 "for f'c < 6,000 psi". |
| Crack-control spacing | D-3 Step 5(b), p. 76; D-5 Step 5(b), p. 86 | s_max = 15 (40,000 / fs) - 2.5 cc. |
| phi | D-5 Step 3(b), p. 84 | 0.75 shear, 0.90 bending, "(ACI 318-19)". |
| beta1 | pp. 30, 52, 75, 86 | "as specified in ACI 318-19"; 0.85 at 4,000 psi and 0.80 at 5,000 psi are the only values printed. |
| Base pressure for structural design | FHWA NHI-06-089 Fig. 10-17, p. 10-39 | "use trapezoidal distribution with qmax and qmin for rocks and structural analysis". |
| Surcharge and the stem | FHWA NHI-06-089 Sec. 10.5.3, p. 10-41 | "the surcharge loads within the width of the base slab are considered for the structural design of the wall stem". |
| Bars | FHWA GEC-10 Table 6-2, p. 6-3 | Diameters and areas, No. 3 to No. 18. |
| Conversions | EM 1110-2-2104 Glossary Section III, p. 129 | ft 0.3048, in. 0.0254, lbf 4.448222 N, psi 6.894757 kPa. |

**Not printed in a held reference:** where the toe and the heel are cut for moment, and their load diagrams (p. 87 sends the base slab to EM 1110-2-2502, not held); a modular ratio; a minimum flexural steel other than the temperature steel; lap and straight development lengths; psi_c at 6,000 psi or more; the loads on a shear key.

## What I built and why

New files in `retaining_walls/`:

- `structural_section.py`: one strip's design in the manual's US units. Steel (Eq. D-8, D-9), capacity (Eq. B-49, B-50), shear (Eq. 5-1), the balanced ratio, the bar choice, the service stress and crack-control spacing, the hook, the temperature steel, Table 2-1's cover. It calls `geotech_references.em_2104` (`design.design_singly_reinforced`, `shear.shear_capacity_one_way_slab`, `flexure_axial.aci_beta1`) by direct import.
- `structural.py`: `analyze_cantilever_wall_structure` and `CantileverWallStructuralResult` (`summary()`, `to_dict()`). It runs `analyze_cantilever_wall` first and keeps its result. The loads come from the wall module's own `_active_thrust_terms`, `_vertical_force_items` and `active_force_components[_water]`. Nothing in `cantilever.py` or `results.py` was edited.
- `structural_steps.py`: the calc package's structural sections.

Edits to existing files:

- `geotech_common/units.py`: eight converters (MPa and psi, kN-m/m and kip-ft/ft, kN/m and kips/ft, mm2/m and sq in/ft), citing p. 129. Added above the decorator section; nothing existing changed.
- `retaining_walls/__init__.py`: two exports. `retaining_walls/calc_steps.py`: three short branches for the new wall type, at the top of `get_input_summary`, `get_calc_steps` and `get_figures`. `retaining_walls/DESIGN.md`: a section.
- `funhouse_agent/adapters/retaining_walls.py`: the method `cantilever_wall_structural_design`, appended at the end of the file (clear of the lines `claude/gm-316-317-warnings-allowed-values` changes). It takes `cantilever_wall`'s parameters with their conditions and left-out cases, plus 16 structural ones.
- `funhouse_agent/adapters/calc_package.py`: `retaining_wall_package` gains `wall_type="cantilever_reinforced"` (selector text, `tools`, `selector_values`, a branch in the generator, the numbers beside the file). The other branch's lines there (1510, 3430) are untouched.
- `funhouse_agent/tests/method_probes.json`: three probes. `docs/user_manual/manual.html`: the new method's row and parameter table, the package's selector row, and the two method counts.

Inputs beyond the wall tool's: `fc_MPa` and `joint_spacing` (required: no page prints a default), `fy_MPa` (default 414, para. 3-5), `stem_cover_mm`, `base_top_cover_mm`, `base_bottom_cover_mm`, bar size and spacing for stem, toe and heel, `beta1`, `modular_ratio`, `max_aggregate_mm`, `lap_length_mm`.

The result carries, for each of stem, toe and heel: the section and where it is printed, the governing load case, Mu and Vu, d, the steel required and provided, the bar, phi Mn, phi Vc, the load cases and each check with its page; then `reinforcement`, `choices` (the stability analysis's and the design's), `not_designed`, and a verdict that names failed checks first.

## Calls made

The brief's ten were built as written; each is in `choices` with its basis. One was built differently, and these are new.

**Different from the brief.** Call 3 says the service-stress route is offered when `modular_ratio` is given. I kept the single load factors as the design route whenever fy is 60,000 psi, and with `modular_ratio` the service stress is worked out and checked beside it, as D-3 Step 5 does after sizing by the single factor (pp. 75-76). The service-stress route replaces the single factors only for another fy, where Table 3-4 does not apply.

**New calls** (each in `choices`):

1. **Which load is which.** D the concrete; EV the soil and a dead-load surcharge on the heel; EH the active thrust without the live load, a dead-load surcharge's thrust included; ES the live-load surcharge; Hs the water's thrust and the uplift. The manual names the loads (Table E-1, p. 92) and prints no rule for a permanent surcharge. The other reading puts the dead-load surcharge in ES, which would move it from case 1 to case 2.
2. **Both columns are worked for both cases.** Para. 3-1e (p. 13) says usual and unusual loads are designed by serviceability, yet Table E-2 prints a strength column for them and D-5 works both. The larger governs. On the stem the serviceability column always governs (2.2 EH against 1.5 EH); on the toe and heel the strength column can.
3. **Strength combinations.** Each of the four D and EV factor pairs is worked on the whole wall, with the base pressure from those factored loads. The passive resistance in front of the wall is left out of the base pressure.
4. **Case 2's surcharge** is tried over the heel and behind it only, and the larger moment and shear are taken. On the hand-check wall the heel's moment is larger with the surcharge behind the heel.
5. **Case 4 (maximum hydrostatic) is not designed.** It needs a second water level and its return period. The water table given is taken as the normal operating level of cases 1 and 2.
6. **The stem** is designed at its base only, for the pressure on the vertical plane through the back of its base, with no axial load.
7. **The uplift** is added to the bearing pressure as an upward load on the slab, at the Hs factor.
8. **Temperature steel on a tapered stem** is on its thickness at the base, for the whole stem (the larger steel). Up the stem and across the base the ratio is 0.003 (para. 2-9d's "transverse (shorter) direction").
9. **The cover's row of Table 2-1** for the stem is by its thickness at the base.
10. **The hook** is checked against the base thickness less the bottom cover and the bottom mat's two layers, from D-5 Step 6(d), "the hooks will rest on the bottom mat". The example's factors are taken as it takes them (uncoated bars, no confining steel, side cover of 6 bar diameters or more).
11. **Crack control with a live-load surcharge and no modular ratio.** The spacing limit is on the 35 ksi limit stress (para. 2-6b(3) allows it): 15 x 40/35 - 2.5 x 2.5 = 10.9 in. So 12 in. is passed over and the main bars go at 6 in. This will be the common result for a wall with traffic. Giving `modular_ratio` or a bar spacing changes it. The owner may prefer a third default spacing between 6 and 12 in.; the manual's examples print only those two.
12. **Metric values.** The design is worked at the manual's inch values and reports the metric values the manual prints beside them (12 in. as 300 mm). The steel area reported is the inch design's (No. 6 at 12 in. is 931 mm2/m, not 947).
13. **fy within 0.5 % of 60,000 psi** is taken as 60,000 psi (414 MPa is 60,045 psi).

## Pinned against the print

`retaining_walls/tests/test_gm312_structural.py`.

| Example | Printed | Module |
|---|---|---|
| D-3 Mu (p. 74) | 11 k-ft | 11.0 |
| D-3 Mn | 147 k-in. | 146.7 |
| D-3 Ku, As (p. 75) | 0.105, 0.43 sq in | 0.1054, 0.430 |
| D-3 bar | No. 6 at 12 in. | No. 6 at 12 in. |
| D-3 rho against 0.25 rho_b | 0.0061 against 0.0071 | 0.00611 against 0.00713 |
| D-3 x, fs (p. 76) | 1.61 in., 24.96 ksi | 1.609, 24.96 |
| D-3 s_max | 19 in. | 19.04 |
| D-5 Vu, Mu from Tables D-2 and D-4 (pp. 80, 84) | 36.85 kips/ft, 428 k-ft/ft | 36.85, 427.9 (rebuilt by statics in the test) |
| D-5 Vc, phi Vc (p. 85) | 61.9, 46.5 kips/ft | 61.94, 46.46 |
| D-5 Ku, As | 0.088, 2.73 sq in/ft | 0.0878, 2.726 |
| D-5 bar | No. 11 at 6 in. | No. 11 at 6 in. |
| D-5 rho against 0.25 rho_b (p. 86) | 0.0071 against 0.0083 | 0.00712 against 0.00838 |
| D-5 psi_c, ldh (p. 87) | 0.93, 24.1 in. | 0.933, 24.11 |
| D-5 temperature steel (p. 87) | 1.26, held to No. 9 at 12 in.; No. 8 at 0.003 | 1.00; No. 8 |

D-5's Tables D-2 and D-4 do print enough to rebuild the stem's loads: the two water triangles, the at-rest earth on the buoyant weight each side, and the wave load at its elevation.

D-5's own slips, not pinned: Step 3(g) (p. 85) uses the 4,000 psi coefficient 2.4956 for 5,000 psi concrete; Step 3(i) prints Ku = 0.088 and then puts 0.111 in the As line (the 2.73 follows from 0.088); Step 5(a) (p. 86) prints x = 9.96 in. and then uses 10.09, and calls the service moment "Mn"; and it prints fs = 34.7 ksi "< 35 ksi, OK" where its own numbers (M = 306 k-ft, As = 3.12, d = 36.5, x = 9.96 or 10.09) give 35.5 ksi, over the 35 ksi limit of Table 3-3: the module gives 35.47 ksi and FAILS the check on the manual's own example, which is the correct arithmetic (found by the review, B6; pinned in `test_d5_service_stress_is_the_arithmetic_not_the_printed_34_7` so nobody "corrects" it to 34.7). D-3 Step 4 cites "paragraph 3-5" on p. 53 for the steel ratio limit, which is para. 3-6.

## A made-up wall, by hand

H = 4.5 m, B = 3.0 m, toe 0.6 m, stem 0.305 m at the top and 0.45 m at the base (back battered), base 0.45 m, embedment 0.9 m. Backfill 19 kN/m3, phi 32 deg; concrete 23.6 kN/m3; live surcharge 12 kPa. f'c 28 MPa, fy 414 MPa, joints 8 m apart. Before this branch the wall tool stopped at the stability checks; the numbers below are new.

**Stem** (h = 4.05 m, Ka = tan^2(29 deg) = 0.3073):

| | By hand | Module |
|---|---|---|
| EH: 0.5 x 0.3073 x 19 x 4.05^2, at h/3 | 47.88 kN/m, 64.64 kN-m/m | 47.88, 64.64 |
| ES: 0.3073 x 12 x 4.05, at h/2 | 14.93 kN/m, 30.24 kN-m/m | 14.93, 30.24 |
| Case 1 serviceability, 2.2 EH | M 142.20, V 105.33 | 142.20, 105.33 |
| Case 1 strength, 1.5 EH | M 96.95 | 96.95 |
| Case 2 serviceability, 1.6 (EH + ES) | M 151.80, V 100.50 | 151.80, 100.50 |
| Case 2 strength, 1.5 EH + 1.6 ES | M 145.34 | 145.34 |
| Mu, Vu | 151.80 kN-m/m (34.13 kip-ft/ft), 105.33 kN/m (7.22 kips/ft) | same |

Steel: h = 17.72 in., 3 in. cover (Table 2-1), No. 5: d = 17.72 - 3 - 0.3125 = 14.40 in. Mn = 34.13 x 12 / 0.9 = 455.0 k-in. Ku = 1 - sqrt(1 - 455.0 / (0.425 x 4.061 x 12 x 14.40^2)) = 0.0544. As = 0.85 x 4.061 x 0.0544 x 12 x 14.40 / 60 = 0.541 sq in/ft (1,146 mm2/m). Least steel 0.003 x 17.72 x 12 / 2 = 0.319. At 12 in. No. 7 (0.60) would do, but case 2's crack-control spacing on the 35 ksi limit is 10.9 in., so 6 in.: No. 5 at 6 in. = 0.62 sq in/ft (1,312 mm2/m). a = 0.62 x 60,000 / (0.85 x 4,061 x 12) = 0.898 in.; phi Mn = 0.9 x 0.62 x 60 x (14.40 - 0.449) / 12 = 38.93 kip-ft/ft (173.2 kN-m/m). phi Vc = 0.75 x 2 sqrt(4,061) x 12 x 14.40 / 1000 = 16.52 kips/ft (241.1 kN/m); the SI form gives 246.8 kN/m. rho = 0.00359 against 0.50 rho_b = 0.01442. The module prints each of these.

**Base, case 1, unfactored:**

| Item | W (kN/m) | Arm from toe (m) |
|---|---|---|
| Base slab 23.6 x 3.0 x 0.45 | 31.86 | 1.500 |
| Stem, rectangle 23.6 x 0.305 x 4.05 | 29.15 | 0.7525 |
| Stem, taper 23.6 x 0.5 x 0.145 x 4.05 | 6.93 | 0.9533 |
| Soil over the battered back | 5.58 | 1.0017 |
| Soil on heel 19 x 1.95 x 4.05 | 150.05 | 2.025 |
| Soil over toe 19 x 0.45 x 0.6 | 5.13 | 0.300 |
| V | 228.70 | |

Thrust on H: 0.5 x 0.3073 x 19 x 4.5^2 = 59.11 kN/m at 1.5 m. Resultant at (387.32 - 88.66) / 228.70 = 1.3059 m from the toe; e = 0.194 m; q_toe = 105.83 kPa, q_heel = 46.63 kPa. The module: 228.70, 1.3059, 105.83, 46.63.

**Toe** (0.6 m): pressure at the face 93.99 kPa. Moment of the pressure about the face 16.92 + 1.42 = 18.34; slab and soil over the toe, 19.17 kPa, 3.45. M = 14.89 kN-m/m; x 2.2 = 32.76, which governs (module 32.76). Shear at d = 0.342 m from the stem: 2.2 x 21.70 = 47.74 kN/m (module 47.74). No. 4 at 6 in. on the 0.319 sq in/ft least steel.

**Heel** (1.95 m): pressure at the back face 85.11 kPa. Down: 87.57 kPa x 1.95^2 / 2 = 166.49. Up: 88.67 + 24.38 = 113.05. M = 53.44 kN-m/m, V = 42.31 kN/m; x 2.2 = 117.57 and 93.08 (module the same). Case 2 with the surcharge behind the heel only: M = 80.26, x 1.6 = 128.41, which governs (module 128.41). Vu = 99.82 kN/m from case 2 strength (0.9 D + 1.5 EH + 1.35 EV + 1.6 ES). No. 5 at 6 in.

Stem hook: psi_c = 4,061/15,000 + 0.6 = 0.871; ldh = 60,000 x 0.871 / (55 x 63.73) x 0.625^1.5 = 7.4 in. (187 mm) against 17.72 - 4 - 0.5 - 0.75 = 12.5 in. available.

The wall's stability on the same inputs is the wall tool's own, unchanged: sliding FOS 1.18 (fails), overturning 3.07, bearing 2.43. The verdict leads with it.

## What I looked at

The calc package of the wall above as a PDF (37 pages, 19 of them structural), rendered to PNG and opened: the basis and load-case page, the stem's loads and combinations page, a steel page, the toe's page, the checks, and the bar schedule and hook page. Changed after looking: the load-case table was six columns with the strength column wrapped a word to a line (now four); the combinations table was ten columns (now seven); the bar schedule was eleven columns (now eight); the hook step printed the bar diameter from a rounded mm value (0.626 in. for 0.625); a recommended check's label said "recommended" twice. The package has no figure of its own: the two stability figures are the plain cantilever package's.

## Tests

Exit codes read directly, never through a pipe.

| Run | Result |
|---|---|
| `retaining_walls/tests/test_gm312_structural.py` (new) | 26 passed |
| `funhouse_agent/tests/test_gm312_wall_structural_adapter.py` (new) | 4 passed |
| `funhouse_agent/tests/test_defaults_used.py -k "retaining or wall"` | 123 passed |
| `test_defaults_used.py`, `test_calc_package_tool_parity.py`, `test_adapter_ergonomics.py`, `test_calc_package_adapter.py`, `test_calc_package_figure_layout.py`, `test_choices_stated_defaults.py`, `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py` | 1301 passed, 23 skipped, 1 failed (see below) |
| `retaining_walls`, `calc_package`, `geotech_common`, `validation_examples`, `funhouse_agent/tests`, and four `webapp/tests` files (fingerprints, public text, docs currency, backlog) | 5963 passed, 31 skipped, 1 xfailed, 3 failed (see below) |
| After the fix below: `retaining_walls`, `calc_package`, the named `funhouse_agent/tests` files with `test_top_level_params.py` and the new adapter tests, fingerprints, public text | 2181 passed, 26 skipped, 1 failed (the `plot_data` probe, already failing on master) |
| `python module_work/doc10/check_manual.py` | OK |

The three failures of the broad run:

- `test_top_level_params.py::test_method_info_declares_only_what_the_adapter_accepts` was mine: the new method left the check of its parameter names to `cantilever_wall`'s, which does not know the structural names. The method now checks its own names first. Fixed, and rerun.
- `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]` (`logx`, `logy`) and `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table` fail the same way on untouched master in this environment (each checked with the branch stashed). Nothing here touches the plotting tool or the pile-group figure.

`webapp/tests/test_page_fingerprints.py` passed without a re-record: the new method and the selector text did not move a page fingerprint.

## Release note

A new tool and a new calc-package type; no existing calculated value changes. `retaining_walls.cantilever_wall_structural_design` designs a cantilever wall's concrete on USACE EM 1110-2-2104 (2023): stem, toe and heel, each with its factored moment and shear, the steel required, a bar size and spacing, the printed checks and the cover, after the stability analysis on the same inputs. `retaining_wall_package` with `wall_type="cantilever_reinforced"` prints the stability sections and then the structural ones. It needs the concrete strength and the joint spacing. It does not design the earthquake, maximum hydrostatic or construction cases, lap or development lengths, bar cut-offs or a shear key's steel, and says so in the result.

## Stopped short of

Named in the result's `not_designed`:

- the earthquake cases (Table E-2 3A, 3B), the maximum hydrostatic case (4) and the construction case (5);
- lap lengths, straight development lengths and bar cut-offs up the stem (ACI 318-19, not held);
- a shear key's steel;
- shear friction across the joint at the base of the stem (para. 5-3 asks for the steel and prints no check);
- deflection (the 0.25 rho_b recommendation is reported);
- counterforts and shear reinforcement.

Also not done, and said where it arises:

- the hook development length at f'c of 6,000 psi or more (`checks_not_run`);
- steel for a toe or heel whose moment reverses in some combination (a warning names it; the other face's bars are not checked for it);
- sections up the stem above its base.

## Library defects met (geotech-references, not edited)

- `em_2104/loads.py`: `table_3_2_permanent_load_factor` and `table_3_2_load_factor` give printed pages "23-24"; Table 3-2 is on pp. 24-25.
- `em_2104/flexure_axial.py`: `balanced_strain_kb` gives p. 29; Eq. 4-3 is on p. 30. `eccentricity_ratio` gives p. 29 for "4-1/4-2"; Eq. 4-2 is on p. 30.
- I did not call either: the load factors are constants in `structural.py`, each cited to the page read.

## Noted, not fixed

- A wall on the tool's own trial proportions with no thickness given has a 0.30 m stem top, which is 5 mm under para. 3-7's 12 in. (30.5 cm) for a wall over 3 m high. The check fails as printed and the verdict says so; giving `stem_thickness_top` 0.305 passes. The 5 m default-proportioned probe wall also fails the hook check (a No. 8 hook needs 379 mm, the 0.5 m base has 360 mm).
- The tool's result is about 39 kB of JSON, a quarter of it `choices`.
- `docs/user_manual/manual.html` line 1594 differs from a rebuild by `*...*` against `<em>...</em>` in one existing parameter description of `cantilever_wall`. It was so before this branch; I kept the committed line.
- `structural_section.design_section` picks 12 in. then 6 in. A given spacing is taken as it is, and a spacing over 12 in. fails the printed limit rather than being refused.

## Review fixes (2026-11-13)

The independent review of a969908 (findings A1, B1 to B6, C1 to C9) and the lead's decisions on it; branch `fix/gm312-review`. One line each.

- **A1** Temperature steel along the wall banded on the metres Table 2-3 (p. 7) prints: below 9 m 0.003, 9 to 12 m 0.004, above 12 m 0.005 (9.0 m was 0.003, 12.1 m was 0.004); the choice and the `joint_spacing` text say so; tests at 8.9, 9.0, 12.0 and 12.1 m.
- **B1** A cover the engineer gives is a required wall check, `clear cover >= Table 2-1 (stem | base top | base bottom)`, in mm against the printed 10 cm / 7.5 cm, with a note naming the value to give; a formed face 12 in. or thinner keeps the old rule.
- **B2** The choice "strength load factors" and the calc package's Table E-2 table say the 1.5 on EH assumes the active state (Table E-2 note b, p. 93, quoted) and name the at-rest basis (EH driving 1.35, para. 3-3d, p. 22) as not built. No new input.
- **B3** A trial bar is chosen only when its clear cover (2.5 db, Table 2-1 Note) and clear distance (para. 2-6a) pass too; the review's wall (H 6.6 m, stem 550 mm, no surcharge) now gets No. 8 @ 150 mm and passes (was No. 11 @ 300, failing the cover).
- **B4** Wording only: the result's development (`confirm`), the anchorage choice, the drawing note, the summary and the calc package's hook step say the factors are D-5 Step 6's (pp. 86-87), that ACI 318-19 (not held) sets when each applies and its own minimum lengths, and that the hook must be confirmed against ACI 318-19. No ACI value written.
- **B5** Para. 3-7 (p. 27) in metric: the wall's height (underside of base to top of stem, the conservative reading, said in a new choice "minimum wall thickness") against 3 m; the stem top against 305 mm or 200 mm; a failure names the value to give. In `cantilever_wall_structural_design` only, a stem top left out is 0.305 m for the stability run and the design (choice "stem thickness at the top", parameter text and default 0.305); `cantilever_wall` unchanged. `when_absent` could not carry it: aec-agent-core refuses a case on a parameter with a default, so the parameter's default (0.305) and description say it, and `defaults_used` lists 0.305.
- **B6** The module's 35.47 ksi for D-5 Step 5(a) is pinned with a comment (the print says 34.7; its own numbers give 35.5); the slip is added to D-5's list above.
- **C1** The "at least" checks read "needs 3.53 in., has 3.00 in." (the least value rounded up); a failed check's note follows it in the verdict.
- **C2** The choice "which loads are which" quotes para. 3-2f(7) (p. 19) for the live surcharge; still no rule printed for a permanent one.
- **C3** The spacing parameters say 300 (the manual's "12 in. (30 cm)"); a given 300, 150, 100 or 75 mm spacing or cover is taken as 12, 6, 4 or 3 in.
- **C6** Bar set F splits into F (top) and G (bottom) when the base's covers differ; "Contract changes" above.
- **C8** FHWA GEC-6 (FHWA-SA-02-054) Sec. 6.6 added to the calc package's REFERENCES.
- **C4, C5, C7, C9** Not changed, by the lead's decision (C4 and C5 are for the drawing code).
- The first bullet of "Noted, not fixed" above (the 0.30 m default top) is answered by B5 for this tool.
