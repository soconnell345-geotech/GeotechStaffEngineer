# GM-290: soe and sheet_pile against the printed references

Checked against the Caltrans Trenching and Shoring Manual (July 2025), FHWA
GEC-4 (Ground Anchors and Anchored Systems) and UFC 3-220-20 (16 January
2025), as printed. Takes in GM-193 (an embedment for two or more anchors in
`fhwa_apparent_pressure_anchored_wall`). Each item gives the page, the printed
form, the module's form and the verdict. "Fixed" items changed the module and
are pinned in `soe/tests/test_gm290_print.py` or
`sheet_pile/tests/test_gm290_print.py`; "noted" items were not changed, and a
follow-up is proposed only where a saved report or a live run's return would
change.

## soe

### S1. Coulomb Kp (`earth_pressure.coulomb_Kp`). Fixed
- Caltrans Eq. 4-3-24, p. 4-13: Kp = cos²(φ + ω) / (cos²ω cos(δ − ω)
  [1 − √(sin(δ + φ) sin(φ + β) / (cos(δ − ω) cos(β − ω)))]²).
- Module: numerator sin²(α + φ), with α = 90° − ω. That is the same as the
  printed form for a vertical wall only. Now sin²(α − φ), the printed form
  for any batter. sheet_pile's copy already had it right.
- Pinned: the printed form at four batters and slopes (ω −8° to 10°,
  β 0° to 10°).

### S2. Sand envelope. Matches
- Caltrans Eq. 8-3-1, GEC-4 Fig. 23 and UFC Fig. 4-51: 0.65 Ka γ H, uniform.

### S3. Soft to medium clay envelope. Fixed
- GEC-4 Fig. 23(c), p. 50, and UFC Fig. 4-52, p. 255: the pressure rises
  over the top 0.25H, then stays constant. Ka = 1 − m (4 Su / γH), with
  m = 1 (GEC-4 Eq. 11, p. 57).
- Caltrans Eq. 8-3-4, p. 8-6: Ka "not to be less than 0.25".
- Module: uniform over the whole height, with the 0.25 floor cited to "FHWA
  guidance". Now shape `ramped_top`, and the floor cites Eq. 8-3-4.

### S4. Clay with 4 < N < 6. Fixed
- Caltrans Sec. 8-3.02, p. 8-6: "use the larger σa from Equations 8-3-2 and
  8-3-3".
- Module: the soft clay value only. Now the larger of Ka γ H and the stiff
  clay ordinate, reported as `governed_by`.
- GEC-4 p. 59 and UFC take Ka = 0.22 for 4 < N ≤ 5.14 and Henkel's equation
  above that. Both need the depth to a firm layer below the base. They are
  given in the choices only, not computed.

### S5. Stiff clay envelope. Fixed. Owner question
- UFC Fig. 4-53, p. 256: 0.4 γH, or 0.2 γH "when movements are minimal and
  the construction period is short".
- Caltrans Eq. 8-3-2, p. 8-5: 0.2 to 0.4 γH, by "best engineering
  judgment".
- GEC-4 Fig. 27, p. 56: the same range.
- Module: 0.2 + 0.2 N/4, which none of the three prints. Now
  `stiff_clay_coefficient`, default 0.4, with the range checked, and stated
  in the choices.

### S6. Braced wall support loads and span moments. Fixed. Owner question
- Caltrans Sec. 8-4.02, Fig. 8-7 (p. 8-10) and Example 8-2, the hinge
  method:
  - T1U is the load above the first support.
  - M1 at the first support is carried by the span below:
    T1L = P/2 + M1/L and T2U = P/2 − M1/L.
  - The other spans are simply supported.
  - The lowest support's load from below is free earth support's, about
    that support.
- GEC-4 Fig. 38, p. 79: the same above, then simple spans down to the base,
  with a subgrade reaction R there.
- UFC Fig. 4-54, p. 257: simple spans, with the span below the lowest
  support a cantilever on it.
- Module: simple spans, with the whole bottom span sent to the lowest
  support but given a simple span's moment. That matches none of the three.
- Now `support_load_method`:
  - `hinge` (default; Caltrans);
  - `hinge_subgrade` (GEC-4);
  - `simple_spans` (UFC).
- Pinned to Example 8-2: T1U 77,474, M1 279,729, T1L 86,093, T2 88,267,
  T3 116,240 and T4U 58,120 lb, each within 0.2 %.
- Under `hinge`, a single support with no FS = 1 moment balance gets a load
  of `None`, with a note; `from_above_kN_per_m` still gives the load from
  above.

### S7. Cantilever moment and shear. Fixed
- Caltrans Example 7-3, p. 7-41: the embedment is found at FS 1.3, then
  "Calculate embedment depth using a factor of safety (FS) equal to 1.0 for
  the purposes of calculating the shear and moments".
- UFC Fig. 4-41 step 5 (p. 242) uses the factored diagram.
- Module: the factored diagram. Now the FS = 1.0 body, with the factored
  value offered in the choices (`moment_body_D0_m`).

### S8. Allowable bending stress. Fixed. Owner question
- The Caltrans examples use 0.60 Fy: Example 7-3, p. 7-43, and Example 8-1.
- GEC-4 p. 81: 0.55 Fy for permanent and critical temporary walls.
- Caltrans p. 9-20, the railroad example, compares against 0.66 Fy.
- Module: 0.66 Fy everywhere. Now `Fb_ratio`, default 0.60, stated in the
  choices.

### S9. Basal heave, Terzaghi. Fixed
- GEC-4 Eq. 35, p. 106: FS = Nc Su / (γH − √2 Su H / B).
- GEC-4 Eq. 37, p. 107: Nc Su / γH, for a wide cut.
- Nc comes from GEC-4 Fig. 51(c), B/L = 0, read from its vector paths.
- The module adds q to γH, as Caltrans Eq. 10-3-2 does.
- Module: Su Nc / (γH + q), with an unprinted Nc table read at
  H / min(B, √2 H) and no side shear. Now Eq. 35 when B > 0 and Eq. 37 when
  B = 0. When the side shear is at least γH + q, there is no FS and the
  check does not pass (a note says so).
- Eq. 35 runs high in a narrow cut. With B 4 m, H 6 m and Su 50 kPa it gives
  FS 173, because the side shear is 106 of the 108 kPa.
- UFC p. 73 note 10: Terzaghi's subtraction "can lead to unreasonably high
  factors of safety for narrow excavations". Fig. 2-7(b) (p. 75, flexible
  wall) is now given beside the result as a note and as
  `FOS_ufc_fig_2_7b`:
  - FBH = (5.7 Su B1 + Su H) / ((γH + q) B1), with B1 = 0.7B;
  - 3.63 in the same case.
- GEC-4 Eq. 36 (a stiff layer at D) is not offered. It is noted in the
  results.

### S10. Basal heave, Bjerrum-Eide. Fixed
- Caltrans Fig. 10-15 gives the square-footing curve, read from the printed
  chart to about ±0.03. Eq. 10-3-4 adjusts it for shape: × (0.84 + 0.16 B/L).
- Module: an unprinted table. Now the chart reading times Eq. 10-3-4.
- Pinned: Nc(square) 8.5 at H/B = 2.

### S11. Basal heave, Caltrans force balance (Sec. 10-3.01). Fixed
- Example 10-2, pp. 10-16 to 10-20: Nc 7.6, FS 1.54.
- Module: Skempton's formula, which gave Nc 7.68. Now Nc is taken as in
  S10: 7.575, FS 1.533. A hard layer is handled through
  `width_factor` = D/B.
- Pinned in V-014.

### S12. FHWA anchored wall, M1 at the upper anchor. Fixed
- GEC-4 Appendix A, Example 1, p. A-6:
  M1 = (13/54) H1² pe + ps H1²/2, which comes to 76.
- Module: (13/54) H1² (pe + ps) = 70.4. Now 75.7.
- V-016's M1 row was PASS (M2,3) / CONVENTION (M1); M1 now passes too.

### S13. FHWA anchored wall with two or more anchors (GM-193). Fixed. Owner question
- The embedment below the lowest anchor now follows the Caltrans hinge
  method, Sec. 8-4.02 steps 3-4: free earth support about the lowest anchor.
  The trapezoid plus ps acts from that anchor down, with Rankine active
  below the excavation against `Kp`.
- New returns: `embedment_D_m`, `embedment_D_FS1_m`,
  `lowest_anchor_load_from_below_kN_per_m` and
  `free_earth_max_moment_kN_m_per_m`.
- `Kp` and `FOS_embedment` are now read on both routes. Before, with two or
  more anchors, they were refused (GM-182).
- GEC-4 Example 1: D 2.954 m at FS 1.3, D′ 2.358 m, load from below
  98.9 kN/m. The free earth body's moment is 107 kN-m/m at z 8.59 m,
  larger than the tributary 75.7. A note says so; `max_moment` stays the
  tributary value.

### Noted, not changed
- Caltrans Fig. 8-4 ramps the cohesive envelope over 2/3 of H1. Not
  offered; GEC-4's and UFC's 0.25H ramp is used.
- N averages Su over the excavated depth, as Caltrans does. GEC-4 reads Su
  below the base, for Henkel's equation.
- Bottom blowout and piping: none of the three prints a form the module
  could follow. The module keeps its own.
- UFC Fig. 4-54 (`simple_spans`): the 80 % reduction of inner span moments
  and the unbalanced force P′H on the buried length are not applied. A note
  says so.
- anchor_design (GEC-4 Table 6, minimum free and bond lengths): matches.
- K0 and tension crack depth: match.
- wall_sections uses manufacturer section data, which is not in the three
  references.
- Caltrans Example 8-2's T4L cannot be pinned: it uses soldier-pile arching
  widths.

### Results before and after (soe)

Every soe call in the soe tests, the validation and package tests, and the
funhouse/calc_package tests matching soe, sheet, anchor or excavation was
recorded before and after the change, with the same inputs. The values that
moved, and why:

| What | Before | After | Cause |
|---|---|---|---|
| Braced, 8 m in sand, two supports, lowest support load (kN/m) | 137.4 | 84.7 | S6: under the hinge method its load from below is free earth support's, not the whole bottom span |
| Braced soe package (8 m), lowest support load (kN/m) | 143.3 | 98.4 | S6 |
| Soft clay H 8 m, Su 30 kPa (N 4.8), apparent pressure (kPa) | 36.0 | 57.6 | S4: the larger of 0.25 γH and 0.4 γH for 4 < N < 6 |
| Soft clay H 6 m, Su 20 kPa (N 5.4), apparent pressure (kPa) | 28.0 | 43.2 | S4 |
| Stiff clay H 6 m, Su 50 kPa, apparent pressure (kPa) | 33.3 | 43.2 | S5: 0.4 γH, not 0.2 + 0.2 N/4 |
| Braced stiff clay, moment at zero shear below the base (kN-m/m) | 104.8 | 131.8 | S5 (envelope) |
| Cantilever soe, maximum moment (kN-m/m), e.g. | 218.1, 256.8, 1,666 | 141.5, 197.2, 1,013 | S7: the FS = 1.0 body; the factored moment stays in the choices |
| Required section modulus (cm³), braced / cantilever workflow tests | 474.1 / 540.8 | 521.5 / 466.4 | S8: 0.60 Fy, not 0.66 Fy (× 1.10 at the same moment); in the cantilever the smaller FS = 1.0 moment (S7) more than offsets it |
| Bjerrum-Eide FS, e.g. | 2.634, 3.148 | 3.044, 3.556 | S10: Nc from Fig. 10-15, not the unprinted table |
| Caltrans force balance, V-014 Nc / FS (printed 7.6 / 1.54) | 7.676 / 1.553 | 7.575 / 1.533 | S11 (with a chart-read Nc given: 1.538 both) |
| Terzaghi FS, B 4 m, H 6 m, Su 50 kPa | 2.778 | 173.0 (UFC Fig. 2-7(b) 3.63 beside it) | S9: Eq. 35's side shear |
| Terzaghi, B 0.5 m, H 6 m, Su 40 kPa | FS 3.185, passes | no FS (side shear 679 > 108 kPa), does not pass; UFC Fig. 2-7(b) 8.46 in the note | S9 |
| FHWA two-anchor M1 (kN-m/m), GEC-4 Example 1 (V-016, WE-SOE-2) | 70.43 | 75.69 | S12 |
| FHWA two-anchor, another wall | 73.7 | 79.0 | S12 |
| FHWA two anchors: embedment, Kp, FOS_embedment | refused or absent | computed and listed | S13 |

Nothing else moved in soe. Within one function's outputs, the records also
showed paired-input moves from the new choice and default parameters
(`support_load_method`, `stiff_clay_coefficient`, `Fb_ratio`). Those come
from the tests that exercise them, not from a value change.
