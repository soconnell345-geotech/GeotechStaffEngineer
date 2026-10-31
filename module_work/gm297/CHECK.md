# GM-297: MSE walls against GEC-11 as printed

Checked against FHWA GEC-11, *Design and Construction of Mechanically
Stabilized Earth Walls and Reinforced Soil Slopes* (FHWA-NHI-10-024 Vol I and
NHI-10-025 Vol II, November 2009), as printed. UFC 3-220-20 (16 January 2025)
was searched as well. Its MSE section ends at p. 228: "Further discussion of
MSE is beyond the scope of this manual". So it gives nothing to check against,
except the general sliding rule of Sec. 4-5.1.3 (p. 223) cited in E1.

Scope:
- `retaining_walls/mse.py` and `reinforcement.py`;
- the MSE parts of `geometry.py`, `calc_steps.py` and `results.py`;
- the MSE text in `funhouse_agent/adapters/retaining_walls.py`.

Each item gives the page, the printed form, the module's form and the
verdict.
- "Fixed" items changed the module. They are pinned in
  `retaining_walls/tests/test_gm297_print.py` and
  `funhouse_agent/tests/test_mse_sliding_cohesion_adapter.py`.
- "Noted" items were not changed. A follow-up is proposed only where a saved
  report or a live run's return would change.

## Internal stability

### I1. Le, the length beyond the failure surface. Fixed
- Vol I Eq. 4-36, p. 4-51: "Le ≥ TMAX / (φ F* α σv C Rc) ≥ 3 ft (1 m)".
- Vol I Eq. 4-37, p. 4-51: L = La + Le.
- Vol I Eqs. 4-38 to 4-40, p. 4-52: La.
- Module before: `Le = max(L − La, 1.0)`.
  - A level with less than 1.0 m beyond the failure surface had its pullout
    worked out on 1.0 m, a length it does not have.
  - The 3 ft minimum was never checked.
- Now:
  - Le = max(L − La, 0).
  - Each level reports `Le_at_least_3ft`.
  - A level with Le under 3 ft (0.914 m) fails, whatever its FOS_pullout.
  - The verdict reads "pullout, rupture or Le under 3 ft at z = ...".
- The calc package's internal table notes the 3 ft rule.

### I2. Geosynthetic F*. Fixed
- Vol I Eq. 3-8, p. 3-21: F* = 2/3 tan φ. The module took 0.67 tan φ.
- Now F* = (2/3) tan φ. Every geosynthetic FOS_pullout falls by 0.5 %.

### I3. Steel grid with no t/St. Fixed
- Vol I Eqs. 3-6 and 3-7, p. 3-21: a grid's F* is 20 (t/St) at the top and
  10 (t/St) at 20 ft (6 m) and below.
- Module before: a grid given no t/St took the ribbed-strip curve (Eqs. 3-4
  and 3-5). The print does not give that curve for grids.
- Now such a grid is refused, unless F_star_tested is given.
  - The F_star_tested choice then says the empirical F* was "not worked out
    here".
  - The built-in grids carry their t/St, so they are unchanged.

### I4. Kr/Ka. Matches; comment fixed
- Vol I Fig. 4-10, p. 4-37:
  - metal strips: 1.7, falling to 1.2 at 20 ft;
  - "Metal Bar Mats & Welded Wire Grids": 2.5, falling to 1.2;
  - geosynthetics: 1.0, with the footnote "*Does not apply to polymer strip
    reinforcement".
- The code matches. A comment that read "strips and bar mats alike" was
  wrong and now cites the figure. The docstring adds the polymer-strip
  footnote.

### I5. Failure surface, H1 and S_eq. Matches
- Vol I Fig. 4-9, p. 4-34:
  - inextensible: 0.3H1 over the upper H1/2, with
    H1 = H + tan β · 0.3H / (1 − 0.3 tan β);
  - extensible: ψ = 45 + φ/2.
- Vol I Fig. 4-11: S_eq = ½ (0.7H) tan β.
- Vol I Eqs. 4-38 to 4-40, p. 4-52: La.

### I6. Pullout and rupture factors. Match
- Vol I pp. 4-40 and 4-41: internal stability takes EV 1.35, and a traffic
  surcharge as EV.
- Vol I Table 4-7, p. 4-48, resistance factors:
  - strips 0.75;
  - grids 0.65 (rigid facing);
  - geosynthetics 0.90;
  - pullout 0.90.
- The module's ASD equivalents are the printed LRFD ones:
  - pullout 1.35 / 0.90 = 1.5;
  - geosynthetic rupture 1.5;
  - steel at Tal = 0.55 Fy (strips) or 0.48 Fy (bar mats), p. 4-77, with
    FOS 1.0.
- Vol I Eq. 4-35, p. 4-49: σv unfactored, traffic left out, C = 2 for strips
  and grids. Vol II E4 p. E4-16 takes γP-EV = 1.00 in pullout, as the module
  does.

### I7. α and the strip F*. Match
- Vol I Table 3-6, p. 3-17: α is 1.0 for metallic, 0.8 for geogrid and 0.6
  for geotextile.
- Vol I Eqs. 3-4 and 3-5, p. 3-20: 1.2 + log Cu ≤ 2.0 at the top, falling to
  tan φ at 20 ft. Cu = 4 when unknown (GM-246).

### I8. Corrosion and Tal. Match
- Vol I Table 3-8, p. 3-26.
- Vol I Eqs. 3-11a and 3-12, pp. 3-22 and 3-28.
- The built-in products were checked in GM-262, 266, 267 and 271.

## External stability

### E1. Sliding and the foundation cohesion. Fixed
- Vol I Eqs. 4-12 and 4-13, p. 4-21: Rr = φτ γEV V1 μ.
- Vol I p. 4-22: μ = min[tan φ'f, tan φ'r, or tan ρ for continuous
  reinforcement]. There is no cohesion term.
- Vol I p. 4-21 Step 3(i) checks sliding along the foundation soil when its
  strength "(based on c'f + tan φ'f and/or cu for cohesive soils)" is less
  than the fill's. It gives no form for cu.
- Module before: R = V tan(min φ) + (2/3) c_foundation L, on the ASD, LRFD
  level and LRFD sloping paths. No held source prints the 2/3 c term. GM-212
  took the same term out of the cantilever wall.
- Now `sliding_cohesion` follows the GM-212 precedent:
  - 'drained': R = V min(tan φ'f, tan φ'r), with the cohesion not counted
    (Eq. 4-12; GEC-6 Eq. 5-38).
  - 'undrained': R = min(cu L, V tan φ'r), with no base friction (Step 3 (i)
    and (ii); cu over the base as GEC-6 Eq. 5-39 gives it).
    - GEC-11 p. 4-19 prints no form of its own: "Sliding resistance along
      the base of the wall is evaluated using the same procedures as for
      spread footings on soil as per Article 10.6.3.4 (AASHTO, 2007)". That
      article is not held.
    - UFC 3-220-20 Sec. 4-5.1.3 (p. 223), for a coarse-grained bearing
      layer under the wall: "Sliding should be checked at two locations.
      The shear resistance, Τ, should be taken as the lowest of (1) the
      resistance at the interface between the base of the wall and the
      bearing material and (2) the resistance within the clay using the
      full undrained shear strength." The basis now cites it beside GEC-6
      Eq. 5-39 (review).
    - Alternative offered (review): EN 1997-1 Eq. 6.5, Rd ≤ 0.4 Vd where
      water or air can reach the interface, as the cantilever's
      `interface_gap_possible` (GM-288). Default false. It has its own
      choice entry when the undrained rule is in force. On the 6 m wall at
      cu 120 it gives ASD 3.340 → 1.981 (191.52 / 96.69) and LRFD 2.227 →
      1.3205 (191.52 / (1.5 × 96.69), the critical combination).
  - The default is undrained when phi_foundation is 0, drained otherwise.
  - UFC's 'friction_plus_adhesion' is refused, because it is a concrete
    base's rule.
  - The undrained rule with no cu is refused.
- The choices give the rule, its basis and the other rule's FOS_sliding (or
  governing CDR). They also note that the old rule is unprinted.
- Both tools take the input. applies_when lists it only when c_foundation is
  given.
- The calc package's sliding step follows the rule:
  - its equation and substitution;
  - its reference, "GEC-11 Vol I Eq. 4-12 (p. 4-21), μ p. 4-22";
  - a note when a c' is not counted.
- Effect: a drained c' no longer adds resistance, so FOS falls. An undrained
  cu now counts in full rather than at 2/3, so FOS rises where the
  foundation governs. See the table below.

### E2. Eccentricity limit on rock. Text fixed
- Vol I p. 4-23: "middle one-half of the base width for soil foundations
  (i.e., emax = L / 4) and middle three-fourths of the base width for rock
  foundations (i.e., emax = 3/8 L)".
- The docstring and the tool text said "use 1/6 for rock". Both now say
  0.375 (3L/8). The default is still 0.25.

### E3. ASD required FOS. Wording fixed
- GEC-11 (2009) designs by LRFD. Vol II App. F.1 (p. F-2) says the ASD
  procedure (AASHTO 2002, NHI-00-043) was not updated. Neither is held.
- The calc package cited "GEC-11 (min 1.5)" and "GEC-11 (min 2.0)". It now
  reads "ASD, required 1.5 (FOS_sliding_req)" and "ASD, required 2
  (FOS_overturning_req)". The values are not changed, and are noted under
  "source not held".

### E4. Bearing. Matches
- Vol I Eq. 4-22, p. 4-28: qn = cf Nc + 0.5 L' γf Nγ, with L' = L − 2eB, and
  L' = L when eB < 0 (GM-179).
- The module works Nc and Nγ out from the formula, so it does not carry over
  the Table 4-6 misprint (see Noted).

### E5. Ka and the sloping backfill. Match
- Vol I Eq. 4-1, p. 4-15: Kab = tan²(45 − φ'b/2), for near-vertical walls
  with a level backslope. The calc step now cites it in place of "Rankine
  (1857)".
- Vol I Eqs. 4-2 and 4-3 cover the sloping case and the broken backslope.
  Checked in GM-262 and the V-056/V-057 tests.

### E6. Preliminary length. Text fixed
- Vol I p. 4-11: "the greater of 0.7H or 8 ft (2.5 m)". The module's
  auto-size already took max(0.7H, 2.5 m). Both tools said "0.7H minimum";
  they now cite the page.

### E7. Sliding along the lowest sheet. Fixed (review)
- Vol I p. 4-21 Step 3(iii): "For sheet type reinforcement, sliding along
  the weaker of the upper and lower soil-reinforcement interfaces. The
  soil-reinforcement friction angle ρ, should preferably be measured by
  means of interface direct shear tests. In absence of testing, it may be
  taken as ⅔ tan φ'r."
- Vol I p. 4-22: μ includes "(for continuous reinforcement) tan ρ".
- Vol I Sec. 3.4.3, Eq. 3-9 (p. 3-21): the interface friction coefficient
  "can be conservatively taken as" ⅔ tan φ "for geotextiles, geogrids and
  geonet type drainage composites" (tests per ASTM D 5321).
- Vol II E1 p. E1-7: "Since this is a sheet type of reinforcement, sliding
  should also be checked at the elevation of the first layer of soil
  reinforcement (and applicable height)." E1 prints no value for it.
- Module before: the base only.
- Now, where the lowest level's reinforcement is not steel:
  - the wall above the lowest level slides on tan ρ: its height is that
    level's depth, with the same L, slope and surcharge; its own weight and
    thrust as at the base;
  - tan ρ from `rho_tested` (deg), else ⅔ tan φ'r;
  - on the ASD, LRFD level and LRFD sloping paths;
  - the lesser of base and sheet governs, and the result names it:
    - ASD: `sliding_surfaces` with FOS_base, FOS_lowest_sheet, the depth and
      tan ρ;
    - LRFD: the sliding dict's `surface`, CDR_base_governing and
      CDR_lowest_sheet_governing;
    - the verdict says "along the lowest sheet, z = … m" when it governs
      and fails;
  - a choice entry gives tan ρ, both values, which governs, and the other
    way (tests, or ⅔ tan φ'r when tested);
  - the calc package adds a "Sliding Along the Lowest Sheet" step, and the
    FOS step takes the lesser.
- mse_wall finds the lowest level itself. The LRFD tool has no
  reinforcement, so it takes `lowest_sheet_depth`; left out, it checks the
  base only and says so in defaults_used.
- `rho_tested` is refused on a steel lowest level, and on the LRFD tool
  without `lowest_sheet_depth`.

Before → after, the 6 m wall (L 4.2 m, γ 19, φr 34, Sv 0.6 m, lowest
level at z = 5.7 m), geogrid:

| Case | Before (base only) | After: base | After: sheet | Governs |
|---|---|---|---|---|
| ASD, φf 28 | 2.633 | 2.633 | 2.344 | sheet |
| ASD, φf 34 | 3.340 | 3.340 | 2.344 | sheet |
| ASD, 15° slope, φf 28 | 2.066 | 2.066 | 1.812 | sheet |
| LRFD CDR, φf 28 (critical, EV 1.0, EH 1.5) | 1.755 | 1.755 | 1.563 | sheet |
| ASD, φf 0, cu 40 | 1.738 | 1.738 | 2.344 | base |
| V-056 test path (E3 geometry, a geogrid) | CDR 1.07 | 1.07 | 0.852 | sheet |

The sheet's ASD 2.344 by hand: V 454.86 × tan ρ 0.4497 / Pa 87.26.

## Worked examples
- V-057 (E1, one run at its eleven layers):
  - F* is now 0.450 at every layer, the 0.45 Table E1-7.5 prints. It was
    0.452.
  - The pullout CDRs match within 0.6 %. Layer 10 is 1.3 % under, because
    its TMAX is 1.4 % over, as the test already noted.
  - Every layer has Le ≥ 3 ft.
- V-009/V-010 (E4):
  - Every level has Le ≥ 3 ft; the least is 10.31 ft, L − 0.3H.
  - The two external-stability tests with a custom mat now give E4's
    St = 6 in. (p. E4-15).
- V-056 (E3): unchanged at the base. One test runs E3's geometry with a
  geogrid; its 1.07 is now asserted as the base CDR, and the sheet governs
  (E7).

## What moved (made-up walls)

Cases 3 to 8 are a 6 m wall with L 4.2 m, γ 19 and φr 34, at the base;
case 9 is a 6 m wall with L 5 m. With geosynthetic reinforcement the lowest
sheet may now govern instead (E7).

| Case | Before | After |
|---|---|---|
| 1. Geogrid, H 4 m, L auto 2.8 m, φ 34, top level | Le 1.00 m, FOS_po 4.263, passes | Le 0.83 m, FOS_po 3.532, **fails** (Le < 3 ft) |
| 2. Geogrid, H 6 m, L 3.5 m, φ 30, top three levels | Le 1.00 m each, FOS_po 3.095, pass | Le 0.21 / 0.56 / 0.90 m, FOS_po 0.644 / 1.711 / 2.777, **all fail** |
| 3. ASD sliding, φf 28, c 10 | 2.923 | 2.633 |
| 4. ASD sliding, φf 0, cu 40 | 1.158, **fails** | 1.738, passes (cu L governs) |
| 5. ASD sliding, φf 0, cu 120 | 3.475 | 3.340 (the fill governs) |
| 6. LRFD sliding, φf 28, c 10 | CDR 1.948 | 1.755 |
| 7. LRFD sliding, φf 0, cu 40 | CDR 0.772, **fails** | 1.158, passes |
| 8. LRFD sliding, slope 15°, φf 28, c 10 | CDR 1.554 | 1.425 |
| 9. Steel grid, no t/St | F* 1.747 (strip curve), FOS_po 8.115 | refused |

Every geosynthetic FOS_pullout also falls by 0.5 % (I2). For example, E1's
GG-I wall test goes from 8.394 to 8.353.

## Noted, not fixed

- **Service-limit eccentricity.** Vol II E4 Table E4-6.3 (p. E4-9) shows an
  L/6 limit at Service I. The module checks the strength limit only.
  Possible BACKLOG item, if the owner wants Service I eccentricity reported.
- **Local or punching shear and lateral squeeze.** Vol I p. 4-30: c* = 0.67c'
  and φ* = atan(0.67 tan φ'f) for a soft foundation; Eq. 4-24, the lateral
  squeeze check. Neither is offered.
- **Face batter of 10° or more.** Vol I Eqs. 4-2 to 4-3 with θ, and Fig.
  4-9's tan(ψ − θ) form. The module takes a vertical face.
- **Seismic and barrier impact.** Vol I Ch. 7 and Vol II E7 cover seismic;
  E6 covers barrier impact. Neither is offered by the MSE tools.
- **Polymer strips.** Fig. 4-10's footnote: the geosynthetic Kr/Ka = 1.0
  "does not apply to polymer strip reinforcement". Their curve is in
  product literature. Source not held.
- **Spacing limits.**
  - St 6 to 24 in. for grids (p. 3-21).
  - Sv ≤ 32 in. (800 mm) (p. 4-38).
  - Neither is checked.
- **Fill strength guidance.** Vol I p. 4-10:
  - the reinforced fill at most 34° unless tested, and never above 40°;
  - the foundation at most 30° "in absence of specific data".
  - Not checked.
- **ASD required FOS and eccentricity.** NHI-00-043 and AASHTO 2002 are not
  held (App. F p. F-2). Not checked: source not held.
- **Misprints in the print.**
  - Vol I Table 4-6 (p. 4-29) prints Nc 37.9 at φ = 39°. The formula gives
    67.9. The module uses the formula.
  - Vol II E4 p. E4-13 prints "1.7Ka = 0.707" where 2.5Ka is meant.

## Owner questions

None needed. The sliding rule follows GM-212's precedent and the printed
forms.

One change the lead may want to show the owner: on a cohesive foundation
(φf 0), cu now counts in full, so FOS_sliding rises by about 50 % where the
foundation governs. Case 4 now passes where it failed. The reinforced fill
still caps the resistance at V tan φr. GEC-11 prints no cu form (p. 4-19
defers to AASHTO 2007 Art. 10.6.3.4, not held), so the full cu rests on
GEC-6 Eq. 5-39 and UFC 3-220-20 p. 223 ("the full undrained shear
strength"). EN 1997-1's 0.4 V limit is offered as `interface_gap_possible`.

Geosynthetic walls' sliding falls where the lowest sheet governs (E7): by
about 11 % on the 6 m wall at φf 28, 30 % at φf 34.
