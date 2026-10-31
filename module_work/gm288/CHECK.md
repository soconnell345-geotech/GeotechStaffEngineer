# retaining_walls (cantilever wall) against GEC-6 and UFC 3-220-20, as printed

BACKLOG GM-288, 2026-10-31; takes in GM-32 and GM-235. Read from the PDFs in
`geotech-references/docs/`: GEC-6 (FHWA-IF-02-054, 2002) chapters 5-7; UFC
3-220-20 (16 January 2025) Sec. 4-3 (earth pressures), Sec. 4-5.1 (rigid
walls, Figs. 4-27 and 4-29) and App. B-1 (the cantilever cut wall); FHWA
NHI-06-089 (2006) Vol II Sec. 10.5 and Example 10-3; EN 1997-1:2004 Sec.
6.5.3. Page numbers are the printed ones.

The module checks a cantilever wall's external stability: the active thrust
on the vertical plane through the back of the heel, sliding, overturning,
the eccentricity and the bearing pressure. Every equation, default and
limit it uses was read against the print. What was fixed, what was added as
an option and what was only noted is below. Pinned to two printed worked
examples, V-061 (`validation_examples/test_published_v061_cantilever_wall.py`);
each reading has a hand check in
`retaining_walls/tests/test_gm288_printed_check.py`.

## Trial proportions (fixed)

- **Print.** NHI Fig. 10-16(a) (p. 10-38, after Teng 1962): stem top 200 mm
  minimum (300 mm preferable), stem base H/12 to H/10, base thickness H/12
  to H/10, toe H/10 to H/8, B 2/5 H to 2/3 H. UFC App. B-1.3 (p. 600): B 50
  to 70 % of H.
- **Module before.** Base thickness 0.60 m whatever the height, toe 0.1 B,
  stem base 0.08 H + 0.2 m: no printed source.
- **Now.** Base thickness H/10, toe H/10, stem base H/10 (at least the top
  thickness); B = 0.6 H is unchanged (inside both printed ranges). Each
  left-out part is listed in `defaults_used` as a trial proportion.

## The stem's batter and the soil over the wall (fixed; GM-235)

- **Print.** UFC Fig. 4-27 (p. 220): "Weight includes ... weight of soil
  above footing for cantilever and counterfort walls"; App. B-1.3 (p. 599)
  counts W_ff, the soil over the toe. UFC Fig. B-2 (p. 598) and NHI Fig.
  10-16(a) and Example 10-3 (p. 10-52) draw the front face battered.
- **Module before.** The back face was always the battered one, and the
  backfill over it was left out (conservative, GM-235); the ground in front
  was always at the top of the base, so no soil over the toe.
- **Now.** `stem_batter` "back" (default; the soil over the battered back
  counts, at its centroid) or "front" (the two examples' wall);
  `embedment_depth` Df, the depth of the underside of the base below the
  ground in front (default the base thickness): the soil over the toe,
  (Df - t) x toe at gamma_foundation, counts as resisting weight. The
  section figure draws both.

## Active thrust

- **Rankine.** UFC Eq. 4-6 (p. 178), NHI Eq. 10-10 (p. 10-14); sloped
  Rankine with the thrust parallel to the backfill. The module matches.
- **Coulomb's wall friction (fixed).** NHI p. 10-20: for cantilever walls
  the pressures act "within the retained soils along a vertical plane
  passing through the heel ... there is soil-to-soil contact ... it is
  generally conservative to assume that the earth pressure is parallel to
  the slope of the backfill"; Example 10-3 (p. 10-53) takes Coulomb's Ka at
  delta = beta = 10 deg. The module took delta = 2/3 phi on that plane,
  which no reference prints for it (a 6 m wall, phi 30, level: FOS_sliding
  1.604 -> 1.285). Now delta = beta by default, `wall_friction_deg` to give
  it; refused with Rankine. UFC Eq. 4-10 is the Ka.
- **The tension zone (fixed).** UFC Fig. 4-5 Note A (p. 183): "In no case
  should negative pressure be considered"; NHI p. 10-15: the pressure within
  the depth of the tension crack is taken as zero. The module subtracted
  2c sqrt(Ka) over the whole height, a pull on the wall near the top
  (unconservative: c' 8 kPa on a 5 m wall, Pa 40.4 -> 47.5 kN/m). Now
  Pa = 0.5 Ka gamma (H - z0)^2 below z0, with a surcharge offsetting the
  cohesion first. NHI p. 10-15 also allows a minimum active pressure of
  5z kN/m2 for c' > 5 kPa; the result's choices give it, and the thrust
  with c neglected, as the alternatives. Not built as an input.
- **Concrete (fixed).** 24 kN/m3, which no reference read prints, is now
  23.6 (150 pcf: UFC App. B-1.3 p. 599; NHI Example 10-3 p. 10-53).

## Sliding

- **Base resistance.** GEC-6 Eqs. 5-38 to 5-40 (pp. 99-100) and UFC Fig.
  4-27 T = (W + Pv) tan delta + Ca Be with Table 4-1: matched since GM-212.
- **Required FS (fixed).** UFC Fig. 4-27 and Sec. 4-5.1.3 (p. 223): 1.5,
  "If passive resistance is included, the minimum FSL should be increased
  to 2.0". The module asked 1.5 in every case. Now 2.0 by default with
  `include_passive`, and the choices say so.
- **Passive resistance in front (fixed).** It acted on the base thickness
  only; now over the embedment Df, with a key from Df to Df + key depth
  (UFC Fig. 4-29, p. 224). Under the undrained rule it counts cu, Kp
  sigma_v + 2 cu sqrt(Kp) (UFC Eq. 4-7, p. 178, and p. 181 for total stress;
  GEC-6 Eq. 5-41 counts Pp in front of a key in clay): 60 kPa clay, 0.5 m
  base, Pp 3.4 -> 62.4 kN/m. Under the drained rules c is not counted in it
  (UFC p. 179). NHI p. 10-40: passive resistance "should not be used in the
  analyses unless the wall extends well below the depth of frost
  penetration, scour or other types of disturbance"; `include_passive`
  stays off by default, and Example 10-3 neglects it.
- **GEC-6 Eq. 5-41 (noted).** ca (B - a1b) + cu a1b + Pp, a1b from the toe
  to the key's front face. The module takes a1b = 0 (ca B + Pp): the same
  with ca = cu, the default, and smaller where base_adhesion is below cu.
  The calc package's key position is for the figure only.
- **EN 1997-1 6.5.3(12)P (added, option; GM-235).** Where water or air can
  reach the interface of an undrained clay, Rd <= 0.4 Vd (Eq. 6.5, p. 65).
  `interface_gap_possible` applies it under the undrained rule; refused with
  the others. GEC-6 and UFC print no such limit.

## Overturning and eccentricity

- **Required FOT (fixed).** NHI Sec. 10.5.5 (p. 10-44): the eccentricity
  criterion "essentially requires that the safety factor ... is
  approximately of 2.0 for soils and 1.5 for rocks"; GEC-6 Sec. 7.3
  (p. 124): about 2.0 for permanent dead loads; UFC Fig. 4-27 and Sec.
  4-5.1.2 (p. 223): 1.5. The module's 2.0 stays the default on soil; 1.5
  with `founded_on_rock`. The choices quote UFC's 1.5.
- **FOT's form (noted).** UFC Fig. 4-27 prints F_OT = W a / (P_H b - P_V f):
  the thrust's vertical component taken off the overturning moment. The
  module counts P_V f in the resisting moment, F_OT = (W a + P_V f) /
  (P_H b). UFC B-1.8 has P_V = 0 and does not tell them apart; with P_V > 0
  the module's is the smaller FOT. The eccentricity, the criterion GEC-6
  Sec. 6.4.1 (p. 119) makes the overturning check ("separate calculation
  for stability with respect to overturning need not be performed"), is the
  same either way. Kept.
- **Eccentricity limit (fixed).** e <= B/6 on soil, B/4 (the middle half) on
  rock: GEC-6 Sec. 6.4.1 (p. 119), UFC Sec. 4-5.1.2 (p. 223), NHI Fig. 10-17
  (p. 10-39). The module had B/6 only. Now `founded_on_rock` gives B/4.

## Bearing pressure (fixed)

- **Print.** On soil the uniform pressure over the effective width, q_eq =
  V / (B - 2e): GEC-6 Eq. 6-6 (p. 118) and Sec. 6.4.2 (p. 119); NHI Fig.
  10-17, "Use uniform stress, qeq, for soils"; UFC Eq. 4-27 and Sec. 4-5.1.4
  (p. 224), for soil that is "relatively ductile", and App. B-1.6 (p. 602),
  q = R/(2 x0). On rock and sensitive clays the trapezoid's q_max (UFC Sec.
  4-5.1.4; NHI Fig. 10-17; Example 10-3 divides q_ult by q_max).
- **Module before.** q_max of the trapezoid against q_allowable in every
  case, which no reference asks on soil (conservative).
- **Now.** `bearing_pressure` "uniform" (default on soil) or "linear"
  (default with `founded_on_rock`); the result gives q_bearing and the
  choices the other pressure's FOS_bearing. q_toe and q_heel are still
  reported.

## Water (not modelled; follow-up)

The module has no water table: the backfill is taken as drained, as NHI
Sec. 10.5.3 and UFC App. B-1.1 assume for their walls. A water table behind
the wall (buoyant unit weight, the hydrostatic pressure, uplift on the
base) is UFC Sec. 4-4.1 and NHI Sec. 10.5.3; not built. The tool's brief now
says the backfill is taken as drained.

## GM-32

- The sliding display's missing Pp with `include_passive` was fixed by
  GM-103; the MSE parts by GM-157 and the GEC-11 built-in reinforcements.
- A descending backfill (backfill_slope < 0) was analysed as level without a
  word. Now refused with the reason: neither Rankine's sloped Ka nor the
  wedge over the heel is for a surface falling from the stem.

## Validation (V-061)

- **UFC App. B-1** (pp. 596-606): R = 2540 B - 4560 at every B, Ph 6,795
  lb/ft, FSL = (1397 B - 2508)/6795, FOT = (1270 B^2 - 4840)/44145, the
  middle third at B = 9.11 ft and the bearing resultant at B = 7.5 ft (x0
  1.805 ft, theta 23.686 deg) to the printed digit. The example rounds Ka
  to 0.25 (phi' set to 36.87 deg here). Table B-4's x0/B 0.35 is the
  resultant without the surcharge (B-1.8's x0 3.36 ft), not B-1.6's.
- **NHI Example 10-3** (pp. 10-52 to 10-56): Ka 0.3495 (printed 0.35), Pa
  within 0.15 %, W and Mw within 0.01 %, FSs 1.72, d 5.55 ft, e 1.0 ft,
  q_max / q_min within 0.5 %, FS bearing 6.04 (printed 6.03) with
  `bearing_pressure="linear"`.

## What moved (15 fixed walls and the owner's 2026-07-14 wall, before -> after)

| Wall | Change |
|---|---|
| 6 m, auto-sized, phi 30 | FOS_sliding 1.227 -> 1.200 (proportions, concrete) |
| 6 m, phi 30, Coulomb | FOS_sliding 1.604 -> 1.285, now fails (delta = beta = 0) |
| 5 m, c' 8 kPa | Pa 40.4 -> 47.5 kN/m; FOS_sliding 2.019 -> 1.755 (tension zone) |
| 5 m, passive, drained | passes 1.546 at 1.5 -> fails at 2.0 required |
| 5 m, passive, cu 60 kPa | Pp 3.4 -> 62.4 kN/m; FOS_sliding 2.846 -> 3.632 |
| 6 m tapered, q_allowable 250 | FOS_bearing 1.99 -> 2.24 (uniform q_eq) |
| 5 m, two surcharges | FOS_bearing 1.47 -> 1.80 (uniform q_eq) |
| Owner wall, Coulomb, delta_b 22 | FOS_sliding 1.339 -> 0.945 (delta = beta = 0 on the plane, concrete 23.6, tension zone) |
| Owner wall, Rankine, delta_b 22 | FOS_sliding 0.978 -> 0.945 (concrete 23.6, tension zone) |
| Owner wall, phi_foundation 22, Coulomb | FOS_sliding 0.566 -> 0.392 (delta = beta = 0, concrete 23.6) |
| Owner wall, Coulomb | FOT 3.03 -> 1.996, now fails the 2.0 required (no vertical thrust component with delta = beta = 0, concrete 23.6) |

Over nine checked walls the switch to q_eq alone raised FOS_bearing 15 to
33 %; with the module's other changes the total moves were +8 to +43 %.
On soil, q_eq raises FOS_bearing over the old q_toe for the same loads by
(1 + 6e/B)(1 - 2e/B) - 1: nothing at e = 0, a third at e = B/6 and beyond.
q_allowable is now the allowable pressure for B' = B - 2e (GEC-6 Sec.
6.4.2, p. 119), and the tool says so.
