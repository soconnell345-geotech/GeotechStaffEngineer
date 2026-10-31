# settlement against GEC-6 and UFC 3-220-20, as printed

BACKLOG GM-285, 2026-10-31. Read from the PDFs in `geotech-references/docs/`:
GEC-6 (FHWA-SA-02-054) Ch. 5 and Appendix B; UFC 3-220-20 (2025), which
sends settlement to DM 7.1, now UFS 3-220-10 (2026), Ch. 5; so UFS 3-220-10
Chs. 4 and 5 too; and FHWA NHI-06-088 Ch. 7 and NHI-06-089 Sec. 8.5, which
GEC-6 and the module lean on for Schmertmann, the time rate and secondary
compression. Page numbers are the printed ones (GEC-6: pdf index = page +
5 in the body, page B-n at n + 157; UFS 3-220-10: page + 27; NHI-06-089:
page 8-n at n + 33; NHI-06-088: page 7-31 at 440).

The methods the module offers: the stress below a footing by 2:1,
Boussinesq or Westergaard (rectangle centre or corner, strip, point load);
immediate settlement by the elastic form (Iw from GEC-6 Table 5-12),
Schmertmann's strain influence method, or Hough's C' method for granular
soil; primary consolidation (NC and OC); the time rate (U against T); and
secondary compression. `SettlementAnalysis` puts these together; the calc
steps print them.

UFC 3-220-20 itself sends settlement to "Chapter 5 of DM 7.1" (pp. 224,
279), names Schmertmann et al. (1978) for the time-dependent elastic
settlement of sands (p. 323) and prints an elastic form (Eq. 5-37, p. 325)
of the UFS mu0 mu1 kind (below). It has no settlement worked example.

## Stress below a footing

- **2:1 (GEC-6 Eq. 5-17, Fig. 5-12, pp. 78, 79).** delta_sigma = q B L / ((B + z)
  (L + z)); for a strip q B / (B + z) (GEC-6 Appendix B Example 4 works it
  so). The module matches (`approximate_2to1`, `approximate_2to1_strip`).
  NHI-06-089 Example 8-3 (p. 8-55): 130 kips on 10 ft x 10 ft at 15 ft,
  208 psf; the module gives 208.0. **Agrees**; pinned.
- **Boussinesq, point load (UFS 3-220-10 Table 4-2, p. 185).**
  sigma_z = 3 Q z^3 / (2 pi R^5). The module matches. **Agrees.**
- **Boussinesq, corner of a rectangle (UFS Table 4-2).** (q0 / 2 pi)
  [atan(x y / z R3) + (x y z / R3)(1/R1^2 + 1/R2^2)]. The module had
  Newmark's arrangement of the same solution; it now prints Table 4-2's
  form; the two agree to 7e-15 over a grid. **Agrees** (no number moves);
  the centre is four corners of B/2 x L/2. GEC-6 Fig. 5-11 is the chart of
  the same stress.
- **Boussinesq, strip (UFS Table 4-2).** (q0 / pi)[alpha + sin alpha cos
  (alpha + 2 delta)]; below the centre delta = -alpha/2, so (q0 / pi)(alpha
  + sin alpha). The module matches. **Agrees.**
- **Westergaard (UFS Fig. 4-10, p. 194).** The figure gives the influence
  factor I below the corner of a rectangle (nu = 0), sigma_z = I q0, m = x/z
  and n = y/z interchangeable, as curves of m from 0.1 to 10 and infinity
  against n from 0.1 to 10. The module had I = atan(m n / sqrt(1 + m^2 +
  n^2)) / 2 pi, which no reference in the library prints and which is about
  28 % below the figure (I(1, 1) 0.083 against 0.116 read; I(1, 10) 0.115
  against 0.151). **Disagrees; fixed**: the figure is digitised
  (`stress_distribution.WESTERGAARD_M`, `_WESTERGAARD_I`,
  `_WESTERGAARD_I_INF`) and read bilinearly in log m and log n
  (`westergaard_influence_factor`, the reading stated in
  `WESTERGAARD_READING` and in the stress choice's basis). The centre is
  four corners; the strip is the curve m = infinity.
  - *How it was read.* The figure was rendered from the PDF at high
    resolution and the curves traced by colour along each vertical grid
    line; the axes were calibrated on the printed grid (I = 0 to 0.25
    linear, n log). At n = 5, 6 and 8 the figure prints the curve labels
    over the curves; there each curve was read across the gap. Which traced
    line was which curve was checked against the closed form for nu = 0
    (Westergaard's own solution, not printed in the library and not used
    in the module); the readings sit within 0.0005 of it (0.001 at most).
    Beyond n = 10 the module runs toward the m = infinity curve in 10/n,
    with I = 0.25 when both are infinite; below 0.1 it scales to zero.
    These are module readings and say so.
  - Pinned: nine figure readings (I(0.1, 10) 0.0223, I(1, 1) 0.116,
    I(1, 10) 0.1514, I(2, 2) 0.1741, I(inf, 10) 0.2383, ...).

## Immediate settlement, elastic (GEC-6 Eq. 5-36, Table 5-12, pp. 96, 97)

- **Equation.** GEC-6 prints delta_v = Cd delta_sigma B (1 - nu^2) / (alpha_E
  E_IR) for a footing on rock (Eq. 5-36); NHI-06-089 Eq. 8-19 and Table
  8-13 (pp. 8-58, 8-59) print the same with the same table. The module's
  `elastic_settlement` is that form with the soil's Es. **Agrees** (the
  form); the docstring now says the print gives it for rock.
- **Table 5-12.** The module's `_CD_*` tables match every printed cell
  (pinned, GM-203). **Agrees.**
- **The default Iw in `SettlementAnalysis`.** Schleicher's closed form for
  the centre of a flexible rectangle (1.122 for a square, 1.53 at L/B = 2;
  printed 1.12 and 1.52), which no reference in the library prints.
  **Disagrees; fixed**: `elastic_influence_factor` is now Table 5-12's
  centre of a flexible footing (1.00 circle, 1.12 square, a rectangle in
  log(L/B) between rows). Moves an elastic result by at most 0.8 % (L/B =
  2: 1.532 to 1.52; a square 1.122 to 1.12).
- **GEC-6 Appendix B Examples 4 and 5 (rock), pp. B-70, B-89.** Cd 1.83
  (L/B 5, read in the average column), 159 kPa, B 10 m, nu 0.14, 0.53 x
  9.8e6 kPa: 0.55 mm (printed 0.5); Cd 1.78 (L/B 3, centre), 10,200 kPa,
  B 1 m, nu 0.2, 0.99 x 52e6 kPa: 0.34 mm (printed 0.3). **Agrees**;
  pinned.
- UFS 3-220-10 Eq. 5-5 (p. 229), s = q0 B mu0 mu1 / Es with Fig. 5-6
  (Christian and Carrier), and UFC 3-220-20 Eq. 5-37: another elastic form
  for soil, **not offered** (noted below).

## Immediate settlement, Schmertmann (NHI-06-089 Sec. 8.5.1.1, pp. 8-45 to 8-53; UFS 3-220-10 Eqs. 5-7 to 5-9, Fig. 5-7, pp. 232, 233)

GEC-6 gives no Schmertmann method (it refers to Munfakh 2000); NHI-06-089
and UFS print it.

- **Equations.** Si = C1 C2 delta_p sum(Iz H / XE) (NHI Eq. 8-16; UFS Eq.
  5-7 with Es), C1 = 1 - 0.5 (p_o / delta_p) >= 0.5 (Eq. 8-17, UFS 5-8),
  C2 = 1 + 0.2 log10(t / 0.1) (Eq. 8-18, UFS 5-9). The module matches;
  t = 0 means C2 = 1 (the print's t = 0.1 year, end of construction).
  **Agrees.**
- **Izp = 0.5 + 0.1 sqrt(delta_p / p_op)** with p_op the vertical effective
  stress at the depth of the peak (Fig. 8-21b). Two disagreements:
  - Without `gamma_soil` the module took p_op = q0 (the overburden at the
    base), printed nowhere, and Izp = 0.5 when q0 = 0. **Fixed**:
    `gamma_soil` is required (p_op = q0 + gamma_soil z_peak); the tool
    refuses without it.
  - The module held Izp to 1.0; no cap is printed. **Fixed**: removed.
- **The diagram (Fig. 8-21a, Table 8-11).** Axisymmetric (L/B = 1): Izb
  0.1, peak at B/2, to 2B; plane strain (L/B >= 10): 0.2, B, 4B; "use
  linear interpolation" between (Example 8-2; Table 8-11 tabulates it).
  UFS Fig. 5-7 prints the two cases. The module took the axisymmetric
  diagram for any L/B up to 10 and plane strain above. **Disagrees;
  fixed**: linear interpolation in L/B, B the lesser side. Table 8-11
  pinned at eight rows.
- **X factor (NHI Eq. 8-16)** for a modulus correlated with N or qc (1.25
  to 1.75): the module takes Es as given; the layer docstring says to give
  X E. **Agrees** (stated).
- **NHI-06-089 Example 8-2 (pp. 8-48 to 8-52).** B 6 ft, L 24 ft, delta_p
  1,655 psf, p_o 345 psf, p_op 840 psf, four layers with XE 142, 426, 426,
  1,159 tsf: Izb 0.133, D_IP 4 ft, D_I 16 ft, Izp 0.640, sum 0.1761 in/tsf,
  C1 0.896, Si 0.1305 in and 0.1566 in at 1 year; printed 0.133, 4, 16,
  0.64, 0.1760, 0.896, 0.130 and 0.156. **Agrees**; pinned (before the fix
  the module could not take the interpolated diagram, so it gave another
  answer).

## Immediate settlement, Hough (GEC-6 Eq. 5-24, Figs. 5-18, 5-19, p. 89)

- **Equation.** delta_H = H (1 / C') log10((sigma'_vo + delta_sigma) /
  sigma'_vo), the stress by 2:1 (as the Appendix B examples take it). The
  module matches. **Agrees.** C' from Fig. 5-19 with the corrected N' is
  the user's (stated).
- **A strip.** GEC-6 Example 4 works a strip with the strip's 2:1 (q B /
  (B + z)); the module took only rectangles. **Fixed**: `L = math.inf`
  gives the strip.
- **Appendix B.** Example 1 is V-022 (unchanged). Example 2 (pp. B-27 to
  B-29): Table B2-3's delta_sigma / q 0.64, 0.33 and layers 11.2 mm, 4.1 mm
  (printed 11.2, 4). Example 3 (pp. B-46 to B-51): every cell of Table B3-4
  (19 to 59 mm) within 1 mm (the example rounds delta_sigma / q to two
  figures and each layer to the mm); the worked layers 5, 3, 6, 3, 1, 1 mm.
  Example 4 (pp. B-68 to B-70): every cell of Table B4-3 (strip) within
  0.6 mm; the worked layers 16 and 4 mm. **Agree**; pinned. Table B2-4's
  D'Appolonia column is not a module method (noted).

## Primary consolidation (GEC-6 Eqs. 5-20, 5-22, pp. 82, 84; UFS 3-220-10 Eqs. 5-10 to 5-14, pp. 239 to 241)

- **Equations.** NC: H Cc / (1 + e0) log(sigma'_f / sigma'_v0); OC with
  sigma'_f <= sigma'_p: Cr; OC past sigma'_p: Cr log(sigma'_p / sigma'_v0)
  + Cc log(sigma'_f / sigma'_p). UFS writes the same in C_er = Cr / (1 +
  e0) and C_ec = Cc / (1 + e0). The module matches. **Agrees.**
- **"Normally consolidated" band.** The module took any OCR within 5 % of 1
  as normally consolidated, printed nowhere. **Fixed**: NC only when
  sigma'_p = sigma'_v0; the OC form runs into the NC one as sigma'_p falls
  to sigma'_v0, so a slightly overconsolidated layer settles a little less
  than before.
- **Underconsolidated (UFS Sec. 5-2.3, pp. 224, 225).** sigma'_p below
  sigma'_v0 means a layer still consolidating, whose sigma'_v0 must come
  from its measured or estimated pore pressures; the print gives no
  formula for it otherwise. The module runs it through the OC-past-
  sigma'_p branch (a negative recompression term), silently. **Fixed in
  part**: the analysis now warns, with the print's reason, and the layer
  says `is_underconsolidated`; the arithmetic is unchanged (a refusal
  stopped soil profiles whose sublayer centres sit a little below a
  layer's sigma'_p, e.g. 60 against 60.07 kPa, which the 5 % band had
  taken as NC; the warning now says so instead).
- **UFS Fig. 5-13 (p. 242):** 3,000 / 3,500 / 1,000 psf, H0 10 ft, C_er
  0.015, C_ec 0.2: 0.126 ft (printed 0.13 ft, 1.5 in). **NHI-06-089 Example
  8-3:** 0.092 ft, 1.10 in (printed 0.09 ft, 1.1 in). **Agree**; pinned.

## Time rate (UFS 3-220-10 Eq. 5-15, Fig. 5-16, p. 245; NHI-06-088 Eq. 7-8, Table 7-4, pp. 7-31, 7-32)

- **Equations.** T = cv t / Hdr^2. U(T): T <= 0.2, U = (2 / sqrt pi) sqrt
  T; T > 0.2, U = 1 - (8 / pi^2) exp(-pi^2 T / 4) (UFS Fig. 5-16). The
  module switched at U = 60 % (T = 0.283) and wrote the second as T =
  -0.9332 log(1 - U) - 0.0851 (the same curve, constants rounded).
  **Disagrees (the switch); fixed**: the print's T = 0.2; U moves by at
  most 0.0035 between T = 0.2 and 0.283.
- **Table 7-4 / Fig. 5-16's table** pinned at every row (U to 0.0015). Its
  U = 0.95, T = 1.163 is not on the printed approximation (which gives T =
  1.129); the module keeps the approximation (noted).
- **NHI-06-088 p. 7-38:** t90 = 0.848 x 10^2 / 0.2 = 424 days; module
  424.0. **Agrees**; pinned.

## Secondary compression (NHI-06-088 Eq. 7-10, p. 7-39; UFS 3-220-10 Eq. 5-18, Fig. 5-24, pp. 254, 255)

- **Equation.** S_s = C_alpha H / (1 + e0) log(t2 / t1) (NHI Eq. 7-10; UFS
  Eq. 5-18 with e0 the initial void ratio). The module matches.
  **Agrees.** `secondary_settlement` defaulted e0 to 1.0, printed nowhere.
  **Fixed**: e0 required (the analysis already passes each layer's e0).
- **t1.** NHI-06-088 Eq. 7-10: "t1 = time when approximately 90 percent
  of primary compression has occurred for the actual clay layer ... as
  determined from Equation 7-8"; the module took t95 from cv.
  **Disagrees; fixed**: t90. Without cv, t1 = 1 year (stated as an
  assumed default, as before).
- **t2.** The print's t2 is "the service life of the structure or any
  other time of interest" (UFS: t after loading); the module's t2 = t1 +
  `t_secondary` (the time after primary). A stated input convention, not
  changed (noted).
- **UFS Fig. 5-24 (p. 255):** C_alpha 0.16, e0 6, H0 10 ft, 1 to 50 years:
  0.388 ft (printed 0.4 ft, 5 in). **Agrees**; pinned.

## Values that move (old -> new)

Every settlement function's inputs and results were recorded over the
settlement, calc_package, ground_improvement and validation tests, the
settlement adapter tests and the cross-checks that call the module, before
and after (4,102 inputs in common; 267 moved, the rest unchanged, and the
moves below explain every one; tests whose inputs had to change are paired
by order). By cause:

- **Westergaard stress: +27 % to +46 %** (e.g. 100 kPa, 2 m strip, z 3 m:
  20.48 -> 28.15 kPa; B = L = 2 m, z 3 m, centre: 6.38 -> 11.72 kPa). The
  old closed form against UFS Fig. 4-10.
- **Secondary compression with cv: +7.7 % to +12.4 %** (e.g. 15.5 -> 16.9
  mm): t1 = t90 (NHI-06-088 Eq. 7-10) for t95, a shorter t1 and a longer
  log(t2 / t1). Without cv (t1 = 1 year) nothing moves.
- **Schmertmann, 1 < L/B < 10: up** (L/B = 1.5: 12.53 -> 13.12 mm, +4.7 %;
  Example 8-2's L/B = 4 takes the interpolated diagram the print works).
  Schmertmann without `gamma_soil` now stops with the reason (it took
  sigma'_vp = q0, so a footing at the surface had Izp = 0.5: the adapter
  probe, 164 kPa on 2.44 m, gave 26.2 mm that way and 39.7 mm with gamma
  18 kN/m3); with it, a square or a strip is unchanged unless Izp was
  over 1.0.
- **Elastic, default Iw: -0.2 % for a square** (1.1222 -> 1.12; 17.90 ->
  17.86 mm), at most 0.8 % for a rectangle.
- **Time rate: U moves by at most 0.35 points (0.58 %) between T = 0.2
  and 0.283** (T = 0.2827: 60.0 -> 59.65 %), and the settlement-time
  curve and the ground_improvement surcharge time curve with it (at most
  0.55 %); t90 by 0.002 % (the rounded constants 0.9332 and 0.0851 for
  the exact 4 / pi^2 and ln(pi^2 / 8)).
- **Consolidation: a layer with sigma'_p within 5 % of sigma'_v0** now
  takes the OC form above it (a little less settlement) and the as-given
  form below it, with the underconsolidated warning (a little more); none
  in the recorded tests had one (the soil-profile test builds layers
  with 60 against 60.07 kPa but computes no settlement).

## Noted, not fixed

- UFS 3-220-10 Eq. 5-5 / UFC 3-220-20 Eq. 5-37 (the mu0 mu1 elastic form
  for soil), GEC-6 Eqs. 5-21 and 5-23 (the mv and C_ce forms of
  consolidation), Leonards' alpha for the over-consolidated case (UFS Fig.
  5-15), the D'Appolonia method (GEC-6 Table B2-4) and an elastic-chart
  stress option for Hough are not module methods. None changes a saved
  report's output; each would be a new option.
- The secondary t2 convention (above).
- The U = 0.95 row of Fig. 5-16's table (above).
- The 1-year t1 without cv, an assumed default stated in the result.
- The Liao-Whitman N' correction for Hough's C' is the user's.
- An underconsolidated layer is computed as given, with a warning (above);
  the print's way is a measured sigma'_v0, which is the user's input.
