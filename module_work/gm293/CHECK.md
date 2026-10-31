# GM-293: seismic_geotech against the print we hold

Checked 2026-10-31 against the PDFs in `geotech-references/docs`, page by
page: FHWA GEC-12 Vol I (FHWA-NHI-16-009, 2016) Sec. 7.4.2.1 (site class,
site factors, spectrum) and Fig. 7-78 (residual strength); FHWA GEC-13 Vol
I (FHWA-NHI-16-027, 2017) Eq. 5-9 and its r_d; FHWA GEC-5 (Fig. 5-19, the
SPT triggering chart, and Ch. 8, dynamic properties); FHWA GEC-7 p. 93
(N-bar); UFC 3-220-20 (16 January 2025) Sec. 4-4.4.1 and Fig. 4-8; FHWA
GEC-4 (FHWA-IF-99-015) Eqs. 40 to 46; FHWA NHI-06-089 p. 10-17; FHWA GEC-11
Vol II (FHWA NHI-10-025) Example E7. Covers every method the module
offers, its calc steps, the seismic_geotech tool text, the liquefaction
tool's `method="nceer2001"` route (it calls `evaluate_liquefaction`, so the
r_d fix reaches it) and the calc package's seismic numbers.

The AASHTO LRFD site-factor tables are not held; GEC-12 reprints them
(Tables 7-26 to 7-29, "after AASHTO 2014"), and the module is checked
against that reprint. ASCE 7-22 and FEMA P-2082 are held but print a
different site-class table (ASCE 7-22 rounds 1,000 psf to 50 kPa, for
one); the module follows AASHTO and was not moved toward them.

## What changed

### Liquefaction triggering, NCEER / Youd et al. (2001)

| Item | Print | Before | Now |
|---|---|---|---|
| CSR | GEC-13 Vol I Eq. 5-9 (p. 5-40): CSR = 0.65 (a_max/g)(σ_v/σ'_v) r_d | same | same, cited |
| r_d | GEC-13 pp. 5-40, 5-41, "for preliminary designs", coefficients "developed for metric units": 1.0 - 0.00765 z (z ≤ 30 ft), 1.174 - 0.0267 z (30 < z ≤ 75 ft), 0.744 - 0.008 z (75 < z ≤ 100 ft), 0.5 (printed "for z < 100 ft", a slip: the other three lines cover the depths above) | limits 9.15 and 23 m; the third line ran on below 30.48 m with no floor (0.464 at 35 m, 0.424 at 40 m) | limits 9.144, 22.86 and 30.48 m; 0.5 below |
| MSF | 10^2.24 / M^2.56 is printed on no held page | Youd et al. (2001) Eq. 20 | unchanged; not checked |
| CRR curve | GEC-5 Fig. 5-19 (p. 5-32) reprints the NCEER SPT chart (clean sand, 15 % and 35 % fines curves) | Youd et al. Eq. 2 | unchanged: the clean-sand curve sits within 0.3 to 0.6 blows of the figure's and the 35 % curve on it (see Noted, not fixed for 15 %) |
| K_σ | Youd et al. (2001), not held | not applied; the choices say so | unchanged |
| Tool text | | Liao & Whitman (1986), limits 9.15 and 23 m, two lines | GEC-13 Eq. 5-9 and its four r_d lines in the steps and choices |

### Residual strength

| Item | Print | Before | Now |
|---|---|---|---|
| Idriss & Boulanger (2007) | GEC-12 Vol I Fig. 7-78 (p. 383): two recommended curves, "void redistribution effects could be significant" and "... expected to be negligible"; s_r = (s_r/σ'_vo) σ'_vo (Table 7-31, Eq. 7-80, p. 384) | exp(N/16 + (N/21.2)^3 - 3.0) capped at 0.6, printed on no held page and on neither curve | both curves read off the figure (`trace_fig7_78.py`; the readings drawn back over the figure sit on the curves); `void_redistribution` "significant" (default) or "negligible"; refused past the chart's end (N 29.07 and 17.08) and below N 0 |
| Seed & Harder (1990) | not held | lower-bound points | unchanged; not checked |
| Tool | | offered `olson_stark` (which the module refused) and not `idriss_boulanger` | offers `seed_harder` and `idriss_boulanger`, and `void_redistribution` for the second |

Fig. 7-78 read: the figure is an embedded image (826 x 556 px); axes from
its own grid lines (1 px = 0.046 blows, 0.0009 in the ratio). The two
curves agree up to N 9, where the negligible curve leaves (a spur marks
it). Each curve's label hides part of it: the significant curve from N
18.9 to 24.4 (taken as a straight line, up to 0.01 high there), the
negligible curve from 0.265 to 0.342 in the ratio.

### Site class and site factors (GEC-12 Vol I Sec. 7.4.2.1)

| Item | Print | Before | Now |
|---|---|---|---|
| Class boundaries | Table 7-26 (p. 379), in ft/s and ksf: 5,000, 2,500, 1,200, 600 ft/s; N 50, 15; s_u 2.0, 1.0 ksf | 1500, 760, 360, 180 m/s; 100, 50 kPa (the SI rounding of another code), so a site between took the stiffer class | 1524, 762, 365.76, 182.88 m/s; 95.76, 47.88 kPa |
| Averaging | upper 100 ft (p. 378) | same | same, cited |
| N in N-bar | GEC-7 p. 93: field N, "not to exceed 100 bpf" | uncorrected N, no limit | field N, 100 at most; the tool text says field N |
| Class E soft clay, Class F | Table 7-26's notes | not mentioned | stated in the steps as not checked by the tool |
| Fpga | Table 7-27 (p. 380), against PGA | against PGA, or against Ss when no PGA was given (the Fa column read for Fpga, printed nowhere) | against PGA only; no PGA, no Fpga (None) |
| Fa, Fv | Tables 7-28, 7-29 (p. 380); straight-line interpolation "for intermediate values in any of the three tables" (p. 379) | same values | same; all 75 cells pinned |
| As | Eq. 7-78 (p. 381): As = Fpga PGA | not reported | reported (`As_g`) with the PGA (`PGA_g`) |
| SDS, SD1 | Eq. 7-79; p. 382 | same | same, cited |
| Spectrum below T0 | Fig. 7-77, Eq. 7-77: from As at T = 0 to SDS at T0 | ASCE 7 form SDS (0.4 + 0.6 T/T0), starting at 0.4 Fpga Ss | GEC-12's line from As; without PGA the line below T0 is not drawn |
| Tool text | | "ASCE 7 site classification" | AASHTO (GEC-12 Tables 7-26 to 7-29); pga gives Fpga and As |

### Mononobe-Okabe

| Item | Print | Before | Now |
|---|---|---|---|
| KAE | UFC 3-220-20 Eq. 4-15 (p. 214); GEC-4 Eq. 42 (p. 114) | same values; cited AASHTO, GEC-3, Kramer (not held) | unchanged values (pinned against Eq. 4-15 written out, four cases, 1e-12); cited to UFC and GEC-4 |
| Sign of the wall batter | UFC Fig. 4-8: θ positive the other way; UFC's β is the backfill slope | not stated | the module's beta = -θ (UFC), = -β (GEC-4); i = UFC's β; in the tool text |
| ΔPAE at 0.6 H | UFC pp. 215-216 (Seed & Whitman 1970) | same | same, cited |
| Fallback where M-O has no solution | UFC Eq. 4-17, ΔPAE = (3/8) kh γ H², printed for a vertical wall, level backfill, no wall friction | Ka + 0.75 kh, uncited | unchanged; cited, and the result says it is used outside its printed range |
| KPE | GEC-4 Eq. 44 (p. 114); its Eq. 45 prints "1 +" in the bracket, which gives the active coefficient at kh = 0 (a slip; UFC Fig. 4-8's static passive form has "1 -") | same values | unchanged values (pinned against Eq. 44 with "1 -", three cases) |
| Passive wall friction | NHI-06-089 p. 10-17: φ/2 the maximum for passive (2/3 φ for active); UFC Fig. 4-8 (p. 187): planar passive "unconservative for δ/φ' > 0.4 and, therefore, should not be used" | the active δ (default 2/3 φ) also used for KPE | own input `delta_passive`, default φ/2; a warning above 0.4 φ naming UFC Fig. 4-8 and the log-spiral alternative (UFC pp. 189-190); the choice offers 0.4 φ |

### Dynamic properties (GEC-5 Ch. 8)

| Item | Print | Before | Now |
|---|---|---|---|
| Gmax from Vs | Eq. 8.1 | same | same, cited |
| CPT sand, Rix & Stokoe | Table 8-7 (p. 8-21): 1634 qc^0.25 σ'_v^0.375 (kPa) | same | same, cited |
| CPT clay, Mayne & Rix (1993) | Table 8-7: Gmax = 49.4 qt^0.695 e^-1.130 (MPa) | 2.78 qc^1.335 (kPa), printed on no held page, no void ratio | Table 8-7's form; `void_ratio` required |
| Hardin | Eq. 8.14 (p. 8-18), Hardin (1978): A = 625, F(e) = 1/(0.3 + 0.7 e²) (Eq. 8.8), OCR^k with k from PI (Table 8-4), n = 0.5 | A default 875 (Taborda et al. 2019, a dense marine sand; not held), no OCR, named Hardin & Black (1968) | A default 625; OCR and PI (k by Table 8-4, straight-line between rows); PI required when OCR ≠ 1; the result gives k; pa taken as 100 kPa |
| Andersen (2015), Ishibashi & Zhang (1993) | printed on no held page (GEC-5 p. 8-23 names Ishibashi & Zhang only) | | unchanged; not checked |

### Calc steps, calc package, review checklist

- Steps and the reference list cite GEC-12, UFC 3-220-20, GEC-13 and GEC-4
  as above (FEMA P-1050 and the AASHTO section numbers, not held, are
  gone); the class table shows the printed boundaries; the Fpga step and
  the As step appear only with a PGA; the K_PE step shows `delta_passive`.
- The calc package's seismic numbers add PGA and As.
- The site-class review checklist states the printed boundaries and that
  Fpga is read against PGA.

## Printed example pinned

V-062 (`validation_examples/test_published_v062_seismic_mse_e7.py`), FHWA
GEC-11 Vol II Example E7 (pp. E7-3 to E7-6): Site Class B, PGA 0.206 g,
S1 0.177 g: Fpga 1.00, Fv 1.0, kmax = Fpga PGA = 0.206 g; φ' 30°, δ 30°,
level backfill, vertical face, kh 0.206, kv 0: KAE 0.4782 (module
0.47823), PAE = 0.5 (0.4782)(125 pcf)(25.64 ft)² = 19.65 k/lft (module
19.649). Also through the tool. No held manual prints a worked CSR,
residual strength or Gmax example; those are pinned to the printed
equations, tables and figure readings (`seismic_geotech/tests/test_print_gm293.py`).

## What moved (made-up inputs)

| Case | Before | Now |
|---|---|---|
| r_d at 35 m | 0.464 | 0.5 |
| r_d at 22.9 m | 0.5626 | 0.5608 |
| layer at 35 m, (N1)60 18, FC 5, a_max 0.3, GWT 2 m, M 7.5 | CSR 0.1763, FS 1.088 | CSR 0.190, FS 1.010 |
| class, Vs30 181 / 362 / 1510 m/s | D / C / A | E / D / B |
| class, s_u-bar 97 / 48.5 kPa | D / E | C / D |
| N-bar, 20 m at N 40 over 10 m at N 500 | 57.7, Class C | 50.0, Class D |
| Fpga, Class D, Ss 1.0, S1 0.4, no PGA | 1.1 (Fa read against Ss) | none; with PGA 0.35, 1.15 and As 0.4025 (both before and now 1.15) |
| I&B s_r/σ'_vo at N 10 / 16 / 20 / 25 | 0.103 / 0.208 / 0.402 / 0.600 | 0.091 / 0.140 / 0.189 / 0.273 (significant) |
| I&B at N 16, negligible | not offered | 0.290 |
| KPE, φ 30°, kh 0.2, no wall friction given | 4.975 (δ 20°) | 4.129 (δ 15°), with the 0.4 φ warning |
| Gmax, CPT clay, qc 1 MPa | 28.1 MPa | needs e: 49.4 MPa at e 1.0, 31.2 MPa at e 1.5 |
| Gmax, Hardin, σ'_m 100 kPa, e 0.7 | 136.1 MPa (A 875) | 97.2 MPa (A 625) |

Calc package figure: the spectrum below T0 now starts at As (or is not
drawn without a PGA).

## Not checked (source not held)

- Youd et al. (2001): MSF, K_σ, the fines correction's equations (compared
  only with GEC-5 Fig. 5-19's curves); the CRR cap at (N1)60cs 30.
- Seed & Harder (1990) residual strength.
- AASHTO LRFD itself (checked through GEC-12's reprint).
- Andersen (2015) Gmax for clays; Ishibashi & Zhang (1993) curves.
- GEC-3 and Kramer (1996), formerly cited for M-O; replaced by UFC and
  GEC-4, which print it.

## Noted, not fixed

- GEC-5 Fig. 5-19's 15 % fines curve sits about 1.8 blows left of the
  module's (the module asks for more blows for the same CRR, the safe
  side); the clean-sand and 35 % curves agree.
- GEC-10 p. 9-42: FS ≤ 1.0 liquefies, and 1.0 to 1.5 may build excess pore
  pressure; the tool reports `liquefiable` at FS < 1.0 only.
- GEC-13: ground improvement targets CRR/CSR above 1.5, 1.3 at least; not
  in the tool.
- GEC-10 Eq. 9-13 (Kramer and Wang) is a second residual strength method;
  not added.
- GEC-7 Tables 4.8 and 4.9 print the FPGA column headers as Ss values (a
  print slip); GEC-12's Table 7-27 is the one used.
- M-O's static KA defaults to Rankine (GM-142 already states it).
- GEC-11 Vol I p. 7-4 puts PAE at h/2 for MSE walls and GEC-4 p. 115 at
  mid-height for anchored walls; the module puts ΔPAE at 0.6 H (UFC) and
  says so.
- GEC-5 Table 8-2 (Eq. 8.10, coarse-grained soils), Eq. 8.12 (K2,max,
  Table 8-3) and Eq. 8.13 (Menq) are not offered.
- The I&B readings carry the label gaps above (up to 0.01 high on the
  significant curve between N 18.9 and 24.4).

## Owner questions

1. Residual strength, Idriss & Boulanger: which of Fig. 7-78's curves is
   the default? Now "void redistribution effects could be significant"
   (the lower: 0.140 at N 16); "negligible" gives 0.290 there.
2. Passive wall friction for KPE: φ/2 (NHI-06-089 p. 10-17's maximum, the
   default now, which always draws the UFC 0.4 φ warning) or 0.4 φ (UFC
   Fig. 4-8's limit)? At φ 30°, kh 0.2: 4.129 against 3.731.

## Gates (targeted; at ≥1000 MB free, one at a time)

- seismic_geotech/tests, validation_examples and foundry_test_harness,
  `-k` seismic, site, liquefaction, M-O, Gmax, residual, V-062: 218
  passed. Two harness tests changed: the retired foundry agent takes no
  PGA, so it gives no Fpga (the workflow takes As from the module).
- test_csr_crr_check, test_liquefaction_router, test_liquefaction_water_table,
  test_liquefaction_to_downdrag, test_post_liquefaction_downdrag,
  test_reviewers: 181 passed.
- test_defaults_used, test_choices_stated_defaults, test_adapter_ergonomics
  (whole files): 967 passed, 14 skipped.
- calc_package/tests and the funhouse calc package tests, `-k` seismic,
  site, liquefaction, minimal, parity: 196 passed. test_calc_package_tool_parity's
  module-side M-O case now gives KPE with delta_passive = phi / 2.
- test_calc_package_adapter, _assumed_inputs, _derived_inputs,
  _figure_conventions, test_steps_from_a_minimal_analysis,
  downdrag/tests/test_post_liquefaction.py, geotech_common/tests/test_adapters.py
  (whole files): 340 passed.
- funhouse_agent/tests `-k` seismic, site class, residual, M-O, Gmax,
  liquefaction: 172 passed.
- webapp/tests/test_page_fingerprints.py and test_worked_examples: 53
  passed, 2 skipped (no re-record).
- test_citations: 5 passed. GEC-5 Eq. 8.1 (p. 8-3), Eq. 8.14 and Table 8-4
  (p. 8-18) are on the gap list: checked on the page, and the reference
  library's chapter 8 leaves them out.
- test_public_text on every changed file: clean.
- Not run: the full suites.
