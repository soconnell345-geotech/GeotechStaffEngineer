# Validation-Problem Inventory

Published worked examples with complete numeric inputs AND published numeric answers,
collected for implementation as correctness checks against the analysis modules.

Sources scanned: local geotech-references PDFs (`geotech-references/docs/`) via PyMuPDF text
extraction + page-image reads for figure data, plus public Itasca FLAC verification docs.
Raw page-text dumps and rendered figure PNGs used during extraction are kept in
`validation_examples/extracts/` (file names referenced per entry). Helper scripts:
`dump_pages.py` (page-range text dump), `find_kw.py` (regex page search).

Already-validated problems deliberately excluded: Fredlund-Krahn 1977, ACADS 1(a),
Griffiths-Lane 1999, Prandtl Nc=5.14 strip on clay, Duncan 2000 reliability.

Units: entries keep the source's units (mostly US customary from FHWA manuals). Our modules
are SI — implementers must convert (1 ft = 0.3048 m, 1 kip = 4.448 kN, 1 ksf = 47.88 kPa,
1 pcf = 0.157087 kN/m3, 1 kip/ft = 14.59 kN/m, 1 kip-ft = 1.356 kN-m).

---

## V-001 GEC-12 North Abutment — Nordlund driven H-pile capacity vs depth
- Source: GEC-12 Vol 3 (FHWA-NHI-16-064), Appendix D, Block 10 North Abutment, Table D-6 + Figure D-7 (PDF pp. 56-63; extracts: `g12_north_abut.txt`, `g12v3_p60.png`)
- Target: `axial_pile` (Nordlund method, cohesionless profile, compression)
- Inputs:
  - Pile: HP 12x74 (A = 21.8 in2; box perimeter/width per HP12x74: d = 12.13 in, bf = 12.22 in), driven from bottom of footing 5 ft below ground surface. Depths below are referenced to bottom of footing.
  - Water table: 15 ft below ground surface.
  - Profile (depths below ground surface): Layer 1, 0-25 ft, loose silty fine sand (SM), gamma = 105 pcf (gamma' = 42.6 pcf below WT), phi' = 33 deg; Layer 2, 25-50 ft, medium dense coarse sand little silt (SP), gamma_sat = 112 pcf (gamma' = 49.6), phi' = 36 deg; Layer 3, 50-97 ft, dense gravel w/ sand (GW), gamma_sat = 125 pcf (gamma' = 62.6), phi' = 36 deg for shaft / 40 deg for toe (design limit); limestone bedrock below 97 ft.
  - Computed with DrivenPiles (FHWA Nordlund), different shaft/toe phi in Layer 3.
- Published answer (nominal resistances, kips, depth below footing bottom):
  - D = 35 ft: Rs = 137.5, Rt = 150.2, Rn = 287.7
  - D = 50 ft: Rs = 250.7, Rt = 428.1, Rn = 678.8
  - D = 60 ft: Rs = 344.1, Rt = 428.1, Rn = 772.2
  - D = 70 ft: Rs = 452.5, Rt = 428.1, Rn = 880.6
  - Toe resistance plateaus at 428.1 kips from ~49 ft (DrivenPiles limiting value in Layer 3).
  - Suggested tolerance: +/-15% on Rs and Rt at each depth (Nordlund chart-interpolation differences).
- Confidence in extraction: high (full table in text; profile read off rendered Figure D-7, cross-checked against the sigma'v table D-4)
- Notes: exercises Nordlund K-delta/CF charts, layered profile, water table handling, separate shaft/toe phi, toe limiting. H-pile perimeter/toe-area convention (box) matters.

## V-002 GEC-12 South Abutment — alpha-method H-pile in clay vs depth
- Source: GEC-12 Vol 3, Appendix D, Block 10 South Abutment, Tables D-96/97/98 + D-100 + Figure D-83 (PDF pp. 209-216; extracts: `g12_south.txt`, `g12v3_p212.png`)
- Target: `axial_pile` (Tomlinson/alpha total-stress method, cohesive profile)
- Inputs:
  - Pile: HP 12x74, bottom of pile cap 5 ft below ground surface (depths below referenced to cap bottom). Box behavior: toe = 9*su on box area (~1.03 ft2), shaft on box perimeter (~4.06 ft).
  - Profile (depth bgs): Layer 1, 0-25 ft, medium silty clay (CL), gamma = 110 pcf; su at sample depths 1/6/11/16/21 ft = 0.650/0.660/0.680/0.700/0.720 ksf. Layer 2, 25-45 ft, stiff silty clay (CL), gamma = 124 pcf; su at 26/31/36/41 ft = 1.79/1.83/1.93/2.00 ksf. Layer 3, 45-96 ft, very stiff silty clay (CL), gamma = 129 pcf; su at 46...96 ft (5-ft steps) = 3.11/3.19/3.30/3.36/3.39/3.50/3.55/3.58/3.60/3.65/3.70 ksf. Limestone bedrock below 96 ft (excluded — rock toe computed by a separate hard-rock procedure).
  - Each sample depth treated as its own su sub-layer (DrivenPiles alpha method).
- Published answer (nominal, kips, depth below cap bottom):
  - D = 40.01 ft: Rs = 165.28, Rt = 28.69, Rn = 193.98
  - D = 70 ft: Rs ~ 318.4, Rt = 32.75, Rn ~ 351.1
  - D = 80 ft: Rs = 369.3, Rt = 33.03, Rn = 402.33
  - D = 90 ft: Rs = 420.33, Rt = 33.68, Rn = 454.01
  - Suggested tolerance: +/-15% (alpha-curve interpolation differences).
- Confidence in extraction: high (su tables + resistance table verbatim; unit weights from rendered Figure D-83)
- Notes: exercises alpha-vs-su curve and 9*su toe for an H-pile in layered clay. Do NOT validate the 91.25-ft rock-toe value (1527 kips) against the alpha method.

## V-003 GEC-12 North Abutment — wave-equation drivability (Smith model)
- Source: GEC-12 Vol 3, Appendix D, D.10.3, Tables D-7/D-8/D-10 (PDF pp. 72-79; extract: `g12_north_drv_np.txt`)
- Target: `wave_equation` (drivability / bearing graph)
- Inputs:
  - Pile: HP 12x74 (A = 21.8 in2, steel, Fy = 50 ksi), driven through the V-001 profile.
  - Hammer: Delmag D36-52 single-acting diesel, ram 7.94 kips, rated energy 89.3 ft-kips, fuel setting 4; helmet weight and hammer-cushion = wave-equation (GRLWEAP) defaults.
  - Smith soil parameters (Table D-8, HP12x74 row): Layers 1-2 shaft quake 0.10 in, toe quake 0.20 in, shaft damping 0.10 (L1) / 0.05 (L2) s/ft, toe damping 0.15 s/ft, setup factor 1.2; Layer 3 shaft quake 0.10 in, toe quake 0.10 in, shaft damping 0.05, toe damping 0.15, setup 1.0. Gain/loss factor 0.833 on long-term resistance to get driving resistance.
  - Soil resistance distribution: Nordlund shaft/toe distribution of V-001 (Table D-6) converted to unit resistances, reduced by gain/loss to nominal DRIVING resistance.
- Published answer:
  - With D36-52 (fuel 4): practical refusal (120 blows/ft) reached at penetration 61 ft with nominal driving resistance Rndr = 746 kips; max compression driving stress 40.1 ksi.
  - With D30-52: 120 bpf at Rndr = 680 kips, depth 55 ft, stresses <= 38.5 ksi.
  - With D46-52: 120 bpf at Rndr = 867 kips, depth 72 ft, max stress 44 ksi.
  - Suggested tolerance: +/-25% on blow count at the stated resistance (or +/-10% on Rndr at 120 bpf) — diesel-hammer modeling differences dominate.
- Confidence in extraction: high for the published numbers; medium for reproducibility (requires a D36-52 hammer model and GRLWEAP-default helmet/cushion in our Smith implementation)
- Notes: exercises bearing graph + driving stress output. Treat as a trend/limit check rather than a tight numeric match.

## V-004 GEC-12 North Abutment — neutral plane location and drag force
- Source: GEC-12 Vol 3, Appendix D, D.17 / Figure D-33 (PDF pp. 111-115; extract: `g12_north_drv_np.txt` lines ~3296-3500)
- Target: `downdrag` (Fellenius neutral-plane method)
- Inputs:
  - HP 12x74 driven 60 ft below footing bottom in the V-001 profile (same Nordlund shaft/toe resistance distribution, Table D-6).
  - Sustained (Service I, no LL) pile-head load Q = 201 kips.
  - 100% toe mobilization (fully mobilized toe = 428.1 kips at 60 ft); negative skin friction fully mobilized above the neutral plane.
- Published answer:
  - Neutral plane at 54 ft below pile head (6 ft above the toe).
  - Maximum axial force in pile Qmax = 486 kips; drag force DF = Qmax - Q = 285 kips.
  - Suggested tolerance: +/-3 ft on NP depth, +/-10% on Qmax/DF.
- Confidence in extraction: high
- Notes: exercises the load-vs-depth / resistance-vs-depth crossing construction at 100% toe mobilization. Resistance distribution can be taken directly from the published Table D-6 values to decouple from the Nordlund calc (or chained with V-001).

## V-005 GEC-12 North Abutment — Meyerhof (1976) pile-group settlement
- Source: GEC-12 Vol 3, Appendix D, D.15, Table D-23 (PDF pp. 96-98; extract: `g12_north_drv_np.txt` lines ~1838-2035)
- Target: `pile_group` or `settlement` (Meyerhof SPT group-settlement method) — CHECK MODULE SUPPORT; formula is S(in) = 4*pf(ksf)*If*sqrt(B_ft)/N160
- Inputs: group B = 5 ft x Z = 41 ft (edge-to-edge), unfactored permanent load Q = 1540 kips → pf = 7.512 ksf; pile penetration 50 ft, embedment into bearing stratum DB = 5 ft → If = 1 - (2/3*DB)/(8B) = 0.92; N160 (avg within B below toe) = 59.
- Published answer: S = 1.04 in at 50 ft penetration. Table D-23 also gives: penetration 40 ft (N160 = 17): 2.63 in; 42.5 ft (N160 = 38): 1.11 in; 45.5 ft: 1.13 in; 55 ft: 0.95 in (config 1&2). Tolerance +/-5% (closed-form).
- Confidence in extraction: high (numbers verbatim)
- Notes: trivial closed-form — only include if a module exposes Meyerhof group settlement; otherwise drop or add to geotech_common checks.

## V-006 GEC-10 Appendix A — drilled shaft side resistance, beta (rational) method
- Source: GEC-10 (FHWA-NHI-18-024), Appendix A design example, Steps 11.5-11.6 (PDF pp. 591-595; extract: `g10_appA.txt`)
- Target: `drilled_shaft` (cohesionless side resistance, GEC-10 beta method)
- Inputs: shaft B = 8.0 ft; geomaterial Layer 3 (sand), layer thickness Delta-z = 20 ft; mean N60 = 30, mean (N1)60 = 21; mean sigma'v over layer = 2,266 psf (full-scour case); pa = 2,116 psf.
- Published answer (intermediate chain + result):
  - phi' = 27.5 + 9.2*log10[(N1)60] = 40 deg; delta = phi'.
  - sigma'p = pa*0.47*(N60)^0.6 = 7,654 psf; OCR = 7,654/4,645 = 1.65 (OCR uses no-scour sigma'v = 4,645 psf).
  - Ko = (1 - sin phi')*OCR^(sin phi') = 0.49; beta = Ko*tan(delta) = 0.41.
  - fSN = beta * sigma'v = 936 psf; RSN = pi*B*Delta-z*fSN = 470.7 kips (text uses 932 psf in the product → 470,684 lb).
  - Suggested tolerance: +/-5% on beta and fSN, +/-7% on RSN.
- Confidence in extraction: high
- Notes: exercises the GEC-10 (Chen & Kulhawy/O'Neill) rational beta chain incl. scour-modified stress. Watch the no-scour-vs-scour sigma'v subtlety in OCR.

## V-007 GEC-10 Appendix A — drilled shaft side resistance, alpha method (clay)
- Source: GEC-10, Appendix A, Step 11.5 item B (PDF pp. 593-594; extract: `g10_appA.txt`)
- Target: `drilled_shaft` (cohesive side resistance, GEC-10 alpha method)
- Inputs: shaft B = 8.0 ft; Layer 4 clay, Delta-z = 15 ft; mean su(UU) = 1,750 psf; mean sigma'vo = 2,114 psf; pa = 2,114 psf (as printed).
- Published answer:
  - UU→CIUC transform: su(UC)/su(CIUC) = 0.893 + 0.513*log10(su/sigma'vo) = 0.851 → su(CIUC) = 2,057 psf.
  - alpha = 0.30 + 0.17/(su(CIUC)/pa) = 0.47.
  - fSN = alpha*su = 976 psf; RSN = pi*B*Delta-z*fSN = 368.1 kips (text product uses 967 psf → 368,124 lb).
  - Suggested tolerance: +/-5% on alpha/fSN.
- Confidence in extraction: high
- Notes: tests whether our GEC-10 alpha implementation includes the su-test-mode transform; if module uses the plain AASHTO alpha (0.55 / reduced for su/pa>1.5), document deviation instead of "fixing" to match.

## V-008 GEC-10 Appendix A — drilled shaft base resistance in sand (0.60*N60)
- Source: GEC-10, Appendix A, Step 11.5 item C (PDF p. 594; extract: `g10_appA.txt`)
- Target: `drilled_shaft` (cohesionless base resistance)
- Inputs: B = 8.0 ft, tip in Layer 7 sand at 201 ft; average uncorrected N60 over 2B below tip = 41.
- Published answer: qBN = 0.60*N60 = 24.6 tsf = 49.2 ksf; RBN = (pi/4)*B^2*qBN = 2,473 kips. Whole-shaft check: sum(phi_i*R_i) = 1087 + 1182 + 1237 = 3,506 kips factored (phi = 0.55 beta side, 0.45 alpha side, 0.50 base). Tolerance +/-3% (closed-form).
- Confidence in extraction: high
- Notes: simple but pins the 0.60*N60 (<= 30 tsf) rule and area math; factored sum exercises resistance-factor plumbing if our module reports factored values.

## V-009 GEC-11 Example E4 — MSE wall external stability (sliding/eccentricity/bearing)
- Source: GEC-11 Vol 2 (FHWA-NHI-10-025), Appendix E, Example E4 (PDF pp. 293-301; extract: `g11_E4.txt`)
- Target: `retaining_walls` (MSE external stability, GEC-11/AASHTO LRFD)
- Inputs:
  - Geometry: exposed height He = 23.64 ft, embedment 2.0 ft → design H = 25.64 ft; reinforcement length L = 18 ft (0.7H); level backfill; segmental precast panels.
  - Reinforced fill: phi'r = 34 deg, gamma_r = 125 pcf. Retained fill: phi'f = 30 deg, gamma_f = 125 pcf (Kaf = 0.333). Foundation: phi'fd = 30 deg, gamma = 125 pcf.
  - Live-load surcharge heq = 2 ft of soil (q = 0.25 ksf). Bearing resistances (given): 10.50 ksf strength, 7.50 ksf service.
  - Load factors: EV 1.35/1.00, EH 1.50/0.90, LL 1.75 (Str I max/min); sliding phi = 1.0, bearing phi = 0.65.
- Published answer:
  - Unfactored: V1 = 57.69 k/ft, F1 = 13.68 k/ft, F2 = 2.13 k/ft; MV1 = 519.21, MF1 = 116.94, MF2 = 27.36 k-ft/ft.
  - Sliding: CDR = 1.85 (Str I max), 2.08 (min), critical max/min CDR = 1.37.
  - Eccentricity (Str I max): eL = 2.87 ft (limit L/4 = 4.50 ft), critical max/min eL = 3.87 ft.
  - Bearing (Str I max): eL = 2.60 ft, B' = 12.79 ft, sigma_v = 6.70 ksf, CDR = 1.57; critical combo sigma_v = 5.86 ksf, CDR = 1.79; Service I sigma_v = 4.66 ksf.
  - Suggested tolerance: +/-2% on forces/moments, +/-0.05 on CDRs.
- Confidence in extraction: high (all tables verbatim)
- Notes: directly exercises the GEC-11 LRFD external-stability bookkeeping incl. LL-on-resisting-side exclusions and max/min load-factor pairing.

## V-010 GEC-11 Example E4 — MSE internal stability (Kr, Tmax, pullout) for steel bar mats
- Source: GEC-11 Vol 2, Example E4 Steps 7.1-7.6, Table E4-7.4 (PDF pp. 302-310; extract: `g11_E4.txt`)
- Target: `retaining_walls` (MSE internal stability, inextensible reinforcement)
- Inputs: same wall as V-009. Steel bar mats (inextensible): Kr/Ka varies 2.5*Ka = 0.707 at Z = 0 to 1.2*Ka = 0.340 at Z = 20 ft (Ka = 0.283 from phi'r = 34); 10 reinforcement levels at Z = 1.87, 4.37, ..., 24.37 ft; panel width wp = 5 ft; Svt = 3.12/2.50.../2.52 ft; live load q = 0.25 ksf (gamma_EV = 1.35 on both soil and surcharge); pullout: F* from 20*(t/St) at Z=0 to 10*(t/St) at Z>=20 ft, t = 0.374 in (W11 transverse), St = 6/12/18 in by level, Le = L - 0.3H = 10.31 ft for Z < H/2 (longer below), unfactored sigma_v for pullout, phi_pullout = 0.90.
- Published answer (Strength I max, per 5-ft panel; Table E4-7.4):
  - Level 1 (Z=1.87): sigma_H = 0.40 ksf, Tmax = 6.25 k; Level 4 (Z=9.37): sigma_H = 1.02 ksf, Tmax = 12.77 k, F* = 0.955, Le = 10.31 ft, phi_p*Pr = 20.75 k/ft; Level 7 (Z=16.87): sigma_H = 1.26 ksf, Tmax = 15.71 k; Level 10 (Z=24.37): sigma_H = 1.51 ksf, Tmax = 19.05 k, Le = 17.24 ft.
  - Hand-calc illustration at Level 4: sigma_H-soil = 0.84 ksf, sigma_H-surcharge = 0.18 ksf, Pr = 23.06 k/ft (nominal).
  - Suggested tolerance: +/-3% on sigma_H/Tmax, +/-5% on Pr.
- Confidence in extraction: high
- Notes: pins the Kr interpolation, tributary-area Tmax, and bar-mat F* pullout. Note the source's own "1.7Ka" typo — numeric anchors (0.707 → 0.340) are the authoritative inputs.

## V-011 GEC-11 Example E7 — MSE seismic: Mononobe-Okabe KAE + sliding CDR
- Source: GEC-11 Vol 2, Example E7 (Example #4 wall under EQ), PDF pp. 355-361 (extracts: `g11_E10.txt`, `g11_E7b.txt`)
- Target: `seismic_geotech` (Mononobe-Okabe active thrust) + optionally `retaining_walls` seismic external
- Inputs: vertical wall (theta = 90 deg), level backfill (beta = 0), phi' = 30 deg, delta = 30 deg, kh = kmax = 0.206, kv = 0; retained-soil gamma_b = 125 pcf; height of pressure plane h = 25.64 ft. Site: PGA = 0.206 g, Fpga = 1.0; kav = 1.024*kmax = 0.211 g; reinforced mass W = 25.64*18*0.125 = 57.68 k/ft.
- Published answer:
  - KAE (total static+dynamic) = 0.4785; PAE = 0.5*gamma*h^2*KAE = 19.65 k/ft (inclined at delta = 30 deg).
  - PIR = 0.5*kav*W = 6.09 k/ft.
  - Sliding: THF = PAE*cos30 + PIR + 0.5*qLS*H*KAE = 17.02 + 6.09 + 1.53 = 24.64 k/ft; V = 57.68 + PAE*sin30 = 67.52 k/ft; R = V*tan30 = 38.98 k/ft; CDR = 1.58.
  - Suggested tolerance: +/-2% on KAE, +/-3% on PAE/CDR.
- Confidence in extraction: high
- Notes: clean M-O check with delta = phi (a stressing case for the M-O formula's delta handling); kv = 0. Our M-O battered-wall fix work makes this a good regression anchor.

## V-012 Caltrans T&S Example 7-1B — cantilever soldier pile wall (simplified method)
- Source: California Trenching & Shoring Manual (July 2025), Ch. 7, Example 7-1B (PDF pp. 133-139; extract: `cal_7_1B.txt`)
- Target: `soe` (cantilever soldier pile) or `sheet_pile` (cantilever, with effective-width factors)
- Inputs:
  - Retained height H = 15 ft; uniform soil gamma = 125 pcf, phi = 35 deg, c = 0; no water. Rankine: Ka = 0.271, Kp = 3.69; wall friction delta = 0.
  - Soldier piles W14x120 at s = 8 ft o.c., in 2-ft-diameter drilled holes; construction surcharge 72 psf over the retained height (driving only).
  - Active below dredge on hole width (2 ft, no arching); passive below dredge on arching-amplified width f*b = (0.08*35)*2 ft = 5.6 ft.
- Published answer:
  - Embedment (moment balance about toe, FS = 1.0): D0 = 12.27 ft; design D = 1.2*D0 = 14.73 ft.
  - Zero shear at Y = 6.0 ft below dredge; Mmax = 379.7 kip-ft (per pile); Vmax = 137.7 kips.
  - Comparison row (conventional/rigorous method): D = 13.53 ft, V = 91.28 kips, M = 379.9 kip-ft, deflection 1.73 in (simplified: 1.64 in).
  - Suggested tolerance: +/-5% on D0, +/-7% on Mmax.
- Confidence in extraction: high (all equations and the cubic D0^3 - 1.2133 D0^2 - 93.432 D0 - 518.75 = 0 printed)
- Notes: exercises soldier-pile effective-width/arching logic and the simplified (toe-moment) cantilever solution; the rigorous-method row doubles as a check for our full cantilever solver.

## V-013 Caltrans T&S Example 8-1 — single-anchor sheet pile wall
- Source: California Trenching & Shoring Manual, Ch. 8, Example 8-1 (PDF pp. 162-171; extracts: `cal_8_1.txt`, `cal_p162.png`)
- Target: `soe` (anchored wall, apparent/trapezoidal pressure) or `sheet_pile` (anchored)
- Inputs:
  - H = 25 ft excavation; single anchor 10 ft below top, inclined 15 deg, anchors at 10 ft horizontal spacing; PZ22 sheet pile (S = 18.10 in3/ft, I = 84.70 in4/ft); steel 42 ksi.
  - Sandy soil: gamma = 115 pcf, phi = 30 deg, c = 0, delta (wall friction) = 15 deg; no water.
  - Ka = 0.333 (Rankine); Kp = 4.7 (Caquot-Kerisel log-spiral 6.3 x R = 0.746 for delta/phi = -0.5).
  - Apparent diagram: total PT = 1.3*P with P = 0.5*gamma*H^2*Ka = 11,980 lb/ft → PT = 15,574 lb/ft; max ordinate sigma_a = PT/((2/3)*H) = 934.4 psf (trapezoid, top transition (2/3)*10 ft, bottom transition at dredge).
- Published answer:
  - Embedment D = 6.09 ft for FS = 1.3 (MR = 1.3*MD about anchor); D' = 4.89 ft for FS = 1.0.
  - Anchor: T1 = T1U + T1L = 6,228 + 8,026 = 14,254 lb/ft of wall; TH = 143.87 kips per anchor (10-ft spacing), T = TH/cos15 = 148.95 kips.
  - Max shear at anchor = 8,026 lb/ft; zero shear 9.69 ft below anchor; Mmax = 22,494 ft-lb/ft (at anchor level); fb = 14,913 psi (OK vs 25,000).
  - Deflection (info only, CT program): max 0.23 in.
  - Suggested tolerance: +/-5% on D and T, +/-10% on Mmax.
- Confidence in extraction: high (given-info figure read from rendered page image: H = 25 ft, anchor depth 10 ft, 15-deg incline, gamma = 115 pcf, phi = 30, delta = 15)
- Notes: exercises the FHWA-style single-anchor apparent diagram, log-spiral passive with FS on passive moments, and anchor-force back-out. Note source applies the 1.3 factor by scaling total active force.

## V-014 Caltrans T&S Example 10-2 — basal heave factor of safety
- Source: California Trenching & Shoring Manual, Ch. 10, Example 10-2 + Figure 10-17 (PDF pp. 229-230; extracts: `cal_10x.txt`, `cal_p230.png`)
- Target: `soe` (basal heave / bottom stability, Bjerrum-Eide Nc)
- Inputs: excavation H = 30 ft, width B = 15 ft, length L = 45 ft; surcharge q = 300 psf; clay c = 500 psf (phi = 0), gamma = 120 pcf.
- Published answer:
  - Nc (H/B = 2, square) = 8.5; corrected for L/B = 3: Nc = 7.6.
  - Resisting: qu*(0.7B) = (c*Nc)*(0.7*15) = 3.8 ksf * 10.5 ft ≈ 40.0 k/ft.
  - Driving: W + 0.7B*q - S = (10.5*30*0.120) + (0.7*15*0.3) - (0.5*30) = 37.8 + 3.15 - 15.0 = 26.0 k/ft (S = side shear c*H on the vertical plane).
  - FS = 40.0/26.0 = 1.54.
  - Suggested tolerance: +/-0.1 on FS (Nc chart read).
- Confidence in extraction: high (calculation figure read from rendered image)
- Notes: exercises rectangular-correction Nc and the side-shear term in the driving block. If our soe heave method omits the sidewall shear S or uses the Terzaghi 1/sqrt(2) width, document the convention difference.

## V-015 Caltrans T&S Examples 10-3/10-4 — Fellenius and Bishop FOS for a specified circle
- Source: California Trenching & Shoring Manual, Ch. 10, Examples 10-3 and 10-4 (PDF pp. 240-243; extracts: `cal_10x.txt`, `cal_p240.png`)
- Target: `slope_stability` (OMS/Fellenius and Simplified Bishop on a SPECIFIED circle)
- Inputs:
  - Soil: gamma = 115 pcf, phi' = 30 deg, c' = 200 psf, no groundwater.
  - Geometry (reconstructed from the slice table, verified to 1%): toe at (0,0); slope face rises at 4V:3H (53.13 deg, height/run = 1.333) to the crest; circle center directly above the toe at (0, 60 ft), R = 60 ft (passes through the toe, exits on the face at x ≈ 57.6 ft).
  - Source discretization: 6 slices, 10 ft wide; slice angles theta_i = asin(x_i/60) at x = 5..55; avg heights 6.46/18.09/27.88/35.40/39.69/37.31 ft; weights 7.43/20.81/32.06/40.71/45.64/42.91 k/ft; arc length L = 113.55 ft.
- Published answer:
  - Fellenius/OMS: FS = [c*L + tan(phi)*sum(W cos theta)] / sum(W sin theta) = (0.2*113.55 + 0.577*137.09)/116.49 = 0.87.
  - Simplified Bishop (same circle): converges to FS ≈ 0.90 (iterations shown: 1.10 @ FSa=1.5, 0.90 @ FSa=0.8).
  - Suggested tolerance: +/-0.05 (source uses only 6 hand slices; our finer slicing will shift slightly).
- Confidence in extraction: high for soil/circle/answers; medium for the slope-face reconstruction (derived, but reproduces all 6 published slice weights to <1%)
- Notes: specified-circle check (NOT a critical-surface search). New geometry distinct from the existing Fredlund-Krahn/ACADS/Griffiths set.

## V-016 GEC-4 Design Example 1 — two-tier anchored soldier-beam wall (apparent pressure, tributary method)
- Source: GEC-4 (FHWA-IF-99-015), Appendix A, Design Example 1 (PDF pp. 207-214; extract: `gec4_ex1.txt`)
- Target: `soe` (multi-anchor apparent envelope, tributary-area anchor loads)
- Inputs:
  - H = 10 m permanent anchored soldier beam wall; anchors at H1 = 2.5 m and 6.25 m below top (H2 = 3.75 m, H3 = 3.75 m); soldier-beam spacing 2.5 m; anchor inclination 15 deg.
  - Soil: medium dense silty sand, gamma = 18 kN/m3, phi' = 33 deg, no groundwater.
  - Traffic surcharge: qs = 0.6 m * 18 = 11 kPa → uniform ps = Ka*qs = 3.2 kPa over full height (Ka = tan^2(45 - 33/2) = 0.295).
  - Apparent envelope (FHWA trapezoid for sand, 2 anchors): pe = 0.65*Ka*gamma*H^2 / (H - H1/3 - H3/3).
- Published answer:
  - pe = 43.6 kN/m2.
  - TH1 = (2/3 H1 + H2/2)*pe + (H1 + H2/2)*ps = 168 kN/m; TH2 = (H2/2 + 23/48 H3)*pe + (H2/2 + H3/2)*ps = 172 kN/m.
  - M1 = (13/54)*H1^2*(pe + ps) = 76 kN-m/m; M2,3 = (1/10)*H2^2*(pe+ps) = 66 kN-m/m → Mmax = 76 kN-m/m.
  - Subgrade reaction R = (3/16)*H3*pe + (H3/2)*ps = 37 kN/m.
  - Anchor design loads: DL1 = TH1*2.5/cos15 = 435 kN, DL2 = 445 kN.
  - Suggested tolerance: +/-3%.
- Confidence in extraction: high (all formulas and numbers printed; already in SI)
- Notes: canonical FHWA anchored-wall worked example; exercises the 2-anchor trapezoid, tributary anchor loads, hinge-method moments, and inclination resolution.

## V-017 Micropile Manual Sample Problem No. 2 — laterally loaded micropile (LPILE benchmark)
- Source: FHWA NHI-05-039 Micropile Manual, Appendix E (PDF pp. 427-436; extract: `mp_sp2.txt` — includes the verbatim LPILE 4.0 input echo)
- Target: `lateral_pile` (p-y FD solver, Reese et al. 1974 sand, fixed head, with axial load)
- Inputs (from the LPILE input echo — complete):
  - Pile: length 12.19 m; head 0.305 m above ground (ground at -0.30 m from pile top); D = 0.19685 m; I = 3.58667e-5 m4; A = 0.008626 m2; E = 199,948 MPa (steel casing section properties; analyze as constant linear EI for the comparison).
  - Soil (Reese sand, static): Layer 1 from -0.305 to 3.048 m: phi = 32 deg, k = 24,430 kN/m3, gamma' = 18.84 kN/m3. Layer 2 from 3.048 to 12.192 m: phi = 30 deg, k = 16,287 kN/m3, gamma' = 17.64 kN/m3. c = 0 both.
  - Loading (Load Case 1, fixed head): V = 44.482 kN shear, slope = 0 at head, axial P = 1,423.4 kN (P-delta included). (Load Case 2 = free head: V = 44.482 kN, M = 0, same axial.)
- Published answer:
  - Fixed head (100% fixity): pile-head deflection = 3.3 mm; max bending moment = -37.3 kN-m (at head).
  - 50% fixity (rotational stiffness 88,964 kN-m/rad): deflection = 8.4 mm; max moment = +27.2 kN-m (max positive ~0.75-2.1 m depth).
  - Suggested tolerance: +/-15% on deflection, +/-10% on Mmax (LPILE 4 vs our FD solver; nonlinear-EI option was on in source but casing stays elastic at these loads).
- Confidence in extraction: high (input file echoed verbatim in the manual)
- Notes: best-quality lateral_pile benchmark in the local library: exact k, gamma', phi, BCs. Exercises Reese sand p-y, axial-load effect, head fixity variants.

## V-018 GEC-13 Example Problem 1 — rammed aggregate pier settlement (two-layer method)
- Source: GEC-13 Vol 1 (FHWA-HIF-24-016 era update), Ch. 5 Sec 4.3.1 (PDF pp. 372-373; extract: `g13_v1_ex.txt`)
- Target: `ground_improvement` (aggregate piers, Lawton/Fox two-layer settlement)
- Inputs:
  - Embankment: q = 125 pcf * 20 ft = 2,500 psf over a 15-ft soft clay layer on rock; clay gamma_sat = 120 pcf, water table at surface (po at mid-depth 7.5 ft = (120-62.4)*7.5 = 432 psf); Cc = 0.25, eo = 0.7.
  - RAP: diameter 2.75 ft (area 5.94 ft2), square(?) spacing s = 5 ft, de = 1.05*s = 5.25 ft → Ra = 5.94/(pi/4*5.25^2) = 0.27; stress concentration ns = 6; pier stiffness modulus kg = 65 pci; piers full-depth (no lower zone).
- Published answer:
  - Unimproved: S = Cc/(1+eo)*H*log((po+dq)/po) = 0.25/1.7*15*log(2932/432) = 1.83 ft = 22 in.
  - Improved: qg = q*ns/(Ra*ns - Ra + 1) = 2500*6/2.35 = 6,383 psf; suz = qg/kg = (6383/144 psi)/65 pci = 0.68 in; lower zone = 0 (rock) → total ≈ 0.7 in.
  - Suggested tolerance: +/-10%.
- Confidence in extraction: high (all arithmetic printed)
- Notes: exercises the RAP upper-zone stiffness-modulus method and the baseline consolidation calc. Watch the de = 1.05s unit-cell convention.

## V-019 GEC-13 Example Problem 2 — stone column settlement improvement (Priebe)
- Source: GEC-13 Vol 1, Ch. 5 Sec 4.3.2 + Figure 5-27 (PDF pp. 373-375; extract: `g13_v1_ex.txt`)
- Target: `ground_improvement` (stone columns / vibro, Priebe improvement factor)
- Inputs: embankment q = 125 pcf * 15 ft = 1,875 psf on 50 ft soft clay (gamma_sat = 120 pcf, WT at surface; po at mid-depth 25 ft = 1,440 psf; Cc = 0.2, eo = 0.6) over dense sand (no settlement). Stone columns: diameter 3.0 ft, spacing 5.7 ft → area ratio A/Ac = (5.7/3.0)^2 = 3.6 (source convention).
- Published answer:
  - Unimproved: S = 0.2/1.6*50*log(3315/1440) = 2.26 ft = 27 in.
  - Priebe settlement-improvement ratio (from chart, Wallays et al. 1983 presentation) = 2.7 → improved settlement = 27/2.7 = 10 in.
  - Suggested tolerance: ratio 2.7 +/- 0.3 (chart read; Priebe n0 with phi_col ≈ 42.5 deg and the source's area-ratio convention).
- Confidence in extraction: high for the numbers; medium for method match (chart-based ratio; our Priebe formula may use Ac/A and phi_col input — verify convention before asserting failure)
- Notes: pins the unimproved consolidation calc tightly; treat the 2.7 ratio as a soft target.

## V-020 GEC-13 Ch. 2 PVD design example — time to 90% consolidation vs drain spacing
- Source: GEC-13 Vol 1, Ch. 2 Sec 4.4 (PDF p. 116; extract: `g13v1_pvd.txt`)
- Target: `ground_improvement` (wick drains / PVD, Barron-Hansbo radial consolidation)
- Inputs: 20 ft NC clay (sand lenses) over rock; cv = 0.1 ft2/day, ch = 2*cv = 0.2 ft2/day; PVD equivalent diameter dw = 2.5 in; triangular pattern (de = 1.05*s); target U = 90%; ideal drains (no smear/well resistance stated).
- Published answer: t90 ≈ 300 days at s ≈ 8 ft; t90 ≈ 500 days at s ≈ 10 ft ("on the order of").
- Confidence in extraction: medium (answers given as approximate; combined vertical+radial contribution ambiguous — with ch-only Barron, s = 8 ft gives t90 in the right range)
- Notes: use generous tolerance (+/-20%); good smoke test of the radial-drainage formula and de/dw plumbing rather than a precision check.

## V-021 GEC-6 Example B-1 — spread footing ultimate bearing capacity (Vesic factors, GW correction)
- Source: GEC-6 (FHWA-SA-02-054), Appendix B, Example 1 (PDF pp. 161-168; extract: `gec6_ex1.txt`)
- Target: `bearing_capacity` (Vesic/AASHTO factors, shape + groundwater corrections)
- Inputs:
  - Square footing (B = L), Df = 2.3 m, bearing in silty sand: phi = 35 deg, c = 0, gamma = 19.6 kN/m3 (all layers); GW at 9.1 m below grade.
  - Surcharge q = 19.6*2.3 = 45.1 kPa; shape factors only (no inclination): s_gamma = 1 - 0.4*B/L = 0.6, sq = 1 + (B/L)tan(phi) = 1.7; dq = 1.0; Nq = 33.3, N_gamma = 48.0 (AASHTO/Vesic, phi = 35).
  - GW corrections at B = 6 m trial: Cwq = 1.0, Cw_gamma = 0.5 + 0.5*[9.1/(2.3 + 1.5*6)] = 0.9.
- Published answer:
  - qult = q*Nq*sq*Cwq + 0.5*gamma*B*N_gamma*s_gamma*Cw_gamma = 2,553 kPa + 254*B(m) kPa.
  - With FS = 3: qall = 851 + 85*B kPa → B = 3 m: 1,106 kPa; B = 4.6 m: 1,242 kPa; B = 6.1 m: 1,369 kPa.
  - Suggested tolerance: +/-3% (closed-form; N-factor table rounding).
- Confidence in extraction: high
- Notes: matches Vesic Nq = 33.30/N_gamma = 48.03 at phi = 35 — direct check of our Vesic path with shape + Cw groundwater factors. (The example's eccentricity/sliding follow-on: B' = 4.75 m, L' = 4.67 m for 4.9 m square footing with ey = 0.077 m, ez = 0.117 m at P = 8,070 kN; q_applied = 364 kPa; sliding FS = 31 — usable as a second assertion set for effective-area handling.)

## V-022 GEC-6 Example B-1 — Hough-method settlement table for granular layers
- Source: GEC-6, Appendix B, Example 1, Tables B1-2/B1-3 (PDF pp. 168-170; extract: `gec6_ex1.txt`)
- Target: `settlement` (granular/SPT method — Hough; verify module support, else 2:1 stress + log formula)
- Inputs:
  - Square footing Df = 2.3 m; layers below footing: L2 silty sand 2.1 m thick (mid Z = 1.05 m, sigma'vo = 65.7 kPa, C' = 65); L3a well-graded sand 4.7 m (Z = 4.45 m, 132 kPa, C' = 120); L3b 3.0 m (Z = 8.3 m, 193 kPa, C' = 102); L4 3.0 m (Z = 11.3 m, 222 kPa, C' = 110). (C' = Hough bearing capacity index from corrected N': 24/36/30/26.)
  - Stress increase by 2:1 method: d_sigma_v = q*B^2/(B+Z)^2 (square footing).
  - dH = H/C' * log10[(sigma'vo + d_sigma)/sigma'vo] per layer, summed.
- Published answer (total settlement, mm):
  - B = 3 m: q = 240 kPa → 21; 290 → 25; 335 → 28; 380 → 30.
  - B = 4.6 m: 240 → 28; 290 → 31; 335 → 34; 380 → 37.
  - B = 6.1 m: 240 → 31; 290 → 35; 335 → 38; 380 → 41.
  - Worked single case (B = 3, q = 240): per-layer 15 + 4 + 1 + 1 = 21 mm.
  - Suggested tolerance: +/-15% (their per-layer values are rounded to mm).
- Confidence in extraction: high
- Notes: only implement if our settlement module exposes a Hough/C'-index granular method or a generic 2:1 + log-stress layer sum; otherwise record as a coverage gap.

## V-023 Itasca FLAC verification — one-dimensional consolidation (Terzaghi/Biot column)
- Source: Itasca FLAC2D/FLAC3D verification "One-Dimensional Consolidation", https://docs.itascacg.com/itasca930/flac3d/zone/test2d/Fluid/1DConsolidation/1dconsolidation2d.html
- Target: `fem2d` (coupled consolidation, plane strain column)
- Inputs:
  - Soil column H = 20 m (20 zones), laterally confined (zero horizontal displacement), base fixed and impermeable; top drained (p = 0) after load application.
  - Elastic: K = 5e8 Pa, G = 2e8 Pa (E = 5.17e8 Pa, nu = 0.292); Biot alpha = 1.0, Biot modulus M = 4e9 Pa; mobility coefficient k = 1e-10 m2/(Pa*s).
  - Applied surface load pz = 1e5 Pa (undrained application, then drainage).
- Published answer (analytical Biot/Terzaghi series):
  - Initial undrained pore pressure p0 = (alpha*M/(K + 4G/3 + alpha^2*M))*pz ≈ 0.981e5 Pa (98,119 Pa).
  - Storage S = 1/M + alpha^2/(K + 4G/3) = 1.554e-9 1/Pa; consolidation coefficient c = k/S = 0.0643 m2/s; pore-pressure decay p(z,t) follows the standard one-D series in t_hat = c*t/H^2 (e.g., at the base, p/p0 ≈ 0.92 at t_hat = 0.1, 0.31 at t_hat = 0.5 — series values).
  - FLAC reports < 5% max relative error vs the analytical series.
  - Suggested tolerance: +/-5% on p/p0 history at base and on final settlement = pz*H/(K + 4G/3) (drained) = 2.61 mm.
- Confidence in extraction: high for inputs; medium for the spot series values (computed from the standard series, not quoted by Itasca — implementer should evaluate the series directly)
- Notes: exercises fem2d coupled consolidation with compressible fluid (finite M). If fem2d assumes incompressible fluid (M → inf), set the equivalent and adjust p0 → pz.

## V-024 Itasca FLAC verification — cylindrical hole in infinite Mohr-Coulomb medium (Salencon)
- Source: Itasca FLAC3D verification "Cylindrical Hole in an Infinite Mohr-Coulomb Material", https://docs.itascacg.com/flac3d700/flac3d/zone/test3d/VerificationProblems/CylinderInMohrCoulomb/salencon.html
- Target: `fem2d` (plane-strain MC plasticity; quarter-symmetry hole unloading)
- Inputs: hole radius a = 1 m (domain >= 5 diameters); isotropic in-situ stress P0 = 30 MPa; internal pressure Pi = 0; K = 3.9 GPa, G = 2.8 GPa (nu ≈ 0.313); MC: c = 3.45 MPa, phi = 30 deg, psi = 0 (non-associated case).
- Published answer (Salencon 1969 analytical; constants quoted by Itasca: Kp = 3.0, q = 2c*sqrt(Kp) = 11.95 MPa):
  - Plastic radius R0 = a*[ (2/(Kp+1)) * (P0 + q/(Kp-1)) / (Pi + q/(Kp-1)) ]^(1/(Kp-1)) = 1.73 m.
  - Radial stress at elastic-plastic boundary sigma_r(R0) = (1/(Kp+1))*(2*P0 - q) = 12.01 MPa.
  - Stresses/displacements along a radial line follow the closed-form Salencon solution; FLAC achieves <= 2.1% average error (4.6% on displacements, non-associated).
  - Suggested tolerance: +/-5% on R0 and on sigma_r/sigma_theta profiles at r/a = 2-5.
- Confidence in extraction: medium (a = 1 m and the R0/sigma_r values derived from the standard Salencon formulas with the quoted constants — verify against the closed form when implementing)
- Notes: classic elasto-plastic verification; complements our existing Prandtl/SRM checks with a stress-redistribution problem. Far-field truncation at 5 diameters causes ~1-2% boundary error — match Itasca's domain or use displacement-controlled far field.

## V-025 Caltrans T&S Example 7-2 — layered active pressure with water (total driving force)
- Source: California Trenching & Shoring Manual, Ch. 7, Example 7-2 (PDF pp. 139-142; extract: `cal_7_1B.txt` tail)
- Target: `geotech_common`/`sheet_pile` (layered Rankine active pressure + hydrostatic water, stress-point diagram)
- Inputs: wall retaining (top down): 4 ft coarse sand and gravel, gamma = 130 pcf, phi = 37 deg (Ka = 0.249); then fine sand, gamma = 102.4 pcf, phi = 30 deg (Ka = 0.333) — 6 ft moist, then 20 ft below the water table (use gamma_sub = 102.4 - 62.4 = 40 pcf as printed); water table 10 ft below top; wall friction 0.
- Published answer:
  - Stress points: sigma1+ = 129.48 psf (Ka1), sigma1- = 173.16 psf (Ka2 applied to same overburden), sigma2 = 377.76 psf, sigma3 = 644.16 psf; water at base u = 1,248 psf.
  - Forces: F1 = 259.0, F2 = 1,039.0, F3 = 613.8, F4 = 7,555.2, F5 = 2,664.0, F6 (water) = 12,480 lb/ft; FTOTAL = 24,610.9 lb/ft.
  - Suggested tolerance: +/-1% (pure arithmetic).
- Confidence in extraction: high
- Notes: cheap, sharp check of layered-Ka stress discontinuities and water handling in the earth-pressure builder used by sheet_pile/soe.

---

# v5.3 additions — Slide2 Slope Stability Verification Manual (public)

Source manual: Rocscience Slide2 Slope Stability Verification Manual,
https://static.rocscience.cloud/assets/verification-and-theory/Slide2/Slide_SlopeStabilityVerification.pdf
(21 MB, kept out of the repo). Raw text of the 21 selected problems:
`module_work/slope_v53/slide2_selected_problems.txt`. These are classic published
referee problems (ACADS/Giam & Donald 1989, Duncan 2000, Loukidis 2003, Baker
2003, Pockoski & Duncan 2000) each carrying per-method + referee factors of
safety. NOTE: the manual's geometry lives in figures that did NOT survive the
text extraction; problems whose geometry could not be reconstructed from the text
+ a known sibling problem are marked SKIPPED(geometry) below.

## V-026 Slide2 #2 = ACADS 1(b) — homogeneous slope, water-filled tension crack
- Source: Slide2 Verification #2 (manual pp. 24-27); ACADS 1(b) [Giam & Donald 1989].
- Target: `slope_stability` (tension crack + water in crack, all methods).
- Geometry: SAME as Slide2 #1 / ACADS 1(a) — the (20,25)-(30,25)-(50,35)-(70,35) m,
  2:1, 10 m slope already validated in slope_stability B3. Analyzed MIRRORED
  (crest on the left, x->90-x) because the module hard-codes the tension crack to
  the slip-surface ENTRY side.
- Inputs: soil c'=32 kPa, phi'=10 deg, gamma=20 kN/m3 (Table 2.1); water-filled
  tension crack, Rankine/Craig-1997 depth zc = 2c'/(gamma*sqrt(Ka)),
  Ka=(1-sin phi')/(1+sin phi')=0.704 -> zc=3.81 m. Search grid centers per manual
  ~x[31,47] y[34,49].
- Published answer (Table 2.2, WITH water crack): Bishop 1.596, Spencer 1.592,
  GLE 1.592, Janbu corrected 1.489; referee FOS = 1.65 [Giam].
- Suggested tolerance: dry crack within +/-3.5% of the Slide2 water-crack values;
  water crack documented as ~6% conservative (CONVENTION).
- Confidence in extraction: high (geometry = known ACADS #1; soil + answers verbatim).
- Notes: exercises the tension-crack strength truncation + hydrostatic crack-water
  thrust. The module is more conservative than Slide2 on the water-filled crack
  (keeps the cracked wedge as zero-strength driving soil + full hydrostatic thrust;
  Slide2 truncates the mass at the crack). B2 refinement candidates: (1) allow the
  tension crack on the exit side (currently entry-only, forcing the mirror);
  (2) mass-truncation crack model. See RESULTS.md V-026.
- **v5.4 update (E4): both candidates BUILT, residual RESOLVED.**
  `tension_crack_side='exit'` removes the mirror workaround (un-mirrored ≡
  mirrored to ~1e-16); `tension_crack_model='truncation'` (Slide2's mass model)
  reproduces the published water-crack FOS to <0.1% (Bishop 1.597 / Spencer
  1.593 / GLE 1.594 vs 1.596 / 1.592 / 1.592) — not tuned. Defaults
  ('entry', 'strength') unchanged; the strength model stays documented as the
  conservative convention. Tests: test_published_v026_e4.py (5).

## V-029 Slide2 #10 = ACADS 5 — homogeneous slope, pore-pressure GRID + ponded water
- Source: Slide2 Verification #10 (manual pp. 55-58); ACADS 5 [Giam & Donald 1989].
- Target: `slope_stability` — but pore pressure is supplied as a discrete GRID of
  (x, z, u) points (TIN-interpolated from the Figure 10.2 flow net).
- Inputs: homogeneous soil c'=11 kPa, phi'=28 deg, gamma=20 kN/m3 (Table 10.1);
  1:2 excavated slope (beta=26.56 deg) below an initially horizontal surface;
  pore pressure from a flow-net grid; ponded water.
- Published answer (Table 10.2): Bishop 1.498, Spencer 1.500, GLE 1.500, Janbu
  corrected 1.457; referee 1.53 [Giam].
- Verdict: **N/A (scope)** — the module has NO pore-pressure-GRID input (it
  supports a piezometric surface `gwt_points` and a per-layer `ru`, but not an
  arbitrary TIN-interpolated u(x,z) grid). The flow-net grid values are a figure
  that did not survive text extraction, and the excavation height is not in the
  text, so an ru/piezo approximation is not defensible here. Capability gap for B2
  (pore-pressure-grid input + TIN interpolation). No test.
- **v5.4 update (E3): capability BUILT** — `SlopeGeometry.pore_pressure_points`
  (scattered (x,z,u) triples, Delaunay-linear TIN interpolation with nearest-
  neighbour fallback, suction clamped ≥0, overrides piezo/ru at slice bases,
  wired through the search; additive, None = byte-identical). Validated
  rigorously by construction (hydrostatic grid ≡ piezometric line to machine
  precision; TIN exact on linear fields; 6 tests, test_published_v029*). The
  published Bishop 1.498 remains UNPINNED — Fig 10.2's grid values are still
  not recoverable and inventing them is forbidden; the ACADS-5-style run is a
  capability demonstration (dry 1.89 → grid 1.25), not a referee match.
- Confidence in extraction: soil + answers high; geometry + PP grid not recoverable
  from text.

## V-030 Slide2 #29 — Duncan (2000) LASH underwater slope (deterministic + probabilistic)
- Source: Slide2 Verification #29 (manual pp. 121-122); Duncan (2000), JGGE 126(4).
- Target: `slope_stability` (undrained noncircular FOS + `probabilistic` reliability).
- Geometry (feet, labeled Fig 29.1): seabed surface (-28,-40),(0,-40),(71,-120),
  (138,-120),(228,-18),(283,-17),(350,-8),(389,22),(461,22); ocean level el 0;
  base el -143. Duncan's "estimated" noncircular failure surface (pixel-traced,
  ~+-3 ft): (138,-120),(150,-117),(170,-105),(185,-100),(205,-93),(221,-85),
  (240,-75),(257,-64),(275,-53),(293,-39),(311,-23),(350,-8).
- Inputs: SF Bay Mud, undrained su = 100 psf at el -20, +9.8 psf/ft; gamma=100 pcf.
  Probabilistic (Table 29.2): gamma std 3.3 pcf; su rate-of-change std 1.2 psf/ft.
- Published answer (Table 29.3): deterministic FOS Janbu-s 1.127 / Janbu-c 1.168 /
  Spencer 1.157 / GLE 1.160 (Duncan quote 1.17); Pf 18% (Taylor series).
- Suggested tolerance: +/-5% on the deterministic FOS; Pf within a few points.
- Confidence in extraction: high (labeled figure; surface pixel-traced).
- Notes: DETERMINISTIC PASS — rigorous methods reproduce Spencer 1.157/Duncan 1.17
  to +2.9% (thin undrained sub-layers for the linear su + ocean at el 0). RELIABILITY
  PASS — Taylor-series helpers reproduce F=1.17->Pf 18%. Input-COV FOSM propagation:
  GAP CLOSED (v5.4 F1) — the new `linear_su` correlated (a,b) law + `gamma_sat` var
  propagate the stated Table 29.2 inputs to COV_F 0.133 -> Pf ~11% (~13% at F=1.17, in
  the Slide2 band); Duncan's closed-form COV_F 0.16 -> Pf 18% is recovered with the full
  coherent su-profile COV. Tests: test_published_v030_fosm_slope.py (5). See RESULTS V-030.

## V-031 Slide2 #57 — Pockoski & Duncan (2000) test slope 3 (2-layer, water, circular)
- Source: Slide2 Verification #57 (manual pp. 203-205); Pockoski & Duncan (2000).
- Target: `slope_stability` (2-layer + water table, circular critical surface).
- Geometry (feet, from labeled Fig 57.1): surface (-70,100),(0,100),(125,150),
  (200,150) — 2.5:1, 50 ft; horizontal material boundary el 90; base el 85; water
  table (-70,100),(0,100),(125,140),(200,140); dry tension crack (auto, shallow).
- Inputs: Sandy clay c'=300 psf/phi'=35/130 pcf over Highly Plastic Clay
  c'=0/phi'=25/130 pcf.
- Published answer (Table 57.3, circular): Spencer 1.422, Bishop 1.417, Janbu-simplified
  1.263, Lowe-Karafiath 1.414, Ordinary 1.319; GOLD-NAIL 1.40.
- Suggested tolerance: +/-1% on Bishop/Spencer vs the Slide2 circular values.
- Confidence in extraction: high (labeled figure vertices).
- Notes: Bishop/Spencer reproduce to <0.6% (PASS). Janbu (module f0-CORRECTED)
  runs above the Slide2 Janbu-SIMPLIFIED, and Ordinary/OMS-with-water below — usual
  method-definition conventions.

## V-032 Slide2 #61 — Baker (2003) example 3, homogeneous Mohr-Coulomb
- Source: Slide2 Verification #61 (manual pp. 213-215); Baker (2003).
- Target: `slope_stability` (homogeneous M-C circular). Power-curve (nonlinear
  envelope) variant is OUT OF SCOPE (module has no power-curve strength).
- Geometry (metres, labeled Fig 61.1): surface (0,0),(6,6),(20,6) — 45 deg (1:1),
  6 m high; base y=0.
- Inputs: homogeneous clay c'=6 kPa, phi'=32 deg, gamma=18 kN/m3.
- Published answer: Spencer 1.366, Janbu-simplified 1.291 (M-C). (Power-curve
  Spencer 1.468 / Baker nonlinear FS 1.48 — out of scope.)
- Suggested tolerance: +/-1% on Spencer.
- Confidence in extraction: high (labeled figure).
- Notes: Spencer reproduces to <0.1% (PASS).

## V-033 Slide2 #62 — Loukidis et al. (2003) ex 1, critical seismic coefficient
- Source: Slide2 Verification #62 (manual pp. 216-218); Loukidis et al. (2003).
- Target: `slope_stability` (pseudo-static SEISMIC, homogeneous, circular thru toe).
- Geometry (metres, labeled Fig 62.1): surface (-50,0),(0,0),(75,25),(150,25) —
  3:1, 25 m high; base y=-25.
- Inputs: homogeneous clay c'=25 kPa, phi'=30 deg, gamma=20 kN/m3; Loukidis'
  critical seismic coefficient (dry) kc=0.432 (also ru=0.5 case kc=0.132).
- Published answer: at kc=0.432 the Spencer FOS = 1.000 (Slide2 circular 1.001).
- Suggested tolerance: +/-2% on the reproduced FOS=1.0.
- Confidence in extraction: high (labeled figure).
- Notes: Spencer 1.005 at kc=0.432 (+0.5%, PASS). This CLEAN, fully-labeled
  geometry validates the module's pseudo-static seismic engine independently of
  the harder 3-layer ACADS #4 seismic case (whose reconstructed layer geometry is
  the limiting factor there, not the seismic engine).

## V-027 Slide2 #4 = ACADS 1(d) — 3-material slope + seismic 0.15g (geometry-limited)
- Source: Slide2 Verification #4 (manual pp. 32-35); ACADS 1(d) [Giam & Donald 1989].
- Target: `slope_stability` (3-layer + pseudo-static seismic kh=0.15).
- Geometry (metres, from Fig 4.1; layer boundaries pixel-read, MODERATE confidence):
  surface (20,25),(30,25),(50,35),(70,35); base y=20; upper matl bnd (base of
  Soil#1) (33,26.5),(40,27),(50,29),(54,31),(70,31); lower matl bnd (Soil#2/#3)
  (40,27),(52,24),(70,24). Soil#1 c'=0/phi'=38, Soil#2 c'=5.3/phi'=23, Soil#3
  c'=7.2/phi'=20, all gamma=19.5. kh=0.15.
- Published answer: STATIC (#3, Table 3.2) Bishop 1.405 / Spencer 1.375 / GLE 1.374;
  SEISMIC 0.15g (#4, Table 4.2) Bishop 1.016 / Spencer 0.991 / GLE 0.989; referee 1.00
  (seismic).
- Verdict: **N/A (geometry-precision — discrepancy analysis)** — the reconstructed
  geometry cannot reproduce EITHER published table, and the discrepancy is isolated
  to the geometry, not the module:
  * The module's per-slice soil assignment was VERIFIED correct against the figure
    (toe/base = orange Soil#3 φ=20, mid-right wedge = green Soil#2, crest cap = yellow
    Soil#1 φ=38).
  * The critical circle is a toe circle DOMINATED by the weak Soil#3 (φ=20) → static
    Bishop 1.02 / seismic 0.77. A full soil-interpretation sweep (Soil#1-top vs
    -base, flip, all-Soil#2, all-Soil#1) found NONE reaching the published static
    1.405: the target sits between homogeneous all-φ23 (static 1.305) and all-φ38
    (1.562), i.e. an EFFECTIVE φ≈29 along the critical surface.
  * The target seismic/static ratio 0.723 matches the HOMOGENEOUS cases (0.712-0.725),
    NOT the layered one (0.760) — and the seismic engine is independently validated
    to +0.5% by Loukidis #62 (V-033). So the module is right; the published surface
    must AVOID the weak Soil#3 toe (Soil#3 confined deeper than the figure read
    implies), which the reconstructed boundaries don't achieve. FOS is hyper-sensitive
    to this. No forced test.
- Confidence in extraction: surface + soils high; layer boundaries moderate.

## V-028 Slide2 #9 = ACADS 4 — weak seam + piezometric surface, noncircular
- Source: Slide2 Verification #9 (manual pp. 50-54); ACADS 4 [Giam & Donald 1989].
- Target: `slope_stability` (thin weak seam, noncircular, water table).
- Geometry (metres, Fig 9.1): surface (20,28),(43,28),(68,40),(84,40); base z=15;
  weak seam = 1 m planar band, does NOT daylight, top [(20,19),(84,37)] bottom
  [(20,18),(84,36)]. Soil#1 c'=28.5/phi'=20/gamma=18.84 (above+below seam); Soil#2
  weak seam c'=0/phi'=10. Piezometric surface (Table 9.3, confirmed vs figure).
- Published answer (noncircular through the seam): NO-optimization Spencer 0.760 /
  GLE 0.720 / Janbu-c 0.734; block-search WITH optimization Spencer 0.707 / GLE
  0.683 / Janbu 0.699; Slope-2000 GLE 0.6878; referee 0.78 [Giam].
- Verdict: **PASS (v5.3 B2)** — RE-VALIDATED after the SS-6 weak-layer search fix.
  Before the fix the `weak_layer` noncircular search returned spurious DEGENERATE
  surfaces (Spencer ~0.05-0.18, a solver-non-convergence artifact); it now reliably
  finds a smooth seam-following critical surface at Spencer 0.792 -- within +1.5% of
  the referee 0.78 (GLE 0.786, Janbu 0.804), bracketed by Slide2's no-opt 0.760 and
  optimized 0.707. The optimized `noncircular_de` reaches ~0.69, matching Slide2's
  optimized 0.707 / Slope-2000 0.6878. Two tests in
  test_published_v035_slope.py (pinned surface + search regression). NOTE: the
  module still supports only ONE surcharge zone (problem has bench + crest — a
  separate B2 capability gap; the loads are secondary to the seam-governed FOS).
- Confidence in extraction: high (label-confirmed geometry).

## V-034 Slide2 #63 — Loukidis (2003) ex 2, 3-layer seismic (geometry-limited)
- Source: Slide2 Verification #63 (manual pp. 219-220); Loukidis et al. (2003).
- Target: `slope_stability` (3-layer + pseudo-static seismic kc=0.155).
- Geometry (metres, Fig 63.1 UNLABELED — pixel-reconstructed, ~+/-1.5 m): surface
  (-30,20),(20,20),(58,39),(77,40),(111,54),(150,54); top/middle bnd daylights
  ~(78,41)->(150,44); middle/lower bnd daylights ~(37,27)->(150,33); base y~-20.
  Top c=4/phi=30/gamma=17, Middle c=25/phi=15/gamma=19, Bottom c=15/phi=45/gamma=19.
- Published answer: Spencer FOS = 1.000 at kc=0.155 (Slide2 0.991).
- Verdict: **N/A (geometry-precision)** — the approximate/unlabeled 3-layer figure
  gives seismic ~0.82 vs 1.0 (same imprecise-3-layer-geometry issue as V-027 #4).
  The seismic engine is validated by V-033 (#62). No passing test.
- Confidence in extraction: geometry approximate (unlabeled figure).

## V-035 Slide2 #79 — Duncan & Wright (2005), cohesionless embankment, infinite slope
- Source: Slide2 Verification #79 (manual pp. 267-269); Duncan & Wright (2005), Fig 14.4.
- Target: `slope_stability.infinite_slope_fos` (planar/translational mechanism).
- Geometry: cohesionless embankment on a stiff foundation; the "very shallow"
  (infinite-slope) surface parallels the 2.5H:1V embankment face (tan beta = 0.4).
  Embankment c'=0, phi'=30, gamma=120 pcf.
- Published answer (Case 2, infinite slope): Bishop/Spencer/GLE all 1.443-1.444;
  referee 1.44 [Duncan & Wright].
- Suggested tolerance: exact (closed form); +/-0.005.
- Confidence in extraction: high (the slope ratio 2.5:1 is the clean value the
  infinite-slope FOS = tan30/tan(beta) reproduces to 1.443).
- Notes: PASS — `infinite_slope_fos` gives FOS = tan(30)/0.4 = 1.443.

## V-036 Slide2 #81 — Duncan & Wright (2005), earth embankment, infinite slope
- Source: Slide2 Verification #81 (manual pp. 273-275); Duncan & Wright (2005), Fig 14.7.
- Target: `slope_stability.infinite_slope_fos`.
- Geometry: 2H:1V embankment (tan beta = 0.5); embankment c'=0, phi'=30, gamma=124 pcf.
- Published answer (Case 2, infinite slope): Bishop/Spencer/GLE all 1.155;
  referee 1.15 [Duncan & Wright].
- Suggested tolerance: exact (closed form); +/-0.005.
- Confidence in extraction: high (2:1 ratio).
- Notes: PASS — `infinite_slope_fos` gives FOS = tan(30)/0.5 = 1.155.

## V-037 Slide2 #95 — USACE 2-stage rapid drawdown (EM 1110-2-1902 App. G)
- Source: Slide2 Verification #95 (manual pp. 307-308); USACE EM 1110-2-1902 (1970).
- Target: `slope_stability.rapid_drawdown_fos` (`method='corps_2stage'`).
- Geometry (feet, Fig 95.1 — EXACT two-segment face, calibrated to the published
  circle): homogeneous embankment, toe (0,0) -> 3H:1V to (220,73) -> 2.5H:1V to the
  crest shoulder (312,110) -> flat crest to (380,110); base el 0. Specified circle
  centre (169.5,210), R=210 (arc through the confirmed entry (72,24) / exit
  (354,110) to <0.5 ft). Water: initial el 110 -> drawdown to el 24. (Figure draws
  the initial level at ~103; the TEXT says 110 — 110 matches the published FOS.)
- Inputs: gamma=135 pcf; effective c'=0/phi'=30; R-envelope cR=1200 psf/phiR=16.
- Published answer: Army-Corps 2-stage FOS 1.347; referee 1.35.
- Verdict: **PASS (under steady-seepage stage-1)** — with the declined stage-1
  phreatic (`stage1_phreatic_points`, ~0.08 gradient) FOS = 1.34 vs 1.347 (~0.6%).
  The combined R/effective (`min`) envelope is confirmed correct (pure R -> ~1.42).
  The flat-phreatic DEFAULT gives the conservative bound 1.21. The exact geometry
  does NOT close the gap on its own (~1.21, if anything below the earlier straight
  face) — the residual was the stage-1 flow net, now reproducible via the option.
- Confidence in extraction: soil + circle + water + geometry exact.

## V-038 Slide2 #96 — Duncan-Wright-Wong 3-stage rapid drawdown (same dam)
- Source: Slide2 Verification #96 (manual pp. 309-310); Duncan, Wright, Wong (1990).
- Target: `slope_stability.rapid_drawdown_fos` (`method='duncan_3stage'`).
- Geometry/inputs: identical to V-037 (only the method differs).
- Published answer: Duncan-Wright-Wong 3-stage FOS 1.443; referee 1.44.
- Verdict: **CONVENTION (approximate)** — flat 1.235, seepage stage-1 1.273; the
  published ORDERING holds (3-stage >= 2-stage at the default) and seepage raises
  it, but a ~12% residual to 1.443 remains at the SAME phreatic that validates the
  2-stage. Isolated to the Duncan-Wright-Wong Kc (anisotropic-consolidation)
  interpolation for a c'=0 soil: the drained (Kc=Kf) envelope falls below the R
  (Kc=1) envelope at low sigma'_fc, so the anisotropic strength gain that lifts the
  published 3-stage above the 2-stage is under-captured. NOT geometry/seepage;
  documented follow-up (see DESIGN.md). Stage-3 substitution left per Duncan-
  Wright-Brandon (2014) Ch. 9.
- **v5.4 update (E2):** re-diagnosed — the Kc interpolation is sound (yields ~1.45
  in isolation); the residual sits in the STAGE-3 drained substitution, whose
  Fellenius estimate of the drawn-down effective normal is inconsistent with the
  rigorous GLE normal used in stage 1 and over-fires the substitution (17/50
  slices). Opt-in `stage3_effective_normal='gle'` (default `'fellenius'`
  preserved) uses the consistent rigorous normal: flat 1.306 / seepage 1.370 vs
  published 1.443 (~5% residual, within the flow-net + LE-N'-at-FOS sensitivity
  seen on #95). Default-flip decision is owner-gated. See slope_stability
  VALIDATION.md §B7 (authoritative) and test_published_v038.

## V-041 Slide2 #98 (Walter Bouldin Dam, 5-material rapid drawdown) — was DEFERRED, now VALIDATED (geometry-limited)
- Source: Slide2 Verification #98 (manual pp. 313-314); Duncan, Wright, Wong (1990).
- Published answers (SEARCH minima): Corps 2-stage 0.931, Lowe-Karafiath 1.075,
  DWW 3-stage 1.039. Water: initial 47 ft -> drawdown 15 ft.
- Geometry recovered (Fig 98.1, printed labels): domain x 0-180, z 0-60; surface
  (0,0)-(100,40)-(140,60)-(180,60); 4 stacked layers (Clayey Sandy Gravel 0-17,
  Cretaceous Clay 17-30, Micaceous Silt/Sand 30-51, Clayey Silty Sand 51-60 at
  x=180) with pinch-outs, plus a riprap veneer on the upper face. Material R-data
  in the manual is complete (Table 98.1).
- Verdict (v5.3): **DEFERRED (search-limited, not data-limited)** — the published
  values are MINIMA over a slip-surface search and `rapid_drawdown_fos` evaluated
  only one specified surface.
- **Verdict (v5.4, E1): VALIDATED — GEOMETRY-LIMITED.** `search_rapid_drawdown`
  (the B2a-search wrapper) now finds drawdown search minima: Corps 0.837 / DWW
  0.938 vs published 0.931 / 1.039 — both ~10% low with the correct method
  ordering (DWW > Corps), on the RECOVERED simplified section (flat-stacked
  layers, pinch-outs simplified, riprap veneer omitted). Wrapper exactness proven
  on the exact #95/#96 section (search min ≤ published specified-circle FOS;
  stage detail reproduces the search FOS to machine precision). NOT tuned. See
  test_published_v041.py and slope_stability VALIDATION.md §B7.

## V-039 Slide2 #104 — Newmark seismic sliding-block displacement (B2b)
- Source: Slide2 Verification #104 (manual pp. 330-331), based on Slide2 Tutorial 28
  "Seismic Analysis with Newmark Method"; Newmark (1965); Jibson (2007).
- Target: `slope_stability.newmark` — `yield_acceleration`, `newmark_displacement`,
  `newmark_jibson2007`.
- Published answers (four scenarios, MMO / uni-modal): no-seismic FS 1.359/1.360;
  seismic k=0.15 FS 0.978/0.980; critical accel Ky=0.139/0.140; Newmark displacement
  5.042/5.081 cm.
- What is / isn't recoverable: the acceleration TIME HISTORY (Tutorial-28 record) and
  the Tutorial-28 geometry are NOT in the manual, so 5.042 cm cannot be reproduced
  directly.
- Verdict: **PASS (integrator + Jibson) / #104 documented (record-limited).**
  (1) The rigid-block integrator reproduces the closed-form rectangular-pulse Newmark
  displacement D = ap(ap−ay)T²/(2ay) exactly. (2) `newmark_jibson2007` reproduces
  Jibson (2007) Eq. 6 (coeffs 0.215 / 2.341 / −1.438, σ=0.510) exactly. (3) The #104
  four-scenario structure is reproduced qualitatively: the published values are
  internally consistent with a near-linear FOS(k) (FS 1.359 at k=0, 0.978 at k=0.15 →
  FS=1 at k≈0.141, matching Ky=0.139), which is what `yield_acceleration` bisects; and
  a Jibson cross-check at Ky=0.139 brackets the published 5.042 cm (1.9 cm at PGA 0.35g,
  7.3 cm at 0.60g).
- Confidence in extraction: published FS/Ky/disp values exact; record + geometry absent.

## V-040 Slide2 #54 — stabilizing micro-piles (Yamagami 2000) (B2d)
- Source: Slide2 Verification #54 (manual pp. 196-198; Yamagami 2000); Ito & Matsui
  (1975) for the plastic-force formula (#106 target, geometry-limited).
- Target: `slope_stability.reinforcement.StabilizingPile` + `ito_matsui_lateral_force`
  / `ito_matsui_pressure`, wired through `compute_reinforcement_forces`.
- Geometry (Fig 54.1, LABELED): surface (-6,0)-(0,0)-(8,4)-(12,4); base to z=-5;
  homogeneous c'=4.9/phi'=10/gamma=15.68. Single pile row at the crest (x≈8.75),
  spaced 1 m, shear strength 10.7 kN per pile. Circular search.
- Published: no-pile 1.102 (Yamagami 1.10); with-pile 1.193 (Yamagami 1.20).
- Verdict: **CONVENTION (pile reinforcement validated).** No-pile critical circle
  1.114 (+1.1%); with the 10.7 kN/m pile on that surface 1.223 (+2.5%). The pile-force
  integration is exact (phi=0 closed form in tests). The +2.5% over-prediction is the
  active-vs-passive support convention (the module reduces the driving moment; a passive
  resisting force would be slightly less effective) + the figure-read pile location +
  the single-pile search subtlety (a re-search WITH the pile finds a pile-avoiding
  surface at ~1.11, so the pile is applied to the recovered critical surface as Slide
  reports). NOT tuned.
- Ito-Matsui FORMULA (#106 Cai & Ugai 2000): implements the ORIGINAL Ito & Matsui
  (1975) Eq. 13 (exp arg tan(pi/8+phi/4), first-term coeff 1/(Nphi·tanphi), second
  cohesion term −c(D1·Fc−2·D2/√Nphi)) + the phi=0 Eq. 23 cohesive limit; unit-tested
  against the paper's exact hand-check values (c=10/phi=20/gamma=18/z=5/D1=2/D2=1.5 →
  105.079 kN/m per m depth; phi=0 c=25/gamma=18/z=4/D1=2/D2=1 → 146.683) PLUS the #106
  spacing TREND (force/metre falls as spacing/diameter grows: #106 FS 1.54/1.37/1.31/
  1.25 for ratio 2/3/4/6). The #106 cross-section (Fig 106.1) is not in the manual
  extract, so a single-surface FOS is not reproduced.
- Confidence in extraction: #54 geometry + soil + pile data exact; #106 geometry absent.

## V-042 Slide2 #36 — Li-Lumb (1987) / Hassan-Wolff (1999) probabilistic slope (F2)
- Source: Slide2 Verification #36 (manual pp. 137-139); Li & Lumb (1987); Hassan &
  Wolff (1999).
- Target: `slope_stability.probabilistic` (fosm_fos + monte_carlo_fos + lognormal RI).
- Geometry (Fig 36.1, LABELED, m): surface (0,5),(5,5),(15,15),(20,15); base el 0;
  homogeneous. Table 36.1 (mean, std): c'=18±3.6, phi=30±3, gamma=18±0.9, ru=0.2±0.02.
  Bishop; variables normal; FOS lognormal.
- Published (Table 36.2): Slide Bishop det-min FOS 1.340 (RI_LN 2.482) / overall 1.350
  (2.393); Hassan-Wolff 1.334 (2.336).
- Verdict: **PASS.** Searched Bishop critical FOS 1.325 (−0.7% vs H-W). FOSM COV_F
  0.1225 reproduces H-W's implied FOS-COV: lognormal_beta at the published FOS 1.334 →
  RI 2.30 (within 1.6% of 2.336); module's own F=1.325 → RI 2.244 (~4%). FOSM≈MC.
  c'-dominated variance (~70%). Needed the additive `ru` probabilistic variable.
- Confidence: high (labeled figure + full property table).

## V-043 Slide2 #39 — Tandjiria (2002) geosynthetic-reinforced embankment (F2)
- Source: Slide2 Verification #39 (manual pp. 147-150); Tandjiria (2002) problem 1.
- Target: `slope_stability` 2-material embankment + `Geosynthetic` reinforcement +
  tension crack; Spencer/GLE.
- Geometry (Fig 39.1/39.2, LABELED, m): surface (0,9),(10,9),(20,3),(30,3); soft-clay
  foundation el 0-3 full width; fill above el 3 x0-20. Sand fill c'=0/phi=37/gamma=17,
  soft clay c'=20/phi=0/gamma=20 (Table 39.2); clay fill both c=20/phi=0/gamma=19.4,
  water-filled crest crack (Table 39.1).
- Published: no-reinf Spencer sand circ 1.209/noncirc 1.188, clay circ 0.975/noncirc
  0.935; reinf force (FS=1.35) sand 44-45/56, clay 169-170/184-190 kN/m.
- Verdict: **PASS (sand no-reinf)** / **CONVENTION (reinf force, clay crack).** Sand
  no-reinf circular 1.180 (−2.4%), noncircular ~1.19 (~+0.2%). Geosynthetic force for
  FS=1.35 ~55 kN/m circ / ~50 noncirc — brackets published 44-56 but split inverted
  (surface-sensitive). Clay: without the unlabeled water crack 1.039 (+6.6% vs 0.975);
  full 2c/γ water column overshoots to ~0.27 — not reproduced, NOT tuned.
- Confidence: geometry + soil exact; crack depth + Slide2 critical surfaces absent.

## V-106 Slide2 #106 — Cai & Ugai (2000) Ito-Matsui pile — N/A (source)
- Source: Slide2 Verification #106 (manual p. 334); Cai & Ugai (2000).
- Verdict: **N/A (source).** Fig 106.1 is an unlabeled raster (no coords, no soil/pile
  properties); geometry + Ito-Matsui pile params live only in the external Cai & Ugai
  paper. Cannot recover without reconstructing/fitting. The Ito-Matsui 1975 force is
  unit-tested vs the paper and a with-pile FOS is validated at V-040 — source-coverage
  gap, not a capability gap. Recorded, not pinned.

## V-044 Slide2 #70 — Duncan & Wright (2005) submerged slope + ponded water (F2)
- Source: Slide2 Verification #70 (manual pp. 233-236); Duncan & Wright (2005) Fig 6.27.
- Target: `slope_stability` submerged slope, ponded-water buttress, circular search.
- Geometry (Fig 70.1/70.2, LABELED, ft): surface (0,15),(30,15),(105,45),(140,45);
  base el 0; homogeneous γ=128 pcf, c'=100 psf, φ'=20°. Fully submerged; ponded water
  at el 75 ft (Case 1) / el 105 ft (Case 2).
- Published: Bishop 1.603 / Spencer 1.599 / GLE 1.599 (circular), ref 1.60; Case 1 = Case 2.
- Verdict: **PASS.** On the Bishop centre-grid critical circle: Bishop 1.597 / Spencer
  1.598 / GLE 1.595 (within 0.4%). Water-level INDEPENDENCE reproduced to ~7 sig figs
  (same circle → 1.5974 at water el 75/105/45). A free Spencer/entry-exit search hits a
  spurious degenerate surface on this ponded geometry; Bishop centre-grid is the robust
  path (Slide auto-refine). Tests: test_published_v044_slope_submerged.py (2).

## V-085/086 Slide2 #85/#86 — Duncan & Wright grouted tiebacks — DEFERRED (search robustness)
- Source: Slide2 #85 (Fig 6.34) / #86 (Fig 7.28); Duncan & Wright (2005).
- Geometry recovered (labeled): #85 saturated clay c=350 psf/φ=0, single 9,000 lb/ft
  support at mid-height, steep 0.5:1 20-ft face; #86 c'=0/φ=37 fill on rigid rock, five
  800 lb/ft tiebacks (el 4/8/12/16/20 ft, 20 ft long).
- Verdict: **DEFERRED (search robustness).** Both blocked by the circular search
  converging to spurious degenerate surfaces on steep / thin-layer / c'=0 sections
  (centres outside the domain or below the rigid base; #86 c'=0 toe circles at 0.55,
  below the 0.94 infinite-slope value) that the supports don't engage. Not tuned/pinned;
  flagged as a search-robustness follow-up (physical-centre / below-base rejection guard,
  like the E1 drawdown-search gate). Anchor mechanic itself exercised at V-043.

---

# Looked for but NOT extracted (and why)

- **Liquefaction (liquepy_agent / seismic_geotech NCEER)** — no clean published worked example with complete inputs + FS values found in the local library (UFC/DM7/GEC refs have none); Boulanger & Idriss (2014, UCD/CGM-14/01) presents case-history plots, not a reproducible single-profile worked calc; Youd et al. (2001) likewise. Best future option: NCHRP/TRB "Pile Design for Downdrag" Appendix G design example (nap.nationalacademies.org/read/27864/chapter/8) pairs a CPT liquefaction calc with downdrag — needs a dedicated extraction pass.
- **reliability (FOSM/PEM/FORM)** — Duncan (2000) already validated; no other complete worked example with published beta/Pf in the local refs. USACE ETL 1110-2-547 examples are a candidate (public) but were not fetched in this pass.
- **Caltrans Example 10-5 (translational slide, FS = 0.47)** — complete inputs and answer extracted, but the source uses a simplistic three-block hand method (Pa = W1 tan(alpha_a - phi), Pp = W3 tan(alpha_p + phi)) whose FS is not comparable to our Spencer/GLE noncircular solver on the same surface; dropped to avoid a false-failure benchmark.
- **Micropile Manual Ch. 6.7 slope-stabilization design example** (Hreq = 650 kN/m etc.) — inputs span several charts/figures (Mult tables, p-y outputs) and the answer depends on a SLIDE/SLOPE-W run with the pile force; too entangled for a module-level check.
- **GEC-13 Vol 2 column-supported embankment (GeogridBridge Examples 1-2) and mass-mixing Ch. 7 example** — methods (load-transfer-platform geogrid tension; deep-mixed shear walls) not covered by our ground_improvement module.
- **GEC-8 (CFA piles)** — chapter 6 has a staged design discussion but the numeric design example tables are figure-images in the PDF (scanned tables); extraction would need OCR/vision pass; deferred.
- **DM7.1/DM7.2 example problems** — the worked examples are embedded in drafting-style figures (vector drawings with scattered text); text extraction scrambles them. A vision pass over the DM7 figure catalog could recover the classic cantilever-sheet-pile and consolidation examples later.
- **GEC-12 Vol 2 Ch. 12 wave-equation example (Figure 12-14)** — graphical bearing-graph output only; the Vol 3 drivability tables (V-003) are strictly better.
- **GEC-5 / GEC-9 / GEC-14 / UFC refs / FHWA pavements** — scanned for example keywords; no self-contained numeric worked examples relevant to our modules (GEC-14 has none; UFC examples are tables of requirements, not calcs).
- **Itasca M-O wall / dynamic verifications** — fem2d has no dynamics; skipped. PLAXIS verification set not needed — Itasca pair (V-023/V-024) covers fem2d adequately given its existing validation depth.
