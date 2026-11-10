# Equation judgment calls: the lead's rulings (2026-11-10)

Owner, 2026-11-10: "for all equation judgement calls, go with whatever you
think is best, but file away a 'later' backlog item for me to revisit it in
the future". Each row below goes into ONE "later" BACKLOG item for the owner
(with a desk card), once the fix branches report their final wording. In
every case the function's description and result state the printed form, the
coded form and the page.

| # | Where | What the print shows | Ruling | Basis |
|---|---|---|---|---|
| 1 | DM 7.1 Eq. 3-2 `relative_density_from_dry_density` (PDF 169, printed 141) | second form, as typeset, has gamma_d-max in the numerator's bracket: negative for any soil looser than its densest state | keep the code (gamma_d-min in the numerator) | the print sets the two forms equal; only the coded form makes them equal |
| 2 | DM 7.1 Eq. 4-7 `trench_load_coefficient` (PDF 223, printed 195) | Cd = [1 - e^(-K mu' H/Bd)] / (2 K mu') | keep the code (2 in the exponent) | with zero wall friction the printed pair (Eq. 4-6, 4-7) gives half the backfill weight; as typeset rigid-pipe loads fall about 40 % |
| 3 | DM 7.1 Eq. 4-8 `flexible_pipe_trench_load` (PDF 225, printed 197) | Wc = Cd gamma_t Bc D with Bc and D both the pipe's outside diameter | keep the code (trench width for one of them) | higher load whenever the trench is wider than the pipe; as typeset is the unsafe side |
| 4 | DM 7.1 Eq. 8-44 `hydraulic_conductivity_benson_trast` (PDF 491, printed 463) | "w = molding water content", no unit; k in m/s | take the bracket term (initial saturation) in PERCENT | in Eq. 8-43 on the same page the same term has coefficient 2.87 and can only be a decimal; here it is 0.087, idle as a decimal (under 0.09 in ln k over the whole range), the same span as Eq. 8-43's in percent. Lead saw the page. Public source being sought (`newrefs`) |
| 4b | DM 7.1 Eq. 8-43 `hydraulic_conductivity_benson_landfill` (same page) | k "in m/s"; w no unit | keep as printed (w as a decimal, k in m/s), with a note that ordinary liner inputs give about 3.6e-8 m/s | the source regression may be in cm/s (would give 3.6e-10 m/s); cannot be settled from the page. Public source being sought |
| 4c | DM 7.1 Eq. 5-23 `drain_spacing_factor` (PDF 295, printed 267) | Fn = [n^2/(n^2 - 1)] ln(n) - (3n^2 - 1)/(4n^2) | follow the typeset form | the page's sentence on the approximation's error fits the old bracketed form slightly better (true from n 4.6 and 12.5 with the typeset form); under 0.1 % apart at n 20 to 50 |
| 4d | DM 7.1 Figure 5-6(a) (PDF 258, printed 230) | below H/B 1.5 the five curves run in one band; no rule between curves | band middle for every L/B (+/-0.03); linear in L/B between curves | what the print can give |
| 5 | Caltrans Eqs. 4-5-1 to 4-5-4 `max_allowable_slope_angle` (PDF 73-74) | Eq. 4-5-2 is not dimensionally consistent; angle can fall as cohesion rises (61.1 deg at c 400 psf, 46.9 deg at 800 psf) | keep as typeset; the result opens with a caution and says to check by a stability analysis | safe side; the cited 1997 ASCE paper is not held |
| 6 | EN 1997-1 Table A.10 for a number of tests the table does not print | n = 1, 2, 3, 4, 5, 7, 10 only, no interpolation rule | refuse, giving the neighbouring printed values | the print gives no rule |
| 7 | GEC-12 Table 7-10 (Brown's method), compression rows for Gravelly Sand to Boulders and Rock (PDF 306, printed 260) | Bb printed equal to Ab (0.888, 2.89); tension rows print 0.0 | return Bb = 0.0, with a note saying what is printed | every tension cell is 0.94 times its compression cell; with 0.888 Eq. 7-21 gives 27.5 ksf at N60 30 against Table 7-8's 2.4 ksf limit. Vibratory rows filed under tension, as typeset |
| 8 | FHWA-NHI-05-037 Table 5-34, CBR = 292/DCP^1.12 (PDF 231, printed 5-53) | "DCP = Penetration index, in./blow" | take the penetration in mm per blow (parameter `dcp_mm_per_blow`); refuse a CBR over 100 % | as printed 0.5 in./blow gives CBR 635 %; in mm the same blow gives 16.9, inside Table 5-28's 3 to 80. SETTLED FROM PUBLIC PRINTS: TM 3-34.48-2 (29 Feb 2016) Figure G-11, p. G-11, "CBR = 292 / PR^1.12" on an axis "DCP INDEX (mm/blow)"; WES Instruction Report GL-92-3 (1992), p. 17, "DCP is the penetration ratio in millimetres per blow". Renders in `sources/renders/` |
| 9 | FHWA-NHI-05-037 Table 5-34, CBR = 75/(1 + 0.728 wPI) | "P200 = % passing No. 200" | keep P200 as a decimal | with the percent the equation gives CBR under 1 for any plastic soil |
| 10 | GEC-7 Table 4.9 F_PGA (PDF 134, printed 94) | column headings PGA <= 0.25, 0.5, 0.75, 1.0, >= 1.25 | use PGA 0.1, 0.2, 0.3, 0.4, 0.5 | the headings are those of the Fa table (Ss); GEC-12 Table 7-27 (PDF 426, printed 380) prints the same values under PGA < 0.1 to > 0.5. Site class D at PGA 0.30: 1.2 (was 1.56). Lead saw both pages |
| 11 | UFC 3-250-11 (2020) Table 2-3 footnote (c) (PDF 21, printed 13) | "PI <= 20 + 50 - percent passing No. 200" | SETTLED FROM A PRINT: PI <= 20 + (50 - percent passing No. 200)/4 | the 16 January 2004 edition (body is TM 5-822-14/AFJMAN 32-1019, 25 October 1994), Table 2-1 footnote b, PDF p. 12, printed p. 2-3, typesets the fraction over 4; the 2020 and 2026 editions lost the divisor and point row 1C at the wrong note. Render: `sources/renders/ufc_3_250_11_2004_p12_footnote_b_crop.png`. At 10 % passing the limit is 30, not 60 |
| 12 | UFC 3-250-01 Eq. 20-3 `taylor_permeability` (PDF 113, printed 93) | (1 - e) in the denominator | keep the coded e^3/(1 + e) | the Taylor form; departure to be stated |
| 13 | AASHTO 1993 Fig 3.5 and Fig 3.6 closed forms (`relative_damage_rigid`, `k_loss_of_support`) | charts; no closed form printed | keep, disclosed as fits (up to 12.5 % off chart reads; LS = 1 matches the printed example) | used by the pavement calc module; re-fit is a later item |
| 14 | DM 7.2 Eq. 3-4 `relative_water_content` | Eq. 3-4 prints w_opt - w_field; footnote A of Tables 3-10, 3-11 defines w_field - w_opt | keep the code (follows the tables) | the tables' limits read that way |
| 15 | DM 7.2 Eq. 6-15 `alpha_coleman_CFA` | prints 0.53 (su/Pa) with no exponent | keep the coded -1 exponent | Figure 6-14 curve E |

| 16 | EM 1110-2-2107 Eq F.10 `hydrostatic_horizontal_component` (pdf 436, printed 427) | the typeset general equation gives -50.0 for the worked example; Table F.3 prints +50.0 | keep the code (sign that integrating the stated integral gives) | matches the manual's worked example and table |
| 17 | EM 1110-2-2107 theta_p `resultant_angle_from_horizontal` (pdf 436, printed 427) | typeset as tan^-1(Ph/Pv); the worked example uses tan^-1(Pv/Ph) = 0.293 | keep the code (Pv/Ph) | matches the worked example |
| 18 | EM 1110-2-2107 theta_Rt `trunnion_reaction_force` (pdf 445, printed 436) | typeset as tan^-1(Rtx/Rty); the worked example uses tan^-1(363/1336) = 0.2653; Table F.8 prints it signed | keep the code; state the departure and the sign | matches the worked example |

| 19 | Wood Handbook Eq 9-24 `torsional_shear_stress_rectangular` (pdf 244-245, printed 9-13) | T/(beta h b^2) | code beta T/(h b^2) | the print's own Figure 9-15 gives beta from 3 to 4.8, a multiplier; as printed the stress is 9 to 23 times too low |
| 20 | Wood Handbook Eq 9-32 `critical_ponding_spacing` | density of water "1000 kg/m3" in SI | use the unit weight 9,810 N/m3 in SI | a mass density with force-per-length inputs makes the spacing 9.81 times too large (unsafe); the inch-pound value printed is a weight density |
| 21 | Wood Handbook Eq 9-41 `elastic_buckling_stress_edgewise_bending` (pdf 249) | 1.44 E d2/(le d1) | code 1.44 E d2^2/(le d1), unless the page's own figure or text contradicts it (fixer to check) | as printed it is not a stress and its value changes with the length unit |

Fixed to the print on the struct branch after the formula re-read: UFC 3-301-01
Eq C-1 `rigid_pipe_fundamental_period` (length not squared, 1/pi missing; a
pipe was called rigid up to hundreds of feet of span); Wood Handbook Eq 9-22
compression cases (edges swapped) and Figure 9-14 coefficients (misread).
Also fixed to the print (not judgment
calls): em_2107 `westergaard_pressure` (Eq 4.3 has a 7/8 factor the code
dropped); em_2104 `earthquake_load_combination` Eq 3-5 (permanent loads
unfactored). Print inconsistencies noted, code follows the stated equation:
EM 2107 Eq 10.1 against F.2/F.3 (a factor of two in the seal friction term);
Eq 4.4 against Eq 4.6 (C = 0.75); EM 2104's D-5 example uses the 4-ksi
coefficient for 5-ksi concrete.

Not equation calls, also decided by the lead on the branches: ufc_backfill
rebuilt from the held 04FA (overturns REF-31's "stays 04N"); ufc_expansive
rebuilt from UFC 3-220-07; functions with no printed basis removed (gec-23,
gec-27, the Komornik-David and Seed functions, backfill Types I-III and
others).

Owner's decisions, 2026-11-10 (evening), in his words:
- New references: "for those 'references to add next', don't build those yet.
  Just put them on a 'later' backlog item." (EM 1110-1-1905 was asked for
  separately and its build continues.)
- DM 7.2 reissue: "Agreed with not rebuilding DM 7.2 (I believe the same
  redesignation happened for 7.1)." Lead's reading: move the citation to UFS
  3-220-20 (2026), no rebuild. DM 7.1 is already cited as UFS 3-220-10 (2026).
- Copyrighted PDFs tracked in the public repo: "The copyright PDFs are OK in
  the repo. I downloaded these freely off the web; my understanding is
  they're on there because they're adopted law." No action.
- Release: "You can release when everything is merged and checked."
