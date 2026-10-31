# ground_improvement — Ground Improvement Methods (GEC-13)

## Purpose
Design tools for four ground improvement techniques: aggregate piers,
wick drains, surcharge preloading, and vibro-compaction.

## References
- FHWA GEC-13 (FHWA-NHI-16-027, Ground Modification Methods), Vols. I and
  II: checked against the print in GM-291 (module_work/gm291/CHECK.md)
- UFC 3-220-10, Soil Mechanics (2026): the smear factor (Eq. 5-25) and the
  combined vertical and radial consolidation (Eq. 5-27, Carrillo 1942)
- Priebe's curve as GEC-13 prints it (Fig. 5-21 curve (4), p. 5-36): the
  module's only Priebe basis since GM-276; no Priebe formula

## Files
- `aggregate_piers.py` — area replacement, SRF, composite modulus, bearing improvement
- `settlement_ratio.py` — the settlement ratio off GEC-13 Fig. 5-21: by E_c/E_s (weighted modulus, Balaam) and the Priebe curve (4)
- `wick_drains.py` — radial consolidation by GEC-13 Eq. 2-1, drain spacing design (bisection)
- `surcharge.py` — preloading with/without drains (imports settlement.time_rate)
- `vibro.py` — vibro-compaction feasibility (fines content, SPT N by Table 4-8) and the probe spacing off Fig. 4-34
- `feasibility.py` — decision support: soil + problem -> method recommendations
- `results.py` — 5 result dataclasses

## Public API
```python
analyze_aggregate_piers(B, L, q_applied, ...) -> AggregatePierResult
analyze_wick_drains(cv, ch, drain_spacing, ...) -> WickDrainResult
analyze_surcharge_preloading(target_consolidation, ...) -> SurchargeResult
analyze_vibro_compaction(fines_content, initial_N_spt, target_relative_density=None, ...) -> VibroResult
evaluate_feasibility(soil_type, fines_content, ...) -> FeasibilityResult
```

## Key Notes
- Wick drains (GM-291): GEC-13 Eq. 2-1 (Rixner et al. 1986a), t = (d_c^2 /
  8 c_h) [F(n) + F_s + F_r] ln[1 / (1 - U_h)], with F(n) = ln(n) - 0.75
  (Eq. 2-2) and d_c = 1.05 s or 1.13 s (p. 2-33). Smear is ignored by
  default (smear_ratio 1.0), as GEC-13 p. 2-35 says it often is; above 1,
  F_s = (k_h/k_s) ln(s), UFC 3-220-10 Eq. 5-25 as printed (GEC-13 prints no
  F_s formula). Until GM-291 the defaults were smear_ratio 2.0 and k_h/k_s
  2.0 with Hansbo's ln(n/s) + (k_h/k_s) ln(s) - 0.75, printed in neither
  manual. Well resistance is not modelled: "rarely significant, and,
  therefore, ignored except for extremely deep drains" (p. 2-36). The drain
  diameter defaults to 2.5 in, "the most common" (p. 2-35; was 0.066 m).
  GEC-13 Sec. 4.4's example (p. 2-38), "on the order of 300 days" at 8 ft
  and "500 days" at 10 ft, runs at 299 and 503 days on the defaults.
- Combined consolidation: U_total = 1 - (1-Uv)*(1-Ur), Carrillo as UFC
  3-220-10 Eq. 5-27 prints it. GEC-13's Eq. 2-1 is radial only: vertical
  drainage "should be considered for major/complex projects" (p. 2-33) and
  "for a conservative approach, no beneficial effects may be considered"
  in preliminary feasibility evaluations (p. 2-34). Results give U_r beside
  U_total; design_drain_spacing targets U_total.
- Wick drains use bisection to find spacing for target consolidation
- feasibility.py takes plain floats, NOT SoilProfile objects
- Vibro (GM-291): the fines screen as GEC-13 prints it, below 10 % ideally
  suited (Fig. 4-27 zone B, p. 4-67), "more than 12 to 15 percent silt OR
  more than 2 percent clay" and densification "generally cannot be
  achieved" (p. 4-65): the module reads 12 to 15 % as marginal and above
  15 % as not feasible (was 15 to 20 % marginal, above 20 % not feasible,
  printed nowhere). The clay fraction is not an input. SPT N is read as a
  density by Table 4-8 (p. 4-74); very dense (N above 50, D_r 85 to 100 %)
  is not feasible, since p. 4-75 finds D_r above 85 % unlikely to be needed
  (was any N above 25). The D50 screen (0.1 / 0.2 mm) and the probe spacing
  from a target SPT N (a rule of thumb, 3.0 m to 1.5 m) are gone, neither
  printed: the spacing is read off Fig. 4-34 (p. 4-77, silty sands, the
  lower bound gradation) by `target_relative_density`, 55 to 90 %, refused
  outside it (`vibro.FIG_4_34`, traced off the printed chart and drawn back
  over it). The adapter refuses D50 and target_N_spt with the reason.
- Feasibility typical sizes (GM-291) are GEC-13's: PVD spacing 3 to 6.5 ft
  (p. 2-33), drain diameter 1.5 to 5.5 in, most commonly 2.5 in (p. 2-35);
  stone columns 1.5 to 4 ft, spacing 5 to 12 ft (p. 5-31); stress ratio 2.0
  to 5.0 measured, 2.5 for preliminary design (pp. 5-31, 5-32), rammed
  aggregate piers 5 to 10 and 10 (p. 5-43). GEC-13 prints no typical
  surcharge height. The aggregate-column strength screen is p. 5-9's:
  refused above 2000 psf (96 kPa), the top of the stone columns' practical
  upper limit, noted within 1000 to 2000 psf and below 300 psf, with
  special care in organics under 300 psf (was 10 to 100 kPa, an organic
  soil under 10 kPa refused). The N_spt < 15 screen for a granular soil is
  not printed and is kept.
- `improved_bearing_capacity` is a FIRST-ORDER estimate using the
  equal-strain stress-concentration factor 1 + as·(n−1) (GI-2). It is not
  the Priebe curve, which GEC-13 prints as a settlement ratio (Fig. 5-21).
- `equivalent_drain_diameter = 2(w+t)/π` (GEC-13 Eq. 2-3; GI-1 fixed in 5.0)
- Aggregate piers state their choices (GM-142, 2026-10-18): n (default
  5.0; GEC-13 for stone columns: measured "between 2.0 and 5.0", p. 5-31,
  and "a ratio of 2.5 is often conservatively used" for preliminary
  design, p. 5-32; for rammed aggregate piers 5 to 10 under embankments and
  10 under rigid footings, p. 5-43; "stress concentration ratios range
  from 4 to 45 for rigid footings", p. 5-45; taken as given, not worked out from the moduli), E_soil (used only for the composite modulus) and the pattern. The calc
  package no longer prints E_column as an input. Since GM-179 the tool
  refuses E_column (and its other names column_modulus, E_pier) with the
  stress-ratio settlement method: there it changes no result, only a
  warning below E_soil.
- Settlement by E_c/E_s (GM-220, 2026-10-20): `settlement_method`
  'stress_ratio' (the default, SRF = 1/[1 + a_s(n - 1)] from n, unchanged)
  or one of GEC-13 (FHWA-NHI-16-027) Fig. 5-21's (p. 5-36) settlement
  ratio curves (settlement without columns over settlement with them,
  against A/A_c; p. 5-36 prints it as "n", which is not the stress ratio n,
  so the module calls it the settlement ratio, GM-279): 'weighted_modulus',
  curve (2), settlement ratio = 1 + a_s(E_c/E_s - 1),
  any E_c/E_s (traced, within 1 % of (2a) up to A/A_c = 5 and 2.2 % beyond,
  within 0.6 % of (2b)); 'balaam_rigid', curves (5b) E_c/E_s = 10 and (5c)
  5, A/A_c 1.75 to 9, linear in E_c/E_s between them; 'balaam_flexible',
  curve (5a) E_c/E_s = 10 only, A/A_c 1.5 to 8.5. The Balaam curves were
  traced off the rendered page, calibrated on the grid lines, checked by
  overlay and tabulated at A/A_c steps of 0.25 to 0.5; between nodes it is
  linear in a_s (within 0.5 % of the trace). Off the drawn ranges they are
  refused, not extended. p. 5-37: the FHWA (1983) Equilibrium Method "is
  roughly equivalent to the Balaam relationships shown in Figure 5-21 and
  represents an average or lower bound estimate suitable for preliminary
  analyses"; the Priebe curve "may be used to evaluate the upper bound
  effectiveness". These methods need E_column (and the tool, E_soil);
  then SRF = 1 / the settlement ratio, S = S_unreinforced / the settlement
  ratio, E_comp = E_s times it, and n sets
  only the improved bearing (a choice only with q_unreinforced). GEC-13
  has no worked example by these curves (Example 2, p. 5-49, reads the
  Priebe curve).
- Priebe (GM-221, GM-276): `priebe_improvement_factor` reads GEC-13's
  printed Priebe curve, Fig. 5-21 curve (4), phi = 37.5 deg
  (`settlement_ratio.PRIEBE_CURVE_4`, A/A_c 1.6 to 8.0, traced and drawn
  back as the Balaam curves, +/- 0.03), and nothing else. GM-221 added the
  curve beside Priebe's basic n0 formula (default phi_column 42.5, printed
  nowhere in GEC-13, 20 % above the curve at Example 2); the owner's answer
  to OWN-70 (2026-10-21): "just the printed curve. discard your formula.
  don't trust formulas from your memory!" GM-276 removed the formula,
  `priebe_basic_improvement_factor` and the `method` choice; the tool
  refuses `phi_column`, `nu_soil` (and their other names) and `method`, each
  with what it uses now, and returns `settlement_ratio` (was `n0`) with
  `phi_column_deg` 37.5. Off A/A_c 1.6 to 8.0 it is refused, not extended.
  GM-276 checked the table on both printed copies: on p. 5-36 it sits on the
  line within 0.015 (95 % of reads) from A/A_c 2.25 to 8 and within 0.04
  below A/A_c 2.25, where the curve is steep; Fig. 5-27 (p. 5-49) agrees.
  From A/A_c about 4.0 the printed (4) runs on one line with (5a). The
  printed curve runs on past 8.0, to about 8.5 to 8.7 where the lines end;
  the table stops at 8.0 and the tool refuses past it, saying so. Example 2: at
  A/A_sc = (5.7/3.0)^2 = 3.61 the curve reads 2.55 on both copies, so 27 in
  becomes 10.6 in; GEC-13 prints 2.7 and 10 in, its arrows (Fig. 5-27)
  drawn at A/A_sc 3.53, settlement ratio 2.63 on the curve, and labelled
  3.6 and 2.7.
  The owner's note on OWN-70, "print hard number, leave to engineer to decide
  with note": the result leads with the curve value, and its note and choice
  quote GEC-13 as printed (`PRIEBE_GEC13_WORDS`): p. 5-37, the Priebe curve
  "may be used to evaluate the upper bound effectiveness and cost at various
  spacings" for preliminary estimates, and the Equilibrium Method, roughly the
  Balaam curves, "an average or lower bound estimate"; p. 5-27, theoretical
  results "supplemented by past experience and sound engineering judgment".
  The design value is the engineer's. A test checks each quoted sentence on
  its page of the PDF when the PDF is present.
  The settlement and composite-modulus steps cited "Priebe (1995)": they
  are GEC-13's stress reduction method (Sec. 4.1.2.6, p. 5-36) with the
  soil stress by Eq. 5-4 (p. 5-32), on the unit cell under equal strain
  (Secs. 4.1.2.1, 4.1.2.4); GEC-13 names Priebe (1995) only as a design
  method (p. 5-27) and for its curve.
- Feasibility screening (GM-179): the depth to the bottom of the
  treatment, depth_to_top_m + thickness_m, is held against GEC-13's
  printed depth limits: aggregate columns "Typically, the maximum
  practical depth of stone columns and rammed aggregate piers is 100 feet
  and 35 feet respectively" (p. 5-9, item 7; 30.5 m and 10.7 m), which
  replaces an uncited 12 m thickness limit; vibro-compaction "a depth of
  about 165 feet, with generally recommended depth of about 100 feet"
  (p. 4-65; 50 m and 30.5 m). The water table is refused: no screen here
  uses it, and GEC-13 finds vibro-compaction "effective both above and
  below the natural groundwater level" (p. 4-65).
- Tests in tests/test_ground_improvement.py; the printed examples and
  screens of GM-291 in tests/test_gec13_print.py
