# WikiLLM Tier-A ledger — wishlist items vs the library's extracted knowledge

*2026-07-18. Tier A = classification from the wiki's LLM-extracted records
(`summary`/`key_parameters`/`key_takeaways` in `library.db`, 7,323 records) —
zero PDF reads. Tier-A corroboration upgrades provenance NOTES; any code change
still requires a Tier-B page check. Wishlist item codes refer to
`module_work/provenance_audit_other.md` (O-) and `_slope.md` (S-).*

## ANSWERED / CORROBORATED by wiki records

| Item | Finding | Action |
|---|---|---|
| S4 DWW 1990 (rapid drawdown case histories) | Record holds the published FOS set (Pilarcitos: Corps 0.82, Lowe-Karafiath & new method 1.05 — our V-052 targets confirmed) AND the real geometry our validation lacked: **"~78 ft high, upstream slope ~2:1 (lower 58 ft) then 3:1"** (vs our assumed uniform 2.5H:1V), γm≈135 pcf, k≈4e-8 cm/s, drawdown 1.7 ft/day; Walter Bouldin description; back-calculated F=1.2/1.0 under Morgenstern assumptions; T=cv·t/D² ≥3 → drained rule. | V-052 note updated (see below). Optional Tier-B: re-run Pilarcitos on the true two-slope section (agent already located printed p. 262 Fig. 3). |
| O8 Duncan 2000 (reliability) | Record confirms LASH FMLV=1.17, VF=16%, Pf=18% (V-030 targets), the lognormal β formula (as implemented in `reliability/`), the 3-sigma rule, and COV ranges (Su 13-40%, φ 2-13%, γ 3-7%, Cc 10-37%) matching our COV-database rows. | Corroborated; per-row COV audit vs the PDF = optional low-priority Tier-B (three copies in library, hydrated paths known). |
| O15 ASTM A615 (rebar) | Records (2012 + 2008 editions) confirm grades/strengths/size range No. 3-18 and that nominal dimensions live in Table 1. | Grade data corroborated. `_REBAR_AREAS` numeric check = small Tier-B (text-layer PDF, cheap). |
| O5 PTI DC35.1-14 (anchors) | Record confirms strand basis (0.6-in A416), A981 bond ≥8,000 lb at 0.01-in slip, service-life definitions, grout-cover minima. | Corroborated for those values. The audit's target — presumptive grout-bond stress tables — not in record → Tier-B page check when soe anchor work resumes. |
| O6 M-O anchor | Library holds seismic-wall docs with printed P_AE form and the K_AE ≈ K_A + (3/4)k_h approximation ("Seismic Design and Behavior of Gravity Retaining Walls"); log-spiral caveat (M-O unconservative for passive when δ>φ/2) also captured. | Anchor SOURCE located; a printed K_AE table still needs Tier-B to anchor our M-O implementation numerically. |
| S8 Ito-Matsui | 1977 DISCUSSION is in the library (record confirms it re-derives the plastic-deformation solution and the Caquot corresponding-states treatment; original is Soils & Foundations 15(4) pp. 45-59). 1975 original NOT in wiki. | Partial: keeps SECONDARY tier but with an in-library primary-adjacent source; Tier-B on the discussion could check Eq. forms. |

## Verified at Tier B

| Item | Verdict |
|---|---|
| S5 Bray & Travasarou 2007 (wave 1) | **CONFIRMED-PRIMARY, zero discrepancies** — every coefficient (incl. rigid a0=-0.22 via printed Eq. 6), both published worked examples reproduced. Report: `bray_travasarou_2007.md`. Docstring caveat retired in `slope_stability/newmark.py`. AUDIT FLAG CLOSED. |
| S12 + S2 LE originals (wave 2) | **CONFIRMED-PRIMARY on every core kernel, zero equation discrepancies**: Bishop 1955 m_alpha/FOS (p. 10 Eqs. 12/13), Spencer 1967 Eq. 5 kernel + F_f=F_m crossing (pp. 14-15), M-P 1965 X=λf(x)E (p. 85 Eq. 18) with the GLE engine classified EQUIVALENT-NOTATION (Fredlund-Krahn discrete form, identical statics), EM 1110-2-1902 App. G Eq. G-12 (p. G-8, exact identity) + min(R,S) rule (p. G-2). Report: `le_method_originals.md`. Docstring provenance upgraded in methods.py/gle.py/rapid_drawdown.py. |
| O8 Duncan 2000 COV database (wave 6) | **VALUES PERFECT — all 15 Duncan-attributed COV rows digit-for-digit (both endpoints) + 3 formulas symbol-for-symbol** vs the in-hand paper (Table 3, p. 310). Five CITATION-level fixes applied: "Table 1"→"Table 3" throughout, bogus "Kulhawy & Trautmann (1996)" attribution → Harr (1984)/Kulhawy (1992) per print, std_from_range N<6 wording corrected (Christian-Baecher discussion, not the 2000 paper), beta_normal co-citation dropped. Report: `duncan_2000_cov.md`. reliability suite 176 green. |
| Kf c'>0 ticket (lead, closed) | **FIXED**: `_Kf` implements printed Eq. G-8 (EM p. G-7) for c'>0, reduces exactly to G-7 at c'=0 (published validations byte-identical); 4 new unit tests incl. printed-equation hand check; V-037/041/048/052 green (31 passed). |
| O7 + O15 + S6 small checks (wave 5) | **3/3 CONFIRMED, zero discrepancies**: ACI Ec=4700√f'c anchored in-hand to 318-08 §8.5.1 (57,000√f'c psi form, p. 111; composite_section.py citation upgraded); ASTM A615-12 Table 1 rebar areas 11/11 exact vs `_REBAR_AREAS`; Das 3rd Ed Eq. 7.51 (p. 434) anisotropic su form algebraically identical to `_anisotropic_su` (45° α-vs-i convention stated correctly; ADP generalizations deliberate+documented). Report: `small_checks_aci_astm_das.md`. |
| O6 AASHTO site factors (wave 4) | **CONFIRMED — 75/75 cells match** AASHTO LRFD 5th Ed (2010) Tables 3.10.3.2-1/2/3 (printed pp. 3-88/89; 9th-Ed copy is DRM-locked, tables edition-stable, renumbered 3.10.3.1-x). Interpolation + Class-F footnotes conform; Fpga=Fa aliasing faithful to print. Report: `aashto_site_factors.md`. site_class.py provenance upgraded. |
| O2 COM624P Reese sand (wave 3) | **REAL DISCREPANCIES FOUND AND FIXED**: the audited A_c=0.55 deep asymptote was WRONG (0.55 is B_c's; Fig 3.12 prints A=0.88 for both loadings), both cyclic tables were v5.3 reconstructions matching the manual at no depth (real A_c/B_c are low hump curves), A_s mid-depths ran +0.12..+0.24 high, k submerged medium/dense −32/−35% vs printed Tables 3.4/3.5, and the "Figs 2.19/2.20 / Table 2.2" citations were stale. ALL FIXED: four tables re-digitized from Figs 3.12/3.13 (page-calibrated reads), k anchors set to printed values, citations corrected, tests updated to the printed asymptotes (127 lateral_pile + V-017 + funhouse green). Report: `reese_sand_com624p.md`. Scope: opt-in `construction="reese1974"` branch + `_sand_k_recommendation`; default "simplified" branch untouched. AUDIT RED-FLAG CLOSED. |

## Follow-up tickets — BOTH CLOSED (2026-07-19)

1. ~~rapid_drawdown Kf for c' > 0~~ — **FIXED**: `_Kf` implements printed
   Eq. G-8; reduces to G-7 at c'=0 (published validations byte-identical);
   V-037/041/048/052 green.
2. ~~V-052 true-section re-run~~ — **DONE (V-052b)**: on the real 2:1/3:1
   section, Corps 0.726 (published 0.823, failure reproduced), LK 1.279 / DWW
   1.266 (published ~1.05) — overshoot shrinks ~29% → ~22%, residual isolated
   to the steep phi'=45 Kc gain (moderate-phi' #98 matches ~10%).

## Wave 7 (final) — EM 2906 / PTI / M-O

| Item | Verdict |
|---|---|
| O14 EM 1110-2-2906 pile_group basis | **EQUIVALENT-NOTATION confirmed**: the EM Ch. 4 Saul/CPGA stiffness method (pp. 4-29/30/31/38/45) is the engine's K = ΣBᵀkB in congruence-transform form; PG-1 simplifications already documented. Cosmetic "Eq. 4-1" citation in calc_steps.py corrected (the EM's Ch. 4 equations are unnumbered; the simplified formula isn't printed there). rigid_cap.py provenance upgraded. |
| O5 PTI bond stresses | **FUTURE-ANCHOR recorded**: module codes GEC-4 Table 4 values (correctly cited; PTI's Tables C6.1-C6.3 printed pp. 47-49 hold DIFFERENT bins — 9 representative values recorded in the report for a future GEC-4-vs-PTI presentation). Bond-length equation + FS=2.0 match PTI §6.7/§6.6. |
| O6 M-O numeric anchor | **CONFIRMED TO THE POUND**: Whitman (1990, ASCE GSP 25) Fig. 11 worked wall (printed p. 831) — φ=35, kh=0.2, H=25 ft → printed P_A=8892 / ΔP_AE=4088 lb/ft reproduced exactly by `mononobe_okabe_KAE` (K_AE=0.3956); Seed-Whitman K_A+(3/4)kh fallback symbol-exact. NEW anchor test in seismic_geotech tests. Long-open "no published M-O anchor" audit gap CLOSED. |

Report: `wave7_em2906_pti_mo.md`.

## Remaining (small, for a future session)

- Optional: Slide2-manual-copy hunt for the raster-recovered geometries.
- Optional: Duncan 2000 closure/Christian-Baecher discussion PDF (N<6 rule).
- Optional: GEC-4-vs-PTI bond-value comparison exhibit for soe anchor reports.

## NEEDS-PAGE-CHECK (records too shallow for the specific values)

- S12 Bishop 1954 / Spencer 1967 / M-P 1965 equations — originals in library
  (`.../Seepage/Resources/Papers/Spencer.pdf`, `M_P.pdf`; Bishop under
  `01/1. Geotechnical/8. Soil Mechanics/_Unsorted papers/`). Wave-1 salvage: M-P
  text extracted clean; Bishop is a scan (page renders); Spencer eq-page pending.
- S2 EM 1110-2-1902 App G (Eq G-12, min(R,S) rule) — reference-tier record.
- O2 COM624P Reese sand A/B/k tables — wave-1 salvage: page offset +22, sand
  section = PDF pp. 362-369.
- AASHTO site factors — 9th Ed copy is FileOpen-DRM-encrypted; use the 2010 5th
  Ed copy (`02 Tech/AASHTO/2010 LRFD Bridge Design Specifications - 5th
  Edition.pdf`), note edition caveat.
- O14 EM 1110-2-2906 (pile_group basis) — reference-tier records, both copies.
- S6 Das Advanced Soil Mechanics §7.18 (anisotropic su) — reference-tier record.
- O7 ACI 318 Ec clause — 318-02/-05/-08 in library (not -19); clause stable, cheap
  text search.

## NOT-IN-WIKI (parked)

- O1 Caquot & Kerisel (1948) tables (DM-7.2 record in library is a lateral-pile
  excerpt; Kp-chart coverage unknown → could try Tier-B on it).
- O4 Peck, Hanson & Thornburn (1974) book (library has Piedmont papers that
  BENCHMARK the PHT settlement method's bias ~3x conservative — useful context,
  not the N-φ chart).
- S3 Lowe & Karafiath (1960) original. S1 Duncan-Wright-Brandon (2014) book.
- S7 Jibson (2007) original (but: Siyahi-Cetin-region 2011 error-analysis paper
  in library documents Jibson-2007's ≥82%-unconservative bias over 42 dam cases —
  worth citing in newmark docs as an applicability caveat someday).
- S9 Wolff & Harr (1987) Cannon Dam original (CGPR workshop volumes may embed
  related material — unconfirmed).

## Bonus finds (not on the wishlist)

- 1995 centrifuge study with p-y GROUP multipliers (3D: 0.8/0.45/0.3;
  5D: 1.0/0.85/0.7; efficiencies 0.74/0.93) — independent corroboration
  candidate for pile_group/lateral_pile p-multiplier defaults (GEC-12 Table 9-2).
- Vesic lecture set + cavity-expansion originals (1972) — would upgrade
  bearing-factor lineage docs if ever needed.
