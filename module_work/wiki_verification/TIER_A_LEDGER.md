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

## Verified at Tier B already (wave 1)

| Item | Verdict |
|---|---|
| S5 Bray & Travasarou 2007 | **CONFIRMED-PRIMARY, zero discrepancies** — every coefficient (incl. rigid a0=-0.22 via printed Eq. 6), both published worked examples reproduced. Report: `bray_travasarou_2007.md`. Docstring caveat retired in `slope_stability/newmark.py`. AUDIT FLAG CLOSED. |

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
