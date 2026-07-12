# Source-Provenance Audit — slope_stability (5.5.0)

Owner-directed audit for the 5.5.0 release. Purpose: (a) make explicit, everywhere
a future engineer would look, WHERE each implemented equation/coefficient/anchor
came from and how much extraction risk it carries; (b) feed the owner's reference-
wiki integration — the gaps below are what the wiki should fill (see WIKI-WISHLIST).

**Scope (slope half):** `slope_stability/*` (methods, gle, rapid_drawdown incl. the
EM G-12 convention, newmark + bray_travasarou, probabilistic, reinforcement /
Ito-Matsui, anisotropic su, infinite slope, search guards) + `validation_examples/`
records.

## Classification legend

- **SOURCE TIER** — `PRIMARY` (original paper/manual PDF read directly, in hand) ·
  `SECONDARY` (authoritative vendor verification manual / USACE EM / textbook) ·
  `MEM+ANCHOR` (memory- or scout-relay-informed, but validated against a published
  anchor) · `MEM-ONLY` (memory-only, no anchor — RED FLAG).
- **EXTRACTION RISK** — `text` (PDF text-layer, low) · `vision/raster` (figure
  geometry recovered from a rendered/raster image — elevated) · `OCR` ·
  `hand` (hand transcription/derivation).
- **ANCHOR** — `source-worked-example` · `secondary` (vendor/textbook value) ·
  `limiting-behavior` (monotonicity/identity only) · `none`.
- **DOCUMENTED?** — is the provenance stated where a future engineer sees it
  (module docstring / DESIGN.md / VALIDATION.md / INVENTORY-RESULTS)?

Extraction methodology is itself documented: `validation_examples/INVENTORY.md`
header records PyMuPDF text extraction + page-image reads, with raw page-text dumps
and rendered figure PNGs retained under `validation_examples/extracts/`
(`dump_pages.py` / `find_kw.py`).

---

## 1. Core limit-equilibrium methods (`methods.py`, `gle.py`)

| Method | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| Ordinary/Fellenius (1927) | Fellenius 1927 (classic) | SECONDARY (textbook eqn) | text/hand | Duncan §7 verification; ACADS | DESIGN.md refs + test_duncan_verification |
| Bishop simplified (1955) | Bishop 1955, Géotechnique 5:7–17 | SECONDARY | text/hand | Duncan §7; ACADS 1(a); Caltrans #10-3/4 (V-015) | DESIGN.md refs |
| Spencer (1967), rigorous | Spencer 1967, Géotechnique 17:11–26 (via `gle.py`) | SECONDARY | text/hand | F&K-1977; Duncan §7 | DESIGN.md refs; gle.py docstring |
| **Spencer / M-P LEGACY (`methods.py`)** | self-declared **APPROXIMATION (SS-1)** — "NOT the textbook-exact Spencer / not an exact GLE"; legacy fallback IGNORES reinforcement | SECONDARY (approx) | hand | rigorous `gle.py` is the cited-exact path | **DOCUMENTED prominently** in methods.py (SS-1) + DESIGN.md |
| Morgenstern–Price / GLE (rigorous) | Fredlund & Krahn 1977; GeoStudio 2022 Ch 2-3; Duncan-Wright-Brandon Ch 6-7 | SECONDARY | text/hand | F&K-1977 method-comparison (excluded-as-already-validated per INVENTORY) | DESIGN.md; gle.py docstring |
| **Janbu (corrected, f0)** — in `gle.py` | Janbu 1973 + F&K 1977 + **Abramson et al. (2002) Fig 6.18 (f0 read from a FIGURE)** | SECONDARY | **vision/figure (f0 from a chart — elevated)** | Duncan §7 | gle.py docstring |

Rigorous methods (Bishop/Spencer/M-P/GLE) are standard published equations reproduced
from textbook form and anchored to Fredlund-Krahn 1977 + Duncan §7.6–7.7
(`test_duncan_verification.py`) + ACADS. **Two flags:** (a) the `methods.py` LEGACY
Spencer/M-P are self-declared approximations (SS-1) that ignore reinforcement — the
rigorous `gle.py` path is the cited-exact one (documented prominently in code); (b) the
**Janbu f0 correction factor is read off Abramson et al. (2002) Fig 6.18** — a
figure/chart read (elevated extraction risk), the one non-text extraction among the core
methods. Tier SECONDARY (textbook), not PRIMARY-in-hand for the original Géotechnique
papers. **DOCUMENTED: yes.**

## 2. Rapid drawdown (`rapid_drawdown.py`)

| Item | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| Corps 2-stage `min(R,S)` | USACE EM 1110-2-1902 App. G, Eq. G-18/19 | SECONDARY (EM read via web-archive PDF by scout, **not primary in hand**) | text (scout web read) | #95 Corps 1.347 (V-037, Slide2 verif — secondary) | DESIGN.md (extensive) + V-037 |
| Kc-interpolation (DWW/LK stage 2) | Duncan-Wright-Brandon 2014 Ch.9; EM G-12 | SECONDARY / MEM+ANCHOR | text/hand | #96 DWW 1.443 (V-038, secondary) | DESIGN.md + V-038 |
| Lowe & Karafiath (1960) 2-stage | L&K 1960 Pan-Am conf. — **original NOT in hand**; Rocscience Slide2 docs confirm "= DWW minus stage 3" | MEM+ANCHOR | text (vendor docs) | #98 LK 1.075 (V-048, secondary, geometry-limited) | DESIGN.md + V-048 |
| Duncan-Wright-Wong (1990) 3-stage | DWW 1990 H.B. Seed symposium — **original NOT in hand**; GEO-SLOPE/Slide2 manuals used | MEM+ANCHOR | text (vendor) | #96/#98 (secondary) | DESIGN.md + V-038/041 |
| **EM G-12 R-envelope convention call** | EM 1110-2-1902 App. G Eq. G-12 (scout web read) + **empirical decision** | MEM+ANCHOR | text + hand-derivation | empirically anchored: raw as-plotted inputs give LK-seepage 1.45 = published DWW 1.443; G-12 transform overshoots to 1.53 | DESIGN.md (explicit convention note) + commit 4f010b8 |
| Stage-3 Fellenius vs GLE normal | Duncan-Wright-Brandon Ch.9 | SECONDARY | hand | #96 gle-option 1.31/1.37 (V-038) | DESIGN.md; **owner-decided default stays 'fellenius'** (plan E2) |

Anchor extraction risk: #95/#96 dam face is **labeled** (Fig 95.1) — text/low; **#98
Walter Bouldin** section is **raster-recovered** (V-041, "recovered simplified section",
~10% geometry-limited — elevated); **#97 Pilarcitos** upstream slope is **NOT published
in prose → assumed 2.5:1** (V-052, CONVENTION, elevated); **#99 pumped-storage** zoning
is figure-only → **N/A** (V-053). **DOCUMENTED: yes**, including the honest EM-not-in-hand
and empirical-convention basis.

## 3. Newmark seismic (`newmark.py`)

| Item | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| Newmark (1965) rigid-block integration | Newmark 1965 (classic) | SECONDARY | text/hand | Slide2 #104 (V-039, secondary) | newmark.py docstring; V-039 |
| Jibson (2007) regression | Jibson 2007, Eng. Geol. 91:209–218 — **constants cited from the paper's Eq. 6** (not memory) | SECONDARY | text | #104 cross-check (secondary) + limiting-behavior | newmark.py docstring (Jibson 2007 Eq 6) |
| **Bray & Travasarou (2007) — flexible branch** | Bray 2007 Ch.14 Eqs 14.4/14.5, **chapter PDF in hand, text-layer extracted (PRIMARY)** | PRIMARY | text | **source worked example reproduced**: ky=0.14/Ts=0.4/Sa=0.48/M=7.2 → ln D 2.29, P(D=0) 0.01 | newmark.py docstring (cites JGGE 133(4) + Ch.14) |
| **B&T rigid branch (a0=−0.22, Ts<0.05)** | Bray 2007 Ch.14 (chapter in hand) — but the rigid coefficient is **NOT reproduced against a published rigid worked example** | PRIMARY-source / anchor-weak | text | **limiting-behavior only** (rigid<flexible ordering; no published rigid D reproduced) | newmark.py docstring provenance note (this audit's doc fix) |

**Honest flag:** the B&T chapter WAS in hand; the *flexible* branch is PRIMARY and
reproduces the paper's own worked example. The *rigid* branch (a0=−0.22 with the 1.5·Ts
term dropped for Ts<0.05 s) is NOT reproduced against a published rigid worked example —
only its limiting behaviour is tested — so its coefficient is documented as unverified
(confirm against the chapter/paper before relying near Ts=0). Jibson's constants are
cited from the paper's Eq. 6 (not memory). **Doc note added to newmark.py.**

## 4. Probabilistic FOS (`probabilistic.py`)

| Item | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| FOSM (Taylor series + cross-terms) | Duncan 2000 (JGGE 126(4)) Eq. 8; USACE ETL 1110-2-556; Baecher & Christian | SECONDARY | hand | Duncan #29 Pf 18% (V-030); H-W #36 β (V-042) | DESIGN.md; VALIDATION; probabilistic.py docstring |
| Monte Carlo (bivariate normal, Cholesky) | standard (Baecher & Christian) | SECONDARY | hand | FOSM≈MC agreement; #36 | DESIGN.md |
| Correlated scalar pairs | generalization of F1 su-law correlation | MEM+ANCHOR | hand | #34 Wolff-Harr Table 34.1 rho (V-045, capability only; full FOS N/A) | DESIGN.md (correlated-pairs bullet) + V-045 |
| lognormal_beta / normal_beta | standard reliability | SECONDARY | hand | #36 RI_LN 2.30 vs H-W 2.336 (V-042) | DESIGN.md |

Anchor extraction risk: #29 surface **pixel-traced** (V-030, labeled figure, elevated);
#34 section **rendered/no labeled coords → full FOS N/A** (V-045, capability validated
against the published rho only); #36 **labeled** (V-042, text). **DOCUMENTED: yes.**

## 5. Reinforcement (`reinforcement.py`)

| Item | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| Ito & Matsui (1975) stabilizing-pile force | Ito & Matsui 1975, Soils & Found. 15(4), Eqs 13/23 — code **explicitly rejects the Hassiotis secondary variant (cites DOI)** | SECONDARY (paper cited; not in hand) + anchor | hand | **hand-check against the paper's own values** (c=10/φ=20/z=5/D1=2/D2=1.5 → 105.079; φ=0 case → 146.683) + #54 Yamagami trend (V-040) | DESIGN.md; V-040; reinforcement.py docstring |
| Soil nails | FHWA GEC-7, Lazarte et al. 2003 | SECONDARY | text | (nails.py disconnected from pipeline) | DESIGN.md refs |
| **Anchors** | **NO literature source — capacity USER-SPECIFIED by design; no pullout/bond model** | n/a (by design) | — | #39 force-equilibrium mechanic (V-043) | reinforcement.py Anchor docstring already states "user-specified allowable tension… no pullout model — use SoilNail" (ADEQUATE) |
| **Geosynthetics** | **NO literature source — capacity USER-SPECIFIED by design; no pullout/durability reduction applied** | n/a (by design) | — | #39 Tandjiria geosynthetic force (V-043) | reinforcement.py Geosynthetic docstring already states "user is responsible for reducing for pullout/durability" (ADEQUATE) |

Ito-Matsui #106 (Cai & Ugai 2000) is **N/A(source)** — Fig 106.1 unlabeled raster,
params in the external paper (V-106). #54 (Yamagami) is labeled. The **Anchor and
Geosynthetic elements are the only reinforcement models with no paper/manual source —
by design their capacity is user-supplied** (there is no pullout/bond model to cite).
JUDGMENT: **adequate, no doc fix needed** — the reinforcement.py Anchor and Geosynthetic
class docstrings already state the user-specified capacity and the explicit absence of a
pullout model, where an API user reads them. **DOCUMENTED: yes.**

## 6. Per-layer strength models + infinite slope (`geometry.py`, `analysis.py`)

| Item | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| SHANSEP `su=S·OCR^m·σ'_v` | Ladd & Foott (1974) SHANSEP | SECONDARY | hand | per-slice hand-calc + OCR^m scaling (test_strength_models) | geometry.py docstring; test_strength_models |
| Generalized Hoek-Brown (instantaneous c-φ) | Hoek, Carranza-Torres & Corkum (2002) GHB | SECONDARY | hand | Balmer-consistency + GSI=100/D=0 classic-HB reduction (test_strength_models) | geometry.py docstring; test_strength_models |
| **Anisotropic su(α) formula** | Casagrande & Carrillo (1944) `su=su_H+(su_V−su_H)sin²(i)` — **via scout web research (Das Adv. Soil Mech. §7.18); NOT primary in hand** | MEM+ANCHOR | hand-derivation | isotropic anchor (Frontiers 13:1581457) 2.287 vs ~2.16-2.2, secondary; Bakklandet (IOP EES, NGI-ADP FEM) 21.5% vs 24%, CONVENTION | DESIGN.md (F6 bullet shows C-C + the sin(2α) reduction) + V-054 |
| **sin(2α) base-angle reduction + α convention** | **my hand-derivation** from C-C sin²(i) + the 45° failure-plane offset (i=α+45°) | MEM+ANCHOR | hand | exact isotropic identity (equal su's == mohr cu) | DESIGN.md + code docstring (α = failure-plane angle, sign convention) |
| Infinite-slope closed form | standard (Duncan & Wright) | SECONDARY | hand | Slide2 #79/#81 (V-035/036, exact) | DESIGN.md; V-035/036 |
| Tension crack (strength / truncation) | Slide2/UTEXAS convention (E4) | SECONDARY | hand | ACADS 1(b) #2 water-crack (V-026, <0.1% truncation) | DESIGN.md (E4); V-026 |
| Pore-pressure grid (TIN u(x,z)) | standard interpolation (E3) | n/a (numerical) | — | ACADS 5 #10 capability (V-029; published grid unrecoverable → demo only) | DESIGN.md (E3); V-029 |

**Honest flag:** the anisotropic su formula's provenance is a *scout web relay* of
Casagrande-Carrillo / Das, not a primary source in hand; the sin(2α) base-angle form
and the α sign-convention mapping are **my hand-derivation** from the cited sin²(i)
relation. Anchors are a *secondary* isotropic value and a *FEM* (not LEM) anisotropy
magnitude → CONVENTION. #105 (the Slide2 anisotropic verification) is **N/A(source)**:
its geometry lives in Slide2 Tutorial 32, not the manual. **DOCUMENTED: yes.**

## 7. Search guards & thrust line (`slices.py`, `search.py`, `analysis.py`)

| Item | Source | Tier | Extraction | Anchor | Documented? |
|---|---|---|---|---|---|
| SS-5 below-base rejection (generalized) | internal engineering logic (not a published claim) | n/a (numerical guard) | — | zero-shift proof on all pinned tests | DESIGN.md (SS-5) |
| SS-6 noncircular degenerate guard | internal | n/a | — | search-robustness tests | DESIGN.md (SS-6) |
| Thrust-line clamp to section | internal (physical bound: thrust ∈ [slip, ground]) | n/a | — | regression test (every boundary within section) | analysis.py docstring + commit 6559899 |

These are numerical/robustness guards, not physics coefficients — no external source
to cite; each is anchored by a zero-shift / within-section invariant test. **DOCUMENTED: yes.**

## 8. Validation-anchor extraction-risk roster (`validation_examples/`)

Elevated-risk (geometry recovered from a **raster/rendered figure**, not labeled prose):

| V- | Problem | Recovery | Verdict |
|---|---|---|---|
| V-027 | ACADS 1(d) #4 | layer boundaries **pixel-read** (~MODERATE conf) | geometry-limited |
| V-030 | Duncan #29 LASH | failure surface **pixel-traced** (labeled fig) | PASS (surface traced) |
| V-034/045-scope | Loukidis #63 / Wolff-Harr #34 | **pixel-reconstructed ±1.5 m** / rendered no-coords | geometry-limited / capability-only |
| V-041 | #98 Walter Bouldin | **recovered simplified section** (flat layers, riprap omitted) | PASS geometry-limited (~10% low) |
| V-052 | #97 Pilarcitos | upstream slope **NOT published → assumed 2.5:1** | CONVENTION |

N/A-from-raster (unrecoverable): #33 (V-046), #99 (V-053), #51 (V-051), #106 (V-106),
#34 full-FOS (V-045), #105 anisotropic (Tutorial-32 geometry).
Text/labeled (low risk): #95/#96 (V-037/038), #79/#81 (V-035/036), #36 (V-042), #70
(V-044), #2/#9/#10 ACADS (V-026/028/029), #85/#86 (V-049/050), Caltrans/GEC anchors.

**DOCUMENTED: yes** — every entry's extraction confidence + recovery basis is stated
in INVENTORY.md per-entry, with raw dumps/PNGs retained under `extracts/`.

---

## WIKI-WISHLIST — documents that would upgrade a tier

Ordered by how much a slope anchor/method would gain if the owner's reference wiki
surfaces the primary source:

1. **Duncan, Wright & Brandon (2014), *Soil Strength and Slope Stability*, 2nd ed. —
   Ch. 9 (rapid drawdown), Ch. 5 (LE methods), anisotropy discussion.** Would move
   Corps/LK/DWW, the Kc interpolation, and the LE methods from SECONDARY→PRIMARY and
   let us confirm the stage-3-normal question independently.
2. **USACE EM 1110-2-1902 (2003 *and* 1970), Appendix G** — primary in hand (currently
   read via a web-archive PDF by the research scout). Confirms Eq. G-12 and the Corps
   `min(R,S)` rule and the G-12 R-envelope convention first-hand.
3. **Lowe, J. & Karafiath, L. (1960),** "Stability of earth dams upon drawdown," 1st
   Pan-Am Conf. SMFE, Vol. 2 — the ORIGINAL. Currently inferred from Rocscience docs
   ("= DWW minus stage 3").
4. **Duncan, Wright & Wong (1990),** "Slope Stability During Rapid Drawdown," H. Bolton
   Seed Memorial Symposium, Vol. 2 — the ORIGINAL case histories (Pilarcitos, Walter
   Bouldin, pumped-storage), incl. the cross-section geometries we had to recover from
   raster / assume (Pilarcitos slope, Bouldin zoning).
5. **Bray, J.D. & Travasarou, T. (2007),** JGGE 133(4):381–392 (and Bray 2007 Ch. 14) —
   primary in hand to CONFIRM the rigid-branch coefficient (a0=−0.22) and reproduce a
   published rigid worked example (currently limiting-behavior only).
6. **Casagrande, A. & Carrillo, N. (1944),** "Shear failure of anisotropic materials,"
   J. Boston Soc. Civ. Eng. 31:74–87, and **Das, *Advanced Soil Mechanics* §7.18** — to
   move the anisotropic su formula from scout-relay to PRIMARY and confirm the sin²(i)
   convention independently of my hand-derivation.
7. **Jibson, R.W. (2007),** "Regression models for estimating coseismic landslide
   displacement," Eng. Geol. 91:209–218 — primary for the Jibson regression constants.
8. **Ito, T. & Matsui, T. (1975),** "Methods to estimate lateral force acting on
   stabilizing piles," Soils & Foundations 15(4) — the ORIGINAL (Eqs 13/23). Currently
   hand-checked against the paper's values, but the paper itself is not in hand.
9. **Wolff, T.F. & Harr, M.E. (1987),** Cannon Dam correlated c'-φ' study — the ORIGINAL
   (the Slide2 #34 section is a rendered no-coordinate figure; full FOS/Pf is N/A).
10. **Slide2 Verification Manual + Tutorial PDFs** (Rocscience) at higher fidelity —
    the *manual* is our SECONDARY anchor for #29/#36/#54/#57/#70/#79/#81/#95–99, and
    **Tutorial 32** holds the #105 anisotropic geometry that is currently N/A(source).
11. **Higher-resolution source figures / original papers' cross-sections** for the
    pixel-recovered geometries (ACADS 1(d) #4, Loukidis #63, Duncan #29, Walter
    Bouldin #98, Pilarcitos #97) — to retire the raster-recovery extraction risk.
12. **Fredlund & Krahn (1977)** and the original **Bishop (1955) / Spencer (1967) /
    Janbu** Géotechnique papers — to move the core LE methods to PRIMARY.

## Net assessment

No **MEM-ONLY (RED FLAG)** items found in the slope module: the citation sweep confirms
the CODE never claims a memory-basis coefficient — every equation traces to a named
source (methods with journal+pages, gle F&K 1977 / GeoStudio 2022 / DWB, rapid_drawdown
EM 1110-2-1902 / L&K 1960 PACSMFE / DWW 1990 / DWB Ch 9, newmark Jibson 2007 Eq 6 + B&T
2007 Ch 14, probabilistic Duncan 2000 Eq 8 / ETL 1110-2-556 / Baecher & Christian), and
Ito-Matsui even rejects the Hassiotis secondary variant with a DOI. All
recovered/reconstructed-geometry and "not pinned / demonstration / NOT-tuned" disclosures
are concentrated in **`slope_stability/VALIDATION.md` §B7–B11** (e.g. B7 the #98 recovered
section; B8 the V-029 reconstructed flow-net explicitly noting inventing values is
"forbidden") + INVENTORY per-entry.

The honest soft spots, all documented, are: (a) the B&T **rigid branch** — the chapter
WAS in hand, but the rigid coefficient (a0=−0.22) is NOT reproduced against a published
rigid worked example (limiting-behavior only → flagged UNVERIFIED in the docstring);
(b) the **legacy `methods.py` Spencer/M-P** are self-declared approximations (SS-1) and
the **Janbu f0** is read off a figure (Abramson Fig 6.18 — elevated) — both documented,
the rigorous `gle.py` path is the cited-exact one; (c) the anisotropic su **formula
provenance** (scout web relay of Casagrande-Carrillo/Das, with the sin(2α) base-angle
form my own hand-derivation) validated only against a secondary isotropic value and a
FEM (not LEM) anisotropy magnitude; (d) the rapid-drawdown **EM/DWW/L&K originals not in
hand** (read via web-archive / vendor manuals), the G-12 convention call empirically
anchored; (e) the **raster-recovered geometries** (#98, #97, #29, ACADS 1(d), Loukidis
#63) that cap several anchors at geometry-limited CONVENTION; and (f) the **Anchor /
Geosynthetic** reinforcement elements carry no literature source by design (user-supplied
capacity, no pullout model — adequately noted in their docstrings). The WIKI-WISHLIST
above lists exactly what would upgrade each.
