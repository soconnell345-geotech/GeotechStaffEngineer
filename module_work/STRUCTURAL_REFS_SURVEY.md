# Public-domain structural reference survey (2026-09-03, web-verified)

Purpose: identify the structural-engineering equivalents of DM7/GEC/UFC for
the reference layer — free, US-government public-domain documents whose
equations/tables/figures can be digitized with page-accurate citations.

**Strategic caveat:** the consensus standards themselves (ASCE 7, ACI 318,
AISC 360, TMS 402, AASHTO LRFD, NDS) are paywalled or free-but-copyrighted
(AISC's free spec PDFs = verification-only, never digitization). The PD
anchors are the government documents that adopt-and-modify them.

## Corrections vs. common assumptions
- UFC 3-310-04 (seismic) SUPERSEDED 2019 — seismic now lives inside
  UFC 3-301-01. The legacy 3-310/3-320 "A" wrappers are all cancelled;
  the current UFC structural story = 3-301-01 + 4-023-03 (+ blast).
- EM 1110-2-2105 (steel) SUPERSEDED by **EM 1110-2-2107 (2022)** which
  absorbed the gate-specific EMs (miter/tainter/bulkhead appendices).
- EM 1110-2-2104 (RC hydraulic structures) has a **Jan 2025 edition** —
  digitize that, not 2016.

## Prioritized inventory

| Reference | Anchors | Edition | ~Pages | PD status |
|---|---|---|---|---|
| UFC 3-301-01 Structural Engineering | loads/wind/snow/seismic; DoD mods to IBC 2024 / ASCE 7-22 | 2023 + C4 (Jun 2025) | ~150-200 | High; born-digital (WBDG) |
| FEMA P-2192 Vol 1 — 2020 NEHRP Design Examples | worked seismic design examples, all materials (ASCE 7-22 basis) | 2021-22 | ~1000 | High; born-digital |
| EM 1110-2-2104 | RC LRFD design: flexure/shear/axial+bending/crack control + hydraulic factors | Jan 2025 | ~130 | High; born-digital |
| EM 1110-2-2107 | steel LRFD: members, connections, fatigue/fracture + gate appendices | 2022 | ~350 | High; born-digital |
| USDA Wood Handbook FPL-GTR-282 | wood properties + beam/column/fastener mechanics | 2021 | ~600 | High (explicit PD) |
| UFC 4-023-03 Progressive Collapse | tie forces, alternate path, acceptance criteria | 2009 + C4 (2024) | ~230 | High |
| GSA Alternate Path Guidelines | progressive collapse, federal civilian | 2016 | ~200 | High |
| NIST NEHRP Seismic Design Technical Briefs 1-13 | per-system seismic doctrine (RC/steel SMF, SCBF, BRBF, diaphragms, masonry, CFS...) | 2008-2017 | ~40-50 ea | High (NIST GCR) |
| FHWA Steel Bridge Design Handbook HIF-16-002 | steel bridge, 19 vols + 6 design examples | 2015 | ~2500 | High (FHWA-authored) |
| FHWA-NHI-15-047 + -15-058 | LRFD superstructure manual + worked examples | 2015 | ~1500 | High |
| FEMA P-751 / FEMA 451 | older seismic example sets | 2012/2006 | ~900/860 | High |
| UFC 3-340-02 Blast | accidental explosions; SDOF, charts, detailing | 2008 + Ch2 2014 | ~1867 | High |
| NASA-STD-5020B | threaded-fastener/bolted-joint analysis | 2021 (reval. 2025) | ~110 | High |
| NAVFAC DM-2 series | legacy 1980s structural | superseded | — | PD but obsolete + image scans — SKIP |

Confirmed paywalled (never digitize): AASHTO LRFD 10th ed, ASCE 7-22,
ACI 318-19/25, TMS 402, NDS. AISC 360-22/341/358: free PDFs, copyrighted —
verification anchors only.

## Start-here shortlist (recommended onboarding order)

1. **UFC 3-301-01 (2023 C4)** — the loads/seismic umbrella, keyed to
   current practice; direct parallel to how the geotech layer uses UFCs.
   Anchors a `loads` module (wind/snow/seismic per DoD-modified ASCE 7-22).
2. **EM 1110-2-2104 (2025) + EM 1110-2-2107 (2022)** — the only PD sources
   carrying actual current LRFD concrete and steel DESIGN equations; their
   hydraulic-structures flavor bridges naturally from our retaining-wall /
   dam-adjacent modules.
3. **FEMA P-2192 Vol 1** — a thousand pages of current worked seismic
   examples; slots straight into the worked_examples verification corpus
   as ground truth for future structural modules.
4. **USDA Wood Handbook (2021)** — the PD wood bible (properties +
   mechanics; NDS adjustment factors stay out of scope).
5. **UFC 4-023-03 + GSA 2016** — progressive collapse: closed-form
   tie-force equations + acceptance tables, highly implementable.

Runners-up: NIST Technical Briefs (doctrine layer once seismic-structural
modules exist); FHWA bridge set only if bridges enter scope.

Pairing with round-1 wrappers: sectionproperties/concreteproperties/PyNite
give the ANALYSIS engines; these references supply the DESIGN checks and
worked-example validation the same way GEC/DM7 anchor the geotech side.

## Verified download links (for the onboarding campaign)

- UFC 3-301-01: wbdg.org/ffc/dod/unified-facilities-criteria-ufc/ufc-3-301-01
  (2023 C4 PDF: nibs-s3-wbdg3-production.s3.us-east-1.amazonaws.com/FFC/DOD/UFC/ufc_3_301_01_2023_c4.pdf)
- UFC 4-023-03 (C4): wbdg.org/ffc/dod/unified-facilities-criteria-ufc/ufc-4-023-03
- EM 1110-2-2104 (2025): publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM%201110-2-2104_2025%2001%2008%20-%20Final.pdf
- EM 1110-2-2107 (2022): publications.usace.army.mil/Portals/76/Users/182/86/2486/EM%201110-2-2107.pdf
- FEMA P-2192 Vol 1: fema.gov/sites/default/files/documents/fema_nehrp_design-examples-training-materials_volume-1.pdf
- NIST briefs (nvlpubs pattern, e.g. #13): nvlpubs.nist.gov/nistpubs/gcr/2017/NIST.GCR.17-917-47.pdf
- FHWA-NHI-15-058: fhwa.dot.gov/bridge/pubs/nhi15058.pdf
- GSA 2016: gsa.gov/system/files/Progressive_Collapse_2016.pdf
- NASA-STD-5020: standards.nasa.gov/standard/NASA/NASA-STD-5020
- USDA Wood Handbook: research.fs.usda.gov/treesearch/62200
- Paywall confirmations: store.transportation.org (AASHTO LRFD 10th);
  aisc.org current standards = free-but-copyrighted (verification only).
