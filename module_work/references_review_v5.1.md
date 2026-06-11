# geotech-references v5.1 Review (submodule @ dafb490)

Date: 2026-06-10. Reviewed from the v5.1-todos worktree; tests run with the parent venv from the submodule root.

**Bottom line:** the library is healthy — all 199 JSON files load, all 3,702 tests pass (twice, ~6–7 s),
the FTS5 text index covers all 17 text references (1,883 sections) and the figure index covers all 20
catalogs with every page resolved, and every spot-checked lookup value (GEC-10, GEC-11, ufc_pavement)
matched its source. The real findings are: (a) the `implementation_gaps` convention is **empty in
practice** — qc_progress.json has no gaps and only registers dm7_1/dm7_2, so the weekday-TODO pipeline
described in CLAUDE.md isn't producing anything; (b) CLAUDE.md is significantly stale (the repo grew
from 14 to 20 reference packages since it was last rewritten); (c) the audit script's schema has
drifted from how GEC chapters are actually authored; (d) the GEC-13 figure catalog is ~90% incomplete;
(e) the synonym table has a demonstrated harmful entry (`cu`) and several easy, defensible additions.

---

## 1. Data integrity — PASS

- **All 199 JSON files parse** (package data, catalogs, manifests, qc_progress.json).
- **Schema consistency:** all 17 text references use the standard section schema. Two cohorts:
  older/manual refs include `summary` per section; gec_6/7/10/11/12/13 omit it (see §6, audit drift).
  gec_12 has 10 sections with extra `subsections`/`cross_references` keys (harmless, but nonstandard).
- **Empty bodies:** 42 (dm7_1) + 30 (dm7_2) sections have empty `body` — all are heading/container
  sections that carry `key_points` instead. Convention, not corruption. No truncated/short bodies anywhere.
- **Pointer integrity:** every `Figure X-Y` cited in dm7_1/dm7_2 section `figures` arrays resolves to a
  figures_catalog entry (0 dangling). Section-numbering vs chapter check: only two benign cases —
  dm7_2 `prologue.json` uses `1-x` ids (the prologue *is* chapter 1), and gec_13 chapter01 numbers its
  top-level sections `1.0…9.0` while ch 2–11 use `<ch>.<sec>.0` (style inconsistency, see §6).
- **Mixed `equations` entry types:** dicts in most refs, bare strings in gec_7/gec_10/gec_13/micropile,
  and *mixed within* gec_6 (60 dict / 14 str), gec_11 (42/48), gec_12 (129/15). Consumers that assume
  dicts will break; normalization is a v5.2 item.

## 2. implementation_gaps inventory — convention is EMPTY; derived inventory below

`references.<key>.implementation_gaps` exists nowhere in the repo except as prose in CLAUDE.md.
qc_progress.json registers **only dm7_1 and dm7_2** (both fully `done`, zero gaps recorded), despite
20 manifests now resolving to PDFs in docs/ (after the two fixes in §7, 20 of 22 resolve; only the
volumes-style gec_11/gec_13 manifests have no top-level `pdf_path`). Either the weekend QC routine
hasn't run since the new references landed, or its auto-registration doesn't handle the new manifests
— worth checking the routine's run history.

**Derived inventory** (cited-in-text figures/tables with no `figure_X_Y_*`/`table_X_Y_*` function,
filtered to lookup-able design content, ranked by likely agent demand — bearing/pile/wall/liquefaction
first). Top authoring targets for weekday sessions:

### Tier 1 — pile/shaft and wall design values (highest agent demand)
| Ref | Item | Why |
|---|---|---|
| gec_7 | Tables 5.2, 5.4–5.11 (8 small LRFD tables) | Complete soil-nail LRFD φ set: overall stability, basal heave, pullout, sliding, tendon, facing flexure/punching/stud. Captions already carry the values — near-mechanical to author. |
| gec_7 | Table 4.2b CPT tip resistance → friction angle | Common parameter lookup. |
| gec_8 | Table 5.1 su–Ir–N*c; Figs 5.2/5.4 (alpha/beta, Coleman & Arcement); Figs 5.3/5.5 LPC CPT side-shear; Table 5.3 group efficiency | CFA-pile axial design core — gec_8 currently has only 1 table function. |
| gec_9 | Tables 3-1/3-2 lateral-load geomaterial parameters; Table 11-1 effective length K; Table 8-1 seismic zones | Lateral deep-foundation design inputs (p-y workflows). |
| gec_12 | Table 7-32 Rndr resistance factors | Closes the driven-pile LRFD set. |
| gec_10 | Table 10-1 (geomaterial properties required per resistance) ; Tables 8-2/8-3 AASHTO load factors | Rounds out the drilled-shaft design tables. |
| micropile | Tables 4-3/4-4 (GEWI / hollow-core bar properties) | Frequent sizing lookups. |

### Tier 2 — bearing/settlement charts (GEC-6, GEC-5)
- gec_6 Fig 5-6 bearing capacity factors vs φ (AASHTO 1996); Fig 5-7 sloping-ground modified factors;
  Table 5-2 shape factors; Fig 5-20 Christian & Carrier μ0/μ1; Fig 5-21 D'Appolonia modulus vs SPT;
  Table 5-11 Cc correlations; Fig 8-3 modulus of subgrade reaction.
- gec_5 Table 9-10 Hoek-Brown mi (cited 2×); Fig 11-19 Ko vs φ'/OCR; Fig 6-36/6-37 PI/LL consolidation
  correlations; Fig 7-30 FVT correction (Chandler); Fig 5-20 **CPT liquefaction triggering charts
  (Youd et al. 2001)** — the only liquefaction-design gap found, and worth Tier-1 treatment if the
  parent's liquefaction API wants a reference-backed CRR chart.

### Tier 3 — ground modification (GEC-13) and RSS (GEC-11)
- gec_13 Fig 2-2 Ur–Th (Barron/Hansbo) and Fig 2-4 drain-spacing chart (PVD design); Figs 5-8/5-9
  Priebe n0 + corrections; Fig 5-3 stone-column applicability; Tables 4-2/4-4 DDC and vibro spacing
  parameters; Tables 3-2/3-3 lightweight-fill properties.
- gec_11 Fig 9-5 Schmertmann RSS reinforcement chart (the only chart-type RSS gap).

Not worth authoring: the long tails are photos, flowcharts, and schematics (e.g., gec_10's 78 "gaps"
are mostly construction photos) — the figure-catalog + vision read-off path already covers them.

## 3. Lookup-function coverage + health — PASS

- Full suite: **3,702 passed** (CLAUDE.md still says 3,529), ~6–7 s, re-run green after fixes.
- **GEC-10 spot checks (all correct):** Fig 10-6 α = 0.30+0.17/(su/pa) → 0.47 @ su/pa=1, 0.64 @ 0.5;
  Table 11-1 p-multipliers 0.70/0.50/0.35 @ 3D; AASHTO group η 0.65/1.0 @ 2.5D/4D; Table 10-2
  N*c = 9.0 @ su ≥ 2000 psf; Table 8-4 φ = 0.45 side-cohesive / 0.40 base-cohesive; Eq 10-21
  fsn = pa·√(qu/pa) → 318 kPa @ qu=1 MPa; Eq 10-22 correctly applies 0.65·αE (0.85 @ RQD 70, closed);
  β=3.0 → pF=0.00135.
- **GEC-11 spot checks (all correct):** Fig 4-10 Kr/Ka — metal strip 1.7→1.2 over 6 m (1.45 @ 3 m),
  bar mat 2.5, geosynthetic 1.0; Table 4-6 φ=34° → Nc 42.16 / Nq 29.44 / Nγ 41.06; Table 4-4 heq.
- **Drift, not bugs:** input-unit drift vs the "All units SI" convention — `table_10_2_nc_base_clay(su_psf)`,
  `table_4_4_traffic_surcharge(wall_height_ft)` take US units (documented in signatures, but the
  CLAUDE.md units claim is wrong as stated). Key-vocabulary friction: row positions are `'lead'/'2nd'/'3rd'`,
  reinforcement types `'metal_strip'/'bar_mat'`, method keys `'side_cohesive_compression'` — error
  messages list valid options (good), but adapter-side curated aliases remain the mitigation.
- **Cross-link gap:** GEC-10's implemented tables (8-4, 9-1, 10-2, 10-3, 11-1, 11-2) are **never cited**
  in any chapter-JSON `tables` array — text retrieval will never point a consumer at the digitized
  lookups. Backfill the pointers in ch 8/9/10/11 sections (cheap, high value).

## 4. Retrieval / query-expansion curation

Index coverage: text FTS5 = all 17 text refs; figure FTS5 = all 20 catalogs, 0 unresolved pages.
Both DBs are lazy temp builds keyed to source mtimes — no staleness mechanism to worry about.

**Demonstrated harmful entries (fix in next curation pass):**
1. `cu` in the undrained-strength group collides with coefficient of uniformity Cu: a query
   "coefficient of uniformity cu" expands to `"undrained shear strength" OR su OR cohesion` and its
   top hit becomes dm7_1 4-8 (stress distribution). Remove bare `cu` (keep `su`).
2. Bare `compaction` in the compaction group: "dynamic compaction" / "rapid impact compaction"
   (deep-compaction technologies) expand to `proctor OR "relative compaction"` (QC concepts).
   Remove the bare member; keep "relative compaction"/"proctor".
3. Substring artifact: multi-word terms match as substrings, so "negative skin friction" fires the
   *side-resistance* group via embedded "skin friction", polluting a downdrag query with
   `"side resistance" OR "shaft resistance"`. Consider longest-match-wins or excluding a term when it
   is part of a longer matched term.
4. Watch-list (not demonstrated): `phi` doubles as the LRFD resistance-factor symbol; `cohesion`
   inside the su group surfaces drained-c′ sections for undrained queries.

**Verified vocabulary gaps (no expansion fires; literal recall thin):** `N160` → **0 hits**;
`(N1)60`, `p-y curve`, `t-z curve` → 1 hit each; `MSE` alone does not expand (group only has the
multi-word "mse wall") and its top hit is dm7_1 7-9, not gec_11; `PVD`, `CSR`, `CPT`, `SPT`,
`dilatometer/DMT`, `pressuremeter/PMT`, `RQD`, `geofoam`, `vibroflotation` all have zero expansion.

**Recommended additions (defensible equivalences):**
`["prefabricated vertical drain","pvd","wick drain","vertical drain"]`,
`["cyclic stress ratio","csr"]` (do NOT merge with the CRR group — demand ≠ capacity),
`["(n1)60","n160","corrected spt blow count","normalized blow count"]`,
`["p-y curve","py curve","lateral load transfer"]`, `["t-z curve","tz curve","axial load transfer"]`,
`["cone penetration test","cpt"]`, `["standard penetration test","spt"]`,
`["dilatometer","dmt"]`, `["pressuremeter","pmt"]`,
`["unconfined compressive strength","ucs","qu"]`, `["rock quality designation","rqd"]`,
`["geofoam","eps","lightweight fill"]`, `["stone column","aggregate column","aggregate pier","vibro-replacement"]`,
`["vibro-compaction","vibroflotation","vibratory probe"]`, `["deep mixing","deep soil mixing","dsm","soil-cement column"]`,
`["continuous flight auger","cfa pile","auger cast pile","augered cast-in-place"]`,
add bare `"mse"` to the MSE group. Also grow the eval's trusted set (still 9 cases) before re-tuning.

## 5. ufc_pavement — REAL and CONSISTENT; CLAUDE.md note is stale

The parent adapter (`funhouse_agent/adapters/ufc_pavement_adapter.py` via `_reference_common.build_lookup_registry`)
exposes exactly **21 entries = 11 canonical + 10 aliases**, matching the parent's listing. All 11
canonical methods were smoke-called against the submodule copy and returned correct, source-consistent
values (CBR→k 130.3 psi/in @ CBR 10; ML → frost group F4; F3 → SSI 3.5; graded crushed aggregate →
design CBR 100; Table 7-2 minimums; Table 9-1 equivalency 1.15 for cement-stabilized SP base; etc.).
`equations.py`/`tables.py` docstrings explicitly scope to **UFC 3-250-01 (14 Nov 2016), roads/streets/
walks/storage — NOT airfields**, and `agents/ufc_pavement_agent.py` says the same. The CLAUDE.md
warnings ("coded from UFC 3-260-02 — needs audit/replacement", "agent references UFC 3-260-02") are
**obsolete** — the audit/replacement evidently happened. Remove both notes.

## 6. Stale / duplicated / inconsistent (v5.2 fodder)

1. **CLAUDE.md is the stalest artifact in the repo.** It omits 8 shipped packages (gec_4, gec_5,
   gec_8, gec_9, gec_14, california_trenching, fhwa_pavements, fema_p2082), still lists three deleted
   ones (fema_p2192, noaa_frost, ufc_dewatering), says "14 agents" (now 21 incl. references_agent),
   "3,529 tests" (3,702), figure catalogs "built for DM7" (now all 20 refs), the ufc_pavement audit
   note (§5), and the query-expansion section still says "uncommitted, on branch ref-retrieval-expansion"
   though it merged at 40f8a06. One deliberate rewrite pass needed.
2. **audit_chapter_text.py schema drift:** it hard-requires `summary` per section, but gec_6/7/10/11/12/13
   were authored without it → gec_10 "fails" with 273 errors (120 = missing summary), gec_11 with 392,
   gec_13 with 188. The weekend QC routine uses this script for validation, so it can never go green on
   these refs. Make `summary` optional (or backfill) and re-baseline.
3. **gec_13 figures_catalog.json is ~90% incomplete:** 33 figures, chapters 1 and 6 only, Vol 1 PDF only
   — ch 2–5 and all of Vol 2 (ch 7–11: deep mixing, grouting, nails, micropiles, geosynthetics) missing.
   Contradicts the 829b897 release note "all figure catalogs 100%". Rebuild over both volumes.
4. **Figure-catalog descriptions thin outside DM7:** ufc_pavement 76/76, ufc_expansive 32/32,
   california_trenching 159/177, gec_12 468/546 entries have no `description` — concept-level
   `figure_search` (the Fig 4-12 log-spiral win) doesn't work for these refs yet.
5. **Manifest duplication/inconsistency:** gec_12.json (volumes-style) duplicates gec_12_v1/v2.json
   (chapters-style); gec_11.json/gec_13.json are volumes-style with no top-level `pdf_path` —
   verify the QC routine's eligibility scan handles the `volumes` shape, then consolidate to one style.
6. **qc_progress.json moribund** (§2): 18 eligible references unregistered; no implementation_gaps ever
   recorded. Check routine trig_015ika6HHrYfcrf7uupLGGTC's recent runs.
7. **_retrieval_db.py docstring** still says "gec_11 and the FEMA/NOAA/UFC references have no narrative
   text" — gec_11 has 183 indexed sections and the module itself indexes them.
8. Minor: mixed `equations` types (§1); gec_13 ch-1 section-id style (§1); units drift vs "All units SI"
   (§3); ufc_backfill labeled UFC 3-220-04N in CLAUDE.md but the PDF is ufc_3_220_04fa_2004.pdf;
   legacy `references/` dir still present; scripts/README.md describes the retired pipeline.

## 7. Mechanical fixes applied in this review (complete list)

1. `scripts/manifests/gec_12_v1.json` — `pdf_path` `../../references/nhi16009_v1.pdf` (git-ignored,
   missing) → `../../docs/GEC 12 vol 1.pdf` (exists). Unblocks weekend-QC eligibility for GEC-12 Vol I.
2. `scripts/manifests/gec_12_v2.json` — same fix → `../../docs/GEC 12 Vol 2.pdf`.

Both verified to resolve; full suite re-run green (3,702 passed). No other files modified.

## Prioritized v5.2 recommendations

1. **Revive the gap pipeline:** check the weekend QC routine's runs; get the 18 unregistered references
   into qc_progress.json; fix audit_chapter_text.py's `summary` requirement so validation can pass.
2. **Author Tier-1 lookups** (§2): the gec_7 LRFD table set (8 trivial tables), gec_8 CFA axial-design
   charts/tables, gec_9 lateral-load parameter tables, gec_12 Table 7-32, gec_10 Table 10-1 — plus
   gec_5 Fig 5-20 CPT liquefaction charts for the liquefaction API.
3. **Query-expansion curation pass:** remove `cu` + bare `compaction`, fix the substring artifact,
   add the ~16 verified-gap groups (§4); grow the eval gold set beyond 9 before re-measuring.
4. **Rebuild gec_13 figures_catalog** over both volumes; backfill catalog `description` fields for the
   worst refs (ufc_pavement, ufc_expansive, california_trenching, gec_12).
5. **Rewrite CLAUDE.md** (one pass covering every §6 item) and fix the _retrieval_db docstring.
6. **Cross-link GEC-10's implemented tables** into its chapter JSON `tables` arrays (§3) — and adopt
   "cite the implemented artifact in text pointers" as an authoring rule.
7. **Normalize `equations` entries** to one shape (dict) across refs; consolidate manifest styles.
