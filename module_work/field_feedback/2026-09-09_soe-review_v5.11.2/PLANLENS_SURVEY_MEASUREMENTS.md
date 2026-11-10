# planlens survey train — measurements on the real submittal (PRIVATE ledger, 2026-09-16)

Companion to `module_work/PLANLENS_PACKAGE_SURVEY.md`. Every number below was
taken on the 260-page submittal in `raw/` (gitignored) or the ten public
Mecklenburg sheets, by the probe scripts beside this file
(`probe_measure.py`, `probe_fuzzy.py`, `probe_quantities.py`,
`probe_dup_hash.py`; `probe_layers_fill.py` lives in
`module_work/drawing_ground_truth/`). planlens branch `feature/survey-step1`.
Nothing here may be copied into the public planlens repo.

## Stored scale (`/VP` viewports, measurement markups)
- Submittal: one populated `/VP`, page 27, `/Subtype /GEO` (georeferenced,
  lat/long near the project), seven empty `/VP []` on the drawing set (pp 7-13).
  Zero of 360 annotations carry `/Measure` or a `*Dimension` intent — the
  reviewer's Bluebeam markups are comments, not measurements.
- Corpus: one viewport (`10.31A`), rectilinear but the untouched 1:1 default
  (`/R ( )`, `/C .01389` = 1/72 → identity), blank unit → `is_calibrated` False.
- 162 other PDFs in the dev tree: no viewports, no measurement markups.
- Conclusion: **no real Bluebeam-calibrated rectilinear page exists in our
  corpus yet.** The parser is spec-built; `/C` meaning (real units per PDF
  point) anchored on the identity viewport; `/R` vs `/X` cross-check warns at
  2 %. Re-run `probe_measure.py` on the first calibrated sheet that arrives.

## PDF layers and fill (drawing IR)
- Submittal drawing sheets pp 7-13: 15-28 distinct layer names per sheet on
  77-100 % of paths (real CAD layer names, some nested with a pipe), while
  `doc.get_ocgs()` returns zero groups. Page 12: 4,457 filled-and-stroked
  paths, 2,876 circle-like (the pile-symbol / boring-dot case). Path counts
  1 / 5,192 / 5,270 / 26,258 / 12,618 / 7,750 / 24,292; `get_drawings()` peaks
  at 0.20 s.
- Corpus (public, in DESIGN.md): only `10.31A` declares OCGs (`0`, `BORDER`,
  `TEXT`, `REV`, `PROPOSED`); 7.8 % of 41,061 paths layered; `closePath` False
  on all 6,669 filled paths, so `filled` is the reliable area signal.
- PyMuPDF returns "" not None for no group (normalised); content on a group
  OFF by default is omitted by MuPDF → `include_hidden_layers=True`.
- Page map: the two path passes became one; median 8.7 s → 5.4 s / 260 pages.

## Fuzzy search threshold (`rapidfuzz`)
15 real callouts, 45 corrupted queries (substitution / drop / transposition),
10 absent words, 7,261 candidates:

| threshold | source found | unrelated hits median / max | absent words hitting |
|---|---|---|---|
| 70 | 45/45 | 10 / 68 | 0/10 |
| 75 | 45/45 | 7 / 44 | 0/10 |
| 80 (default) | 45/45 | 5 / 40 | 0/10 |
| 85 | 45/45 | 3 / 28 | 0/10 |
| 90 | 43/45 | 1 / 28 | 0/10 |
| 95 | 21/45 | 0 / 15 | 0/10 |

Single-word queries are the binding case (45/45 at 75, 41/45 at 80); the
answer ranked first for 44/45. Candidates shorter than the query are scored
whole (otherwise one-character lines scored 100 against everything).

## Quantities from prose — bake-off
| corpus | extractor | TP | FN | FP | precision | recall | crashes |
|---|---|---|---|---|---|---|---|
| 40 invented sentences | regex | 37 | 1 | 1 | 0.97 | 0.97 | 0 |
| 40 invented sentences | quantulum3 | 23 | 15 | 14 | 0.62 | 0.61 | 9 |
| 40 real sentences | regex | 16 | 15 | 0 | 1.00 | 0.52 | 0 |
| 40 real sentences | quantulum3 | 11 | 20 | 114 | 0.09 | 0.35 | 2 |

quantulum3 NOT adopted (its ten extra mentions were all wrong: volts from the
V in `2H:1V`, bytes, atomic mass units, invented range midpoints; ImportError
mid-parse without its classifier extra). Regex's real-corpus misses are all
one printout form, `Pc (kip): 437.0` — a follow-up if calc printouts matter.

## Duplicate scans (dHash, 64 bits)
| measurement | distance |
|---|---|
| a page vs itself, re-rendered / re-opened | 0 |
| ten corpus sheets vs each other (n=45) | min 7, median 27, max 38 |
| closest different pair the rule compares (submittal) | 4 |
| drawing sheets of one set (submittal) | 11, 15, 15, 17 |
| consecutive boring logs (text-gated, never compared) | 1, 3, 6 |
| a rescan of one page (dpi / JPEG / offset / skew) | 2-8 high-contrast, 17-26 otherwise |

`DUP_HASH_DISTANCE = 2`; gated to needs_ocr / scanned / figure / drawing_sheet
/ <50 chars, never blank (13 pages qualify: 0.45-0.64 s vs 2.3-3.3 s ungated).
The rule claims "placed twice", not "scanned twice"; a text hash that differs
overrides the picture (the two D-size fixture sheets are within threshold).
Neither corpus holds a real duplicate — positive case is synthetic only.

## Nexus firewall probe (owner, on-cluster, 2026-09-16)
`nexus_pip_install` loaded 858 blocked versions. `rapidfuzz`: 172 versions
available, installed first try → pin `rapidfuzz==3.14.6`. `mcp`: 64 versions
available, installed first try → pin `mcp==2.2.0 idna==3.19 pydantic==2.12.5
pyjwt==2.10.1 starlette==1.6.0` (the helper pins transitive deps that have
blocked versions). Both import. Coupling check on pydantic/starlette vs the
streamlit stack still to do before the app adopts `[mcp]`.

## Shipped
planlens 0.4.0 (PyPI 2026-09-16 18:14 UTC) and geotech-staff-engineer 5.18.0
(PyPI 2026-09-16 19:45 UTC). Cluster install not yet confirmed at time of writing.
