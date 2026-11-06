# DR-11 label vocabulary on the saved corpus: summary
Measured 2026-11-06 on the saved private corpus records by a local run (report_ingest/recompute.py at bc46ba9; no model calls). Generic labels and counts only; the per-report detail stays local and private.

## What was read
214 report.record.json files (76 distinct), 382 calc-stage run files (104 distinct calcs), 10 hand-truth calcs -> 1,335 calculation readings, 965 unique (report, first page), 11 reports (14 with bound documents), 45,022 labels. 0.4 % of labels unclassified by the generic classifier.

## Counts per kind (readings / unique / reports / recomputed / admin-artefact share)
lateral_pile 384/289/4/no recipe/11 % | settlement 255/197/8/0/19 % | other 165/156/8/n/a/8 % | axial_pile 157/133/5/no recipe/7 % | retaining_wall 93/56/7/no recipe/12 % | slope_stability 88/46/5/held/8 % | site_response 67/23/9/no recipe/30 % | pavement 60/37/8/no recipe/7 % | shallow_foundation_bearing 54/21/7/0/2 % | liquefaction 9/5/2/no recipe/11 % | ground_improvement 3/2/1/no recipe/3 %

## Why nothing recomputed
Bearing (54): 43 not recomputed, 11 not compared (no claim; 9 empty). Several cases in one Calculation (scenario x stratum x load case, 8-20 claims) 41 readings (8 unique); only gamma' printed 38 (7); shape several/none 23/18; ratio taken as width 28 (8).
Settlement (255): Schmertmann 193 (76 %); no method and fits neither 25; combined 6; inputs missing 16 (Es per layer 14, no applied pressure 11). Elastic printed in 31, consolidation in 3.
What-if (measurement only): case split alone -> 0 of 217 cases (193 stop at gamma'). Split + gamma' run as the submerged case + shape from the case's own labels -> 5 of 10 unique bearing calcs with claims (4 reports); 196 claim rows: 181 agree within 10 %, 14 within options, 1 differs, median |diff| 3.7 %.

## Labels taken wrongly (labels / readings (unique) -> fix)
- ratio B/L, L/B, B/Df, B/elevation -> width/length, B/L: 95 / 42 (22) -> `^b\b(?!\s*/)`, `^l\b(?!\s*/)`
- correlation ratio q/Nc, qc/N -> q_net: 23 / 2 (2) -> exclude `^q_?c?\s*/`
- water (pore) pressure -> q_net: 6 / 3 (3) -> exclude `water`
- drain thickness -> thickness: 2 / 2 (2) -> exclude `drain`
- stress RATIO -> sigma_v0: 4 / 4 (2) -> exclude `\bratio\b`
- yes/no setting or sentence fragment -> Es, B: 8 / 8 (6) -> skip text-valued and sentence-like labels (>8 words, "yes or no", leading "x) ")
- unconfined strength (qu = 2 su) -> su: 1 / 1 (1) -> exclude `unconfined`

## Labels no pattern matches (labels (unique calcs) -> proposal)
- settlement as a symbol rho / Se / delta / delta-h (rho missing from _GREEK): 178 labels in 63 readings (50 unique), all Schmertmann today -> `_GREEK["ρ"]="rho"`; claim `settlement|^rho\b|^s_?e\b|^delta( ?h)?\b`, length unit required
- height of soil layer (H) -> thickness: 80 (46) -> `height of (the )?(soil )?layer|layer height`
- qall / qa -> allowable claim: 24 (1 bearing calc) -> `allowable|\bq_?all?\b|\bq_?a\b`
- depth to water / water depth -> gwt_depth: 18 (1) -> `depth to (ground ?)?water|water depth` (a depth relative to the footing needs Df added, stated)
- Unit Wt. / soil density -> unit_weight: 263 (mostly slope, lateral pile; 1 bearing) -> `unit w(eigh)?t\b|\bdensity\b`
- depth factors "d q", "dg" -> depth-factor detection: 21 (1) -> `\bd ?_?(c|q|gamma|g)\b` (harmless today)
- prefixed Es (layer name first) -> Es: 95 (18), per-layer -> `\be_?s\b(?!ur)`, low value until layer tables
- gamma' alone (106 labels, 7 bearing calcs): a RULE for the owner, not an alias
Do not alias: sigma'zD / sigma'zp (not mid-layer sigma'v0); Is, If (not Iw); Esur (not Es); an allowable "from settlement" among inputs (an input, not a claim).

## Reader-side
"..., last row of the table" on 2,112 labels in 246 readings (bearing 31, settlement 83, axial 40, slope 52, site response 13, pavement 16): table rows lost. Case qualifiers live inside labels in several word orders. 12 % of labels are admin or artefact (30 % in site response).

## Next-slice coverage (proxy: readings / unique (reports))
- retaining wall static Ka/Kp/K0 claimed, phi printed: 60 / 38 (7); about 30 unique with one phi -> enough (soe / sheet_pile Rankine and Coulomb, retaining_walls Rankine on a slope)
- retaining wall seismic M-O, phi + kh/PGA printed: 46 / 32 (7); M-O named in 64/93 readings -> enough (seismic_geotech mononobe_okabe_KAE/KPE)
- rigid pavement D (AASHTO 1993): 31 / 14 (7) -> second slice
- flexible SN: 16 / 5 (2) -> thin
- axial pile (diameter, length, capacity): 74 / 61 (4) -> needs whole table rows
- lateral pile: 12 / 9 (3) of 289 -> not yet
- liquefaction: 0 of 5 -> no
- site class (N-bar/Vs + class or spectral value): 27 / 11 (6) -> small

## Recommendation
1. In bearing and settlement first: case split by printed qualifier, the false-positive fixes, rho/Se and qall, and the owner's gamma' rule (reproduces 181/196 claim rows).
2. New kind: retaining-wall earth-pressure coefficients (static plus Mononobe-Okabe): best coverage, existing module functions, no layer table.
3. Rigid pavement D second; piles, liquefaction and Schmertmann wait on the reader keeping whole table rows.
