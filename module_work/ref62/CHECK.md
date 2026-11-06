# REF-62: three more GEC-13 table functions that cited tables printing something else

Branches: refs `claude/ref-62-gec13-tables` (head d9c841c, from refs main
c0ed2b9), app `claude/ref-62-gec13-tables` (from master df301b3). The app
branch moves the `geotech-references` submodule pointer to d9c841c, so CI
tests against the refs this change needs. This item makes no refs version
bump and no change to the app's `geotech-references>=` pin; the release
needs both (see "Notes for the lead").

## What the print says

GEC-13 is FHWA-NHI-16-027 (2017): Vol I for chapters 1-5, Vol II for
chapters 6-10. The PDFs in the refs repo's `docs/` were read with PyMuPDF.
Tables 4-3, 8-1 and 8-6 were rendered to PNG and read off the page; the
extracted text matches.

| Old function | What GEC-13 prints under that number | What GEC-13 does print on the subject |
|---|---|---|
| `table_4_1_ddc_parameters` | Table 4-1 "Equipment Requirements for Different Sized Tampers" (Vol I p. 4-34): crane and cable size by tamper weight | Table 4-3 "Applied Energy Guidelines" (Vol I p. 4-46): unit applied energy per unit volume by deposit type. Also Eq. 4-1 (depth, p. 4-40) and Eq. 4-2 (applied energy, p. 4-45). |
| `table_8_2_jet_grouting_systems` | Table 8-2 "Components of Permeation Grouts Used by Brachman et al. 2004" (Vol II p. 8-38) | Table 8-1 "Advantages and Disadvantages of Jet Grouting Systems" (Vol II p. 8-26); Table 8-6 "Range of Typical Soilcrete Strengths - Single-Fluid System" (p. 8-71, Burke 2004); text: columns 2.5 to 15 ft in diameter depending on the method (p. 8-71) |
| `table_7_2_deep_mixing_strength` | Nothing: chapter 7 has no List of Tables (only Figures 7-1 to 7-8, Vol II p. 7-iii) | Text only: "The specified unconfined compressive strength, which on recent projects has varied from about 100 psi to 500 psi, although strengths at the lower end of this range have been most common" (p. 7-23); mass mixing cores hard to recover below 100 psi (p. 7-38); a 26 psi dry-method design example (p. 7-32) |

**Table 4-3, p. 4-46** (as printed):

| Type of deposit | ft-lbf/ft3 (kJ/m3) | % standard Proctor |
|---|---|---|
| Pervious coarse-grained soil (Zone 1 of Figure 4-5) | 4,100 to 5,200 (200 to 250) | 33 to 41 |
| Semipervious fine-grained soils (Zone 2 and clay fills above the water table and Zone 3 of Figure 4-5) | 5,200 to 7,200 (250 to 350) | 41 to 60 |
| Landfills | 12,400 to 22,700 (600 to 1100) | 100 to 180 |

Note: standard Proctor energy equals 12,400 ft-lbf/ft3.

**Table 8-6, p. 8-71**: clean sands and gravel 750 to 1,250 psi; silts and
silty sands 500 to 750; clays 250 to 500; organic silts and peats less than
250.

**Table 8-1, p. 8-26**: five systems (Single Fluid, Double Fluid, Triple
Fluid, SuperJet/MegaJet/UltraJet/STRAJet, X-Jet), each with a bulleted list
of advantages and of disadvantages. No numbers.

## The values against the print

Other held manuals searched (PDF text): FHWA NHI-06-088/089, GEC-4, GEC-5,
GEC-7, UFS 3-220-10, UFC 3-220-20, the Caltrans T&S manual, UFC 3-250-11.
The refs text database was searched too. UFC 3-220-20 Figures 1-8 (dynamic
compaction, p. 45) and 1-9 (deep mixing, p. 46) were rendered: they are
qualitative summaries that point to GEC-13 (FHWA 2017) and print no values.
None of the three functions' values is printed anywhere held:

- **DDC.** The old function gave energy per pass in ft-lb/ft2 (5,000-8,000,
  5,000-10,000, 8,000-12,000), 5-15 passes, 1-6 weeks between passes, and
  maximum depths of 36, 30 and 20 ft. GEC-13 prints none of these. It gives
  energy per unit volume (Table 4-3), total applied energies of about
  21,000 to 62,000 ft-lbf/ft2 on most projects (p. 4-45), and "generally a
  waiting period of at least a few days between passes" (p. 4-11).
- **Jet grouting.** The old function gave diameters per system (single
  300-600 mm, double 600-1,000, triple 600-2,000) and strengths
  (1-5, 1-7, 1-10 MPa). GEC-13 prints no diameter or strength per system.
  Its only per-soil strengths are Table 8-6, single fluid, 250 to 1,250 psi
  (1.7 to 8.6 MPa).
- **Deep mixing.** The old function gave seven soil/method rows,
  100-4,000 kPa. GEC-13 prints one range, 100 to 500 psi (690 to
  3,450 kPa), for specified strengths on recent projects, with no soil
  or method split.

## What changed and why

Refs `geotech_references/gec_13/tables.py`:

1. **`table_4_1_ddc_parameters` re-cited as `table_4_3_applied_energy`**
   (option a). GEC-13 does print a soil-keyed DDC design table, Table 4-3,
   so the subject has a printed answer; the function now returns it as
   printed. The rows keep the old keys `pervious_coarse` and
   `semi_pervious` (the third is now `landfills`, not `impervious`). Each
   row gives the unit applied energy in ft-lbf/ft3 and kJ/m3, the percent
   of standard Proctor energy, the reference and a note (multiply by the
   deposit thickness; loose deposits take the high end; saturated Zone 3
   clays not recommended, p. 4-21). The name changed because the old one
   cited Table 4-1.
2. **`table_8_2_jet_grouting_systems` re-cited as
   `table_8_1_jet_grouting_systems`** (option a). It returns Table 8-1's
   five systems with their printed advantages and disadvantages. Its note
   gives the printed column range (2.5 to 15 ft, p. 8-71) and points to
   Table 8-6 for strength. The diameter and strength fields are gone,
   because no held manual prints them per system.
3. **`table_8_6_soilcrete_strength` added.** It carries the strength
   subject of the old jet grouting function, from the one table that
   prints it: Table 8-6, single fluid, in psi as printed, with kPa
   converted (1 psi = 6.894757 kPa). The "less than 250" row has
   `ucs_psi_low` None. The note gives p. 8-71's caveats: strength follows
   cement content whatever the system, design values at most 50 % of the
   ultimate, and a production field test is essential.
4. **`table_7_2_deep_mixing_strength` removed** (option b). There is no
   table to re-cite it to, and turning one sentence into a lookup keyed by
   soil and method would invent the split. The subject is answered by the
   text: gec13 `search_sections` ("unconfined compressive strength" or
   "deep mixing") finds section 7 p. 7-23 in `text/chapter07.json`, which
   carries the 100 to 500 psi sentence.

None of the three had catalog, manifest or search-database entries (the
search databases index the chapter text and figure catalogs, not the
functions). The refs `agents/gec13_agent.py` docstring now lists Table 4-3,
Table 8-1 and Table 8-6 instead of "Deep dynamic compaction parameters
(Table 4-1)". The GEC-13 function count stays 15 (3 out, 3 in), so the
README row (15) and the package total are unchanged.

**A print slip found on the way.** The worked example on p. 4-46 takes
5,200 for a loose 26 ft Zone 1 deposit and prints "41,000 ft-lbf/ft2".
5,200 x 26 is 135,200 ft-lbf/ft2; 41,000 is that divided by 3.28 (a
feet-to-metres slip). The table's own unit columns agree with each other
(250 kJ/m3 = 5,221 ft-lbf/ft3; 12,400 ft-lbf/ft3 = 594 kJ/m3, the
"100 %" landfill row), so the table is used as printed. The docstring says
so, and the test checks the unit columns rather than the example.

**App** (`funhouse_agent/adapters/_reference_common.py`, `_CURATED_ALIASES`):

| Name a model might call (gec13) | Before | After |
|---|---|---|
| `ddc_parameters` | auto alias of `table_4_1_ddc_parameters` | curated alias of `table_4_3_applied_energy` |
| `ddc_applied_energy` | did not exist | curated alias of `table_4_3_applied_energy` |
| `applied_energy` | did not exist | auto alias of `table_4_3_applied_energy` |
| `jet_grouting_systems` | auto alias of `table_8_2_...` | auto alias of `table_8_1_jet_grouting_systems` |
| `soilcrete_strength`, `jet_grout_strength` | did not exist | auto / curated alias of `table_8_6_soilcrete_strength` |
| `deep_mixing_strength`, `table_7_2_deep_mixing_strength` | the old lookup | unknown method; the did-you-mean scores gec13's own methods |

No other app file names these functions: `funhouse_agent/`, `webapp/`,
`calc_package/`, `report_ingest/`, `docs/`, `validation_examples/`,
`method_probes.json` and `worked_examples.json` were grepped for each name
and its stripped alias. The reference methods are on neither page's tool
list, so no tool description, prompt or page fingerprint changed.

## Before and after

| Call (gec13) | Before | After |
|---|---|---|
| `ddc_parameters`, soil_type "pervious" | 3 rows ("pervious" is inside "semi_pervious" and "impervious"); Zone 1: 5,000-8,000 ft-lb/ft2 per pass, 5-10 passes, 1 week, max depth 36 ft, cited "Table 4-1" | 2 rows (pervious_coarse, semi_pervious); Zone 1: 4,100-5,200 ft-lbf/ft3 (200-250 kJ/m3), 33-41 % Proctor, Table 4-3, p. 4-46 |
| `ddc_parameters`, "impervious" | Zone 3: 8,000-12,000 per pass, 10-15 passes, max 20 ft | no row matches (Table 4-3 groups Zone 3 clay fills above the water table with Zone 2; "semi" or "clay" finds it) |
| `jet_grouting_systems`, "triple" | 600-2,000 mm, 1-10 MPa | Table 8-1's triple fluid advantages and disadvantages; no diameter or strength |
| `soilcrete_strength`, "clays" | did not exist | 250-500 psi (1,724-3,447 kPa), single fluid, Table 8-6, p. 8-71 |
| `deep_mixing_strength`, soft clay WDM | 500-2,000 kPa, cited "Table 7-2" | unknown method; p. 7-23 text through `search_sections`: 100-500 psi specified on recent projects |

## Tests

Refs (`/tmp/v`, Python 3.12, the refs branch installed editable,
`geotech_references.__file__` -> `/home/user/geotech-references/...`),
judged by exit code:
- `tests/test_gec13_tables.py`, `test_gec13_figures.py`, `test_gec13_text.py`,
  `test_references_agent.py`, `test_cover_citations.py`,
  `test_figure_captions.py`: exit 0, 314 passed, 2 xfailed.
- New: `TestTable43` (every printed cell of Table 4-3; the two unit columns
  agree; the zone filter), `TestTable81` (the five systems in order; single
  and triple fluid lists word for word; no number fields), `TestTable86`
  (every printed row; the kPa conversion), `TestRef62Removed` (the three
  old names are gone).
- `tests/test_gec13*.py` collects 201 (was 197); the whole refs suite
  collects 6,545 (was 6,541). The refs CLAUDE.md counts and its GEC-13
  table paragraph are updated.

App (`/tmp/v`, `pip install -e packages/aec-agent-core -e .`, the refs branch
installed editable over the PyPI copy), judged by exit code:
- `funhouse_agent/tests/test_reference_adapters.py`,
  `test_method_name_aliases.py`, `test_method_aliases.py`,
  `test_gm179_unused_inputs.py`, `test_reference_result_sizes.py`,
  `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  exit 0, 472 passed. No fingerprint moved, so nothing was re-recorded.
- New: `TestRef62Gec13Recited` in `test_reference_adapters.py` (the old
  names and `deep_mixing_strength` are gone; `ddc_parameters` returns the
  same as `table_4_3_applied_energy` with the landfill row's printed
  12,400 and 22,700; `jet_grouting_systems` reaches Table 8-1;
  `jet_grout_strength` gives clays 250-500 psi).

## Release note

GEC-13's deep dynamic compaction, jet grouting and deep mixing lookups now
give only what GEC-13 prints, so their answers move. The DDC lookup
(`table_4_3_applied_energy`, also called as `ddc_parameters`) returns
GEC-13 Table 4-3, p. 4-46: unit applied energy per unit volume by deposit
type (Zone 1 4,100 to 5,200 ft-lbf/ft3, Zone 2 and dry clay fills 5,200 to
7,200, landfills 12,400 to 22,700), instead of energies per pass, numbers
of passes, waiting times and maximum depths that no manual prints; multiply
by the deposit thickness for the energy at the surface. The jet grouting
lookup (`table_8_1_jet_grouting_systems`, also `jet_grouting_systems`)
returns Table 8-1's advantages and disadvantages for five systems, with no
diameter or strength per system (GEC-13 gives 2.5 to 15 ft columns overall,
p. 8-71). A new lookup, `table_8_6_soilcrete_strength`, gives Table 8-6's
single-fluid soilcrete strengths by soil (clays 250 to 500 psi, up to
1,250 psi in clean sands and gravel). The deep mixing strength lookup
(`table_7_2_deep_mixing_strength`, `deep_mixing_strength`) is removed:
GEC-13 prints only that specified strengths on recent projects ran about
100 to 500 psi (p. 7-23), which `search_sections` finds. The old
`table_4_1_ddc_parameters` and `table_8_2_jet_grouting_systems` names are
gone.

## Noted, not fixed

- `figure_4_1_ddc_depth` (in `tables.py`) cites Figure 4-1 and "Lukas
  (1995)" and uses n by soil type (0.5-0.6 pervious, 0.35-0.5
  semi-pervious, 0.35-0.40 impervious, 0.35-0.5 waste). GEC-13 prints the
  relation as Eq. 4-1, p. 4-40, with n "approximately 0.5 for most soils",
  0.3 to 0.8 in Figure 4-21, and 0.4 typical for landfills. It has no
  soil-keyed n table. This would change a reference answer, so the lead may
  want an item for it; it was outside this item.
- GEC-13 Table 4-1 (tamper equipment) and the other printed tables in
  chapters 4 and 8 have no lookup functions. Nothing calls for them.
- The p. 4-46 worked example's 41,000 ft-lbf/ft2 (above) is a print slip.
  Nothing in the code uses it.

## Notes for the lead

1. Merge the refs branch first, then this app branch, and move the app's
   `geotech-references` submodule pointer to the refs merge commit in the
   app merge (this branch already points at refs d9c841c). Otherwise CI
   fails: `TestRef62Gec13Recited` finds the old names.
2. The release needs a refs version bump and a raised
   `geotech-references>=` pin. REF-61 already made that necessary; REF-62
   removes or renames three more public functions. Until then a PyPI
   install keeps the old values.
