# REF-61: three GEC-13 table functions that cited tables printing something else

Branches: refs `claude/ref-61-gec13-tables` (d3760dc), app
`claude/ref-61-gec13-tables`. No refs version bump, no change to the app's
`geotech-references>=` pin.

## What the print says

GEC-13 is FHWA-NHI-16-027 (2017), Vol II for chapters 6-10. The PDF text
was read with PyMuPDF.

| Function (refs `gec_13/tables.py`) | What GEC-13 prints under that number | Held manual that prints this subject |
|---|---|---|
| `table_9_2_nail_bond_strength` | Table 9-2 "Subgrade Stabilization Methods for Pavement Support" (Vol II p. 9-3) | GEC-7 (FHWA-NHI-14-007) Table 4.4a coarse-grained and 4.4b fine-grained (p. 84), Table 4.5 rock (p. 85). Bond is listed by construction method and soil, in psi. |
| `table_10_1_micropile_bond_stress` | Table 10-1 "Geosynthetic Properties Required for Reinforced Embankments" (Vol II p. 10-12) | FHWA-NHI-05-039 Micropile manual, Table 5-3 (p. 5-21), kPa (psi), Types A-D |
| `table_11_1_geosynthetic_reduction_factors` | No Table 11-1: GEC-13 has 10 chapters, and this was left over from REF-25's old chapters 9-11 | GEC-11 (FHWA-NHI-10-024) Vol I: RF_ID in Table 3-9 (p. 3-32); typical RF_CR by polymer in an unnumbered list on p. 3-33 (PET 2.5 to 1.6, PP 5 to 4.0, HDPE 5 to 2.6); RF_D for PET in Table 3-11 (p. 3-36) |

GEC-13 Vol I and Vol II were also searched for "bond". GEC-13 prints no
bond-strength table.

## The values against the print

None of the three tables matches its subject's printed table:
- **Soil nail.** The function's rows are keyed by su and SPT N. Examples:
  "cohesive stiff, su 50-150 kPa" gives 40-100 kPa, and "hard rock" gives
  500-3000 kPa. GEC-7 Table 4.4 has no such rows. Its closest rows are
  augered stiff clay at 6-9 psi (41-62 kPa) and, in Table 4.5, basalt at
  73-87 psi (503-600 kPa).
- **Micropile.** Only the first row matches Table 5-3 (soft silt & clay:
  A 35-70, B 35-95, C 50-120 kPa); the function folded C and D into one
  "C/D" column. Some rows are not in Table 5-3 at all ("medium stiff clay
  su 50-100 kPa", "soft rock", "hard rock"). Rows that are in it differ:
  - dense sand, Type B: 95-180 in the function, 120-360 in the print;
  - hard rock, Type B: 500-2500 in the function, but the print gives rock
    for Type A only (granite and basalt 1,380-4,200).
- **Geosynthetic RF.** No row matches:
  - HDPE geogrid: RF_ID 1.05-2.50 and RF_CR 2.0-5.0 in the function;
    Table 3-9 gives 1.20-1.45 (Type 1 backfill) and 1.10-1.20 (Type 2),
    and p. 3-33 gives RF_CR 2.6 to 5;
  - PP: RF_CR 1.5-4.5 in the function, 4.0 to 5 in the print.

## What changed and why

Option (b): **all three functions were removed** from the refs repo, along
with their data and tests. They cannot be re-cited:
- no held manual prints their values;
- correcting them to the print would turn each into a copy, under GEC-13's
  name, of a function that already exists with the print's values and
  cites the right manual:
  - `gec_7.tables.table_4_4a_bond_strength_coarse` / `_4_4b_..._fine` /
    `_4_5_bond_strength_rock`;
  - `micropile.tables.table_5_3_alpha_bond` (checked against p. 5-21: all
    11 rows and 4 types match);
  - `gec_11.tables.table_3_9_installation_damage` (checked against p. 3-32:
    all 7 rows match) and `table_3_11_pet_durability`.

The three functions had no catalog, manifest or search entries; their only
references in refs were their tests, the refs CLAUDE.md inventory and the
docstring of `gec_13/figures.py` `equation_11_1_ltds`. That docstring now
points to GEC-11 Table 3-9 (p. 3-32), the RF_CR list (p. 3-33) and
Table 3-11 (p. 3-36).

**App:** `funhouse_agent/adapters/_reference_common.py` `_CURATED_ALIASES`
changed:
- the two GEC-13 entries are gone;
- `micropile_grout_bond_strength` is now an alias of the micropile
  adapter's `table_5_3_alpha_bond`, which already had `grout_bond_strength`;
- `soil_nail_bond_strength` is not re-pointed. GEC-7 splits bond across
  three tables (coarse, fine, rock), and sending a clay question to the
  coarse table would be wrong. GEC-7's `soil_nail_bond_strength_coarse` /
  `_fine` / `_rock` aliases stay, and the dispatch layer's did-you-mean
  offers them.

No tool description, prompt or page fingerprint changed: the reference
methods are not on either page's tool list.

## Before and after

| Call | Before | After |
|---|---|---|
| gec13 `micropile_grout_bond_strength`, dense sand, Type B | 95-180 kPa, cited "GEC-13 Table 10-1" | no such method in gec13 |
| micropile `micropile_grout_bond_strength`, sand (fine-coarse, med.-very dense), Type B | alias did not exist | 120-360 kPa, FHWA-NHI-05-039 Table 5-3, p. 5-21 |
| gec13 `soil_nail_bond_strength`, cohesive stiff | 40-100 kPa, cited "GEC-13 Table 9-2" | no such method; gec7 `soil_nail_bond_strength_fine`, augered stiff clay, gives 6-9 psi (41-62 kPa) |
| gec13 `table_11_1_geosynthetic_reduction_factors`, HDPE geogrid | RF_ID 1.05-2.50, RF_CR 2.0-5.0 | removed; gec11 `table_3_9_installation_damage` HDPE uniaxial: 1.20-1.45 / 1.10-1.20 |

## Tests

Refs (venv with pymupdf/numpy/scipy, the branch installed editable), judged
by exit code:
- `tests/test_gec13_tables.py tests/test_gec13_figures.py`: exit 0,
  136 passed.
- The same files plus `test_references_agent.py`, `test_cover_citations.py`,
  `test_figure_captions.py`, `test_micropile*.py`, `test_gec11*.py` and
  `test_gec7*.py`: 593 passed, 2 xfailed.
- New: `TestRemovedTables`. It checks the three names are gone, and pins
  micropile Table 5-3 cells (35-70 A soft clay, 145-385 D coarse sand,
  1,380-4,200 granite) and GEC-11 Table 3-9 cells (slit film 1.60-3.00 /
  1.10-2.00; HDPE 1.20-1.45) as printed.
- `tests/test_gec13*.py` now collects 197 tests (was 212: 20 removed, 5
  added). The refs CLAUDE.md total, 6,540, is the old 6,555 minus 15. This
  venv cannot collect the whole suite (4 modules need packages it lacks),
  but the same 15-test drop shows with and without the change.

App: see the bottom of this file.

## Release note

GEC-13 no longer answers soil-nail bond strength, micropile grout bond
strength or geosynthetic reduction factors: those values matched no printed
table. Ask GEC-7 Tables 4.4/4.5, micropile Table 5-3 or GEC-11 Table 3-9
instead. `micropile_grout_bond_strength` now returns the micropile manual's
Table 5-3, so its answers move (for example, dense sand Type B goes from
95-180 to 120-360 kPa).

## Noted, not fixed

- GEC-13's other `tables.py` functions (Tables 1-2, 1-3, 1-6, 2-1, 3-1, 4-1,
  7-2, 8-2) were not checked against the print in this item. They come from
  the same pre-REF-25 era. Checking them could change a reference answer,
  so the lead may want an item for it.
- GEC-11's RF_CR typical ranges (p. 3-33) have no lookup function.
- `gec_11.tables.table_3_9_installation_damage`'s docstring describes Type 2
  backfill as "max 3/4 in (19 mm)". The print says 3/4 in. (20 mm), D50
  about #30 (0.7 mm). This is wording only; the values are right.

### App tests

Run with `/tmp/va` (Python 3.12, `pip install -e packages/aec-agent-core -e .`)
and the refs branch installed editable over the PyPI copy
(`geotech_references.__file__` -> `/home/user/geotech-references/...`), judged
by exit code:
- `funhouse_agent/tests/test_reference_adapters.py`,
  `test_method_aliases.py`, `test_method_name_aliases.py`,
  `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  exit 0, 380 passed. No fingerprint moved, so nothing was re-recorded.
- Every app test file naming `gec13` or `_CURATED_ALIASES`
  (`test_gm179_unused_inputs.py`, `test_reference_adapters.py`): exit 0,
  329 passed.
- New: `TestRef61Gec13Removed` in `test_reference_adapters.py`. It checks
  that the three names and `soil_nail_bond_strength` are gone from the
  gec13 registry, and that micropile `micropile_grout_bond_strength`
  returns Table 5-3's 35-70 kPa (soft silt & clay, Type A).

**Merge order:** merge the refs branch first, then this app branch. If the
app branch merges while the app still pins refs 1.5.0, nothing breaks:
removing an alias only takes a name away, and the gec13 functions stay
callable under their table names until the refs pin moves.
