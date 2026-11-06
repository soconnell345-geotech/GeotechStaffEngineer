# REF-61: three GEC-13 table functions that cited tables printing something else

Branches: refs `claude/ref-61-gec13-tables` (head 23f1d7e), app
`claude/ref-61-gec13-tables` (7ee139a). The review fixes are refs cef7d66
and an app commit, both on `merge/ref61` (see "Review and fixes" below).
This item makes no refs version bump and no change to the app's
`geotech-references>=` pin; the release needs both (see below).

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
  `_fine` / `_rock` aliases stay. The did-you-mean does not offer them for
  a GEC-13 call: it scores only the called module's own methods. A gec13
  `soil_nail_bond_strength` call gets "Unknown method", suggests
  `table_7_2_deep_mixing_strength`, `table_8_2_jet_grouting_systems` and
  `load_chapter`, and lists gec13's methods. gec13
  `micropile_grout_bond_strength` / `grout_bond_strength` behave the same
  way, with no pointer to `micropile`. GEC-7's tables are suggested only
  on a `gec7` call.

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
table. Old GEC-13 calls to the three removed lookups now get an
unknown-method error, and the error does not point to another module. This
covers `table_9_2_nail_bond_strength`, `table_10_1_micropile_bond_stress`,
`table_11_1_geosynthetic_reduction_factors` and their aliases
(`nail_bond_strength`, `soil_nail_bond_strength`, `micropile_bond_stress`,
`micropile_grout_bond_strength`, `grout_bond_strength`,
`geosynthetic_reduction_factors`). Each subject is answered instead by:
- soil-nail bond strength: GEC-7 (`gec7`) Tables 4.4a (coarse-grained) and
  4.4b (fine-grained), p. 84, and Table 4.5 (rock), p. 85;
- micropile grout bond strength: the micropile manual (`micropile`)
  Table 5-3, p. 5-21;
- geosynthetic reduction factors: GEC-11 (`gec11`) Vol I Table 3-9 (RF_ID,
  p. 3-32), Table 3-11 (RF_D for PET, p. 3-36), and the typical RF_CR
  ranges by polymer on p. 3-33 (PET 2.5 to 1.6, PP 5 to 4.0, HDPE 5 to
  2.6). That list is text only: `gec11` `search_sections` finds section
  3.5.2.c, which carries it.

`micropile_grout_bond_strength` on `micropile` now returns the micropile
manual's Table 5-3, so its answers move (for example, dense sand Type B
goes from 95-180 to 120-360 kPa). GEC-11 `table_3_11_pet_durability`
asked for 'geotextile' now returns both printed rows (1.6 / 2.0 and
1.15 / 1.3), because Table 3-11 lists geotextiles in both; before, it
returned only 1.6 / 2.0.

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
  returns the same dict as `table_5_3_alpha_bond`, with
  `alpha_bond_min_kpa` 35 and `alpha_bond_max_kpa` 70 (Table 5-3, soft
  silt & clay, Type A).

**Merge order:** merge the refs branch first, then this app branch, and
move the app's `geotech-references` submodule pointer to the refs merge
commit in the same app merge. Without that move, CI fails: it checks out
the submodule and installs it, and `TestRef61Gec13Removed` then finds the
three names still in the gec13 registry (the review ran this: 3 failed,
2 passed against refs f62a1d1).

## Review and fixes (2026-11-06)

An independent review checked the change against the PDFs and agreed that
removing all three functions was right. Its findings, and what was done:

1. **CI fails unless the submodule moves in the same merge** (high, merge).
   Not a code change. The merge-order paragraph above is corrected. The
   lead moves the app's submodule pointer to the refs merge commit in the
   app merge.
2. **PyPI users keep the old values until refs is released** (medium,
   release). Not fixable in the branch; see note 2 below.
3. **`equation_11_1_ltds` gave the removed table's unprinted ranges**
   (refs cef7d66). The `rf_id`, `rf_cr` and `rf_cbd` descriptions now give
   GEC-11 Vol I as printed, each checked on the page:
   - RF_ID 1.10 to 3.00, Table 3-9, p. 3-32;
   - RF_CR PET 2.5 to 1.6, PP 5 to 4.0, HDPE 5 to 2.6, p. 3-33;
   - RF_D "can vary typically from 1.1 to 2.0", p. 3-33; for PET,
     Table 3-11 1.15 to 2.0, and a default of 1.3 when the Table 3-12
     index criteria are met, p. 3-36.
4. **Three more GEC-13 table functions are mis-cited** (Tables 4-1, 7-2,
   8-2). Not fixed here: the lead files it as its own backlog item.
5. **The did-you-mean claim was wrong.** Corrected under "What changed and
   why": a gec13 call gets suggestions from gec13's own methods only. No
   cross-module redirect was added; the review marked it optional, with no
   eval evidence of the miss.
6. **README counted 18 GEC-13 functions** (refs cef7d66). Counted with
   `inspect`: gec_13 15, the FHWA group 143, the package 1,230. Every other
   README row matched the count.
7. **The new app test's assertion was weak** (this app commit). The test
   now checks that the alias returns the same dict as
   `table_5_3_alpha_bond`, `soil_type` `silt_clay_soft`, `micropile_type`
   `A`, `alpha_bond_min_kpa` 35 and `alpha_bond_max_kpa` 70.
8. **CHECK details.** The refs commit is named correctly (23f1d7e; the
   old name d3760dc does not exist in the refs repo). The release note now
   names GEC-11 Table 3-11 and the p. 3-33 RF_CR list beside Table 3-9,
   says the old gec13 calls now get an unknown-method error, and names the
   module that answers each subject.
9. **GEC-11 Table 3-11 labels** (refs cef7d66). Read on p. 3-36 (rendered
   to PNG): the review's reading is right. Row 1 is "Geotextiles, Mn <
   20,000, 40 < CEG < 50" (1.6 / 2.0); row 2 is "Coated geogrids,
   Geotextiles, Mn > 25,000, CEG < 30" (1.15 / 1.3). The labels now match,
   so `table_3_11_pet_durability("geotextile")` returns both rows; before
   it returned only 1.6 / 2.0. The docstring says why, and
   `tests/test_gec11_tables.py` pins both rows, labels and values, as
   printed. The review's other label notes (micropile glacial till and
   silt/clay wording, GEC-11 Table 3-9 "19 mm") are wording only and were
   not changed.

The refs CLAUDE.md now reads gec_11 136 tests and 6,541 collected (one
test added; `pytest --collect-only -q tests/` collects 6,541).

**Notes for the lead:**
1. The app merge must move the `geotech-references` submodule pointer to
   the refs merge commit (the lead does this). Otherwise CI fails (finding
   1).
2. The release needs a refs version bump and a raised
   `geotech-references>=` pin in the app's `pyproject.toml`. Three public
   functions are removed, so 1.6.0 or 2.0.0 is the owner's call at
   release. Until then, a PyPI install keeps the old gec13 functions and
   values.

**Tests after the fixes** (Windows venv, `PYTHONPATH` set to the refs
worktree, `geotech_references.__file__` confirmed; judged by exit code;
logs in `C:/dev/tmp/ref61_fix/`):
- Refs: `test_gec13_tables.py`, `test_gec13_figures.py`,
  `test_gec11_figures.py`, `test_gec11_tables.py`, `test_gec11_text.py`,
  `test_references_agent.py`, `test_cover_citations.py`: exit 0, 335
  passed, 2 xfailed.
- App (`funhouse_agent.__file__` in the app worktree):
  `funhouse_agent/tests/test_reference_adapters.py`,
  `test_method_name_aliases.py`, `test_method_aliases.py`,
  `test_gm179_unused_inputs.py`, `webapp/tests/test_page_fingerprints.py`,
  `webapp/tests/test_public_text.py`: exit 0, 400 passed. No fingerprint
  moved.
