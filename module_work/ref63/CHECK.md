# REF-63: GEC-13's deep dynamic compaction depth, Eq. 4-1 as printed

Branches `claude/ref-63-ddc-depth` in geotech-references (52a38c1) and in
the app (the submodule pinned there).

## What the print says

GEC-13 Vol I, FHWA-NHI-16-027, Ch. 4, Sec. 2.3.3.1 "Depth and Degree of
Improvement", p. 4-40 (read off the PDF, `docs/GEC 13 Vol 1.pdf`, PDF page
index 257, and matching the library's text, section 4.2.3.3.1):

> D = n (WH)^1/2   [Eq. 4-1]
> where, D = depth of improvement in meters
> n = empirical coefficient that is approximately 0.5 for most soils. As
> shown in Figure 4-21, n varies between 0.3 and 0.8. A value of 0.4 is
> typically used for landfills.
> W = mass of tamper in Megagrams (metric tonnes = 1.10 US ton)
> H = drop height in meters (1 meter = 3.28 feet)

p. 4-41: "Figure 4-21. Trend between apparent maximum depth of influence
and energy per drop" (Mayne et al. 1984). No table of n by soil type is
printed. The case history on p. 4-61 uses n = 0.4 for a landfill.

GEC-13 Figure 4-1 (p. 4-2) is "Schematic illustration of deep dynamic
compaction", so the old function name cited a figure that is not the
relation.

## What changed

geotech-references:
- `geotech_references/gec_13/tables.py`: `equation_4_1_ddc_depth(weight_tonnes,
  drop_height_m, deposit="most_soils", n=None)` replaces the soil-type
  table (`_DDC_N_COEFFICIENTS`, four Lukas 1995 types, n 0.35 to 0.6, not
  printed). n = 0.5 by default ("most soils"), `deposit="landfill"` gives
  0.4, a given `n` must lie in 0.3 to 0.8 (Figure 4-21's range, refused
  outside). The result always carries the depth at the n used
  (`depth_m`), the depths at 0.3 and 0.8 (`depth_low_m`, `depth_high_m`)
  with `range_basis`, `n_basis`, `reference` ("FHWA-NHI-16-027 (GEC-13)
  Vol I, Ch. 4, Eq. 4-1, p. 4-40; range of n from Figure 4-21 (Mayne et
  al. 1984), p. 4-41") and `choices` (n, given or default, basis,
  alternatives). The four old soil types are refused with a message
  giving the printed n values.
- `figure_4_1_ddc_depth` kept as an alias (same signature, returns exactly
  what `equation_4_1_ddc_depth` returns; its docstring says so).
- `agents/gec13_agent.py`: both names carry the Eq. 4-1 source label in
  `_FUNCTION_REFS`.
- `tests/test_gec13_tables.py`: the old `TestFigure41` (soil types)
  replaced by `TestEquation41DdcDepth` (by hand at n = 0.3, 0.4, 0.5, 0.65,
  0.8 with made-up W = 12 Mg, H = 18 m; round numbers W H = 100; landfill
  0.4; n outside the range refused; old soil types refused; citation and
  docstring; alias identity; agent source labels).
- `CLAUDE.md`: GEC-13 row (16 functions with the alias, 222 tests), the
  table-functions line, suite count 6,691.

App:
- `geotech-references` submodule pinned at 52a38c1.
- No alias change needed: `_reference_common`'s mechanical alias
  `ddc_depth` now resolves to `equation_4_1_ddc_depth` (alphabetically
  first; both names are the same relation anyway). Grep of `webapp/`,
  `funhouse_agent/`, `calc_package/`, `report_ingest/`, `docs/`,
  `validation_examples/`: nothing calls `figure_4_1_ddc_depth`, `ddc_depth`
  or a soil type such as `pervious_coarse`. `ground_improvement` does not
  call it.

## Before and after (made-up W = 20 Mg, H = 20 m, (W H)^0.5 = 20 m)

| Call | Before | After |
|---|---|---|
| default | 10.0 to 12.0 m ("pervious_coarse", n 0.5-0.6) | 10.0 m (n = 0.5), range 6.0 to 16.0 m (n 0.3-0.8) |
| semi_pervious | 7.0 to 10.0 m | refused (not printed) |
| impervious | 7.0 to 8.0 m | refused (not printed) |
| waste_fills | 7.0 to 10.0 m | refused; `deposit="landfill"` gives 8.0 m (n = 0.4) |

## Tests

- refs `tests/test_gec13_tables.py`: 97 passed; all GEC-13 tests 222 passed.
- refs whole suite (`pytest -n 8 tests`): 6,690 passed, 1 xfailed
  (includes the agent/registry and cover-citation tests). Collected 6,691.
- app `funhouse_agent/tests/test_reference_adapters.py`,
  `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`:
  336 passed. No fingerprint moved (no app tool description changed), so
  no re-record.
- app `test_reference_result_sizes.py`, `test_defaults_used.py`,
  `test_method_aliases.py`, `test_method_name_aliases.py`,
  `test_catalog_routing_hints.py`: 951 passed, 23 skipped, 1 failed. The
  failure is `test_defaults_used[profile_figure|plot_data]` (`logx`/`logy`
  change nothing in the result), in a module this change does not touch
  (no reference lookup in its path); noted for the lead, not fixed.

## Release note

GEC-13's deep dynamic compaction depth (`equation_4_1_ddc_depth`, old name
`figure_4_1_ddc_depth`, alias `ddc_depth`) now follows Eq. 4-1 (p. 4-40) as
printed: it returns one depth at n = 0.5 ("most soils") by default, or at
n = 0.4 for a landfill, with the depths at n = 0.3 and 0.8 (Figure 4-21's
range) beside it. The soil types it took before (`pervious_coarse`,
`semi_pervious`, `impervious`, `waste_fills`; n 0.35 to 0.6 after Lukas
1995) are not printed in GEC-13 and are refused. For a 20 Mg tamper
dropped 20 m the default answer moves from 10.0 to 12.0 m to 10.0 m
(range 6.0 to 16.0 m); a waste fill's 7.0 to 10.0 m becomes a landfill's
8.0 m.

## Noted, not fixed

- `figure_4_1_ddc_depth` is listed in the app's method catalog beside the
  new name (the app scans the module's public functions); its brief says
  "Alias of equation_4_1_ddc_depth".
- Figure 4-21's data points are not digitized; only its three printed
  lines (0.3, 0.5, 0.8) are used.
