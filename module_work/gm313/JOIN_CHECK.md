# GM-313, the join: a reinforced wall's calc package ends with its drawing

Branch `feat/wall-package-sheet` (from `merge/gm313`). The owner's ask: "a
concrete cantilever retaining wall with reinforcing and dimensions and call
outs at the end of a calc package". Before this, the design
(`cantilever_wall_structural_design`, GM-312) and the drawing layer
(`dxf_export`, GM-313) were not connected: the agent had to call
`draw_wall_section` itself.

Contract changes: none. The `reinforced_wall_section/1` keys are unchanged;
the join only fills the shear key's `key_width` and `key_distance_from_toe`
where the package was given them (they were already in the contract).

## What changed, and why

All in the agent layer, `funhouse_agent/adapters/calc_package.py`, which may
import both modules; `retaining_walls`, `dxf_export` and `calc_package`
still import none of each other.

1. `retaining_wall_package`, `wall_type="cantilever_reinforced"`: after the
   design runs, `_wall_drawing_description` takes the result's
   `reinforcement` description, `_wall_sheet` draws it with
   `dxf_export.wall_section_drawing` and `render_sheet`, and the sheet goes
   to `generate_calc_package(sheets=[SheetData])`: the PDF's last page (the
   sheet's own vector page, landscape, at the sheet's size); the PNG under
   "Drawings" in html and latex. The DXF and the PNG are written beside the
   package with its own stem, `<stem>_reinforcement.dxf` and `.png`
   (verified saves), so in the app they land in the conversation folder.
2. Title block from the package's own header inputs: project name, project
   number, engineer (drawn by), checker (checked by), date (today's when none
   is given, as the package prints it). A field over the title block's 80
   characters is cut with "..." and said in `drawing_warnings` (the drawing
   layer would otherwise refuse it and cost the sheet). Title, stamp and
   layout are the drawing code's.
3. Sheet size: drawn on letter first; `drawing.sheet.scale` is the scale
   `wall_section_drawing` chose (the largest standard scale that fits). If
   it is smaller than 1:50, or letter takes the wall at no scale, the wall
   is drawn again on tabloid. No helper was added to `dxf_export`: the
   drawing already answers "which scale would this sheet give".
4. Shear key: the design gives only its depth. The package's own
   `key_distance_from_toe` and `key_width` (already checked by
   `_shear_key_place`) go into the description, each where given; one not
   given stays null and the sheet draws it as the package's figure does
   (front face under the stem's, as wide as the stem) and says so.
5. Result: `drawing_dxf_path`, `drawing_png_path`, `drawing_sheet`,
   `drawing_scale`, `drawing_warnings`; `output_path` is still the package.
   `render_figures` for the package saves the sheet's PNG with the figures
   (`reinforcement_sheet.png`, title "Reinforcement sheet") and reports its
   size, scale and warnings.
6. The drawing never costs the calculation: a sheet that cannot be drawn,
   or that `generate_calc_package` cannot take (it is then run again without
   it), leaves the package written without the sheet, the drawing keys null,
   and `warnings` saying why. In `render_figures` it costs the sheet only.
   The sheet is drawn after the calculation's raised warnings are taken, so
   a warning the drawing raises is not printed in the package as the
   calculation's.
7. Texts made true: the `wall_type` text (was "no reinforcement drawing");
   the matching row of `docs/user_manual/manual.html` hand-edited, and the
   `draw_wall_section` description row; `draw_wall_section`'s `description`
   text now says `key_width`/`key_distance_from_toe` may be null with a key
   and `end.ldh_mm` may be null; the package's `returns` name the new keys;
   one paragraph each in `retaining_walls/DESIGN.md` and
   `dxf_export/DESIGN.md`.

No new parameter. Nothing lets the model give code or a path for the sheet;
the stamp cannot be turned off.

## What I looked at

The pdf package of a made-up 5 m wall with a 12 kPa live surcharge (5.0 m
high, B 3.1 m, toe 0.65 m, stem 0.305/0.5 m, base 0.5 m): 38 pages, pages
1-37 letter portrait, page 38 the sheet, 17 x 11 in. (tabloid, 1:50;
letter would have drawn it at 1:100). The last page rendered at 110 dpi
(`join_package_last_page.png` here) and the `_reinforcement.png` (200 dpi)
read: stamp, section with hatch, ground both sides, bars A-G with callouts,
dimensions (5000, 4500, 500, 650/500/1950, 3100, covers 75/75/75/100, stem
top 305), bar schedule, notes, scale bar, and the title block carrying the
package's header (project, number, drawn, checked, date). Nothing to fix in
the join; the sheet is the drawing layer's, unchanged.

## Tests

New `funhouse_agent/tests/test_wall_package_sheet.py` (made-up walls): a pdf
package has one page more than the same package with the drawing made to
fail, its earlier pages unchanged and its last page tabloid landscape; the
DXF beside it reads back with a clean audit and its callout marks and texts,
and its schedule's marks, sizes and spacings, are exactly the design's
`reinforcement` bars; the title block is the package's header; a tall wall
goes to tabloid because letter would be 1:100 and a low one stays on letter;
an html package holds the sheet's PNG (byte for byte the file beside it); a
key with its place given is drawn there (concrete outline read from the
DXF), one without is drawn under the stem with the note; a drawing made to
raise leaves the package written with `warnings` saying why and no sheet
files; `render_figures` returns the sheet's PNG with the figures.

## Gate (local, one test file at a time, by exit code, 2026-11-13)

Run on the uncommitted tree from `feat/wall-package-sheet` (based on addfdcf).
Every file exited 0; the page fingerprint test passed with no re-record.

| # | Test file | Result |
|---|---|---|
| 1 | `funhouse_agent/tests/test_wall_package_sheet.py` | 8 passed |
| 2 | `funhouse_agent/tests/test_gm312_wall_structural_adapter.py` | 5 passed |
| 3 | `funhouse_agent/tests/test_defaults_used.py -k "wall or retaining"` | 126 passed, 734 deselected |
| 4 | `funhouse_agent/tests/test_calc_package_tool_parity.py` | 181 passed |
| 5 | `funhouse_agent/tests/test_calc_package_adapter.py` | 84 passed |
| 6 | `funhouse_agent/tests/test_calc_package_figure_layout.py` | 13 passed |
| 7 | `funhouse_agent/tests/test_writer_output_paths.py` | 11 passed |
| 8 | `dxf_export/tests/test_drawing_sheet.py` | 70 passed |
| 9 | `calc_package/tests/test_sheets.py` | 11 passed, 1 skipped |
| 10 | `webapp/tests/test_page_fingerprints.py` | 13 passed |
| 11 | `webapp/tests/test_public_text.py` | 5 passed |

## Review (independent, Opus, 2026-11-13)

PASS WITH FIXES, nothing that had to be fixed for the numbers. The reviewer
ran the structural design for the same 5 m wall itself and compared every
bar set (A to G: mark, size, spacing, place, steel required and provided)
with the DXF's schedule and callouts and with both PNGs: they agree, and
the package's `reinforcement` equals the design tool's. Dimensions, the
title block, the tabloid sheet at 1:50, the failure path (the drawing made
to raise: the package written, `warnings` says why, no sheet files), the
html package's PNG and the data-only promise all held. Fixed after it:
texts the join had made untrue (`retaining_walls/DESIGN.md` twice, the key
parameters' descriptions and their manual rows, a code comment), and with
the key's place given and its width not, the sheet and the package's own
wall figure no longer name the stem as the key's place: they say its width
is assumed (`dxf_export/wall_section.py`, `retaining_walls/calc_steps.py`,
with a test each).

## Noted, not fixed

- The title block's DWG field is left blank.
- A wall that also draws at 1:100 on tabloid still goes to tabloid (the rule
  as written: letter when the wall fits at 1:50, else tabloid).
- Callout D (toe, bottom) has its leader on the bottom bar under the heel:
  correct, as that bar is drawn full length (note 8), but an engineer looks
  for the toe steel at the toe (the drawing layer's layout, GM-313).
