# GM-313: the drawing layer, and the first drawing

Branch `claude/gm-313-drawing-layer`, from master 0dac9c2. Two commits:
the work (4fea1bc) and the page fingerprints re-recorded on their own
(84386fa), then this file.

## Contract changes

None. `reinforced_wall_section/1` is read exactly as the brief's table
defines it: no key renamed, dropped or added. Three readings of it that the
other session should know:

- `in_section` with `face: "both"` is refused ("give one set per face");
  the table allows `"both"` for a set placed in each face, and the example
  uses it only for `along_wall` sets.
- `end` (a hook) is drawn only on a stem bar with `runs: "in_section"` and
  `into: "base"`; anywhere else it is refused by name.
- `heel_length`, if given, must equal `base_width - toe_length -
  stem_thickness_base` within 2 mm, or the description is refused.

The route the brief asked for works: ezdxf's drawing add-on renders the
DXF document itself, with no system fonts. The matplotlib fallback was not
needed.

## What the print says

- **USACE EM 1110-2-2104 (1 November 2023), para. 2-10 a and Figure 2-1,
  p. 9** (PDF page index 14 of
  `geotech-references/docs/EM_1110-2-2104_2025.pdf`; I rendered the page
  and looked at it). The text: "Full moment transfer is attained by
  turning the hooks from the beam as shown on the left side of Figure
  2-1." The left sketch, "Correct Reinforcement Orientation", has the bar
  at one face of the stem turned at the bottom of the base TOWARD THE
  OTHER FACE of the stem, under the stem. The right sketch, "Incorrect",
  turns it away from the stem.
- So a hooked back-face (retained-side) stem bar is drawn turned toward
  the toe, and a hooked front-face bar toward the heel. Tested.
- **No held reference prints a standard hook's bend diameter or
  extension.** I searched the reference library's text for "standard
  hook", "bend diameter" and "12 d_b" under `em_2104`, `ufc_structural`
  and the GEC folders and found nothing. I did not use values from memory.
  The hook's bend (a 2 mm radius on the sheet) and its tail (to the line
  of the stem's opposite face) are drawn schematically, and every wall
  sheet's note 6 says so: "The hook's bend and tail are schematic, not
  detailed."

## Calls made

1. **Hook shape is schematic** (above). The other form would be a
   detailed standard hook, which needs a printed bend diameter and
   extension that the library does not hold.
2. **One model space, no paper space.** The object is at true size in mm;
   the sheet is drawn round it N times its printed size, so plotting at
   1:N gives the sheet. A paper-space layout with a viewport is the other
   form; one model space opens the same in every CAD program and renders
   the same in the add-on.
3. **The PNG is the PDF page rasterised** (PyMuPDF, 200 dpi), not a second
   render through the add-on's matplotlib backend. It cannot differ from
   the PDF.
4. **Wall dimensions are in whole millimetres with no second unit**, and
   the sheet says "All dimensions are in millimetres." A second unit in
   brackets is supported by the layer (`second_unit`: in, ft, ft-in, m)
   but does not fit a 450 mm dimension at 1:50. The callouts carry both
   units, from the description.
5. **Bars thinner than prints legibly are drawn wider than to scale** (0.5
   mm on the sheet for a line, 0.9 mm for a dot) and the result's warnings
   say so. At 1:50 a No. 6 bar to scale is 0.38 mm.
6. **The retained ground is drawn from the top of the stem**; the
   description gives a slope and no level.
7. **A stem bar stands on the base's bottom steel** (it runs down to the
   bottom in-section bars' top). If the description's `ldh_mm` is more than
   the depth that leaves, the result warns (the front-battered example
   does: 310 mm given, 284 mm available).
8. **`draw_sheet` takes the sheet size and scale inside its `drawing`**;
   `draw_wall_section` takes them as `sheet_size` and `scale`.
9. **Sheet sizes are always landscape**, as the brief's list reads.

## What I changed and why

`dxf_export/` (new files; `writer.py` untouched):

| File | What |
|---|---|
| `drawing.py` | The typed description: `Drawing`, `Outline`, `Line`, `Bar`, `Dimension`, `Callout`, `Table`, `TitleBlock`, `Sheet`. Strict `from_dict` (unknown keys refused by name, in plain words), `to_dict`. |
| `sheet.py` | `build_dxf`: one ezdxf document. Our own dimension style `GSE-MM` (the stock `EZDXF` style multiplies by 100), two text styles, 16 fixed layers, the border, title block, notes, table, scale bar and stamp. Chooses the scale. |
| `render.py` | `render_sheet`, `write_sheet`: PDF page from the document at exactly 1:N, PNG from the PDF. |
| `_fonts.py` | Registers matplotlib's DejaVu Sans with ezdxf's font manager. |
| `wall_section.py` | `validate_wall_description`, `wall_section_drawing`. |
| `results.py` | `DrawingSheetResult` added. |
| `DESIGN.md` | A section "Drawing sheets". |

`calc_package/`: `SheetData`, `CalcPackageData.sheets`,
`generate_calc_package(sheets=)`, `latex_renderer.append_sheets`. PDF: the
package is rendered without the sheets and each sheet's own vector page is
appended (`insert_pdf`) on either leg. HTML and LaTeX source: a last
section "Drawings" with the PNG. `calc_package` does not import
`dxf_export` (tested).

`funhouse_agent/adapters/dxf_export.py`: `draw_wall_section` and
`draw_sheet`, added at the end of the file with `METHOD_INFO.update`, away
from the `units` lines the `claude/gm-316-317-warnings-allowed-values`
branch changes (I diffed that branch's copy of the file first). Each takes
data only, writes `<name>.dxf`, `.pdf`, `.png` into the conversation
folder, and returns `output_path`, `pdf_path`, `png_path`, the sheet,
scale, counts and warnings. A name that does not end in `.dxf` is refused.

Also: two probes in `method_probes.json`; one writer-path test; the
module's one-line brief in `MODULE_REGISTRY` (fingerprint re-recorded in
its own commit: only the geotech page's system prompt moves, 26,465 to
26,556 characters; the catalog-size test passes); the user manual's rows,
hand-edited with the builder's own table functions, and
`manual_content.py`.

## Fonts and the read-only home

- **No system fonts.** `_fonts.ensure_fonts()` adds matplotlib's
  `DejaVuSans.ttf` and `DejaVuSans-Bold.ttf` to ezdxf's font manager in
  the process, before anything is measured or drawn; the sheet's two text
  styles name those files. Our copy replaces a system copy of the same
  name, so every host draws the same sheet.
- **How it is tested.** A fresh process empties ezdxf's font manager
  (`font_manager.clear()`, asserting neither DejaVu nor Arial is known),
  with HOME set to a path below a regular file, where nothing can be
  created by any user. It renders a sheet with every kind of lettering.
  The pixels equal those of a normal process. Without the registration
  ezdxf reports "no fonts available, not even fallback fonts"; I ran that
  once by hand to confirm the test can fail.
- **The font cache.** ezdxf loads or builds `~/.cache/ezdxf` when first
  imported and already carries on when it cannot write there. The test
  above covers it.

## Before and after

No calculated value changes. The made-up example
(`dxf_export/tests/fixtures/wall_section_example.json`, the brief's):

| | Before | After |
|---|---|---|
| A wall section with bars, dimensions and callouts | not possible | `wall_example.dxf`, `.pdf`, `.png` |
| Sheet and scale | - | letter landscape, 1:50, chosen by the code |
| DXF entities | - | 12 DIMENSION, 6 LEADER, 1 AR-CONC HATCH, 6 bar sets (44 dots), 204 entities on 11 layers, audit clean |
| Dimensions measured | - | 4500, 4050, 450 (x3), 2700, 1800, 300, 100, 75 (x3) mm |
| Calc package PDF with one sheet | PNG figures only | one more page, the sheet's own size, vector, no image |
| Calc package with no sheet | - | HTML and LaTeX source byte-identical to master (SHA-1 compared) |

## What I looked at, and what I changed after looking

Every sheet was rendered to PNG and opened.

- **The general test sheet.** Lettering overflowed its boxes: the stamp
  ran out of its frame and notes ran past the border. The style's width
  factor was not reaching TEXT entities; each now carries it.
- **The four wall sheets** (`wall_example`, `wall_low`,
  `wall_tall_long_heel` on tabloid, `wall_front_batter_key`):
  - the leaders to the base's bar sets crossed; the bottom bar's leader
    now starts further from the heel and callouts are ordered by height;
  - the base's top-cover number touched the front ground line on the
    keyed wall; numbers moved outside a dimension now sit closer to it;
  - the bar schedule was 2 mm too wide for a letter sheet; its "Place"
    text is shorter and the title explains "along".
- **Aligned dimensions.** Given right to left, the number printed upside
  down. They are now measured left to right on the same side.
- **Small dimensions.** A 60 mm dimension at 1:10 had its number over its
  arrowheads. A number now needs room for itself and both arrowheads, or
  it is set beyond `p2`.
- **EM 1110-2-2104 Fig. 2-1**, to read the hook's direction.

Still visible and left: the top extension line of the height dimensions
runs from the stem's top to the left and reads a little like a ground
line; the stem's cover numbers at the top are close together on a thin
stem. Both are legible.

## Tests (pytest exit codes)

Python 3.12 venv; ezdxf 1.4.5, PyMuPDF 1.28.2, matplotlib 3.11.2, planlens
0.15.1, geotech-references 1.6.0.

| Tests | Result |
|---|---|
| `dxf_export/tests` (61 new in `test_drawing_sheet.py`) | 98 passed, exit 0 |
| `calc_package/tests` (11 new in `test_sheets.py`) | 274 passed, 4 skipped (pdflatex not installed), exit 0 |
| `funhouse_agent/tests/test_defaults_used.py -k dxf` | 24 passed, exit 0 |
| `funhouse_agent/tests/test_writer_output_paths.py` | 11 passed, exit 0 |
| `funhouse_agent/tests/test_adapter_ergonomics.py` | 170 passed, exit 0 |
| `funhouse_agent/tests/test_catalog_routing_hints.py` | 14 passed, exit 0 |
| `webapp/tests/test_page_fingerprints.py` | 13 passed, exit 0 (after the re-record) |
| `webapp/tests/test_public_text.py` | 5 passed, exit 0 |
| All of `funhouse_agent/tests` | 4,240 passed, 30 skipped, 2 failed, exit 1 |
| `packages/aec-agent-core/tests/test_core_imports.py` | 5 passed |
| `module_work/doc10/check_manual.py` | OK |

The 2 failures in `funhouse_agent/tests` are not this branch's: both fail
the same way on master's code in this environment
(`test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
and `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`).
I checked by stashing the branch's changes and running them.

`check_manual.py --rebuild` reports one differing line pair (a soe
parameter table near line 1595). It differs on master too; the rows this
branch added match the rebuild.

## Not verified

- **The pdflatex PDF leg.** pdflatex is not installed here, so
  `test_the_latex_pdf_leg_appends_the_sheet_too` was skipped. The append
  is the same function on both legs and runs on the finished PDF; the
  Story leg is tested.
- **Opening the DXF in a CAD program.** None here. `doc.audit()` is clean
  on every sheet (tested) and the file reads back with ezdxf.
- **In the app.** Not run in the web app; the three files land in the
  conversation folder through the dispatch layer (tested).

## Release note

None: no calculated value changes. New for a user: `dxf_export` can draw a
sheet (`draw_sheet`, `draw_wall_section`), and a calc package can end with
one (`generate_calc_package(sheets=...)`; the agent's calc-package tools
do not pass sheets yet: that is the join after GM-312).

## Noted, not fixed

- ezdxf moves each line of a pattern hatch by a random hair when it
  renders, so two renders of one sheet differ in a few pixels of the
  concrete hatch. The DXF is identical.
- A sheet's PDF is about 310 KB, because lettering is outlines.
- The PDF's lettering is not selectable text (accepted in the brief).
- A CAD program without DejaVu Sans shows the DXF in its substitute font.
- A wall with several in-section sets on one face stacks them inward in
  the order given; not drawn in any example.
- The hint in "the largest standard scale that fits is 1:N", when a given
  wall scale is refused, comes from the layout at the refused scale and
  could be one step off.
- `geotech-references`: no defect found. The library holds no printed
  standard-hook geometry (ACI 318 is not a held reference).

## Review fixes (2026-11-13)

From the independent review of b7372c3 (merge after fixes). One line per
finding; "not fixed" items are the lead's or noted.

- A1 fixed: the PDF page is the sheet's own rectangle (`render_box` from
  `SheetInfo.origin`, no margins); `_fonts.wrap` breaks a word wider than
  its column; tested with a 290-character word and a 160-character title
  (the border lands 10 mm inside the page).
- A2 fixed: a shear key given by its depth only (GM-312) is drawn under the
  stem, as wide as the stem's base (as `retaining_walls`' wall figure);
  only its depth is dimensioned; a note and a warning say it is assumed.
- A3 fixed: the three GM-312 walls fit letter (plain 1:50, key_surcharge
  1:100, front_batter 1:50): the basis is not repeated, the drawing's own
  notes are two, notes and callouts at a 3.8 mm pitch, table rows 4.5 mm,
  the notes strip 80 mm and 100 or 120 only when the notes need it.
- A4 fixed: every dimension number is placed where the code lays it out
  (`dimtix` 1, an explicit location); a chain link too narrow for its
  number gets a row of its own, the toe's number set outward; the key
  depth's number gets room; tested: no two numbers overlap on any GM-312
  wall.
- A5 fixed: stem bars stand on the whole bottom mat (bars along the wall
  included); tested.
- B1 fixed: the basis is appended only when no note carries it.
- B2 fixed: at most 20,000 points in a drawing; a wall asking for more
  than 5,000 bars along the wall is refused before layout.
- B3 fixed: a test runs `render_pdf(renderer="pdflatex")` with the compile
  stood in for and checks the sheet's page is appended.
- B4, B5 fixed: the A1 test; the three GM-312 descriptions are a fixture
  (`dxf_export/tests/fixtures/gm312_real_descriptions.json`): each
  validates, draws on letter, audits clean, prints no "None".
- C1 fixed: control characters dropped from text. C2 fixed: a title past
  three lines ends in "..." with a warning.
- C3 (a battered face's cover measured horizontally), C4 (the second unit
  is text, not DIMALT), C5 (base leaders run diagonally), C7 (ezdxf
  private attributes, with a fallback): noted, not fixed.
- C6 (`wall.dxf` overwrites an existing `wall.pdf`/`wall.png` in the
  folder): for the lead, a product call.
- Lead's L1: dimension, extension and leader lines are the thinnest (0.13
  mm); ground lines 0.50 mm with 45-degree strokes under them (0.25 mm).
- Lead's L2: one example set is committed (`wall_example` .dxf/.pdf/.png)
  and `wall_real_design.png` (GM-312's key_surcharge wall);
  `make_examples.py --all` still writes the low, tall and keyed walls
  named above, uncommitted.
