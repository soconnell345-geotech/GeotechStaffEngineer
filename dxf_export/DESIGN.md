# dxf_export — Design Notes

## Purpose

Export cross-section geometry to DXF files. Complements `dxf_import` and
`pdf_import` to enable round-trip workflows: PDF/DXF → extract → analyze →
export new DXF.

## Layer Conventions

| Layer            | Color | Entity     | Source field          |
|------------------|-------|------------|-----------------------|
| SURFACE          | 3     | LWPOLYLINE | surface_points        |
| BOUNDARY_<name>  | 1     | LWPOLYLINE | boundary_profiles     |
| GWT              | 4     | LWPOLYLINE | gwt_points            |
| NAILS            | 6     | LINE       | nail_lines            |
| ANNOTATIONS      | 7     | TEXT       | text_annotations      |

Colors follow AutoCAD ACI standard: 1=red, 3=green, 4=cyan, 6=magenta, 7=white.

## Coordinate System

- Coordinates are (x, z) matching the geotechnical convention used by
  `dxf_import` and `slope_stability`.
- DXF writes to the XY plane — the z coordinate maps to DXF's Y axis.

## Units

The `$INSUNITS` header is set based on the `units` parameter:
- 6 = meters (default)
- 2 = feet
- 4 = millimeters
- 5 = centimeters
- 1 = inches

## DXF Version

Default is `R2010` — widely compatible with AutoCAD 2010+ and most DXF readers.

## Input Formats

### Raw geometry (export_to_dxf / to_dxf_bytes)
- `surface_points`: `[(x, z), ...]`
- `boundary_profiles`: `{"Clay": [(x, z), ...], ...}`
- `gwt_points`: `[(x, z), ...]` or None
- `nail_lines`: `[{"x_head": ..., "z_head": ..., "x_tip": ..., "z_tip": ...}, ...]`
- `text_annotations`: `[{"text": "...", "x": ..., "y": ...}, ...]`

### Parse result (export_parse_result)
- Accepts `DxfParseResult` or `PdfParseResult` directly
- PdfParseResult is auto-converted via `pdf_import.to_dxf_parse_result()`

## Dependencies

- `ezdxf >= 1.4` (same optional dependency as `dxf_import`)

## Drawing sheets (BACKLOG GM-313)

Fixed code that takes a typed description of a drawing and writes three
files of the ONE drawing: a DXF, a one-page vector PDF sheet and a PNG. No
model-written code anywhere, so it works where plot code is off.

| File | Holds |
|------|-------|
| `drawing.py` | The typed description: `Drawing` and its parts (`Outline`, `Line`, `Bar`, `Dimension`, `Callout`, `Table`, `TitleBlock`, `Sheet`), each with strict `from_dict` (an unknown key is refused by name) and `to_dict`. JSON schema name `drawing_sheet/1`. |
| `sheet.py` | `build_dxf(drawing)`: the description as one ezdxf document. Chooses the scale, lays out the sheet. |
| `render.py` | `render_sheet` / `write_sheet`: the PDF page drawn from that document by ezdxf's drawing add-on (PyMuPDF backend), and the PNG rasterised from that PDF page. |
| `wall_section.py` | The first drawing: `wall_section_drawing(description)` lays out a `reinforced_wall_section/1` description by fixed rules. |
| `_fonts.py` | The lettering: one font the code always has. |

### One drawing, three files

The object is drawn in model space at true size in MILLIMETRES. The sheet
(border, title block, notes, table, scale bar, stamp) is drawn round it in
model space too, N times its printed size, so plotting model space at 1:N
gives the sheet. `render.py` does exactly that: `Frontend` +
`PyMuPdfBackend`, a page of the sheet's size, `fit_page=False`,
`scale=1/N`, no page margins, and the page is the sheet's own rectangle in
model space (`render_box`, from `SheetInfo.origin`), whatever the content
holds: the border lands 10 mm inside the page (tested, also with a word
far wider than its column, which is broken). ezdxf's default, centring on
the content's box, let one over-long word slide the whole sheet, stamp
and all, off the page (GM-313 review A1). The PNG is the
PDF page rasterised at 200 dpi, so it cannot differ from the PDF. Black on
white (`ColorPolicy.BLACK`); the DXF keeps layer colours for CAD.

No paper-space layout or viewport is used: one model space opens the same
in every CAD program and renders the same in the add-on.

The PDF's lettering is drawn as outlines (not selectable text); accepted.

### What a description may say, and what is the code's

A description holds outlines (closed polygons, an optional hatch from
`concrete`, `earth`, `gravel`, `none`), lines, bars, dimensions, callouts,
notes, one small table, the title block's fields and the sheet (size, and
optionally the scale). It names what a thing IS (`layer: "concrete"`), not
how it is drawn.

Fixed by the code and closed to a description: layer names, colours and
line weights (`sheet.LAYERS`, all `GSE-...`); text heights (2.5 mm on the
sheet, 3.5 mm for the title and the stamp; tested); the border and the
title block's layout; the scale bar; and the stamp `CALCULATION SKETCH -
NOT FOR CONSTRUCTION` on every sheet (`sheet.STAMP`; no input removes it).

- **Dimensions** are real DIMENSION entities in our own style `GSE-MM`
  (`dimlfac` 1, `dimdec` 0, sized on the sheet and scaled by `dimscale` =
  N). The stock `EZDXF` style multiplies the measurement by 100 and is not
  used. The entity's text is `<>` (its own measurement) plus an optional
  suffix and, in brackets, the same measurement in a second unit worked
  out by the code: a description has no key for a dimension's number.
  Whole millimetres. `offset_mm` is on the sheet. A number wider than its
  dimension is set beyond `p2`.
- **Bars**: a bar in the plane of the drawing is an LWPOLYLINE whose
  constant width is the bar's diameter, its corners rounded with bulges; a
  bar seen end-on is a solid HATCH circle. A bar that would print under
  0.5 mm wide (0.9 mm across, end-on) at the sheet's scale is drawn at
  that least size, and the result's warnings say so.
- **Callouts**: a LEADER entity to a landing, an optional mark in a
  circle, the text wrapped by the code and set as TEXT lines.
- **Scale**: N of 1:N from 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500,
  1000. Left out, the code takes the smallest N at which the object with
  its dimensions and callouts fits the drawing area. A given scale that
  does not fit is refused with the one that does; so are notes or a table
  too big for the sheet. Nothing is written on a refusal.
- **Sheets**: `letter` (default, as a calc package's pages), `tabloid`,
  `a4`, `a3`, always landscape. The right-hand strip, 80 mm wide (100 or
  120 mm, by the code, only when the notes do not fit 80), holds the notes
  over the title block; the table and the scale bar sit at the bottom
  left; the stamp is boxed at the top left.

### Fonts: no system fonts, read-only home

ezdxf's drawing add-on takes fonts from the system's font folders and the
app's hosts may have none. `_fonts.ensure_fonts()` registers the DejaVu
Sans and DejaVu Sans Bold TTF files that matplotlib ships (a required
package) with ezdxf's font manager, in this process only, before anything
is measured or rendered; the sheet's two text styles name those files, and
our copy wins over a system copy of the same name, so one description draws
the same sheet on every host. Text widths (wrapping, table columns, "does
this number fit") are measured with the same font. Styles carry a width
factor of 0.85.

ezdxf loads or builds a font cache file under `~/.cache/ezdxf` when it is
first imported; it already carries on when it cannot write there. Tested in
a fresh process with ezdxf's font manager emptied and HOME set to a path
nothing can be written under: the sheet renders pixel-for-pixel the same.

A DXF opened in a CAD program that has no DejaVu Sans is shown in that
program's substitute font; the geometry and the dimensions are unaffected.

ezdxf moves each line of a pattern hatch by a random hair when it renders
(`ezdxf.render.hatching`), so two renders of a sheet with a concrete hatch
differ in a few hatch pixels. The DXF is the same.

### The reinforced wall section

Input: the `reinforced_wall_section/1` dict the wall module's structural
design returns (geometry in metres and degrees, materials, bar sets, notes,
basis). Neither module imports the other. `validate_wall_description`
refuses unknown keys by name and checks the geometry adds up.

Frame: mm, x from the toe's outer face toward the heel (retained soil on
the right), y up from the underside of the base.

Layout rules (fixed; the sheet's notes state the first three):

1. A stem bar that lies in the section runs from one cover below the top
   of the stem, parallel to its face, down into the base to stand on the
   base's whole bottom mat (its bars along the wall included). A base bar that lies in the section runs the
   full width of the base, one cover in from each end. No bar is cut off
   or lapped.
2. Bars along the wall are dots at their spacing, inside the in-section
   bars of the same face (their offset is the larger of their own cover
   and the in-section bars' cover plus diameter).
3. **Hooks.** A stem bar with `end.type = "standard_hook_90"` is turned at
   the bottom of the base toward the OPPOSITE face of the stem, across the
   joint: a back-face (retained-side) bar turns toward the toe. That is the
   "Correct Reinforcement Orientation" of USACE EM 1110-2-2104 (1 November
   2023) Figure 2-1, p. 9 (the bar at one face of the stem turns under the
   stem toward its other face); turned away from the stem is the figure's
   "Incorrect". No held reference prints a standard hook's bend diameter
   or extension, so the bend (a 2 mm radius on the sheet) and the tail (to
   the line of the stem's opposite face) are SCHEMATIC, and note 6 of the
   sheet says the hook is not detailed. If the description's `ldh_mm` is
   more than the depth from the top of the base to the outside of the hook
   as drawn, the result warns.
4. The retained ground is drawn from the top of the stem's back face at
   `backfill_slope_deg`; the ground in front at `embedment_depth` above the
   underside of the base.
5. Dimensions (mm): base thickness and stem height chained on the left,
   the wall height outside them; toe, stem thickness at the base and heel
   chained below, the base width outside them; the stem's top thickness
   above, with the stem's covers on the line under it; the base's covers
   at the toe's face; a shear key's distance, width and depth. Every
   number is placed where the code lays it out (`dimtix`, an explicit
   location); a chain link whose number is too wide for it gets a row of
   its own under the chain, the toe's set outward, so no number prints
   over another's (tested on GM-312's three walls).
   A shear key given by its depth only (GM-312 gives no width or
   position) is drawn under the stem, as wide as the stem's base, as the
   calc package's own wall figure draws it; only its depth is dimensioned,
   and a note and a warning say its width and position are assumed.
   At most 5,000 bars along the wall are laid out (a description asking
   for more is refused before layout); any drawing holds at most 20,000
   points.
6. One callout per bar set (its mark in a circle and its `callout` text).
   Front-face stem sets leave to the left, everything else to the right,
   in one column beyond the heel, ordered by height so no two leaders
   cross; texts are spaced so none overlap (tested on four walls).
7. The bar schedule (mark, size, spacing, place, steel required and
   provided), the description's notes, its basis, and three notes of the
   drawing's own.

The layout depends on the scale (callouts and dimension lines are placed in
sheet millimetres), so it is laid out at each standard scale in turn and
the first that fits is taken.
