# Mecklenburg ground-truth pairs (harvested 2026-09-04)

Real-drafting-practice ground truth for the drawing-intelligence
composition layer: the SAME standard detail as native CAD (DWG) and the
plotted PDF the fleet actually works with.

- **Source**: Mecklenburg County NC Stormwater Services, Land
  Development Standard Drawings (public municipal standards portal at
  stormwaterservices.mecknc.gov/Standard-Drawings, hosted on
  mecknc.widencollective.com). 179 DWG+PDF paired details available;
  10 diverse pairs harvested here (see MANIFEST.json for external IDs,
  direct URLs, sizes, hashes).
- **Truth files** (`*.truth.json`): per-sheet extraction of native
  LEADER / MULTILEADER / DIMENSION / TEXT / MTEXT / INSERT entities
  from the DXF via ezdxf — vertex chains, annotation text, defpoints,
  text insert/height/rotation, all in model-space coordinates (all ten
  sheets are drawn in model space at paper scale, units = inches;
  paper-space layouts are empty). MULTILEADER/MTEXT text retains MTEXT
  formatting codes (`\A1;{\H0.7x;...}`) — strip before comparing.
- **DXFs are NOT committed** (10–19 MB each). Regenerate from the DWGs
  with ODA File Converter (installed per-user at
  `%LOCALAPPDATA%\Programs\ODA\ODAFileConverter 27.1.0\`, silent-MSI
  install; signed, passes Windows Application Control — note the
  unsigned LibreDWG dwg2dxf.exe is BLOCKED by App Control on this
  machine):
  `ODAFileConverter.exe <in_dir> <out_dir> ACAD2018 DXF 0 1 *.DWG`

## Key sheets for scoring

| Sheet  | Native truth                                   |
|--------|------------------------------------------------|
| 21.01  | 13 LEADER + 5 DIMENSION (+7,987 lines — dense) |
| 3001   | 7 LEADER + 10 DIMENSION                        |
| 10.31A | 5 MULTILEADER + 1 DIMENSION + 40 CIRCLE (bubbles) |
| all    | 10–126 TEXT/MTEXT entities per sheet           |

## THE FLEET-REALITY FINDING (verified on all 10 PDFs)

Every plotted PDF here has **zero extractable text and zero embedded
fonts** — all lettering is SHX-stroked vector geometry (the only
embedded image is the county seal). Vector linework is rich
(784–10,095 paths/sheet), so geometry-side composition (leaders,
dimensions, clouds, bubbles) works on the vector layer, but
`find_text`-style queries on such sheets REQUIRE the raster/OCR leg
(render → OCR → map boxes back to IR coordinates) even though the PDF
is "vector". Agency plots with TrueType fonts would carry a text
layer; both realities must be supported. This promotes B7 from
"scanned sheets only" to a required leg of the primary path.

License note: public municipal standard drawings, used as test
fixtures only; not redistributed as a product feature.
