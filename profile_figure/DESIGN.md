# profile_figure — design notes

Generic, parameter-driven subsurface profile schematic (matplotlib → PNG).
Built 2026-09-04 from field feedback: asked for "a figure showing the pile and
a subsurface profile", the agent had **no drawing tool at all** and delivered a
PDF with an `[image]` placeholder plus an HTML table with coloured rows
presented as a "figure"
(`module_work/field_feedback/2026-09-04_praia-downdrag_v5.11.2/FINDINGS.md`,
item 3). This module is the capability that was missing.

## Scope

It draws whatever it is handed. It knows **no analysis** — no consolidation, no
capacity, no pore pressure — so every module family (downdrag, settlement,
axial pile, bearing, walls, liquefaction, …) can call it with the profile it
already has. Anything computed must be passed in, either as a layer
`description` or as an `annotation`.

**General pool, not a specialist scope.** The agent tool is deliberately NOT a
member of `PAVEMENT_MODULES` / `STRUCTURAL_MODULES` or any reviewer scope in
`funhouse_agent/dispatch.py`: subsurface stratigraphy is a geotech capability,
and those scopes are minimal by doctrine (see their comments). The unscoped
agent — the one the prompt nudge is written for — sees every analysis module,
so it reaches this tool without a scope entry. `worked_examples` is the same
shape: named in the shared prompt, in no specialist scope. Add it to a scope
only if that specialist's owner decides it belongs there, and bump the pinned
count in `test_reviewers.py` with it (2026-09-05 decision).

## Conventions

- **SI, elevations internally.** Every input may be given as an elevation
  (`*_elevation`) or as a depth below ground (`*_depth`); depths are converted
  with `elevation = ground - depth` at resolve time. `axis` controls only how
  the vertical axis is *labelled*.
- **`axis="auto"`** picks `depth` when the caller expressed the profile purely
  in thicknesses/depths, and `elevation` as soon as any elevation appears. The
  drawn geometry is identical either way.
- **Vertical scale is true; horizontal is schematic.** A 1-D profile has no
  width, so bands span a unit-width axis and a pile/footing is drawn at an
  exaggerated width (`_FOUNDATION_WIDTH`) with its real dimension in the label.
  The figure footer says so — do not let a reader scale a width off it.
- **Layers must stack continuously.** A gap or overlap raises a `ValueError`
  naming the layer and the size of the gap. A silently-drawn gap would be a
  misleading figure, and an LLM assembling a profile from a boring log gets
  this wrong in exactly that way.
- **Never colour alone.** Each band carries its name as text plus a distinct
  hatch from the house palette (`slope_stability.plotting`'s earth tones), so
  the figure survives greyscale printing and colour-vision deficiency.
- **Axis ticks sit at the layer interfaces**, not on a regular grid: the
  interface elevations are the numbers an engineer reads off a profile. Ticks
  closer than ~3 % of the plot height are thinned so labels never overprint.

## Structure

| File | Role |
|------|------|
| `geometry.py` | Resolution + validation. Pure Python, no matplotlib. |
| `plotting.py` | Drawing; house palette; `build_profile_figure` → `(fig, ax)`. |
| `profile.py` | `render_profile_figure` — resolve, draw, PNG (file + base64). |
| `results.py` | `ProfileFigureResult` (resolved geometry + image + `summary()`). |

`build_profile_figure` is exported so a module can drop the section into a
bigger figure or a calc package; `render_profile_figure` closes its own figure
(long-running hosts must not accumulate open figures).

## Embedding in reports

`render_profile_figure` returns **both** a saved PNG (`output_path`) and the
same image as base64 (`image_base64` / `data_uri()` / `img_tag()`), because the
two consumers differ:

- the web app renders a saved `.png` artifact inline in chat automatically
  (`webapp/core.py` `_ARTIFACT_KIND_BY_EXT`), so the agent should save it;
- `calc_package.html_to_pdf` needs an embeddable image. It now inlines a real
  local PNG/JPEG path for the caller, so an agent can write
  `<img src="/abs/path/profile.png">` and never handle base64 itself; a data
  URI still works for callers that already have one.

## Deliberate non-goals

- **Not a boring log.** No sample intervals, blow counts by depth, or lab
  results columns. `subsurface_characterization` owns real log data and its
  Plotly depth profiles.
- **Not a cross-section.** Horizontally varying stratigraphy, slope geometry
  and slip surfaces belong to `slope_stability.plotting` / `dxf_export`, which
  draw true 2-D sections. This is the 1-D column at a boring/pile location.
- **No unit conversion.** SI in, SI out, like every other analysis module
  (`pavement_design` is the documented US-customary exception).
