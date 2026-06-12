# geo_project — DESIGN

The canonical Project document + staged, human-gated MODEL-SETUP workflow
for 2D LE (slope_stability) and FEM (fem2d) models.

## The design inversion (why this exists)

Even frontier LLMs are unreliable at pulling exact geometry out of an image.
So this system **inverts the trust direction**: the agent NEVER claims to
have read geometry correctly. Instead:

1. Geometry enters as **numbers** (template parameters, DXF/PDF vector
   coordinates, typed-in points — or, worst case, a vision *draft*).
2. The machine renders those numbers back as an **echo-back cross-section**
   (PNG + a numbered vertex table) — `geo_project.render.echo_back`.
3. The **human confirms visually**. Numbers→image is easy for a person to
   verify at a glance; image→numbers is never to be trusted.

Vision-extracted geometry is quarantined: `geometry.provenance =
'vision_draft'` makes `validate()` emit the **blocking error GEOM007** until
the human has confirmed the geometry stage. Builders and `project_run`
refuse on blocking errors, so a vision draft physically cannot reach an
analysis without a human sign-off.

## Package map

| module       | role |
|--------------|------|
| `schema.py`  | versioned dataclasses → `Project`; JSON round-trip with unknown-key tolerance |
| `validate.py`| deterministic checks → `Issue{level, code, message, fix_hint}` list |
| `builders.py`| `to_slope_geometry`, `to_fem_kwargs`, `run_analyses` |
| `templates.py`| parametric geometry generators (`simple_slope`, `benched_slope`, `embankment_on_foundation`, `cut_with_berm`) |
| `ingest.py`  | `discover_dxf`/`from_dxf`, `from_pdf_vector`, `from_points`, `from_vision_draft` |
| `render.py`  | `echo_back(project, path)` → PNG + numbered vertex table |

Hard dependency: numpy only. matplotlib (render), ezdxf (DXF), PyMuPDF
(PDF) are optional and lazy-imported.

## Schema reference (schema_version = 1)

```
Project
├── meta            ProjectMeta{name, description, units='SI', schema_version}
├── geometry        Geometry{surface_points[(x,z)...],
│                            layer_boundaries{name: [(x,z)...]},
│                            provenance: user|dxf|pdf_vector|template|vision_draft}
├── stratigraphy    [Layer{name, material, top_elevation?, bottom_elevation?,
│                          bottom_boundary?(name into layer_boundaries)}]
│   └── material    Material{strength_model: mohr_coulomb|undrained|shansep|hoek_brown,
│                            gamma, gamma_sat?, phi?, c_prime?, cu?,
│                            shansep_S?, shansep_m?, ocr?, su_min,
│                            hb_sigci?, hb_gsi?, hb_mi?, hb_D,
│                            E?, nu?, psi (FEM),
│                            probabilistic{param: {cov(frac), dist, source}}}
├── water           Water{gwt_points?, ru, ponded(bool, declared intentional)}
├── loads           Loads{surcharges[Surcharge{q, x_start?, x_end?, label}], kh}
├── reinforcement   Reinforcement{nails[Nail...], anchors[Anchor...],
│                                 geosynthetics[GeosyntheticLayer...]}
├── analyses        [LEAnalysis{method, n_slices, search{surface_type, nx, ny,
│                       n_trials, n_points, seed}, probabilistic?{kind: fosm|
│                       monte_carlo, variables, n, seed}}
│                    | FEMAnalysis{nx, ny, depth?, x_extend?, srf_tol,
│                       element_type, srf_range}]   (dispatch on 'type')
├── confirmations   Confirmations{geometry, materials, water_loads}   ← gates
└── assumptions     [Assumption{field, value, source, note}]          ← ledger
```

Conventions: all SI (m, kPa, kN/m3, degrees, kN/m of run); x left→right,
z = elevation. Field names deliberately mirror `SlopeSoilLayer` /
`analyze_slope_srm` so the builders are near-mechanical.

**Layer elevations** chain: a layer's top is the surface crest (first layer)
or the layer above's bottom; its bottom is a flat `bottom_elevation` or a
named `bottom_boundary` polyline (representative elevation = the polyline's
max z, the dxf_import convention; the polyline itself becomes
`bottom_boundary_points` / `layer_polylines` downstream).

**Serialization**: `to_json()` / `Project.from_json()`. Documents carry
`schema_version`; unknown keys at ANY depth are tolerated (dropped) on load.

## Provenance + confirmation model

* `provenance` ∈ user | dxf | pdf_vector | template | **vision_draft**.
  Only vision_draft changes behavior (GEOM007 blocks until the geometry
  gate is set). The others are recorded for the audit trail.
* `confirmations` are the three staged human gates. They are set ONLY by the
  setup agent's `request_confirmation` tool (patching `confirmations.*` is
  refused), and **editing a stage's data clears that stage's gate**:
  `geometry*`/`stratigraphy[*]` (non-material) → geometry;
  `stratigraphy[*].material*` → materials; `water*`/`loads*`/
  `reinforcement*` → water_loads.
* `assumptions` is the explicit ledger: every default taken (template
  gammas, ingest section bottoms, nail structural defaults, proposed
  strengths, COVs) with its source — `cov_lookup` rows cite Duncan (2000) /
  ISSMGE TC304 / Phoon & Kulhawy. The confirmation payload always shows the
  ledger so the human approves the assumptions, not just the picture.

## Validation codes

GEOM001/002 surface shape · GEOM003 no layers · GEOM004 gap/overlap/zero
thickness · GEOM005 top layer below crest · GEOM006 shallow section (warn) ·
**GEOM007 vision draft unconfirmed (blocking)** · GEOM008 undetermined
elevations · GEOM009 short boundary (warn) · WATER001 ponded (info when
`water.ponded` declared, else warn) · WATER002 GWT below section ·
WATER003 ru range · LOAD001-004 surcharge/kh · REINF001 outside section ·
MAT001-008 material completeness FOR THE REQUESTED ANALYSES (FEM needs
E+nu; SHANSEP needs S/m/OCR-with-source; Hoek-Brown needs sigci/GSI/mi;
MAT007 FEM×shansep/hoek_brown unsupported) · UNIT001-005 unit-sanity
heuristics (gamma 10–25, phi<50, nu, strength magnitudes, E range) ·
ANAL001 unknown analysis/method.

`has_errors(issues)` is the run gate; warnings/infos surface at the
confirmation stages.

## The setup sub-agent (funhouse_agent/deep)

`funhouse_agent.deep.setup_agent.build_setup_subagent()` returns a
deepagents SubAgent spec (`name='model_setup'`) over
`setup_tools.make_setup_tools(store)`:

`project_new` (template | dxf+mapping | pdf | points | vision_draft) ·
`dxf_discover` · `project_show` · `project_validate` · `project_patch`
(targeted dot-path set/append/delete; resets gates) · `project_render`
(echo-back) · `cov_lookup` (cited COV rows) · `request_confirmation` (THE
gate) · `project_run` (refuses until all three gates + clean validation).

Attach with `build_deep_agent(..., enable_setup_agent=True,
setup_store=ProjectStore(), setup_render_dir=...)`. **OFF by default**:
the tool set carries mutable per-build state (the ProjectStore), so hosts
that only ask calculation questions never pay for the extra tool surface.

### Staged protocol (the system prompt enforces order)

1. **GEOMETRY** — create → render → confirm. DXF/PDF always goes through
   `dxf_discover` + ask-the-user layer mapping (never guessed).
2. **STRATIGRAPHY / MATERIALS** — strength model + parameters per layer via
   `project_patch`; proposed values are labeled assumptions with cov_lookup
   citations → confirm (form_schema lists every assumed parameter).
3. **WATER / LOADS / REINFORCEMENT** — re-render so the human SEES the GWT
   and reinforcement → confirm (an explicitly-empty stage still confirms).
4. **ANALYSIS PLAN** — append analyses → `request_confirmation
   (stage='analysis_plan')` (review-only; logged to the ledger, no gate).
5. **RUN** — `project_run`; its refusal on missing gates is correct
   behavior, not an error.

### Confirmation mechanics (both offline-tested)

`request_confirmation(stage, summary_markdown, form_schema?, user_response?)`
builds the payload
`{type: 'model_setup_confirmation', stage, summary_markdown, image_path,
vertex_table, form_schema: [{name(patch path), label, type: number|choice|
bool|text, options?, default?, unit?}], assumptions, resume_contract}`.

* **Durable path** (running under LangGraph WITH a checkpointer — detected
  via the `__pregel_checkpointer` configurable): the tool calls
  `langgraph.types.interrupt(payload)`; the run pauses and the payload
  surfaces under `__interrupt__`. The host collects the human's answer and
  resumes with `Command(resume={"approved": bool, "edits": {path: value}})`.
  On resume the edits are applied as patches FIRST (approve-with-edits =
  correct then confirm), then the stage gate is set.
* **Chat fallback** (no checkpointer / plain tool call): the tool returns
  `status='needs_user'` with the payload rendered as numbered chat
  questions; the agent relays them verbatim and calls the tool again with
  `user_response={approved, edits}` after the user replies.

### Notebook forms (designed, not shipped)

The owner deprioritized form polish vs. the staged flow, and this venv has
no ipywidgets — so the chat fallback is the shipped UX. The designed
renderer (for `funhouse_agent/deep/notebook.py` when picked up): map
`form_schema` items → widgets (number → `BoundedFloatText` with the unit as
description suffix, choice → `Dropdown(options)`, bool → `Checkbox`,
text → `Text`), plus Approve / Request-changes buttons; the handler diffs
widget values against defaults to build `edits` and resumes the thread with
`Command(resume={"approved": ..., "edits": ...})` via
`DeepNotebookChat`'s thread config. The interrupt payload was shaped so
this renderer needs no agent-side changes.

## Demo

`geo_project/demo_model_setup.py` replays the scripted staged flow and
writes `docs/examples/model_setup_walkthrough.md` + the echo-back PNG —
regenerate with:

```
python -m geo_project.demo_model_setup
```
