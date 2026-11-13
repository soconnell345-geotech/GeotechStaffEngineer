# Cloud helper brief: drawings and figures (BACKLOG GM-312 to GM-315)

Read `COMMON_MODULES.md` (beside this file) first: its rules bind every task here. This file adds what is common to this work, then has one section per cloud session. Your routine's prompt names your section; do that section only.

The owner's ask (2026-11-12): "if I wanted to draw a concrete cantilever retaining wall with reinforcing and dimensions and call outs at the end of a calc package, could it do it?" Today it cannot. The wall figure is a to-scale schematic with no bars; the wall module stops at sliding, overturning and bearing; a calc package embeds PNG images only. The owner also asked whether figures got worse where model-written plot code is switched off; nobody has measured it.

Sections: GM-312 (the wall's structural design), GM-313 (the drawing layer and the first drawing), GM-315 (the fixed plotting tool's new figure kinds). These three run at the same time. Two more are written here and are NOT started yet: the join of GM-312 and GM-313 (after both are merged), and GM-314 (the figure test set, which starts when GM-328's sample-package script is on master).

## Common to every task here

- **Line numbers** below are from master 50ef609 and may have moved; find the code by what it does.
- **Calls.** Where a held reference prints nothing, or two printed forms differ, do not stop for the owner: follow the calls this file already makes; for a new one, choose, state both forms in the tool description and the result, and list it in CHECK.md under "Calls made" (the owner's rule of 2026-11-10). A value or formula from memory is never a call you may make: stop short and say so.
- **Look at what you draw.** Every figure, drawing or sheet you produce is rendered to PNG and opened with the Read tool, and you fix what you see, before you call it done. Say in CHECK.md what you looked at and what you changed after looking.
- **Another session's branch is in flight:** `claude/gm-316-317-warnings-allowed-values` changes a few lines of `funhouse_agent/adapters/calc_package.py`, `adapters/dxf_export.py`, `adapters/retaining_walls.py` and the core's `charts/plot_tool.py`. Before you start: `git fetch origin claude/gm-316-317-warnings-allowed-values && git diff origin/master...FETCH_HEAD -- <the files you share with it>`. Keep your edits away from the lines it changes, so the two merge cleanly.
- **Do not edit:** `BACKLOG.md`, `HANDOFF.md`, `CLAUDE.md`, `slope_stability/`, `geotech-references/` (write library defects in CHECK.md), `packages/aec-document-review/` (except its recorded fingerprint file, GM-315 only), any `pyproject.toml` version or pin, and the files this brief gives to another section.
- **No new dependency.** ezdxf 1.4 or newer, matplotlib 3.8 or newer and PyMuPDF 1.27 are already required by the app.
- **Commit trailer.** End each commit message with the Co-Authored-By line of the model you are running as (for example `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`), not the one COMMON_MODULES prints.
- **Setup.** COMMON_MODULES' command installs planlens and geotech-references from PyPI at the versions master needs (planlens 0.15, geotech-references 1.6.0), whatever versions its text names.

### The reinforced wall section: the description GM-312 writes and GM-313 draws

This is the contract between the two sessions. It is a plain dict (JSON types only), so neither module imports the other. GM-312 returns it; GM-313 validates and draws it. Do not rename, drop or add a key without saying so at the top of CHECK.md under "Contract changes": the lead reconciles the two branches.

| Key | Type | Meaning |
|---|---|---|
| `schema` | str | `"reinforced_wall_section/1"` |
| `title` | str | One line for the title block, for example "Cantilever retaining wall: reinforcement" |
| `geometry` | dict | The wall as the module resolved it, metres and degrees: `wall_height` (underside of the base to the top of the stem), `base_width`, `toe_length`, `heel_length`, `stem_thickness_top`, `stem_thickness_base`, `base_thickness`, `stem_batter` (`"back"` or `"front"`), `embedment_depth`, `backfill_slope_deg`, `has_shear_key` (bool), `key_depth`, `key_width`, `key_distance_from_toe` (the last three `null` without a key) |
| `materials` | dict | `fc_MPa`, `fy_MPa` |
| `bars` | list of dict | One entry per bar set (below) |
| `notes` | list of str | Lines for the sheet's notes block: concrete strength and steel grade in both unit systems, covers, the design basis, and what was NOT designed (lap and straight development lengths, bar cut-offs, the shear key's steel) |
| `basis` | str | One line naming the manual and edition the design follows |

Each entry of `bars`:

| Key | Type | Meaning |
|---|---|---|
| `mark` | str | `"A"`, `"B"`, ...: unique, the letter the callout and the schedule carry |
| `element` | str | `"stem"`, `"toe"`, `"heel"` or `"base"` (a set that serves the whole base slab) |
| `face` | str | For the stem `"back"` (the retained side) or `"front"`; for the base `"top"` or `"bottom"`; `"both"` for a set placed in each face |
| `runs` | str | `"in_section"`: the bar lies in the plane of the section and is drawn as a line (vertical stem bars, bars across the base). `"along_wall"`: the bar runs along the wall, is seen end-on and is drawn as dots at its spacing |
| `role` | str | `"flexure"` or `"temperature_shrinkage"` |
| `size` | str | US designation as the manual uses it, `"No. 4"` to `"No. 11"` |
| `diameter_mm` | float | Nominal diameter |
| `spacing_mm` | float | Centre to centre |
| `cover_mm` | float | Clear cover from the named face to this bar set's outermost bar |
| `end` | dict or null | `null` = straight. Else `{"type": "standard_hook_90", "into": "base", "ldh_mm": 310.0}`: a bar that turns into the named element with a standard hook, and the hook development length the design computed |
| `callout` | str | The text the drawing prints beside the mark, in both unit systems, for example `"No. 6 @ 300 mm (12 in.), back face"` |
| `As_required_mm2_per_m`, `As_provided_mm2_per_m` | float or null | For the bar schedule |

Drawing rules that are GM-313's, not the description's: an `in_section` stem bar runs the full height of the stem and into the base; an `in_section` base bar runs the full width of the base; `along_wall` bars sit inside the `in_section` bars of the same face. No bar is cut off or lapped in version 1, and the sheet says so.

A made-up example (not a design; use it as a test fixture):

```json
{
  "schema": "reinforced_wall_section/1",
  "title": "Cantilever retaining wall: reinforcement (made-up example)",
  "geometry": {"wall_height": 4.5, "base_width": 2.7, "toe_length": 0.45, "heel_length": 1.8,
               "stem_thickness_top": 0.3, "stem_thickness_base": 0.45, "base_thickness": 0.45,
               "stem_batter": "back", "embedment_depth": 0.9, "backfill_slope_deg": 0.0,
               "has_shear_key": false, "key_depth": 0.0, "key_width": null, "key_distance_from_toe": null},
  "materials": {"fc_MPa": 28.0, "fy_MPa": 414.0},
  "bars": [
    {"mark": "A", "element": "stem", "face": "back", "runs": "in_section", "role": "flexure",
     "size": "No. 6", "diameter_mm": 19.1, "spacing_mm": 300.0, "cover_mm": 75.0,
     "end": {"type": "standard_hook_90", "into": "base", "ldh_mm": 310.0},
     "callout": "No. 6 @ 300 mm (12 in.), back face",
     "As_required_mm2_per_m": 880.0, "As_provided_mm2_per_m": 947.0},
    {"mark": "B", "element": "stem", "face": "front", "runs": "in_section", "role": "temperature_shrinkage",
     "size": "No. 4", "diameter_mm": 12.7, "spacing_mm": 300.0, "cover_mm": 75.0, "end": null,
     "callout": "No. 4 @ 300 mm (12 in.), front face",
     "As_required_mm2_per_m": 400.0, "As_provided_mm2_per_m": 430.0},
    {"mark": "C", "element": "stem", "face": "both", "runs": "along_wall", "role": "temperature_shrinkage",
     "size": "No. 4", "diameter_mm": 12.7, "spacing_mm": 300.0, "cover_mm": 75.0, "end": null,
     "callout": "No. 4 @ 300 mm (12 in.), each face, along the wall",
     "As_required_mm2_per_m": 400.0, "As_provided_mm2_per_m": 430.0},
    {"mark": "D", "element": "toe", "face": "bottom", "runs": "in_section", "role": "flexure",
     "size": "No. 5", "diameter_mm": 15.9, "spacing_mm": 300.0, "cover_mm": 100.0, "end": null,
     "callout": "No. 5 @ 300 mm (12 in.), bottom",
     "As_required_mm2_per_m": 610.0, "As_provided_mm2_per_m": 663.0},
    {"mark": "E", "element": "heel", "face": "top", "runs": "in_section", "role": "flexure",
     "size": "No. 6", "diameter_mm": 19.1, "spacing_mm": 300.0, "cover_mm": 75.0, "end": null,
     "callout": "No. 6 @ 300 mm (12 in.), top",
     "As_required_mm2_per_m": 900.0, "As_provided_mm2_per_m": 947.0},
    {"mark": "F", "element": "base", "face": "both", "runs": "along_wall", "role": "temperature_shrinkage",
     "size": "No. 4", "diameter_mm": 12.7, "spacing_mm": 300.0, "cover_mm": 75.0, "end": null,
     "callout": "No. 4 @ 300 mm (12 in.), top and bottom, along the wall",
     "As_required_mm2_per_m": 400.0, "As_provided_mm2_per_m": 430.0}
  ],
  "notes": ["Concrete f'c = 28 MPa (4,060 psi). Reinforcement fy = 414 MPa (60,000 psi).",
            "Lap and straight development lengths and bar cut-offs are not designed."],
  "basis": "USACE EM 1110-2-2104 (1 November 2023), strength design with the serviceability single load factors"
}
```

## GM-312: the cantilever wall's structural design

Branch `claude/gm-312-wall-structural-design`. CHECK at `module_work/gm312/CHECK.md`. Module: `retaining_walls`, its adapter and its calc package.

**What to build.** The design of the wall's concrete: stem, toe and heel, each with its factored moment and shear at its critical section, the steel required, a bar size and spacing, the checks the manual prints, and the cover. On USACE EM 1110-2-2104, "Strength Design for Reinforced Concrete Hydraulic Structures", the edition the library holds (cover: 1 November 2023; it supersedes the 2016 edition and has no hydraulic factor). ACI 318 is not held.

**Read the print itself.** `git submodule update --init geotech-references`, then `geotech-references/docs/EM_1110-2-2104_2025.pdf` with PyMuPDF: printed page p is `doc[p + 5]`. The library's `text/chapter0N.json` for this manual is paraphrased and stops at chapter 5, so do not quote from it. The library's functions are `geotech_references.em_2104` (`design`, `flexure_axial`, `shear`, `loads`, `serviceability`, `reinforcement`; US customary units; each returns a dict with its printed page). Call them by direct import, as `pavement_design` calls `geotech_references.aashto_1993` (`pavement_design/DESIGN.md:19-21`: the reference package is a shared library, outside the rule against cross-module imports). Five of the library's page strings are wrong (`loads.table_3_2_*` say pp. 23-24, the table is on pp. 24-25; `flexure_axial.balanced_strain_kb` says p. 29, Eq. 4-3 is on p. 30): cite every page from the PDF yourself, and list library defects you meet in CHECK.md.

**What the print has** (found by a read of the PDF on 2026-11-12; confirm each on the page before you use it, and where this list and the page differ the page wins and CHECK.md says so):

| Need | Where printed |
|---|---|
| Load cases for an earth retaining wall, serviceability and strength columns | Table E-2, p. 93, with its notes (pp. 92-94); permanent load factors para 3-3d, p. 22; Table 3-2, pp. 24-25 |
| Usual and unusual loads are designed by the serviceability requirements; extreme loads by strength | para 3-1e, p. 13 |
| Service steel stress limits; the single load factors that stand in for them (2.2 usual, 1.6 unusual; fy = 60,000 psi only; beams and one-way slabs) | para 3-4a, Tables 3-3 and 3-4, p. 26 |
| Forces at critical sections from factored loads; with a single load factor, the unfactored results times the factor | para 3-1f, pp. 13-14 |
| Steel ratio not above 0.50 of balanced (must); 0.25 of balanced recommended | para 3-6 and 3-4b(2), p. 27; balanced ratio as printed on pp. 53 and 75 |
| Minimum wall thickness | para 3-7, p. 27 |
| Steel for a moment, singly reinforced | Eq. D-8 and D-9, p. 72 (`design.design_singly_reinforced`); capacity Eq. B-49, B-50, p. 51 |
| Effective depth | para 2-3, p. 4; the examples, pp. 75 and 84 |
| Shear strength of walls and slabs without shear steel | Eq. 5-1 and para 5-2, p. 35 (`shear.shear_capacity_one_way_slab`) |
| Critical sections for SHEAR: the base of the stem; the toe at d from the front face of the stem; the heel at the back face of the stem | para 5-3, pp. 35-36; Fig. 5-1, p. 36 |
| Cover for sections thicker than 12 in. | Table 2-1, p. 5 |
| Bar spacing: 12 in. (30 cm) centre to centre at most; clear distance | para 2-6, pp. 4-5 |
| Temperature and shrinkage steel; primary steel never less than it | para 2-9 and Table 2-3, pp. 6-8 |
| Splices: none above No. 11 | para 2-8, p. 6 |
| The stem-to-base joint; stem steel developed into the base | para 2-10a and Fig. 2-1, p. 9; para 5-3, p. 36 |
| Worked examples | D-3, pp. 73-76 (a wall stem from a given moment); D-5, pp. 79-87 (a floodwall stem: loads, moment, shear, steel, hook) |
| The base pressure for structural design is the trapezoid or triangle, not the uniform one | FHWA NHI-06-089 Fig. 10-17, p. 10-39; FHWA GEC-6 Sec. 6.6, pp. 121-122 |
| A surcharge within the base width counts for the stem's design | FHWA NHI-06-089 Sec. 10.5.3, p. 10-41 |
| Bar diameters and areas, No. 3 to No. 18 | FHWA GEC-10 (FHWA-NHI-18-024) Table 6-2, p. 6-3 |
| Unit conversion factors | EM 1110-2-2104 Glossary Section III, p. 129 |

**Calls already made** (the lead's, for the owner to revisit; build them as written and repeat each in `choices` with its basis):

1. **Strength reduction factors.** The manual takes them from ACI 318-19 (p. 21) and prints the values only in its examples: "For shear; phi = 0.75 (ACI 318-19). For bending; phi = 0.90 (ACI 318-19)" (D-5, p. 84). Use 0.90 and 0.75, citing p. 84.
2. **beta1.** The manual takes it from ACI 318-19 (p. 30) and prints only 0.85 at 4,000 psi (pp. 52, 75) and 0.80 at 5,000 psi (p. 86). Default: the library's `flexure_axial.aci_beta1`, named in `choices` as ACI 318-19's value, not printed in the manual as a rule; an input `beta1` overrides it. Pin the two printed values. beta1 enters only the balanced steel ratio.
3. **Serviceability.** Default: the single load factors of Table 3-4, which need no modulus. They are printed for fy = 60,000 psi (414 MPa) only: refuse another fy on this route. The service-stress route (Table 3-3) needs a modular ratio, for which the manual prints no rule (its examples use Es = 29,000 ksi and Ec = 3,605 ksi at 4,000 psi): offer it only when the engineer gives `modular_ratio`.
4. **Eq. 5-1.** Its two printed forms differ by about 2 % (2 sqrt(f'c) in psi is 0.166 sqrt(f'c) in MPa; the print says 0.17). Use the US form on converted values, as the manual's examples do, and state the SI form's result beside it.
5. **Toe and heel moment.** No held reference prints where the toe and the heel are cut for moment, or their load diagrams; EM 1110-2-2104 p. 87 sends the reader to EM 1110-2-2502, which is not held. Take the moment at the face of the stem (front face for the toe, back face for the heel), by statics from the free body: the base pressure up, the slab's own weight and the soil, surcharge and water above it down. Say in `choices` and in the calc package that the section is not printed in a held reference.
6. **Development.** The manual sends development, anchorage and laps to ACI 318-19 (para 2-4, p. 4) and prints one formula, the standard hook's development length ldh, in D-5 Step 6 (pp. 86-87). Use that formula for the stem's main bars hooked into the base, and check ldh against the depth of base available. Straight development lengths, lap lengths and bar cut-offs are NOT computed: say so in the result (`not_designed`, with the reason) and in the description's `notes`. An optional input may carry a lap length the engineer supplies; it is printed, never checked.
7. **Crack control.** Para 2-6b, p. 5: for usual loads the 12 in. spacing meets it when the chapter 3 serviceability is met; for unusual loads the manual modifies ACI's spacing formula, which it prints only in its examples (p. 76). Use the formula as printed there, with para 2-6b's modifications, citing both pages.
8. **Cover.** Table 2-1 gives the default where it prints one. For a section 12 in. (30 cm) or thinner the table says "According to ACI 318-19": require the cover from the engineer, with a message that names the 2 in. the manual's own example uses (D-3, p. 75).
9. **Minimum flexural steel.** The manual prints only "not less than that required for shrinkage and temperature" (para 2-9g, p. 8). Apply that and say that no other minimum is printed.
10. **Bars.** US sizes No. 4 to No. 11. The choice follows the manual's examples: the smallest bar that gives the steel at 12 in. spacing; if No. 11 does not, 6 in. (D-5). The engineer may give the size, the spacing or both.

**Stop short of:** earthquake cases (Table E-2 3A, 3B) and the construction case; a shear key's steel; bar cut-offs up the stem; counterforts; anything else the print does not give. Name each in CHECK.md and in the result's `not_designed`.

**Where it lives.**
- New files in `retaining_walls/`: the design (suggested `structural.py`) and its result dataclass with `summary()` and `to_dict()`. The design runs `analyze_cantilever_wall` with the same inputs first and keeps its result, so the stability checks and the concrete are one wall. Reuse the module's own pressure functions (`earth_pressure.active_force_components` and `_water` take any height; `cantilever._active_thrust_terms`, `_bearing_terms` and `_vertical_force_items` hold Ka, the base pressures and the weights). Other sessions are fixing functions in `cantilever.py`, `calc_steps.py` and `results.py`: ADD to those files where you must, and do not rewrite or reorder what is there.
- Inputs: the wall tool's inputs, plus concrete strength, steel yield, covers, joint spacing (for Table 2-3), bar size and spacing where the engineer fixes them, `beta1`, `modular_ratio`, the largest aggregate where a rule needs it. A default only where a page prints one; otherwise the input is required and the message says why.
- Units: SI in and out (m, kN, kPa, MPa, mm). The manual is US customary: every structural quantity is returned in both (`Mu_kNm_per_m` and `Mu_kipft_per_ft`, and so on), and the calc package shows both. Missing converters (MPa and psi, kN-m/m and kip-ft/ft, mm2/m and sq in/ft) go in `geotech_common/units.py`, citing p. 129.
- The result carries, for each of stem, toe and heel: the section's place, the governing load case, Mu and Vu, d, the steel required and provided, the bar set, phi Mn, phi Vc, each check with its page, and pass or fail; then `reinforcement`, the description defined above (exact keys); `choices`; `not_designed`; a verdict that names failed checks first, as the wall result does.
- Agent: a new method in `funhouse_agent/adapters/retaining_walls.py` (suggested `cantilever_wall_structural_design`), with parameter descriptions that say what the code does and cite the pages; a probe in `method_probes.json`; the manual row.
- Calc package: `retaining_wall_package` gains a fourth `wall_type`, `"cantilever_reinforced"`, through the existing selector (`_PACKAGE_TOOLS`, `funhouse_agent/adapters/calc_package.py:603-655`): the stability sections as today, then the structural sections (loads and cases, stem, toe, heel, steel and checks, each step with its equation, the numbers put in, and the page). No drawing: that is GM-313 and the join.

**Pin against the print.**
- D-3 (pp. 73-76) through your section-design function: the given 5 k-ft, Mu = 11 k-ft, d = 6 in., As = 0.43 sq in, No. 6 at 12 in., the 0.25 rho_b check (0.0061 against 0.0071). If you build the service-stress route, also fs = 24.96 ksi and the 19 in. spacing.
- D-5 Step 3 (pp. 84-85): Vu = 36.9 kips/ft against phi Vc = 46.5 kips/ft; As = 2.73 sq in/ft, No. 11 at 6 in.; Step 6: ldh = 24.1 in. D-5 has slips of its own (Step 3(g) uses the 4,000 psi coefficient for 5,000 psi concrete; a Ku printed two ways; x printed 9.96 then used as 10.09): do not pin those lines, and list them.
- D-3 gives no wall height or soil, so it cannot check a stem moment worked from earth pressure. If D-5's Tables D-2 to D-4 (pp. 80-84) print enough to rebuild its stem loads, reproduce Mu = 428 k-ft/ft and Vu = 36.9 kips/ft from them; if not, say what is missing.
- One made-up wall worked by hand in CHECK.md, step by step, beside the module's numbers: the stem's moment and shear from the pressure diagram, the toe's and heel's from the base pressure and the weights, and the steel.

**Tests** (exit codes): `retaining_walls/tests` (new files for the design), `calc_package/tests` for retaining walls, `funhouse_agent/tests/test_defaults_used.py -k "retaining or wall"`, `test_calc_package_tool_parity.py`, `test_adapter_ergonomics.py`, `test_calc_package_adapter.py`, `test_calc_package_figure_layout.py`, `webapp/tests/test_page_fingerprints.py` (a method added inside an adapter should not move it; if it moves, find out why before re-recording), `webapp/tests/test_public_text.py`.

**Leave alone:** `bearing_capacity/concrete_design.py` (an ACI 318-19 footing design no tool reaches; the owner decides its fate), `dxf_export/`, `calc_package/` outside what a new wall_type needs, the core.

## GM-313: the drawing layer, and the first drawing

Branch `claude/gm-313-drawing-layer`. CHECK at `module_work/gm313/CHECK.md`. Modules: `dxf_export`, `calc_package`, the `dxf_export` adapter.

**What to build.** Fixed code that takes a typed description of a drawing and writes (1) a DXF with real DIMENSION entities, layers, hatches and leaders, (2) the same drawing as a one-page vector PDF sheet, (3) the same drawing as a PNG. No model-written code anywhere, so it works where plot code is off. First drawing: the reinforced cantilever wall section, from the description defined in "Common" above (use the made-up example as your fixture; GM-312 produces real ones in parallel).

**One drawing, three files.** Build ONE ezdxf document and render that same document to the PDF page and the PNG with ezdxf's own drawing add-on (`ezdxf.addons.drawing`: `Frontend`, `RenderContext`, `matplotlib.MatplotlibBackend`, `pymupdf.PyMuPdfBackend`), so the sheet in the calc package is the DXF the engineer opens, not a second drawing that can drift from it. A trial on 2026-11-12 with ezdxf 1.4.4 showed this route works: linear DIMENSION entities, an AR-CONC hatch, LEADER, MULTILEADER, TEXT and MTEXT all rendered to PNG and to a vector PDF page (lettering comes out as outlines, so it is not selectable text; accepted). Three things that trial found, for you to settle:
- the stock `EZDXF` dimension style multiplies the measurement by 100: define your own dimension and text styles;
- ezdxf took its font from the system (Arial on that machine). The app's hosts may have no system fonts. Make the rendering work with none (for example by pointing ezdxf's font lookup at the DejaVu Sans that matplotlib ships), test it with the system fonts hidden, and say in CHECK.md how;
- ezdxf builds a font cache on first use: make sure a read-only home folder does not break a render.
If the add-on cannot be made to work without system fonts, fall back to painting the same typed description with matplotlib, and say so at the top of CHECK.md.

**The typed description (the layer's input).** Dataclasses with validation and a JSON form, in `dxf_export/` (suggested files: `drawing.py` for the types, `sheet.py` for the DXF, `render.py` for PDF and PNG, `wall_section.py` for the first drawing). It holds what the owner listed and no more:
- outlines (closed polygons on a named layer, with an optional hatch from a fixed list of patterns: concrete, earth, gravel, none);
- lines and polylines;
- bars: a path with the bar's diameter (drawn as a heavy line with bend radii) or dots for bars seen end-on;
- dimensions: two points and an offset, horizontal, vertical or aligned. The number is the DIMENSION entity's own measurement: a description cannot type a different number over it. A unit suffix and a second unit in brackets are allowed;
- callouts: a leader from a point on the drawing to a text;
- notes, a small table (the bar schedule), and the title block's fields;
- the sheet: size (`letter` by default, to match the calc package's pages; `tabloid`, `a4`, `a3`), landscape, the scale (chosen by the code from a short list of standard scales so the drawing fits, or given), drawing units mm.
Fixed by the code and not open to the description: layer names, colours and line weights; text heights (lettering 2.5 mm or more on the sheet); the border and title block; a scale bar; and the stamp "CALCULATION SKETCH - NOT FOR CONSTRUCTION" on every sheet. Unknown keys are refused by name. Geometry is drawn in model space at true size; keep it simple enough to open cleanly in a CAD program (run `doc.audit()` in a test and expect no errors).

**The first drawing: the reinforced wall section** (`wall_section_drawing(description) -> Drawing`, fixed layout rules): the concrete outline to scale with a concrete hatch; the ground lines in front and behind; every bar set of the description (rules in "Common"); the stem's hooked bars turned as EM 1110-2-2104 Fig. 2-1 (p. 9, in `geotech-references/docs/EM_1110-2-2104_2025.pdf`; printed page p is PDF page index p + 5) draws them, and say which way and why; dimensions for the wall height, stem height, base width, toe, heel, stem thickness top and bottom, base thickness and each cover; one callout per bar set (mark and `callout` text), placed so that no two overlap and none crosses the concrete where it can go outside; a bar schedule (mark, size, spacing, place, steel required and provided); the notes; the title block. Draw the conventions the owner already set for figures where they apply (`profile_figure/DESIGN.md:236-243`; no unnecessary ink). Check the layout on at least three made-up walls of different proportions (a low wall, a tall wall with a long heel, a front-battered stem with a shear key), look at each PNG, and commit the example's `.dxf`, `.pdf` and `.png` under `module_work/gm313/` for the lead and the owner to open.

**Into the calc package.** `calc_package` today has one page size, PNG figures only, and no page merge (`data_model.py:107 FigureData`; `latex_renderer.py:1732 render_pdf`; the PyMuPDF Story leg re-encodes every PNG to JPEG at 1,100 px, which would ruin a drawing). Add a `SheetData` (title, caption, the PDF page's bytes, the PNG's base64) to the data model and `sheets=` to `generate_calc_package`: in the PDF, each sheet is appended as its own vector page after the last page, on either PDF leg (PyMuPDF `insert_pdf`; a package with no sheet must come out exactly as today); in HTML, a last section "Drawings" shows the PNG; in LaTeX source, the PNG. `calc_package` stays pure data and rendering: it does not import `dxf_export`.

**To the agent and the chat.** Two new methods in `funhouse_agent/adapters/dxf_export.py`: `draw_wall_section` (a `reinforced_wall_section` description plus sheet options) and `draw_sheet` (a general typed description). Each writes `<name>.dxf`, `<name>.pdf` and `<name>.png` into the conversation folder (any new file there gets a download card; `.dxf` already has one; the PNG shows inline) and returns `output_path` (the DXF), `pdf_path`, `png_path`, the sheet size and scale used, counts and warnings. What the model may ask for is exactly the typed description: data, never code; no file paths beyond a name; no way to remove the stamp. Parameter descriptions say what the code does. Add probes to `method_probes.json` and the manual rows. If the module's one-line brief in `MODULE_REGISTRY` must change to stay true, the page fingerprint moves: re-record in a commit of its own, and keep the catalog under its size caps (`funhouse_agent/tests/test_catalog_routing_hints.py`).

**Not yours:** `retaining_walls/`, `funhouse_agent/adapters/calc_package.py` and `adapters/retaining_walls.py` (GM-312's, and the join's); the core; `webapp/` (one card per file is accepted for now).

**Tests** (exit codes): `dxf_export/tests` (new files: the description's validation; the DXF read back with ezdxf and checked for DIMENSION, HATCH, leader and layer counts and for a clean audit; the wall drawing on the fixture; a render with the system fonts hidden), `calc_package/tests` (a package with a sheet has one more page of the sheet's size, and the same package without one is unchanged), `funhouse_agent/tests/test_defaults_used.py -k dxf`, `test_writer_output_paths.py`, `test_adapter_ergonomics.py`, `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_public_text.py`.

## GM-315: the fixed plotting tool draws what only plot code draws today

Branch `claude/gm-315-plot-data-panels`. CHECK at `module_work/gm315/CHECK.md`. Code: the shared core, `packages/aec-agent-core/aec_agent_core/` (read `COMMON_DOCREVIEW.md` beside this file for how the core and Document Review are installed and tested; its other rules are for another team's tasks).

**Why.** `plot_data` draws a figure from a JSON description with no model-written code. Its own description says: "It cannot draw a structure to scale with diagrams beside it, or several panels sharing one axis" (`deep/vision_kit.py:273-276`). Those figures, and contours, come only from model-written plot code today, which is off for Document Review on the shared hosts. The figure helper's checklist (item 4, `deep/figure_helper.py:82-146`) asks for "a wall or pile ... drawn to scale with its diagrams beside it" even where no tool on the host can draw it.

**What to build**, in `charts/data_plot.py` and `charts/plot_tool.py` (one axes today: `plt.subplots` at `data_plot.py:1591`; every placement and check helper takes a single `ax`):
1. **Panels sharing the vertical axis.** Several panels side by side on one elevation (or depth) axis, each with its own series, x label, short title, reference lines and notes; relative widths; panels that hold the same quantity on one x scale; a reference line that crosses every panel (a water table, a neutral plane); a figure title and a footnote.
2. **A section to scale beside its diagrams.** One panel is a section drawn to scale (one metre across = one metre up), sharing its elevation axis with the diagram panels beside it. For that the tool needs: a fill between a curve and a base line (a pressure diagram), layer bands across a panel with a name and a hatch, a dimension line (two heads, its text, set off from what it measures), and an arrow for a resultant with its value.
3. **Contours.** From a grid (x, y and a matrix) or from scattered points (triangulated with matplotlib's own `tri`; no scipy): lines or filled, chosen or automatic levels, labelled lines, a colour scale named with its quantity and unit; to scale when the axes are lengths; outlines and points drawn over it.
Existing inputs keep drawing what they draw today: the existing tests pass unchanged.

**Acceptance: the code examples, redrawn without code.** `charts/figure_examples.py` holds three matplotlib scripts given to the helper as plot-code examples: `wall_pressures`, `settlement_stress_profile`, `pile_design_profile` (`CODE_EXAMPLES`, :670-949). Write each as a `plot_data` parameter set that draws the same figure, add them to `EXAMPLES` so the figure helper can fetch them where plot code is off, and add one contour example on made-up numbers. For each, render the script's figure (through the plotting box, as `funhouse_agent/deep/tests/test_figure_helper.py` does) and the `plot_data` figure, open both PNGs, and check the `plot_data` one against the checklist's eleven items. Commit the pairs under `module_work/gm315/` with a short table in CHECK.md: what matches, and what `plot_data` still cannot do. The layout check (`charts/layout_ink.layout_check`) must come back empty on each and lettering must print at 7 pt or more at the calc package's width.

**The interactive twin.** `build_data_plot_figure` draws a Plotly twin of single-axes figures. For panels and contours, build the twin where it is straightforward (subplots sharing y; a contour trace); where it is not, return no twin and say so in `warnings`. Do not ship a twin that shows something different from the PNG.

**What the model reads.** A new top-level key goes in four places (`VALID_PLOT` and `PLOT_DATA_INFO` in `plot_tool.py`, `render_data_plot`'s signature, the tool function at `deep/vision_kit.py:1286-1298`, and the described-key list at `vision_kit.py:289-295`). The description is about 5,300 characters already and is sent with every call on both pages: add at most about 900 characters for all of this, and put the detail in the examples. Then make every sentence that says only code can draw these true again: `_PLOT_APP_NOTE` (`vision_kit.py:262-276`), the helper's notes (`deep/figure_helper.py:302-306`), the plotting box's description (`plot_box/__init__.py:864-869`), the geotech calc helper's text (`funhouse_agent/deep/agent.py:434-440`). The checklist's wording itself stays. Plot code remains the tool for what `plot_data` still cannot draw: say what that is.

**Fingerprints.** The description changes on both pages, so `webapp/tests/test_page_fingerprints.py` and `packages/aec-document-review/tests/test_review_fingerprints.py` both move. Re-record both in ONE last commit of their own, with nothing else in it (another session re-records them this week too; the lead will re-record again at the merge and needs to be able to drop that commit).

**May edit:** core `charts/` (all of it), `deep/vision_kit.py` (the `plot_data` tool function, its description list and `_PLOT_APP_NOTE` only), `deep/figure_helper.py` and `plot_box/__init__.py` (only the sentences named above), `funhouse_agent/deep/agent.py` (only the sentences named above), `funhouse_agent/adapters/profile_figure_adapter.py`, `profile_figure/DESIGN.md`, the tests of these, `funhouse_agent/tests/method_probes.json` (the `plot_data` row), the two recorded fingerprint files. **Not yours:** any version or pin (the core releases on another session's schedule), every other core file, `packages/aec-document-review/` source.

**Tests** (exit codes): `profile_figure/tests/test_data_plot*.py` (and new files for panels, sections and contours), `geotech_common/tests/test_layout_ink.py`, `packages/aec-agent-core/tests/test_core_charts.py`, `funhouse_agent/deep/tests/test_figure_helper.py`, `funhouse_agent/tests/test_defaults_used.py -k plot_data`, the core-only install's tests for the files you touched (COMMON_DOCREVIEW's setup), both fingerprint tests, `webapp/tests/test_public_text.py`.

## The join: a reinforced wall's calc package ends with its drawing (NOT STARTED: after GM-312 and GM-313 are merged)

Branch `claude/gm-313-wall-package-sheet`. CHECK at `module_work/gm313/JOIN_CHECK.md`.

`retaining_wall_package` with `wall_type="cantilever_reinforced"` takes the design's `reinforcement` description, draws it with `dxf_export`'s wall drawing, hands the sheet to `generate_calc_package(sheets=...)` and writes the `.dxf` beside the package, in `funhouse_agent/adapters/calc_package.py` (the agent layer may import both modules). The result names the package, the DXF and the PNG. `render_figures` saves the sheet's PNG with the other figures. Tests: a package for a made-up wall has the sheet as its last page, and the bar marks, sizes and spacings on the sheet are the design's (read the DXF back). One validation run of the description GM-312 really returns against GM-313's validator; any key the two disagree on is fixed here. Look at the finished PDF's last page and commit it, with the DXF, under `module_work/gm313/`.

## GM-314: the figure test set (NOT STARTED: start when GM-328's script is on master)

Branch `claude/gm-314-figure-test-set`. CHECK at `module_work/gm314/CHECK.md`. Code: a new folder `module_work/figure_test_set/`, and `module_work/live_smoke/` only where the harness needs an option it lacks.

**First, confirm the bed is there.** GM-328 adds `docs/examples/build_sample_packages.py`: it builds one calc package per `*_package` method from a published worked example into a folder named on the command line (default `docs/examples/out/`, gitignored), with each figure as its own PNG under `figures/` and an `index.json` listing them. If that script is not on master, stop and say so. Read the script and one `index.json` it writes before you design anything. Every case in this test set is one of that script's cases: never write inputs of your own.

**What is known** (2026-11-12). Every figure in a canned calc package is drawn by fixed code, so it is the same with model-written plot code on or off. The difference shows only in a figure the agent is ASKED to draw (the figure helper, or a report the calc helper writes itself). No request has been run with plot code off, nothing scores a drawn figure against the figure helper's checklist (`packages/aec-agent-core/aec_agent_core/deep/figure_helper.py:82-146`, eleven items), and no contact sheet exists. `module_work/live_smoke/` is the harness that drives the app's real turn path (`run.py flow --wave NAME --cap USD --model ID --ids ...`; a flow's `env` block sets switches for that flow; `--fake` runs a scripted model at no cost). Its flows F52 to F66 are figure requests on made-up data, all but four with plot code on.

**What to build.**
1. **The requests** (`module_work/figure_test_set/requests.json`, built by a script from `index.json`, not typed by hand): about 20 figure requests. Each names its sample package in `index.json`, carries the numbers to draw (taken from that sample's inputs and results, by running the script's own case; the request gives the agent the numbers, it does not ask for the analysis again), says in an engineer's words what figure is wanted, and lists the facts a reader must find on it (two or three values, the axis ranges, the named lines). Choose them to cover what only plot code draws today and what the fixed tool draws: a wall to scale with its pressure diagrams; a sheet pile or braced excavation with pressure, shear and moment sharing the depth axis; a pile or shaft beside its soil column with load and resistance sharing elevation; stress profiles in panels; contours (fem2d); a slope section at true scale with its slip surface; a plan drawn as a drawing; a time curve; a case comparison; a spectrum; a histogram with its marked values. Where a kind has no sample package, reuse one of the existing figure flows (F56 to F66) as it is written rather than write a new case.
2. **The arms.** Each request becomes one flow per arm: `GEOTECH_PLOT_CODE` "1" and "0", the figure helper on in both. An arm is a switch setting on a commit: record the commit and the switches with every run, so the same requests can be run again on a later master (after GM-315) and laid beside these.
3. **Collect.** After a wave: each request's final figure for each arm (the PNG the turn ended with, not a superseded draft), the request, and how it was drawn (the `plot_data` arguments or the code, from `.figure_code/`), plus the fixed-code figure of the same case from `index.json` where the sample has one.
4. **Score sheet.** One entry per figure: each of the eleven checklist items as pass, fail or not applicable with one line of reason (item 8's "every figure of a job" and item 11's hover text cannot be judged from one PNG: not applicable), then two overall lines: does the figure show the request's facts correctly (the values read back), and what is wrong with it in one sentence. The scoring is done by a vision model that does not know the arm: give each figure a coded file name and keep the key apart. Write the scorer's instructions as one file a reader model is handed with the PNG and the request (the checklist's text imported from the code, never copied), and a script that merges the returned scores. The lead runs the scoring locally; do not build an API call for it.
5. **Contact sheet.** One self-contained HTML page: a row per request, a column per arm (and the fixed-code figure where there is one), each figure at a size a person can read, its eleven marks and the one-sentence fault under it; at the top a table per arm of passes by item and by figure kind. This is the page the owner looks at: plain, no decoration.
6. **Dry run at no cost.** With `--fake` the plumbing runs end to end. Also build the contact sheet for the fixed-code figures alone, from `index.json`, so the sheet exists before any paid run. Commit code and `requests.json`; outputs go to a gitignored folder.

**Not yours to run.** The agent runs are paid model calls and wait for the owner's word; the cloud session has no key and must not try. In CHECK.md give the command lines for the lead (waves, caps, the model ids the harness takes) and an estimate of the cost from the harness's own price table and the recorded figure waves (about $0.06 to $0.82 a figure flow on Sonnet 5.5 in October).

**Tests** (exit codes): `module_work/live_smoke/test_harness.py` and `test_arms.py`, new tests for the request builder (every request names a case that `index.json` has; no request carries a number the sample does not), the collector and the sheet on fake outputs, `webapp/tests/test_public_text.py`.
