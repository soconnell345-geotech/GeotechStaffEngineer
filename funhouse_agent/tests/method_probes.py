"""The probe table behind the METHOD_INFO guard (BACKLOG GM-158, GM-150).

``method_probes.json`` holds, for every analysis-side method, the smallest
inputs it runs on: ``"agent|method"`` (or ``"agent|method#variant"`` where
one method needs several, e.g. one p-y model each) -> ``{"params": {...}}``,
with ``"before": [[agent, method, params], ...]`` for a method that works on
what an earlier call loaded. The inputs came from the test suite's own
calls, cut down to the parameters each method needs, so every parameter a
probe leaves out runs at its default and the guard can check that
default. A few probes also give a seed (a sampler with its own generator)
or a coarse mesh or search grid (the guard runs each probe twice), so
those defaults are the ones not checked here.

A file reader's probe names a sample in ``funhouse_agent/eval_samples/``
as ``"<eval_samples>/<file>"``, or a test fixture as ``"<repo>/<path>"``,
made an absolute path here (the guard runs each probe in a scratch
folder); a probe whose method needs an optional
backend names it under ``"needs"`` (an importable module), and the guard
skips it where that is not installed.

Each method either has a probe or is in :data:`EXCLUDED` with the reason.
"""

import importlib.util
import json
from pathlib import Path

#: The sample files the file readers' probes read (GM-160).
REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "funhouse_agent" / "eval_samples"
_TAGS = {"<eval_samples>/": SAMPLES, "<repo>/": REPO}


def _with_samples(x):
    """``x`` with every ``"<eval_samples>/<file>"`` or ``"<repo>/<path>"``
    made an absolute path."""
    if isinstance(x, str):
        for tag, root in _TAGS.items():
            if x.startswith(tag):
                return str(root / x[len(tag):])
    if isinstance(x, dict):
        return {k: _with_samples(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_with_samples(v) for v in x]
    return x


PROBES = _with_samples(json.loads(
    (Path(__file__).with_name("method_probes.json")).read_text(
        encoding="utf-8")))


#: Import name -> the package to install, for a probe's ``"needs"``.
_PACKAGE_OF = {"python_ags4": "python-ags4", "pydiggs": "pydiggs"}


def skip_reason(key: str):
    """Why this probe cannot run here, naming the package, or None
    (BACKLOG INF-61, as INF-13 names a missing dependency): its module is
    left out of the registry because its package is not installed (Pynite
    on numpy 1.26), or its method needs an optional backend that is not."""
    from funhouse_agent.adapters import ABSENT_MODULES
    agent = key.split("#")[0].split("|")[0]
    if agent in ABSENT_MODULES:
        return (f"{ABSENT_MODULES[agent]} is not installed, so the "
                f"{agent} tool is not listed")
    needs = (PROBES.get(key) or {}).get("needs")
    if needs and importlib.util.find_spec(needs) is None:
        return f"{_PACKAGE_OF.get(needs, needs)} is not installed"
    return None


def listed(keys):
    """The keys whose module is in the registry here."""
    from funhouse_agent.adapters import ABSENT_MODULES
    return [k for k in keys if k.split("#")[0].split("|")[0]
            not in ABSENT_MODULES]

_FILE = "reads a file the call names; no file fixture here"
_HANDLE = "works on a handle an earlier file read returned"

#: Methods with no probe, and why.
EXCLUDED = {
    "drawing_ir|digitize_drawing": _FILE,
    "drawing_ir|get_entities": _HANDLE,
    "drawing_ir|query_drawing": _HANDLE,
    "drawing_ir|search_drawing_set": _FILE,
    "drawing_ir|snip_region": _FILE,
    "pdf_import|build_fem_inputs": _HANDLE,
    "pdf_import|build_slope_geometry": _HANDLE,
    "pdf_import|cleanup_geometry": _HANDLE,
    "pdf_import|cross_check": _HANDLE,
    "pdf_import|discover_pdf_content": _FILE,
    "pdf_import|extract_vector_geometry": _FILE,
    "pdf_import|propose_role_mapping": _HANDLE,
    "pdf_import|propose_scale": _HANDLE,
    "opensees|pm4sand_cyclic_dss": "needs OpenSeesPy, an optional backend "
                                   "that is not always present",
    "opensees|site_response_1d": "needs OpenSeesPy, an optional backend "
                                 "that is not always present",
}

#: Declared parameters an adapter reads and removes before its own name
#: check, so the check's list leaves them out: (method, name) -> how.
READ_BEFORE_THE_CHECK = {
    ("ground_improvement|aggregate_piers", "area_replacement_ratio"):
        "turned into a spacing for the column diameter and pattern",
}

#: Probes that take several seconds a run even on a coarse grid.
SLOW = frozenset({"slope_stability|search_rapid_drawdown"})

#: Result keys that share a parameter's name but report something else,
#: so ``defaults_used`` and the result may differ there: (probe key, name)
#: -> what the result's value is.
ECHO_IS_NOT_THE_INPUT = {
    ("calc_package|seismic_package", "analysis_type"):
        "the package's title",
    ("calc_package|fem2d_package", "analysis_type"):
        "the package's title",
    ("calc_package|downdrag_package", "neutral_plane_method"):
        "the method 'auto' chose",
    ("downdrag|downdrag_analysis", "neutral_plane_method"):
        "the method 'auto' chose",
    ("gstools|variogram", "n_bins"): "the lag classes that hold pairs",
    ("reliability|pem", "scheme"): "the scheme's full name",
    ("salib|sobol_analysis", "n_samples"): "the model outputs analysed",
}

#: Declared inputs that change nothing in their probe's result by design
#: (GM-160): ``(probe key, name) -> what the input does``. The guard
#: gives every default a probe's run listed (``defaults_used``) another
#: value and fails where the result does not move, past the input's own
#: echo and the result's ``choices``; an input here is skipped. A name
#: whose effect the probe's inputs cannot show (a clamp the probe does not
#: reach, a branch it does not take) goes in :data:`NOT_REACHED`, never
#: here. Filled by the GM-184 sweep (2026-10-19).
_HAMMER_LABEL = ("a custom hammer's label in the result and the calc "
                 "package: every type is modelled as a rigid ram, so the "
                 "ram, stroke and efficiency set the blow")
_PLANE_STRAIN = ("plane strain: the thickness scales the stiffness, the "
                 "self-weight and the loads alike (and, since GM-204, the "
                 "beams, springs and wall joints), so the per-metre "
                 "results do not change")
LABEL_ONLY = {
    # GM-310 (OWN-82): a layer naming no wall friction takes none under
    # log_spiral, and at delta = 0 every source's Kp is Rankine's; the
    # #log_spiral and #interface_soil probes (a delta given) reach it
    ("sheet_pile|cantilever_wall#log_spiral_left_out", "log_spiral_source"):
        "the probe's layer names no wall friction, so delta = 0, where "
        "every printed source of the log-spiral Kp gives Rankine's",
    **{(f"wave_equation|{m}", n): _HAMMER_LABEL
       for m in ("bearing_graph", "drivability", "single_blow")
       for n in ("hammer_type", "hammer_custom_name")},
    **{("calc_package|wave_equation_package", n): _HAMMER_LABEL
       for n in ("hammer_type", "hammer_custom_name")},
    **{(f"fem2d|{m}", "t"): _PLANE_STRAIN
       for m in ("fem2d_foundation", "fem2d_gravity", "fem2d_local_fos",
                 "fem2d_slope_srm", "fem2d_excavation", "fem2d_staged",
                 "fem2d_srm_mesh_refinement_study")},
    ("calc_package|render_tables", "output_path"):
        "names the HTML file written; the tables in it are the same",
    ("fem2d|fem2d_seepage", "t"): (
        "plane strain: the thickness scales the flow matrix and the nodal "
        "reactions alike, and the flow is reported per metre run (GM-222)"),
}

#: Declared inputs that act, but not on their probe's inputs or not in
#: what the probe's run returns: ``(probe key, name) -> why the probe
#: does not show it``. Filled by the GM-184 sweep (2026-10-19).
_ALL_CLAY = "the probe's profile is all clay: the beta method is for sand"
_ALL_SAND = "the probe's profile is all sand: the alpha method is for clay"
_NO_ROCK = "the probe has no rock layer, so there is no socket"
_CASING = ("0.5 m of casing lies inside the top 1.5 m that GEC-10 already "
           "leaves out of the side resistance in clay")
_TOE_LIMIT = ("the probe's toe stress stays below GEC-12 Fig. 7-15's "
              "limiting q_L, which the default takes")
_ONE_SAND = ("one sand layer: averaging phi over 3b about the toe gives "
             "the same angle")
_ADHESION = ("'auto' already picks the concrete and timber adhesion curve "
             "for the probe's pile")
_GRID = ("the depth grid is the larger of L / 0.25 m, n_sublayers per "
         "layer and 50, which n_sublayers + 1 does not change here")
_CONVERGES = "the probe converges well within the limit"
_EAC = ("a step of 1 psi moves the layer thicknesses by less than the "
        "design's rounding")
_FIGURE = ("acts on the drawn figure only; the result carries the file, "
           "not its pixels")
_NO_FRICTION = ("the probe's wall friction is 0, where Coulomb's "
                "coefficients are Rankine's")
_LINEAR = ("the probe's soil model is linear, so there is no strain "
           "iteration")
_SUBLAYERS = ("the probe's layer is already thinner than wave_frac times "
              "the shortest wavelength at max_freq_hz")
_PHI_FDN = ("sliding takes the lesser of the backfill and foundation phi; "
            "the probe's backfill phi is the lesser")
_LIMITS = "the probe's run stops well before the limit"
_GEF = "'auto' already reads the probe's .gef file with the gef engine"
_VIBRO = ("acts in the printed calc steps (the probe spacing off "
          "GEC-13 Fig. 4-34); the vibro package writes no table and "
          "returns no number")
_DRAINED = ("the probe's soil is elastic and the reported movements are "
            "the load stage's alone, so the initial hydrostatic state "
            "does not show")
_ITERATIONS = "the probe converges in fewer iterations"
_BLOWUP = ("the probe's strength reduction fails by non-convergence "
           "before either blow-up factor")
_NO_ARTESIAN = ("the probe has no water, so no slice's pore-water force "
                "exceeds its weight and the frictional term is never "
                "negative (GM-140 step 3)")
_NO_LOW_M_ALPHA = ("no trial surface of the probe's search has m_alpha "
                   "below 0.2 on a slice, so keeping and leaving out such "
                   "surfaces give the same search (GM-140 step 3)")
# INF-77 T1 (b): why a parameter a probe gives does not move its run
_ONE_LAYER = ("the probe's one layer reaches below the toe (or the wall); "
              "a thicker one changes nothing")
_NODE_INDEX = ("a doubled node index is out of range and raises IndexError, "
               "not a worded refusal")
_ONE_FEM_LAYER = ("the probe's one layer fills the mesh, so a bottom below "
                  "the mesh's base moves nothing")
_TOP_CUT = ("the top layer's top_elevation above the ground surface is cut "
            "off at the surface")
_NO_LOAD = "the probe carries no load, so every force and movement is 0"
_CARRIED = ("a build tool carries the property into its output under its "
            "own name, set aside as the input's echo; there it is the "
            "effect")
_PY_LINE = ("every point of the probe's curve lies past its initial line "
            "(k z y is above p at the first point), so the initial "
            "stiffness moves nothing")
_STRONG = ("c of 1,000,000 kPa keeps the probe's material elastic, so its "
           "strength never acts")
_STAGGERED = ("the default staggered scheme makes no excess pore pressure "
              "from a load (the result's note says so), so k has no excess "
              "field to move")
_ONE_D = ("a uniform load (or self-weight) on a level layer is "
          "one-dimensional, so any width gives the same result")
_SALIB_STEP = ("SALib's Morris and Sobol analyses read which input a step "
               "moved and the outputs Y; the bounds (and the scale of X) "
               "set the sampling, not the analysis")
_TOTAL_STRESS = ("the probe's one clay layer is read by the alpha method "
                 "and a 9 cu base, which take cu, not the overburden")
NOT_REACHED = {
    **{(f"slope_stability|{m}", "negative_friction_term"): _NO_ARTESIAN
       for m in ("analyze_slope", "search_critical_surface")},
    **{(f"slope_stability|{m}", "low_m_alpha"): _NO_LOW_M_ALPHA
       for m in ("search_critical_surface", "search_rapid_drawdown")},
    ("slope_stability|analyze_slope", "m_alpha_rule"):
        "the probe's surface has m_alpha above 0.2 on every slice, so "
        "which slices the check looks at changes nothing (GM-140 step 3)",
    ("dxf_import|parse_geometry", "ignore_header_units"):
        "the probe gives no units, so there is no difference from the "
        "drawing's header to ignore",
    ("axial_pile|axial_pile_capacity#clay", "adhesion_curve"): _ADHESION,
    ("axial_pile|axial_pile_capacity#clay", "adhesion_chart"):
        "the probe gives no fig_7_18_case, so no Fig. 7-18 chart is read",
    ("axial_pile|capacity_vs_depth", "toe_stress_limit"): _TOE_LIMIT,
    ("axial_pile|capacity_vs_depth", "toe_phi_zone"): _ONE_SAND,
    ("axial_pile|capacity_vs_depth#clay", "adhesion_curve"): _ADHESION,
    # GM-229: su past Fig. 7-17's curves, above the line the probe gives
    ("axial_pile|axial_pile_capacity#overconsolidated", "adhesion_curve"):
        _ADHESION,
    ("axial_pile|capacity_vs_depth#overconsolidated", "adhesion_curve"):
        _ADHESION,
    ("calc_package|axial_pile_package", "toe_stress_limit"): _TOE_LIMIT,
    ("calc_package|axial_pile_package", "toe_phi_zone"): _ONE_SAND,
    # GM-241: K/Ko goes with the GEC-10 beta, the default; pa reaches
    # clay too since GEC-10's alpha became the default (GM-244)
    **{(k, n): _ALL_CLAY for k in (
        "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|lrfd_capacity", "calc_package|drilled_shaft_package")
       for n in ("beta_method", "K_over_Ko")},
    **{(k, n): why for k in (
        "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|capacity_vs_depth#sand",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|drilled_shaft_capacity#sand",
        "drilled_shaft|lrfd_capacity",
        "drilled_shaft|lrfd_capacity#sand")
       for n, why in (("socket_diameter", _NO_ROCK),
                      ("rock_side_C", _NO_ROCK),
                      ("rock_side_method", _NO_ROCK))},
    # casing_depth: these probes have clay over the top 1.5 m; the
    # all-sand probes take side resistance from the ground surface, so
    # their casing reaches the result (GM-241 review)
    **{(k, "casing_depth"): _CASING for k in (
        "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|drilled_shaft_capacity#sand",
        "drilled_shaft|lrfd_capacity")},
    **{(k, n): _NO_ROCK for k in ("calc_package|drilled_shaft_package",
                                  "calc_package|drilled_shaft_package#sand")
       for n in ("rock_side_C", "rock_side_method")},
    **{(k, n): _ALL_SAND for k in (
        "drilled_shaft|capacity_vs_depth#sand",
        "drilled_shaft|drilled_shaft_capacity#large_base",
        "drilled_shaft|lrfd_capacity#sand",
        "calc_package|drilled_shaft_package#sand")
       for n in ("alpha_method", "su_test_type")},
    **{("drilled_shaft|drilled_shaft_capacity#large_base", n): _NO_ROCK
       for n in ("socket_diameter", "rock_side_C", "rock_side_method")},
    # GM-221: the rock probe's soil is clay over rock
    **{(k, n): "the probe's soil is clay over rock: the beta method is "
       "for sand" for k in ("drilled_shaft|drilled_shaft_capacity#rock",
                            "drilled_shaft|drilled_shaft_capacity"
                            "#fractured_rock",
                            "drilled_shaft|drilled_shaft_capacity"
                            "#hoek_brown")
       for n in ("beta_method", "K_over_Ko")},
    **{(k, "casing_depth"): _CASING for k in (
        "drilled_shaft|drilled_shaft_capacity#rock",
        "drilled_shaft|drilled_shaft_capacity#fractured_rock",
        "drilled_shaft|drilled_shaft_capacity#hoek_brown")},
    # GM-244: at RQD 70 or more the 'rqd' base also takes 2.5 qu; the
    # #fractured_rock probe (RQD 40, the conditions of Eq. 10-23 stated
    # met) reaches it, and the #hoek_brown probe (RQD 40 with GSI and m_i,
    # the Hoek-Brown base, GM-251)
    ("drilled_shaft|drilled_shaft_capacity#rock", "rock_base_method"):
        "the probe's rock has RQD 100 (given: GM-310 refuses a rock base "
        "with no RQD), where the 'rqd' base also takes q_b = 2.5 qu",
    ("downdrag|downdrag_analysis", "n_sublayers"): _GRID,
    # GM-228: the warning when the two rules for the depth below the toe
    # disagree; the #toe_in_sand probe (26 % apart) reaches it
    ("downdrag|downdrag_analysis", "influence_depth_warning_percent"):
        "the probe's two rules give pile head settlements 11 % apart, "
        "under 25 % and 37.5 % alike",
    ("downdrag|downdrag_analysis#toe_in_sand", "n_sublayers"): _GRID,
    ("downdrag|post_liquefaction_downdrag", "n_sublayers"): _GRID,
    # GM-207: the result as returned leaves out the basis text that names
    # these choices (detail='full' has it); no number of the probe moves
    ("downdrag|post_liquefaction_downdrag", "skin_friction_stress"):
        "the probe has no fill or drawdown, so the initial and final "
        "effective stress are the same",
    ("downdrag|post_liquefaction_downdrag", "influence_depth_criterion"):
        "the probe's soil below the toe has no compressibility, so it "
        "settles by nothing to either depth",
    ("downdrag|post_liquefaction_downdrag",
     "influence_depth_warning_percent"):
        "the probe's soil below the toe has no compressibility, so the "
        "two rules give the same result and nothing is warned",
    ("calc_package|downdrag_package", "n_sublayers"): _GRID,
    ("calc_package|downdrag_package", "influence_depth_criterion"):
        "the probe's soil below the toe is marked as not settling",
    ("calc_package|downdrag_package", "influence_depth_warning_percent"):
        "the probe's soil below the toe is marked as not settling, so the "
        "two rules give the same result and nothing is warned",
    # GM-216: the same probe with a ground elevation
    ("calc_package|downdrag_package#elevation", "n_sublayers"): _GRID,
    ("calc_package|downdrag_package#elevation",
     "influence_depth_criterion"):
        "the probe's soil below the toe is marked as not settling",
    ("calc_package|downdrag_package#elevation",
     "influence_depth_warning_percent"):
        "the probe's soil below the toe is marked as not settling, so the "
        "two rules give the same result and nothing is warned",
    **{("ground_improvement|feasibility", n):
       "a bearing problem needs both the required and current bearing"
       for n in ("required_bearing_kPa", "current_bearing_kPa")},
    ("ground_improvement|feasibility", "predicted_settlement_mm"):
        "0.5 mm stays below the allowable settlement",
    ("ground_improvement|feasibility", "allowable_settlement_mm"):
        "the probe predicts no settlement, so 75 mm allows it as 50 mm does",
    ("ground_improvement|feasibility", "depth_to_top_m"):
        "the probe's treatment reaches 0.5 m, well short of GEC-13's depth "
        "limits (10.7 m, 30.5 m, 50 m) that it is held against",
    ("calc_package|ground_improvement_package",
     "target_relative_density"): _VIBRO,
    ("calc_package|ground_improvement_package", "pattern"): _VIBRO,
    **{(f"lateral_pile|get_py_curve#{m}", "loading"):
       "at the probe's depth (z / b about 8) the cyclic curve is the "
       "static one; it differs near the surface (and, for SandReese's "
       "'reese1974' construction, by B at any depth)"
       for m in ("SandAPI", "SandReese")},
    ("lateral_pile|get_py_curve#WeakRock", "rock_top"):
        "the probe's depth is more than 3 b below the rock top either way",
    ("lateral_pile|lateral_pile_analysis", "max_iterations"): _CONVERGES,
    ("calc_package|lateral_pile_package", "max_iterations"): _CONVERGES,
    # GM-287: with API Fig. 6.8.6-1's C1-C3 the probe's sand stays near
    # its initial line at 100 kN
    **{(k, "tolerance"): "the probe converges in the same three "
       "iterations at 1e-5 and 1.5e-5" for k in (
           "lateral_pile|lateral_pile_analysis",
           "calc_package|lateral_pile_package")},
    **{(k, "i_c_limit"): "every Ic in the probe is below 2.6, so a higher "
       "limit adds no point" for k in (
           "liquefaction|liquefaction_analysis", "liquepy|cpt_liquefaction")},
    ("pavement_design|compare_pavement_methods", "eac_psi"): _EAC,
    ("calc_package|pavement_design_package", "eac_psi"): _EAC,
    ("calc_package|pavement_design_package", "cbr_subbase"):
        "a subbase CBR of 31 against 30 moves no layer past the design's "
        "rounding",
    ("pavement_design|rigid_pavement_design", "thickness_increment_in"):
        "D = 8.985 in rounds up to 9.0 in at either increment",
    ("settlement|schmertmann_settlement", "shape"):
        "the strain factor diagram changes for a strip only; other shapes "
        "take it from L / B, and the probe is a square",
    **{(k, "units"): "sets the DXF header's $INSUNITS only; the result "
       "carries the file, not its header" for k in (
           "dxf_export|export_geometry_to_dxf",
           "dxf_export|export_to_dxf_bytes")},
    ("fem2d|fem2d_consolidation", "gwt"): _DRAINED,
    ("fem2d|fem2d_consolidation", "gamma_w"): _DRAINED,
    ("fem2d|fem2d_consolidation#monolithic", "gwt"): _DRAINED,
    ("fem2d|fem2d_consolidation#monolithic", "gamma_w"): _DRAINED,
    ("fem2d|fem2d_consolidation#monolithic", "t"):
        "the plane-strain thickness scales the load, stiffness, storage "
        "and flow of the coupled solve alike, so the column does not show it",
    ("fem2d|fem2d_consolidation#monolithic", "startup_steps"):
        "2 -> 3: two start-up steps already damp the jump at the drained "
        "boundary; 0 (none) leaves the Crank-Nicolson ringing, 16.8 kPa at "
        "1e8 s in this probe (fem2d/tests/test_consolidation_startup_fill.py)",
    ("fem2d|fem2d_staged", "gamma_w"):
        "the probe's phases give no water table",
    **{(k, "max_iter"): _ITERATIONS for k in (
        "fem2d|fem2d_excavation", "fem2d|fem2d_excavation#interface",
        "fem2d|fem2d_footing_capacity",
        "fem2d|fem2d_footing_capacity#rigid", "fem2d|fem2d_local_fos",
        "fem2d|fem2d_slope_srm", "fem2d|fem2d_staged")},
    ("fem2d|fem2d_local_fos", "blowup_factor"): _BLOWUP,
    ("fem2d|fem2d_slope_srm", "blowup_factor"): _BLOWUP,
    ("fem2d|fem2d_local_fos", "tol"): _ITERATIONS,
    ("fem2d|fem2d_staged", "tol"): _ITERATIONS,
    **{("fem2d|fem2d_srm_mesh_refinement_study", n): (
        "the study reports only each mesh's FOS, found to the SRF "
        "tolerance, and the probe's FOS does not move with it")
       for n in ("max_iter", "blowup_factor", "tol", "n_load_steps")},
    ("fem2d|fem2d_srm_mesh_refinement_study", "conv_tol"):
        "the probe's two meshes differ by 4.8 %, beyond both 3 % and 4.5 %, "
        "so the converged flag reads the same",
    ("fem2d|fem2d_slope_srm", "mesh_check"):
        "it adds the result's mesh_check block, which shares the input's "
        "name, so the guard leaves it out with the echo",
    ("pile_group|pile_group_simple", "Mx"):
        "the probe's piles all lie on y = 0, so a moment about x loads none",
    **{("pystrata|eql_site_response", n): _LINEAR for n in (
        "strain_ratio", "tolerance", "max_iterations")},
    ("pystrata|eql_site_response#darendeli", "max_iterations"):
        _ITERATIONS,
    **{(k, n): _SUBLAYERS for k in (
        "pystrata|eql_site_response", "pystrata|linear_site_response")
       for n in ("max_freq_hz", "wave_frac")},
    ("reliability|form", "max_iterations"): _CONVERGES,
    ("calc_package|reliability_package#form", "max_iterations"):
        _CONVERGES,
    ("calc_package|reliability_package#pem", "scheme"):
        "one random variable: Rosenblueth's 2n+1 multiplicative form then "
        "gives the full two-point estimate's mean and COV exactly",
    **{(k, "phi_foundation"): _PHI_FDN for k in (
        "retaining_walls|mse_lrfd_external_stability",
        "retaining_walls|mse_wall", "retaining_walls|mse_wall#surcharge",
        "retaining_walls|mse_wall#per_level",
        "retaining_walls|mse_wall#geogrid",
        "retaining_walls|mse_wall#depths")},
    # GM-307: a batter under 10 deg is near vertical (GEC-11 Vol I p.
    # 4-15); the probe's step from 0 stays below it, which only the
    # choices then report
    **{(k, "face_batter"): (
        "the probe's step stays below the 10 deg from which GEC-11 Vol I "
        "p. 4-15 takes a face as battered; a near-vertical face is "
        "vertical, which the choices report")
       for k in ("retaining_walls|mse_wall",
                 "retaining_walls|mse_wall#surcharge",
                 "retaining_walls|mse_wall#per_level",
                 "retaining_walls|mse_wall#geogrid",
                 "retaining_walls|mse_wall#depths",
                 "calc_package|retaining_wall_package#mse",
                 "calc_package|retaining_wall_package#mse_bearing")},
    ("calc_package|retaining_wall_package#surcharge",
     "surcharge_in_bearing"):
        "the probe gives no q_allowable, so the bearing check it enters "
        "is not run",
    **{(k, n): _LIMITS for k in ("salib|morris_analysis",
                                 "salib|sobol_analysis")
       for n in ("max_evaluations", "time_budget_s")},
    ("salib|morris_analyze", "seed"):
        "the probe's model is linear, so every bootstrap sample gives the "
        "same mu* and the confidence interval is 0",
    **{(k, "pressure_method"): _NO_FRICTION for k in (
        "sheet_pile|anchored_wall", "sheet_pile|cantilever_wall",
        "calc_package|sheet_pile_package")},
    ("soe|check_bottom_blowout", "FOS_required"):
        "the probe's factor of safety clears both requirements",
    ("soe|design_ground_anchor", "anchor_angle_deg"):
        "GEC-4's 4.5 m minimum free length governs at either angle",
    ("soe|design_ground_anchor", "soil_type"):
        "the guard's other value, sand_gravel_loose, prints the same "
        "145 kN/m as sand_medium in GEC-4 Table 6, and the 4.5 m minimum "
        "bond length governs either way",
    ("soe|fhwa_apparent_pressure_anchored_wall", "FOS_embedment"):
        "the probe's wall reaches no moment balance at FS = 1.3 within 6H, "
        "so no embedment is reported at either factor",
    **{(k, "engine"): _GEF for k in ("subsurface|parse_bore",
                                     "subsurface|parse_cpt")},
    ("subsurface|plot_atterberg_limits", "use_elevation"): _FIGURE,
    **{("subsurface|plot_cross_section", n): _FIGURE
       for n in ("column_width", "show_gwl")},
    **{("subsurface|plot_parameter_vs_depth", n): _FIGURE
       for n in ("color_by", "show_bands")},
    **{("subsurface|plot_plan_view", n): _FIGURE
       for n in ("color_by", "label_field")},
    **{("profile_figure|plot_data", n): _FIGURE
       for n in ("scale_units", "legend_position", "grid")},
    ("subsurface|spt_correction", "rod_length_m"):
        "the rod length correction C_R is 1.0 for rods over 10 m",
    ("wave_equation|drivability", "refusal_blow_count"):
        "the probe's blow counts stay below either refusal",
    ("worked_examples|find_worked_examples", "limit"):
        "the probe's topic matches one example",
    ("calc_package|fem2d_package", "FOS_required"):
        "the probe's FOS (1.48) is below both 1.5 and 2.25, so "
        "meets_FOS_required reads the same; a FOS of 1.2 required passes "
        "(funhouse_agent/tests/test_fem2d_package_gm328.py)",
    ("calc_package|html_to_pdf", "allow_unembeddable"):
        "the probe's HTML has no image to embed",
    # INF-77 T1 (b): parameters a probe gives (``layers[0].thickness`` is
    # a field of the first item of a list of dicts) that do not move it
    **{(k, "layers[0].thickness"): _ONE_LAYER for k in (
        "axial_pile|axial_pile_capacity",
        "axial_pile|axial_pile_capacity#clay",
        "axial_pile|axial_pile_capacity#overconsolidated",
        "calc_package|soe_package", "calc_package|soe_package#braced",
        "downdrag|downdrag_analysis", "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|capacity_vs_depth#sand",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|drilled_shaft_capacity#large_base",
        "drilled_shaft|lrfd_capacity", "drilled_shaft|lrfd_capacity#sand",
        "sheet_pile|anchored_wall", "sheet_pile|cantilever_wall",
        "sheet_pile|cantilever_wall#interface_soil",
        "sheet_pile|cantilever_wall#log_spiral",
        "sheet_pile|cantilever_wall#log_spiral_left_out",
        "soe|apparent_pressure", "soe|braced_excavation",
        "soe|braced_excavation#log_spiral", "soe|cantilever_excavation",
        "soe|cantilever_excavation#log_spiral")},
    ("lateral_pile|lateral_pile_analysis", "layers[0].bottom"): _ONE_LAYER,
    **{(k, "soil_layers[0].bottom_elevation"): _ONE_FEM_LAYER for k in (
        "fem2d|fem2d_consolidation", "fem2d|fem2d_consolidation#monolithic")},
    **{(k, "soil_layers[0].top_elevation"): _TOP_CUT for k in (
        "fem2d|fem2d_excavation", "fem2d|fem2d_excavation#interface")},
    ("pile_group|pile_group_6dof", "axial_stiffness"): _NO_LOAD,
    **{("pynite|continuous_beam", n): _NO_LOAD for n in ("E", "I")},
    **{("pynite|frame_analysis", n): _NO_LOAD for n in (
        "members[0].A", "members[0].E", "members[0].Iz", "nodes[0].x",
        "nodes[0].y")},
    **{("dxf_import|build_fem_inputs", f"soil_properties[0].{n}"): _CARRIED
       for n in ("E", "gamma", "nu", "phi")},
    ("dxf_import|build_slope_geometry", "soil_properties[0].phi"): _CARRIED,
    ("lateral_pile|get_py_curve#SandReese", "k"): _PY_LINE,
    ("lateral_pile|get_py_curve#SoftClayJeanjean", "Gmax"): _PY_LINE,
    ("lateral_pile|get_py_curve#StiffClayBelowWT", "ks"): _PY_LINE,
    ("lateral_pile|get_py_curve#WeakRock", "Er"): _PY_LINE,
    ("lateral_pile|get_py_curve#StiffClayBelowWT", "gamma"):
        "at the probe's depth the flow-around resistance 11 c b governs, "
        "which does not read gamma",
    **{("fem2d|fem2d_staged", f"material_props[0].{n}"): _STRONG
       for n in ("c", "phi", "psi")},
    ("fem2d|fem2d_seepage", "elements"): _NODE_INDEX,
    ("fem2d|fem2d_seepage", "head_bcs"): _NODE_INDEX,
    ("fem2d|fem2d_staged", "elements"): _NODE_INDEX,
    ("fem2d|fem2d_consolidation", "k"): _STAGGERED,
    ("fem2d|fem2d_consolidation#monolithic", "k"):
        "the probe reports t = 0 and 1e8 s only: the loading instant and "
        "the end of consolidation",
    ("fem2d|fem2d_gravity", "width"): _ONE_D,
    ("calc_package|fem2d_package#consolidation", "width"): _ONE_D,
    ("calc_package|fem2d_package#consolidation", "soil_layers[0].gamma"):
        "the settlements are the load stage's, measured from the end of the "
        "self-weight stage, on linear elastic soil",
    ("salib|morris_analyze", "X"): _SALIB_STEP,
    ("salib|morris_analyze", "bounds"): _SALIB_STEP,
    ("salib|sobol_analyze", "bounds"): _SALIB_STEP,
    ("axial_pile|capacity_vs_depth#clay", "layers[0].unit_weight"):
        _TOTAL_STRESS,
    ("drilled_shaft|capacity_vs_depth", "layers[0].unit_weight"):
        _TOTAL_STRESS,
    **{("calc_package|pile_group_package#cap", n): _FIGURE
       for n in ("cap_outline", "property_line")},
    **{("profile_figure|subsurface_profile", f"layers[0].{n}"): _FIGURE
       for n in ("cu", "unit_weight")},
    ("subsurface|plot_cross_section", "annotate_parameter"): _FIGURE,
    ("axial_pile|axial_pile_capacity#beta", "pile_type"):
        "for the probe's pile and sand, pipe_open's lesser of the plugged "
        "and unplugged cases is the plugged one, which bears as pipe_closed",
    ("axial_pile|capacity_vs_depth", "wall_thickness"):
        "a closed pipe bears on its full section, so the wall thickness "
        "sets only the steel area, which this run does not print",
    ("calc_package|downdrag_package#elevation", "ground_elevation"):
        "it sets the elevations drawn and echoed; the package's numbers go "
        "by depth below it",
    ("calc_package|pile_group_package#eta", "cu"):
        "the probe's cu and the changed one (75 kPa) both lie under the "
        "2 ksf (96 kPa) eta_g is read against",
    ("downdrag|downdrag_analysis", "layers[0].settling"):
        "the probe applies no fill or drawdown, so no layer settles either "
        "way and the neutral plane is the force balance alone",
    ("downdrag|post_liquefaction_downdrag", "layers[0].phi"):
        "the layer gives beta, so its shaft friction does not read phi",
    ("ground_improvement|surcharge_preloading", "surcharge_kPa"):
        "it sets only the equivalent preconsolidation sigma_v0 + surcharge, "
        "and the probe gives no sigma_v0; the settlement is S_ultimate as "
        "given",
    ("lateral_pile|composite_section_ei", "section_type"):
        "with no bar ring a cased_concrete section is the probe's filled "
        "pipe",
    **{("pile_group|group_efficiency", n):
       "for block failure only, as the text says; the probe asks eta_g only"
       for n in ("n_rows", "n_cols")},
    ("retaining_walls|cantilever_wall#cohesion", "c_foundation"):
        "sliding is 'drained' (the default with phi_foundation above 0), "
        "which does not count c, as the text says; the probe gives no "
        "bearing inputs, so bearing is not run",
    **{(k, "theory"): "the probe's wall is vertical with level backfill and "
       "no wall friction, where Coulomb's K is Rankine's" for k in (
           "retaining_walls|earth_pressure_coefficient#coulomb",
           "retaining_walls|earth_pressure_coefficient#coulomb_passive")},
    ("retaining_walls|mse_lrfd_external_stability", "phi_backfill"):
        "the loads come from the retained fill's phi, and sliding takes the "
        "lesser of the reinforced fill and foundation phi, here the "
        "foundation's",
    ("settlement|consolidation_settlement", "layers[0].Cr"):
        "the probe's layer gives no sigma_p, so it is normally consolidated "
        "and Cr is not read",
    ("soe|design_ground_anchor", "phi_deg"):
        "the unbonded length is GEC-4's 4.5 m minimum, longer than the "
        "45 + phi/2 surface asks at the probe's depth",
    ("sheet_pile|cantilever_wall#log_spiral_left_out", "pressure_method"):
        "the probe's layer names no wall friction, so delta = 0, where the "
        "log-spiral Kp is Rankine's",
    ("worked_examples|find_worked_examples", "topic"):
        "a suffix leaves the topic's words, so the search finds the same "
        "example",
}

#: fem2d_package runs the fem2d tools (GM-328), so an input that does not
#: show on a tool's probe does not show on the package's probe of the same
#: tool either, for the tool's reason: its entries are the tools', one
#: source.
_FEM2D_PACKAGE_PROBES = {
    "calc_package|fem2d_package": "fem2d|fem2d_slope_srm",
    "calc_package|fem2d_package#consolidation": "fem2d|fem2d_consolidation",
    "calc_package|fem2d_package#excavation": "fem2d|fem2d_excavation",
    "calc_package|fem2d_package#seepage": "fem2d|fem2d_seepage",
    "calc_package|fem2d_package#staged": "fem2d|fem2d_staged",
}
for _table in (LABEL_ONLY, NOT_REACHED):
    for _package, _tool in _FEM2D_PACKAGE_PROBES.items():
        _table.update({(_package, _n): _why
                       for (_k, _n), _why in list(_table.items())
                       if _k == _tool})

#: Mismatches the guard finds and leaves, each with its BACKLOG item.
#: slope_stability is frozen by the owner until the integral review.
KNOWN = {
    ("slope_stability|analyze_slope", "method"):
        "GM-140: 'bishop' is listed, but a noncircular slip_points surface "
        "runs Spencer",
}

#: Probes the GM-160 guard leaves out, by agent or by probe: why.
_GM140 = ("GM-140: slope_stability is frozen by the owner until its "
          "integral review, and its probes run over 45 minutes")
FROZEN = {
    "slope_stability": _GM140,
    "calc_package|slope_report_package": _GM140,
    "calc_package|slope_stability_package": _GM140,
}

#: Inputs a run lists as taken from another input (GM-165) that, given
#: back, add a row the package prints only for a given input, the numbers
#: the same: (probe key, name) -> why. The guard does not give them back.
SHOWN_ONLY_WHEN_GIVEN = {
    ("calc_package|drilled_shaft_package", "socket_diameter"):
        "the calc steps print a rock-socket diameter row only when it is "
        "given; the probe has no rock layer",
}

#: Probes whose run takes 2 s or more on a laptop, left out of the INF-77
#: sweep over the parameters they give (T1 b; their defaults are still
#: changed): together they would add about 4 CPU-minutes a job.
_SLOW_RUN = "its run takes 2 to 5 s and it gives 5 to 17 parameters"
GIVEN_LEFT_OUT = {k: _SLOW_RUN for k in (
    "calc_package|fem2d_package", "calc_package|fem2d_package#excavation",
    "calc_package|retaining_wall_package#reinforced",
    "fem2d|fem2d_footing_capacity", "fem2d|fem2d_footing_capacity#rigid",
    "fem2d|fem2d_local_fos", "fem2d|fem2d_slope_srm",
    "fem2d|fem2d_srm_mesh_refinement_study")}

#: METHOD_INFO ``returns`` keys a probe's result leaves out by design whose
#: text does not say when they come back (INF-77 T1 a): (method, key) ->
#: when they do. Any other named key the result lacks fails the guard,
#: unless its text says when (a condition word, a case label such as
#: "Cohesive: ...", or "with" a declared parameter), or it is a warnings
#: or notes key (there only when there is something to say).
_BY_METHOD = ("the result of the method it names; a run of another method "
              "leaves it out")
_BLOCK = ("the block-failure keys come with pile_length and su1 and su2 "
          "(or one cu), the cohesive block check; the probe asks eta_g "
          "only")
RETURNED_SOMETIMES = {
    ("calc_package|fem2d_package", "extra"):
        "names the fem2d tool's own keys, not one key",
    ("calc_package|pavement_design_package", "extra"):
        "names the design's own numbers, not one key",
    ("calc_package|html_to_pdf", "images_embedded"):
        "with local image files in the HTML; the probe's has none",
    ("calc_package|html_to_pdf", "included"):
        "with a data-include in the HTML; the probe's has none",
    ("calc_package|html_to_pdf", "unembeddable"):
        "with a figure that cannot be embedded; the probe's HTML has none",
    ("calc_package|pile_group_package", "group_efficiency_row"):
        "when eta_g is read off a printed row; the probes' is not",
    **{("calc_package|reliability_package", k): _BY_METHOD for k in (
        "beta", "pf", "beta_normal", "pf_normal", "beta_lognormal",
        "pf_lognormal")},
    ("calc_package|retaining_wall_package", "drawing_scale"):
        "with the drawing sheet; null, so left out, without it",
    ("fem2d|fem2d_excavation", "wall_water_pressures"):
        "with water on the wall; the probes have none",
    ("liquefaction|liquefaction_analysis", "layer_results"):
        "the SPT routes' per-layer table; the probe runs a CPT",
    ("pavement_design|flexible_pavement_design_ufc", "frost_governs"):
        "with frost inputs that give a frost group; the probe gives none",
    **{("pile_group|group_efficiency", k): _BLOCK for k in (
        "block_failure_phi", "block_failure_Nc", "block_width_m",
        "block_length_m")},
    **{("pile_group|group_efficiency", k):
       "with soil_type (eta_g read off the printed table); the #block "
       "probe asks block failure only"
       for k in ("s_over_b", "group_efficiency_row")},
    **{("pile_group|group_uplift", k): "with uplift_load and "
       "axial_capacity_tension (the governing resistance); the probes give "
       "no uplift_load" for k in ("group_uplift_utilization",
                                  "group_uplift_adequate")},
    ("retaining_walls|earth_pressure_coefficient", "choices"):
        "for the log-spiral passive coefficient only",
    ("seismic_geotech|seismic_earth_pressure", "delta_passive_deg"):
        "with include_passive, beside KPE",
}


# ---------------------------------------------------------------------------
# GM-316: every warning a run raises that must reach the agent is in its
# result. Checked on every probe inside the defaults guard's own run
# (test_defaults_used; review B5: no second pass over the probes).

def _warning_dirs() -> tuple:
    """The folders of the app's calculation packages, each ending in a
    separator (``funhouse_agent.core_hooks.WARNING_PACKAGES``)."""
    import os
    from funhouse_agent.core_hooks import WARNING_PACKAGES
    out = []
    for name in WARNING_PACKAGES:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        for loc in (getattr(spec, "submodule_search_locations", None)
                    or ()):
            out.append(os.path.normcase(os.path.abspath(loc)) + os.sep)
    return tuple(out)


WARNING_DIRS = _warning_dirs()


def must_reach(w) -> bool:
    """Whether a recorded warning must reach the agent: a numpy
    floating-point notice anywhere, or any warning but a notice raised in
    a calculation package (``aec_agent_core.call_warnings``'s rule)."""
    import os
    from aec_agent_core.call_warnings import _NOISE, _NUMPY_FP
    if issubclass(w.category, _NOISE):
        return False
    if issubclass(w.category, RuntimeWarning) and _NUMPY_FP.match(
            str(w.message)):
        return True
    return os.path.normcase(os.path.abspath(w.filename)).startswith(
        WARNING_DIRS)


def lost_warnings(recorded, result) -> list:
    """The texts of the ``recorded`` warnings (a ``catch_warnings(record=
    True)`` list under ``simplefilter("always")``) that must reach the agent
    and are not in ``result``'s ``warnings`` (a list cut with "and N more"
    loses none)."""
    got = result.get("warnings") if isinstance(result, dict) else None
    got = [str(g) for g in (got or [])]
    if any(g.startswith("and ") and g.endswith(" more") for g in got):
        return []
    return [str(w.message) for w in recorded
            if must_reach(w) and str(w.message) not in got]

