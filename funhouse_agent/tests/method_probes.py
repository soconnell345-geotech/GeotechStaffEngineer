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
    **{(f"wave_equation|{m}", n): _HAMMER_LABEL
       for m in ("bearing_graph", "drivability", "single_blow")
       for n in ("hammer_type", "hammer_custom_name")},
    **{("calc_package|wave_equation_package", n): _HAMMER_LABEL
       for n in ("hammer_type", "hammer_custom_name")},
    **{(f"fem2d|{m}", "t"): _PLANE_STRAIN
       for m in ("fem2d_foundation", "fem2d_gravity", "fem2d_local_fos",
                 "fem2d_slope_srm", "fem2d_excavation", "fem2d_staged")},
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
_VIBRO = ("acts in the printed calc steps (spacing and improvement "
          "ratio); the vibro package writes no table and returns no "
          "number")
_DRAINED = ("the probe's soil is elastic and the reported movements are "
            "the load stage's alone, so the initial hydrostatic state "
            "does not show")
_ITERATIONS = "the probe converges in fewer iterations"
_BLOWUP = ("the probe's strength reduction fails by non-convergence "
           "before either blow-up factor")
NOT_REACHED = {
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
    # GM-241: K/Ko and pa go with the GEC-10 beta, the default
    **{(k, n): _ALL_CLAY for k in (
        "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|lrfd_capacity", "calc_package|drilled_shaft_package")
       for n in ("beta_method", "K_over_Ko", "pa")},
    **{(k, n): why for k in (
        "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|capacity_vs_depth#sand",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|drilled_shaft_capacity#sand",
        "drilled_shaft|lrfd_capacity",
        "drilled_shaft|lrfd_capacity#sand")
       for n, why in (("socket_diameter", _NO_ROCK),
                      ("rock_side_C", _NO_ROCK))},
    # casing_depth: these probes have clay over the top 1.5 m; the
    # all-sand probes take side resistance from the ground surface, so
    # their casing reaches the result (GM-241 review)
    **{(k, "casing_depth"): _CASING for k in (
        "drilled_shaft|capacity_vs_depth",
        "drilled_shaft|drilled_shaft_capacity",
        "drilled_shaft|drilled_shaft_capacity#sand",
        "drilled_shaft|lrfd_capacity")},
    ("calc_package|drilled_shaft_package", "rock_side_C"): _NO_ROCK,
    ("calc_package|drilled_shaft_package#sand", "rock_side_C"): _NO_ROCK,
    **{(k, "alpha_method"): _ALL_SAND for k in (
        "drilled_shaft|capacity_vs_depth#sand",
        "drilled_shaft|lrfd_capacity#sand",
        "calc_package|drilled_shaft_package#sand")},
    # GM-221: the rock probe's soil is clay over rock
    **{("drilled_shaft|drilled_shaft_capacity#rock", n):
       "the probe's soil is clay over rock: the beta method is for sand"
       for n in ("beta_method", "K_over_Ko", "pa")},
    ("drilled_shaft|drilled_shaft_capacity#rock", "casing_depth"): _CASING,
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
    ("calc_package|ground_improvement_package", "target_N_spt"): _VIBRO,
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
    **{("settlement|schmertmann_settlement", n):
       "the strain factor diagram changes only for a strip or L / B over "
       "10; the probe is a square" for n in ("L", "shape")},
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
        "fem2d|fem2d_footing_capacity", "fem2d|fem2d_local_fos",
        "fem2d|fem2d_slope_srm", "fem2d|fem2d_staged")},
    ("fem2d|fem2d_local_fos", "blowup_factor"): _BLOWUP,
    ("fem2d|fem2d_slope_srm", "blowup_factor"): _BLOWUP,
    ("fem2d|fem2d_local_fos", "tol"): _ITERATIONS,
    ("fem2d|fem2d_staged", "tol"): _ITERATIONS,
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
        "retaining_walls|mse_wall", "retaining_walls|mse_wall#surcharge")},
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
    ("calc_package|html_to_pdf", "allow_unembeddable"):
        "the probe's HTML has no image to embed",
    # GM-242: the first cantilever_wall probe with include_passive
    ("retaining_walls|cantilever_wall#log_spiral_toe", "has_shear_key"):
        "the probe gives no key_depth, and a key 0 m deep is taken as no key",
}

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
