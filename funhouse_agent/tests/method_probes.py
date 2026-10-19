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

#: Probes whose run takes several seconds, so giving each listed default
#: other values takes a minute or more (GM-160's check only).
SLOW_TO_VARY = frozenset({
    "calc_package|downdrag_package", "calc_package|lateral_pile_package",
    "calc_package|retaining_wall_package",
    "calc_package|retaining_wall_package#surcharge",
    "calc_package|sheet_pile_package", "calc_package|slope_report_package",
    "calc_package|wave_equation_package", "fem2d|fem2d_excavation",
    "fem2d|fem2d_footing_capacity", "fem2d|fem2d_local_fos",
    "fem2d|fem2d_slope_srm", "gstools|conditional_simulation",
    "slope_stability|monte_carlo_fos",
    "slope_stability|search_critical_surface",
    "slope_stability|yield_acceleration"})

#: Probes GM-160's check does not vary, and why.
NOT_VARIED = {
    k: "each run is a full critical-surface search (minutes), and "
       "slope_stability is frozen by the owner until the integral review"
    for k in ("slope_stability|monte_carlo_fos",
              "slope_stability|search_critical_surface",
              "slope_stability|search_rapid_drawdown")}

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
#: here.
LABEL_ONLY = {
    ('calc_package|wave_equation_package', 'hammer_custom_name'):
        'a label in the result and calc package',
    ('calc_package|wave_equation_package', 'hammer_type'):
        'a label: every hammer is a rigid ram (GEC-12 Sec. 12.6.2); see its description',
    ('ground_improvement|aggregate_piers', 'E_column'):
        'its description says: not used in the calculation, only a warning below E_soil',
    ('profile_figure|plot_data', 'grid'):
        'changes the figure only; the result is the figure file',
    ('profile_figure|plot_data', 'legend_position'):
        'changes the figure only; the result is the figure file',
    ('profile_figure|plot_data', 'logx'):
        'changes the figure only; the result is the figure file',
    ('profile_figure|plot_data', 'logy'):
        'changes the figure only; the result is the figure file',
    ('profile_figure|plot_data', 'scale_units'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_atterberg_limits', 'use_elevation'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_cross_section', 'column_width'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_cross_section', 'show_gwl'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_parameter_vs_depth', 'color_by'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_parameter_vs_depth', 'show_bands'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_plan_view', 'color_by'):
        'changes the figure only; the result is the figure file',
    ('subsurface|plot_plan_view', 'label_field'):
        'changes the figure only; the result is the figure file',
    ('wave_equation|bearing_graph', 'hammer_custom_name'):
        'a label in the result and calc package',
    ('wave_equation|bearing_graph', 'hammer_type'):
        'a label: every hammer is a rigid ram (GEC-12 Sec. 12.6.2); see its description',
    ('wave_equation|drivability', 'hammer_custom_name'):
        'a label in the result and calc package',
    ('wave_equation|drivability', 'hammer_type'):
        'a label: every hammer is a rigid ram (GEC-12 Sec. 12.6.2); see its description',
    ('wave_equation|single_blow', 'hammer_custom_name'):
        'a label in the result and calc package',
    ('wave_equation|single_blow', 'hammer_type'):
        'a label: every hammer is a rigid ram (GEC-12 Sec. 12.6.2); see its description',
}

#: Declared inputs that act, but not on their probe's inputs: ``(probe
#: key, name) -> why the probe does not show it``.
NOT_REACHED = {
    ('bearing_capacity|bearing_capacity_analysis', 'vertical_load'):
        'inclination factors need a horizontal load; the probe has none',
    ('bearing_capacity|bearing_capacity_analysis#terzaghi', 'vertical_load'):
        'inclination factors need a horizontal load; the probe has none',
    ('calc_package|axial_pile_package', 'toe_stress_limit'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|bearing_capacity_package', 'vertical_load'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|bearing_capacity_package#terzaghi', 'vertical_load'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|downdrag_package', 'fill_unit_weight'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|drilled_shaft_package', 'beta_method'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|html_to_pdf', 'allow_unembeddable'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|lateral_pile_package', 'Q'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|lateral_pile_package', 'head_condition'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|lateral_pile_package', 'max_iterations'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|lateral_pile_package', 'pile_E'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|lateral_pile_package', 'stickup'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|lateral_pile_package', 'tolerance'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|retaining_wall_package', 'has_shear_key'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|retaining_wall_package#surcharge', 'has_shear_key'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|retaining_wall_package#surcharge', 'surcharge_in_bearing'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|sheet_pile_package', 'pressure_method'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|slope_report_package', 'time_budget_s'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|slope_report_package', 'tol'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|slope_stability_package', 'FOS_required'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'damping_model'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'damping_side'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'damping_toe'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'efficiency'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'helmet_weight'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'pile_E'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'pile_unit_weight'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'quake_side'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'quake_toe'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'segment_length'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('calc_package|wave_equation_package', 'skin_fraction'):
        'the package returns its file and headline values; the input acts in the printed calculation',
    ('downdrag|downdrag_analysis', 'fill_unit_weight'):
        'the probe places no fill',
    ('downdrag|downdrag_method_comparison', 'bearing_nu'):
        'only PILENEG uses it, and the probe gives too little for PILENEG (skipped)',
    ('drilled_shaft|capacity_vs_depth', 'beta_method'):
        'the probe has no cohesionless side layer',
    ('drilled_shaft|capacity_vs_depth', 'casing_depth'):
        '0.5 m of casing lies inside the 1.5 m top exclusion zone',
    ('drilled_shaft|capacity_vs_depth', 'socket_length'):
        'the probe has no rock layer',
    ('drilled_shaft|drilled_shaft_capacity', 'beta_method'):
        'the probe has no cohesionless side layer',
    ('drilled_shaft|drilled_shaft_capacity', 'casing_depth'):
        '0.5 m of casing lies inside the 1.5 m top exclusion zone',
    ('drilled_shaft|drilled_shaft_capacity', 'socket_length'):
        'the probe has no rock layer',
    ('drilled_shaft|lrfd_capacity', 'beta_method'):
        'the probe has no cohesionless side layer',
    ('drilled_shaft|lrfd_capacity', 'casing_depth'):
        '0.5 m of casing lies inside the 1.5 m top exclusion zone',
    ('drilled_shaft|lrfd_capacity', 'socket_length'):
        'the probe has no rock layer',
    ('fem2d|fem2d_consolidation', 'gamma_w'):
        'the probe has no water',
    ('fem2d|fem2d_consolidation', 'gwt'):
        "the probe's soil is weightless (gamma 0), so the water level moves no stress",
    ('fem2d|fem2d_excavation', 'gamma_w'):
        'the probe has no water',
    ('fem2d|fem2d_excavation', 'max_iter'):
        'a solver setting; the probe converges well inside it',
    ('fem2d|fem2d_local_fos', 'blowup_factor'):
        "a failure criterion the probe's SRM run does not reach",
    ('fem2d|fem2d_local_fos', 'gamma_w'):
        'the probe has no water',
    ('fem2d|fem2d_slope_srm', 'blowup_factor'):
        "a failure criterion the probe's SRM run does not reach",
    ('fem2d|fem2d_slope_srm', 'gamma_w'):
        'the probe has no water',
    ('fem2d|fem2d_staged', 'gamma_w'):
        'the probe has no water',
    ('ground_improvement|feasibility', 'allowable_settlement_mm'):
        'the probe states no bearing or settlement problem, only a soil type',
    ('ground_improvement|feasibility', 'current_bearing_kPa'):
        'the probe states no bearing or settlement problem, only a soil type',
    ('ground_improvement|feasibility', 'depth_to_top_m'):
        'the probe states no bearing or settlement problem, only a soil type',
    ('ground_improvement|feasibility', 'predicted_settlement_mm'):
        'the probe states no bearing or settlement problem, only a soil type',
    ('ground_improvement|feasibility', 'required_bearing_kPa'):
        'the probe states no bearing or settlement problem, only a soil type',
    ('gstools|conditional_simulation', 'bounds_mode'):
        'acts on values outside the bounds; the probe has none',
    ('lateral_pile|composite_section_ei', 'E_bar'):
        "the probe's section has no reinforcing ring",
    ('lateral_pile|composite_section_ei', 'bar_diameter'):
        "the probe's section has no reinforcing ring",
    ('lateral_pile|composite_section_ei', 'n_bars'):
        "the probe's section has no reinforcing ring",
    ('lateral_pile|get_py_curve#SandAPI', 'loading'):
        "the probe's depth is past where cyclic and static curves meet (z/D > 5)",
    ('lateral_pile|get_py_curve#SandReese', 'loading'):
        "the probe's depth is past where cyclic and static curves meet (z/D > 5)",
    ('lateral_pile|get_py_curve#WeakRock', 'loading'):
        "the probe's depth is past where cyclic and static curves meet (z/D > 5)",
    ('lateral_pile|get_py_curve#WeakRock', 'rock_top'):
        "the probe's depth is past 3D below the rock surface, where p_ur no longer varies",
    ('lateral_pile|lateral_pile_analysis', 'Q'):
        'the probe applies no head load, so the pile does not move',
    ('lateral_pile|lateral_pile_analysis', 'head_condition'):
        'the probe applies no head load, so the pile does not move',
    ('lateral_pile|lateral_pile_analysis', 'max_iterations'):
        'a solver setting; the probe converges well inside it',
    ('lateral_pile|lateral_pile_analysis', 'pile_E'):
        'the probe applies no head load, so the pile does not move',
    ('lateral_pile|lateral_pile_analysis', 'tolerance'):
        'a solver setting; the probe converges well inside it',
    ('liquefaction|liquefaction_analysis', 'a_ratio'):
        'corrects q_c with the pore pressure u2, which the probe does not give',
    ('liquepy|cpt_liquefaction', 'a_ratio'):
        'corrects q_c with the pore pressure u2, which the probe does not give',
    ('liquepy|field_correlations', 'a_ratio'):
        'corrects q_c with the pore pressure u2, which the probe does not give',
    ('pavement_design|design_traffic_esals', 'pavement_type'):
        'the probe gives truck factors, so no LEF is looked up',
    ('pavement_design|design_traffic_esals', 'pt'):
        'the probe gives truck factors, so no LEF is looked up',
    ('pavement_design|design_traffic_esals', 'sn'):
        'the probe gives truck factors, so no LEF is looked up',
    ('pavement_design|rigid_pavement_design', 'thickness_increment_in'):
        'D = 8.985 in rounds up to 9.0 in at every increment tried',
    ('pynite|continuous_beam', 'A'):
        'a continuous beam under transverse load carries no axial force',
    ('pystrata|eql_site_response', 'max_freq_hz'):
        "the probe's layers are already thinner than the wavelength rule asks",
    ('pystrata|eql_site_response', 'max_iterations'):
        'a solver setting; the probe converges well inside it',
    ('pystrata|eql_site_response', 'tolerance'):
        'a solver setting; the probe converges well inside it',
    ('pystrata|eql_site_response', 'wave_frac'):
        "the probe's layers are already thinner than the wavelength rule asks",
    ('pystrata|linear_site_response', 'max_freq_hz'):
        "the probe's layers are already thinner than the wavelength rule asks",
    ('pystrata|linear_site_response', 'wave_frac'):
        "the probe's layers are already thinner than the wavelength rule asks",
    ('reliability|form', 'max_iterations'):
        'a solver setting; the probe converges well inside it',
    ('retaining_walls|cantilever_wall', 'has_shear_key'):
        'a key needs key_depth, which the probe does not give',
    ('retaining_walls|cantilever_wall#surcharge', 'has_shear_key'):
        'a key needs key_depth, which the probe does not give',
    ('salib|morris_analysis', 'max_evaluations'):
        "a ceiling that refuses a larger design; the probe's design is under it",
    ('salib|morris_analysis', 'time_budget_s'):
        "a ceiling that refuses a longer run; the probe's run is under it",
    ('salib|sobol_analysis', 'max_evaluations'):
        "a ceiling that refuses a larger design; the probe's design is under it",
    ('salib|sobol_analysis', 'time_budget_s'):
        "a ceiling that refuses a longer run; the probe's run is under it",
    ('sheet_pile|anchored_wall', 'pressure_method'):
        'with no wall friction and a level backfill Coulomb and the log-spiral Kp equal Rankine',
    ('sheet_pile|cantilever_wall', 'pressure_method'):
        'with no wall friction and a level backfill Coulomb and the log-spiral Kp equal Rankine',
    ('slope_stability|analyze_slope', 'tol'):
        'a solver setting; the probe converges well inside it',
    ('slope_stability|compare_methods_table', 'reverse_curvature'):
        "the probe's circle does not turn back above its centre",
    ('slope_stability|compare_methods_table', 'tol'):
        'a solver setting; the probe converges well inside it',
    ('slope_stability|newmark_displacement', 'polarity'):
        "the probe's record is constant, so both polarities drive it alike",
    ('slope_stability|rapid_drawdown_fos', 'reverse_curvature'):
        "the probe's circle does not turn back above its centre",
    ('slope_stability|rapid_drawdown_fos', 'tol'):
        'a solver setting; the probe converges well inside it',
    ('slope_stability|yield_acceleration', 'reverse_curvature'):
        "the probe's circle does not turn back above its centre",
    ('slope_stability|yield_acceleration', 'tol'):
        'a solver setting; the probe converges well inside it',
    ('soe|check_bottom_blowout', 'FOS_required'):
        "the probe's FOS clears every required value tried",
    ('soe|design_ground_anchor', 'anchor_angle_deg'):
        "the unbonded length is held at the 4.5 m minimum (GEC-4) at the probe's depths",
    ('wave_equation|drivability', 'refusal_blow_count'):
        "the probe's blow counts stay below every value tried",
}

#: Inputs GM-160's sweep (2026-10-19) found changing nothing on their probe
#: and that are not yet traced: each is a follow-up, to be used, described,
#: or moved to one of the tables above with why.
TO_TRACE = {
    ('calc_package|downdrag_package', 'influence_depth_criterion'):
        "moved nothing in the package's headline; not traced",
    ('calc_package|downdrag_package', 'n_sublayers'):
        'the discretisation moved nothing; not traced',
    ('calc_package|ground_improvement_package', 'pattern'):
        'moved nothing in the package result; not traced',
    ('calc_package|ground_improvement_package', 'target_N_spt'):
        'moved nothing in the package result; not traced',
    ('calc_package|pavement_design_package', 'cbr_subbase'):
        'default None listed; not traced',
    ('calc_package|pavement_design_package', 'eac_psi'):
        'the asphalt modulus moved nothing; not traced',
    ('calc_package|settlement_package', 'q_overburden'):
        'moved nothing on the probe; not traced',
    ('calc_package|slope_report_package', 'f_interslice'):
        'moved nothing in the package result; slope_stability frozen (GM-140)',
    ('concrete_props|rc_rectangular_section', 'n_interaction_points'):
        'the result does not return the interaction diagram; not traced',
    ('downdrag|downdrag_analysis', 'n_sublayers'):
        'the discretisation moved nothing; not traced',
    ('dxf_export|export_geometry_to_dxf', 'units'):
        'the DXF content is not in the result; whether units scale or only label is not traced',
    ('dxf_export|export_to_dxf_bytes', 'units'):
        'the DXF content is not in the result; whether units scale or only label is not traced',
    ('fem2d|fem2d_foundation', 't'):
        'plane-strain results per metre: t looks to cancel; not traced',
    ('fem2d|fem2d_gravity', 't'):
        'plane-strain results per metre: t looks to cancel; not traced',
    ('fem2d|fem2d_local_fos', 't'):
        'plane-strain results per metre: t looks to cancel; not traced',
    ('fem2d|fem2d_slope_srm', 'mesh_check'):
        "the finer mesh's FOS may land only in the notes; not traced",
    ('fem2d|fem2d_slope_srm', 't'):
        'plane-strain results per metre: t looks to cancel; not traced',
    ('fem2d|fem2d_staged', 't'):
        'plane-strain results per metre: t looks to cancel; not traced',
    ('lateral_pile|get_py_curve#StiffClayAboveWT', 'loading'):
        'cyclic stiff clay above the water table matched static at z = 5 m; not traced',
    ('liquefaction|liquefaction_analysis', 's_g'):
        'specific gravity moved nothing; not traced',
    ('liquepy|cpt_liquefaction', 's_g'):
        'specific gravity moved nothing; not traced',
    ('liquepy|field_correlations', 'i_c_limit'):
        'the Ic limit moved nothing in the correlations; not traced',
    ('liquepy|field_correlations', 's_g'):
        'specific gravity moved nothing; not traced',
    ('pavement_design|compare_pavement_methods', 'eac_psi'):
        'the asphalt modulus moved nothing; not traced',
    ('pavement_design|performance_period', 'pt'):
        'terminal serviceability moved nothing in the period; not traced',
    ('salib|morris_analyze', 'seed'):
        'the confidence intervals may not resample; not traced',
    ('settlement|combined_settlement_analysis', 'immediate_method'):
        'moved nothing on the probe; not traced',
    ('settlement|combined_settlement_analysis', 'q_overburden'):
        'moved nothing on the probe; not traced',
    ('settlement|elastic_settlement', 'shape'):
        "its description says 'for reference; supply Iw'; a rectangle with no L runs as a square (GM-142 refuses that elsewhere); not traced",
    ('slope_stability|analyze_slope', 'reinforcement_force'):
        '0 -> 0.5 kN/m moved no FOS digit; slope_stability frozen (GM-140)',
    ('slope_stability|compare_methods_table', 'reinforcement_force'):
        '0 -> 0.5 kN/m moved no FOS digit; slope_stability frozen (GM-140)',
    ('slope_stability|fosm_fos', 'n_slices'):
        'FOSM moved nothing with the slice count; slope_stability frozen (GM-140)',
    ('slope_stability|fosm_fos', 'reinforcement_force'):
        '0 -> 0.5 kN/m moved no FOS digit; slope_stability frozen (GM-140)',
    ('slope_stability|rapid_drawdown_fos', 'reinforcement_force'):
        '0 -> 0.5 kN/m moved no FOS digit; slope_stability frozen (GM-140)',
    ('slope_stability|yield_acceleration', 'reinforcement_force'):
        '0 -> 0.5 kN/m moved no FOS digit; slope_stability frozen (GM-140)',
}

#: Mismatches the guard finds and leaves, each with its BACKLOG item.
#: slope_stability is frozen by the owner until the integral review.
KNOWN = {
    ("slope_stability|analyze_slope", "method"):
        "GM-140: 'bishop' is listed, but a noncircular slip_points surface "
        "runs Spencer",
}
