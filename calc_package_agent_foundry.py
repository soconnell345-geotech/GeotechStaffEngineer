"""
Calculation Package Agent - Palantir Foundry AIP Agent Studio Version.

Register these three functions as tools in AIP Agent Studio:
  1. calc_package_agent        - Generate a calculation package
  2. calc_package_list_methods - Browse available methods
  3. calc_package_describe_method - Get detailed parameter docs

FOUNDRY SETUP:
  - These functions accept and return JSON strings for LLM compatibility
  - Generates self-contained HTML calculation packages
"""

import json
import math
import numpy as np
from functions.api import function


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _clean_result(result: dict) -> dict:
    return {k: _clean_value(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Method builders — run analysis + generate calc package
# ---------------------------------------------------------------------------

def _generate_bearing_capacity_package(params: dict) -> dict:
    """Run bearing capacity analysis and generate calc package."""
    from bearing_capacity.footing import Footing
    from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
    from bearing_capacity.capacity import BearingCapacityAnalysis
    from calc_package import generate_calc_package

    footing = Footing(
        width=params["width"],
        length=params.get("length"),
        depth=params.get("depth", 0.0),
        shape=params.get("shape", "strip"),
        base_tilt=params.get("base_tilt", 0.0),
        eccentricity_B=params.get("eccentricity_B", 0.0),
        eccentricity_L=params.get("eccentricity_L", 0.0),
    )

    layer1 = SoilLayer(
        cohesion=params.get("cohesion", 0.0),
        friction_angle=params.get("friction_angle", 0.0),
        unit_weight=params["unit_weight"],
    )

    gwt_depth = params.get("gwt_depth")
    soil = BearingSoilProfile(layer1=layer1, gwt_depth=gwt_depth)

    analysis = BearingCapacityAnalysis(
        footing=footing,
        soil=soil,
        load_inclination=params.get("load_inclination", 0.0),
        ground_slope=params.get("ground_slope", 0.0),
        vertical_load=params.get("vertical_load", 0.0),
        factor_of_safety=params.get("factor_of_safety", 3.0),
        ngamma_method=params.get("ngamma_method", "vesic"),
        factor_method=params.get("factor_method", "vesic"),
    )
    result = analysis.compute()

    return _build_html_response(
        "bearing_capacity", result, analysis, params,
        analysis_type="Bearing Capacity",
        extra={"q_ultimate_kPa": round(result.q_ultimate, 1),
               "q_allowable_kPa": round(result.q_allowable, 1),
               "factor_of_safety": result.factor_of_safety},
    )


def _generate_lateral_pile_package(params: dict) -> dict:
    """Run lateral pile analysis and generate calc package."""
    from lateral_pile import Pile, LateralPileAnalysis
    from lateral_pile.soil import SoilLayer
    from lateral_pile.py_curves import (
        SoftClayMatlock, StiffClayBelowWT, StiffClayAboveWT,
        SoftClayJeanjean, SandReese, SandAPI, WeakRock,
    )
    from calc_package import generate_calc_package

    PY_MODEL_MAP = {
        "SoftClayMatlock": SoftClayMatlock,
        "StiffClayBelowWT": StiffClayBelowWT,
        "StiffClayAboveWT": StiffClayAboveWT,
        "SoftClayJeanjean": SoftClayJeanjean,
        "SandReese": SandReese,
        "SandAPI": SandAPI,
        "WeakRock": WeakRock,
    }

    pile = Pile(
        length=params["pile_length"],
        diameter=params["pile_diameter"],
        thickness=params.get("pile_thickness"),
        E=params["pile_E"],
        moment_of_inertia=params.get("moment_of_inertia"),
    )

    layers = []
    for layer_dict in params["soil_layers"]:
        model_name = layer_dict["py_model"]
        model_cls = PY_MODEL_MAP[model_name]
        model_params = {k: v for k, v in layer_dict.items()
                       if k not in ("top", "bottom", "py_model", "description")}
        py_model = model_cls(**model_params)
        layers.append(SoilLayer(
            top=layer_dict["top"],
            bottom=layer_dict["bottom"],
            py_model=py_model,
            description=layer_dict.get("description"),
        ))

    analysis = LateralPileAnalysis(pile, layers)
    result = analysis.solve(
        Vt=params.get("Vt", 0.0),
        Mt=params.get("Mt", 0.0),
        Q=params.get("Q", 0.0),
        head_condition=params.get("head_condition", "free"),
    )

    return _build_html_response(
        "lateral_pile", result, analysis, params,
        analysis_type="Lateral Pile",
        extra={"y_top_mm": round(result.y_top * 1000, 2),
               "max_moment_kNm": round(result.max_moment, 1),
               "converged": result.converged},
    )


def _generate_slope_stability_package(params: dict) -> dict:
    """Run slope stability analysis and generate calc package."""
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.analysis import analyze_slope
    from calc_package import generate_calc_package

    soil_layers = []
    for ld in params["soil_layers"]:
        soil_layers.append(SlopeSoilLayer(
            name=ld["name"],
            top_elevation=ld["top_elevation"],
            bottom_elevation=ld["bottom_elevation"],
            gamma=ld["gamma"],
            gamma_sat=ld.get("gamma_sat"),
            phi=ld.get("phi", 0.0),
            c_prime=ld.get("c_prime", 0.0),
            cu=ld.get("cu", 0.0),
            analysis_mode=ld.get("analysis_mode", "drained"),
        ))

    surface_points = [tuple(p) for p in params["surface_points"]]
    gwt_points = None
    if "gwt_points" in params and params["gwt_points"]:
        gwt_points = [tuple(p) for p in params["gwt_points"]]

    geom = SlopeGeometry(
        surface_points=surface_points,
        soil_layers=soil_layers,
        gwt_points=gwt_points,
        surcharge=params.get("surcharge", 0.0),
        kh=params.get("kh", 0.0),
    )

    result = analyze_slope(
        geom,
        xc=params["xc"],
        yc=params["yc"],
        radius=params["radius"],
        method=params.get("method", "bishop"),
        n_slices=params.get("n_slices", 30),
        FOS_required=params.get("FOS_required", 1.5),
        include_slice_data=True,
        compare_methods=params.get("compare_methods", False),
    )

    analysis_dict = {"geom": geom}
    return _build_html_response(
        "slope_stability", result, analysis_dict, params,
        analysis_type="Slope Stability",
        extra={"FOS": round(result.FOS, 3), "method": result.method,
               "is_stable": result.is_stable},
    )


def _generate_settlement_package(params: dict) -> dict:
    """Run settlement analysis and generate calc package."""
    from settlement import SettlementAnalysis, ConsolidationLayer, SchmertmannLayer
    from calc_package import generate_calc_package

    consol_layers = None
    if "consolidation_layers" in params and params["consolidation_layers"]:
        consol_layers = []
        for ld in params["consolidation_layers"]:
            consol_layers.append(ConsolidationLayer(
                thickness=ld["thickness"],
                depth_to_center=ld["depth_to_center"],
                e0=ld["e0"],
                Cc=ld["Cc"],
                Cr=ld["Cr"],
                sigma_v0=ld["sigma_v0"],
                sigma_p=ld.get("sigma_p"),
                description=ld.get("description", ""),
            ))

    schm_layers = None
    if "schmertmann_layers" in params and params["schmertmann_layers"]:
        schm_layers = []
        for ld in params["schmertmann_layers"]:
            schm_layers.append(SchmertmannLayer(
                depth_top=ld["depth_top"],
                depth_bottom=ld["depth_bottom"],
                Es=ld["Es"],
            ))

    analysis = SettlementAnalysis(
        q_applied=params.get("q_applied", 0.0),
        q_overburden=params.get("q_overburden", 0.0),
        B=params.get("B", 1.0),
        L=params.get("L", 1.0),
        footing_shape=params.get("footing_shape", "square"),
        stress_method=params.get("stress_method", "2:1"),
        immediate_method=params.get("immediate_method", "elastic"),
        Es_immediate=params.get("Es_immediate"),
        nu=params.get("nu", 0.3),
        schmertmann_layers=schm_layers,
        time_years_schmertmann=params.get("time_years_schmertmann", 0.0),
        consolidation_layers=consol_layers,
        cv=params.get("cv"),
        Hdr=params.get("Hdr"),
        drainage=params.get("drainage", "double"),
        C_alpha=params.get("C_alpha"),
        e0_secondary=params.get("e0_secondary", 1.0),
        t_secondary=params.get("t_secondary", 0.0),
    )
    result = analysis.compute()

    return _build_html_response(
        "settlement", result, analysis, params,
        analysis_type="Settlement",
        extra={"total_settlement_mm": round(result.total * 1000, 2),
               "immediate_mm": round(result.immediate * 1000, 2),
               "consolidation_mm": round(result.consolidation * 1000, 2),
               "secondary_mm": round(result.secondary * 1000, 2)},
    )


def _generate_axial_pile_package(params: dict) -> dict:
    """Run axial pile capacity analysis and generate calc package."""
    from axial_pile import AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, PileSection
    from calc_package import generate_calc_package

    pile = PileSection(
        name=params.get("pile_name", "Custom Pile"),
        pile_type=params.get("pile_type", "h_pile"),
        area=params["pile_area"],
        perimeter=params["pile_perimeter"],
        tip_area=params["pile_tip_area"],
        width=params["pile_width"],
        depth=params.get("pile_depth"),
    )

    layers = []
    for ld in params["soil_layers"]:
        layers.append(AxialSoilLayer(
            soil_type=ld["soil_type"],
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            friction_angle=ld.get("friction_angle", 0.0),
            cohesion=ld.get("cohesion", 0.0),
            delta_phi_ratio=ld.get("delta_phi_ratio"),
            description=ld.get("description", ""),
        ))
    soil = AxialSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))

    analysis = AxialPileAnalysis(
        pile=pile, soil=soil,
        pile_length=params["pile_length"],
        factor_of_safety=params.get("factor_of_safety", 2.5),
        method=params.get("method", "auto"),
    )
    result = analysis.compute()

    return _build_html_response(
        "axial_pile", result, analysis, params,
        analysis_type="Axial Pile Capacity",
        extra={"Q_ultimate_kN": round(result.Q_ultimate, 1),
               "Q_allowable_kN": round(result.Q_allowable, 1),
               "Q_skin_kN": round(result.Q_skin, 1),
               "Q_tip_kN": round(result.Q_tip, 1)},
    )


def _generate_drilled_shaft_package(params: dict) -> dict:
    """Run drilled shaft analysis and generate calc package."""
    from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis
    from calc_package import generate_calc_package

    shaft = DrillShaft(
        diameter=params["diameter"],
        length=params["length"],
        socket_diameter=params.get("socket_diameter"),
        socket_length=params.get("socket_length", 0.0),
        bell_diameter=params.get("bell_diameter"),
        casing_depth=params.get("casing_depth", 0.0),
        concrete_fc=params.get("concrete_fc", 28000.0),
    )

    layers = []
    for ld in params["soil_layers"]:
        layers.append(ShaftSoilLayer(
            soil_type=ld["soil_type"],
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            cu=ld.get("cu", 0.0),
            phi=ld.get("phi", 0.0),
            N60=ld.get("N60", 0),
            qu=ld.get("qu", 0.0),
            RQD=ld.get("RQD", 100.0),
            description=ld.get("description", ""),
        ))
    soil = ShaftSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))

    analysis = DrillShaftAnalysis(
        shaft=shaft, soil=soil,
        factor_of_safety=params.get("factor_of_safety", 2.5),
    )
    result = analysis.compute()

    return _build_html_response(
        "drilled_shaft", result, analysis, params,
        analysis_type="Drilled Shaft Capacity",
        extra={"Q_ultimate_kN": round(result.Q_ultimate, 1),
               "Q_allowable_kN": round(result.Q_allowable, 1),
               "Q_skin_kN": round(result.Q_skin, 1),
               "Q_tip_kN": round(result.Q_tip, 1)},
    )


def _generate_downdrag_package(params: dict) -> dict:
    """Run downdrag analysis and generate calc package."""
    from downdrag import DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile
    from calc_package import generate_calc_package

    layers = []
    for ld in params["soil_layers"]:
        layers.append(DowndragSoilLayer(
            soil_type=ld["soil_type"],
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            cu=ld.get("cu", 0.0),
            phi=ld.get("phi", 0.0),
            settling=ld.get("settling", False),
            alpha=ld.get("alpha"),
            beta=ld.get("beta"),
            Cc=ld.get("Cc"),
            e0=ld.get("e0"),
            sigma_p=ld.get("sigma_p"),
            description=ld.get("description", ""),
        ))
    soil = DowndragSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth", 0.0))

    analysis = DowndragAnalysis(
        pile_length=params["pile_length"],
        pile_diameter=params["pile_diameter"],
        pile_E=params["pile_E"],
        pile_unit_weight=params.get("pile_unit_weight", 78.5),
        Q_dead=params["Q_dead"],
        soil=soil,
        fill_thickness=params.get("fill_thickness", 0.0),
        fill_unit_weight=params.get("fill_unit_weight", 20.0),
        gw_drawdown=params.get("gw_drawdown", 0.0),
        structural_capacity=params.get("structural_capacity"),
        allowable_settlement=params.get("allowable_settlement"),
        Nt=params.get("Nt"),
    )
    result = analysis.compute()

    return _build_html_response(
        "downdrag", result, analysis, params,
        analysis_type="Downdrag",
        extra={"neutral_plane_m": round(result.neutral_plane_depth, 2),
               "dragload_kN": round(result.dragload, 1),
               "max_pile_load_kN": round(result.max_pile_load, 1),
               "pile_settlement_mm": round(result.pile_settlement * 1000, 2)},
    )


def _generate_seismic_package(params: dict) -> dict:
    """Run seismic geotechnical analysis and generate calc package."""
    from seismic_geotech import (
        classify_site, site_coefficients, mononobe_okabe_KAE,
        seismic_earth_pressure, evaluate_liquefaction,
    )
    from seismic_geotech.results import (
        SiteClassResult, SeismicEarthPressureResult, LiquefactionResult,
    )
    from calc_package import generate_calc_package

    analysis_type = params.get("analysis_type", "site_classification")
    analysis_dict = dict(params)

    if analysis_type == "site_classification":
        site_class = classify_site(
            vs30=params.get("vs30"),
            n_bar=params.get("n_bar"),
            su_bar=params.get("su_bar"),
        )
        result = site_coefficients(site_class, Ss=params["Ss"], S1=params["S1"])
    elif analysis_type == "seismic_earth_pressure":
        KAE = mononobe_okabe_KAE(
            phi_deg=params["phi"], delta_deg=params.get("delta", params["phi"] * 2 / 3),
            kh=params["kh"], kv=params.get("kv", 0.0),
        )
        result_dict = seismic_earth_pressure(
            gamma=params["gamma"], H=params["H"], KAE=KAE,
            KA=params.get("KA", 0.333),
        )
        result = SeismicEarthPressureResult(
            KAE=KAE, KPE=0.0,
            PAE_total=result_dict["PAE_total"],
            PA_static=result_dict["PA_static"],
            delta_PAE=result_dict["delta_PAE"],
            height_of_application=result_dict["height_of_application"],
            phi=params["phi"], delta=params.get("delta", params["phi"] * 2 / 3),
            kh=params["kh"], kv=params.get("kv", 0.0),
        )
    elif analysis_type == "liquefaction":
        layer_results = evaluate_liquefaction(
            layer_depths=params["layer_depths"],
            layer_N160=params["layer_N160"],
            layer_FC=params["layer_FC"],
            layer_gamma=params["layer_gamma"],
            amax_g=params["amax_g"],
            gwt_depth=params["gwt_depth"],
            M=params.get("M", 7.5),
        )
        result = LiquefactionResult(
            layer_results=layer_results,
            amax_g=params["amax_g"],
            magnitude=params.get("M", 7.5),
            gwt_depth=params["gwt_depth"],
        )
    else:
        raise ValueError(f"Unknown analysis_type: {analysis_type}")

    return _build_html_response(
        "seismic_geotech", result, analysis_dict, params,
        analysis_type="Seismic Geotechnical",
        extra={"seismic_analysis_type": analysis_type},
    )


def _generate_retaining_wall_package(params: dict) -> dict:
    """Run retaining wall analysis and generate calc package."""
    from retaining_walls import (
        CantileverWallGeometry, MSEWallGeometry, Reinforcement,
        analyze_cantilever_wall, analyze_mse_wall,
    )
    from calc_package import generate_calc_package

    wall_type = params.get("wall_type", "cantilever")
    analysis_dict = dict(params)

    if wall_type == "cantilever":
        geom = CantileverWallGeometry(
            wall_height=params["wall_height"],
            base_width=params.get("base_width"),
            toe_length=params.get("toe_length"),
            stem_thickness_top=params.get("stem_thickness_top", 0.3),
            stem_thickness_base=params.get("stem_thickness_base"),
            base_thickness=params.get("base_thickness", 0.6),
            backfill_slope=params.get("backfill_slope", 0.0),
            surcharge=params.get("surcharge", 0.0),
        )
        result = analyze_cantilever_wall(
            geom=geom,
            gamma_backfill=params["gamma_backfill"],
            phi_backfill=params["phi_backfill"],
            c_backfill=params.get("c_backfill", 0.0),
            phi_foundation=params.get("phi_foundation"),
            c_foundation=params.get("c_foundation", 0.0),
            q_allowable=params.get("q_allowable"),
            gamma_concrete=params.get("gamma_concrete", 24.0),
            FOS_sliding=params.get("FOS_sliding", 1.5),
            FOS_overturning=params.get("FOS_overturning", 2.0),
            pressure_method=params.get("pressure_method", "rankine"),
        )
        analysis_dict["geom"] = geom
        extra = {"FOS_sliding": round(result.FOS_sliding, 2),
                 "FOS_overturning": round(result.FOS_overturning, 2)}
    else:
        raise ValueError("MSE wall calc package not yet implemented via Foundry agent")

    return _build_html_response(
        "retaining_walls", result, analysis_dict, params,
        analysis_type="Retaining Wall",
        extra=extra,
    )


def _generate_ground_improvement_package(params: dict) -> dict:
    """Run ground improvement analysis and generate calc package."""
    from ground_improvement import (
        analyze_wick_drains, analyze_aggregate_piers,
        analyze_surcharge_preloading, analyze_vibro_compaction,
    )
    from calc_package import generate_calc_package

    method = params.get("method", "wick_drains")
    analysis_dict = dict(params)

    if method == "wick_drains":
        result = analyze_wick_drains(
            spacing=params["spacing"],
            ch=params["ch"],
            cv=params["cv"],
            Hdr=params["Hdr"],
            time=params["time"],
            dw=params.get("dw", 0.066),
            pattern=params.get("pattern", "triangular"),
            smear_ratio=params.get("smear_ratio", 2.0),
            kh_ks_ratio=params.get("kh_ks_ratio", 2.0),
        )
    elif method == "aggregate_piers":
        result = analyze_aggregate_piers(
            diameter=params["diameter"],
            spacing=params["spacing"],
            length=params["length"],
            E_pier=params.get("E_pier", 50000.0),
            E_soil=params.get("E_soil", 5000.0),
            phi_pier=params.get("phi_pier", 45.0),
            applied_stress=params.get("applied_stress", 100.0),
        )
    elif method == "surcharge":
        result = analyze_surcharge_preloading(
            H_clay=params["H_clay"],
            cv=params["cv"],
            surcharge=params["surcharge"],
            sigma_v0=params["sigma_v0"],
            Cc=params["Cc"],
            e0=params["e0"],
            sigma_p=params.get("sigma_p"),
            target_consolidation=params.get("target_consolidation", 90.0),
        )
    elif method == "vibro":
        result = analyze_vibro_compaction(
            spacing=params["spacing"],
            N1_before=params["N1_before"],
            depth=params["depth"],
        )
    else:
        raise ValueError(f"Unknown ground improvement method: {method}")

    return _build_html_response(
        "ground_improvement", result, analysis_dict, params,
        analysis_type="Ground Improvement",
        extra={"method": method},
    )


def _generate_wave_equation_package(params: dict) -> dict:
    """Run wave equation analysis and generate calc package."""
    from wave_equation import (
        get_hammer, make_cushion_from_properties, Cushion,
        discretize_pile, generate_bearing_graph,
    )
    from calc_package import generate_calc_package

    hammer = get_hammer(params["hammer_name"])
    if "cushion_stiffness" in params:
        cushion = Cushion(stiffness=params["cushion_stiffness"],
                         cor=params.get("cushion_cor", 0.8))
    else:
        cushion = make_cushion_from_properties(
            area=params["cushion_area"],
            thickness=params["cushion_thickness"],
            elastic_modulus=params["cushion_E"],
            cor=params.get("cushion_cor", 0.8),
        )
    pile = discretize_pile(
        length=params["pile_length"],
        area=params["pile_area"],
        elastic_modulus=params["pile_E"],
        segment_length=params.get("segment_length", 1.0),
        unit_weight_material=params.get("pile_unit_weight", 78.5),
    )
    result = generate_bearing_graph(
        hammer=hammer, cushion=cushion, pile=pile,
        skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", 0.5),
        R_min=params.get("R_min", 200.0),
        R_max=params.get("R_max", 2000.0),
        R_step=params.get("R_step", 200.0),
    )

    analysis_dict = {"hammer": hammer, "cushion": cushion, "pile": pile}
    return _build_html_response(
        "wave_equation", result, analysis_dict, params,
        analysis_type="Wave Equation",
        extra={"n_points": len(result.blow_counts)},
    )


def _generate_pile_group_package(params: dict) -> dict:
    """Run pile group analysis and generate calc package."""
    from pile_group import (
        GroupPile, create_rectangular_layout, GroupLoad,
        analyze_vertical_group_simple, analyze_group_6dof,
    )
    from calc_package import generate_calc_package

    if "layout" in params and params["layout"] == "rectangular":
        piles = create_rectangular_layout(
            n_rows=params["n_rows"],
            n_cols=params["n_cols"],
            spacing_x=params["spacing_x"],
            spacing_y=params["spacing_y"],
        )
    else:
        piles = []
        for pd in params["piles"]:
            piles.append(GroupPile(
                x=pd["x"], y=pd["y"],
                axial_capacity_compression=pd.get("axial_capacity_compression"),
                axial_capacity_tension=pd.get("axial_capacity_tension"),
                label=pd.get("label", ""),
            ))

    load = GroupLoad(
        Vx=params.get("Vx", 0.0), Vy=params.get("Vy", 0.0),
        Vz=params.get("Vz", 0.0),
        Mx=params.get("Mx", 0.0), My=params.get("My", 0.0),
        Mz=params.get("Mz", 0.0),
    )

    if params.get("method_6dof", False):
        result = analyze_group_6dof(piles=piles, load=load)
    else:
        result = analyze_vertical_group_simple(piles=piles, load=load)

    analysis_dict = {
        "piles": piles, "load": load,
        "pile_diameter": params.get("pile_diameter", 0.3),
        "pile_spacing": params.get("pile_spacing", 1.5),
    }
    return _build_html_response(
        "pile_group", result, analysis_dict, params,
        analysis_type="Pile Group",
        extra={"n_piles": result.n_piles,
               "group_efficiency": round(result.group_efficiency, 3)
               if hasattr(result, "group_efficiency") else None},
    )


def _generate_sheet_pile_package(params: dict) -> dict:
    """Run sheet pile wall analysis and generate calc package."""
    from sheet_pile import WallSoilLayer, analyze_cantilever, analyze_anchored
    from calc_package import generate_calc_package

    layers = []
    for ld in params["soil_layers"]:
        layers.append(WallSoilLayer(
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            friction_angle=ld.get("friction_angle", 30.0),
            cohesion=ld.get("cohesion", 0.0),
            description=ld.get("description", ""),
        ))

    wall_type = params.get("wall_type", "cantilever")
    analysis_dict = dict(params)
    analysis_dict["soil_layers"] = layers

    if wall_type == "cantilever":
        result = analyze_cantilever(
            soil_layers=layers,
            excavation_depth=params["excavation_depth"],
            gwt_depth_active=params.get("gwt_depth_active"),
            gwt_depth_passive=params.get("gwt_depth_passive"),
            surcharge=params.get("surcharge", 0.0),
            FOS_passive=params.get("FOS_passive", 1.5),
            pressure_method=params.get("pressure_method", "rankine"),
        )
        extra = {"embedment_m": round(result.embedment_depth, 2),
                 "max_moment_kNm_per_m": round(result.max_moment, 1)}
    else:
        result = analyze_anchored(
            soil_layers=layers,
            excavation_depth=params["excavation_depth"],
            anchor_depth=params["anchor_depth"],
            gwt_depth_active=params.get("gwt_depth_active"),
            gwt_depth_passive=params.get("gwt_depth_passive"),
            surcharge=params.get("surcharge", 0.0),
            FOS_passive=params.get("FOS_passive", 1.5),
            pressure_method=params.get("pressure_method", "rankine"),
        )
        extra = {"embedment_m": round(result.embedment_depth, 2),
                 "anchor_force_kN_per_m": round(result.anchor_force, 1),
                 "max_moment_kNm_per_m": round(result.max_moment, 1)}

    return _build_html_response(
        "sheet_pile", result, analysis_dict, params,
        analysis_type="Sheet Pile Wall",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Common HTML response builder
# ---------------------------------------------------------------------------

def _build_html_response(module, result, analysis, params, analysis_type, extra=None):
    """Build standardized response dict with HTML calc package."""
    from calc_package import generate_calc_package

    output_path = params.get("output_path")
    html = generate_calc_package(
        module=module,
        result=result,
        analysis=analysis,
        project_name=params.get("project_name", "Project"),
        project_number=params.get("project_number", ""),
        engineer=params.get("engineer", ""),
        checker=params.get("checker", ""),
        company=params.get("company", ""),
        output_path=output_path,
    )

    response = {
        "status": "success",
        "analysis_type": analysis_type,
        "html_length": len(html),
        "output_path": output_path,
        "html": html if params.get("return_html", False) else None,
    }
    if extra:
        response.update(extra)
    return response


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "bearing_capacity_package": _generate_bearing_capacity_package,
    "lateral_pile_package": _generate_lateral_pile_package,
    "slope_stability_package": _generate_slope_stability_package,
    "settlement_package": _generate_settlement_package,
    "axial_pile_package": _generate_axial_pile_package,
    "drilled_shaft_package": _generate_drilled_shaft_package,
    "downdrag_package": _generate_downdrag_package,
    "seismic_package": _generate_seismic_package,
    "retaining_wall_package": _generate_retaining_wall_package,
    "ground_improvement_package": _generate_ground_improvement_package,
    "wave_equation_package": _generate_wave_equation_package,
    "pile_group_package": _generate_pile_group_package,
    "sheet_pile_package": _generate_sheet_pile_package,
}

METHOD_INFO = {
    "bearing_capacity_package": {
        "category": "calc_package",
        "description": (
            "Run bearing capacity analysis and generate a Mathcad-style "
            "HTML calculation package with equations, figures, and checks."
        ),
        "params": {
            "width": "Footing width B (m) [REQUIRED]",
            "unit_weight": "Soil unit weight (kN/m3) [REQUIRED]",
            "friction_angle": "Friction angle (deg), default 0",
            "cohesion": "Cohesion (kPa), default 0",
            "depth": "Embedment depth Df (m), default 0",
            "shape": "'strip','square','rectangular','circular', default 'strip'",
            "length": "Footing length L (m), required for rectangular",
            "factor_of_safety": "FS, default 3.0",
            "ngamma_method": "'vesic','meyerhof','hansen', default 'vesic'",
            "gwt_depth": "Groundwater depth (m), optional",
            "project_name": "Project name for header",
            "engineer": "Engineer name for header",
            "output_path": "File path to save HTML, optional",
            "return_html": "If true, include HTML string in response",
        },
        "returns": "dict with q_ultimate_kPa, q_allowable_kPa, html_length, output_path",
    },
    "lateral_pile_package": {
        "category": "calc_package",
        "description": (
            "Run lateral pile analysis and generate an LPILE-style "
            "HTML calculation package with deflection/moment/shear plots."
        ),
        "params": {
            "pile_length": "Pile length (m) [REQUIRED]",
            "pile_diameter": "Pile diameter (m) [REQUIRED]",
            "pile_E": "Young's modulus (kPa) [REQUIRED]",
            "pile_thickness": "Wall thickness (m), optional (solid if omitted)",
            "soil_layers": (
                "List of layer dicts, each with: top, bottom, py_model "
                "(model name string), plus model-specific params "
                "(c, gamma, eps50, J, phi, k, etc.) [REQUIRED]"
            ),
            "Vt": "Lateral load at head (kN), default 0",
            "Mt": "Moment at head (kN-m), default 0",
            "Q": "Axial load (kN), default 0",
            "head_condition": "'free' or 'fixed', default 'free'",
            "project_name": "Project name for header",
            "engineer": "Engineer name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with y_top_mm, max_moment_kNm, converged, html_length",
    },
    "slope_stability_package": {
        "category": "calc_package",
        "description": (
            "Run slope stability analysis and generate a SLIDE-style "
            "HTML calculation package with cross-section and slip circle plots."
        ),
        "params": {
            "surface_points": "List of [x, z] ground surface coordinates [REQUIRED]",
            "soil_layers": (
                "List of layer dicts with: name, top_elevation, bottom_elevation, "
                "gamma, phi, c_prime (or cu, analysis_mode='undrained') [REQUIRED]"
            ),
            "xc": "Circle center x (m) [REQUIRED]",
            "yc": "Circle center y/elevation (m) [REQUIRED]",
            "radius": "Circle radius (m) [REQUIRED]",
            "method": "'bishop','fellenius','spencer', default 'bishop'",
            "n_slices": "Number of slices, default 30",
            "FOS_required": "Required FOS, default 1.5",
            "compare_methods": "If true, compute FOS for all methods",
            "gwt_points": "List of [x, z] GWT coordinates, optional",
            "kh": "Seismic coefficient, default 0",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with FOS, method, is_stable, html_length",
    },
    "settlement_package": {
        "category": "calc_package",
        "description": (
            "Run settlement analysis (immediate + consolidation + secondary) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "q_applied": "Applied bearing pressure (kPa) [REQUIRED]",
            "q_overburden": "Overburden pressure at footing base (kPa)",
            "B": "Footing width (m), default 1",
            "L": "Footing length (m), default 1",
            "footing_shape": "'square','rectangular','circular', default 'square'",
            "stress_method": "'2:1','boussinesq','westergaard', default '2:1'",
            "immediate_method": "'elastic' or 'schmertmann', default 'elastic'",
            "Es_immediate": "Elastic modulus for immediate method (kPa)",
            "consolidation_layers": (
                "List of dicts: thickness, depth_to_center, e0, Cc, Cr, "
                "sigma_v0, sigma_p, description"
            ),
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with total_settlement_mm, immediate_mm, consolidation_mm, html_length",
    },
    "axial_pile_package": {
        "category": "calc_package",
        "description": (
            "Run axial pile capacity analysis (Nordlund/Tomlinson/Beta) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "pile_name": "Pile designation string",
            "pile_type": "'h_pile','pipe_pile','concrete_pile'",
            "pile_area": "Pile cross-section area (m2) [REQUIRED]",
            "pile_perimeter": "Pile perimeter (m) [REQUIRED]",
            "pile_tip_area": "Pile tip area (m2) [REQUIRED]",
            "pile_width": "Pile width/diameter (m) [REQUIRED]",
            "pile_length": "Embedded pile length (m) [REQUIRED]",
            "soil_layers": (
                "List of dicts: soil_type ('cohesionless'/'cohesive'), thickness, "
                "unit_weight, friction_angle, cohesion, description [REQUIRED]"
            ),
            "factor_of_safety": "FS, default 2.5",
            "method": "'auto','nordlund','tomlinson','beta', default 'auto'",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with Q_ultimate_kN, Q_allowable_kN, Q_skin_kN, Q_tip_kN, html_length",
    },
    "drilled_shaft_package": {
        "category": "calc_package",
        "description": (
            "Run drilled shaft capacity analysis (GEC-10 alpha/beta/rock) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "diameter": "Shaft diameter (m) [REQUIRED]",
            "length": "Shaft length (m) [REQUIRED]",
            "soil_layers": (
                "List of dicts: soil_type ('cohesive'/'cohesionless'/'rock'), thickness, "
                "unit_weight, cu, phi, N60, qu, RQD, description [REQUIRED]"
            ),
            "gwt_depth": "Groundwater depth (m), optional",
            "factor_of_safety": "FS, default 2.5",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with Q_ultimate_kN, Q_allowable_kN, Q_skin_kN, Q_tip_kN, html_length",
    },
    "downdrag_package": {
        "category": "calc_package",
        "description": (
            "Run downdrag analysis (Fellenius neutral plane method) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "pile_length": "Pile length (m) [REQUIRED]",
            "pile_diameter": "Pile diameter (m) [REQUIRED]",
            "pile_E": "Pile Young's modulus (kPa) [REQUIRED]",
            "Q_dead": "Dead load at pile head (kN) [REQUIRED]",
            "soil_layers": (
                "List of dicts: soil_type, thickness, unit_weight, cu, phi, "
                "settling (bool), alpha, beta, Cc, e0, sigma_p [REQUIRED]"
            ),
            "fill_thickness": "Fill thickness (m), default 0",
            "fill_unit_weight": "Fill unit weight (kN/m3), default 20",
            "structural_capacity": "Factored structural capacity (kN), optional",
            "allowable_settlement": "Allowable settlement (m), optional",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with neutral_plane_m, dragload_kN, max_pile_load_kN, html_length",
    },
    "seismic_package": {
        "category": "calc_package",
        "description": (
            "Run seismic geotechnical analysis (site class, M-O pressures, "
            "or liquefaction) and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "analysis_type": (
                "'site_classification', 'seismic_earth_pressure', or "
                "'liquefaction' [REQUIRED]"
            ),
            "vs30": "Average shear wave velocity top 30m (m/s) — site classification",
            "Ss": "Short-period spectral acceleration (g) — site classification",
            "S1": "1-second spectral acceleration (g) — site classification",
            "phi": "Backfill friction angle (deg) — seismic earth pressure",
            "kh": "Horizontal seismic coefficient — seismic earth pressure",
            "gamma": "Soil unit weight (kN/m3) — seismic earth pressure",
            "H": "Wall height (m) — seismic earth pressure",
            "layer_depths": "List of layer midpoint depths (m) — liquefaction",
            "layer_N160": "List of corrected SPT N values — liquefaction",
            "layer_FC": "List of fines content (%) — liquefaction",
            "layer_gamma": "List of unit weights (kN/m3) — liquefaction",
            "amax_g": "Peak ground acceleration (g) — liquefaction",
            "gwt_depth": "Groundwater depth (m) — liquefaction",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with seismic_analysis_type, html_length",
    },
    "retaining_wall_package": {
        "category": "calc_package",
        "description": (
            "Run retaining wall stability analysis (cantilever or MSE) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "wall_type": "'cantilever' or 'mse', default 'cantilever'",
            "wall_height": "Wall height H (m) [REQUIRED]",
            "gamma_backfill": "Backfill unit weight (kN/m3) [REQUIRED]",
            "phi_backfill": "Backfill friction angle (deg) [REQUIRED]",
            "toe_length": "Toe length (m), optional",
            "base_thickness": "Base thickness (m), default 0.6",
            "surcharge": "Surcharge pressure (kPa), default 0",
            "pressure_method": "'rankine' or 'coulomb', default 'rankine'",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with FOS_sliding, FOS_overturning, html_length",
    },
    "ground_improvement_package": {
        "category": "calc_package",
        "description": (
            "Run ground improvement analysis (wick drains, aggregate piers, "
            "surcharge, or vibro) and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "method": "'wick_drains','aggregate_piers','surcharge','vibro' [REQUIRED]",
            "spacing": "Element spacing (m) — wick drains, aggregate piers, vibro",
            "ch": "Horizontal cv (m2/yr) — wick drains",
            "cv": "Vertical cv (m2/yr) — wick drains, surcharge",
            "Hdr": "Drainage path (m) — wick drains",
            "time": "Design time (years) — wick drains",
            "diameter": "Pier diameter (m) — aggregate piers",
            "length": "Pier length (m) — aggregate piers",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with method, html_length",
    },
    "wave_equation_package": {
        "category": "calc_package",
        "description": (
            "Run wave equation analysis (Smith 1-D model) and generate "
            "a bearing graph HTML calculation package."
        ),
        "params": {
            "hammer_name": "Hammer designation string [REQUIRED]",
            "cushion_area": "Cushion area (m2) [REQUIRED]",
            "cushion_thickness": "Cushion thickness (m) [REQUIRED]",
            "cushion_E": "Cushion elastic modulus (Pa) [REQUIRED]",
            "pile_length": "Pile length (m) [REQUIRED]",
            "pile_area": "Pile cross-section area (m2) [REQUIRED]",
            "pile_E": "Pile elastic modulus (Pa) [REQUIRED]",
            "skin_fraction": "Fraction of resistance from skin, default 0.5",
            "R_min": "Min resistance for bearing graph (kN), default 200",
            "R_max": "Max resistance for bearing graph (kN), default 2000",
            "R_step": "Resistance step (kN), default 200",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with n_points, html_length",
    },
    "pile_group_package": {
        "category": "calc_package",
        "description": (
            "Run pile group analysis (rigid cap load distribution) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "layout": "'rectangular' for auto-layout, or provide 'piles' list",
            "n_rows": "Number of rows — rectangular layout",
            "n_cols": "Number of columns — rectangular layout",
            "spacing_x": "X spacing (m) — rectangular layout",
            "spacing_y": "Y spacing (m) — rectangular layout",
            "piles": "List of dicts: x, y, label — custom layout",
            "Vz": "Vertical load (kN), default 0",
            "Mx": "Moment about x (kN-m), default 0",
            "My": "Moment about y (kN-m), default 0",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with n_piles, group_efficiency, html_length",
    },
    "sheet_pile_package": {
        "category": "calc_package",
        "description": (
            "Run sheet pile wall analysis (cantilever or anchored) "
            "and generate a Mathcad-style HTML calculation package."
        ),
        "params": {
            "wall_type": "'cantilever' or 'anchored', default 'cantilever'",
            "excavation_depth": "Excavation depth (m) [REQUIRED]",
            "soil_layers": (
                "List of dicts: thickness, unit_weight, friction_angle, "
                "cohesion, description [REQUIRED]"
            ),
            "gwt_depth_active": "GWT depth on active side (m), optional",
            "gwt_depth_passive": "GWT depth on passive side (m), optional",
            "surcharge": "Surface surcharge (kPa), default 0",
            "anchor_depth": "Anchor depth (m) — anchored walls",
            "FOS_passive": "FOS on passive resistance, default 1.5",
            "pressure_method": "'rankine' or 'coulomb', default 'rankine'",
            "project_name": "Project name for header",
            "output_path": "File path to save HTML, optional",
        },
        "returns": "dict with embedment_m, max_moment_kNm_per_m, html_length",
    },
}


# ---------------------------------------------------------------------------
# Foundry agent functions (standard 3-function pattern)
# ---------------------------------------------------------------------------

@function
def calc_package_agent(method: str, params_json: str) -> str:
    """Generate a professional Mathcad-style calculation package.

    Runs the specified analysis and produces a self-contained HTML document
    with inputs, equations, step-by-step calculations, figures, and
    engineering checks.

    Args:
        method: One of 'bearing_capacity_package', 'lateral_pile_package',
                'slope_stability_package', 'settlement_package',
                'axial_pile_package', 'drilled_shaft_package',
                'downdrag_package', 'seismic_package',
                'retaining_wall_package', 'ground_improvement_package',
                'wave_equation_package', 'pile_group_package',
                'sheet_pile_package'.
        params_json: JSON string with analysis parameters (see describe_method).

    Returns:
        JSON string with results summary and HTML output info.
    """
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    if method not in METHOD_REGISTRY:
        return json.dumps({
            "error": f"Unknown method '{method}'",
            "available_methods": list(METHOD_REGISTRY.keys()),
        })

    try:
        result = METHOD_REGISTRY[method](params)
        return json.dumps(_clean_result(result), default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "method": method})


@function
def calc_package_list_methods(category: str = "all") -> str:
    """List available calc package generation methods.

    Args:
        category: Filter by category. 'all' returns everything.

    Returns:
        JSON string with method names, categories, and descriptions.
    """
    methods = []
    for name, info in METHOD_INFO.items():
        if category == "all" or info["category"] == category:
            methods.append({
                "method": name,
                "category": info["category"],
                "description": info["description"],
            })
    return json.dumps({"methods": methods, "count": len(methods)})


@function
def calc_package_describe_method(method: str) -> str:
    """Get detailed parameter documentation for a calc package method.

    Args:
        method: Method name from calc_package_list_methods.

    Returns:
        JSON string with parameter descriptions and return info.
    """
    if method not in METHOD_INFO:
        return json.dumps({
            "error": f"Unknown method '{method}'",
            "available_methods": list(METHOD_INFO.keys()),
        })
    return json.dumps(METHOD_INFO[method])
