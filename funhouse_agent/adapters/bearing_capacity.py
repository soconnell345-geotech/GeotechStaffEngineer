"""Bearing capacity adapter — flat dict → BearingCapacityAnalysis → dict."""

from funhouse_agent.adapters import (apply_aliases, declare_aliases,
                                     declare_applies_when,
                                     reject_unknown_params, require_params)
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (
    bearing_capacity_Nc, bearing_capacity_Nq, bearing_capacity_Ngamma,
    TERZAGHI_NGAMMA_SOURCES, terzaghi_Nc, terzaghi_Nq, terzaghi_Ngamma,
    terzaghi_local_shear,
)

# Every top-level parameter _run_bearing_capacity_analysis consumes. Anything
# else is rejected loudly so an invented name is never silently dropped.
_ANALYSIS_VALID_PARAMS = (
    "width", "length", "depth", "shape", "base_tilt", "eccentricity_B",
    "eccentricity_L", "cohesion", "friction_angle", "unit_weight",
    "layer1_thickness", "layer1_description", "layer2_cohesion",
    "layer2_friction_angle", "layer2_unit_weight", "layer2_description",
    "gwt_depth", "load_inclination", "ground_slope", "vertical_load",
    "factor_of_safety", "ngamma_method", "factor_method", "failure_mode",
)


#: Common names for two inputs: the groundwater depth, and Vesic's "base
#: inclination" (the tilt of the footing base from horizontal).
_ANALYSIS_ALIASES = {
    "water_table_depth": "gwt_depth", "groundwater_depth": "gwt_depth",
    "base_inclination": "base_tilt",
}

#: factor_method drives the shape/depth/inclination factors, which exist in
#: the Vesic and Meyerhof forms; Hansen is an N-gamma option. "terzaghi" is
#: Terzaghi (1943) whole: his Nc, Nq, N-gamma and shape coefficients.
_FACTOR_METHODS = ("vesic", "meyerhof", "terzaghi")


def run_bearing_capacity_analysis(params: dict, *,
                                  method: str = "bearing_capacity_analysis"):
    """(result, analysis) of ``bearing_capacity_analysis``: the tool's own
    code path, also run by ``calc_package.bearing_capacity_package``
    (GM-141). ``method`` names the caller in error messages."""
    params = apply_aliases(params, _ANALYSIS_ALIASES)
    reject_unknown_params(params, _ANALYSIS_VALID_PARAMS, method=method)
    require_params(params, ["width", "unit_weight"], method=method,
                   valid=_ANALYSIS_VALID_PARAMS)
    factor_method = str(params.get("factor_method", "vesic")).strip().lower()
    if factor_method not in _FACTOR_METHODS:
        raise ValueError(
            f"{method}: factor_method must be one of "
            f"{list(_FACTOR_METHODS)} (it sets the shape, depth and "
            f"inclination factors; 'terzaghi' = Terzaghi 1943 throughout), "
            f"got '{params.get('factor_method')}'. Hansen's N-gamma is "
            f"chosen separately with ngamma_method='hansen'.")
    params = {**params, "factor_method": factor_method}
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
        thickness=params.get("layer1_thickness"),
        description=params.get("layer1_description", ""),
    )
    layer2 = None
    if "layer2_unit_weight" in params:
        layer2 = SoilLayer(
            cohesion=params.get("layer2_cohesion", 0.0),
            friction_angle=params.get("layer2_friction_angle", 0.0),
            unit_weight=params["layer2_unit_weight"],
            description=params.get("layer2_description", ""),
        )
    soil = BearingSoilProfile(layer1=layer1, layer2=layer2, gwt_depth=params.get("gwt_depth"))
    analysis = BearingCapacityAnalysis(
        footing=footing, soil=soil,
        load_inclination=params.get("load_inclination", 0.0),
        ground_slope=params.get("ground_slope", 0.0),
        vertical_load=params.get("vertical_load", 0.0),
        factor_of_safety=params.get("factor_of_safety", 3.0),
        ngamma_method=params.get("ngamma_method"),
        factor_method=params.get("factor_method", "vesic"),
        failure_mode=params.get("failure_mode", "general"),
    )
    return analysis.compute(), analysis


def _run_bearing_capacity_analysis(params: dict) -> dict:
    return run_bearing_capacity_analysis(params)[0].to_dict()


def _terzaghi_factors(phi, params: dict) -> dict:
    source = str(params.get("ngamma_method") or "kumbhojkar").strip().lower()
    mode = str(params.get("failure_mode", "general")).strip().lower()
    if mode not in ("general", "local"):
        raise ValueError("failure_mode must be 'general' or 'local'")
    phi_used = phi if mode == "general" else terzaghi_local_shear(0.0, phi)[1]
    out = {
        "Nc": round(terzaghi_Nc(phi_used), 2),
        "Nq": round(terzaghi_Nq(phi_used), 2),
        "Ngamma": round(terzaghi_Ngamma(phi_used, source), 2),
        "method": "terzaghi",
        "ngamma_method": source,
        "Ngamma_source": TERZAGHI_NGAMMA_SOURCES[source],
        "Ngamma_alternatives": {
            k: round(terzaghi_Ngamma(phi_used, k), 2)
            for k in TERZAGHI_NGAMMA_SOURCES
            if k != source and (k != "kumbhojkar" or phi_used <= 50)},
        "failure_mode": mode,
        "friction_angle_deg": phi,
    }
    if mode == "local":
        out["phi_used_deg"] = round(phi_used, 3)
        out["note"] = ("Local shear: factors at phi* = atan(2/3 tan phi); use "
                       "them with c* = 2/3 c (Terzaghi 1943; UFC 3-220-20 "
                       "Table 5-4).")
    return out


def _run_bearing_capacity_factors(params: dict) -> dict:
    valid = ("friction_angle", "method", "ngamma_method", "failure_mode")
    reject_unknown_params(params, valid, method="bearing_capacity_factors")
    require_params(params, ["friction_angle"],
                   method="bearing_capacity_factors", valid=list(valid))
    phi = params["friction_angle"]
    method = str(params.get("method", "vesic")).strip().lower()
    if method == "terzaghi":
        return _terzaghi_factors(phi, params)
    if (params.get("ngamma_method")
            or str(params.get("failure_mode", "general")).lower() != "general"):
        raise ValueError(
            "bearing_capacity_factors: ngamma_method and failure_mode go "
            "with method='terzaghi'; for the others, method IS the N-gamma "
            "choice.")
    return {
        "Nc": round(bearing_capacity_Nc(phi), 2),
        "Nq": round(bearing_capacity_Nq(phi), 2),
        "Ngamma": round(bearing_capacity_Ngamma(phi, method), 2),
        "method": method,
        "friction_angle_deg": phi,
    }


METHOD_REGISTRY = {
    "bearing_capacity_analysis": _run_bearing_capacity_analysis,
    "bearing_capacity_factors": _run_bearing_capacity_factors,
}

METHOD_INFO = {
    "bearing_capacity_analysis": {
        "category": "Bearing Capacity",
        "brief": "Full bearing capacity analysis for a shallow foundation (Vesic/Meyerhof/Hansen, or Terzaghi 1943 via factor_method='terzaghi').",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Footing width B (m)."},
            "length": {"type": "float", "required": False, "description": "Footing length L (m). Omit for strip."},
            "depth": {"type": "float", "required": False, "default": 0.0, "description": "Embedment depth Df (m)."},
            "shape": {"type": "str", "required": False, "default": "strip", "allowed_values": ["strip", "square", "rectangular", "circular"], "description": "Footing shape."},
            "cohesion": {"type": "float", "required": False, "default": 0.0, "description": "Soil cohesion c (kPa)."},
            "friction_angle": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle phi (degrees)."},
            "unit_weight": {"type": "float", "required": True, "description": "Soil unit weight gamma (kN/m3)."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth below surface (m)."},
            "layer2_cohesion": {"type": "float", "required": False, "description": "Second layer cohesion (kPa)."},
            "layer2_friction_angle": {"type": "float", "required": False, "description": "Second layer phi (degrees)."},
            "layer2_unit_weight": {"type": "float", "required": False, "description": "Second layer gamma (kN/m3). Triggers 2-layer analysis."},
            "layer1_thickness": {"type": "float", "required": False, "description": "First layer thickness (m) for 2-layer."},
            "layer1_description": {"type": "str", "required": False, "description": "First layer label for the result."},
            "layer2_description": {"type": "str", "required": False, "description": "Second layer label for the result."},
            "factor_of_safety": {"type": "float", "required": False, "default": 3.0, "description": "Factor of safety."},
            "factor_method": {"type": "str", "required": False, "default": "vesic", "allowed_values": ["vesic", "meyerhof", "terzaghi"], "description": "Shape/depth/inclination-factor method. Hansen applies to N-gamma only: use ngamma_method='hansen'. 'terzaghi' = Terzaghi (1943): his Nc, Nq, N-gamma and shape coefficients (1.3cNc; 0.4/0.3 gamma B Ngamma square/circular), no depth or inclination factors; not for inclined loads, slopes or tilted bases (UFC 3-220-20 Table 5-3)."},
            "ngamma_method": {"type": "str", "required": False, "default_depends_on": {"parameter": "factor_method", "values": {"terzaghi": "kumbhojkar"}, "otherwise": "vesic"}, "allowed_values": ["vesic", "meyerhof", "hansen", "kumbhojkar", "coduto"], "description": "Ngamma factor method (may differ from factor_method). Default depends on factor_method: 'kumbhojkar' with factor_method='terzaghi', else 'vesic'. With factor_method='terzaghi' (Terzaghi gave N-gamma only as a chart): 'kumbhojkar' (Kumbhojkar 1993 as tabulated in Das) or 'coduto' (Coduto et al. 2016 approximation, UFC 3-220-20 Table 5-2); the result reports the other beside it."},
            "failure_mode": {"type": "str", "required": False, "default": "general", "allowed_values": ["general", "local"], "description": "factor_method='terzaghi' only: 'local' applies Terzaghi's local-shear reduction c* = 2/3 c, tan(phi*) = 2/3 tan(phi) (UFC 3-220-20 Table 5-4)."},
            "base_tilt": {"type": "float", "required": False, "default": 0.0, "description": "Base tilt angle (degrees)."},
            "eccentricity_B": {"type": "float", "required": False, "default": 0.0, "description": "Load eccentricity along B (m)."},
            "eccentricity_L": {"type": "float", "required": False, "default": 0.0, "description": "Load eccentricity along L (m)."},
            "load_inclination": {"type": "float", "required": False, "default": 0.0, "description": "Load inclination from vertical (degrees)."},
            "ground_slope": {"type": "float", "required": False, "default": 0.0, "description": "Ground slope angle (degrees)."},
            "vertical_load": {"type": "float", "required": False, "default": 0.0, "description": "Applied vertical load (kN) for inclination factors."},
        },
        "returns": {
            "q_ultimate_kPa": "Ultimate bearing capacity.",
            "q_allowable_kPa": "Allowable bearing capacity (q_ult / FOS).",
            "Nc": "Cohesion factor.", "Nq": "Overburden factor.", "Ngamma": "Unit weight factor.",
            "Ngamma_alternatives": "Terzaghi: N-gamma from the other cited source.",
            "notes": "Terzaghi: method basis, N-gamma source, shape interpolation, Df/B and local-shear notes.",
        },
    },
    "bearing_capacity_factors": {
        "category": "Bearing Capacity",
        "brief": "Quick lookup of Nc, Nq, Ngamma for a given friction angle (Vesic/Meyerhof/Hansen N-gamma, or Terzaghi's factors).",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "description": "Friction angle phi (degrees, 0-50)."},
            "method": {"type": "str", "required": False, "default": "vesic", "allowed_values": ["vesic", "meyerhof", "hansen", "terzaghi"], "description": "Ngamma method; 'terzaghi' returns Terzaghi's (1943) Nc, Nq and Ngamma."},
            "ngamma_method": {"type": "str", "required": False, "default": "kumbhojkar", "allowed_values": ["kumbhojkar", "coduto"], "description": "method='terzaghi' only: Terzaghi N-gamma source (Kumbhojkar 1993 as tabulated in Das, or Coduto et al. 2016 / UFC 3-220-20 Table 5-2)."},
            "failure_mode": {"type": "str", "required": False, "default": "general", "allowed_values": ["general", "local"], "description": "method='terzaghi' only: 'local' gives the factors at phi* = atan(2/3 tan phi)."},
        },
        "returns": {"Nc": "Cohesion factor.", "Nq": "Overburden factor.", "Ngamma": "Unit weight factor."},
    },
}

declare_aliases(METHOD_INFO, {
    "bearing_capacity_analysis": _ANALYSIS_ALIASES})

# When each default applies (GM-119): the failure mode and Terzaghi's
# N-gamma source go with Terzaghi's factors only.
declare_applies_when(METHOD_INFO, {
    "bearing_capacity_analysis": {
        "failure_mode": {"factor_method": ["terzaghi"]}},
    "bearing_capacity_factors": {
        "ngamma_method": {"method": ["terzaghi"]},
        "failure_mode": {"method": ["terzaghi"]}},
})
