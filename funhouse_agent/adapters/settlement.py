"""Settlement adapter — flat dict → settlement functions → dict."""

from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, declare_aliases, declare_applies_when,
    field_keys, reject_unknown_params, require_keys, require_params,
)
from geotech_common.choices import MODULE, choice, how_given
from settlement.immediate import (
    GEC6_RIGID_SQUARE_NOTE, GEC6_TABLE_5_12_BASIS, SchmertmannLayer,
    elastic_settlement,
    gec6_influence_factor, schmertmann_settlement,
)
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer,
)
from settlement.hough import HoughLayer, hough_settlement
from settlement.stress_distribution import stress_at_depth
from settlement.analysis import SettlementAnalysis

# Keys each layer dict may carry: the dataclass fields it builds (Hough also
# its aliases below). Anything else was read with .get and silently left
# out, e.g. a consolidation layer's 'pc' ran it as normally consolidated
# with sigma_p unset (GM-58).
_SCHMERTMANN_KEYS = field_keys(SchmertmannLayer)
_CONSOLIDATION_KEYS = field_keys(ConsolidationLayer)
_HOUGH_ALIASES = {"C": "C_prime", "Cprime": "C_prime",
                  "sigma_vo": "sigma_v0",
                  "depth_to_mid": "depth_to_center",
                  "depth_mid": "depth_to_center"}
_HOUGH_KEYS = field_keys(HoughLayer) + tuple(_HOUGH_ALIASES)
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_CONSOLIDATION_KEY_HINTS = {
    "pc": ("sigma_p",), "preconsolidation": ("sigma_p",),
    "preconsolidationpressure": ("sigma_p",),
    "pastpressure": ("sigma_p",), "sigmac": ("sigma_p",),
    "e": ("e0",),
}
_SCHMERTMANN_KEY_HINTS = {"e": ("Es",), "modulus": ("Es",),
                          "top": ("depth_top",), "bottom": ("depth_bottom",)}


#: Other names for the net bearing pressure, and the combined analysis's.
_Q_ALIASES = {"q": "q_net", "q_applied": "q_net", "pressure": "q_net"}
_COMBINED_ALIASES = {
    "q_net": "q_applied", "q": "q_applied",
    # the footing's own descriptive names (2026-10 Foundry eval, SE-4)
    "foundation_pressure": "q_applied", "foundation_width": "B",
    "foundation_length": "L",
}


_ELASTIC_KEYS = ("q_net", "B", "Es", "nu", "Iw", "shape", "L", "rigidity",
                 "point")


def _elastic_iw_choice(params: dict) -> tuple:
    """``(Iw, choices entry)``: the influence factor given, or GEC-6 Table
    5-12's C_d for the footing's shape, rigidity and point with the table's
    other values for it (GM-203: Iw was 1.0 for every footing, which matches
    no printed point of a square)."""
    Iw = params.get("Iw")
    if Iw is not None:
        return Iw, choice(
            "influence factor Iw", f"{Iw:g}", how_given(Iw, None, name="Iw"),
            "Se = q B (1 - nu^2) Iw / Es (GEC-6 Eq. 5-36)",
            ["GEC-6 Table 5-12's C_d for the shape, rigidity and point "
             "(leave Iw out)"])
    shape = params.get("shape", "square")
    rigidity = params.get("rigidity", "flexible")
    point = params.get("point", "center")
    B, L = params["B"], params.get("L")
    Iw, where = gec6_influence_factor(shape, rigidity, point, B=B, L=L)
    others = []
    for rig, pt, label in (
            ("rigid", "center", "the footing rigid (rigidity 'rigid')"),
            ("flexible", "average", "the average over a flexible footing "
             "(point 'average')"),
            ("flexible", "corner", "a corner of a flexible footing "
             "(point 'corner')")):
        try:
            v = gec6_influence_factor(shape, rig, pt, B=B, L=L)[0]
        except ValueError:
            continue
        if abs(v - Iw) > 1e-9:
            others.append(f"{v:.3g} for {label}")
    basis = GEC6_TABLE_5_12_BASIS
    if where.startswith("square"):
        # a square uses or lists the printed rigid square, which is above
        # the flexible average (GM-214)
        basis += ". " + GEC6_RIGID_SQUARE_NOTE
    return Iw, choice(
        "influence factor Iw",
        f"{Iw:.3g}: GEC-6 Table 5-12, {where} (shape '{shape}', rigidity "
        f"'{rigidity}', point '{point}')",
        MODULE, basis, others + ["give Iw (Iw)"])


def _run_elastic_settlement(params: dict) -> dict:
    params = apply_aliases(params, _Q_ALIASES)
    reject_unknown_params(params, _ELASTIC_KEYS, method="elastic_settlement")
    require_params(params, ["q_net", "B", "Es"], method="elastic_settlement",
                   valid=list(_ELASTIC_KEYS))
    Iw, entry = _elastic_iw_choice(params)
    Se = elastic_settlement(q=params["q_net"], B=params["B"],
                            Es=params["Es"], nu=params.get("nu", 0.3),
                            Iw=Iw)
    return {"immediate_settlement_m": round(Se, 6),
            "immediate_settlement_mm": round(Se * 1000, 2),
            "Iw": round(Iw, 4), "method": "elastic", "choices": [entry]}


def _run_schmertmann_settlement(params: dict) -> dict:
    params = apply_aliases(params, _Q_ALIASES)
    reject_unknown_params(
        params, ("q_net", "q_overburden", "B", "L", "layers", "shape",
                 "time_years", "gamma_soil"),
        method="schmertmann_settlement")
    require_params(params, ["q_net", "B", "layers"], method="schmertmann_settlement")
    check_nested_list(params["layers"], _SCHMERTMANN_KEYS,
                      method="schmertmann_settlement", label="layers",
                      hints=_SCHMERTMANN_KEY_HINTS)
    for l in params["layers"]:
        require_keys(l, ["depth_top", "depth_bottom", "Es"], method="schmertmann_settlement")
    layers = [SchmertmannLayer(depth_top=l["depth_top"], depth_bottom=l["depth_bottom"], Es=l["Es"]) for l in params["layers"]]
    Se = schmertmann_settlement(
        q_net=params["q_net"], q0=params.get("q_overburden", 0.0),
        B=params["B"], L=params.get("L", params["B"]),
        layers=layers, footing_shape=params.get("shape", "square"),
        time_years=params.get("time_years", 0.0),
        gamma_soil=params.get("gamma_soil"),
    )
    return {"schmertmann_settlement_m": round(Se, 6), "schmertmann_settlement_mm": round(Se * 1000, 2), "method": "schmertmann"}


def _run_consolidation_settlement(params: dict) -> dict:
    reject_unknown_params(
        params, ("layers", "delta_sigma", "delta_sigmas", "q_net", "B", "L",
                 "stress_method"),
        method="consolidation_settlement")
    require_params(params, ["layers"], method="consolidation_settlement")
    check_nested_list(params["layers"], _CONSOLIDATION_KEYS,
                      method="consolidation_settlement", label="layers",
                      hints=_CONSOLIDATION_KEY_HINTS)
    for l in params["layers"]:
        require_keys(l, ["thickness", "depth_to_center", "e0", "Cc", "Cr", "sigma_v0"],
                     method="consolidation_settlement")
    layers = [
        ConsolidationLayer(
            thickness=l["thickness"], depth_to_center=l["depth_to_center"],
            e0=l["e0"], Cc=l["Cc"], Cr=l["Cr"],
            sigma_v0=l["sigma_v0"], sigma_p=l.get("sigma_p"),
            description=l.get("description", ""),
        ) for l in params["layers"]
    ]
    delta_sigmas = params.get("delta_sigmas")
    if delta_sigmas is None and "delta_sigma" in params:
        delta_sigmas = [params["delta_sigma"]] * len(layers)

    layer_results = []
    S_total = 0.0
    if delta_sigmas is not None:
        for layer, ds in zip(layers, delta_sigmas):
            Sc = consolidation_settlement_layer(layer, ds)
            S_total += Sc
            layer_results.append({"depth_m": layer.depth_to_center, "settlement_mm": round(Sc * 1000, 3), "delta_sigma_kPa": round(ds, 2)})
    else:
        # the stress increase is worked out from the footing: it needs the
        # pressure and the width (no silent q_net = 0 or B = 1 m, GM-142)
        require_params(params, ["q_net", "B"],
                       method="consolidation_settlement",
                       valid=["delta_sigma", "delta_sigmas", "q_net", "B",
                              "L", "stress_method"])
        q_net = params["q_net"]
        B = params["B"]
        L = params.get("L", B)
        method = params.get("stress_method", "2:1")
        for layer in layers:
            ds = stress_at_depth(q_net, B, L, layer.depth_to_center, method=method)
            Sc = consolidation_settlement_layer(layer, ds)
            S_total += Sc
            layer_results.append({"depth_m": layer.depth_to_center, "settlement_mm": round(Sc * 1000, 3), "delta_sigma_kPa": round(ds, 2)})

    out = {"consolidation_settlement_m": round(S_total, 6), "consolidation_settlement_mm": round(S_total * 1000, 2), "layer_breakdown": layer_results}
    if delta_sigmas is None:
        # The stress increase was worked out here, by a method (GM-115).
        from geotech_common.choices import DEFAULT, choice
        from settlement.stress_distribution import stress_method_choice
        out["choices"] = [stress_method_choice(method)]
        if "L" not in params:
            out["choices"].append(choice(
                "footing length", f"L = B = {B:g} m (square)", DEFAULT,
                "L not given: a square footing", ["L given (L)"]))
    return out


def _run_hough_settlement(params: dict) -> dict:
    params = apply_aliases(params, _Q_ALIASES)
    reject_unknown_params(
        params, ("q_net", "B", "L", "layers"),
        method="hough_settlement")
    require_params(params, ["q_net", "B", "layers"], method="hough_settlement",
                   valid=["q_net", "B", "L", "layers"])
    check_nested_list(params["layers"], _HOUGH_KEYS,
                      method="hough_settlement", label="layers")
    layers = []
    for l in params["layers"]:
        l = apply_aliases(l, _HOUGH_ALIASES)
        require_keys(l, ["thickness", "depth_to_center", "sigma_v0", "C_prime"],
                     method="hough_settlement")
        layers.append(HoughLayer(
            thickness=l["thickness"], depth_to_center=l["depth_to_center"],
            sigma_v0=l["sigma_v0"], C_prime=l["C_prime"],
            description=l.get("description", ""),
        ))
    result = hough_settlement(
        layers=layers, q_net=params["q_net"], B=params["B"],
        L=params.get("L"),
    )
    return result.to_dict()


def run_combined_settlement_analysis(
        params: dict, *, method: str = "combined_settlement_analysis"):
    """(result, analysis) of ``combined_settlement_analysis``: the tool's
    own code path, also run by ``calc_package.settlement_package``
    (GM-141). ``method`` names the caller in error messages."""
    params = apply_aliases(params, _COMBINED_ALIASES)
    reject_unknown_params(
        params, ("q_applied", "q_overburden", "B", "L", "shape",
                 "immediate_method", "Es", "nu", "schmertmann_layers",
                 "consolidation_layers", "cv", "Hdr", "drainage", "C_alpha",
                 "e0_secondary", "t_secondary", "stress_method",
                 "time_years_schmertmann", "Iw"),
        method=method)
    require_params(params, ["q_applied", "B"], method=method)
    consol_layers = None
    check_nested_list(params.get("consolidation_layers"),
                      _CONSOLIDATION_KEYS,
                      method=method,
                      label="consolidation_layers",
                      hints=_CONSOLIDATION_KEY_HINTS)
    check_nested_list(params.get("schmertmann_layers"), _SCHMERTMANN_KEYS,
                      method=method,
                      label="schmertmann_layers",
                      hints=_SCHMERTMANN_KEY_HINTS)
    if "consolidation_layers" in params:
        consol_layers = []
        for l in params["consolidation_layers"]:
            require_keys(l, ["thickness", "depth_to_center", "e0", "Cc", "Cr",
                             "sigma_v0"],
                         method=method,
                         item_label="consolidation_layers[]")
            consol_layers.append(ConsolidationLayer(
                thickness=l["thickness"], depth_to_center=l["depth_to_center"],
                e0=l["e0"], Cc=l["Cc"], Cr=l["Cr"],
                sigma_v0=l["sigma_v0"], sigma_p=l.get("sigma_p"),
                description=l.get("description", ""),
            ))
    schmertmann_layers = None
    if "schmertmann_layers" in params:
        schmertmann_layers = []
        for l in params["schmertmann_layers"]:
            require_keys(l, ["depth_top", "depth_bottom", "Es"],
                         method=method,
                         item_label="schmertmann_layers[]")
            schmertmann_layers.append(SchmertmannLayer(
                depth_top=l["depth_top"], depth_bottom=l["depth_bottom"],
                Es=l["Es"]))

    analysis = SettlementAnalysis(
        q_applied=params["q_applied"], q_overburden=params.get("q_overburden", 0.0),
        # L not given: the module works it out from the shape and says
        # so in its choices (GM-142)
        B=params["B"], L=params.get("L"),
        footing_shape=params.get("shape", "square"),
        immediate_method=params.get("immediate_method", "elastic"),
        Es_immediate=params.get("Es"), nu=params.get("nu", 0.3),
        Iw_immediate=params.get("Iw"),
        schmertmann_layers=schmertmann_layers, consolidation_layers=consol_layers,
        cv=params.get("cv"), Hdr=params.get("Hdr"),
        drainage=params.get("drainage", "double"),
        C_alpha=params.get("C_alpha"),
        e0_secondary=params.get("e0_secondary"),
        t_secondary=params.get("t_secondary", 0.0),
        stress_method=params.get("stress_method", "2:1"),
        time_years_schmertmann=params.get("time_years_schmertmann", 0.0),
    )
    return analysis.compute(), analysis


def _run_combined_settlement(params: dict) -> dict:
    return run_combined_settlement_analysis(params)[0].to_dict()


METHOD_REGISTRY = {
    "elastic_settlement": _run_elastic_settlement,
    "schmertmann_settlement": _run_schmertmann_settlement,
    "consolidation_settlement": _run_consolidation_settlement,
    "hough_settlement": _run_hough_settlement,
    "combined_settlement_analysis": _run_combined_settlement,
}

METHOD_INFO = {
    "elastic_settlement": {
        "category": "Immediate Settlement",
        "brief": "Elastic settlement: Se = q*B*(1-nu^2)/Es * Iw.",
        "parameters": {
            "q_net": {"type": "float", "required": True, "description": "Net bearing pressure (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "Es": {"type": "float", "required": True, "description": "Soil elastic modulus (kPa)."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio."},
            "Iw": {"type": "float", "required": False, "description": "Influence factor in Se = q B (1 - nu^2) Iw / Es. Not given: GEC-6 Table 5-12's shape and rigidity factor C_d for the shape, rigidity and point (1.12 at the centre of a flexible square; 0.99 for a rigid square, 0.79 for a rigid circle)."},
            "shape": {"type": "str", "required": False, "default": "square", "allowed_values": ["square", "rectangular", "circular"], "description": "Footing shape, for GEC-6 Table 5-12's Iw when Iw is not given. A strip has no Iw (it grows without limit with L/B): give shape 'rectangular' with L, or Iw."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Required with shape 'rectangular' and no Iw: GEC-6 Table 5-12 by L/B, interpolated in log(L/B) between its printed rows (1.5 to 10,000)."},
            "rigidity": {"type": "str", "required": False, "default": "flexible", "allowed_values": ["flexible", "rigid"], "description": "Footing rigidity, for GEC-6 Table 5-12's Iw when Iw is not given. Rigid: 0.99 for a square and 0.79 for a circle, the same at every point; the table prints no rigid rectangle (give Iw)."},
            "point": {"type": "str", "required": False, "default": "center", "allowed_values": ["center", "corner", "mid_short", "mid_long", "average", "edge"], "description": "Where under a flexible footing, for GEC-6 Table 5-12's Iw when Iw is not given: the centre, a corner, the middle of the short or long side, or the average over the area ('edge' for a circle)."},
        },
        "returns": {"immediate_settlement_mm": "Settlement in mm.",
                    "Iw": "The influence factor used.",
                    "choices": "The influence factor: given, or GEC-6 Table 5-12's for the shape, rigidity and point, with the table's other values for the footing. Tell the user."},
    },
    "schmertmann_settlement": {
        "category": "Immediate Settlement",
        "brief": "Schmertmann (1978) strain influence factor method.",
        "parameters": {
            "q_net": {"type": "float", "required": True, "description": "Net bearing pressure (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Default B."},
            "layers": {"type": "array", "required": True, "description": "Array of {depth_top, depth_bottom, Es} dicts."},
            "q_overburden": {"type": "float", "required": False, "default": 0.0, "description": "Overburden pressure at footing base (kPa)."},
            "shape": {"type": "str", "required": False, "default": "square", "allowed_values": ["square", "rectangular", "strip"], "description": "Footing shape."},
            "time_years": {"type": "float", "required": False, "default": 0.0, "description": "Time for creep factor."},
            "gamma_soil": {"type": "float", "required": False, "description": "Effective unit weight (kN/m3) below the base, used for sigma'_vp at the PEAK-influence depth in Izp = 0.5 + 0.1*sqrt(dq/sigma'_vp). STRONGLY recommended for embedded footings: without it sigma'_vp falls back to the base overburden q_overburden, which OVERSTATES Izp (and settlement)."},
        },
        "returns": {"schmertmann_settlement_mm": "Settlement in mm."},
    },
    "consolidation_settlement": {
        "category": "Consolidation",
        "brief": "Cc/Cr e-log(p) consolidation settlement with layer summation.",
        "parameters": {
            "layers": {"type": "array", "required": True, "description": "Array of {thickness (m), depth_to_center (m), e0, Cc, Cr (recompression index; ~Cc/10 if unknown), sigma_v0 (initial effective stress at layer center, kPa, must be > 0), sigma_p (preconsolidation, kPa, optional)} dicts. All of thickness/depth_to_center/e0/Cc/Cr/sigma_v0 are required."},
            "delta_sigma": {"type": "float", "required": False, "description": "Uniform stress increase (kPa). Or use delta_sigmas array."},
            "delta_sigmas": {"type": "array", "required": False, "description": "Per-layer stress increases (kPa), one per layer."},
            "q_net": {"type": "float", "required": False, "description": "Net pressure for auto stress distribution; required without delta_sigma(s)."},
            "B": {"type": "float", "required": False, "description": "Footing width for stress distribution (m); required without delta_sigma(s)."},
            "L": {"type": "float", "required": False, "description": "Footing length for stress distribution (m). Defaults to B."},
            "stress_method": {"type": "str", "required": False, "default": "2:1", "allowed_values": ["2:1", "boussinesq", "westergaard"], "description": "Stress distribution method."},
        },
        "returns": {"consolidation_settlement_mm": "Total consolidation settlement in mm.", "layer_breakdown": "Per-layer results.", "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives."},
    },
    "hough_settlement": {
        "category": "Granular Settlement",
        "brief": "Hough (1959) granular C'-index settlement: dH = H/C'*log10[(sigma'_vo + delta_sigma)/sigma'_vo] per layer, with a 2:1 stress increase (GEC-6). Use for sands/gravels with SPT data (C' is the Hough bearing-capacity index from corrected N', NOT Cc/(1+e0)).",
        "parameters": {
            "q_net": {"type": "float", "required": True, "description": "Net bearing pressure at footing base (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of granular {thickness (m), depth_to_center (m, mid-depth below footing base), sigma_v0 (effective overburden at layer center, kPa, > 0), C_prime (Hough bearing-capacity index, dimensionless, > 0)} dicts. All four keys are required per layer."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Defaults to B (square footing); 2:1 increase is q*B*L/((B+z)(L+z))."},
        },
        "returns": {"total_settlement_mm": "Total Hough settlement in mm.", "layer_breakdown": "Per-layer results (delta_sigma, sigma_v0, C', settlement)."},
    },
    "combined_settlement_analysis": {
        "category": "Combined Analysis",
        "brief": "Full analysis: immediate + consolidation + secondary + time curve.",
        "parameters": {
            "q_applied": {"type": "float", "required": True, "description": "Applied pressure (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "Es": {"type": "float", "required": False, "description": "Elastic modulus for immediate settlement (kPa), for the elastic method. Not given: no immediate settlement is computed (0)."},
            "consolidation_layers": {"type": "array", "required": False, "description": "Array of consolidation layer dicts."},
            "schmertmann_layers": {"type": "array", "required": False, "description": "Array of Schmertmann layer dicts."},
            "cv": {"type": "float", "required": False, "description": "Coefficient of consolidation (m2/yr). With consolidation_layers, adds the time curve and t50/t90 to the result."},
            "Hdr": {"type": "float", "required": False, "description": "Drainage path length (m) for the time curve. Default: consolidating thickness / 2 (double drainage) or the whole thickness (single)."},
            "immediate_method": {"type": "str", "required": False, "default": "elastic", "allowed_values": ["elastic", "schmertmann"], "description": "Immediate settlement method."},
            "drainage": {"type": "str", "required": False, "default": "double", "allowed_values": ["single", "double"], "description": "Drainage condition for time rate."},
            "stress_method": {"type": "str", "required": False, "default": "2:1", "allowed_values": ["2:1", "boussinesq", "westergaard"], "description": "Stress distribution method."},
            "q_overburden": {"type": "float", "required": False, "default": 0.0, "description": "Overburden pressure at footing base (kPa)."},
            "L": {"type": "float", "required": False, "description": "Footing length (m). Not given: L = B for a square or circular footing; a strip is taken as infinitely long (plane strain), which the elastic method refuses unless Iw is given (its Iw grows with L/B without limit); a rectangular footing needs L."},
            "shape": {"type": "str", "required": False, "default": "square", "allowed_values": ["square", "rectangular", "strip", "circular"], "description": "Footing shape."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio for immediate settlement."},
            "Iw": {"type": "float", "required": False, "description": "Elastic influence factor for the elastic method. Not given: Schleicher's value at the centre of a flexible footing (1.12 for a square, growing with L/B; 1.0 for a circle)."},
            "C_alpha": {"type": "float", "required": False, "description": "Secondary compression index. Enables secondary settlement."},
            "e0_secondary": {"type": "float", "required": False, "description": "One void ratio for the secondary settlement of every consolidation layer. Not given: each layer's own e0, layer by layer."},
            "t_secondary": {"type": "float", "required": False, "default": 0.0, "description": "Time for secondary settlement (years)."},
            "time_years_schmertmann": {"type": "float", "required": False, "default": 0.0, "description": "Creep time for Schmertmann immediate method (years)."},
        },
        "returns": {
            "total_mm": "Total settlement = immediate + consolidation + secondary (mm).",
            "immediate_mm": "Immediate settlement (mm).",
            "consolidation_mm": "Primary consolidation settlement (mm).",
            "secondary_mm": "Secondary compression settlement (mm); 0 without C_alpha.",
            "immediate_method": "Immediate-settlement method used.",
            "stress_method": "Stress-distribution method used.",
            "consolidation_layers": "Per-layer consolidation breakdown (with consolidation_layers).",
            "time_settlement_curve": "With cv: list of {time_years, settlement_mm, U_percent} points, t = 0 then 0.01-100 years log-spaced; settlement_mm = immediate + U(t) x primary consolidation (Terzaghi 1-D; secondary not included).",
            "t50_years": "With cv: time to 50 % average degree of consolidation (years).",
            "t90_years": "With cv: time to 90 % average degree of consolidation (years).",
            "cv_m2_per_year": "With cv: the coefficient of consolidation used (m2/year).",
            "Hdr_m": "With cv: the drainage path length used (m): Hdr if given, else the consolidating thickness / 2 (double drainage) or the whole thickness (single).",
            "time_settlement_basis": "With cv: one sentence stating how the time curve is computed.",
            "choices": "What the run chose and how: what, chosen, how (given, default or module), basis, alternatives.",
        },
    },
}

declare_aliases(METHOD_INFO, {
    "elastic_settlement": _Q_ALIASES,
    "schmertmann_settlement": _Q_ALIASES,
    "hough_settlement": _Q_ALIASES,
    "combined_settlement_analysis": _COMBINED_ALIASES,
})

# When each default applies (GM-119): the stress increase to a stress worked
# out from the footing pressure, nu to the elastic method with a modulus,
# the creep time to Schmertmann, the drainage to a time curve with no Hdr
# given, the secondary time to a run with C_alpha (e0_secondary has no
# default since GM-142: each layer's own e0, stated in the choices).
declare_applies_when(METHOD_INFO, {
    # the shape, rigidity and point pick GEC-6 Table 5-12's Iw when Iw is
    # not given (GM-203); a rigid footing settles the same at every point
    "elastic_settlement": {
        "shape": {"Iw": "unset"},
        "rigidity": {"Iw": "unset"},
        "point": {"Iw": "unset", "rigidity": ["flexible"]},
        "L": {"Iw": "unset", "shape": ["rectangular"]},
    },
    "consolidation_settlement": {
        "stress_method": {"delta_sigma": "unset", "delta_sigmas": "unset"},
    },
    "combined_settlement_analysis": {
        "nu": {"immediate_method": ["elastic"], "Es": "set"},
        "Es": {"immediate_method": ["elastic"]},
        "time_years_schmertmann": {"immediate_method": ["schmertmann"]},
        "stress_method": {"consolidation_layers": "set"},
        "drainage": {"cv": "set", "consolidation_layers": "set",
                     "Hdr": "unset"},
        "t_secondary": {"C_alpha": "set"},
    },
})


# What a left-out input with no default stands for (BACKLOG GM-165):
# the footing length, the drainage path, the influence factor and the
# secondary void ratio.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
declare_when_absent(METHOD_INFO, {
    "elastic_settlement": {
        "Iw": ("GEC-6 Table 5-12's C_d for the shape, rigidity and point",
               "the result's Iw and choices give the value and the "
               "table's other points (GEC-6 Table 5-12, p. 97)")},
    "schmertmann_settlement": {"L": "B"},
    "consolidation_settlement": {"L": "B"},
    "hough_settlement": {"L": "B"},
    "combined_settlement_analysis": {
        "L": ("set by shape and B",
              "L = B for a square or circular footing; a strip is taken as "
              "infinitely long (plane strain)"),
        "Hdr": ("half the consolidating thickness, or all of it with single "
                "drainage", "the drainage path for the time curve from the "
                "drainage given"),
        "Iw": ("Schleicher's value at the centre of a flexible footing",
               "1.12 for a square, growing with L/B; 1.0 for a circle"),
        "e0_secondary": ("each layer's own e0",
                         "the secondary settlement of each consolidation "
                         "layer uses that layer's void ratio"),
        # GM-184: the result's choices already said so
        "Es": ("no immediate settlement",
               "the elastic method needs Es, so the immediate settlement "
               "is not computed (0); give Es, or immediate_method "
               "'schmertmann' with its layers")},
})
