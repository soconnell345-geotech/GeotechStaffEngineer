"""Settlement adapter — flat dict → settlement functions → dict."""

from settlement.immediate import elastic_settlement, SchmertmannLayer, schmertmann_settlement
from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer,
)
from settlement.stress_distribution import stress_at_depth
from settlement.analysis import SettlementAnalysis


def _run_elastic_settlement(params: dict) -> dict:
    Se = elastic_settlement(
        q=params["q_net"], B=params["B"], Es=params["Es"],
        nu=params.get("nu", 0.3), Iw=params.get("Iw", 1.0),
        shape=params.get("shape", "square"),
    )
    return {"immediate_settlement_m": round(Se, 6), "immediate_settlement_mm": round(Se * 1000, 2), "method": "elastic"}


def _run_schmertmann_settlement(params: dict) -> dict:
    layers = [SchmertmannLayer(depth_top=l["depth_top"], depth_bottom=l["depth_bottom"], Es=l["Es"]) for l in params["layers"]]
    Se = schmertmann_settlement(
        q_net=params["q_net"], q0=params.get("q_overburden", 0.0),
        B=params["B"], L=params.get("L", params["B"]),
        layers=layers, footing_shape=params.get("shape", "square"),
        time_years=params.get("time_years", 0.0),
    )
    return {"schmertmann_settlement_m": round(Se, 6), "schmertmann_settlement_mm": round(Se * 1000, 2), "method": "schmertmann"}


def _run_consolidation_settlement(params: dict) -> dict:
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
        q_net = params.get("q_net", 0)
        B = params.get("B", 1.0)
        L = params.get("L", B)
        method = params.get("stress_method", "2:1")
        for layer in layers:
            ds = stress_at_depth(q_net, B, L, layer.depth_to_center, method=method)
            Sc = consolidation_settlement_layer(layer, ds)
            S_total += Sc
            layer_results.append({"depth_m": layer.depth_to_center, "settlement_mm": round(Sc * 1000, 3), "delta_sigma_kPa": round(ds, 2)})

    return {"consolidation_settlement_m": round(S_total, 6), "consolidation_settlement_mm": round(S_total * 1000, 2), "layer_breakdown": layer_results}


def _run_combined_settlement(params: dict) -> dict:
    consol_layers = None
    if "consolidation_layers" in params:
        consol_layers = [
            ConsolidationLayer(
                thickness=l["thickness"], depth_to_center=l["depth_to_center"],
                e0=l["e0"], Cc=l["Cc"], Cr=l["Cr"],
                sigma_v0=l["sigma_v0"], sigma_p=l.get("sigma_p"),
            ) for l in params["consolidation_layers"]
        ]
    schmertmann_layers = None
    if "schmertmann_layers" in params:
        schmertmann_layers = [SchmertmannLayer(depth_top=l["depth_top"], depth_bottom=l["depth_bottom"], Es=l["Es"]) for l in params["schmertmann_layers"]]

    analysis = SettlementAnalysis(
        q_applied=params["q_applied"], q_overburden=params.get("q_overburden", 0.0),
        B=params["B"], L=params.get("L", params["B"]),
        footing_shape=params.get("shape", "square"),
        immediate_method=params.get("immediate_method", "elastic"),
        Es_immediate=params.get("Es"), nu=params.get("nu", 0.3),
        schmertmann_layers=schmertmann_layers, consolidation_layers=consol_layers,
        cv=params.get("cv"), Hdr=params.get("Hdr"),
        drainage=params.get("drainage", "double"),
        C_alpha=params.get("C_alpha"),
        e0_secondary=params.get("e0_secondary", 1.0),
        t_secondary=params.get("t_secondary", 0.0),
        stress_method=params.get("stress_method", "2:1"),
        time_years_schmertmann=params.get("time_years_schmertmann", 0.0),
    )
    return analysis.compute().to_dict()


METHOD_REGISTRY = {
    "elastic_settlement": _run_elastic_settlement,
    "schmertmann_settlement": _run_schmertmann_settlement,
    "consolidation_settlement": _run_consolidation_settlement,
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
            "shape": {"type": "str", "required": False, "default": "square", "description": "square/rectangular/circular."},
        },
        "returns": {"immediate_settlement_mm": "Settlement in mm."},
    },
    "schmertmann_settlement": {
        "category": "Immediate Settlement",
        "brief": "Schmertmann (1978) strain influence factor method.",
        "parameters": {
            "q_net": {"type": "float", "required": True, "description": "Net bearing pressure (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "layers": {"type": "array", "required": True, "description": "Array of {depth_top, depth_bottom, Es} dicts."},
            "q_overburden": {"type": "float", "required": False, "default": 0.0, "description": "Overburden pressure at footing base (kPa)."},
            "shape": {"type": "str", "required": False, "default": "square", "description": "square/rectangular."},
            "time_years": {"type": "float", "required": False, "default": 0.0, "description": "Time for creep factor."},
        },
        "returns": {"schmertmann_settlement_mm": "Settlement in mm."},
    },
    "consolidation_settlement": {
        "category": "Consolidation",
        "brief": "Cc/Cr e-log(p) consolidation settlement with layer summation.",
        "parameters": {
            "layers": {"type": "array", "required": True, "description": "Array of {thickness, depth_to_center, e0, Cc, Cr, sigma_v0, sigma_p} dicts."},
            "delta_sigma": {"type": "float", "required": False, "description": "Uniform stress increase (kPa). Or use delta_sigmas array."},
            "q_net": {"type": "float", "required": False, "description": "Net pressure for auto stress distribution."},
            "B": {"type": "float", "required": False, "description": "Footing width for stress distribution (m)."},
            "stress_method": {"type": "str", "required": False, "default": "2:1", "description": "Stress distribution method (2:1 or boussinesq)."},
        },
        "returns": {"consolidation_settlement_mm": "Total consolidation settlement in mm.", "layer_breakdown": "Per-layer results."},
    },
    "combined_settlement_analysis": {
        "category": "Combined Analysis",
        "brief": "Full analysis: immediate + consolidation + secondary + time curve.",
        "parameters": {
            "q_applied": {"type": "float", "required": True, "description": "Applied pressure (kPa)."},
            "B": {"type": "float", "required": True, "description": "Footing width (m)."},
            "Es": {"type": "float", "required": False, "description": "Elastic modulus for immediate settlement (kPa)."},
            "consolidation_layers": {"type": "array", "required": False, "description": "Array of consolidation layer dicts."},
            "schmertmann_layers": {"type": "array", "required": False, "description": "Array of Schmertmann layer dicts."},
            "cv": {"type": "float", "required": False, "description": "Coefficient of consolidation (m2/yr)."},
            "Hdr": {"type": "float", "required": False, "description": "Drainage path length (m)."},
        },
        "returns": {"total_settlement_mm": "Total settlement.", "time_settlement_curve": "Settlement vs time data."},
    },
}
