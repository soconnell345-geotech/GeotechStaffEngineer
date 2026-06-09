"""Unified liquefaction adapter — single discoverable liquefaction tool.

Routes a liquefaction-triggering request to the correct underlying module by
**input type** and **method**:

    Input type
    ----------
    * CPT input  (q_c / f_s present)  -> liquepy B&I-2014 CPT (LPI/LSN/LDI).
    * SPT input  (N values present)   -> SPT procedure, per ``method``.

    Method (SPT only; CPT is always B&I-2014 via liquepy)
    ------
    * "bi2014"    (DEFAULT) -> liquepy B&I-2014 SPT triggering
                              (liquepy_agent.analyze_spt_liquefaction).
    * "nceer2001"           -> legacy NCEER / Youd-2001 SPT procedure
                              (seismic_geotech.evaluate_liquefaction), for
                              code-compliance work that still cites NCEER-2001.

Design decisions (locked by owner):
- B&I-2014 is the DEFAULT for both SPT and CPT.
- NCEER/Youd-2001 SPT stays available behind method="nceer2001".
- Unification happens here, at the agent/tool layer (analysis modules never
  import each other). The underlying per-module functions are untouched.

liquepy ships a packaged B&I-2014 CPT triggering object but NO packaged SPT
triggering object — its SPT entry points are field correlations only. The SPT
B&I-2014 path therefore lives in ``liquepy_agent.analyze_spt_liquefaction``,
which composes liquepy's tested B&I-2014 building blocks (CRR/rd/CSR/K_sigma).
"""

from funhouse_agent.adapters import clean_result


_CPT_KEYS = ("q_c", "qc", "f_s", "fs", "u_2", "u2")
_SPT_KEYS = ("N160", "n160", "N1_60", "n1_60", "N", "n_values")

# Method names the agent may guess; normalize to canonical method ids.
_METHOD_NORMALIZE = {
    "bi2014": "bi2014",
    "boulanger_idriss": "bi2014",
    "boulanger_idriss_2014": "bi2014",
    "boulanger_and_idriss_2014": "bi2014",
    "b&i": "bi2014",
    "bi": "bi2014",
    "nceer2001": "nceer2001",
    "nceer": "nceer2001",
    "youd2001": "nceer2001",
    "youd": "nceer2001",
    "seed_idriss": "nceer2001",
    "simplified": "nceer2001",
}


def _check_liquepy():
    from liquepy_agent import has_liquepy
    if not has_liquepy():
        raise ValueError(
            "liquepy is not installed (required for B&I-2014). "
            "Install with: pip install liquepy — or use method='nceer2001' "
            "for the native SPT procedure."
        )


def _detect_input_type(params: dict) -> str:
    """Return 'CPT' or 'SPT' from the parameter keys, or raise if ambiguous."""
    explicit = str(params.get("input_type", "")).strip().lower()
    if explicit in ("cpt", "spt"):
        return explicit.upper()

    has_cpt = any(k in params for k in _CPT_KEYS)
    has_spt = any(k in params for k in _SPT_KEYS)

    if has_cpt and not has_spt:
        return "CPT"
    if has_spt and not has_cpt:
        return "SPT"
    if has_cpt and has_spt:
        raise ValueError(
            "Ambiguous input: both CPT (q_c/f_s) and SPT (N) data present. "
            "Set input_type='CPT' or input_type='SPT'."
        )
    raise ValueError(
        "Could not detect input type. Provide CPT data (q_c, f_s) or SPT data "
        "(N160 blow counts), or set input_type explicitly."
    )


def _normalize_method(params: dict, input_type: str) -> str:
    raw = params.get("method")
    if raw is None:
        return "bi2014"  # default for both SPT and CPT
    key = str(raw).strip().lower()
    method = _METHOD_NORMALIZE.get(key)
    if method is None:
        raise ValueError(
            f"Unknown method '{raw}'. Use 'bi2014' (default, Boulanger & "
            f"Idriss 2014) or 'nceer2001' (NCEER / Youd et al. 2001 SPT)."
        )
    if input_type == "CPT" and method == "nceer2001":
        raise ValueError(
            "method='nceer2001' applies to SPT input only. CPT triggering uses "
            "Boulanger & Idriss (2014) via liquepy — omit method or use 'bi2014'."
        )
    return method


def _first(params: dict, *keys, default=None, required=False, label=""):
    for k in keys:
        if k in params and params[k] is not None:
            return params[k]
    if required:
        raise ValueError(
            f"Missing required parameter{(' ' + label) if label else ''}: "
            f"provide one of {list(keys)}."
        )
    return default


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _run_cpt_bi2014(params: dict) -> dict:
    _check_liquepy()
    from liquepy_agent import analyze_cpt_liquefaction

    result = analyze_cpt_liquefaction(
        depth=_first(params, "depth", required=True, label="depth"),
        q_c=_first(params, "q_c", "qc", required=True, label="q_c"),
        f_s=_first(params, "f_s", "fs", required=True, label="f_s"),
        u_2=_first(params, "u_2", "u2"),
        gwl=_first(params, "gwl", "gwt_depth", default=1.0),
        pga=_first(params, "pga", "amax_g", default=0.25),
        m_w=_first(params, "m_w", "magnitude", "M", default=7.5),
        a_ratio=params.get("a_ratio", 0.8),
        i_c_limit=params.get("i_c_limit", 2.6),
        cfc=params.get("cfc", 0.0),
        unit_wt_method=params.get("unit_wt_method", "robertson2009"),
        gamma_predrill=params.get("gamma_predrill", 17.0),
        s_g=params.get("s_g", 2.65),
        p_a=params.get("p_a", 101.0),
    )
    out = clean_result(result.to_dict())
    out["method"] = "bi2014"
    out["input_type"] = "CPT"
    return out


def _run_spt_bi2014(params: dict) -> dict:
    _check_liquepy()
    from liquepy_agent import analyze_spt_liquefaction

    result = analyze_spt_liquefaction(
        depth=_first(params, "depth", "depths", required=True, label="depth"),
        n1_60=_first(params, "N160", "n160", "N1_60", "n1_60", "N", "n_values",
                     required=True, label="N160"),
        fc=_first(params, "FC", "fc", "fines", required=True, label="FC"),
        gamma=_first(params, "gamma", "unit_weight", required=True, label="gamma"),
        amax_g=_first(params, "amax_g", "pga", "amax", required=True, label="amax_g"),
        gwt_depth=_first(params, "gwt_depth", "gwl", required=True, label="gwt_depth"),
        m_w=_first(params, "m_w", "magnitude", "M", default=7.5),
        c_0=params.get("c_0", 2.8),
    )
    return clean_result(result.to_dict())


def _run_spt_nceer2001(params: dict) -> dict:
    from seismic_geotech.liquefaction import evaluate_liquefaction

    layer_results = evaluate_liquefaction(
        layer_depths=_first(params, "depth", "depths", required=True, label="depth"),
        layer_N160=_first(params, "N160", "n160", "N1_60", "n1_60", "N", "n_values",
                          required=True, label="N160"),
        layer_FC=_first(params, "FC", "fc", "fines", required=True, label="FC"),
        layer_gamma=_first(params, "gamma", "unit_weight", required=True, label="gamma"),
        amax_g=_first(params, "amax_g", "pga", "amax", required=True, label="amax_g"),
        gwt_depth=_first(params, "gwt_depth", "gwl", required=True, label="gwt_depth"),
        M=_first(params, "m_w", "magnitude", "M", default=7.5),
        gamma_w=params.get("gamma_w", 9.81),
    )
    n_liq = sum(1 for r in layer_results if r.get("liquefiable"))
    min_fos = min((r["FOS_liq"] for r in layer_results), default=99.9)
    return clean_result({
        "method": "nceer2001",
        "input_type": "SPT",
        "n_layers": len(layer_results),
        "gwt_depth_m": round(float(_first(params, "gwt_depth", "gwl")), 2),
        "amax_g": round(float(_first(params, "amax_g", "pga", "amax")), 3),
        "m_w": round(float(_first(params, "m_w", "magnitude", "M", default=7.5)), 1),
        "min_fos": round(min_fos, 3),
        "n_liquefiable": n_liq,
        "layer_results": layer_results,
    })


def _run_liquefaction_analysis(params: dict) -> dict:
    """Unified entry point: route by input type and method."""
    input_type = _detect_input_type(params)
    method = _normalize_method(params, input_type)

    if input_type == "CPT":
        return _run_cpt_bi2014(params)
    # SPT
    if method == "nceer2001":
        return _run_spt_nceer2001(params)
    return _run_spt_bi2014(params)


METHOD_REGISTRY = {
    "liquefaction_analysis": _run_liquefaction_analysis,
}

METHOD_INFO = {
    "liquefaction_analysis": {
        "category": "Liquefaction",
        "brief": (
            "Unified liquefaction triggering. Auto-routes by input type: CPT "
            "(q_c/f_s) -> Boulanger & Idriss 2014 (liquepy, with LPI/LSN/LDI); "
            "SPT (N160) -> B&I-2014 by default, or NCEER/Youd-2001 via "
            "method='nceer2001' for code-compliance work."
        ),
        "parameters": {
            "depth": {"type": "array", "required": True,
                      "description": "Layer mid-depths / CPT depths (m)."},
            # SPT inputs
            "N160": {"type": "array", "required": False,
                     "description": "SPT (N1)60 blow counts (presence => SPT input)."},
            "FC": {"type": "array", "required": False,
                   "description": "Fines content (%) per layer (SPT)."},
            "gamma": {"type": "array", "required": False,
                      "description": "Total unit weight (kN/m3) per layer (SPT)."},
            "gwt_depth": {"type": "float", "required": False,
                          "description": "Groundwater depth (m) (SPT; alias gwl)."},
            "amax_g": {"type": "float", "required": False,
                       "description": "Peak ground acceleration (g) (SPT; alias pga)."},
            # CPT inputs
            "q_c": {"type": "array", "required": False,
                    "description": "Cone tip resistance (kPa) (presence => CPT input)."},
            "f_s": {"type": "array", "required": False,
                    "description": "Sleeve friction (kPa) (CPT)."},
            "u_2": {"type": "array", "required": False,
                    "description": "Pore pressure behind cone (kPa) (CPT, optional)."},
            "gwl": {"type": "float", "required": False,
                    "description": "Groundwater level depth (m) (CPT)."},
            "pga": {"type": "float", "required": False,
                    "description": "Peak ground acceleration (g) (CPT)."},
            # shared
            "m_w": {"type": "float", "required": False, "default": 7.5,
                    "description": "Moment magnitude (alias magnitude)."},
            "method": {"type": "str", "required": False, "default": "bi2014",
                       "allowed_values": ["bi2014", "nceer2001"],
                       "description": (
                           "bi2014 = Boulanger & Idriss 2014 (DEFAULT, both SPT "
                           "and CPT). nceer2001 = NCEER / Youd et al. 2001 SPT "
                           "procedure (SPT input only).")},
            "input_type": {"type": "str", "required": False,
                           "allowed_values": ["CPT", "SPT"],
                           "description": (
                               "Optional override if both/neither CPT and SPT "
                               "keys are present.")},
        },
        "returns": {
            "input_type": "Detected input type ('CPT' or 'SPT').",
            "method": "Triggering method used ('bi2014' or 'nceer2001').",
            "min_fos": "Minimum factor of safety against liquefaction.",
            "n_liquefiable": "Number of liquefiable layers/points.",
            "layer_results": "Per-layer CSR/CRR/FoS (SPT).",
            "lpi": "Liquefaction Potential Index (CPT only).",
            "lsn": "Liquefaction Severity Number (CPT only).",
            "ldi_m": "Lateral Displacement Index in m (CPT only).",
        },
    },
}
