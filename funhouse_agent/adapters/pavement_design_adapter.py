"""Adapter for the pavement_design module (AASHTO 1993 pavement design).

Bridges flat JSON parameters to pavement_design's dataclass/kwarg API.
UNITS: US customary (psi, pci, inches, kips, 18-kip ESALs) -- the AASHTO
1993 Guide is US-customary native; this module intentionally does NOT use
SI. NOTE (calc-package convention): results echo every defaulted /
midpoint-selected coefficient in 'notes' so reports can state the basis.
"""

from funhouse_agent.adapters import (apply_aliases, clean_result,
                                     reject_unknown_params, require_params)

_FLEX_ALIASES = {
    "esals": "w18", "design_esals": "w18", "w_18": "w18",
    "reliability": "reliability_pct", "r_pct": "reliability_pct",
    "subgrade_modulus_psi": "mr_psi", "mr": "mr_psi",
    "resilient_modulus_psi": "mr_psi",
    "seasonal_mr_psi": "monthly_mr_psi",
    "p_t": "pt", "p_o": "po", "dpsi": "delta_psi", "psi_loss": "delta_psi",
}

_FLEX_VALID = ("w18", "reliability_pct", "zr", "so", "mr_psi",
               "monthly_mr_psi", "pt", "po", "delta_psi", "layers",
               "thickness_increment_in", "enforce_minimums",
               "swelling", "frost", "design_period_yr")


def _run_flexible(params: dict) -> dict:
    from pavement_design import design_flexible_pavement

    p = apply_aliases(params, _FLEX_ALIASES)
    reject_unknown_params(p, _FLEX_VALID, method="flexible_pavement_design",
                          aliases=_FLEX_ALIASES)
    require_params(p, ["w18", "layers"], method="flexible_pavement_design",
                   valid=_FLEX_VALID)
    res = design_flexible_pavement(**p)
    return clean_result(res.to_dict())


_RIGID_ALIASES = {
    "esals": "w18", "design_esals": "w18", "w_18": "w18",
    "reliability": "reliability_pct", "r_pct": "reliability_pct",
    "modulus_of_rupture_psi": "sc_psi", "sc": "sc_psi", "s_c_psi": "sc_psi",
    "concrete_modulus_psi": "ec_psi", "ec": "ec_psi",
    "k": "k_pci", "k_value_pci": "k_pci",
    "subgrade_modulus_psi": "mr_psi", "mr": "mr_psi",
    "p_t": "pt", "p_o": "po", "dpsi": "delta_psi", "psi_loss": "delta_psi",
    "d_in": "slab_thickness_in", "slab_d_in": "slab_thickness_in",
}

_RIGID_VALID = ("w18", "sc_psi", "ec_psi", "reliability_pct", "zr", "so",
                "pt", "po", "delta_psi", "j", "pavement_type",
                "shoulder_type", "load_transfer_devices", "cd",
                "drainage_quality", "pct_saturation_time", "k_pci", "mr_psi",
                "composite_k", "slab_thickness_in", "thickness_increment_in",
                "swelling", "frost", "design_period_yr")


def _run_rigid(params: dict) -> dict:
    from pavement_design import design_rigid_pavement

    p = apply_aliases(params, _RIGID_ALIASES)
    reject_unknown_params(p, _RIGID_VALID, method="rigid_pavement_design",
                          aliases=_RIGID_ALIASES)
    require_params(p, ["w18", "sc_psi", "ec_psi"],
                   method="rigid_pavement_design", valid=_RIGID_VALID)
    res = design_rigid_pavement(**p)
    return clean_result(res.to_dict())


_TRAFFIC_ALIASES = {
    "growth_rate": "growth_rate_pct", "growth_pct": "growth_rate_pct",
    "design_period": "design_period_yr", "years": "design_period_yr",
    "dd": "directional_factor", "dl": "lane_factor",
    "lanes_per_direction": "num_lanes_per_direction",
    "lanes": "num_lanes_per_direction",
    "p_t": "pt",
}

_TRAFFIC_VALID = ("growth_rate_pct", "design_period_yr", "axle_groups",
                  "vehicles", "base_year_w18_two_way", "pavement_type",
                  "sn", "d_in", "pt", "directional_factor",
                  "num_lanes_per_direction", "lane_factor")


def _run_traffic(params: dict) -> dict:
    from pavement_design import compute_design_esals

    p = apply_aliases(params, _TRAFFIC_ALIASES)
    reject_unknown_params(p, _TRAFFIC_VALID, method="design_traffic_esals",
                          aliases=_TRAFFIC_ALIASES)
    res = compute_design_esals(**p)
    return clean_result(res.to_dict())


def _run_effective_mr(params: dict) -> dict:
    from geotech_references.aashto_1993.equations import \
        effective_roadbed_resilient_modulus

    p = apply_aliases(params, {"seasonal_mr_psi": "monthly_mr_psi",
                               "mr_values": "monthly_mr_psi"})
    reject_unknown_params(p, ("monthly_mr_psi",),
                          method="effective_subgrade_modulus")
    require_params(p, ["monthly_mr_psi"], method="effective_subgrade_modulus",
                   valid=("monthly_mr_psi",))
    return clean_result(effective_roadbed_resilient_modulus(
        p["monthly_mr_psi"]))


_PERF_ALIASES = {
    "reliability": "reliability_pct",
    "growth_rate": "growth_rate_pct",
    "base_year_esals": "base_year_w18",
    "k": "k_pci", "sc": "sc_psi", "ec": "ec_psi",
}

_PERF_VALID = ("pavement_type", "delta_psi_design", "base_year_w18",
               "growth_rate_pct", "swelling", "frost",
               "max_performance_period_yr", "initial_trial_yr", "sn",
               "mr_psi", "d_in", "sc_psi", "ec_psi", "j", "cd", "k_pci",
               "pt", "zr", "so", "reliability_pct")


def _run_performance(params: dict) -> dict:
    from geotech_references.aashto_1993.tables import \
        standard_normal_deviate_zr
    from pavement_design import estimate_performance_period

    p = apply_aliases(params, _PERF_ALIASES)
    reject_unknown_params(p, _PERF_VALID, method="performance_period",
                          aliases=_PERF_ALIASES)
    require_params(p, ["pavement_type", "delta_psi_design", "base_year_w18"],
                   method="performance_period", valid=_PERF_VALID)
    if p.get("zr") is None and p.get("reliability_pct") is not None:
        p["zr"] = standard_normal_deviate_zr(p.pop("reliability_pct"))["zr"]
    else:
        p.pop("reliability_pct", None)
    return clean_result(estimate_performance_period(**p))


_UFC_FLEX_VALID = ("passes_18kip", "cbr_subgrade", "cbr_base", "cbr_subbase",
                   "frost", "thickness_increment_in")

_UFC_RIGID_VALID = ("passes_18kip", "flexural_strength_psi", "k_pci",
                    "subgrade", "stabilized_foundation",
                    "thickness_increment_in")

_COMPARE_VALID = ("passes_18kip", "cbr_subgrade", "mr_psi",
                  "reliability_pct", "so", "pt", "cbr_base", "cbr_subbase",
                  "eac_psi", "ebs_psi", "esb_psi")

_UFC_ALIASES = {"passes": "passes_18kip", "esals": "passes_18kip",
                "w18": "passes_18kip", "cbr": "cbr_subgrade",
                "r_psi": "flexural_strength_psi",
                "modulus_of_rupture_psi": "flexural_strength_psi",
                "k": "k_pci", "reliability": "reliability_pct"}


def _run_flexible_ufc(params: dict) -> dict:
    from pavement_design import design_flexible_pavement_ufc

    p = apply_aliases(params, _UFC_ALIASES)
    reject_unknown_params(p, _UFC_FLEX_VALID,
                          method="flexible_pavement_design_ufc",
                          aliases=_UFC_ALIASES)
    require_params(p, ["passes_18kip", "cbr_subgrade"],
                   method="flexible_pavement_design_ufc",
                   valid=_UFC_FLEX_VALID)
    return clean_result(design_flexible_pavement_ufc(**p))


def _run_rigid_ufc(params: dict) -> dict:
    from pavement_design import design_rigid_pavement_ufc

    p = apply_aliases(params, _UFC_ALIASES)
    reject_unknown_params(p, _UFC_RIGID_VALID,
                          method="rigid_pavement_design_ufc",
                          aliases=_UFC_ALIASES)
    require_params(p, ["passes_18kip", "flexural_strength_psi"],
                   method="rigid_pavement_design_ufc",
                   valid=_UFC_RIGID_VALID)
    return clean_result(design_rigid_pavement_ufc(**p))


def _run_compare(params: dict) -> dict:
    from pavement_design import compare_flexible_pavement_methods

    p = apply_aliases(params, _UFC_ALIASES)
    reject_unknown_params(p, _COMPARE_VALID,
                          method="compare_pavement_methods",
                          aliases=_UFC_ALIASES)
    require_params(p, ["passes_18kip"], method="compare_pavement_methods",
                   valid=_COMPARE_VALID)
    return clean_result(compare_flexible_pavement_methods(**p))


METHOD_REGISTRY = {
    "flexible_pavement_design": _run_flexible,
    "rigid_pavement_design": _run_rigid,
    "design_traffic_esals": _run_traffic,
    "effective_subgrade_modulus": _run_effective_mr,
    "performance_period": _run_performance,
    "flexible_pavement_design_ufc": _run_flexible_ufc,
    "rigid_pavement_design_ufc": _run_rigid_ufc,
    "compare_pavement_methods": _run_compare,
}

_SWELL_DESC = ("Roadbed swelling spec (Appendix G / Fig G.4): {vr_in: "
               "potential vertical rise (in), ps_pct: percent of area "
               "subject to swell, theta: swell rate constant (~0.04-0.20)}. "
               "Requires design_period_yr.")
_FROST_DESC = ("Frost heave spec (Appendix G / Fig G.8): {phi_mm_day: heave "
               "rate, pf_pct: percent of area subject to heave, "
               "delta_psi_max: max loss (Fig G.7, from drainage quality x "
               "frost depth)}. Requires design_period_yr.")
_ENV_PARAMS = {
    "swelling": {"type": "object", "required": False,
                 "description": _SWELL_DESC},
    "frost": {"type": "object", "required": False,
              "description": _FROST_DESC},
    "design_period_yr": {"type": "number", "required": False,
                         "description": "Analysis period (yr); required "
                                        "with swelling/frost -- their dPSI "
                                        "loss is subtracted from the design "
                                        "budget before the solve."},
}

_LAYER_DESC = (
    "List of layer dicts, TOP-DOWN; first must be layer_type 'asphalt'. "
    "Each: {layer_type: asphalt|granular_base|granular_subbase|"
    "cement_treated_base|bituminous_treated_base; modulus_psi (EAC/EBS/ESB "
    "-- required on non-surface layers in design mode); ucs_7day_psi "
    "(cement-treated); marshall_stability_lb (bituminous-treated); a "
    "(direct coefficient override); m (drainage, unbound layers) or "
    "drainage_quality + pct_saturation_time ('<1%'|'1-5%'|'5-25%'|'>25%'); "
    "thickness_in (set on EVERY layer for a check instead of a design); "
    "min_thickness_in}."
)

METHOD_INFO = {
    "flexible_pavement_design": {
        "category": "pavement design",
        "brief": ("AASHTO 1993 flexible pavement design (Fig 3.1/3.2): "
                  "required SN + layered D1/D2/D3 split with drainage m, "
                  "Sec 3.1.4 minimums and forward check; or adequacy check "
                  "of a given section. US customary units."),
        "parameters": {
            "w18": {"type": "number", "required": True,
                    "description": "Design-lane 18-kip ESALs over the "
                                   "performance period (use "
                                   "design_traffic_esals to build it)."},
            "layers": {"type": "array", "required": True,
                       "description": _LAYER_DESC},
            "reliability_pct": {"type": "number", "required": False,
                                "description": "Reliability R % (50-99.99), "
                                               "Table 4.1 -> ZR. Or give zr."},
            "zr": {"type": "number", "required": False,
                   "description": "Standard normal deviate directly "
                                  "(overrides reliability_pct)."},
            "so": {"type": "number", "required": False, "default": 0.45,
                   "description": "Overall standard deviation (flexible "
                                  "range 0.40-0.50)."},
            "mr_psi": {"type": "number", "required": False,
                       "description": "Effective roadbed resilient modulus, "
                                      "psi (or monthly_mr_psi)."},
            "monthly_mr_psi": {"type": "array", "required": False,
                               "description": "Seasonal roadbed MR list "
                                              "(psi) -> Fig 2.3/2.4 "
                                              "effective MR."},
            "pt": {"type": "number", "required": False, "default": 2.5,
                   "description": "Terminal serviceability."},
            "po": {"type": "number", "required": False, "default": 4.2,
                   "description": "Initial serviceability."},
            "delta_psi": {"type": "number", "required": False,
                          "description": "Serviceability loss directly "
                                         "(else po - pt)."},
            "thickness_increment_in": {"type": "number", "required": False,
                                       "default": 0.5,
                                       "description": "Round-UP increment."},
            "enforce_minimums": {"type": "boolean", "required": False,
                                 "default": True,
                                 "description": "Apply Sec 3.1.4 minimum "
                                                "AC/base thicknesses."},
            **_ENV_PARAMS,
        },
        "returns": {"sn_required": "SN over the roadbed",
                    "sn_provided": "SN of the final section",
                    "layers": "per-layer a, m, thickness_in, basis",
                    "sn_stack": "SN required over each foundation",
                    "w18_capacity": "forward-check capacity",
                    "adequate": "bool"},
    },
    "rigid_pavement_design": {
        "category": "pavement design",
        "brief": ("AASHTO 1993 rigid (PCC slab) design (Fig 3.7): required "
                  "slab thickness D with k from direct value, MR/19.4, or "
                  "the full Sec 3.2 composite-k worksheet (iterated with D); "
                  "or adequacy check. US customary units."),
        "parameters": {
            "w18": {"type": "number", "required": True,
                    "description": "Design-lane 18-kip ESALs."},
            "sc_psi": {"type": "number", "required": True,
                       "description": "PCC modulus of rupture (28-day, "
                                      "third-point), psi."},
            "ec_psi": {"type": "number", "required": True,
                       "description": "PCC elastic modulus, psi."},
            "reliability_pct": {"type": "number", "required": False,
                                "description": "Reliability R % -> ZR "
                                               "(or give zr)."},
            "zr": {"type": "number", "required": False,
                   "description": "Standard normal deviate directly."},
            "so": {"type": "number", "required": False, "default": 0.35,
                   "description": "Overall standard deviation (rigid "
                                  "0.30-0.40)."},
            "pt": {"type": "number", "required": False, "default": 2.5,
                   "description": "Terminal serviceability (also in the "
                                  "strength-term exponent)."},
            "po": {"type": "number", "required": False, "default": 4.5,
                   "description": "Initial serviceability."},
            "delta_psi": {"type": "number", "required": False,
                          "description": "Serviceability loss directly."},
            "j": {"type": "number", "required": False,
                  "description": "Load transfer coefficient J directly "
                                 "(else Table 2.6 midpoint from "
                                 "pavement_type/shoulder_type/"
                                 "load_transfer_devices)."},
            "pavement_type": {"type": "string", "required": False,
                              "default": "plain_jointed_jrcp",
                              "allowed_values": ["plain_jointed_jrcp",
                                                 "crcp"],
                              "description": "For the Table 2.6 J lookup."},
            "shoulder_type": {"type": "string", "required": False,
                              "default": "asphalt",
                              "allowed_values": ["asphalt", "tied_pcc"],
                              "description": "For the Table 2.6 J lookup."},
            "load_transfer_devices": {"type": "boolean", "required": False,
                                      "default": True,
                                      "description": "Dowels present."},
            "cd": {"type": "number", "required": False, "default": 1.0,
                   "description": "Drainage coefficient directly (else "
                                  "Table 2.5 midpoint from "
                                  "drainage_quality)."},
            "drainage_quality": {"type": "string", "required": False,
                                 "allowed_values": ["excellent", "good",
                                                    "fair", "poor",
                                                    "very_poor"],
                                 "description": "For the Table 2.5 Cd "
                                                "lookup."},
            "pct_saturation_time": {"type": "string", "required": False,
                                    "allowed_values": ["<1%", "1-5%",
                                                       "5-25%", ">25%"],
                                    "description": "Moisture exposure for "
                                                   "the Cd lookup."},
            "k_pci": {"type": "number", "required": False,
                      "description": "Effective modulus of subgrade "
                                     "reaction directly, pci. Give exactly "
                                     "one of k_pci / mr_psi / composite_k."},
            "mr_psi": {"type": "number", "required": False,
                       "description": "Roadbed MR, psi -> simplified "
                                      "k = MR/19.4 (no subbase)."},
            "composite_k": {"type": "object", "required": False,
                            "description": "Sec 3.2 worksheet spec: "
                                           "{seasonal: [{mr_psi, ...}], "
                                           "dsb_in, esb_psi, dsg_ft, ls}; "
                                           "slab_d_in is iterated "
                                           "automatically."},
            "slab_thickness_in": {"type": "number", "required": False,
                                  "description": "Give to CHECK a slab "
                                                 "instead of designing."},
            "thickness_increment_in": {"type": "number", "required": False,
                                       "default": 0.5,
                                       "description": "Round-UP increment."},
            **_ENV_PARAMS,
        },
        "returns": {"d_required_in": "unrounded required slab thickness",
                    "d_provided_in": "rounded / supplied slab",
                    "k_pci": "effective k used", "k_basis": "how k was "
                    "obtained", "w18_capacity": "forward check",
                    "adequate": "bool"},
    },
    "design_traffic_esals": {
        "category": "pavement design",
        "brief": ("Design-lane cumulative 18-kip ESALs: axle-load spectrum "
                  "(Appendix D LEFs, single/tandem/triple), per-vehicle "
                  "truck factors, or a base-year total; compound growth + "
                  "directional DD + lane DL."),
        "parameters": {
            "axle_groups": {"type": "array", "required": False,
                            "description": "[{axle_config: single|tandem|"
                                           "triple, load_kips, daily_count "
                                           "(two-way)}]; LEFs at the "
                                           "assumed sn/d_in/pt."},
            "vehicles": {"type": "array", "required": False,
                         "description": "[{description, daily_count, "
                                        "truck_factor (ESALs/pass)}]."},
            "base_year_w18_two_way": {"type": "number", "required": False,
                                      "description": "First-year two-way "
                                                     "ESALs directly."},
            "growth_rate_pct": {"type": "number", "required": False,
                                "default": 0,
                                "description": "Annual growth rate %."},
            "design_period_yr": {"type": "number", "required": False,
                                 "default": 20,
                                 "description": "Performance period, yr."},
            "pavement_type": {"type": "string", "required": False,
                              "default": "flexible",
                              "allowed_values": ["flexible", "rigid"],
                              "description": "For LEF selection."},
            "sn": {"type": "number", "required": False, "default": 5.0,
                   "description": "Assumed SN for flexible LEFs."},
            "d_in": {"type": "number", "required": False, "default": 9.0,
                     "description": "Assumed slab D for rigid LEFs."},
            "pt": {"type": "number", "required": False, "default": 2.5,
                   "description": "Terminal serviceability for LEFs."},
            "directional_factor": {"type": "number", "required": False,
                                   "default": 0.5,
                                   "description": "DD split (0.3-0.7)."},
            "num_lanes_per_direction": {"type": "integer", "required": False,
                                        "default": 2,
                                        "description": "For the DL table."},
            "lane_factor": {"type": "number", "required": False,
                            "description": "DL fraction directly (else "
                                           "table midpoint)."},
        },
        "returns": {"w18_design_lane": "feed to the design methods",
                    "w18_two_way_total": "before DD/DL",
                    "growth_factor": "compound multiplier",
                    "axle_breakdown": "per-group LEFs"},
    },
    "effective_subgrade_modulus": {
        "category": "pavement design",
        "brief": ("Effective roadbed resilient modulus from seasonal "
                  "(monthly) MR values via the Fig 2.3/2.4 relative-damage "
                  "average (flexible design input)."),
        "parameters": {
            "monthly_mr_psi": {"type": "array", "required": True,
                               "description": "Seasonal roadbed MR values, "
                                              "psi (typically 12)."},
        },
        "returns": {"effective_mr_psi": "design MR",
                    "uf_values": "per-season relative damage"},
    },
    "flexible_pavement_design_ufc": {
        "category": "pavement design",
        "brief": ("UFC 3-250-01 (DoD roads/parking) flexible design — the "
                  "CBR-method ALTERNATIVE to AASHTO: Figure E-1 cover "
                  "curves, Table 7-2 minimums, optional seasonal-frost "
                  "reduced-subgrade-strength check. US customary."),
        "parameters": {
            "passes_18kip": {"type": "number", "required": True,
                             "description": "Design-life passes of the "
                                            "18-kip single axle (design "
                                            "lane). NOT interchangeable "
                                            "with AASHTO W18 conceptually "
                                            "— same count, different "
                                            "damage model."},
            "cbr_subgrade": {"type": "number", "required": True,
                             "description": "Subgrade design CBR (%)."},
            "cbr_base": {"type": "number", "required": False, "default": 80,
                         "description": "Base course design CBR."},
            "cbr_subbase": {"type": "number", "required": False,
                            "description": "Subbase CBR (omit for a "
                                           "2-layer section)."},
            "frost": {"type": "object", "required": False,
                      "description": "{uscs_class, finer_than_0_02mm_pct} "
                                     "-> Ch 19 reduced-subgrade-strength "
                                     "frost check; governing section "
                                     "reported."},
        },
        "returns": {"section": "layers + total (in)",
                    "frost_section": "when frost-susceptible",
                    "frost_governs": "bool"},
    },
    "rigid_pavement_design_ufc": {
        "category": "pavement design",
        "brief": ("UFC 3-250-01 rigid (plain concrete) design — Figure F-1 "
                  "(flexural strength x k x 18-kip passes) with the Eq 13-1 "
                  "stabilized-foundation reduction. US customary."),
        "parameters": {
            "passes_18kip": {"type": "number", "required": True,
                             "description": "18-kip single-axle passes."},
            "flexural_strength_psi": {"type": "number", "required": True,
                                      "description": "Concrete flexural "
                                                     "strength R (psi)."},
            "k_pci": {"type": "number", "required": False,
                      "description": "Modulus of subgrade reaction (or "
                                     "give subgrade for Table 10-1)."},
            "subgrade": {"type": "object", "required": False,
                         "description": "{uscs_group, moisture_pct} -> "
                                        "Table 10-1 k."},
            "stabilized_foundation": {"type": "object", "required": False,
                                      "description": "{ef_psi, hs_in} -> "
                                                     "Eq 13-1 reduction."},
        },
        "returns": {"hd_required_in": "F-1 solve",
                    "slab_provided_in": "rounded design slab"},
    },
    "compare_pavement_methods": {
        "category": "pavement design",
        "brief": ("Run AASHTO 1993 AND UFC 3-250-01 flexible designs on a "
                  "shared 18-kip pass count and compare sections side by "
                  "side, with every cross-guide assumption (CBR<->Mr "
                  "correlation, AASHTO-only reliability) stated in notes."),
        "parameters": {
            "passes_18kip": {"type": "number", "required": True,
                             "description": "18-kip single-axle passes "
                                            "(= AASHTO W18; LEF 1.0 by "
                                            "definition)."},
            "cbr_subgrade": {"type": "number", "required": False,
                             "description": "Subgrade CBR (Mr derived via "
                                            "FHWA 2555*CBR^0.64). Give "
                                            "this OR mr_psi."},
            "mr_psi": {"type": "number", "required": False,
                       "description": "Subgrade Mr (CBR derived by "
                                      "inversion)."},
            "reliability_pct": {"type": "number", "required": False,
                                "default": 95,
                                "description": "AASHTO-side reliability."},
            "so": {"type": "number", "required": False, "default": 0.45,
                   "description": "AASHTO overall standard deviation."},
            "cbr_base": {"type": "number", "required": False, "default": 80,
                         "description": "Base CBR (both sides; AASHTO EBS "
                                        "derived unless ebs_psi given)."},
            "cbr_subbase": {"type": "number", "required": False,
                            "default": 30,
                            "description": "Subbase CBR (both sides)."},
            "eac_psi": {"type": "number", "required": False,
                        "default": 400000,
                        "description": "AASHTO asphalt modulus."},
            "ebs_psi": {"type": "number", "required": False,
                        "description": "AASHTO base modulus override."},
            "esb_psi": {"type": "number", "required": False,
                        "description": "AASHTO subbase modulus override."},
        },
        "returns": {"aashto_1993": "SN + layers + total",
                    "ufc_3_250_01": "layers + total",
                    "delta_total_thickness_in": "UFC minus AASHTO",
                    "notes": "cross-guide assumptions"},
    },
    "performance_period": {
        "category": "pavement design",
        "brief": ("Predicted performance period of a DESIGNED section under "
                  "roadbed swelling / frost heave (the guide's Table 3.1 "
                  "iteration): environmental dPSI shrinks the traffic "
                  "budget until trial period and traffic-derived period "
                  "converge."),
        "parameters": {
            "pavement_type": {"type": "string", "required": True,
                              "allowed_values": ["flexible", "rigid"],
                              "description": "Which design equation gives "
                                             "the section's W18 capacity."},
            "delta_psi_design": {"type": "number", "required": True,
                                 "description": "Total design dPSI "
                                                "(po - pt)."},
            "base_year_w18": {"type": "number", "required": True,
                              "description": "First-year design-lane "
                                             "ESALs."},
            "growth_rate_pct": {"type": "number", "required": False,
                                "default": 0,
                                "description": "Annual traffic growth %."},
            "swelling": {"type": "object", "required": False,
                         "description": _SWELL_DESC},
            "frost": {"type": "object", "required": False,
                      "description": _FROST_DESC},
            "max_performance_period_yr": {"type": "number",
                                          "required": False, "default": 20,
                                          "description": "Cap / default "
                                                         "trial seed."},
            "sn": {"type": "number", "required": False,
                   "description": "Flexible: as-built structural number."},
            "mr_psi": {"type": "number", "required": False,
                       "description": "Flexible: effective roadbed MR."},
            "d_in": {"type": "number", "required": False,
                     "description": "Rigid: as-built slab thickness."},
            "sc_psi": {"type": "number", "required": False,
                       "description": "Rigid: PCC modulus of rupture."},
            "ec_psi": {"type": "number", "required": False,
                       "description": "Rigid: PCC elastic modulus."},
            "j": {"type": "number", "required": False,
                  "description": "Rigid: load transfer J."},
            "cd": {"type": "number", "required": False,
                   "description": "Rigid: drainage Cd."},
            "k_pci": {"type": "number", "required": False,
                      "description": "Rigid: design k."},
            "pt": {"type": "number", "required": False, "default": 2.5,
                   "description": "Terminal serviceability."},
            "zr": {"type": "number", "required": False,
                   "description": "Standard normal deviate (or give "
                                  "reliability_pct)."},
            "reliability_pct": {"type": "number", "required": False,
                                "description": "Reliability R % -> ZR."},
            "so": {"type": "number", "required": True,
                   "description": "Overall standard deviation (same as the "
                                  "design)."},
        },
        "returns": {"performance_period_yr": "converged period",
                    "rows": "per-iteration Table 3.1 worksheet",
                    "converged": "bool"},
    },
}
