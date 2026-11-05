"""Pile group adapter — rigid cap analysis, 6-DOF, efficiency."""

from funhouse_agent.adapters import (
    apply_aliases, check_nested_list, declare_aliases, field_keys,
    reject_unknown_params, require_keys, require_params, without_unused,
)
from pile_group import (
    GroupPile, create_rectangular_layout, group_efficiency_factor,
    block_failure_resistance, group_plan_dimensions, p_multiplier,
    GroupLoad, analyze_vertical_group_simple, analyze_group_6dof,
    meyerhof_group_settlement,
)
from pile_group.group_efficiency import (
    P_MULTIPLIER_METHODS, PILE_TYPES, SOIL_TYPES, SU_LIMIT_KPA,
)


# Keys a piles[] dict may carry: GroupPile's fields. Anything else was read
# with .get and silently left out, e.g. 'batter' ran the pile vertical
# (GM-58).
_PILE_KEYS = field_keys(GroupPile)
# Words the near-miss matcher cannot place. Suggestions only, never renamed.
_PILE_KEY_HINTS = {
    "batter": ("batter_x", "batter_y"), "ka": ("axial_stiffness",),
    "kl": ("lateral_stiffness",), "kaxial": ("axial_stiffness",),
    "klateral": ("lateral_stiffness",), "name": ("label",),
    "id": ("label",), "capacity": ("axial_capacity_compression",),
}


def _build_piles(params):
    if "n_rows" in params and "n_cols" in params:
        # Accept a single 'spacing' for both directions.
        sx = params.get("spacing_x", params.get("spacing"))
        sy = params.get("spacing_y", params.get("spacing"))
        if sx is None or sy is None:
            raise ValueError(
                "Rectangular layout requires spacing_x and spacing_y (m), "
                "or a single 'spacing' applied to both directions."
            )
        piles = create_rectangular_layout(
            n_rows=params["n_rows"], n_cols=params["n_cols"],
            spacing_x=sx, spacing_y=sy,
            axial_stiffness=params.get("axial_stiffness"), lateral_stiffness=params.get("lateral_stiffness"),
        )
        for p in piles:
            if params.get("axial_capacity_compression"):
                p.axial_capacity_compression = params["axial_capacity_compression"]
            if params.get("axial_capacity_tension"):
                p.axial_capacity_tension = params["axial_capacity_tension"]
        return piles
    elif "piles" in params:
        check_nested_list(params["piles"], _PILE_KEYS, method="pile_group",
                          label="piles", hints=_PILE_KEY_HINTS)
        for pd in params["piles"]:
            require_keys(pd, ["x", "y"], method="pile_group", item_label="piles[]")
        return [GroupPile(
            x=pd["x"], y=pd["y"], batter_x=pd.get("batter_x", 0.0), batter_y=pd.get("batter_y", 0.0),
            axial_stiffness=pd.get("axial_stiffness"), lateral_stiffness=pd.get("lateral_stiffness"),
            axial_capacity_compression=pd.get("axial_capacity_compression"),
            axial_capacity_tension=pd.get("axial_capacity_tension"), label=pd.get("label", ""),
        ) for pd in params["piles"]]
    raise ValueError("Provide n_rows/n_cols for rectangular layout or 'piles' array.")


def _build_load(params):
    return GroupLoad(Vx=params.get("Vx", 0.0), Vy=params.get("Vy", 0.0), Vz=params.get("Vz", 0.0),
                     Mx=params.get("Mx", 0.0), My=params.get("My", 0.0), Mz=params.get("Mz", 0.0))


# Every top-level parameter the group-analysis methods consume.
_GROUP_VALID_PARAMS = (
    "n_rows", "n_cols", "spacing_x", "spacing_y", "spacing", "piles",
    "axial_stiffness", "lateral_stiffness", "axial_capacity_compression",
    "axial_capacity_tension", "Vx", "Vy", "Vz", "Mx", "My", "Mz",
)


#: What the simple method ignores: it carries no lateral load or torsion and
#: uses no pile stiffness (GM-110), so these are refused rather than dropped.
_SIMPLE_IGNORES = ("Vx", "Vy", "Mz", "axial_stiffness", "lateral_stiffness")


# The group methods as (result, inputs in force): their own code path, also
# run by ``calc_package.pile_group_package`` (GM-141). The inputs in force
# are the piles and load built, and the layout inputs given. ``method``
# names the caller in error messages.

def _layout_used(params, piles, load):
    used = {"piles": piles, "load": load}
    used.update({k: params[k] for k in ("n_rows", "n_cols", "spacing_x",
                                        "spacing_y", "spacing")
                 if params.get(k) is not None})
    return used


def run_pile_group_simple(params, *, method="pile_group_simple"):
    params = without_unused(
        params, METHOD_INFO["pile_group_simple"]["refused_parameters"])
    reject_unknown_params(
        params, [k for k in _GROUP_VALID_PARAMS if k not in _SIMPLE_IGNORES],
        method=method)
    piles, load = _build_piles(params), _build_load(params)
    return (analyze_vertical_group_simple(piles, load),
            _layout_used(params, piles, load))


def run_pile_group_6dof(params, *, method="pile_group_6dof"):
    reject_unknown_params(params, _GROUP_VALID_PARAMS, method=method)
    piles = _build_piles(params)
    missing = [p.label or f"#{i}" for i, p in enumerate(piles)
               if p.axial_stiffness is None]
    if missing:
        raise ValueError(
            f"{method} needs every pile's axial_stiffness (kN/m) — "
            f"missing on {missing[:6]}{' ...' if len(missing) > 6 else ''}. "
            "Give 'axial_stiffness' at the top level with n_rows/n_cols, or on "
            "each piles[] entry, and 'lateral_stiffness' (kN/m) as well when "
            "vertical piles must carry Vx/Vy. For vertical load and moments "
            "only, pile_group_simple needs no stiffness.")
    load = _build_load(params)
    return analyze_group_6dof(piles, load), _layout_used(params, piles, load)


def _with_uplift_note(result, load):
    """The result's dict, with the group uplift note when a pile carries
    tension (GM-289 and its review); a group with no pile in tension is
    unchanged."""
    out = result.to_dict()
    note = tension_note(out["pile_forces"], load.Vz)
    if note:
        out["group_uplift_note"] = note
    return out


def _run_pile_group_simple(params):
    result, used = run_pile_group_simple(params)
    return _with_uplift_note(result, used["load"])


def _run_pile_group_6dof(params):
    result, used = run_pile_group_6dof(params)
    return _with_uplift_note(result, used["load"])


_EFFICIENCY_PARAMS = (
    "n_rows", "n_cols", "pile_diameter", "spacing", "spacing_x",
    "spacing_y", "soil_type", "cap_in_contact", "cu", "pile_type",
    "pile_length", "su1", "su2", "row_position", "spacing_diameter_ratio",
    "p_multiplier_method",
)

_PILE_TYPE_TEXT = {
    "driven": "driven piles", "drilled_shaft": "drilled shafts",
    "cfa": "CFA piles", "ddp": "drilled displacement piles",
    "micropile": "micropiles",
}


def _eta_g_basis(pile_type: str, s_over_b: float) -> str:
    """Where eta_g comes from, in the print's words, for the choices."""
    if pile_type == "driven":
        text = ("GEC-12 Sec. 7.2.2.1 (Vol I p. 282): driven piles in "
                "cohesionless soil not underlain by a weak deposit, with no "
                "jetting or predrilling, 1.0 (jetting or predrilling 'can "
                "result in group efficiencies less than 1', no value "
                "printed); Sec. 7.2.2.2 (p. 284): clay with su < 2 ksf and "
                "the cap not in firm contact with the ground, 0.65 at s = "
                "2.5 diameters to 1.0 at 6, linear between; su < 2 ksf with "
                "the cap in firm contact, or su above 2 ksf, 1.0. UFC "
                "3-220-20 Table 6-26 (p. 432) prints the same")
    else:
        text = (f"UFC 3-220-20 Table 6-26 (p. 432), the "
                f"{_PILE_TYPE_TEXT[pile_type]} column, interpolated between "
                f"the printed spacings (p. 431)")
    text += (". The group resistance is the lesser of n eta_g R_n and the "
             "resistance to block failure (GEC-12 Vol I p. 284 step 4; UFC 3-220-20 Eq. "
             "6-37); for a group in cohesionless soil over a weak deposit, "
             "block failure is checked too (GEC-12 Vol I p. 282 item 2)")
    if s_over_b < 3.0:
        text += (". Below 3 diameters: GEC-12 recommends at least 3 "
                 "(Vol I pp. 282, 284), UFC 3-220-20 the same (p. 431)")
    return text


def _p_multiplier_spacing(params):
    """s/D for the p-multiplier: spacing_diameter_ratio, else the spacing
    over the diameter. The p-multiplier takes the spacing in the direction
    of the load, so spacing_x and spacing_y that differ, with no single
    spacing, are refused, naming spacing_diameter_ratio."""
    if params.get("spacing_diameter_ratio") is not None:
        return params["spacing_diameter_ratio"]
    s = params.get("spacing")
    sx, sy = params.get("spacing_x"), params.get("spacing_y")
    if s is None and sx is not None and sy is not None and sx == sy:
        s = sx
    if s is None or params.get("pile_diameter") is None:
        raise ValueError(
            "group_efficiency (p-multiplier): give spacing_diameter_ratio, "
            "the centre-to-centre spacing of the rows in the direction of "
            "the load over the pile diameter, or that spacing as 'spacing' "
            "with pile_diameter.")
    return s / params["pile_diameter"]


def _run_group_efficiency(params):
    from geotech_common.choices import MODULE, choice, how_given
    reject_unknown_params(params, _EFFICIENCY_PARAMS,
                          method="group_efficiency")
    wants_eta = params.get("soil_type") is not None
    wants_block = any(params.get(k) is not None
                      for k in ("pile_length", "su1", "su2"))
    wants_pm = params.get("row_position") is not None
    if not (wants_eta or wants_block or wants_pm):
        raise ValueError(
            "group_efficiency: nothing to work out. Give soil_type "
            "('cohesionless' or 'cohesive') with pile_diameter and spacing "
            "for the group efficiency factor eta_g (GEC-12 Sec. 7.2.2; UFC "
            "3-220-20 Table 6-26); pile_length with su1 and su2 (or one cu "
            "for both) and the layout (n_rows, n_cols, spacing, "
            "pile_diameter) for block failure (GEC-12 Eqs. 7-36, 7-37); or "
            "row_position for the lateral p-multiplier (GEC-12 Vol I p. 370; UFC "
            "3-220-20 Table 6-38).")
    results, choices = {}, []
    if wants_eta:
        require_params(params, ["pile_diameter"],
                       method="group_efficiency (eta_g)",
                       valid=list(_EFFICIENCY_PARAMS))
        sx = params.get("spacing_x", params.get("spacing"))
        sy = params.get("spacing_y", params.get("spacing"))
        given = [v for v in (sx, sy) if v is not None]
        if not given:
            raise ValueError(
                "group_efficiency (eta_g): missing 'spacing' (centre-to-"
                "centre, m), or spacing_x and spacing_y.")
        s = min(given)
        d = params["pile_diameter"]
        pile_type = params.get("pile_type", "driven")
        r = group_efficiency_factor(
            params["soil_type"], s / d,
            cap_in_contact=params.get("cap_in_contact"),
            su_kPa=params.get("cu"), pile_type=pile_type)
        results["group_efficiency"] = round(r["eta_g"], 4)
        results["s_over_b"] = round(s / d, 3)
        results["group_efficiency_row"] = r["row"]
        choices.append(choice(
            "group efficiency factor eta_g",
            f"{r['eta_g']:.3f} at s/b = {s / d:.2f} ({r['row']}, "
            f"{_PILE_TYPE_TEXT[pile_type]})", MODULE,
            basis=_eta_g_basis(pile_type, s / d)))
        choices.append(choice(
            "pile type (UFC 3-220-20 Table 6-26 column)", pile_type,
            how_given(pile_type, "driven", "pile_type"),
            alternatives=[f"pile_type '{t}'" for t in PILE_TYPES
                          if t != pile_type]))
        if r["cap_in_contact_taken"]:
            choices.append(choice(
                "pile cap contact with the ground",
                "not in contact (the lower eta_g)", MODULE,
                basis="cap_in_contact not given, and the row depends on it",
                alternatives=[f"cap in firm contact (cap_in_contact true): "
                              f"eta_g {r['alternative']:.3f}"]))
        if r["su_taken"]:
            choices.append(choice(
                "clay's undrained strength against 2 ksf",
                "su < 2 ksf (the lower eta_g)", MODULE,
                basis="cu not given, and the row depends on it (2 ksf = "
                      f"{SU_LIMIT_KPA:.2f} kPa)",
                alternatives=["su of 2 ksf or more (cu given): eta_g "
                              "1.000"]))
        if sx is not None and sy is not None and sx != sy:
            choices.append(choice(
                "spacing for eta_g", f"{s:g} m, the smaller of spacing_x "
                f"{sx:g} m and spacing_y {sy:g} m", MODULE,
                basis="the print takes one centre-to-centre spacing; the "
                      "closer piles give the lower eta_g"))
    if wants_block:
        require_params(params, ["n_rows", "n_cols", "pile_diameter",
                                "pile_length"],
                       method="group_efficiency (block failure)",
                       valid=list(_EFFICIENCY_PARAMS))
        sx = params.get("spacing_x", params.get("spacing"))
        sy = params.get("spacing_y", params.get("spacing"))
        if sx is None or sy is None:
            raise ValueError(
                "group_efficiency (block failure): missing 'spacing' "
                "(centre-to-centre, m), or spacing_x and spacing_y.")
        cu = params.get("cu")
        su1 = params.get("su1", cu)
        su2 = params.get("su2", cu)
        missing = [n for n, v in (("su1", su1), ("su2", su2)) if v is None]
        if missing:
            raise ValueError(
                f"group_efficiency (block failure): give "
                f"{' and '.join(missing)} (kPa): su1 the weighted average "
                f"undrained strength along the group's perimeter over the "
                f"embedded length, su2 the average to 2B below the toes "
                f"(GEC-12 Eq. 7-36, Vol I p. 286), or one cu for both.")
        B, Z = group_plan_dimensions(params["n_rows"], params["n_cols"],
                                     sx, sy, params["pile_diameter"])
        pile_type = params.get("pile_type", "driven")
        r = block_failure_resistance(B, Z, params["pile_length"], su1, su2,
                                     pile_type=pile_type)
        results["block_failure_kN"] = round(r["R_ng_kN"], 1)
        results["block_failure_phi"] = r["phi_bl"]
        results["block_failure_factored_kN"] = (
            None if r["R_r_kN"] is None else round(r["R_r_kN"], 1))
        if r["phi_bl"] is None:
            results["block_failure_factored_note"] = (
                f"No held page prints a resistance factor for block failure "
                f"of {_PILE_TYPE_TEXT[pile_type]}: GEC-12 Table 7-1 (Vol I p. 224) "
                f"gives 0.60 (driven piles, cohesive soil) and UFC 3-220-20 "
                f"Table 6-27 (p. 433) 0.60 for driven piles, 0.55 for "
                f"drilled shafts and 0.60 for micropiles, with fine-grained "
                f"soil below the group; the nominal resistance is given.")
        else:
            choices.append(choice(
                "resistance factor for block failure",
                f"phi_bl = {r['phi_bl']:.2f} ({_PILE_TYPE_TEXT[pile_type]})",
                how_given(params.get("pile_type"), "driven", "pile_type"),
                basis=("GEC-12 Table 7-1 (Vol I p. 224): 'Block Failure, phi_bl, "
                       "Cohesive, 0.60'; UFC 3-220-20 Table 6-27 (p. 433, "
                       "AASHTO 2020): driven piles 0.60, drilled shafts 0.55, "
                       "micropiles 0.60, fine-grained (cohesive) soil below "
                       "the group"
                       + ("; AASHTO: reduced by 20 % for micropiles in "
                          "marginal ground conditions, not applied here"
                          if pile_type == "micropile" else ""))))
        results["block_failure_Nc"] = round(r["Nc"], 3)
        results["block_width_m"] = round(B, 3)
        results["block_length_m"] = round(Z, 3)
        choices.append(choice(
            "bearing capacity factor Nc (block failure)", f"{r['Nc']:.2f}",
            MODULE,
            basis=(f"GEC-12 Eq. 7-37 (Vol I p. 286; UFC 3-220-20 Eq. 6-39): Nc = "
                   f"5[1 + D/(5B)][1 + B/(5Z)] = {r['Nc_eq_7_37']:.2f}"
                   + (", limited to 9" if r["Nc_eq_7_37"] > 9 else
                      ", below 9 for this shallow or wide group (GEC-12: "
                      "'generally 9', by the equation for small embedment "
                      "and/or large widths)")
                   + f"; B = {B:.2f} m and Z = {Z:.2f} m to the outer faces "
                     "of the outer piles (GEC-12 Fig. 7-31)")))
        # UFC App. B-5.4 measures the block centre to centre of the outer
        # piles; given as the other reading, with its resistance
        b_cc = sorted(((params["n_cols"] - 1) * sx,
                       (params["n_rows"] - 1) * sy))
        centres = ("centre to centre of the outer piles, as UFC 3-220-20 "
                   "App. B-5.4 (pp. 637-638) takes them (18 ft for its 7x7 "
                   "group of 1 ft piles at 3 ft)")
        if b_cc[0] > 0:
            r_cc = block_failure_resistance(b_cc[0], b_cc[1],
                                            params["pile_length"], su1, su2)
            centres += (f": B = {b_cc[0]:.2f} m, Z = {b_cc[1]:.2f} m, "
                        f"nominal {r_cc['R_ng_kN']:.1f} kN")
        choices.append(choice(
            "group width and length B and Z (block failure)",
            f"to the outer faces of the outer piles: B = {B:.2f} m, "
            f"Z = {Z:.2f} m", MODULE,
            basis="GEC-12 Fig. 7-31; GEC-12 Vol. III p. 62 takes them "
                  "'from exterior pile edge to exterior pile edge'",
            alternatives=[centres]))
        one = [n for n in ("su1", "su2") if params.get(n) is None]
        if one:
            choices.append(choice(
                "undrained strength along the sides and below the toes",
                f"cu = {cu:g} kPa for "
                + ("both" if len(one) == 2 else one[0]),
                how_given(cu, None, "cu"),
                basis="GEC-12 Eq. 7-36 (Vol I p. 286) takes su1, the weighted "
                      "average along the group's perimeter over the "
                      "embedded length, and su2, the average to 2B below "
                      "the toes (UFC 3-220-20 p. 433: 2B to 3B, whichever "
                      "gives the lower)",
                alternatives=["su1 and su2 given separately"]))
    if wants_pm:
        sd = _p_multiplier_spacing(params)
        method = params.get("p_multiplier_method", "aashto")
        pm = p_multiplier(params["row_position"], sd, method)
        results["p_multiplier"] = round(pm, 3)
        basis = (
            "AASHTO as GEC-12 prints it (Vol I pp. 367, 370): 0.8, 0.4 and 0.3 "
            "for the lead, second and third and later rows at 3b, 1.0, "
            "0.85 and 0.7 at 5b, interpolated between. From 5b, a straight "
            "line to 1.00 at 6b: the module's reading of UFC 3-220-20 p. "
            "465, 'AASHTO (2020) guidance, extrapolated to a pile spacing "
            "of 6·b' (Table 6-38, p. 466, prints 1.00 for s/b > 6). The "
            "sentence could also mean AASHTO's 3b-5b line carried on to "
            "6b, which gives 0.9 for the third and later rows at 6b (rows "
            "1 and 2 reach 1.0 by then). GEC-12 Vol I p. 365: consider whether "
            "lower values, as in its Table 7-23, apply"
            if method == "aashto" else
            "Mokwa (1999), UFC 3-220-20 Table 6-38: p_m = a (s/b) + c "
            "<= 1 by row")
        choices.append(choice(
            "p-multiplier", f"{pm:.3f} (row {params['row_position']}, "
            f"s/D = {sd:.2f})",
            how_given(method, "aashto", "p_multiplier_method"),
            basis=basis,
            alternatives=[f"p_multiplier_method '{m}': "
                          f"{p_multiplier(params['row_position'], sd, m):.3f}"
                          for m in P_MULTIPLIER_METHODS if m != method]
            + _pm_line_to_6b(params["row_position"], sd, pm, method)))
    results["choices"] = choices
    return results


def _pm_line_to_6b(row: int, sd: float, pm: float, method: str) -> list:
    """The other reading of UFC p. 465 between 5b and 6b, AASHTO's 3b-5b
    line carried on (at most 1), where it differs from the module's."""
    if method != "aashto" or not 5.0 < sd <= 6.0:
        return []
    p3, p5 = p_multiplier(row, 3.0), p_multiplier(row, 5.0)
    line = min(p5 + (p5 - p3) / 2.0 * (sd - 5.0), 1.0)
    if abs(line - pm) < 5e-4:
        return []
    return [f"AASHTO's 3b-5b line carried on to s/D = {sd:.2f} (the other "
            f"reading of UFC 3-220-20 p. 465): {line:.3f}"]


# Group uplift (GM-289): GEC-12 Sec. 7.2.3.2, UFC 3-220-20 Eqs. 6-40 to
# 6-43. The group_uplift tool runs it on a rectangular layout; the pile
# group calc package on the piles it ran, when the group carries net
# tension. Both go through group_uplift_check.

#: The inputs of the block that group_uplift_check reads from a call.
UPLIFT_BLOCK_PARAMS = ("pile_length", "soil_type", "unit_weight", "layers",
                       "cu", "cap_weight", "pile_type")
_UPLIFT_LAYER_KEYS = ("thickness", "unit_weight")


def _uplift_layers(params, method):
    """(thickness, unit weight) pairs from ``layers``, else one layer of
    ``unit_weight`` over the embedded length."""
    if params.get("layers") is not None:
        check_nested_list(params["layers"], _UPLIFT_LAYER_KEYS,
                          method=method, label="layers")
        for layer in params["layers"]:
            require_keys(layer, list(_UPLIFT_LAYER_KEYS), method=method,
                         item_label="layers[]")
        return [(layer["thickness"], layer["unit_weight"])
                for layer in params["layers"]]
    if params.get("unit_weight") is None:
        raise ValueError(
            f"{method}: give unit_weight, the soil's effective unit weight "
            f"(kN/m3; below the water table the buoyant), or layers "
            f"[{{thickness, unit_weight}}, ...] from the top of the block "
            f"down to the pile toes.")
    return [(params["pile_length"], params["unit_weight"])]


#: What a block over part of the group, the piles in tension, adds to
#: each choice's ``what`` (GM-289 review).
TENSION_PILES = "piles in tension"


def group_uplift_check(B, Z, n_piles, params, *, piles_resistance=None,
                       uplift_load=None, method="group_uplift",
                       group_piles=None, cap_share=None, over=None):
    """The group's uplift resistance (``group_uplift_resistance``) on a
    block B x Z at the toes (m, to the outer pile faces) from ``params``'
    :data:`UPLIFT_BLOCK_PARAMS`, as (results, choices): the block, its
    factored resistance, the piles' sum (``piles_resistance``, the sum of
    their factored tension resistances, kN) and the lesser, and with
    ``uplift_load`` (kN) the load over it.

    ``cap_share`` (GM-289 review): the block is over the piles in tension
    of a larger group, ``group_piles`` piles under the cap (the small-group
    rule counts them), and takes that share of ``cap_weight``, the whole
    cap's, as GEC-12 Vol. III p. 207 takes the cap over the back row in
    tension. ``over`` names those piles in the block's choice (GM-289
    review 2: the outer row of them), else 'the n piles in tension'."""
    from geotech_common.choices import MODULE, choice, how_given
    from pile_group.group_efficiency import (
        group_uplift_resistance, small_group_reduction, GROUP_UPLIFT_PHI,
        SMALL_GROUP_PILES)
    soil_type = params["soil_type"]
    pile_type = params.get("pile_type") or "driven"
    cap_total = params.get("cap_weight") or 0.0
    part = cap_share is not None
    cap = cap_total * cap_share if part else cap_total
    group_piles = n_piles if group_piles is None else group_piles
    layers = _uplift_layers(params, method)
    if soil_type == "cohesive" and params.get("cu") is None:
        raise ValueError(
            f"{method}: soil_type 'cohesive' needs cu, the weighted average "
            f"undrained shear strength over the embedded length along the "
            f"group's perimeter (kPa; GEC-12 Eq. 7-41's su).")
    r = group_uplift_resistance(
        B, Z, params["pile_length"], n_piles, soil_type, layers,
        su=params.get("cu"), cap_weight=cap, pile_type=pile_type,
        piles_resistance=piles_resistance, group_piles=group_piles)
    sand = soil_type == "cohesionless"
    out = {"block_width_m": round(r["B_m"], 3),
           "block_length_m": round(r["Z_m"], 3)}
    if sand:
        out["block_top_width_m"] = round(r["top_width_m"], 3)
        out["block_top_length_m"] = round(r["top_length_m"], 3)
    out["block_volume_m3"] = round(r["volume_m3"], 2)
    out["block_soil_weight_kN"] = round(r["soil_weight_kN"], 1)
    out["block_cap_weight_kN"] = round(cap, 1)
    if not sand:
        out["block_side_shear_kN"] = round(r["shear_kN"], 1)
    if part:
        out["block_piles"] = n_piles
        out["block_group_piles"] = group_piles
        out["block_cap_share"] = round(cap_share, 4)
    if params.get("layers") is not None:
        out["block_layers"] = [
            {"top_m": round(s["top_m"], 3), "bottom_m": round(s["bottom_m"], 3),
             "unit_weight_kN_m3": round(s["unit_weight"], 3),
             "volume_m3": round(s["volume_m3"], 2),
             "weight_kN": round(s["weight_kN"], 1)} for s in r["slices"]]
    out["group_uplift_block_kN"] = round(r["R_ug_kN"], 1)
    out["group_uplift_phi"] = (None if r["phi_ug"] is None
                               else round(r["phi_ug"], 3))
    out["group_uplift_block_factored_kN"] = (
        None if r["R_r_kN"] is None else round(r["R_r_kN"], 1))
    out["piles_tension_resistance_kN"] = (
        None if piles_resistance is None else round(piles_resistance, 1))
    governing = r.get("governing_kN")
    out["group_uplift_resistance_kN"] = (
        None if governing is None else round(governing, 1))
    out["group_uplift_governs"] = {"block": "block",
                                   "piles": "individual piles"}.get(
                                       r.get("governs"))
    notes = []
    if r["phi_ug"] is None:
        notes.append(
            f"No held page prints a resistance factor for group uplift of "
            f"{_PILE_TYPE_TEXT[pile_type]}: GEC-12 Table 7-1 (Vol I p. 224) gives "
            f"0.50 for driven piles and UFC 3-220-20 Table 6-28 (p. 434) "
            f"0.50 for driven piles, 0.45 for drilled shafts and 0.50 for "
            f"micropiles; the block's nominal resistance is given.")
    elif piles_resistance is None:
        notes.append(
            "The piles' factored tension resistance was not given "
            "(axial_capacity_tension, phi_up R_s per pile, GEC-12 Eq. "
            "7-38): the block's factored resistance is given, and the "
            "group's is the lesser of it and the piles' sum (GEC-12 Vol I p. 291), "
            "so no more than it.")
    if uplift_load is not None:
        out["uplift_load_kN"] = round(uplift_load, 1)
        if governing is not None:
            out["group_uplift_utilization"] = round(uplift_load / governing,
                                                    3)
            out["group_uplift_adequate"] = uplift_load <= governing
    if notes:
        out["group_uplift_note"] = " ".join(notes)

    choices = []
    if governing is not None:
        which = ("the piles' sum governs" if r["governs"] == "piles"
                 else "the block governs")
        ufc = piles_resistance + 0.9 * cap
        ufc_text = (f"{ufc:.1f} kN with the cap's {cap:g} kN"
                    if cap else "the cap's weight not given")
        choices.append(choice(
            "group uplift resistance",
            f"{governing:.1f} kN, the lesser of the piles' sum "
            f"{piles_resistance:.1f} kN and the block's {r['R_r_kN']:.1f} "
            f"kN: {which}", MODULE,
            basis=("GEC-12 Vol I p. 291 (AASHTO 2014): 'limit uplift resistance "
                   "of a pile group to the lesser of: 1. The sum of "
                   "individual pile resistance in axial tension. 2. The "
                   "resistance in axial tension of the pile group "
                   "considered as a block', the sum phi_up R_s n (Eq. "
                   "7-40), the block phi_ug R_ug (Eq. 7-39); UFC 3-220-20 "
                   "Eq. 6-40 (p. 435) the same"),
            alternatives=[
                f"UFC 3-220-20 Eq. 6-40 adds to the piles' sum each pile's "
                f"and the cap's effective weight, factored by 0.9 (p. 434), "
                f"n(R_r,s + W_r,e) + W_r,e,cap: {ufc_text}, and 0.9 times "
                f"each pile's effective weight (not taken here)"]))
    if sand:
        block = (f"the soil above the toes within sides at 1H:4V, "
                 f"{r['B_m']:.2f} x {r['Z_m']:.2f} m at the toes to "
                 f"{r['top_width_m']:.2f} x {r['top_length_m']:.2f} m at "
                 f"the top, the piles taken as soil")
        basis = ("GEC-12 Vol I p. 291, Fig. 7-33 (after Tomlinson 1994): 'the "
                 "effective weight of the block of soil extending upward "
                 "from the pile toe level at a slope of 1H:4V ... the "
                 "weights of the piles within the soil block are "
                 "considered equal to the weight of the soil'; UFC 3-220-20 "
                 "Eq. 6-41 (p. 436), and for layers the plan dimensions "
                 "at each layer's bottom")
    else:
        block = (f"the pile/soil block {r['B_m']:.2f} x {r['Z_m']:.2f} m "
                 f"by the embedded length, its effective weight and the "
                 f"undrained shear on its sides, 2D(B + Z)su, the piles "
                 f"taken as soil")
        basis = ("GEC-12 Eq. 7-41 (Vol I p. 292), R_ug = 2D(B + Z)su + W_g, Fig. "
                 "7-34 (after Tomlinson 1994); UFC 3-220-20 Eqs. 6-42 and "
                 "6-43 (p. 436) the same; UFC p. 435: 'Usually, the "
                 "difference between the unit weight of the foundation "
                 "material and the soil may be neglected by treating all "
                 "materials as having the unit weight of soil'")
    if part:
        block = f"over {over or f'the {n_piles} piles in tension'}: " + block
    choices.append(choice(
        "group uplift block", block,
        how_given(soil_type, None, "soil_type"),
        basis=basis + ("; B and Z to the outer faces of the outer piles "
                       "(GEC-12 Vol. III pp. 52-53 and 205 take the group "
                       "length 'from exterior pile edge to exterior pile "
                       "edge', and one row's width as the pile's)")
        + ("; over the piles in tension of a group in compression, as "
           "GEC-12 Vol. III checks the back row of an abutment group: "
           "'only a single row of piles is providing the tension "
           "resistance rather than the entire pile group' (pp. 51, 203)"
           if part else "")))
    if r["phi_ug"] is not None:
        driven = pile_type == "driven"
        chosen = f"phi_ug = {r['phi_ug']:.2f}"
        if r["reduction"] < 1.0:
            chosen += (f" ({r['phi_ug_table']:.2f} reduced by 20 % for a "
                       f"small group of {group_piles} piles)")
        elif part and small_group_reduction(n_piles, pile_type) < 1.0:
            chosen += (f" (the group's {group_piles} piles under the cap, "
                       f"not a small group)")
        basis = ("GEC-12 Table 7-1 (Vol I p. 224): 'Group Uplift Resistance, "
                 "phi_ug | Sand and Clay | 0.50'; UFC 3-220-20 Table 6-28 "
                 "(p. 434, AASHTO 2020), 'group block uplift, all soils': "
                 "driven piles 0.50, drilled shafts 0.45, micropiles 0.50")
        basis += ("; GEC-12 Vol I p. 225: for a small group, 'less than 3 piles "
                  "per substructure unit in this manual', AASHTO (2014) "
                  "reduces the factors of Tables 7-1 and 7-2 by 20 %"
                  if driven else
                  "; UFC 3-220-20 Table 6-28 note A: reduced by 20 % for small "
                  "pile groups (<5 piles)")
        if pile_type == "micropile":
            basis += (", and for micropiles in marginal ground conditions, "
                      "not applied here")
        alternatives = []
        if driven and small_group_reduction(group_piles, pile_type, "ufc") \
                < r["reduction"]:
            phi_ufc = GROUP_UPLIFT_PHI[pile_type] * 0.8
            alternatives.append(
                f"UFC 3-220-20 Table 6-28 note A's small group (fewer than "
                f"{SMALL_GROUP_PILES['ufc']} piles): phi_ug {phi_ufc:.2f}, "
                f"the block {phi_ufc * r['R_ug_kN']:.1f} kN")
        choices.append(choice(
            "resistance factor for group uplift", chosen,
            how_given(params.get("pile_type"), "driven", "pile_type"),
            basis=basis, alternatives=alternatives))
    cap_text = f"{cap:g} kN"
    cap_basis = ("GEC-12 Eq. 7-41's W_g is 'the effective weight of the "
                 "pile/soil block including pile cap weight'; UFC 3-220-20 "
                 "p. 435 includes the cap's effective weight in the block in "
                 "either soil")
    if part and cap_total:
        cap_text = (f"{cap:.1f} kN, {n_piles}/{group_piles} of the cap's "
                    f"{cap_total:g} kN, the share over the piles in tension")
        cap_basis += ("; GEC-12 Vol. III p. 207 takes the cap over the back "
                      "row in tension only, 'conservatively taken as the "
                      "tributary' area, half the cap's width over one of two "
                      "rows; the module takes the share by the count of "
                      "piles, the tributary area of evenly spaced piles")
    choices.append(choice(
        "pile cap weight in the block", cap_text,
        how_given(params.get("cap_weight"), 0.0, "cap_weight"),
        basis=cap_basis + ("; left out, none, the safe side"
                           if cap_total == 0 else ""),
        alternatives=(["the cap's effective weight (cap_weight, kN), where "
                       "the load does not already carry it"]
                      if cap_total == 0 else
                      ["the tributary area of an uneven layout, as "
                       "cap_weight to the group_uplift tool"] if part
                      else [])))
    if part:
        for c in choices:
            c["what"] = f"{c['what']}, {TENSION_PILES}"
    return out, choices


_UPLIFT_PARAMS = ("n_rows", "n_cols", "spacing", "spacing_x", "spacing_y",
                  "pile_diameter", "axial_capacity_tension", "uplift_load",
                  "piles_under_cap") + UPLIFT_BLOCK_PARAMS


def _run_group_uplift(params):
    method = "group_uplift"
    reject_unknown_params(params, _UPLIFT_PARAMS, method=method)
    require_params(params, ["n_rows", "n_cols", "pile_diameter",
                            "pile_length", "soil_type"],
                   method=method, valid=list(_UPLIFT_PARAMS))
    sx = params.get("spacing_x", params.get("spacing"))
    sy = params.get("spacing_y", params.get("spacing"))
    if sx is None or sy is None:
        raise ValueError(
            f"{method}: missing 'spacing' (centre-to-centre, m), or "
            f"spacing_x and spacing_y.")
    n = params["n_rows"] * params["n_cols"]
    under_cap = params.get("piles_under_cap")
    if under_cap is not None and not (isinstance(under_cap, int)
                                      and under_cap >= n):
        raise ValueError(
            f"{method}: piles_under_cap must be a whole number, the piles "
            f"under the cap, at least the {n} in the block; got "
            f"{under_cap!r}.")
    B, Z = group_plan_dimensions(params["n_rows"], params["n_cols"], sx, sy,
                                 params["pile_diameter"])
    q = params.get("axial_capacity_tension")
    out, choices = group_uplift_check(
        B, Z, n, params, piles_resistance=None if q is None else n * q,
        uplift_load=params.get("uplift_load"), method=method,
        group_piles=under_cap)
    out["n_piles"] = n
    out["choices"] = choices
    return out


def tension_note(pile_forces, Vz: float):
    """What a rigid-cap result says when a pile carries tension (GM-289
    review: GEC-12 Vol. III checks the back row in tension of a group in
    compression as a block, pp. 51, 203), from its ``pile_forces``
    (``axial_kN``, compression positive); None with no pile in tension."""
    tension = [-pf["axial_kN"] for pf in pile_forces if pf["axial_kN"] < 0]
    if not tension:
        return None
    n, k, total = len(pile_forces), len(tension), sum(tension)
    rule = ("GEC-12 Sec. 7.2.3.2 (Vol I p. 291; UFC 3-220-20 Eq. 6-40) limits the "
            "uplift resistance of piles in tension to the lesser of the sum "
            "of their tension resistances and their resistance as a block; "
            "the pile-by-pile check does not cover the block")
    if k == n:
        return (f"Every pile carries tension; the group carries a net "
                f"uplift of {-Vz:,.1f} kN. {rule}: check it with "
                f"group_uplift (uplift_load {-Vz:g}).")
    note = (f"{k} of {n} piles carry tension, {total:,.1f} kN in all (the "
            f"most {max(tension):,.1f} kN). {rule}, and GEC-12 Vol. III "
            f"checks the back row in tension of a group in compression so "
            f"(pp. 51, 203): check them with group_uplift over the piles in "
            f"tension (n_rows 1 for one row, piles_under_cap {n}, "
            f"uplift_load {total:g}).")
    if Vz < 0:
        note += (f" The group carries a net uplift of {-Vz:,.1f} kN: check "
                 f"the whole group as well (uplift_load {-Vz:g}).")
    return note


# Meyerhof (1976) SPT group-settlement aliases (accept a few common synonyms
# for the SI plan dimensions / pressure / SPT count without silently dropping).
_MEYERHOF_ALIASES = {
    "B": "group_width", "width": "group_width",
    "L": "group_length", "Z": "group_length", "length": "group_length",
    "Q": "load_kN", "load": "load_kN", "load_permanent_kN": "load_kN",
    "pf": "pf_kPa", "net_pressure_kPa": "pf_kPa",
    "DB": "embedment_DB", "embedment": "embedment_DB", "embedment_m": "embedment_DB",
    "N": "N160", "N1_60": "N160", "N1_60_avg": "N160",
}

_MEYERHOF_VALID_PARAMS = (
    "group_width", "group_length", "N160", "load_kN", "pf_kPa", "embedment_DB",
    "silty_sand",
)


def _run_meyerhof_group_settlement(params):
    params = apply_aliases(params, _MEYERHOF_ALIASES)
    reject_unknown_params(params, _MEYERHOF_VALID_PARAMS,
                          method="meyerhof_group_settlement")
    require_params(params, ["group_width", "group_length", "N160"],
                   method="meyerhof_group_settlement",
                   valid=_MEYERHOF_VALID_PARAMS)
    if params.get("load_kN") is None and params.get("pf_kPa") is None:
        raise ValueError(
            "meyerhof_group_settlement: provide either load_kN (total "
            "permanent load, kN) or pf_kPa (design foundation pressure, "
            "kPa)."
        )
    from geotech_common.choices import choice, how_given
    db = params.get("embedment_DB", 0.0)
    silty = bool(params.get("silty_sand", False))
    out = meyerhof_group_settlement(
        group_width=params["group_width"],
        group_length=params["group_length"],
        N160=params["N160"],
        load_kN=params.get("load_kN"),
        pf_kPa=params.get("pf_kPa"),
        embedment_DB=db,
        silty_sand=silty,
    )
    out["choices"] = [
        choice("embedment for If (GEC-12 Eq. 7-52)",
               f"{db:g} m: If = {out['influence_factor']:.3f}",
               how_given(db, 0.0, "embedment_DB"),
               basis=("D' = 2/3 of it. GEC-12 Vol. I p. 319 takes D' = 2/3 "
                      "of the pile embedded length (UFC 3-220-20 Eq. 6-47, "
                      "Z/(12B), the same); GEC-12 Vol. III Block 15 (Table "
                      "D-23) takes 2/3 of the embedment into the bearing "
                      "stratum. Left at 0, If = 1.0, the upper bound"),
               alternatives=["the pile embedded length",
                             "the embedment into the bearing stratum"]),
        choice("Meyerhof equation",
               "Eq. 7-51, silty sand (8)" if silty else "Eq. 7-50, sand (4)",
               how_given(silty, False, "silty_sand"),
               basis=("GEC-12 Vol I p. 319, for a pile group in a homogeneous sand "
                      "not underlain by a more compressible soil "
                      "(conservative); over cohesive deposits GEC-12 sends "
                      "to Sec. 7.3.5.3, the equivalent footing. Elastic "
                      "compression of the piles is not included (Eq. 7-48)"),
               alternatives=["silty_sand " + ("false: Eq. 7-50" if silty
                                              else "true: Eq. 7-51")]),
    ]
    return out


METHOD_REGISTRY = {
    "pile_group_simple": _run_pile_group_simple,
    "pile_group_6dof": _run_pile_group_6dof,
    "group_efficiency": _run_group_efficiency,
    "group_uplift": _run_group_uplift,
    "meyerhof_group_settlement": _run_meyerhof_group_settlement,
}

_UPLIFT_NOTE_RETURN = (
    "Only when a pile carries tension: how many and their tension, to be "
    "checked as a block as well with group_uplift (GEC-12 Sec. 7.2.3.2; "
    "Vol. III checks the back row in tension of a group in compression), "
    "and the whole group in net tension (Vz < 0). Report it.")

METHOD_INFO = {
    "pile_group_simple": {
        "category": "Pile Group",
        "brief": "Simplified elastic vertical group analysis.",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Rows for rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Columns for rectangular layout."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m). Or give a single 'spacing' for both directions."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m). Or give a single 'spacing' for both directions."},
            "spacing": {"type": "float", "required": False, "description": "Uniform center-to-center spacing (m), used for both x and y."},
            "piles": {"type": "array", "required": False, "description": "Explicit pile list [{x, y, batter_x, batter_y, ...}] instead of n_rows/n_cols."},
            "Vz": {"type": "float", "required": True, "description": "Vertical load (kN)."},
            "Mx": {"type": "float", "required": False, "default": 0.0, "description": "Moment about x (kN-m)."},
            "My": {"type": "float", "required": False, "default": 0.0, "description": "Moment about y (kN-m)."},
            "axial_capacity_compression": {"type": "float", "required": False, "description": "Per-pile compression capacity (kN) for utilization checks (rectangular layout)."},
            "axial_capacity_tension": {"type": "float", "required": False, "description": "Per-pile tension capacity (kN) (rectangular layout)."},
        },
        "returns": {"pile_forces": "Per-pile axial forces.", "max_compression_kN": "Max compression.", "group_uplift_note": _UPLIFT_NOTE_RETURN},
    },
    "pile_group_6dof": {
        "category": "Pile Group",
        "brief": "General 6-DOF rigid cap analysis (axial + lateral + moments).",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Rows for rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Columns."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m). Or give a single 'spacing' for both directions."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m). Or give a single 'spacing' for both directions."},
            "spacing": {"type": "float", "required": False, "description": "Uniform center-to-center spacing (m), used for both x and y."},
            "piles": {"type": "array", "required": False, "description": "Explicit pile list [{x, y, batter_x, batter_y, axial_stiffness, lateral_stiffness, label}] instead of n_rows/n_cols."},
            "axial_stiffness": {"type": "float", "required": True, "description": "Pile-head axial stiffness ka (kN/m), e.g. from a load test or E*A/L for an end-bearing pile. REQUIRED for every pile: give it here with n_rows/n_cols, or on each piles[] entry."},
            "lateral_stiffness": {"type": "float", "required": False, "description": "Pile-head lateral stiffness kl (kN/m), here with n_rows/n_cols or on each piles[] entry. Needed for vertical piles to carry Vx/Vy (0 if omitted)."},
            "Vx": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load x (kN)."},
            "Vy": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load y (kN)."},
            "Vz": {"type": "float", "required": False, "default": 0.0, "description": "Vertical load (kN)."},
            "Mx": {"type": "float", "required": False, "default": 0.0, "description": "Moment about x (kN-m)."},
            "My": {"type": "float", "required": False, "default": 0.0, "description": "Moment about y (kN-m)."},
            "Mz": {"type": "float", "required": False, "default": 0.0, "description": "Torsion about z (kN-m)."},
            "axial_capacity_compression": {"type": "float", "required": False, "description": "Per-pile compression capacity (kN) for utilization checks (rectangular layout)."},
            "axial_capacity_tension": {"type": "float", "required": False, "description": "Per-pile tension capacity (kN) (rectangular layout)."},
        },
        "returns": {"pile_forces": "Per-pile forces.", "cap_displacement": "Cap displacement.", "group_uplift_note": _UPLIFT_NOTE_RETURN},
    },
    "group_efficiency": {
        "category": "Pile Group",
        "brief": "Pile group axial and lateral group effects as GEC-12 and UFC 3-220-20 print them: the group efficiency factor eta_g, block failure, the row p-multiplier.",
        "parameters": {
            "n_rows": {"type": "int", "required": False, "description": "Number of rows (for block failure)."},
            "n_cols": {"type": "int", "required": False, "description": "Number of columns (for block failure)."},
            "pile_diameter": {"type": "float", "required": False, "description": "Pile diameter or width b (m)."},
            "spacing": {"type": "float", "required": False, "description": "Centre-to-centre pile spacing (m), both directions."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m), if it differs from spacing. eta_g takes the smaller of x and y."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m), if it differs from spacing."},
            "soil_type": {"type": "str", "required": False, "allowed_values": list(SOIL_TYPES), "description": "Gives the group efficiency factor eta_g: 'cohesionless' (UFC's coarse-grained) or 'cohesive' (fine-grained), the stratum giving most of the resistance (GEC-12 Vol I p. 281). Driven piles, GEC-12 Secs. 7.2.2.1-7.2.2.2 (Vol I pp. 282, 284): cohesionless not underlain by a weak deposit, 1.0; clay with su < 2 ksf and the cap not in firm contact with the ground, 0.65 at s = 2.5b to 1.0 at 6b, linear between; else 1.0. Other pile types: UFC 3-220-20 Table 6-26 (see pile_type). Below the table's minimum spacing (2.5b driven piles and drilled shafts, 3b the others) the call is refused. The group resistance is the lesser of n*eta_g*R_n and block failure."},
            "cap_in_contact": {"type": "bool", "required": False, "description": "Whether the pile cap is in firm contact with the ground (sets eta_g's row). Left out where the row depends on it, the lower eta_g (not in contact) is taken and the result's choices give the other."},
            "cu": {"type": "float", "required": False, "description": "Undrained shear strength of the clay (kPa): sets eta_g's row against 2 ksf (95.76 kPa) for soil_type 'cohesive' (left out where it matters, su < 2 ksf is taken, said in the choices); and, for block failure, su1 and su2 where those are not given."},
            "pile_type": {"type": "str", "required": False, "default": "driven", "allowed_values": list(PILE_TYPES), "description": "The column of UFC 3-220-20 Table 6-26 for eta_g: 'driven' (GEC-12), 'drilled_shaft', 'cfa', 'ddp' (drilled displacement) or 'micropile'; and the resistance factor for block failure: 0.60 driven (GEC-12 Table 7-1, Vol I p. 224), 0.55 drilled shafts, 0.60 micropiles (UFC 3-220-20 Table 6-27, p. 433); none printed for 'cfa' or 'ddp' (nominal only)."},
            "pile_length": {"type": "float", "required": False, "description": "Pile embedded length D (m). With su1 and su2 (or cu), n_rows, n_cols, spacing and pile_diameter, gives the resistance to block failure: GEC-12 Eq. 7-36, R = 2D(B+Z)su1 + B*Z*su2*Nc, Nc = 5[1 + D/(5B)][1 + B/(5Z)] <= 9 (Eq. 7-37), B and Z the group's width and length to the outer pile faces (Fig. 7-31). Nominal, and factored by pile_type's printed resistance factor (block_failure_factored_kN)."},
            "su1": {"type": "float", "required": False, "description": "Block failure: weighted average undrained shear strength along the group's perimeter over the embedded length (kPa); for cohesionless soils along the sides, their weighted average unit shaft resistance (GEC-12 Vol I p. 287)."},
            "su2": {"type": "float", "required": False, "description": "Block failure: average undrained shear strength below the toes to a depth of 2B (kPa; UFC 3-220-20 p. 433: 2B to 3B, whichever lower), that of the weak layer for a group in sand over it."},
            "row_position": {"type": "int", "required": False, "description": "Row number, 1 = the lead row (UFC 3-220-20 Fig. 6-25): gives the row's lateral p-multiplier p_m."},
            "spacing_diameter_ratio": {"type": "float", "required": False, "description": "s/b for the p-multiplier, s the spacing of the rows in the direction of the load (from spacing/pile_diameter if omitted). Below 3 the call is refused (p_m is printed from 3b)."},
            "p_multiplier_method": {"type": "str", "required": False, "default": "aashto", "allowed_values": list(P_MULTIPLIER_METHODS), "description": "'aashto' (default): AASHTO's p_m as GEC-12 prints it (Vol I pp. 367, 370): rows 1, 2, 3+ = 0.8, 0.4, 0.3 at 3b and 1.0, 0.85, 0.7 at 5b, interpolated (0.90, 0.625, 0.5 at 4b, as GEC-12 Vol. III takes them), to 1.0 at 6b and beyond (UFC 3-220-20 Table 6-38). 'mokwa': Mokwa (1999), UFC 3-220-20 Table 6-38: p_m = 0.06(s/b)+0.64, 0.11(s/b)+0.34, 0.14(s/b)+0.16, 0.16(s/b)+0.04 for rows 1, 2, 3, 4+, each <= 1."},
        },
        "returns": {"group_efficiency": "eta_g (with soil_type).", "s_over_b": "The s/b eta_g was read at.", "group_efficiency_row": "The printed row read.", "block_failure_kN": "Nominal resistance to block failure (with pile_length).", "block_failure_phi": "The printed resistance factor for block failure (null for CFA or DDP, none printed).", "block_failure_factored_kN": "phi x the nominal resistance (null where no factor is printed).", "block_failure_Nc": "Nc used.", "block_width_m": "Group width B.", "block_length_m": "Group length Z.", "p_multiplier": "The row's p-multiplier (with row_position).", "choices": "Each value's basis and alternatives; report them."},
    },
    "group_uplift": {
        "category": "Pile Group",
        "brief": "Uplift resistance of a pile group in net tension, or of the piles in tension of a group (GEC-12 Vol. III checks the back row in tension of an abutment group in compression), as GEC-12 Sec. 7.2.3.2 and UFC 3-220-20 Eqs. 6-40 to 6-43 print it: the lesser of the sum of the piles' factored tension resistances and the group's as a block (cohesionless soil: the effective weight of the soil within sides at 1H:4V from the toes; cohesive soil: the block's weight plus 2D(B+Z)su on its sides), factored by phi_ug.",
        "parameters": {
            "n_rows": {"type": "int", "required": True, "description": "Rows of the group, or of the piles in tension (1 for one row, as GEC-12 Vol. III checks a back row in tension)."},
            "n_cols": {"type": "int", "required": True, "description": "Columns."},
            "spacing": {"type": "float", "required": False, "description": "Centre-to-centre pile spacing (m), both directions."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m), if it differs from spacing."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m), if it differs from spacing."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter or width b (m): the block's width and length at the toes are to the outer faces of the outer piles, (n - 1)s + b."},
            "pile_length": {"type": "float", "required": True, "description": "Pile embedded length D (m), the block's height."},
            "soil_type": {"type": "str", "required": True, "allowed_values": list(SOIL_TYPES), "description": "'cohesionless': the block is the soil within sides at 1H:4V from the toes up (GEC-12 Fig. 7-33; UFC 3-220-20 Eq. 6-41), its effective weight. 'cohesive': the block is B x Z x D, its effective weight plus the undrained shear on its sides, 2D(B+Z)su (GEC-12 Eq. 7-41; UFC 3-220-20 Eqs. 6-42, 6-43). The piles are taken as soil."},
            "unit_weight": {"type": "float", "required": False, "description": "The soil's effective unit weight (kN/m3; below the water table the buoyant) over the embedded length. Or give layers."},
            "layers": {"type": "array", "required": False, "description": "The soil from the top of the block down, [{thickness (m), unit_weight (effective, kN/m3)}, ...], to the toes at least, in place of one unit_weight (e.g. the water table inside the block). In cohesionless soil each layer's volume is taken between its bottom and top plan dimensions (UFC 3-220-20 p. 436; GEC-12 Vol. III Eq. D-1)."},
            "cu": {"type": "float", "required": False, "description": "soil_type 'cohesive': the weighted average undrained shear strength over the embedded length along the group's perimeter (kPa), GEC-12 Eq. 7-41's su."},
            "cap_weight": {"type": "float", "required": False, "default": 0.0, "description": "The pile cap's effective weight over the block (kN), part of the block (GEC-12 Eq. 7-41's W_g 'including pile cap weight'; UFC 3-220-20 p. 435); for the piles in tension of a group, the cap over them (GEC-12 Vol. III p. 207 takes the tributary area over the back row in tension). Leave it out (0, the safe side) where the uplift load already carries it."},
            "pile_type": {"type": "str", "required": False, "default": "driven", "allowed_values": list(PILE_TYPES), "description": "The block's resistance factor phi_ug: 0.50 driven (GEC-12 Table 7-1, Vol I p. 224), 0.45 drilled shafts, 0.50 micropiles (UFC 3-220-20 Table 6-28, p. 434); none printed for 'cfa' or 'ddp' (nominal only). Reduced by 20 % for a small group: fewer than 3 driven piles (GEC-12 Vol I p. 225), fewer than 5 of the others (UFC 3-220-20 Table 6-28 note A)."},
            "axial_capacity_tension": {"type": "float", "required": False, "description": "One pile's factored tension resistance phi_up R_s (kN; GEC-12 Eq. 7-38, phi_up from Table 7-1 by the static method, 0.60 with a static load test or 0.50 dynamic testing with signal matching, Table 7-2; for a small group reduced by 20 %, GEC-12 Vol I p. 225). With it, the piles' sum n phi_up R_s (Eq. 7-40) and the governing resistance, the lesser."},
            "uplift_load": {"type": "float", "required": False, "description": "The factored uplift on the piles in the block (kN, positive): the net uplift on a group in tension, or the summed tension of the piles in tension; for the utilization, the load over the governing resistance."},
            "piles_under_cap": {"type": "int", "required": False, "description": "The piles under the cap when the block is part of the group (the piles in tension), for the small-group rule, which counts the piles per substructure unit (GEC-12 Vol I p. 225). At least n_rows x n_cols."},
        },
        "returns": {"block_width_m": "Block width B at the toes.", "block_length_m": "Block length Z at the toes.", "block_top_width_m": "Cohesionless: the block's width at the top.", "block_top_length_m": "Cohesionless: its length at the top.", "block_volume_m3": "The block's volume.", "block_soil_weight_kN": "The soil's effective weight in the block (piles taken as soil).", "block_cap_weight_kN": "The cap weight added.", "block_side_shear_kN": "Cohesive: 2D(B+Z)su.", "block_layers": "With layers: each layer's share.", "group_uplift_block_kN": "The block's nominal uplift resistance R_ug.", "group_uplift_phi": "phi_ug applied (null where none is printed).", "group_uplift_block_factored_kN": "phi_ug R_ug.", "piles_tension_resistance_kN": "n phi_up R_s (with axial_capacity_tension).", "group_uplift_resistance_kN": "The group's factored uplift resistance, the lesser of the two (null without axial_capacity_tension).", "group_uplift_governs": "'block' or 'individual piles'.", "group_uplift_utilization": "uplift_load over the governing resistance.", "group_uplift_adequate": "uplift_load <= the governing resistance.", "group_uplift_note": "What is missing or not printed; report it.", "choices": "Each value's basis and alternatives; report them."},
    },
    "meyerhof_group_settlement": {
        "category": "Pile Group",
        "brief": "Meyerhof (1976) SPT pile-group settlement in sand, GEC-12 Eqs. 7-50 to 7-52: S[in] = 4·pf[ksf]·If·√B[ft]/N1(60) (8 for silty sand). SI in, mm out.",
        "parameters": {
            "group_width": {"type": "float", "required": True, "description": "Group plan width B, the SMALLER plan dimension (m)."},
            "group_length": {"type": "float", "required": True, "description": "Group plan length Z, the larger plan dimension (m). Used with group_width to form the plan area when load_kN is given."},
            "N160": {"type": "float", "required": True, "description": "Average corrected SPT N1(60) within a depth B below the pile toe (dimensionless)."},
            "load_kN": {"type": "float", "required": False, "description": "Total unfactored permanent load Q (kN); pf = Q/(B·Z), B and Z to the outer pile faces (GEC-12 Vol. III p. 62). Provide this OR pf_kPa."},
            "pf_kPa": {"type": "float", "required": False, "description": "Design foundation pressure pf (kPa), supplied directly. Takes precedence over load_kN. Provide this OR load_kN."},
            "embedment_DB": {"type": "float", "required": False, "default": 0.0, "description": "The embedment D whose 2/3 is D' in If = 1 − D'/(8B) >= 0.5 (Eq. 7-52), m. GEC-12 Vol. I p. 319 and UFC 3-220-20 Eq. 6-47 take the pile embedded length; GEC-12 Vol. III Table D-23 the embedment into the bearing stratum. Default 0.0 (If = 1.0, the upper bound)."},
            "silty_sand": {"type": "bool", "required": False, "default": False, "description": "True: Eq. 7-51 for silty sand (coefficient 8, twice Eq. 7-50's)."},
        },
        "returns": {"settlement_mm": "Group settlement (mm).", "settlement_in": "Group settlement (inches).", "influence_factor": "Embedment factor If.", "pf_kPa": "Design foundation pressure used (kPa).", "equation": "'7-50' or '7-51'.", "choices": "The embedment and equation taken, with the print's alternatives."},
    },
}

declare_aliases(METHOD_INFO, {"meyerhof_group_settlement": _MEYERHOF_ALIASES})

# What pile_group_simple ignores, refused with the reason (GM-110), so a
# lateral load is never dropped without a word; a 0 carries none and passes.
_LATERAL = {
    "reason": ("pile_group_simple distributes only Vz, Mx and My; lateral "
               "loads (Vx, Vy) and torsion (Mz) need pile_group_6dof"),
    "accepts": [0]}
_STIFFNESS = ("pile_group_simple assumes a rigid cap on equal piles and "
              "uses no pile stiffness; pile_group_6dof takes it")
METHOD_INFO["pile_group_simple"]["refused_parameters"] = {
    "Vx": _LATERAL, "Vy": _LATERAL, "Mz": _LATERAL,
    "axial_stiffness": _STIFFNESS, "lateral_stiffness": _STIFFNESS}


# What a left-out input with no default stands for (BACKLOG GM-165):
# no pile-head lateral stiffness, and the s/D from the spacing.
from funhouse_agent.adapters import declare_when_absent  # noqa: E402
declare_when_absent(METHOD_INFO, {
    "pile_group_6dof": {
        "lateral_stiffness": ("0", "with no pile-head lateral stiffness, "
                              "vertical piles carry no horizontal load (Vx, "
                              "Vy)")},
    "group_efficiency": {
        "spacing_diameter_ratio": ("spacing / pile_diameter",
                                   "the s/D for the p-multiplier, from the "
                                   "spacing and pile diameter given"),
        "cap_in_contact": ("not in contact, where eta_g's row depends on it",
                           "the lower eta_g of GEC-12 Sec. 7.2.2.2 and UFC "
                           "3-220-20 Table 6-26; the result's choices give "
                           "the in-contact value"),
        "cu": ("su < 2 ksf, where a clay's eta_g row depends on it",
               "the lower eta_g of GEC-12 Sec. 7.2.2.2 and UFC 3-220-20 "
               "Table 6-26; the result's choices give the other")},
})
METHOD_INFO["pile_group_6dof"]["parameters"]["lateral_stiffness"][
    "applies_when"] = {"piles": "unset"}
from funhouse_agent.adapters import declare_applies_when  # noqa: E402
declare_applies_when(METHOD_INFO, {
    "group_efficiency": {
        "spacing_diameter_ratio": {"row_position": "set"},
        "cap_in_contact": {"soil_type": "set"},
        "cu": [{"soil_type": ["cohesive"]}, {"pile_length": "set"}],
        "pile_type": [{"soil_type": "set"}, {"pile_length": "set"}],
        "p_multiplier_method": {"row_position": "set"}},
    "group_uplift": {
        "cu": {"soil_type": ["cohesive"]},
        "unit_weight": {"layers": "unset"}},
})
declare_when_absent(METHOD_INFO, {
    "group_uplift": {
        "piles_under_cap": (
            "n_rows x n_cols, the piles in the block", "the small-group rule "
            "counts the piles per substructure unit (GEC-12 Vol I p. 225); give "
            "piles_under_cap when the block is part of the group"),
        "axial_capacity_tension": (
            "the block alone", "the group's uplift resistance is the lesser "
            "of the piles' sum and the block (GEC-12 Vol I p. 291), so no more "
            "than the block's; give each pile's factored tension resistance "
            "for the lesser")},
})
