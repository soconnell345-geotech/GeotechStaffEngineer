"""Retaining walls adapter — cantilever + MSE wall analysis + earth pressure coefficients."""

import dataclasses
import math

from funhouse_agent.adapters import (
    apply_aliases, check_nested_keys, declare_aliases, declare_applies_when,
    declare_when_absent, reject_unknown_params, require_params,
    without_unused,
)
from geotech_common.log_spiral import TABLE_4_2_NOTE as _TABLE_4_2_NOTE
from retaining_walls import earth_pressure as _ep
from retaining_walls.geometry import CantileverWallGeometry, MSEWallGeometry
from retaining_walls.cantilever import (
    GAMMA_CONCRETE, SLIDING_COHESION, analyze_cantilever_wall,
    default_bearing_pressure, default_FOS_overturning, default_FOS_sliding,
    default_sliding_cohesion,
)
from retaining_walls.mse import (
    GEOSYNTHETIC_WIDTH_NEEDS_SH, NEEDS_PHI_FOUNDATION, analyze_mse_wall,
    check_external_stability_lrfd, table_alpha,
)
from retaining_walls.reinforcement import (
    Reinforcement, RIBBED_STEEL_STRIP_50x4, WELDED_WIRE_GRID_W11,
    BAR_MAT_E4_3W11, BAR_MAT_E4_4W15_ST1, BAR_MAT_E4_4W15_ST1_5,
    GEOGRID_GEC11_E1_GG1, GEOGRID_GEC11_E1_GG2, GEOGRID_GEC11_E1_GG3,
)

# Names the agent commonly reaches for, mapped to the adapter's canonical names.
_WALL_ALIASES = {"phi": "phi_backfill", "gamma": "gamma_backfill",
                 "friction_angle": "phi_backfill", "unit_weight": "gamma_backfill",
                 "c": "c_backfill", "cohesion": "c_backfill",
                 "height": "wall_height", "H": "wall_height",
                 "delta_b": "delta_base", "base_friction_angle": "delta_base",
                 "adhesion": "base_adhesion", "ca": "base_adhesion"}

# MSE reinforced fill is select granular — there is no cohesion input, so do
# not alias c/cohesion to c_backfill (the reject message would name a key the
# caller never sent).
_MSE_ALIASES = {k: v for k, v in _WALL_ALIASES.items() if v != "c_backfill"}
# mse_wall alone takes the backfill's uniformity coefficient (GM-246) and
# the reinforcement's coverage ratio and tested F* and alpha (GM-256)
_MSE_WALL_ALIASES = {**_MSE_ALIASES, "Cu": "Cu_backfill",
                     "uniformity_coefficient": "Cu_backfill",
                     "Sh": "reinforcement_horizontal_spacing",
                     "horizontal_spacing": "reinforcement_horizontal_spacing",
                     "Rc": "coverage_ratio",
                     "reinforcement_depths": "level_depths"}

# Each built-in is a GEC-11 Vol II worked example's reinforcement as printed
# (GM-262, GM-267, GM-271).
_REINFORCEMENT_DB = {
    "ribbed_steel_strip_50x4": RIBBED_STEEL_STRIP_50x4,
    "welded_wire_grid_w11": WELDED_WIRE_GRID_W11,
    # GM-266: Example E4's other mats, Table E4-7.4 (p. E4-18)
    "bar_mat_e4_3w11_w11x0_5": BAR_MAT_E4_3W11,
    "bar_mat_e4_4w15_w11x1": BAR_MAT_E4_4W15_ST1,
    "bar_mat_e4_4w15_w11x1_5": BAR_MAT_E4_4W15_ST1_5,
    "geogrid_gec11_e1_gg1": GEOGRID_GEC11_E1_GG1,
    "geogrid_gec11_e1_gg2": GEOGRID_GEC11_E1_GG2,
    "geogrid_gec11_e1_gg3": GEOGRID_GEC11_E1_GG3,
}

_E1_GEOGRIDS = [f"'{key}' (Tal {r.Tallowable:.4g} kN/m)"
                for key, r in _REINFORCEMENT_DB.items()
                if r.type == "geogrid"]
_E1_GEOGRIDS = ", ".join(_E1_GEOGRIDS[:-1]) + " or " + _E1_GEOGRIDS[-1]
_GEOGRID_INSTEAD = (
    "Give one of GEC-11 Vol II Example E1's geogrids, as Table E1-7.3 (p. "
    f"E1-15) prints them: {_E1_GEOGRIDS}; or a custom reinforcement, "
    "reinforcement_type 'geogrid' with reinforcement_Tallowable, the "
    "product's long-term strength Tult / RF for the wall's backfill and "
    "design life (GEC-11 Vol I Eq. 3-12, p. 3-28, RF = RFID x RFCR x RFD)")

#: What list_reinforcement says the built-ins are (GM-271 review).
_BUILT_INS_ARE_EXAMPLES = (
    "Each built-in is the reinforcement a GEC-11 Vol II worked example "
    "prints (E3's strip, E4's bar mats, E1's geogrids), with that example's "
    "Tal, design life and backfill; they are not products. Where the wall's "
    "reinforcement is specified, give it as a custom reinforcement: "
    "reinforcement_type with reinforcement_Tallowable (and "
    "reinforcement_width for steel), Fy Ac / b at 0.55 Fy for strips or "
    "0.48 Fy for bar mats with Ac after corrosion over the design life "
    "(GEC-11 Vol I Eq. 3-11a, p. 3-22; p. 4-77), or for a geosynthetic Tult "
    "/ RF with the product's Tult and RF = RFID x RFCR x RFD for the wall's "
    "backfill and design life (Eq. 3-12, p. 3-28)")

#: Built-in names no longer offered, refused with what to give instead.
_RETIRED_REINFORCEMENT = {
    # GM-267: its 75 mm width was the tool's own
    "ribbed_steel_strip_75x4": (
        "its 75 mm width was the tool's own, where GEC-11's ribbed strip is "
        "50 mm wide (Vol I p. 2-16; Vol II Example E3, p. E3-16). Give "
        "'ribbed_steel_strip_50x4', E3's 50 x 4 mm strip as printed (Tal "
        f"{RIBBED_STEEL_STRIP_50x4.Tallowable:.4g} kN/m of strip width, "
        f"{RIBBED_STEEL_STRIP_50x4.Tallowable * RIBBED_STEEL_STRIP_50x4.width:.3g}"
        " kN a strip), with reinforcement_horizontal_spacing or "
        "coverage_ratio; or a custom reinforcement, reinforcement_type "
        "'metallic_strip' with reinforcement_Tallowable (0.55 Fy Ec, GEC-11 "
        "Vol I p. 4-77) and reinforcement_width"),
    # GM-271: named for one maker's product line, and no recorded source
    # for their Tal
    "geogrid_ux1600": ("its Tal, 21.9 kN/m, had no recorded source. "
                       + _GEOGRID_INSTEAD),
    "geogrid_ux1700": ("its Tal, 29.2 kN/m, had no recorded source. "
                       + _GEOGRID_INSTEAD),
}


# Reinforcement-type aliases. The bar-mat / welded-grid Kr/Ka (2.5->1.2) and
# F* (20 t/St -> 10 t/St) curves are keyed off the dataclass type
# "metallic_grid"; expose the friendlier "bar_mat"/"welded_grid" names too.
_REINFORCEMENT_TYPE_ALIASES = {
    "bar_mat": "metallic_grid",
    "barmat": "metallic_grid",
    "welded_grid": "metallic_grid",
    "welded_wire_grid": "metallic_grid",
    "steel_grid": "metallic_grid",
    "grid": "metallic_grid",
    "strip": "metallic_strip",
    "ribbed_strip": "metallic_strip",
    "metallic_strip": "metallic_strip",
    "metallic_grid": "metallic_grid",
    "geogrid": "geogrid",
    "geotextile": "geotextile",
    "geosynthetic": "geosynthetic",
}


def _with_coverage(reinforcement, params):
    """``reinforcement`` with the call's width, horizontal spacing and
    coverage ratio (GM-256): a copy, never the shared built-in."""
    width = params.get("reinforcement_width")
    if width is not None and reinforcement.width is not None \
            and not math.isclose(width, reinforcement.width, rel_tol=1e-6):
        raise ValueError(
            f"mse_wall: {reinforcement.name} is {reinforcement.width:g} m "
            f"wide, and its Tallowable is per metre of that width; for "
            f"another width give a custom reinforcement "
            f"(reinforcement_Tallowable, reinforcement_type, "
            f"reinforcement_width)")
    return dataclasses.replace(
        reinforcement,
        width=reinforcement.width if reinforcement.width is not None
        else width,
        horizontal_spacing=params.get("reinforcement_horizontal_spacing"),
        coverage_ratio=params.get("coverage_ratio"))


#: The custom reinforcement's own inputs, which a built-in does not take.
_CUSTOM_ONLY = ("reinforcement_Tallowable", "reinforcement_type",
                "reinforcement_thickness",
                "reinforcement_transverse_spacing")


#: What an entry of reinforcement_per_level takes (GM-266): a built-in's
#: name, or a custom reinforcement's own inputs without their
#: "reinforcement_" prefix (given with it, the prefix is dropped).
_LEVEL_KEYS = ("name", "type", "Tallowable", "width", "thickness",
               "transverse_spacing")
_LEVEL_CUSTOM_ONLY = ("type", "Tallowable", "thickness", "transverse_spacing")
_PER_LEVEL_TAKES = (
    "reinforcement_per_level: one entry a level from the top down, each a "
    "built-in name (see list_reinforcement) or a custom reinforcement "
    "{Tallowable, type, width, thickness, transverse_spacing, name}")


def _per_level_values(params, name, n):
    """reinforcement_horizontal_spacing or coverage_ratio as one value a
    level: the one value at every level, or the list's (GM-261), which
    must have one a level."""
    value = params.get(name)
    if not isinstance(value, (list, tuple)):
        return [value] * n
    if len(value) != n:
        raise ValueError(
            f"mse_wall: {name} gives {len(value)} values, one per level "
            f"from the top, and the wall has {n} levels, one entry of "
            "reinforcement_per_level each: give one value a level, or one "
            "number for the wall")
    return list(value)


def _per_level_reinforcement(params, geom):
    """The reinforcement at each level from reinforcement_per_level
    (GM-266), each with its level's Sh or Rc: a list of Reinforcement, one
    a level from the top down."""
    entries = params["reinforcement_per_level"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError(f"mse_wall: {_PER_LEVEL_TAKES}; got {entries!r}")
    single = [k for k in ("reinforcement_name", "reinforcement_width")
              + _CUSTOM_ONLY if params.get(k) is not None]
    if single:
        raise ValueError(
            f"mse_wall does not take {', '.join(single)} with "
            "reinforcement_per_level, which gives each level its own "
            "reinforcement: put each level's name, Tal, type and width in its "
            "entry (" + _PER_LEVEL_TAKES + ")")
    n = geom.n_reinforcement_levels
    if len(entries) != n:
        depths = ", ".join(f"{z:.2f}" for z in geom.reinforcement_depths)
        if geom.level_depths is not None:
            # GM-273: the levels at the depths given
            source = ("level_depths gives them: give one entry of "
                      "reinforcement_per_level for each depth there, or "
                      "one depth for each entry")
        else:
            source = (f"wall_height {geom.wall_height:g} m over "
                      f"reinforcement_spacing {geom.reinforcement_spacing:g} "
                      f"m sets them")
        raise ValueError(
            f"mse_wall: reinforcement_per_level gives {len(entries)} "
            f"reinforcements, one per reinforcement level from the top down, "
            f"and this wall has {n} levels, at depths {depths} m ({source})")
    Shs = _per_level_values(params, "reinforcement_horizontal_spacing", n)
    Rcs = _per_level_values(params, "coverage_ratio", n)
    return [_level_reinforcement(i, entry, Sh, Rc)
            for i, (entry, Sh, Rc) in enumerate(zip(entries, Shs, Rcs))]


def _level_reinforcement(i, entry, Sh, Rc):
    """One entry of reinforcement_per_level, at level i + 1 from the top."""
    label = f"reinforcement_per_level[{i}] (level {i + 1} from the top)"
    if isinstance(entry, str):
        entry = {"name": entry}
    if not isinstance(entry, dict):
        raise ValueError(f"mse_wall: {label} must be a built-in name or a "
                         f"dict; got {entry!r}. {_PER_LEVEL_TAKES}")
    entry = {(k[len("reinforcement_"):] if isinstance(k, str)
              and k.startswith("reinforcement_") else k): v
             for k, v in entry.items()}
    for k in ("horizontal_spacing", "coverage_ratio"):
        if k in entry:
            raise ValueError(
                f"mse_wall: {label} gives {k}; a spacing or coverage ratio "
                f"per level is reinforcement_horizontal_spacing or "
                f"coverage_ratio as a list, one value a level from the top")
    check_nested_keys(entry, _LEVEL_KEYS, method="mse_wall", label=label)
    if entry.get("width") is not None and Rc is not None:
        raise ValueError(
            f"mse_wall: {label} gives a width with coverage_ratio at its "
            "level: give coverage_ratio, or the width with "
            "reinforcement_horizontal_spacing (Rc = b/Sh, GEC-11 Vol I Eq. "
            "3-1, p. 3-12), not both: the width is used only for Rc")
    name = entry.get("name")
    key = name.lower() if isinstance(name, str) else None
    if key in _RETIRED_REINFORCEMENT:
        raise ValueError(f"mse_wall: {label}: '{key}' is no longer offered: "
                         + _RETIRED_REINFORCEMENT[key])
    level = {"reinforcement_width": entry.get("width"),
             "reinforcement_horizontal_spacing": Sh, "coverage_ratio": Rc}
    if key in _REINFORCEMENT_DB:
        unused = [k for k in _LEVEL_CUSTOM_ONLY if entry.get(k) is not None]
        if unused:
            raise ValueError(
                f"mse_wall: {label} gives {', '.join(unused)} with the "
                f"built-in '{key}', which carries its own Tal, type and bar "
                "geometry (see list_reinforcement): leave them out, or give "
                "a custom entry with Tallowable and type and no built-in name")
        try:
            r = _with_coverage(_REINFORCEMENT_DB[key], level)
        except ValueError as e:
            raise ValueError(f"mse_wall: {label}: "
                             + str(e).removeprefix("mse_wall: ")) from None
    else:
        if entry.get("Tallowable") is None:
            raise ValueError(
                f"mse_wall: {label}: "
                + (f"'{name}' is not a built-in ({sorted(_REINFORCEMENT_DB)}; "
                   "see list_reinforcement). " if name else "")
                + "Give a built-in name, or a custom reinforcement with its "
                "Tallowable (kN/m of its own width) and type")
        if entry.get("type") is None:
            raise ValueError(
                f"mse_wall: {label}: a custom reinforcement needs its type: "
                "'metallic_strip', 'metallic_grid' (bar mat, welded grid), "
                "'geogrid' or 'geotextile', which sets the pullout curves and "
                "alpha (GEC-11 Vol I Table 3-6, p. 3-17); or 'geosynthetic' "
                "with alpha_tested")
        r_type = _REINFORCEMENT_TYPE_ALIASES.get(entry["type"], entry["type"])
        r = Reinforcement(
            name=name or f"Custom {r_type}",
            type=r_type, Tallowable=entry["Tallowable"],
            width=entry.get("width"),
            thickness=entry.get("thickness") or 0.0,
            transverse_spacing=entry.get("transverse_spacing") or 0.0,
            horizontal_spacing=Sh, coverage_ratio=Rc)
    if not r.is_metallic and entry.get("width") is not None and Sh is None:
        raise ValueError(f"mse_wall: {label}: "
                         + GEOSYNTHETIC_WIDTH_NEEDS_SH.format(
                             b=entry["width"]))
    return r


def _build_reinforcement(params):
    if isinstance(params.get("reinforcement_name"), (list, tuple)):
        raise ValueError(
            "mse_wall: reinforcement_name takes one built-in name for every "
            "level; for a reinforcement per level give "
            + _PER_LEVEL_TAKES)
    if params.get("coverage_ratio") is not None \
            and params.get("reinforcement_width") is not None:
        # GM-256 review: the width serves only Rc = b/Sh, so with Rc given
        # it would be taken and not used
        raise ValueError(
            "mse_wall: give coverage_ratio, or reinforcement_width with "
            "reinforcement_horizontal_spacing (Rc = b/Sh, GEC-11 Vol I Eq. "
            "3-1, p. 3-12), not both: the width is used only for Rc")
    name = params.get("reinforcement_name", "").lower()
    if name in _RETIRED_REINFORCEMENT:
        raise ValueError(
            f"mse_wall: reinforcement_name '{name}' is no longer offered: "
            + _RETIRED_REINFORCEMENT[name])
    if name in _REINFORCEMENT_DB:
        # GM-262 review: a built-in carries its own Tal, type and bar
        # geometry, so these were taken and not used
        unused = [k for k in _CUSTOM_ONLY if params.get(k) is not None]
        if unused:
            raise ValueError(
                f"mse_wall does not take {', '.join(unused)} with the "
                f"built-in reinforcement_name '{name}': the built-in "
                "carries its own Tal, type and bar geometry (see "
                "list_reinforcement). Leave them out, or give a custom "
                "reinforcement: reinforcement_Tallowable with "
                "reinforcement_type and no built-in name")
        r = _with_coverage(_REINFORCEMENT_DB[name], params)
    else:
        r = _custom_reinforcement(params)
    width = params.get("reinforcement_width")
    if not r.is_metallic and width is not None \
            and params.get("reinforcement_horizontal_spacing") is None:
        # GM-256 review: geosynthetic strips at a spacing (GEC-11 p. 3-12);
        # with no Sh the width would be taken and not used, and Rc = 1
        # would overstate a strip layer
        raise ValueError("mse_wall: " + GEOSYNTHETIC_WIDTH_NEEDS_SH.format(
            b=width))
    return r


def _custom_reinforcement(params):
    """A custom reinforcement from the call's reinforcement_* inputs."""
    if params.get("reinforcement_Tallowable") is None:
        given = params.get("reinforcement_name")
        not_built_in = (f"reinforcement_name '{given}' is not a built-in. "
                        if given else "")
        raise ValueError(
            f"mse_wall: {not_built_in}provide a built-in reinforcement_name "
            f"({sorted(_REINFORCEMENT_DB)}; see list_reinforcement) or a custom "
            "reinforcement_Tallowable (kN/m) with reinforcement_type and "
            "optional reinforcement_width/reinforcement_thickness "
            "(and, for a bar mat / welded grid: reinforcement_transverse_spacing "
            "= St with reinforcement_thickness = transverse-bar diameter t)."
        )
    r_type_in = params.get("reinforcement_type")
    if r_type_in is None:
        # GM-256: the type sets the alpha and the coverage a custom
        # reinforcement takes; there is no default type
        raise ValueError(
            "mse_wall: a custom reinforcement (reinforcement_Tallowable) "
            "needs its reinforcement_type: 'metallic_strip', "
            "'metallic_grid' (bar mat, welded grid), 'geogrid' or "
            "'geotextile'. The type sets the pullout curves and the scale "
            "effect correction factor alpha, GEC-11 Vol I Table 3-6 (p. "
            "3-17): 1.0 for steel, 0.8 for geogrids, 0.6 for geotextiles; "
            "or 'geosynthetic' with alpha_tested, an alpha from pullout "
            "tests")
    r_type = _REINFORCEMENT_TYPE_ALIASES.get(r_type_in, r_type_in)
    return Reinforcement(
        name=params.get("reinforcement_name", "Custom"),
        type=r_type,
        Tallowable=params["reinforcement_Tallowable"],
        width=params.get("reinforcement_width"),
        thickness=params.get("reinforcement_thickness", 0.0),
        transverse_spacing=params.get("reinforcement_transverse_spacing", 0.0),
        horizontal_spacing=params.get("reinforcement_horizontal_spacing"),
        coverage_ratio=params.get("coverage_ratio"),
    )


def run_cantilever_wall(params, *, method="cantilever_wall"):
    """(result, inputs in force) of ``cantilever_wall``: the tool's own code
    path, also run by ``calc_package.retaining_wall_package`` (GM-141).
    The inputs in force are the geometry analysed (``geom``, auto-sized
    parts included) and each input given or defaulted, by the tool's
    names. ``method`` names the caller in error messages."""
    params = apply_aliases(params, _WALL_ALIASES)
    reject_unknown_params(
        params,
        ("wall_height", "base_width", "toe_length", "stem_thickness_top",
         "stem_thickness_base", "base_thickness", "has_shear_key", "key_depth",
         "backfill_slope", "surcharge", "dead_load_surcharge",
         "gamma_backfill", "phi_backfill",
         "c_backfill", "phi_foundation", "c_foundation", "q_allowable",
         "gamma_concrete", "FOS_sliding_required", "FOS_overturning_required",
         "pressure_method", "include_passive", "gamma_foundation",
         "delta_base", "base_adhesion", "live_load_resists",
         "dead_load_resists", "surcharge_in_bearing", "sliding_cohesion",
         "passive_theory", "log_spiral_source",
         "passive_wall_friction_deg", "stem_batter", "embedment_depth",
         "wall_friction_deg", "interface_gap_possible", "founded_on_rock",
         "bearing_pressure"),
        method=method)
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method=method)
    geom = CantileverWallGeometry(
        wall_height=params["wall_height"],
        base_width=params.get("base_width"), toe_length=params.get("toe_length"),
        stem_thickness_top=params.get("stem_thickness_top", 0.30),
        stem_thickness_base=params.get("stem_thickness_base"),
        base_thickness=params.get("base_thickness"),
        has_shear_key=params.get("has_shear_key", False),
        key_depth=params.get("key_depth", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        surcharge=params.get("surcharge", 0.0),
        dead_load_surcharge=params.get("dead_load_surcharge", 0.0),
        stem_batter=params.get("stem_batter", "back"),
        embedment_depth=params.get("embedment_depth"),
    )
    include_passive = params.get("include_passive", False)
    rock = bool(params.get("founded_on_rock", False))
    used = {
        "geom": geom,
        "gamma_backfill": params["gamma_backfill"],
        "phi_backfill": params["phi_backfill"],
        "c_backfill": params.get("c_backfill", 0.0),
        "phi_foundation": params.get("phi_foundation"),
        "c_foundation": params.get("c_foundation", 0.0),
        "q_allowable": params.get("q_allowable"),
        "gamma_concrete": params.get("gamma_concrete", GAMMA_CONCRETE),
        # GM-288: 2.0 with the passive resistance counted, 1.5 on rock
        "FOS_sliding_required": params.get(
            "FOS_sliding_required", default_FOS_sliding(include_passive)),
        "FOS_overturning_required": params.get(
            "FOS_overturning_required", default_FOS_overturning(rock)),
        "pressure_method": params.get("pressure_method", "rankine"),
        "include_passive": include_passive,
        "gamma_foundation": params.get("gamma_foundation"),
        "delta_base": params.get("delta_base"),
        "base_adhesion": params.get("base_adhesion"),
        "live_load_resists": bool(params.get("live_load_resists", False)),
        "dead_load_resists": bool(params.get("dead_load_resists", False)),
        "surcharge_in_bearing": bool(params.get("surcharge_in_bearing",
                                                True)),
        # GM-212: the printed rule for the foundation cohesion in sliding
        "sliding_cohesion": (params.get("sliding_cohesion")
                             or default_sliding_cohesion(
                                 params.get("phi_foundation"))),
        # GM-242: the passive coefficient in front of the wall
        "passive_theory": PASSIVE_THEORY_ALIASES.get(
            params.get("passive_theory", "rankine"),
            params.get("passive_theory", "rankine")),
        "log_spiral_source": params.get("log_spiral_source"),
        "passive_wall_friction_deg": params.get("passive_wall_friction_deg"),
        # GM-288
        "wall_friction_deg": params.get("wall_friction_deg"),
        "interface_gap_possible": bool(params.get("interface_gap_possible",
                                                  False)),
        "founded_on_rock": rock,
        "bearing_pressure": (params.get("bearing_pressure")
                             or default_bearing_pressure(rock)),
    }
    result = analyze_cantilever_wall(
        geom, gamma_backfill=used["gamma_backfill"],
        phi_backfill=used["phi_backfill"], c_backfill=used["c_backfill"],
        phi_foundation=used["phi_foundation"],
        c_foundation=used["c_foundation"], q_allowable=used["q_allowable"],
        gamma_concrete=used["gamma_concrete"],
        # the defaults left to the module, so its choices say they are
        FOS_sliding=params.get("FOS_sliding_required"),
        FOS_overturning=params.get("FOS_overturning_required"),
        pressure_method=used["pressure_method"],
        include_passive=used["include_passive"],
        gamma_foundation=used["gamma_foundation"],
        delta_base=used["delta_base"],
        base_adhesion=used["base_adhesion"],
        live_load_resists=used["live_load_resists"],
        dead_load_resists=used["dead_load_resists"],
        surcharge_in_bearing=used["surcharge_in_bearing"],
        sliding_cohesion=used["sliding_cohesion"],
        passive_theory=used["passive_theory"],
        log_spiral_source=used["log_spiral_source"],
        passive_wall_friction_deg=used["passive_wall_friction_deg"],
        wall_friction_deg=used["wall_friction_deg"],
        interface_gap_possible=used["interface_gap_possible"],
        founded_on_rock=rock,
        bearing_pressure=params.get("bearing_pressure"),
    )
    return result, used


def _run_cantilever_wall(params):
    result, used = run_cantilever_wall(params)
    geom = used["geom"]
    output = result.to_dict()
    # The geometry actually analysed, auto-sized parts included, with the
    # height's datum named (GM-100: a model read H as the retained height).
    output["geometry"] = {
        "wall_height_m": geom.wall_height,
        "wall_height_basis": "underside of the base slab to the top of the stem",
        "stem_height_m": round(geom.stem_height, 3),
        "base_thickness_m": geom.base_thickness,
        "base_width_m": geom.base_width,
        "toe_length_m": geom.toe_length,
        "heel_length_m": round(geom.heel_length, 3),
        "stem_thickness_top_m": geom.stem_thickness_top,
        "stem_thickness_base_m": geom.stem_thickness_base,
        # GM-288: the battered face and the ground in front
        "stem_batter": geom.stem_batter,
        "embedment_depth_m": geom.Df,
    }
    # Surface the sliding-interface values actually used, so the caller can
    # verify the basis instead of guessing (owner wall session 2026-07-14);
    # the result carries them since GM-143, so they are not worked out twice.
    from retaining_walls.cantilever import _sliding_dict
    sl = _sliding_dict(result.sliding_terms, used["FOS_sliding_required"])
    output["sliding_basis"] = {
        "sliding_cohesion": sl["sliding_cohesion"],
        "delta_base_deg": sl["delta_base_deg"],
        "base_adhesion_kPa": sl["base_adhesion_kPa"],
        "adhesion_width_m": sl["adhesion_width_m"],
        "driving_force_kN_per_m": sl["driving_force_kN_per_m"],
        "resisting_force_kN_per_m": sl["resisting_force_kN_per_m"],
        "Pa_kN_per_m": sl["Pa_kN_per_m"],
        "Pa_vertical_kN_per_m": sl["Pa_vertical_kN_per_m"],
        "thrust_inclination_deg": sl["thrust_inclination_deg"],
    }
    terms = result.sliding_terms
    if sl.get("Pp_kN_per_m"):
        # with include_passive: on the embedment, or over the key depth
        # (GM-210), with cu under the undrained rule (GM-288)
        output["sliding_basis"]["Pp_kN_per_m"] = sl["Pp_kN_per_m"]
        if terms.get("Pp_cohesion"):
            output["sliding_basis"]["Pp_cohesion_kN_per_m"] = round(
                terms["Pp_cohesion"], 1)
    if terms.get("gap_limited"):
        # EN 1997-1 Eq. 6.5 held the undrained resistance (GM-288)
        output["sliding_basis"]["base_resistance_held_to_0_4V"] = True
    if terms.get("passive_theory"):
        # the coefficient it took (GM-242)
        output["sliding_basis"]["passive_theory"] = terms["passive_theory"]
        output["sliding_basis"]["Kp"] = round(terms["Kp"], 4)
        if terms["log_spiral_source"]:
            output["sliding_basis"]["log_spiral_source"] = \
                terms["log_spiral_source"]
            output["sliding_basis"]["passive_wall_friction_deg"] = round(
                terms["passive_delta"], 4)
    return output


#: level_depths and reinforcement_spacing both given (GM-273).
_DEPTHS_OR_SPACING = (
    "mse_wall takes level_depths or reinforcement_spacing, not both: "
    "level_depths places every level (each then takes its contributory "
    "height, GEC-11 Vol I pp. 4-46, 4-47), so the spacing Sv would be used "
    "nowhere. Give the depths alone, or leave them out and give "
    "reinforcement_spacing for levels at Sv (i + 1/2)")


def run_mse_wall(params, *, method="mse_wall"):
    """(result, inputs in force) of ``mse_wall``: the tool's own code path,
    also run by ``calc_package.retaining_wall_package`` (GM-141). The inputs
    in force are the geometry (``geom``), the reinforcement and each input
    given or defaulted, by the tool's names. ``method`` names the caller in
    error messages."""
    params = apply_aliases(params, _MSE_WALL_ALIASES)
    reject_unknown_params(
        params,
        ("wall_height", "reinforcement_length", "reinforcement_spacing",
         "surcharge", "backfill_slope", "broken_backslope_height",
         "gamma_backfill", "phi_backfill",
         "reinforcement_name", "reinforcement_type",
         "reinforcement_Tallowable", "reinforcement_width",
         "reinforcement_thickness",
         "reinforcement_transverse_spacing",
         "reinforcement_horizontal_spacing", "coverage_ratio",
         "phi_foundation", "c_foundation", "q_allowable",
         "phi_retained", "gamma_retained", "traffic_vertical_resists",
         "live_load_resists", "surcharge_in_bearing", "Cu_backfill",
         "F_star_tested", "alpha_tested", "reinforcement_per_level",
         "level_depths", "sigma_H_average"),
        method=method)
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method=method)
    if params.get("level_depths") is not None \
            and params.get("reinforcement_spacing") is not None:
        # GM-273: the depths place every level; Sv would place none
        raise ValueError(_DEPTHS_OR_SPACING)
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        reinforcement_spacing=params.get("reinforcement_spacing", 0.60),
        surcharge=params.get("surcharge", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        broken_backslope_height=params.get("broken_backslope_height"),
        level_depths=params.get("level_depths"),
    )
    if params.get("reinforcement_per_level") is not None:
        # GM-266: a reinforcement a level, as GEC-11 Vol II Example E4
        # changes its mats with depth (Table E4-7.4, p. E4-18)
        reinforcement = _per_level_reinforcement(params, geom)
    else:
        reinforcement = _build_reinforcement(params)
    used = {
        "geom": geom, "reinforcement": reinforcement,
        "gamma_backfill": params["gamma_backfill"],
        "phi_backfill": params["phi_backfill"],
        "phi_foundation": params.get("phi_foundation"),
        "c_foundation": params.get("c_foundation", 0.0),
        "q_allowable": params.get("q_allowable"),
        "phi_retained": params.get("phi_retained"),
        "gamma_retained": params.get("gamma_retained"),
        "traffic_vertical_resists": bool(
            params.get("traffic_vertical_resists", False)),
        "live_load_resists": bool(params.get("live_load_resists", False)),
        "surcharge_in_bearing": bool(params.get("surcharge_in_bearing",
                                                True)),
        "Cu_backfill": params.get("Cu_backfill"),
        "F_star_tested": params.get("F_star_tested"),
        "alpha_tested": params.get("alpha_tested"),
        "sigma_H_average": params.get("sigma_H_average"),
    }
    result = analyze_mse_wall(
        geom, gamma_backfill=used["gamma_backfill"],
        phi_backfill=used["phi_backfill"], reinforcement=reinforcement,
        phi_foundation=used["phi_foundation"],
        c_foundation=used["c_foundation"], q_allowable=used["q_allowable"],
        phi_retained=used["phi_retained"],
        gamma_retained=used["gamma_retained"],
        traffic_vertical_resists=used["traffic_vertical_resists"],
        live_load_resists=used["live_load_resists"],
        surcharge_in_bearing=used["surcharge_in_bearing"],
        Cu_backfill=used["Cu_backfill"],
        F_star_tested=used["F_star_tested"],
        alpha_tested=used["alpha_tested"],
        sigma_H_average=used["sigma_H_average"],
    )
    return result, used


def _run_mse_wall(params):
    result, used = run_mse_wall(params)
    output = result.to_dict()
    r = used["reinforcement"]
    # GM-266: each level's, from the top, when given per level
    output["reinforcement_name"] = (r.name if isinstance(r, Reinforcement)
                                    else [ri.name for ri in r])
    return output


def _run_mse_lrfd_external_stability(params):
    """AASHTO/GEC-11 LRFD external-stability CDR set (sliding / eccentricity /
    bearing) under Strength I (max + min) and Service I combinations."""
    params = without_unused(
        apply_aliases(params, _MSE_ALIASES),
        METHOD_INFO["mse_lrfd_external_stability"]["refused_parameters"])
    # No reinforcement_spacing: external stability does not read it, so it
    # is refused, not dropped (GM-110).
    reject_unknown_params(
        params,
        ("wall_height", "reinforcement_length", "surcharge",
         "backfill_slope", "broken_backslope_height",
         "gamma_backfill", "phi_backfill",
         "phi_foundation", "phi_retained", "gamma_retained", "live_load",
         "bearing_resistance_strength", "bearing_resistance_service",
         "gamma_foundation", "phi_bearing",
         "c_foundation", "gamma_EV_max", "gamma_EV_min", "gamma_EH_max",
         "gamma_EH_min", "gamma_LL", "phi_sliding", "ecc_limit_ratio",
         "traffic_vertical_resists"),
        method="mse_lrfd_external_stability")
    require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                   method="mse_lrfd_external_stability")
    if (params.get("gamma_foundation") is not None
            and params.get("phi_foundation") is None):
        raise ValueError(NEEDS_PHI_FOUNDATION)
    geom = MSEWallGeometry(
        wall_height=params["wall_height"],
        reinforcement_length=params.get("reinforcement_length"),
        surcharge=params.get("surcharge", 0.0),
        backfill_slope=params.get("backfill_slope", 0.0),
        broken_backslope_height=params.get("broken_backslope_height"),
    )
    # Only forward load/resistance factors the caller actually set (so the
    # GEC-11 defaults live in one place — the module function).
    factor_keys = ("gamma_EV_max", "gamma_EV_min", "gamma_EH_max",
                   "gamma_EH_min", "gamma_LL", "phi_sliding", "ecc_limit_ratio",
                   "phi_bearing")
    factors = {k: params[k] for k in factor_keys if k in params}
    result = check_external_stability_lrfd(
        geom, gamma_backfill=params["gamma_backfill"],
        phi_backfill=params["phi_backfill"],
        phi_foundation=params.get("phi_foundation", params["phi_backfill"]),
        phi_retained=params.get("phi_retained"),
        gamma_retained=params.get("gamma_retained"),
        live_load=params.get("live_load"),
        c_foundation=params.get("c_foundation", 0.0),
        bearing_resistance_strength=params.get("bearing_resistance_strength"),
        bearing_resistance_service=params.get("bearing_resistance_service"),
        gamma_foundation=params.get("gamma_foundation"),
        traffic_vertical_resists=bool(
            params.get("traffic_vertical_resists", False)),
        **factors,
    )
    result["reinforcement_length_m"] = geom.reinforcement_length
    return result


#: Names of the log-spiral passive Kp a theory input also takes (GM-242):
#: 'log_spiral' is the method's name in UFC 3-220-20 and the Caltrans
#: The log-spiral passive theory of the wall's toe by its other names
#: (GM-242): 'caquot_kerisel' is what earth_pressure_coefficient calls it.
PASSIVE_THEORY_ALIASES = {"caquot_kerisel": "log_spiral",
                          "logspiral": "log_spiral",
                          "log-spiral": "log_spiral"}
#: Trenching and Shoring Manual; 'caquot_kerisel' stays the name results
#: give.
LOG_SPIRAL_THEORY_ALIASES = {"log_spiral": "caquot_kerisel",
                             "logspiral": "caquot_kerisel",
                             "log-spiral": "caquot_kerisel"}
_LOG_SPIRAL_THEORY = ["caquot_kerisel", "log_spiral"]

_K_ALIASES = {"phi": "phi_deg", "friction_angle": "phi_deg",
              "phi_backfill": "phi_deg",
              "delta": "delta_deg", "wall_friction": "delta_deg",
              "beta": "beta_deg", "backfill_slope": "beta_deg",
              "slope_angle": "beta_deg",
              "alpha": "alpha_deg", "wall_angle": "alpha_deg",
              "theta": "theta_deg", "ocr": "OCR",
              "overconsolidation_ratio": "OCR"}


def _log_spiral_kp(p, phi, beta, source):
    """The log-spiral passive Kp from ``source`` (GM-183, GM-232), with
    the wall friction it used and its ``choices``: the route the UFC
    3-220-20 default took by the wall's geometry, its equation or figure
    and page, and every other printed source's Kp for the same case; and
    the wall friction, given or left out, as the engineer's call, with the
    Kp at delta = 0 beside it (GM-253). A left-out wall friction is
    refused where the source prints none (GM-253)."""
    from geotech_common import log_spiral as _ls
    from geotech_common.choices import choice, how_given
    theta = p.get("theta_deg", 0.0) or 0.0
    delta = p.get("delta_deg")
    route = None
    if source == "ufc_3_220_20":
        r = _ls.ufc_3_220_20_route(phi, delta, beta, theta)
        K, delta, route = r["Kp"], r["delta_deg"], r["route"]
        basis = r["where"] + ". " + _ls.SOURCE_BASIS[source]
        if route == "dm7_2_eq_4_12":
            basis += ". " + _ls.EQ_VS_CHARTS_NOTE
    else:
        if delta is None and phi <= 0:
            delta = 0.0
        if delta is None and source in _ls.LEVEL_FRONT_NEEDS_DELTA:
            # the 1948 chart and Eq. 4-12 by name print no wall friction
            # (Caltrans p. 4-29 begins 'Given delta'): refused, as soe and
            # the wall toe refuse it (GM-242, GM-253); the geometry the
            # source takes is checked first, at a trial delta = phi
            _ls.log_spiral_Kp(phi, phi, beta, source=source, theta_deg=theta)
            raise ValueError(_ls.LEVEL_FRONT_NEEDS_DELTA[source].format(
                name="delta_deg"))
        K = _ls.log_spiral_Kp(phi, delta, beta, source=source,
                              theta_deg=theta)
        if delta is None:
            # UFC Figs. 4-12 and 4-13 by name: the one wall friction they
            # are drawn for
            delta = _ls.DM_DELTA_RATIO * phi
        basis = _ls.SOURCE_BASIS[source]
    if "dm7_2_eq_4_12" in (source, route) and _ls.above_the_charts(phi,
                                                                   delta):
        basis += ". " + _ls.HIGH_FRICTION_NOTE
    choices = [choice(
        "Log-spiral passive Kp (log_spiral_source)", source,
        how_given(source, _ls.DEFAULT_SOURCE, name="log_spiral_source"),
        basis + _LOG_SPIRAL_NOTE,
        _ls.other_sources(phi, delta, beta, theta,
                          used=(source, route)))]
    # the wall friction, given or left out: the same words either way, the
    # value it took in defaults_used when left out, and the Kp with none
    chosen = f"{delta:.3g} deg" + (
        f" (delta/phi' {delta / phi:.2f})" if phi else "")
    choices.append(choice(
        "Wall friction delta (delta_deg)", chosen,
        how_given(p.get("delta_deg"), None, name="delta_deg",
                  took=round(delta, 4)),
        _ls.wall_friction_note(source, left_out=(
            "With phi' = 0 there is no wall friction (delta runs 0 to "
            "phi')." if phi <= 0 else None)),
        [_no_wall_friction(phi, beta, theta, source)] if delta else ()))
    return K, delta, theta, route, choices


def _no_wall_friction(phi, beta, theta, source):
    """The log-spiral Kp at delta = 0, as an alternative beside the wall
    friction used (GM-253), from what prints one for the geometry: on level
    ground before a vertical face Rankine's Kp, which UFC 3-220-20 Eq. 4-12
    gives at delta = 0; on sloping ground the Caquot-Kerisel (1948) chart
    with R from its delta/phi = 0 column (UFC Fig. 4-13 is drawn for 0.66
    only); before a battered face Coulomb's Kp, as no log-spiral chart here
    takes a battered face at delta = 0."""
    from geotech_common import log_spiral as _ls
    coulomb = ("FHWA NHI-06-089 p. 10-19 calls Coulomb's passive results "
               "'unreliable ... for wall friction angle values greater than "
               "phi'/3' and says 'If the angle of wall friction delta is "
               "small, the failure surface is almost linear'")
    if theta:
        K = _ep.coulomb_Kp(phi, 0.0, 90.0 - theta, beta)
        how = (f"Coulomb's (theory 'coulomb', alpha_deg {90.0 - theta:g} "
               "= 90 - theta_deg): no log-spiral chart here gives a "
               "battered face at delta = 0, and " + coulomb)
    elif beta:
        try:
            K = _ls.caquot_kerisel_1948_Kp(phi, 0.0, beta_deg=beta)
            how = ("the Caquot-Kerisel (1948) chart's, with R from its "
                   "delta/phi = 0 column (Caltrans T&S Fig. 4-20, p. 4-30)")
            if source != "caquot_kerisel_1948":
                how += "; UFC 3-220-20 Fig. 4-13 is drawn for 0.66 only"
        except ValueError:
            K = _ep.coulomb_Kp(phi, 0.0, 90.0, beta)
            how = ("Coulomb's (theory 'coulomb'): the case is off the 1948 "
                   "chart and UFC Fig. 4-13 is drawn for 0.66 only, and "
                   + coulomb)
    else:
        K = _ep.rankine_Kp(phi)
        how = "Rankine's, which UFC 3-220-20 Eq. 4-12 gives at delta = 0"
    return f"delta = 0 (no wall friction): Kp {K:.3g}, {how}"


def _run_earth_pressure_coefficient(params):
    """Thin wrapper over retaining_walls.earth_pressure Ka/Kp/K0 functions."""
    from geotech_common import log_spiral as _ls
    from geotech_common.choices import choice, how_given
    names = ("phi_deg", "theory", "state", "delta_deg", "beta_deg",
             "alpha_deg", "theta_deg", "log_spiral_source", "OCR")
    p = without_unused(
        apply_aliases(params, _K_ALIASES),
        METHOD_INFO["earth_pressure_coefficient"]["refused_parameters"])
    if "k0_sloping_ground" in p:
        # the same refusal dispatch gives, for a caller that skips it
        raise ValueError(
            "earth_pressure_coefficient does not take "
            "['k0_sloping_ground']: k0_sloping_ground: "
            + _NO_K0_FORM["reason"] + ".")
    reject_unknown_params(p, names, method="earth_pressure_coefficient")
    require_params(p, ["phi_deg"], method="earth_pressure_coefficient",
                   valid=list(names))
    # 'log_spiral' names the same coefficient as 'caquot_kerisel' (GM-242)
    theory = LOG_SPIRAL_THEORY_ALIASES.get(p.get("theory", "rankine"),
                                           p.get("theory", "rankine"))
    state = p.get("state", "active")
    phi = p["phi_deg"]
    beta = p.get("beta_deg", 0.0)
    if theory not in ("rankine", "coulomb", "caquot_kerisel"):
        raise ValueError(f"Unknown theory '{theory}'. Allowed: "
                         "['rankine', 'coulomb', 'caquot_kerisel', "
                         "'log_spiral'].")
    if state not in ("active", "passive", "at_rest"):
        raise ValueError(f"Unknown state '{state}'. Allowed: ['active', 'passive', 'at_rest'].")
    _refuse_unused_k_inputs(p, theory, state)
    source = p.get("log_spiral_source", _ls.DEFAULT_SOURCE)
    _refuse_unused_k_forms(p, theory, state, source, beta)
    ocr = p.get("OCR")
    if ocr is not None and float(ocr) < 1.0:
        raise ValueError(
            "earth_pressure_coefficient: OCR is the overconsolidation "
            f"ratio, at least 1 (1 = normally consolidated); got OCR {ocr}.")
    choices = []
    log_spiral = theory == "caquot_kerisel" and state == "passive"

    if theory == "caquot_kerisel":
        # Log-spiral PASSIVE coefficient Kp (v5.3), from the printed source
        # log_spiral_source names (GM-183), by default UFC 3-220-20's for
        # the wall's geometry (GM-232). Unlike Coulomb it does not
        # over-predict Kp at high wall friction delta/phi.
        if state != "passive":
            raise ValueError(
                "theory 'caquot_kerisel' ('log_spiral') is a passive (Kp) "
                "log-spiral coefficient; "
                "use state='passive' (it has no active/at-rest form).")
        K, ls_delta, theta, route, ls_choices = _log_spiral_kp(
            p, phi, beta, source)
        symbol = "Kp"
        choices.extend(ls_choices)
    elif state == "at_rest":
        # EN 1997-1 Eq. 9.1: K0 = (1 - sin phi') sqrt(OCR) (GM-245)
        ocr_used = float(ocr) if ocr is not None else 1.0
        K0_level = _ep.K0(phi) * math.sqrt(ocr_used)
        K, symbol = K0_level, "K0"
        choices.append(choice(
            "Overconsolidation ratio OCR in the at-rest K0",
            f"OCR {ocr_used:g}",
            how_given(ocr, 1.0, name="OCR"),
            _k0_ocr_text(phi, ocr_used, K0_level)))
        if beta:
            # EN 1997-1 Eq. 9.2, the one form (GM-243, owner OWN-67)
            K = K0_level * (1.0 + math.sin(math.radians(beta)))
            choices.append(choice(
                "At-rest K0 behind sloping ground", "EN 1997-1 Eq. 9.2",
                "module",
                f"K0;beta = K0 (1 + sin beta) = {K0_level:.3f} (1 + sin "
                f"{beta:g}) = {K:.3f}. " + _K0_SLOPE_BASIS))
    elif theory == "rankine":
        if state == "active":
            K = _ep.rankine_Ka_sloped(phi, beta) if beta else _ep.rankine_Ka(phi)
            symbol = "Ka"
        else:
            K, symbol = _ep.rankine_Kp(phi), "Kp"
    else:  # coulomb
        delta = p.get("delta_deg")
        if delta is None:
            # GM-257: none, given in defaults_used as the 0 it took
            delta = 0.0
            how_given(None, None, name="delta_deg", took=0.0)
        alpha = p.get("alpha_deg", 90.0)
        if state == "active":
            K, symbol = _ep.coulomb_Ka(phi, delta, alpha, beta), "Ka"
        else:
            K, symbol = _ep.coulomb_Kp(phi, delta, alpha, beta), "Kp"
    delta = p.get("delta_deg")
    if log_spiral:
        # the wall friction the log-spiral Kp used: given, or 0.66 phi'
        # (UFC) or phi (the 1948 chart at R = 1) when left out
        delta = round(ls_delta, 4)
    elif delta is None:
        delta = 0.0
    if state == "at_rest":
        name = ("en_1997_1_eq_9_2" if beta else
                "en_1997_1_eq_9_1" if ocr_used > 1.0 else "jaky_at_rest")
    else:
        name = theory
    out = {
        "K": round(K, 4),
        "coefficient": symbol,
        "theory": name,
        "state": state,
        "phi_deg": phi,
        "delta_deg": delta,
        "beta_deg": beta,
        "alpha_deg": p.get("alpha_deg", 90.0),
    }
    if log_spiral:
        out["theta_deg"] = theta
        out["log_spiral_source"] = source
        if route:
            out["log_spiral_route"] = route
    if choices:
        out["choices"] = choices
    return out


# The log-spiral sources side by side (GM-183, GM-232), for the choice's
# basis; each other source's Kp for the case is in the alternatives.
_LOG_SPIRAL_NOTE = (
    ". UFC 3-220-20 footnote 21 (p. 191): the 1982 DM 7.2 charts 'used "
    "delta/phi' = 1 along with reduction ratios', which 'were averages that "
    "introduced substantial inaccuracy for some cases'; the Caquot-Kerisel "
    "(1948) chart takes that form. " + _TABLE_4_2_NOTE)

# The at-rest K0 behind sloping ground: EN 1997-1 Eq. 9.2 always (GM-243;
# the owner, OWN-67: "Always Eurocode 9.2. That's a standard in textbooks
# and grad class notes."). The basis the result's choices give, after the
# arithmetic.
_K0_SLOPE_BASIS = (
    "EN 1997-1:2004 Sec. 9.5.2(4), Eq. 9.2 (p. 104): where the ground "
    "slopes upwards from the wall at beta <= phi' to the horizontal, "
    "K0;beta relates the horizontal component of the effective at-rest "
    "earth pressure to the effective overburden pressure, and the resulting "
    "force is taken parallel to the ground surface. K0;beta times the "
    "effective vertical stress is the horizontal earth pressure itself, "
    "the resultant acts at beta to the horizontal, and no further cos beta "
    "reduction applies. beta (beta_deg) is the "
    "slope of the ground rising away from the back of the wall, from the "
    "horizontal; Eq. 9.2 does not cover ground falling away from the wall. "
    "K0 is EN 1997-1 Eq. 9.1 (p. 104), (1 - sin phi') times the square "
    "root of OCR (the OCR choice above gives the value used)")


def _k0_ocr_text(phi, ocr, k0):
    """The choice text for the OCR in EN 1997-1 Eq. 9.1 (GM-245); whether
    it was given or left at 1 is the choice's `how`."""
    text = (f"EN 1997-1:2004 Sec. 9.5.2(3), Eq. 9.1 (p. 104), for a "
            f"horizontal ground surface: K0 = (1 - sin phi') x sqrt(OCR) = "
            f"(1 - sin {phi:g}) x sqrt({ocr:g}) = {k0:.3f}. ")
    if ocr == 1.0:
        text += ("OCR 1 is normally consolidated ground; give OCR for "
                 "overconsolidated ground. ")
    else:
        text += ("The standard says (p. 104): 'The formula should not be "
                 "used for very high values of OCR.' It prints no limit "
                 "value, so whether this OCR is too high is your call.")
    return text.rstrip()

# The input that chose the sloping form before GM-243, refused with this
# reason; a call naming EN 1997-1 Eq. 9.2, the one form, passes.
_NO_K0_FORM = {
    "reason": (
        "the at-rest K0 behind ground rising from the wall is always EN "
        "1997-1:2004 Eq. 9.2 (p. 104), K0;beta = K0 (1 + sin beta); the "
        "tool has no other form. Leave k0_sloping_ground out"),
    "accepts": ["en_1997_1_eq_9_2"]}


def _refuse_unused_k_forms(p, theory, state, source, beta):
    """The log-spiral source is for the log-spiral Kp (GM-183), refused
    where the call does not use it; the at-rest K0 behind sloping ground
    (EN 1997-1 Eq. 9.2, GM-243) is refused where Eq. 9.2 does not reach."""
    from geotech_common import log_spiral as _ls
    if source not in _ls.SOURCES:
        raise ValueError(f"earth_pressure_coefficient: log_spiral_source "
                         f"must be one of {list(_ls.SOURCES)}, got "
                         f"{source!r}.")
    if source != _ls.DEFAULT_SOURCE and not (
            theory == "caquot_kerisel" and state == "passive"):
        raise ValueError(
            f"earth_pressure_coefficient: log_spiral_source {source!r} is "
            "the source of the log-spiral passive Kp (theory "
            "'caquot_kerisel', state 'passive'); this call does not use it. "
            "Leave it out.")
    if state != "at_rest" or not beta:
        return
    if beta < 0:
        raise ValueError(
            "earth_pressure_coefficient: the at-rest K0 behind sloping "
            "ground is EN 1997-1 Eq. 9.2, for ground rising from the wall "
            f"(beta_deg > 0); got beta_deg {beta}, ground falling away from "
            "the wall, which Eq. 9.2 does not cover.")
    if beta > p["phi_deg"]:
        raise ValueError(
            "earth_pressure_coefficient: EN 1997-1 Eq. 9.2 is for ground "
            f"sloping up at beta <= phi'; got beta {beta} on phi' "
            f"{p['phi_deg']}.")


# What each coefficient takes besides phi (GM-184): Rankine's active Ka a
# sloping backfill, its Kp neither wall friction nor a slope; Coulomb's
# wall friction, a battered back and a sloping backfill; the log-spiral Kp
# (Caquot-Kerisel) wall friction and, since GM-183, sloping ground; the
# at-rest K0, since GM-183, sloping ground (EN 1997-1 Eq. 9.2, GM-243). A
# value the chosen one does not take is refused, never echoed as if it had
# been used.
_K_TAKES = {
    ("rankine", "active"): ("beta_deg",),
    ("rankine", "passive"): (),
    ("coulomb", "active"): ("delta_deg", "alpha_deg", "beta_deg"),
    ("coulomb", "passive"): ("delta_deg", "alpha_deg", "beta_deg"),
    ("caquot_kerisel", "passive"): ("delta_deg", "beta_deg", "theta_deg"),
}
_K_NEUTRAL = {"delta_deg": 0.0, "alpha_deg": 90.0, "beta_deg": 0.0,
              "theta_deg": 0.0, "OCR": 1.0}
_K_SAYS = {"delta_deg": "wall friction", "alpha_deg": "a battered wall back",
           "beta_deg": "a sloping backfill",
           "theta_deg": "a battered wall face",
           "OCR": "an overconsolidation ratio"}
# What to use instead, by state and input. Caltrans Trenching and Shoring
# Manual Sec. 4-3: Rankine passive "must NOT be used" for a sloping
# backfill, Coulomb's passive form is not recommended either, and the
# Log-Spiral method (Sec. 4-6) is the one for passive pressure with wall
# friction or slope.
_CT = "the Caltrans Trenching and Shoring Manual"
_OCR_INSTEAD = ("OCR is for the at-rest K0 (EN 1997-1 Eq. 9.1, state "
                "'at_rest'); active and passive coefficients take none")
_K_INSTEAD = {
    ("active", "delta_deg"): "theory 'coulomb' takes wall friction",
    ("active", "alpha_deg"): "theory 'coulomb' takes a battered back",
    ("active", "OCR"): _OCR_INSTEAD,
    ("passive", "OCR"): _OCR_INSTEAD,
    ("active", "theta_deg"): (
        "theory 'coulomb' takes a battered back as alpha_deg, its angle "
        "from horizontal (alpha_deg = 90 - theta_deg)"),
    ("passive", "delta_deg"): (
        "theory 'caquot_kerisel' (the Log-Spiral Kp " + _CT + " Sec. 4-6 "
        "recommends for passive pressure; UFC 3-220-20 p. 190: passive "
        "pressure 'should be calculated using the log spiral method and not "
        "the Coulomb method') or 'coulomb' takes wall friction"),
    ("passive", "alpha_deg"): (
        "theory 'coulomb' takes a battered back; theory 'caquot_kerisel' "
        "(the log-spiral Kp, UFC 3-220-20 Fig. 4-12) takes it as theta_deg, "
        "the face's angle from vertical (theta_deg = 90 - alpha_deg)"),
    ("passive", "theta_deg"): (
        "theory 'caquot_kerisel' (the log-spiral Kp, UFC 3-220-20 Fig. "
        "4-12) takes a battered face; 'coulomb' takes it as alpha_deg = 90 "
        "- theta_deg, though UFC 3-220-20 p. 190 says passive pressure "
        "'should be calculated using the log spiral method and not the "
        "Coulomb method'"),
    ("passive", "beta_deg"): (
        "Rankine's Kp is not for a sloping backfill (" + _CT + " Sec. 4-3: "
        "it gives the same Kp for +beta and -beta); theory 'caquot_kerisel' "
        "takes a slope (by default UFC 3-220-20 Fig. 4-13, the log-spiral "
        "Kp for a vertical face with sloping ground; the Log-Spiral Kp for "
        "sloping ground of its Sec. 4-6 Fig. 4-20, the Caquot-Kerisel (1948) "
        "chart, with log_spiral_source 'caquot_kerisel_1948'), and 'coulomb' "
        "does, though the manual does not recommend Coulomb for passive "
        "pressure"),
    ("caquot_kerisel", "passive", "alpha_deg"): (
        "the log-spiral Kp takes a battered face as theta_deg, UFC "
        "3-220-20's theta, the face's angle from vertical (theta_deg = 90 - "
        "alpha_deg; Fig. 4-12)"),
}
_K_AT_REST_INSTEAD = (
    "the at-rest forms here (Jaky's, and for sloping ground EN 1997-1 Eq. "
    "9.2) are for a vertical wall and take no wall friction")


def _capital(text):
    return text[:1].upper() + text[1:]


def _refuse_unused_k_inputs(p, theory, state):
    takes = (("beta_deg", "OCR") if state == "at_rest"
             else _K_TAKES.get((theory, state), tuple(_K_NEUTRAL)))
    unused = [n for n, neutral in _K_NEUTRAL.items()
              if n not in takes and p.get(n) is not None
              and float(p[n]) != neutral]
    if not unused:
        return
    if state == "at_rest":
        which = ("the at-rest K0 (Jaky's 1 - sin(phi) behind a vertical "
                 "wall, whatever the theory)")
        instead = [_K_AT_REST_INSTEAD]
    else:
        which = f"the {theory} {state} coefficient"
        instead = [_K_INSTEAD.get((theory, state, n), _K_INSTEAD[(state, n)])
                   for n in unused]
    raise ValueError(
        f"earth_pressure_coefficient: {which} does not take "
        + " or ".join(f"{_K_SAYS[n]} ({n} {p[n]})" for n in unused)
        + ". " + _capital("; ".join(instead)) + ". Otherwise leave "
        + ("it" if len(unused) == 1 else "them") + " out (or at "
        + ", ".join(f"{n} {_K_NEUTRAL[n]:g}" for n in unused) + ").")


def _run_list_reinforcement(params):
    result = {}
    for key, r in _REINFORCEMENT_DB.items():
        result[key] = {"name": r.name, "type": r.type,
                       "Tallowable_kN_per_m": round(r.Tallowable, 2),
                       "width_m": r.width,
                       "alpha": table_alpha(r),
                       "mse_wall_needs": _coverage_needs(r)}
        if r.basis:
            result[key]["Tallowable_basis"] = r.basis    # GM-262
    return {"reinforcement_types": result,
            # GM-271 review: worked-example values, not products
            "built_ins": _BUILT_INS_ARE_EXAMPLES,
            "Tallowable": ("Tal per metre of the reinforcement's own width "
                           "(GEC-11 Vol I Eq. 3-11a); mse_wall multiplies "
                           "it by the coverage ratio Rc = b/Sh for the "
                           "tensile resistance per metre of wall (p. "
                           "4-47) and compares that with the unfactored "
                           "Tmax at a factor of safety of 1.0 for steel, "
                           "whose Tal is an allowable strength (0.55 Fy "
                           "or 0.48 Fy, p. 4-77), and of 1.5 for a "
                           "geosynthetic, whose Tal is its long-term "
                           "strength Tult / RF (Eq. 3-12): GEC-11's 1.35 "
                           "/ phi (Table 4-7)")}


#: What mse_wall needs besides a built-in reinforcement's name (GM-256).
_COVERAGE_NEEDS = {
    "metallic_strip": "reinforcement_horizontal_spacing (Sh, m), or "
                      "coverage_ratio",
    "metallic_grid": "reinforcement_width (b, m, centre to centre of the "
                     "outside longitudinal bars) with "
                     "reinforcement_horizontal_spacing (Sh, m), or "
                     "coverage_ratio",
    **{kind: "nothing: a continuous layer, Rc = 1; for strips at less "
             "than full coverage, reinforcement_width with "
             "reinforcement_horizontal_spacing, or coverage_ratio"
       for kind in ("geogrid", "geotextile", "geosynthetic")},
}


def _coverage_needs(r):
    """What mse_wall needs for a built-in's coverage: a steel one that
    carries its own width needs only its spacing (GM-262)."""
    if r.is_metallic and r.width is not None:
        return _COVERAGE_NEEDS["metallic_strip"]
    return _COVERAGE_NEEDS[r.type]


METHOD_REGISTRY = {
    "cantilever_wall": _run_cantilever_wall,
    "mse_wall": _run_mse_wall,
    "mse_lrfd_external_stability": _run_mse_lrfd_external_stability,
    "list_reinforcement": _run_list_reinforcement,
    "earth_pressure_coefficient": _run_earth_pressure_coefficient,
}

METHOD_INFO = {
    "cantilever_wall": {
        "category": "Cantilever Wall",
        "brief": "Full cantilever retaining wall stability analysis (sliding, overturning, eccentricity, bearing); the backfill is taken as drained, with no water table.",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "TOTAL wall height H (m), measured from the UNDERSIDE of the base slab (footing) to the top of the stem: H = base_thickness + stem height. The active thrust acts over H on the vertical plane through the back of the heel. If you know the height retained above the top of the footing, add base_thickness (4.0 m retained on a 0.5 m footing is wall_height 4.5)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees)."},
            "base_width": {"type": "float", "required": False, "description": "Base width (m). Auto-sized if omitted."},
            "toe_length": {"type": "float", "required": False, "description": "Toe length (m). Auto-sized if omitted."},
            "stem_thickness_top": {"type": "float", "required": False, "default": 0.30, "description": "Stem thickness at top (m)."},
            "stem_thickness_base": {"type": "float", "required": False, "description": "Stem thickness at base (m)."},
            "base_thickness": {"type": "float", "required": False, "description": "Base slab thickness (m). Auto-sized if omitted: H/10 (FHWA NHI-06-089 Fig. 10-16(a), p. 10-38: H/12 to H/10)."},
            "stem_batter": {"type": "str", "required": False, "default": "back", "allowed_values": ["back", "front"], "description": "Which face of a tapered stem (thicker at its base than at its top) is battered. 'back' (default): the front face vertical, the backfill over the battered back counted as resisting weight. 'front': the back face vertical, as UFC 3-220-20 App. B-1 Fig. B-2 (p. 598) and FHWA NHI-06-089 Fig. 10-16(a) and Example 10-3 (p. 10-52) draw it."},
            "embedment_depth": {"type": "float", "required": False, "description": "Depth of the underside of the base below the ground in front of the wall, Df (m), 0 to wall_height. Left out: the base thickness, the ground in front at the top of the base. Deeper, the soil over the toe counts as resisting weight (UFC 3-220-20 Fig. 4-27, p. 220: 'Weight includes ... weight of soil above footing'; App. B-1.3, p. 599), at gamma_foundation, and include_passive's passive resistance acts over Df."},
            "has_shear_key": {"type": "bool", "required": False, "default": False, "description": "Include a shear key below the base (with key_depth). It acts only with include_passive: the sliding check then takes the passive resistance over the depth of the key only, from the underside of the base (depth embedment_depth below the ground in front) to the bottom of the key (embedment_depth + key_depth), the trapezoid UFC 3-220-20 Fig. 4-29 draws below the toe (p. 224: 'the passive pressure is applied only along the depth of the key and not along the entire burial depth of the wall'; GEC-6 Fig. 5-22 likewise), in place of the passive over the embedment; with the undrained rule it counts cu, as GEC-6 Eq. 5-41 (p. 101) counts Pp in front of the key. GEC-6 Sec. 5.4 (pp. 100-101): keys in granular soils 'are not very efficient in developing passive pressures'; its key method is for stiff, overconsolidated clays and soft rock. UFC: 'it is also prudent to consider some amount of future excavation in front of the wall'."},
            "key_depth": {"type": "float", "required": False, "default": 0.0, "description": "Shear key depth below the underside of the base slab (m)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform LIVE-load surcharge on the backfill (kPa), e.g. traffic. Its thrust K*q*H drives every check. Over the heel it is NOT a resisting force in sliding, overturning or the eccentricity check by default (live_load_resists), and it does load the base in the bearing pressures (surcharge_in_bearing). A pavement or slab is not a surcharge here: model it as retained soil up to the ground surface (it then rests on the heel and resists like the backfill). Any other surcharge is transient live load: its thrust counts in every check, but its weight over the heel does not resist sliding or overturning. A permanent load you want carried as a surcharge goes in dead_load_surcharge."},
            "dead_load_surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Uniform DEAD-load (permanent) surcharge on the backfill (kPa), a permanent load that is not part of the retained soil (FHWA NHI-06-089 Sec. 10.4.2). Pavements and slabs are normally modelled as retained soil up to the ground surface instead. Its thrust drives every check, added to the live load's. Over the heel it is not a resisting force by default (dead_load_resists=true counts it as resisting weight in sliding and overturning); it loads the base in the bearing pressures (surcharge_in_bearing)."},
            "live_load_resists": {"type": "bool", "required": False, "default": False, "description": "Count the live-load surcharge over the heel (and its share of a sloped or Coulomb thrust's vertical component) as a resisting force in sliding, overturning and the eccentricity check. Default false: FHWA NHI-06-089 Sec. 10.5.5 (p. 10-43), 'the live load surcharge is not considered as a stabilizing force over the heel of the wall when sliding resistance is being checked'; UFC 3-220-20 App. B-1.2 (p. 597), 'For overturning and sliding checks, it is assumed that the traffic pressure does not act over the wall heel'. true = the free body with the load over the heel, which none of these references uses for a live load. The result's choices say which was used."},
            "dead_load_resists": {"type": "bool", "required": False, "default": False, "description": "Count the dead-load surcharge over the heel as a resisting force in sliding, overturning and the eccentricity check. Default false: FHWA NHI-06-089 Sec. 10.5.3 (p. 10-41), for CIP semi-gravity walls the surcharge loads are 'conservatively neglected within the width of the base slab since they contribute to overturning and sliding resistance'. true = counted as a permanent load: GEC-11 Vol I p. 4-22 (AASHTO), loads that increase sliding resistance are included 'only if those loads are permanent'."},
            "surcharge_in_bearing": {"type": "bool", "required": False, "default": True, "description": "Count the surcharges over the heel in the bearing pressures (q_toe, q_heel, FOS_bearing). Default true: UFC 3-220-20 App. B-1.6 (p. 602), 'it is conservative to include the surcharge over the heel of the wall, if this is possible to occur, in the bearing capacity check'; GEC-11 Vol I p. 4-26. false = the surcharge taken as located behind the heel (NHI-06-089 p. 10-41)."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (degrees), rising away from the wall. A backfill descending away from the wall (below 0) is refused."},
            "c_backfill": {"type": "float", "required": False, "default": 0.0, "description": "Backfill cohesion (kPa). The active pressure Ka sigma_v - 2 c sqrt(Ka) is taken as zero where it is negative, the tension zone (UFC 3-220-20 Fig. 4-5 Note A; FHWA NHI-06-089 p. 10-15); the result's choices give the thrust with c neglected and with NHI's minimum active pressure of 5z kN/m2 beside it."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg), FULL value — the sliding check applies delta_b = (2/3)*phi_foundation INTERNALLY. Do NOT pre-reduce (passing 2/3*phi here double-counts the reduction). To set the base friction angle directly, use delta_base instead. Defaults to phi_backfill. phi_foundation = 0 gives an undrained strength (c_foundation = cu): sliding_cohesion then defaults to 'undrained'."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa), FULL value, no 2/3 factor. The sliding check counts it by sliding_cohesion: NOT at all with 'drained' (the default unless phi_foundation is 0), as the undrained shear strength cu with 'undrained' (ca = cu) or 'friction_plus_adhesion' (Ca from UFC 3-220-20 Table 4-1). To set the adhesion directly, use base_adhesion."},
            "sliding_cohesion": {"type": "str", "required": False, "default_depends_on": {"parameter": "phi_foundation", "values": {0: "undrained"}, "otherwise": "drained"}, "allowed_values": ["drained", "undrained", "friction_plus_adhesion"], "description": "How the sliding check counts the foundation soil's cohesion, by the rule a reference prints. Default depends on phi_foundation: 'undrained' when phi_foundation is 0 (c_foundation is then cu), else 'drained'. 'drained': R = V tan delta_b, c_foundation NOT counted (GEC-6 Sec. 5.4 Eq. 5-38, p. 100: 'Equations 5-38 and 5-39 should not be combined for use at the same time'; EN 1997-1 6.5.3(10): 'Any effective cohesion c' should be neglected'). 'undrained': R = ca B alone, no friction, ca = c_foundation taken as the undrained shear strength cu, or base_adhesion (GEC-6 Eq. 5-39: 'ca is typically taken as equal to the undrained shear strength'). 'friction_plus_adhesion': R = V tan delta_b + Ca Be, Ca from UFC 3-220-20 Table 4-1 (p. 184) at c_foundation taken as the undrained shear strength (up to 191.5 kPa), or base_adhesion; Be = B - 2e, the effective base width (UFC 3-220-20 Fig. 4-27 and Sec. 4-5.1.3, p. 223, which adds that the minimum FS 'may also need to be increased when base adhesion is included'). The result's choices give the rule used, its basis and each other rule's FOS_sliding on the same wall."},
            "delta_base": {"type": "float", "required": False, "description": "Base interface friction angle delta_b (deg), used DIRECTLY in the sliding check with NO 2/3 factor. Overrides the (2/3)*phi_foundation default. delta_b = phi_foundation for concrete cast against the ground (GEC-6 Sec. 5.4, p. 99: tan delta 'will be equal to the tangent of the friction angle (tan phi) for the soil supporting the footing'). The result's choices state the delta_b used, its basis and the other reading. Refused with sliding_cohesion 'undrained', which counts no friction."},
            "base_adhesion": {"type": "float", "required": False, "description": "Base adhesion ca (kPa), used DIRECTLY with sliding_cohesion 'undrained' (in place of cu) or 'friction_plus_adhesion' (in place of the UFC 3-220-20 Table 4-1 value). Refused with 'drained', which counts no adhesion: give sliding_cohesion with it."},
            "include_passive": {"type": "bool", "required": False, "default": False, "description": "Include passive resistance in front of the wall in the sliding check (Rankine's Kp unless passive_theory says otherwise; UFC 3-220-20 p. 219: 'typically ignored', and with it considered FSL >= 2.0, Fig. 4-27, so FOS_sliding_required then defaults to 2.0), acting over the embedment, from the ground in front to the underside of the base (embedment_depth, default the base thickness), or with has_shear_key over the depth of the key only (from embedment_depth to embedment_depth + key_depth below the ground in front, UFC 3-220-20 Fig. 4-29). With the undrained rule it counts the foundation's cu, Kp sigma_v + 2 cu sqrt(Kp) (UFC 3-220-20 Eq. 4-7, p. 178, and p. 181); with the other rules c_foundation is not counted in it (p. 179)."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation soil unit weight (kN/m3): the passive resistance in front of the wall and the soil over the toe (embedment_depth below the top of the base). Defaults to gamma_backfill."},
            "passive_theory": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "log_spiral", "caquot_kerisel"], "description": "With include_passive, the passive coefficient in front of the wall: 'rankine' (default) = Rankine's Kp, no wall friction; 'log_spiral' (or 'caquot_kerisel', the same) = the log-spiral Kp of log_spiral_source at passive_wall_friction_deg. UFC 3-220-20 (Sec. 4-5.1, Figs. 4-27 and 4-29, pp. 219-224) names no theory for it; choices give the other theory's Kp and FOS_sliding and the FOS with no passive."},
            "log_spiral_source": {"type": "str", "required": False, "default": "ufc_3_220_20", "allowed_values": ["ufc_3_220_20", "caquot_kerisel_1948", "dm7_2_eq_4_12", "dm7_2_fig_4_12", "dm7_2_fig_4_13"], "description": "With passive_theory 'log_spiral', the printed source of Kp, as earth_pressure_coefficient takes it: 'ufc_3_220_20' (default) = UFC 3-220-20 Eq. 4-12 (p. 191), its log-spiral Kp for a vertical face and level ground at any wall friction; 'caquot_kerisel_1948' = the Caltrans Trenching and Shoring Manual Fig. 4-20 chart, and 'dm7_2_eq_4_12' = UFC Eq. 4-12 by name, both needing passive_wall_friction_deg given; 'dm7_2_fig_4_12' / 'dm7_2_fig_4_13' = UFC's charts, drawn for delta/phi' 0.66 only."},
            "passive_wall_friction_deg": {"type": "float", "required": False, "description": "With passive_theory 'log_spiral', the wall friction delta in front of the wall (deg), 0 to phi_foundation: the engineer's call (UFC 3-220-20 Table 4-1, p. 184). Left out: delta/phi' = 0.66 (UFC p. 191, 'a common assumption'); refused with 'caquot_kerisel_1948' or 'dm7_2_eq_4_12', which need it given (Caltrans p. 4-29: 'Given delta')."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa) for the bearing check. Without it the bearing check is NOT RUN: FOS_bearing and passes_bearing are null and checks_not_run lists 'bearing' (q_toe / q_heel are still reported)."},
            "gamma_concrete": {"type": "float", "required": False, "default": 23.6, "description": "Concrete unit weight (kN/m3). Default 23.6, 150 pcf, as UFC 3-220-20 App. B-1.3 (p. 599) and FHWA NHI-06-089 Example 10-3 (p. 10-53) take it."},
            "FOS_sliding_required": {"type": "float", "required": False, "default_depends_on": {"parameter": "include_passive", "values": {True: 2.0}, "otherwise": 1.5}, "description": "Required sliding FOS; the check passes at or above it. Default depends on include_passive: 1.5 (GEC-6 Eq. 5-40, p. 100; UFC 3-220-20 Fig. 4-27; FHWA NHI-06-089 Fig. 10-17), 2.0 with the passive resistance counted (UFC 3-220-20 Fig. 4-27, p. 220, and Sec. 4-5.1.3, p. 223: 'If passive resistance is included, the minimum FSL should be increased to 2.0'). The result's choices say which was taken."},
            "FOS_overturning_required": {"type": "float", "required": False, "default_depends_on": {"parameter": "founded_on_rock", "values": {True: 1.5}, "otherwise": 2.0}, "description": "Required overturning FOS; the check passes at or above it. Default depends on founded_on_rock: 2.0 on soil, 1.5 on rock (FHWA NHI-06-089 Sec. 10.5.5, p. 10-44: 'approximately of 2.0 for soils and 1.5 for rocks'; GEC-6 Sec. 7.3, p. 124: 'greater than about 2.0 for permanent dead loads'). UFC 3-220-20 Sec. 4-5.1.2 (p. 223) asks FOT >= 1.5 for any wall."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory. The active thrust acts on the vertical plane through the back of the heel. 'rankine' (default): the thrust parallel to the backfill slope. 'coulomb': Coulomb's Ka (UFC 3-220-20 Eq. 4-10) at the wall friction on that plane, wall_friction_deg, by default the backfill slope (FHWA NHI-06-089 p. 10-20 and Example 10-3; delta = 0 behind level backfill). At that default Coulomb's Ka on a vertical plane is Rankine's (sloped) Ka, the same thrust (NHI Example 10-3's 0.35 at phi 30, beta 10 is both), so 'coulomb' changes the result only with wall_friction_deg given."},
            "wall_friction_deg": {"type": "float", "required": False, "description": "Wall friction delta on the vertical plane through the back of the heel (deg), with pressure_method 'coulomb', 0 to phi_backfill. Left out: the backfill slope (FHWA NHI-06-089 p. 10-20: on that plane 'there is soil-to-soil contact ... it is generally conservative to assume that the earth pressure is parallel to the slope of the backfill'). Refused with 'rankine'. The result's choices give the delta used and the other reading's Ka."},
            "interface_gap_possible": {"type": "bool", "required": False, "default": False, "description": "Whether water or air can reach the base-clay interface. With sliding_cohesion 'undrained' (an undrained clay under the base) the base resistance is then held to 0.4 V (EN 1997-1 Sec. 6.5.3(12)P, Eq. 6.5, p. 65). GEC-6 and UFC 3-220-20 print no such limit. Refused with the other rules. The result's choices give FOS_sliding under the other reading."},
            "founded_on_rock": {"type": "bool", "required": False, "default": False, "description": "Whether the wall is founded on rock. On rock the eccentricity limit is B/4, the middle half, in place of B/6 (GEC-6 Sec. 6.4.1, p. 119; UFC 3-220-20 Sec. 4-5.1.2, p. 223; FHWA NHI-06-089 Fig. 10-17), FOS_overturning_required defaults to 1.5 and bearing_pressure to 'linear'."},
            "bearing_pressure": {"type": "str", "required": False, "default_depends_on": {"parameter": "founded_on_rock", "values": {True: "linear"}, "otherwise": "uniform"}, "allowed_values": ["uniform", "linear"], "description": "The bearing pressure compared with q_allowable. Default depends on founded_on_rock: 'uniform' on soil, 'linear' on rock. 'uniform' (the default on soil): q_eq = V / B', B' = B - 2e (GEC-6 Eq. 6-6 and Sec. 6.4.2, p. 119; FHWA NHI-06-089 Fig. 10-17: 'Use uniform stress, qeq, for soils'; UFC 3-220-20 Eq. 4-27 and Sec. 4-5.1.4, p. 224, for soil that is 'relatively ductile'). 'linear' (the default with founded_on_rock): the larger edge pressure of the trapezoid or triangle, q_toe or q_heel (UFC Sec. 4-5.1.4: 'For more brittle materials, such as rock and sensitive clays'; FHWA NHI-06-089 Fig. 10-17). The result's choices give the other pressure's FOS_bearing."},
        },
        "returns": {"verdict": "The result in plain words: the FAILED checks first with their numbers, then those that pass, then those not checked. Report it to the user, failures first.", "checks_failed": "Checks that were run and failed: sliding, overturning, bearing, eccentricity (resultant outside the middle third, or with founded_on_rock the middle half). Never leave one out of the answer.", "checks_not_run": "Checks not run ('bearing' when no q_allowable was given). Say so; never call the wall adequate in bearing.", "all_checks_pass": "true only when every check ran and passed; false when one failed; null when none failed but a check was not run.", "FOS_sliding": "Factor of safety against sliding (null when there is no driving force: the check passes).", "FOS_overturning": "Factor of safety against overturning (null when there is no driving force).", "FOS_bearing": "q_allowable / q_bearing_kPa; null when the bearing check was not run.", "q_bearing_kPa": "The bearing pressure compared with q_allowable, by bearing_pressure: the uniform q_eq = V / (B - 2e) on soil, the larger of q_toe and q_heel with 'linear' (the default on rock).", "q_toe_kPa": "Bearing pressure at the toe.", "q_heel_kPa": "Bearing pressure at the heel.", "eccentricity_m": "Eccentricity of the resultant on the loads the overturning check counts (the eccentricity check: at most eccentricity_limit_m, B/6, or B/4 with founded_on_rock).", "eccentricity_bearing_m": "Present when it differs: the eccentricity on the bearing check's loads (the surcharge over the heel counted), which gives q_toe / q_heel.", "choices": "How the sliding check counted the foundation cohesion (sliding_cohesion, when there is cohesion or adhesion: the rule, its quoted basis and each other rule's FOS_sliding), the base friction delta_b it used (2/3 phi_f by default, or delta_base; none with the undrained rule), how the surcharges over the heel were counted in sliding / overturning / eccentricity and in bearing, and with include_passive the passive coefficient in front of the wall (passive_theory, with the other theory's and no passive's FOS_sliding; with 'log_spiral' its source and wall friction) and whether it counts c_foundation; with Coulomb the wall friction on the heel plane; with c_backfill the tension zone; the required FOS_sliding with the passive counted and FOS_overturning on rock; the bearing pressure compared with q_allowable (with the other pressure's FOS_bearing); with the undrained rule EN 1997-1's 0.4 V limit; each with the basis and the other reading. Tell the user.", "geometry": "The geometry analysed, auto-sized parts included, with stem_batter and embedment_depth_m (Df); wall_height is from the underside of the base to the top of the stem.", "sliding_basis": "The sliding-check basis actually used: sliding_cohesion (the rule for the foundation cohesion), delta_base_deg (null with the undrained rule), base_adhesion_kPa (0 with the drained rule) and adhesion_width_m (B, or Be = B - 2e with friction_plus_adhesion; null with drained), driving/resisting forces, Pa and its vertical component + inclination, and with include_passive the passive resistance Pp_kN_per_m counted (Pp_cohesion_kN_per_m its cu part under the undrained rule), its passive_theory and Kp (with 'log_spiral' its log_spiral_source and passive_wall_friction_deg), and base_resistance_held_to_0_4V when interface_gap_possible held it — verify these match your intended assumptions."},
    },
    "earth_pressure_coefficient": {
        "category": "Earth Pressure",
        "brief": "Lateral earth pressure coefficient Ka/Kp/K0 (Rankine, Coulomb, or the log-spiral passive Kp: by default UFC 3-220-20's Eq. 4-12, Fig. 4-12 or Fig. 4-13 by wall geometry, or the Caquot-Kerisel 1948 chart; K0 also behind sloping ground).",
        "parameters": {
            "phi_deg": {"type": "float", "required": True, "description": "Soil friction angle (degrees). Alias: phi."},
            "theory": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb", "caquot_kerisel", "log_spiral"], "description": "Earth pressure theory. 'caquot_kerisel' or 'log_spiral' (the same) = the log-spiral PASSIVE coefficient Kp (state must be 'passive'; UFC 3-220-20 p. 190: passive pressure 'should be calculated using the log spiral method and not the Coulomb method', which over-predicts Kp at high wall friction), from the printed source log_spiral_source names: by default UFC 3-220-20's for the wall's geometry, 'caquot_kerisel_1948' for the Caquot-Kerisel (1948) chart. Theory is ignored for at_rest, which uses K0 = (1 - sin(phi)) sqrt(OCR) (EN 1997-1 Eq. 9.1; OCR 1 is Jaky's 1 - sin(phi)), and behind ground rising from the wall (beta_deg > 0) EN 1997-1 Eq. 9.2, K0 (1 + sin beta)."},
            "state": {"type": "str", "required": False, "default": "active", "allowed_values": ["active", "passive", "at_rest"], "description": "Pressure state: Ka (active), Kp (passive), or K0 (at rest). 'caquot_kerisel' requires 'passive'."},
            "delta_deg": {"type": "float", "required": False, "description": "Wall friction angle (degrees): Coulomb and caquot_kerisel only; Rankine and K0 take none and refuse a non-zero value. Alias: delta. Not given, it depends on the theory: Coulomb takes delta = 0, no wall friction; caquot_kerisel takes delta = 0.66 phi' with the UFC 3-220-20 sources (p. 191: 'a common assumption', and the one value its Figs. 4-12 and 4-13 are drawn for), and 'caquot_kerisel_1948' and 'dm7_2_eq_4_12' need it given and refuse a call without it (Caltrans T&S p. 4-29: 'Given delta, beta, and phi'); defaults_used gives the value taken. With the default source, a battered face (theta_deg) or sloping ground (beta_deg) at another wall friction is refused: UFC prints none. The wall friction is the engineer's call (UFC 3-220-20 Table 4-1, p. 184, by wall material and soil; FHWA NHI-06-089 p. 10-17 suggests at most 1/2 phi' for passive); the result's choices give the delta used and the Kp at delta = 0."},
            "beta_deg": {"type": "float", "required": False, "default": 0.0, "description": "Slope of the ground behind the wall (degrees, + rising away from the wall; for a passive Kp, the ground in front of the wall). Rankine active, Coulomb, caquot_kerisel (by default UFC 3-220-20 Fig. 4-13: a vertical face, beta/phi' -0.6 to +1 at delta/phi' 0.66; the Caquot-Kerisel 1948 chart, Caltrans T&S Fig. 4-20: beta/phi -0.9 to +1, phi 10 to 45 deg, any wall friction; Eq. 4-12 and Fig. 4-12 are for level ground) and at_rest (EN 1997-1:2004 Eq. 9.2, p. 104: K0;beta = K0 (1 + sin beta), the horizontal component of the at-rest pressure, for ground sloping upwards from the wall at 0 < beta <= phi'; ground falling away is refused). Rankine passive is for level ground and refuses a non-zero value. Aliases: beta, backfill_slope."},
            "alpha_deg": {"type": "float", "required": False, "default": 90.0, "description": "Wall back face angle from horizontal (degrees, 90 = vertical; below 90 when the soil overhangs the face, as UFC 3-220-20 Fig. 4-7 draws a gravity wall's back, where its theta = 90 - alpha_deg is positive). Coulomb only; the others take a vertical back and refuse another value (the log-spiral Kp takes a battered face as theta_deg = 90 - alpha_deg). Alias: alpha."},
            "theta_deg": {"type": "float", "required": False, "default": 0.0, "description": "With theory 'caquot_kerisel' (passive): the wall face's angle from vertical (degrees), UFC 3-220-20's theta (Figs. 4-7, 4-12): positive when the face slopes down into the soil it bears on, so the soil overhangs it (Kp falls; Fig. 4-12 draws the toe of the face farther into the soil than its top), negative the other way (Kp rises); theta_deg = 90 - Coulomb's alpha_deg. A non-zero value reads UFC 3-220-20 Fig. 4-12 (p. 192): level ground only, theta -30 to +40 deg, delta/phi' 0.66 only, Kp up to 15. Alias: theta."},
            "OCR": {"type": "float", "required": False, "default": 1.0, "description": "Overconsolidation ratio for the at-rest K0 only (state 'at_rest'): EN 1997-1:2004 Sec. 9.5.2(3), Eq. 9.1 (p. 104), K0 = (1 - sin phi') x sqrt(OCR), then Eq. 9.2 behind sloping ground. 1 = normally consolidated (the default; the result's choices say whether it was given or taken as 1). At least 1. The standard says 'The formula should not be used for very high values of OCR' and prints no limit, so the choices quote that for OCR above 1 and leave the call to you. Aliases: ocr, overconsolidation_ratio. Refused with active or passive, which take none."},
            "log_spiral_source": {"type": "str", "required": False, "default": "ufc_3_220_20", "allowed_values": ["ufc_3_220_20", "caquot_kerisel_1948", "dm7_2_eq_4_12", "dm7_2_fig_4_12", "dm7_2_fig_4_13"], "description": "With theory 'caquot_kerisel', the printed source of the log-spiral passive Kp. 'ufc_3_220_20' (default): UFC 3-220-20 (DM 7.2), the source the wall's geometry calls for (pp. 191-193): Eq. 4-12 (p. 191) for a vertical face and level ground, at any wall friction; Fig. 4-12 (p. 192) for a battered face (theta_deg) with level ground and Fig. 4-13 (p. 193) for a vertical face with sloping ground (beta_deg), both drawn for delta/phi' = 0.66 only. UFC prints nothing for a battered face with sloping ground, nor for those two charts at another wall friction: the default refuses those and names what covers them. 'caquot_kerisel_1948': the Caquot and Kerisel (1948) chart, Kp at delta = phi times the printed reduction factor R(delta/phi), Caltrans Trenching and Shoring Manual Fig. 4-20 (p. 4-30), the method its Sec. 4-6 uses for passive pressure; a vertical wall, level or sloping ground, any wall friction, given as delta_deg. 'dm7_2_eq_4_12', 'dm7_2_fig_4_12', 'dm7_2_fig_4_13': each UFC source on its own. They differ: at phi 30, delta/phi 0.66 on level ground the 1948 chart gives 5.4, Figs. 4-12 and 4-13 5.3, Eq. 4-12 5.1. UFC 3-220-20 footnote 21 (p. 191): the 1982 DM 7.2 charts 'used delta/phi' = 1 along with reduction ratios', which 'were averages that introduced substantial inaccuracy for some cases'; the 1948 chart takes that form. UFC 3-220-20 Table 4-2 (p. 194) is not used: its row headed delta/phi' = 0.5 prints Fig. 4-13's values for 0.66 (5.3, 8.0, 12, 20 at phi' 30, 35, 40, 45) and Coulomb's Kp at 0.66 (6.1, 9.8, 18, 44); Eq. 4-12 at 0.5 gives 4.46, 6.33, 9.33, 14.3. The result's choices give the route taken and each other source's Kp."},
        },
        "returns": {"K": "Earth pressure coefficient value.", "coefficient": "Ka, Kp, or K0.", "theory": "Theory used: for at_rest 'jaky_at_rest' on level ground at OCR 1, 'en_1997_1_eq_9_1' on level ground at OCR above 1, else 'en_1997_1_eq_9_2' (EN 1997-1 Eq. 9.2, ground rising from the wall).", "delta_deg": "The wall friction used. Left out: 0 for Coulomb (Rankine and K0 take none, 0); 0.66 phi' for caquot_kerisel with the UFC sources, while the 1948 chart and Eq. 4-12 by name refuse it.", "theta_deg": "With caquot_kerisel: the wall face angle from vertical used.", "log_spiral_source": "With caquot_kerisel: the printed source of the Kp.", "log_spiral_route": "With the default 'ufc_3_220_20': the UFC source the geometry called for ('dm7_2_eq_4_12', 'dm7_2_fig_4_12' or 'dm7_2_fig_4_13').", "choices": "The log-spiral source and route, with its basis and each other source's Kp for the case, and the wall friction used, given or left out, with the Kp at delta = 0 beside it; or for at_rest the K0 equation (EN 1997-1 Eq. 9.1, with whether OCR was given or taken as 1, and Eq. 9.2 behind sloping ground) with its arithmetic, page and the sign of beta. Tell the user."},
    },
    "mse_wall": {
        "category": "MSE Wall",
        "brief": "MSE wall external + internal stability analysis (GEC-11), level or sloping backfill (incl. a broken backslope).",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Design wall height H (m): the height of the reinforced soil mass, incl. embedment (GEC-11)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (degrees)."},
            "reinforcement_name": {"type": "str", "required": False, "description": "Built-in reinforcement name for every level, or omit and define a custom reinforcement via reinforcement_type + reinforcement_Tallowable; for a different reinforcement at different levels give reinforcement_per_level instead. The built-ins are reinforcements GEC-11 Vol II's worked examples print, with their Tal as printed: 'ribbed_steel_strip_50x4' (Example E3's 50 x 4 mm ribbed strip), 'welded_wire_grid_w11' (Example E4's 4W11 bar mat, W11 transverse wires at 0.5 ft), 'bar_mat_e4_3w11_w11x0_5', 'bar_mat_e4_4w15_w11x1' and 'bar_mat_e4_4w15_w11x1_5' (E4's other mats, Table E4-7.4, p. E4-18), 'geogrid_gec11_e1_gg1', 'geogrid_gec11_e1_gg2' and 'geogrid_gec11_e1_gg3' (Example E1's geogrids GG-I to GG-III); list_reinforcement gives each one's Tal and its basis. They are the worked examples' reinforcements, not products: where the wall's reinforcement is specified, give it as a custom reinforcement with its own Tal (for a geosynthetic, Tult / RF for the wall's backfill and design life). A steel strip or grid also needs its horizontal spacing (reinforcement_horizontal_spacing) or coverage_ratio. A built-in carries its own Tal, type and bar geometry, and refuses reinforcement_Tallowable, reinforcement_type, reinforcement_thickness and reinforcement_transverse_spacing."},
            "reinforcement_per_level": {"type": "list", "required": False, "description": "A reinforcement for each level, from the top down, instead of reinforcement_name and the custom reinforcement_* inputs, as GEC-11 Vol II Example E4 takes 4W11 mats at Levels 1, 3 and 4, a 3W11 mat at Level 2 and 4W15 mats from Level 5 down (Table E4-7.4, p. E4-18). One entry a level, as many as the wall has levels (one a depth of level_depths; without it wall_height / reinforcement_spacing rounded down, at reinforcement_spacing x (i + 1/2) from the top); a list of another length is refused, naming the levels' depths. Each entry is a built-in name (see reinforcement_name and list_reinforcement) or a custom reinforcement {Tallowable, type, width, thickness, transverse_spacing, name}, the keys as the reinforcement_* inputs without the prefix. reinforcement_horizontal_spacing and coverage_ratio apply level by level, one number for the wall or a list with one per level. Pullout and rupture take each level's own Tal, width, Rc, F* and alpha, and each level of the result names its reinforcement. Example, a 3 m wall at reinforcement_spacing 0.75 (four levels) with reinforcement_horizontal_spacing 1.524: [\"welded_wire_grid_w11\", \"bar_mat_e4_3w11_w11x0_5\", \"bar_mat_e4_4w15_w11x1\", {\"name\": \"Mat X\", \"type\": \"metallic_grid\", \"Tallowable\": 150, \"width\": 0.457, \"thickness\": 0.0095, \"transverse_spacing\": 0.457}]. Left out, one reinforcement holds at every level (reinforcement_name, or the custom reinforcement_* inputs). This tool refuses a wall that mixes steel and geosynthetic levels, its choice: GEC-11 Vol I p. 4-33 and Fig. 4-9 give one failure surface for inextensible and another for extensible reinforcement, and none for a wall with both."},
            "reinforcement_horizontal_spacing": {"type": "float or list", "required": False, "description": "Centre-to-centre horizontal spacing Sh (m) of the strips, grids or sheets at one level (GEC-11 Vol I p. 3-12): one number for the wall, or a list with one per reinforcement level from the top down, as GEC-11 Vol II Example E3 changes it with depth (Table E3-7.3, p. E3-19: 2 strips a 5 ft panel at Levels 1 to 8, 3 at Levels 9 to 12); a list must have as many values as the wall has levels. With the reinforcement's width b it gives the coverage ratio Rc = b/Sh (Eq. 3-1), which scales the pullout resistance (Eq. 4-35, p. 4-49) and the tensile resistance (Tal x Rc, p. 4-47) to a metre of wall. Required for a steel strip or grid unless coverage_ratio is given instead; a call with neither is refused, since GEC-11 prints no spacing to assume (Vol II Example E3 uses 50 mm strips at 0.76 and 0.51 m, 2 or 3 per 1.5 m panel). A geosynthetic with neither is a continuous layer, Rc = 1 (p. 3-13). Aliases: Sh, horizontal_spacing."},
            "coverage_ratio": {"type": "float or list", "required": False, "description": "Coverage ratio Rc = b/Sh (GEC-11 Vol I Eq. 3-1, p. 3-12), more than 0 and at most 1, given directly instead of reinforcement_horizontal_spacing with reinforcement_width (not both): one number for the wall, or a list with one per reinforcement level from the top down. Alias: Rc."},
            "F_star_tested": {"type": "float or list", "required": False, "description": "Pullout resistance factor F* from pullout tests on the product in the wall's backfill (GEC-11 Vol I Table 3-6 note (i), p. 3-17; Appendix B): one number at every level, or [[depth_m, F*], ...] with depths below the top of the wall, linear between and the end value beyond. Replaces the empirical F* (Eqs. 3-4 to 3-8); note (i) accepts the empirical values only for Table 3-1 granular fill with Cu >= 4 ('For fill outside these limits, tests must be run'). The result's choices give FOS_pullout at the empirical F* beside it."},
            "alpha_tested": {"type": "float", "required": False, "description": "Scale effect correction factor alpha from pullout tests on reinforcements of different lengths (GEC-11 Vol I p. 3-16, Appendix B), more than 0 and at most 1. Left out, Table 3-6's alpha for the reinforcement type (p. 3-17): 1.0 for steel strips and grids, 0.8 for geogrids, 0.6 for geotextiles; a reinforcement_type 'geosynthetic' needs it. Table 3-6 note (ii): 'Pullout testing to determine alpha is recommended if alpha shown in table is less than 1.0'. The result's choices give FOS_pullout at the table's alpha beside a tested one."},
            "Cu_backfill": {"type": "float", "required": False, "default": 4.0, "description": "Uniformity coefficient Cu = D60/D10 of the reinforced backfill, for the pullout resistance factor of ribbed steel strips: F* = 1.2 + log Cu at the top of the wall, 2.0 maximum, falling linearly to tan phi_backfill at 20 ft (6 m) and below (GEC-11 Vol I Eqs. 3-4 and 3-5, p. 3-20). Left out, Cu = 4 (F* 1.8 at the top): p. 3-20, 'If the specific Cu for the wall backfill is unknown at design time, a Cu = 4 should be assumed' for fills meeting its Sec. 3.1. Table 3-6 note (i) (p. 3-17) accepts these empirical values only for Table 3-1 granular fill with Cu >= 4 ('For fill outside these limits, tests must be run'); a Cu below 4 is computed and the choices say so. A steel grid given t/St and a geosynthetic keep their own F*, which takes no Cu; given with them, Cu is checked against Table 3-6 note (i) only. The result's choices give the F* used and FOS_pullout at the top level and the lowest at the other reading, or for a grid or geosynthetic whether the given Cu is within note (i). With F_star_tested it is in no check: the tested F* replaces the empirical one, and the F* choice says what the Cu bears on (a strip's empirical F* given beside the tested one, and note (i)'s check on the empirical F*). Aliases: Cu, uniformity_coefficient."},
            "reinforcement_type": {"type": "str", "required": False, "allowed_values": ["metallic_strip", "metallic_grid", "bar_mat", "welded_grid", "geogrid", "geotextile", "geosynthetic"], "description": "Custom reinforcement type, required with reinforcement_Tallowable. Drives the internal-stability lateral-stress-ratio Kr/Ka and pullout F* curves: ribbed STRIP (metallic_strip: Kr/Ka 1.7->1.2, F* 1.2 + log Cu (2.0 at most; Cu_backfill) -> tan-phi), steel bar mat / welded grid (metallic_grid / bar_mat / welded_grid: Kr/Ka 2.5->1.2, F* 20(t/St)->10(t/St) — supply reinforcement_thickness=t and reinforcement_transverse_spacing=St), or geogrid / geotextile (Kr/Ka 1.0, F* 0.67 tan phi). It also sets the scale effect correction factor alpha, GEC-11 Vol I Table 3-6 (p. 3-17): 1.0 steel, 0.8 geogrid, 0.6 geotextile; 'geosynthetic' (either kind) needs alpha_tested."},
            "reinforcement_Tallowable": {"type": "float", "required": False, "description": "Long-term allowable tensile strength Tal (kN/m) of a custom reinforcement, per metre of the reinforcement's own width: a strength per strip or per mat is that divided by its width b. The rupture check compares Tal x Rc, per metre of wall (p. 4-47), with the unfactored Tmax. For steel it requires a factor of safety of 1.0, so Tal is an allowable, a fraction of GEC-11 Vol I Eq. 3-11a (p. 3-22), Fy Ac / b with Ac the area after corrosion: 0.55 Fy for strips and 0.48 Fy for bar mats (p. 4-77); a bar mat's Ac is the number of longitudinal bars times one bar's corroded area (Fig. 3-5, p. 3-24). For a geosynthetic it requires 1.5 (1.35 / phi 0.90, Table 4-7), so Tal is the long-term strength Tult / RF (Eq. 3-12, p. 3-28), not divided by a factor of safety. Required when reinforcement_name is not a built-in."},
            "reinforcement_length": {"type": "float", "required": False, "description": "Reinforcement length (m). Auto-sized (0.7H minimum) if omitted."},
            "reinforcement_spacing": {"type": "float", "required": False, "default": 0.6, "description": "Vertical reinforcement spacing Sv (m): the levels sit at Sv (i + 1/2) below the top of the wall, wall_height / Sv of them rounded down, each taking Sv in Tmax and sigma_H at its depth; when wall_height is not a whole number of spacings the bottom level's contributory height runs from the midpoint above it to the base of the wall, more than Sv, with sigma_H at its middle (GEC-11 Vol I p. 4-47, Fig. 4-14). A spacing of twice wall_height or more is refused. Not taken with level_depths, which places the levels itself."},
            "level_depths": {"type": "list", "required": False, "description": "The depths z (m) of the reinforcement levels below the top of the wall, from the top down, instead of reinforcement_spacing, as GEC-11 Vol II's examples print Z (Example E4, Table E4-7.1, p. E4-11: 1.87 ft, then 2.5 ft apart to 24.37 ft). Each must lie inside the wall (0 < z < wall_height) and below the one above; a list that does not is refused. Each level then takes its contributory height Svt (GEC-11 Vol I p. 4-46): from the midpoints between it and its neighbours, the top level's from the top of the wall and the bottom level's to its base (p. 4-47, Fig. 4-14), and Tmax = sigma_H x Svt with sigma_H averaged over Svt (sigma_H_average). Each level of internal_results then also gives Svt_m and sigma_H_kPa. reinforcement_per_level then needs one entry a depth. Example, a 6 m wall: [0.4, 1.15, 1.9, 2.65, 3.4, 4.15, 4.9, 5.65]. Alias: reinforcement_depths."},
            "sigma_H_average": {"type": "str", "required": False, "default": "trapezoidal", "allowed_values": ["trapezoidal", "midpoint"], "description": "With level_depths: how sigma_H is averaged over each level's contributory height Svt, the two GEC-11 Vol I p. 4-47 gives: 'trapezoidal' (default), the average of sigma_H at the top and the bottom of Svt, as Vol II Examples E3 and E4 take it at every level (pp. E3-15, E4-13); 'midpoint', sigma_H at the middle of Svt. They differ only where Kr/Ka changes with depth (steel reinforcement, within 6.1 m (20 ft) of the top). Refused without level_depths."},
            "reinforcement_width": {"type": "float", "required": False, "description": "Gross width b (m) of the strip, grid or sheet; for a grid, centre to centre of the outside longitudinal bars (GEC-11 Vol I p. 3-12, Fig. 3-3). Required with reinforcement_horizontal_spacing for Rc = b/Sh, except for a built-in that carries its own (ribbed_steel_strip_50x4: 0.05 m; welded_wire_grid_w11 and E4's 4W15 mats: 0.457 m, 4 wires at 0.152 m; bar_mat_e4_3w11_w11x0_5: 0.305 m, 3 wires)."},
            "reinforcement_thickness": {"type": "float", "required": False, "default": 0.0, "description": "Strip thickness (m) for a strip; transverse-bar diameter t (m) for a steel bar mat / welded grid (the t in F* = 20 t/St)."},
            "reinforcement_transverse_spacing": {"type": "float", "required": False, "default": 0.0, "description": "Transverse-bar (grid) spacing St (m) for a steel bar mat / welded grid (the St in F* = 20 t/St). Used only for bar_mat/welded_grid/metallic_grid."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Live-load (traffic) surcharge (kPa), e.g. heq*gamma. Externally its thrust drives every check; over the reinforced zone it is not a resisting force in sliding or overturning and it loads the base for bearing (GEC-11 Fig. 4-2; live_load_resists, surcharge_in_bearing). Internally it is in Tmax and not in the pullout resistance, which GEC-11 takes 'neglecting traffic loads' (p. 4-49). With a sloping backfill it is taken only with broken_backslope_height, as traffic on the level ground behind the crest; refused on a slope that continues."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Slope of the backfill surface above the wall, beta (degrees from horizontal), rising away from the face; 0 = level. GEC-11 Section 4.4.4: Coulomb Kab with the interface angle delta = beta (Eqs. 4-2/4-3); thrust F_T = 1/2*Kab*gamma_retained*h^2 on h = H + L*tan(beta), inclined at beta (Eq. 4-7); the slope fill over the reinforced zone is added to the weight; internally the slope adds S_eq = 1/2*(0.7H)*tan(beta) of fill (Fig. 4-11). Must not be steeper than phi_retained."},
            "phi_retained": {"type": "float", "required": False, "description": "Friction angle (deg) of the retained fill behind the reinforced zone, for the external thrust. Defaults to phi_backfill."},
            "gamma_retained": {"type": "float", "required": False, "description": "Unit weight (kN/m3) of the retained fill behind the reinforced zone. Defaults to gamma_backfill."},
            "broken_backslope_height": {"type": "float", "required": False, "description": "For a broken backslope (GEC-11 Fig. 4-4): the height S (m) of the slope crest above the top of the wall, with level ground behind it. Omit for a slope that continues. Kab and the thrust inclination then use I, the angle from the top of the wall to the ground 2H behind the face, and h = H + min(L*tan(beta), S)."},
            "traffic_vertical_resists": {"type": "bool", "required": False, "default": False, "description": "Broken backslope with traffic (surcharge) behind the crest only: count the traffic thrust's vertical component F2*sin(I) as sliding and eccentricity resistance, as GEC-11 Example E1 does (Vol II pp. E1-7 to E1-9). Default false: the GEC-11 text counts loads that increase sliding resistance 'only if those loads are permanent' (Vol I p. 4-22). Bearing counts it either way (p. 4-26). The result's backslope.traffic_basis says which was used."},
            "live_load_resists": {"type": "bool", "required": False, "default": False, "description": "Count the live-load surcharge over the reinforced zone (q*L; behind the crest of a broken backslope, q over the level ground on the zone) as resisting weight in sliding and overturning. Default false: GEC-11 Vol I p. 4-19, 'The live load surcharge is not considered as a stabilizing force when checking sliding'; p. 4-21 (Eq. 4-12), 'the live load is excluded since it increases sliding stability'; p. 4-22, loads that increase sliding resistance only 'if those loads are permanent', and the eccentricity (overturning) check 'only applies the live load above the retained backfill' (Fig. 4-2). true = the free body with the load over the zone, W = gamma*H*L + q*L. The result's choices say which was used."},
            "surcharge_in_bearing": {"type": "bool", "required": False, "default": True, "description": "Count the live-load surcharge over the reinforced zone in the bearing pressure sigma_v and its eccentricity. Default true: GEC-11 Vol I p. 4-26, 'The bearing check applies live load above both the reinforced zone and the retained backfill' (Fig. 4-2; Eqs. 4-19, 4-20). false = the live load taken as not over the zone."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg)."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa)."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa). Without it the external bearing check is NOT RUN (FOS_bearing null, checks_not_run lists 'bearing'); sigma_v is still reported."},
        },
        "returns": {"verdict": "The result in plain words: failed checks first, then those not checked. Report it to the user.", "choices": "How a live-load surcharge over the reinforced zone was counted in sliding / overturning and in bearing; the coverage ratio Rc (given, b/Sh, or 1 for a continuous geosynthetic) and the tensile resistance Tal x Rc per metre of wall; a built-in reinforcement's Tal and where it comes from (the built-in grid: the rigid facing it assumes, with the flexible-facing value beside it); for a geosynthetic, the factor of safety 1.5 its rupture check requires (each level's FOS_rupture_required: 1.0 for steel, 1.5 for a geosynthetic); where FOS_rupture is below it, the levels, and a stronger reinforcement at those levels (reinforcement_per_level) or a closer spacing; with reinforcement_per_level each level's Tal by runs of the same reinforcement; a tested F* or alpha with FOS_pullout at the empirical value; and for ribbed steel strips the pullout factor F* at the top of the wall (Cu_backfill, given or the Cu = 4 GEC-11 p. 3-20 says to assume when unknown; for another reinforcement a given Cu checked against GEC-11 Table 3-6 note (i)); with level_depths each level's contributory height and how sigma_H was averaged over it (sigma_H_average), without them the levels equally spaced at reinforcement_spacing, each over its contributory height (the bottom level's to the base) with sigma_H at its middle; each with the basis and the other reading. Tell the user.", "checks_failed": "Checks run and failed: sliding, overturning, bearing, internal (pullout or rupture at some level).", "checks_not_run": "Checks not run ('bearing' when no q_allowable).", "reinforcement_name": "The reinforcement's name; with reinforcement_per_level, a list of each level's from the top.", "internal_results": "Each level from the top: depth_m, Tmax, the pullout resistance Pr and the tensile resistance Tal x Rc per metre of wall, F*, alpha, Rc, FOS_pullout and FOS_rupture with FOS_rupture_required; with reinforcement_per_level also its reinforcement; with level_depths also Svt_m, its contributory height, and sigma_H_kPa, the unfactored sigma_H averaged over it (Tmax = sigma_H x Svt).", "all_checks_pass": "true only when every check ran and passed; false when one failed; null when none failed but a check was not run.", "FOS_sliding": "Sliding FOS.", "FOS_overturning": "Overturning FOS.", "FOS_bearing": "Bearing FOS q_allowable / sigma_v; null when not checked.", "passes_external": "true / false, or null when nothing failed but bearing was not checked.", "backslope": "With a sloping backfill only: the GEC-11 basis used (Kab, interface angle, h, F_T and its components, the slope fill weight V2; S_eq, sigma_2 and H1 for internal stability). Each internal level then also gives Zp_m, the depth below the slope used for pullout."},
    },
    "mse_lrfd_external_stability": {
        "category": "MSE Wall",
        "brief": "MSE wall external stability by AASHTO/GEC-11 LRFD: sliding / eccentricity / bearing capacity:demand ratios (CDRs) under Strength I (max+min) and Service I, level or sloping backfill. Distinct from mse_wall, which reports ASD factors of safety.",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Design wall height H (m), incl. embedment. Alias: height."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Reinforced fill unit weight (kN/m3). Alias: gamma."},
            "phi_backfill": {"type": "float", "required": True, "description": "Reinforced fill friction angle (deg). Alias: phi."},
            "reinforcement_length": {"type": "float", "required": False, "description": "Reinforcement length L (m). Auto-sized (0.7H) if omitted."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Live-load surcharge q (kPa) over the mass (e.g. heq*gamma). Used as the LL unless live_load overrides. With a broken backslope it is traffic on the level ground behind the crest (thrust Kab*q*h, inclined at I; GEC-11 Example E1); refused on a slope that continues."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Slope of the backfill surface above the wall, beta (degrees from horizontal), rising away from the face; 0 = level. GEC-11 Section 4.4.6 (Examples E3, E1): Coulomb Kab with delta = beta; F_T = 1/2*Kab*gamma_retained*h^2 on h = H + L*tan(beta) (Eq. 4-7), inclined at beta; F_T*cos(beta) drives and F_T*sin(beta) resists, both EH-factored (Eqs. 4-11, 4-13, 4-16, 4-21); the slope fill over the zone (V2) is an EV load. Must not be steeper than phi_retained."},
            "broken_backslope_height": {"type": "float", "required": False, "description": "For a broken backslope (GEC-11 Fig. 4-4): the height S (m) of the slope crest above the top of the wall, with level ground behind it. Omit for a slope that continues. Kab and the thrust inclination then use I, the angle from the top of the wall to the ground 2H behind the face, and h = H + min(L*tan(beta), S)."},
            "traffic_vertical_resists": {"type": "bool", "required": False, "default": False, "description": "Broken backslope with traffic (surcharge) behind the crest only: count the traffic thrust's vertical component F2*sin(I) as sliding and eccentricity resistance, as GEC-11 Example E1 does (Vol II pp. E1-7 to E1-9). Default false: the GEC-11 text counts loads that increase sliding resistance 'only if those loads are permanent' (Vol I p. 4-22). Bearing counts it either way (p. 4-26). The result's backslope.traffic_basis says which was used."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation friction angle (deg); the sliding interface uses min(phi_backfill, phi_foundation). Defaults to phi_backfill."},
            "phi_retained": {"type": "float", "required": False, "description": "Retained (behind-mass) fill friction angle (deg) for the external active thrust. Defaults to phi_backfill."},
            "gamma_retained": {"type": "float", "required": False, "description": "Retained fill unit weight (kN/m3). Defaults to gamma_backfill."},
            "live_load": {"type": "float", "required": False, "description": "Live-load surcharge q (kPa) if different from `surcharge`."},
            "bearing_resistance_strength": {"type": "float", "required": False, "description": "Factored bearing resistance qR at the strength limit (kPa) = phi_b*qn; enables the strength/critical bearing CDR = qR/sigma_v. Give it or gamma_foundation, not both."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation soil unit weight (kN/m3). Given in place of bearing_resistance_strength, it works the strength bearing resistance out for each Strength I combination, as GEC-11 Vol II Example E2 does: qn = c_foundation*Nc + 0.5*L'*gamma_foundation*Ngamma (GEC-11 Vol I Eq. 4-22; Nc, Ngamma from Table 4-6 for phi_foundation, which must then be given: the reinforced fill's phi_backfill is not taken for it) over that combination's L' = L - 2eB (L' = L when eB < 0), for a level grade in front of the wall and no groundwater influence, the wall's embedment neglected (p. 4-28); qR = phi_bearing*qn (Eq. 4-23). For a slope in front of the wall or a high water table work qn out by AASHTO 10.6.3.1.2 and give bearing_resistance_strength."},
            "phi_bearing": {"type": "float", "required": False, "default": 0.65, "description": "Bearing resistance factor phi_b for the qn worked out from gamma_foundation (GEC-11 Table 4-5: 0.65)."},
            "bearing_resistance_service": {"type": "float", "required": False, "description": "Service bearing resistance (kPa); enables the Service I bearing CDR."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation cohesion (kPa) for sliding adhesion (2/3 c), and c_f in qn when gamma_foundation is given."},
            "gamma_EV_max": {"type": "float", "required": False, "default": 1.35, "description": "Max vertical-earth (EV) load factor (AASHTO Strength I)."},
            "gamma_EV_min": {"type": "float", "required": False, "default": 1.00, "description": "Min vertical-earth (EV) load factor."},
            "gamma_EH_max": {"type": "float", "required": False, "default": 1.50, "description": "Max horizontal-earth (EH) load factor."},
            "gamma_EH_min": {"type": "float", "required": False, "default": 0.90, "description": "Min horizontal-earth (EH) load factor."},
            "gamma_LL": {"type": "float", "required": False, "default": 1.75, "description": "Live-load (LL) factor."},
            "phi_sliding": {"type": "float", "required": False, "default": 1.0, "description": "Sliding resistance factor (AASHTO 11.10.5.3)."},
            "ecc_limit_ratio": {"type": "float", "required": False, "default": 0.25, "description": "Limiting-eccentricity ratio e_max/L (0.25 = L/4 for MSE on soil; use 1/6 on rock)."},
        },
        "returns": {"sliding": "Sliding CDRs (strength max/min, critical, governing).", "eccentricity": "eL (strength max, critical) vs e_limit.", "bearing": "sigma_v (strength/critical/service) + bearing CDRs; checked false and passes null when no bearing resistance was given. With gamma_foundation: qn and qR for Strength I max and the critical combination, Nc, Ngamma, phi_bearing and resistance_basis (GEC-11 Eq. 4-22).", "choices": "With gamma_foundation: how the factored bearing resistance qR was worked out (phi_b, given or default, and the Eq. 4-22 basis) and the other way (a given bearing_resistance_strength). Tell the user.", "passes": "true only when every check ran and passed; false when one failed; null when none failed but one was not run.", "checks_failed": "Checks that failed. Report every one.", "checks_not_run": "Checks not run for want of a bearing resistance: 'bearing (Strength I)', 'bearing (Service I)'. Say so.", "backslope": "With a sloping backfill only: the GEC-11 basis used (Kab, interface angle, h, F_T and its components, the slope fill weight V2)."},
    },
    "list_reinforcement": {
        "category": "MSE Wall",
        "brief": "List the built-in reinforcements (GEC-11 Vol II worked examples' strip, bar mats and geogrids) with their Tal and its basis.",
        "parameters": {},
        "returns": {"reinforcement_types": "The built-ins by name: GEC-11 Vol II worked examples' reinforcements, each with its Tal as printed, its basis (Tallowable_basis) and what mse_wall needs with it; not products.", "built_ins": "What the built-ins are, and how to give the reinforcement specified for the wall.", "Tallowable": "What Tal is and how the rupture check takes it."},
    },
}

declare_aliases(METHOD_INFO, {
    "cantilever_wall": _WALL_ALIASES,
    "mse_wall": _MSE_WALL_ALIASES,
    "mse_lrfd_external_stability": _MSE_ALIASES,
    "earth_pressure_coefficient": _K_ALIASES,
})

# Names the MSE methods do not take, with the reason dispatch gives (GM-110).
# A sloping backfill is taken since GM-112; what it does not cover (a slope
# steeper than phi, a surcharge on a slope that continues) is refused by
# the module with its reason.
METHOD_INFO["mse_lrfd_external_stability"]["refused_parameters"] = {
    "reinforcement_spacing": (
        "external stability does not depend on it; mse_wall's internal "
        "checks use it")}
METHOD_INFO["mse_lrfd_external_stability"]["refused_parameters"][
    "Cu_backfill"] = (
        "external stability does not depend on it; mse_wall's pullout "
        "check uses it (the ribbed steel strip F*, GEC-11 Eq. 3-4)")
# GM-256: the reinforcement's coverage and tested pullout factors
METHOD_INFO["mse_lrfd_external_stability"]["refused_parameters"].update({
    name: ("external stability does not depend on it; mse_wall's pullout "
           "and rupture checks use it")
    for name in ("reinforcement_horizontal_spacing", "coverage_ratio",
                 "F_star_tested", "alpha_tested")})
# GM-179: taken and not used before; mse_lrfd_external_stability uses it.
METHOD_INFO["mse_wall"]["refused_parameters"] = {
    "gamma_foundation": (
        "mse_wall does not use the foundation soil's unit weight: its "
        "bearing check compares the bearing pressure with q_allowable (it "
        "works no bearing resistance out) and it counts no passive "
        "resistance. Give q_allowable, or run mse_lrfd_external_stability "
        "with gamma_foundation (and phi_foundation, c_foundation) to work "
        "the bearing resistance out by GEC-11 Eq. 4-22"),
    # GM-269: taken and not used before (the dataclass keeps a built-in's)
    "reinforcement_Fy": (
        "mse_wall takes no yield strength: the rupture check works from "
        "the reinforcement's Tal (reinforcement_Tallowable). For steel give "
        "the allowable worked out from Fy and the corroded section, "
        "GEC-11 Vol I Eq. 3-11a (p. 3-22), Fy Ac / b: 0.55 Fy for strips "
        "and 0.48 Fy for bar mats (p. 4-77)")}
# GM-243: the sloping-ground at-rest form is no longer a choice.
METHOD_INFO["earth_pressure_coefficient"]["refused_parameters"] = {
    "k0_sloping_ground": _NO_K0_FORM}

# When each default applies (GM-119): the key depth to a wall with a shear
# key, the custom reinforcement's properties to a custom reinforcement (a
# reinforcement_Tallowable and no built-in name), and the traffic thrust's
# vertical component to a sloping backfill carrying a surcharge; since
# GM-165 the foundation soil's unit weight to passive resistance, its
# friction angle to the base friction it sets and to passive, the base
# adhesion to a foundation cohesion, and a crest height to a sloping
# backfill; since GM-184 each earth pressure input to the coefficients
# that take it, and a shear key to passive resistance (it acts only
# there).
_TRAFFIC = {"traffic_vertical_resists": {"backfill_slope": "set",
                                         "surcharge": "set"}}
_COULOMB = ["coulomb"]
_ACTIVE_PASSIVE = ["active", "passive"]
_FRICTION = [r for r in SLIDING_COHESION if r != "undrained"]
_ADHESION = [r for r in SLIDING_COHESION if r != "drained"]
declare_applies_when(METHOD_INFO, {
    "earth_pressure_coefficient": {
        "theory": {"state": _ACTIVE_PASSIVE},
        # GM-257: Coulomb's, and the log-spiral Kp's (GM-232)
        "delta_deg": [{"theory": _COULOMB, "state": _ACTIVE_PASSIVE},
                      {"theory": _LOG_SPIRAL_THEORY, "state": ["passive"]}],
        "alpha_deg": {"theory": _COULOMB, "state": _ACTIVE_PASSIVE},
        # since GM-183 the log-spiral Kp and the at-rest K0 take a slope
        "beta_deg": [{"theory": ["rankine"], "state": ["active"]},
                     {"theory": _COULOMB, "state": _ACTIVE_PASSIVE},
                     {"theory": _LOG_SPIRAL_THEORY, "state": ["passive"]},
                     {"state": ["at_rest"]}],
        "theta_deg": {"theory": _LOG_SPIRAL_THEORY, "state": ["passive"]},
        "OCR": {"state": ["at_rest"]},
        "log_spiral_source": {"theory": _LOG_SPIRAL_THEORY,
                              "state": ["passive"]}},
    "cantilever_wall": {
        # the key acts only through the passive resistance in front of it
        # (cantilever.py: Pp over the key depth when has_shear_key, GM-210)
        "has_shear_key": {"include_passive": [True]},
        "key_depth": {"has_shear_key": [True], "include_passive": [True]},
        # GM-288: the passive resistance, and the soil over the toe
        "gamma_foundation": [{"include_passive": [True]},
                             {"embedment_depth": "set"}],
        # GM-242: the passive coefficient in front of the wall
        "passive_theory": {"include_passive": [True]},
        "log_spiral_source": {"include_passive": [True],
                              "passive_theory": ["log_spiral",
                                                 "caquot_kerisel"]},
        "passive_wall_friction_deg": {"include_passive": [True],
                                      "passive_theory": ["log_spiral",
                                                         "caquot_kerisel"]},
        # GM-212: the rule for the foundation cohesion in sliding counts
        # the friction (drained, friction_plus_adhesion), the adhesion
        # (undrained, friction_plus_adhesion) or both
        "phi_foundation": [{"delta_base": "unset",
                            "sliding_cohesion": _FRICTION},
                           {"include_passive": [True]}],
        "delta_base": {"sliding_cohesion": _FRICTION},
        "sliding_cohesion": [{"c_foundation": "set"},
                             {"base_adhesion": "set"}],
        "c_foundation": {"sliding_cohesion": _ADHESION,
                         "base_adhesion": "unset"},
        "base_adhesion": {"sliding_cohesion": _ADHESION},
        "live_load_resists": {"surcharge": "set"},
        "dead_load_resists": {"dead_load_surcharge": "set"},
        "surcharge_in_bearing": [{"surcharge": "set"},
                                 {"dead_load_surcharge": "set"}],
        # GM-288: at the default delta = beta Coulomb's Ka on the vertical
        # plane through the heel is Rankine's, so the theory moves the
        # result only with a wall friction given
        "pressure_method": {"wall_friction_deg": "set"},
        "wall_friction_deg": {"pressure_method": ["coulomb"]},
        "interface_gap_possible": {"sliding_cohesion": ["undrained"]},
        "bearing_pressure": {"q_allowable": "set"}},
    "mse_wall": {**{
        name: {"reinforcement_Tallowable": "set",
               "reinforcement_name": "unset"}
        for name in ("reinforcement_type",
                     "reinforcement_thickness",
                     "reinforcement_transverse_spacing")}, **_TRAFFIC,
        # GM-256: Rc = b/Sh, or Rc given; neither: 1 for a geosynthetic
        "reinforcement_width": {"reinforcement_horizontal_spacing": "set"},
        "coverage_ratio": {"reinforcement_horizontal_spacing": "unset"},
        "live_load_resists": {"surcharge": "set"},
        "surcharge_in_bearing": {"surcharge": "set"},
        "broken_backslope_height": {"backfill_slope": "set"},
        # GM-273: the depths given place the levels, Sv otherwise
        "reinforcement_spacing": {"level_depths": "unset"},
        "sigma_H_average": {"level_depths": "set"},
        # GM-246: the ribbed steel strip F*; a steel grid given no t/St
        # falls back to it, and the choices then report the Cu it took
        "Cu_backfill": [
            {"reinforcement_name": ["ribbed_steel_strip_50x4"]},
            {"reinforcement_Tallowable": "set", "reinforcement_name": "unset",
             "reinforcement_type": ["metallic_strip", "strip",
                                    "ribbed_strip"]}]},
    "mse_lrfd_external_stability": {
        **_TRAFFIC,
        "broken_backslope_height": {"backfill_slope": "set"},
        # GM-179: the resistance is given or worked out, one or the other
        "phi_bearing": {"gamma_foundation": "set"},
        "bearing_resistance_strength": {"gamma_foundation": "unset"}},
})

# What a left-out input with no default stands for (GM-165): a part the
# module sizes, the foundation and retained soils taken as the backfill,
# the base friction and adhesion it works out, a slope that continues, and
# a bearing check not run.
_NOT_SIZED = "a trial proportion; give it for the wall as designed"
_REINFORCEMENT_LENGTH = (
    "0.7 H, at least 2.5 m",
    "GEC-11's preliminary reinforcement length L = max(0.7H, 8 ft "
    "[2.5 m]); give reinforcement_length for the wall as designed")
_SLOPE_CONTINUES = (
    "a slope that continues",
    "with a backfill slope and no crest height, the slope is taken to "
    "continue behind the wall (GEC-11 Sec. 4.4.4)")
declare_when_absent(METHOD_INFO, {
    "cantilever_wall": {
        "base_width": ("B = 0.6 H, sized by the module", _NOT_SIZED),
        # GM-288: FHWA NHI-06-089 Fig. 10-16(a)'s proportions
        "toe_length": ("H/10, sized by the module", _NOT_SIZED),
        "stem_thickness_base": (
            "the larger of H/10 and the stem thickness at the top, sized by "
            "the module", _NOT_SIZED),
        "base_thickness": ("H/10, sized by the module", _NOT_SIZED),
        "embedment_depth": (
            "the base thickness",
            "the ground in front of the wall at the top of the base: no soil "
            "over the toe, and include_passive's passive resistance over the "
            "base thickness; give embedment_depth for the wall as designed"),
        "wall_friction_deg": (
            "the backfill slope",
            "Coulomb's wall friction on the vertical plane through the back "
            "of the heel, delta = beta (FHWA NHI-06-089 p. 10-20; 0 behind "
            "level backfill)"),
        "phi_foundation": "phi_backfill",
        "gamma_foundation": "gamma_backfill",
        "delta_base": ("2/3 of the foundation soil friction angle",
                       "the sliding check's base friction delta_b = (2/3) "
                       "phi_foundation; give delta_base to set it directly"),
        "base_adhesion": ("worked out from c_foundation by the sliding "
                          "rule",
                          "with sliding_cohesion 'undrained' ca = "
                          "c_foundation as cu (GEC-6 Eq. 5-39), with "
                          "'friction_plus_adhesion' Ca from UFC 3-220-20 "
                          "Table 4-1 at c_foundation (Fig. 4-27); give "
                          "base_adhesion to set it directly"),
        "q_allowable": ("bearing not checked",
                        "with no allowable bearing pressure the bearing "
                        "check is not run (checks_not_run lists it)"),
        "passive_wall_friction_deg": (
            "delta/phi' = 0.66",
            "delta/phi' = 0.66 (UFC 3-220-20 p. 191, 'a common "
            "assumption'); refused with 'caquot_kerisel_1948' or "
            "'dm7_2_eq_4_12', which need it given"),
    },
    "earth_pressure_coefficient": {
        # GM-257: what a left-out wall friction means depends on the
        # theory; each run gives the value it took in defaults_used
        "delta_deg": (
            "by theory: Coulomb delta = 0 (no wall friction); "
            "caquot_kerisel delta = 0.66 phi'",
            "Coulomb's Ka and Kp take the wall friction given, and none "
            "left out; the log-spiral Kp's UFC 3-220-20 sources take "
            "delta/phi' = 0.66 (p. 191: 'a common assumption', the one "
            "value Figs. 4-12 and 4-13 are drawn for), while "
            "'caquot_kerisel_1948' and 'dm7_2_eq_4_12' refuse a call "
            "without it (Caltrans T&S p. 4-29: 'Given delta, beta, and "
            "phi'). The wall friction is the engineer's call: UFC 3-220-20 "
            "Table 4-1 (p. 184) by wall material and soil"),
    },
    "mse_wall": {
        "reinforcement_length": _REINFORCEMENT_LENGTH,
        # GM-273; the bottom level's contributory height, GM-277
        "level_depths": (
            "the levels at reinforcement_spacing (i + 1/2) below the top, "
            "wall_height / reinforcement_spacing of them rounded down, each "
            "taking Sv and sigma_H at its depth, the bottom level's "
            "contributory height down to the base",
            "a trial layout. GEC-11 Vol I p. 4-46: for equally spaced "
            "levels Sv is the spacing and sigma_H at the level is at the "
            "center of the contributory height; p. 4-47 and Fig. 4-14: the "
            "bottom layer's Sv runs from the midpoint above it to the "
            "bottom of the wall, so when wall_height is not a whole number "
            "of spacings the bottom level takes Sv plus the wall below the "
            "last whole spacing, sigma_H at the middle of it. Give "
            "level_depths for the wall as designed"),
        # GM-256: a steel strip or grid with neither is refused
        "coverage_ratio": (
            "Rc 1, a continuous geosynthetic layer",
            "GEC-11 Vol I p. 3-13: 'Rc = 1 in the case of continuous "
            "reinforcement'; give reinforcement_horizontal_spacing (with "
            "reinforcement_width) or coverage_ratio for strips"),
        "F_star_tested": (
            "the empirical F*",
            "GEC-11 Vol I Eqs. 3-4 to 3-8 (pp. 3-20, 3-21), for use 'in "
            "the absence of product and backfill specific test data' "
            "(Table 3-6 note (i), p. 3-17)"),
        "alpha_tested": (
            "Table 3-6's alpha for the reinforcement type",
            "GEC-11 Vol I Table 3-6 (p. 3-17): 1.0 for steel strips and "
            "grids, 0.8 for geogrids, 0.6 for geotextiles (p. 3-16: 'In the "
            "absence of test data'); a 'geosynthetic' of no stated kind is "
            "refused"),
        "phi_retained": "phi_backfill",
        "gamma_retained": "gamma_backfill",
        "broken_backslope_height": _SLOPE_CONTINUES,
        "phi_foundation": "phi_backfill",
        "q_allowable": ("bearing not checked",
                        "with no allowable bearing pressure the external "
                        "bearing check is not run (checks_not_run lists "
                        "it)"),
    },
    "mse_lrfd_external_stability": {
        "reinforcement_length": _REINFORCEMENT_LENGTH,
        "broken_backslope_height": _SLOPE_CONTINUES,
        "phi_foundation": "phi_backfill",
        "phi_retained": "phi_backfill",
        "gamma_retained": "gamma_backfill",
        "live_load": "surcharge",
        "bearing_resistance_strength": (
            "bearing not checked at the strength limit",
            "with no factored bearing resistance qR and no gamma_foundation "
            "to work it out the Strength I bearing CDR is not computed "
            "(checks_not_run lists it)"),
        "bearing_resistance_service": (
            "bearing not checked at the service limit",
            "with no service bearing resistance the Service I bearing CDR "
            "is not computed (checks_not_run lists it)"),
    },
})
