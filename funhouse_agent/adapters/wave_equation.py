"""Wave equation adapter — Smith 1-D (bearing graph, drivability)."""

from funhouse_agent.adapters import (declare_applies_when,
                                     declare_when_absent,
                                     reject_unknown_params, require_params)
from wave_equation import (
    Hammer, get_hammer, list_hammers, Cushion, make_cushion_from_properties,
    discretize_pile, SoilSetup, simulate_blow, generate_bearing_graph, drivability_study,
)
from wave_equation.drivability import REFUSAL_BLOWS_PER_M
from wave_equation.driving_systems import (
    PRINTED_DRIVING_SYSTEMS, PRINTED_OPTIONS, driving_system_for_hammer,
    get_driving_system, require_helmet_weight)
from wave_equation.hammer import diesel_warning
from wave_equation.soil_model import quake_weighting_choice

#: Steel, 492 lb/ft3 (GEC-12 Vol 2 p. 462), the module's default.
_STEEL_UNIT_WEIGHT = 77.3
#: Toe Smith damping, 0.15 s/ft (GEC-12 Vol 2 p. 211).
_DAMPING_TOE = 0.49

# The embedded length and the shaft resistance's distribution over it
# (GM-301), single blow and bearing graph.
_EMBEDMENT_PARAMS = ("embedded_length", "shaft_distribution")

# Hammer + cushion params shared by every method's valid set.
_HAMMER_PARAMS = ("hammer_name", "hammer_custom_name", "ram_weight", "stroke",
                  "efficiency", "hammer_type")
_CUSHION_PARAMS = ("cushion_stiffness", "cushion_area", "cushion_thickness",
                   "cushion_E", "cushion_cor", "driving_system")


def _build_hammer(params):
    if "hammer_name" in params:
        return get_hammer(params["hammer_name"])
    require_params(params, ["ram_weight", "stroke"],
                   method="custom hammer (no hammer_name)",
                   valid=_HAMMER_PARAMS)
    return Hammer(name=params.get("hammer_custom_name", "Custom"), ram_weight=params["ram_weight"],
                   stroke=params["stroke"], efficiency=params.get("efficiency", 0.67),
                   hammer_type=params.get("hammer_type", "single_acting"))


def _driving_system(params, hammer):
    """The printed driving system in force and how it came: the one named
    by ``driving_system`` ("given"), else the named hammer's where a held
    reference prints one ("default"; GEC-12 Vol 2 Table 12-2, p. 183),
    else (None, None)."""
    if params.get("driving_system"):
        return get_driving_system(params["driving_system"]), "given"
    if "hammer_name" in params:
        system = driving_system_for_hammer(hammer.name)
        if system is not None:
            return system, "default"
    return None, None


def _build_driving_system(params, hammer):
    """(cushion, helmet weight in kN, choices) of a call (GM-299).

    A cushion given by stiffness or by properties needs its COR, and the
    helmet weight is then needed too. With no cushion given, the printed
    driving system (:func:`_driving_system`) supplies the cushion, its COR
    and the helmet weight, each unless given. No held reference prints a
    cushion or helmet for any other hammer, so a call with neither is
    refused, the printed options in the error (before GM-299: a 500,000
    kN/m cushion, COR 0.80 and a 5 kN helmet, none printed). The result's
    choices state the cushion and helmet the run took and how."""
    from geotech_common.choices import DEFAULT, GIVEN, how_given
    own = "cushion_stiffness" in params or "cushion_area" in params
    system, came = _driving_system(params, hammer)
    if own and params.get("driving_system"):
        raise ValueError(
            "Give either a cushion (cushion_stiffness, or cushion_area + "
            "cushion_thickness + cushion_E) or driving_system, not both.")
    if own:
        if "cushion_stiffness" not in params:
            require_params(params, ["cushion_area", "cushion_thickness",
                                    "cushion_E"],
                           method="cushion from properties",
                           valid=_CUSHION_PARAMS)
        if "cushion_cor" not in params:
            raise ValueError(
                "cushion_cor is required with a cushion given: no held "
                "reference prints one coefficient of restitution for every "
                "cushion. " + PRINTED_OPTIONS)
        if "cushion_stiffness" in params:
            cushion = Cushion(stiffness=params["cushion_stiffness"],
                              cor=params["cushion_cor"])
        else:
            cushion = make_cushion_from_properties(
                area=params["cushion_area"],
                thickness=params["cushion_thickness"],
                elastic_modulus=params["cushion_E"],
                cor=params["cushion_cor"])
        helmet = require_helmet_weight(params.get("helmet_weight"))
        how = how_given(cushion.stiffness, None, name=(
            "cushion_stiffness" if "cushion_stiffness" in params
            else "cushion_area"))
        return cushion, helmet, [_driving_system_choice(
            cushion, helmet, how, hammer, None)]
    if system is None:
        raise ValueError(
            "A cushion is required: give cushion_stiffness (kN/m) or "
            "cushion_area + cushion_thickness + cushion_E, with cushion_cor "
            "and helmet_weight, or driving_system. No held reference prints "
            f"a cushion or helmet for the {hammer.name}. " + PRINTED_OPTIONS)
    # The printed system, each of its values unless the call gave it; the
    # cushion as its stiffness and COR, as a cushion given by them is
    cor = params.get("cushion_cor", system.cor)
    cushion = Cushion(stiffness=system.stiffness, cor=cor)
    helmet = require_helmet_weight(params.get("helmet_weight",
                                              system.helmet_weight))
    if came == "default":
        how = DEFAULT
        for name, value in (("cushion_stiffness", system.stiffness),
                            ("cushion_cor", cor),
                            ("helmet_weight", helmet)):
            if name not in params:
                how_given(value, None, name=name, took=value)
    else:
        how = how_given(system.hammer, None, name="driving_system")
    return cushion, helmet, [_driving_system_choice(
        cushion, helmet, how, hammer, system if how == GIVEN else None)]


def _driving_system_choice(cushion, helmet, how, hammer, named):
    """The choices entry of the cushion and helmet a run took (GM-299):
    the same words however they came, so a run given its own defaults
    reads the same; ``named``, a driving system the call named, adds what
    it was printed for when that is another hammer."""
    from geotech_common.choices import choice
    systems = "; ".join(s.describe() for s in PRINTED_DRIVING_SYSTEMS.values())
    basis = ("The hammer cushion (the model's one cushion spring) and the "
             "helmet lumped with the pile head. Ideally the contractor's "
             "submittal for the actual hammer (GEC-12 Vol 2 p. 226). Left "
             "out, the named hammer's driving system as a held reference "
             f"prints it, which only these have: {systems}. Any other "
             "hammer needs them given.")
    if named is not None and named.hammer != hammer.name:
        basis += (f" driving_system '{named.hammer}' is printed for the "
                  f"{named.hammer}, here run with the {hammer.name}.")
    return choice(
        "hammer cushion and helmet",
        f"k = {cushion.stiffness:,.0f} kN/m, COR {cushion.cor:g}, "
        f"helmet {helmet:.2f} kN",
        how, basis=basis,
        alternatives=[f"driving_system '{k}'" for k in PRINTED_DRIVING_SYSTEMS]
        + ["cushion_stiffness or cushion properties with cushion_cor and "
           "helmet_weight"])


def run_bearing_graph(params, *, method="bearing_graph"):
    """(result, inputs in force) of ``bearing_graph``: the tool's own code
    path, also run by ``calc_package.wave_equation_package`` (GM-141). The
    inputs in force are the hammer, cushion and pile models and each
    soil and range input given or defaulted, by the tool's names.
    ``method`` names the caller in error messages."""
    reject_unknown_params(
        params,
        _HAMMER_PARAMS + _CUSHION_PARAMS + (
            "pile_length", "pile_area", "pile_E", "segment_length",
            "pile_unit_weight", "skin_fraction", "quake_side", "quake_toe",
            "damping_side", "damping_toe", "R_min", "R_max", "R_step",
            "helmet_weight", "damping_model") + _EMBEDMENT_PARAMS,
        method=method)
    require_params(params, ["pile_length", "pile_area"], method=method)
    hammer = _build_hammer(params)
    cushion, helmet, drive_choices = _build_driving_system(params, hammer)
    pile_E = params.get("pile_E", 200e6)
    segment_length = params.get("segment_length", 1.0)
    pile_unit_weight = params.get("pile_unit_weight", _STEEL_UNIT_WEIGHT)
    pile = discretize_pile(length=params["pile_length"], area=params["pile_area"],
                            elastic_modulus=pile_E,
                            segment_length=segment_length,
                            unit_weight_material=pile_unit_weight)
    used = {
        "hammer": hammer, "cushion": cushion, "pile": pile,
        "pile_E": pile_E, "segment_length": segment_length,
        "pile_unit_weight": pile_unit_weight,
        "skin_fraction": params.get("skin_fraction", 0.5),
        "quake_side": params.get("quake_side", 0.0025),
        "quake_toe": params.get("quake_toe", 0.0025),
        "damping_side": params.get("damping_side", 0.16),
        "damping_toe": params.get("damping_toe", _DAMPING_TOE),
        "R_min": params.get("R_min", 200.0),
        "R_max": params.get("R_max", 2000.0),
        "R_step": params.get("R_step", 200.0),
        "helmet_weight": helmet,
        "damping_model": params.get("damping_model", "smith"),
        # GM-301: the shaft resistance over the embedded length
        "embedded_length": params.get("embedded_length"),
        "shaft_distribution": params.get("shaft_distribution", "uniform"),
    }
    bg = generate_bearing_graph(
        hammer, cushion, pile, skin_fraction=used["skin_fraction"],
        quake_side=used["quake_side"], quake_toe=used["quake_toe"],
        damping_side=used["damping_side"], damping_toe=used["damping_toe"],
        R_min=used["R_min"], R_max=used["R_max"], R_step=used["R_step"],
        helmet_weight=used["helmet_weight"],
        damping_model=used["damping_model"],
        shaft_distribution=used["shaft_distribution"],
        embedded_length=used["embedded_length"],
    )
    bg.choices = list(bg.choices) + drive_choices
    return bg, used


def _run_bearing_graph(params):
    return run_bearing_graph(params)[0].to_dict()


def _run_drivability(params):
    reject_unknown_params(
        params,
        _HAMMER_PARAMS + _CUSHION_PARAMS + (
            "pile_area", "pile_E", "pile_unit_weight", "depths", "R_at_depth",
            "skin_fractions", "segment_length", "helmet_weight",
            "refusal_blow_count", "damping_model", "quake_side",
            "quake_toe", "damping_side", "damping_toe", "pile_length",
            "shaft_distribution"),
        method="drivability")
    require_params(params, ["pile_area", "depths", "R_at_depth"],
                   method="drivability")
    hammer = _build_hammer(params)
    cushion, helmet, drive_choices = _build_driving_system(params, hammer)
    result = drivability_study(
        hammer, cushion, pile_area=params["pile_area"], pile_E=params.get("pile_E", 200e6),
        pile_unit_weight=params.get("pile_unit_weight", _STEEL_UNIT_WEIGHT), depths=params["depths"],
        R_at_depth=params["R_at_depth"], skin_fractions=params.get("skin_fractions"),
        segment_length=params.get("segment_length", 1.0),
        helmet_weight=helmet,
        refusal_blow_count=params.get("refusal_blow_count",
                                      REFUSAL_BLOWS_PER_M),
        damping_model=params.get("damping_model", "smith"),
        # GM-300: the quakes and damping factors, as the other tools take
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", _DAMPING_TOE),
        # GM-301: the pile in the leads at every depth, and the shaft
        # resistance's distribution over the embedded length
        pile_length=params.get("pile_length"),
        shaft_distribution=params.get("shaft_distribution", "uniform"),
    )
    result.choices = list(result.choices) + drive_choices
    return result.to_dict()


def _run_list_hammers(params):
    hammers = {}
    for name in list_hammers():
        h = get_hammer(name)
        hammers[name] = {"ram_weight_kN": h.ram_weight, "stroke_m": h.stroke,
                          "rated_energy_kNm": round(h.energy, 1), "efficiency": h.efficiency}
    return {"hammers": hammers}


def _run_single_blow(params):
    reject_unknown_params(
        params,
        _HAMMER_PARAMS + _CUSHION_PARAMS + (
            "pile_length", "pile_area", "pile_E", "segment_length",
            "pile_unit_weight", "R_total", "R_ultimate", "skin_fraction",
            "quake_side", "quake_toe", "damping_side", "damping_toe",
            "damping_model", "helmet_weight") + _EMBEDMENT_PARAMS,
        method="single_blow")
    require_params(params, ["pile_length", "pile_area"], method="single_blow")
    hammer = _build_hammer(params)
    cushion, helmet, drive_choices = _build_driving_system(params, hammer)
    pile = discretize_pile(
        length=params["pile_length"], area=params["pile_area"],
        elastic_modulus=params.get("pile_E", 200e6),
        segment_length=params.get("segment_length", 1.0),
        unit_weight_material=params.get("pile_unit_weight", _STEEL_UNIT_WEIGHT),
    )
    # Module keyword is R_ultimate; METHOD_INFO advertises R_total — accept both.
    R = params.get("R_total", params.get("R_ultimate"))
    if R is None:
        raise ValueError("single_blow: missing required parameter 'R_total' "
                         "(total ultimate soil resistance, kN).")
    soil = SoilSetup(
        R_ultimate=R,
        skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", _DAMPING_TOE),
        damping_model=params.get("damping_model", "smith"),
        shaft_distribution=params.get("shaft_distribution", "uniform"),
        embedded_length=params.get("embedded_length"),
    )
    result = simulate_blow(
        hammer, cushion, pile, soil,
        helmet_weight=helmet,
    )
    perm_set = result.permanent_set
    blow_count = round(1.0 / perm_set, 1) if perm_set > 0 else float("inf")
    out = {
        "permanent_set_mm": round(perm_set * 1000, 3),
        "blow_count_per_m": blow_count,
        "max_pile_stress_MPa": round(result.max_compression_stress / 1000, 2),
        "max_tension_stress_MPa": round(result.max_tension_stress / 1000, 2),
        "max_pile_force_kN": round(result.max_pile_force, 1),
    }
    # The quake weights no page prints, with the other weightings' blow
    # counts, and a diesel's warning (GM-294 review).
    quakes = quake_weighting_choice(
        soil.quake_side, soil.quake_toe,
        [(soil.skin_fraction, result.max_toe_displacement)])
    choices = ([quakes] if quakes else []) + drive_choices
    if choices:
        out["choices"] = choices
    warning = diesel_warning(hammer)
    if warning:
        out["warnings"] = [warning]
    return out


METHOD_REGISTRY = {
    "single_blow": _run_single_blow,
    "bearing_graph": _run_bearing_graph,
    "drivability": _run_drivability,
    "list_available_hammers": _run_list_hammers,
}

# Shared hammer/cushion doc fragment (custom hammer in place of hammer_name;
# cushion by stiffness or by properties; default cushion if omitted).
_HAMMER_CUSHION_DOC = {
    "ram_weight": {"type": "float", "required": False, "description": "Custom hammer ram weight (kN) — used with stroke when hammer_name is omitted."},
    "stroke": {"type": "float", "required": False, "description": "Custom hammer stroke (m)."},
    "efficiency": {"type": "float", "required": False, "default": 0.67, "description": "Custom hammer efficiency: the share of the potential ram energy (ram weight x stroke) that reaches the ram as kinetic energy at impact (GEC-12 Vol 2 (FHWA-NHI-16-009) Sec. 12.6.2, p. 205). GEC-12 Vol 2 prints 0.67 for an air hammer and 0.80 for an open end diesel (Example 6, p. 185), and say 0.95 for a hydraulic hammer whose ram kinetic energy is measured (p. 205). Default 0.67."},
    "hammer_custom_name": {"type": "str", "required": False, "default": "Custom", "description": "Custom hammer label."},
    "hammer_type": {"type": "str", "required": False, "default": "single_acting", "allowed_values": ["single_acting", "diesel", "hydraulic"], "description": "Custom hammer type: a label in the result and calc package only. Every type is modelled as a rigid ram striking at v = sqrt(2 g stroke efficiency); a diesel hammer's combustion pressure is not modelled, and its result carries a warning (GEC-12 Vol 2 p. 206: its stroke rises with the resistance, and too high a stroke gives non-conservative nominal resistances). Give the type's losses through efficiency (GEC-12 Vol 2 (FHWA-NHI-16-009) Sec. 12.6.2, p. 205: relatively high, say 0.95, for a hydraulic hammer whose ram kinetic energy is measured and displayed)."},
    "segment_length": {"type": "float", "required": False, "default": 1.0, "description": "Pile segment length (m): about 1 m (GEC-12 Vol 2 Sec. 12.3, p. 155; GRLWEAP's 3.3 ft, p. 213). Shorter segments for stiff, uncushioned driving systems (p. 210)."},
    "cushion_stiffness": {"type": "float", "required": False, "description": "Hammer cushion stiffness (kN/m), with cushion_cor and helmet_weight. Or give cushion_area + cushion_thickness + cushion_E: stiffness = E A / t (GEC-12 Vol 2 p. 155); two materials in series 1/k = 1/k1 + 1/k2 (p. 209). The model has one cushion spring, between the ram and the helmet (whose weight is lumped with the pile head); GEC-12 Vol 2 Fig. 12-2 (p. 157) has a hammer cushion and a pile cushion with the helmet between. Left out with no cushion properties: the hammer's printed driving system where a held reference prints one (hammer_name 'Vulcan 014', GEC-12 Vol 2 Table 12-2, p. 183), or the one named by driving_system; for any other hammer the call is refused and the error lists the printed options (GEC-12 prints no cushion by hammer otherwise; ideally the contractor's submittal, Vol II p. 226)."},
    "cushion_area": {"type": "float", "required": False, "description": "Cushion area (m2) — with cushion_thickness and cushion_E."},
    "cushion_thickness": {"type": "float", "required": False, "description": "Cushion thickness (m)."},
    "cushion_E": {"type": "float", "required": False, "description": "Cushion elastic modulus (kPa), with the nominal (uncompressed) thickness. Plywood pile cushions 30 ksi (207,000 kPa) new and 75 ksi (517,000 kPa) used; oak boards 60 and 90 ksi (GEC-12 Vol 2 p. 209)."},
    "cushion_cor": {"type": "float", "required": False, "description": "Hammer cushion coefficient of restitution, required with a cushion given by stiffness or properties (no held reference prints one COR for every cushion). GEC-12 Vol 2: the hammer cushions of its examples 0.90 to 0.92 (pp. 174, 183); wood 0.50, a badly worn softwood cushion 0.25 (p. 156); a refined analysis adjusts it between 0.10 and 1.0 (p. 216). With a printed driving system and no cushion given: that system's COR unless given (Vulcan 014 0.91, ICE 42S 0.92, p. 183)."},
    "pile_E": {"type": "float", "required": False, "default": 200e6, "description": "Pile elastic modulus (kPa). Default steel, 29,000 ksi (GEC-12 Vol 1 p. 317). Concrete: on the order of 4,000 ksi, varying with its strength (Vol 1 p. 317); its dynamic modulus is usually higher than the static (Vol 2 p. 213)."},
    "pile_unit_weight": {"type": "float", "required": False, "default": 77.3, "description": "Pile material unit weight (kN/m3). Default steel, 492 lb/ft3 (GEC-12 Vol 2 p. 462)."},
    "helmet_weight": {"type": "float", "required": False, "description": "Helmet weight (kN): helmet, hammer cushion, striker plate and inserts (GEC-12 Vol 2 p. 226), lumped with the pile head; 0 with no helmet mass. Required with a cushion given. With a printed driving system and no cushion given: that system's helmet unless given (Vulcan 014 1.67 kips, 7.43 kN; ICE 42S 2.05 kips, 9.12 kN; p. 183). GEC-12's examples give 1.0 to 4.07 kips (4.4 to 18.1 kN; Vol II pp. 163, 167, 174, 179, 183)."},
    "driving_system": {"type": "str", "required": False, "allowed_values": sorted(PRINTED_DRIVING_SYSTEMS), "any_case": "separators", "description": "A driving system (hammer cushion and helmet) as a held reference prints it, by its hammer: GEC-12 Vol 2 Example 6, Table 12-2 (p. 183), on a steel pipe pile. 'Vulcan 014': Nycast, 208 ksi, 234 in2, 6.0 in (1.42 million kN/m), COR 0.91, helmet 1.67 kips; 'ICE 42S': Blue Nylon, 175 ksi, 398 in2, 2.0 in (6.10 million kN/m), COR 0.92, helmet 2.05 kips. cushion_cor and helmet_weight, if given, replace the system's. Not with cushion_stiffness or cushion properties. Left out: the named hammer's own printed system, which only hammer_name 'Vulcan 014' has; any other hammer needs a cushion given. A run with another hammer says so in its choices."},
}

_DAMPING_MODEL_DOC = (
    "Smith damping form (UFC 3-220-20 (2025) Sec. 6-9.2, pp. 487-488): "
    "'smith' (default), damping proportional to the mobilized static "
    "resistance, J x R_static x v, set 'in most instances' (GEC-12 Vol 2 "
    "p. 242); 'smith_viscous', proportional to the fully mobilized static "
    "resistance, J x R_ult x v, which GEC-12 Vol 2 (p. 242) selects for a "
    "residual stress analysis or a vibratory hammer, and sometimes for a "
    "refined analysis's match.")

# Smith soil constants (single blow and bearing graph).
_SOIL_DOC = {
    "quake_side": {"type": "float", "required": False, "default": 0.0025, "description": "Shaft quake (m): 0.10 in (0.0025 m), usually satisfactory (GEC-12 Vol 2 Sec. 12.6.7, p. 211)."},
    "quake_toe": {"type": "float", "required": False, "default": 0.0025, "description": "Toe quake (m). Default 0.10 in (0.0025 m), GEC-12 Vol 2's toe quake for a non-displacement pile, e.g. an H-pile (p. 171). A displacement pile (closed end pipe, concrete): pile diameter or width D over 120 to D over 60 (p. 211), D/60 in medium or dense sand and D/120 in very dense soil (Examples 1, 3, 6, pp. 162, 170, 182); GEC-12 Vol 3 (p. 40) also gives H-piles D/60 and D/120. Hard rock: 0.04 in (0.001 m) (p. 212). Saturated or sensitive soils can reach well over D/60 (p. 211)."},
    "damping_side": {"type": "float", "required": False, "default": 0.16, "description": "Shaft Smith damping (s/m), GEC-12 Vol 2 Sec. 12.6.7 (p. 211): 0.05 s/ft (0.16 s/m, the default) non-cohesive soils, 0.10 s/ft (0.33) silty sands, clayey sands and sandy silts, 0.15 s/ft (0.49) cohesive silts and sandy clays, 0.20 s/ft (0.66) cohesive soils. For restrike, perhaps twice as high (p. 211)."},
    "damping_toe": {"type": "float", "required": False, "default": 0.49, "description": "Toe Smith damping (s/m): about 0.15 s/ft (0.49 s/m) for most soils (GEC-12 Vol 2 p. 211); lower on hard rock, e.g. 0.05 s/ft (0.16 s/m) (p. 212)."},
}

# The embedded length and the shaft resistance's distribution (GM-301).
_SHAFT_DISTRIBUTION_DOC = {"type": "str", "required": False, "default": "uniform", "allowed_values": ["uniform", "triangular"], "description": "How the shaft resistance (skin_fraction of R_ult) is spread over the embedded length, as GEC-12 Vol 2's examples print it: 'uniform' (default), the same per metre (Example 5, p. 179: '92% uniform shaft resistance'); 'triangular', rising linearly from none at grade to its largest at the pile toe (Example 1, p. 162, Fig. 12-3; Example 6, p. 182: 'distributed triangularly along the embedded portion of the pile shaft'). A distribution from a static analysis layer by layer (Example 4, Fig. 12-9) is not offered. GEC-12 Vol 2 p. 212: the stresses, tension above all, are sensitive to the distribution."}
_EMBEDMENT_DOC = {
    "embedded_length": {"type": "float", "required": False, "description": "The pile's embedded length, its penetration below grade (m), at most pile_length: the shaft resistance acts only below grade, the pile above it standing free (GEC-12 Vol 2 p. 224: 'Penetration refers to the analyzed pile toe penetration below grade'; Example 1, p. 162, a 65 ft pile driven 62 ft; Example 6, p. 182, 66 ft with 'an embedded length of 52.5 feet'). Left out: pile_length, the pile embedded over its full length."},
    "shaft_distribution": _SHAFT_DISTRIBUTION_DOC,
}

METHOD_INFO = {
    "single_blow": {
        "category": "Wave Equation",
        "brief": "Simulate a single hammer blow — set, stress, blow count.",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name (use list_available_hammers)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile area (m2)."},
            "R_total": {"type": "float", "required": True, "description": "Total soil resistance (kN). Alias: R_ultimate."},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5, "description": "Skin friction fraction."},
            "damping_model": {"type": "str", "required": False, "default": "smith", "allowed_values": ["smith", "smith_viscous"], "description": _DAMPING_MODEL_DOC},
            **_SOIL_DOC,
            **_EMBEDMENT_DOC,
            **_HAMMER_CUSHION_DOC,
        },
        "parameter_aliases": {"R_ultimate": "R_total"},
        "returns": {
            "permanent_set_mm": "Pile penetration per blow (mm): maximum pile toe displacement less an average of the shaft and toe quakes (GEC-12 Vol 2 p. 158), weighted by the shares of R_ult on the shaft and at the toe: the module's assumption, since the page names no weights (choices gives the blow counts of a plain average and of the toe quake alone).",
            "blow_count_per_m": "Blows per meter.",
            "max_pile_stress_MPa": "Max compressive stress in the pile (MPa), the pile top's without the helmet's inertia.",
            "max_tension_stress_MPa": "Max tensile stress (MPa).",
            "choices": "The quake weights (when the quakes differ).",
            "warnings": "A diesel hammer's (it is run at its rated energy).",
        },
    },
    "bearing_graph": {
        "category": "Wave Equation",
        "brief": "Generate bearing capacity vs blow count graph.",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name (use list_available_hammers to see options)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-sectional area (m2)."},
            "R_min": {"type": "float", "required": False, "default": 200.0, "description": "Min resistance (kN)."},
            "R_max": {"type": "float", "required": False, "default": 2000.0, "description": "Max resistance (kN)."},
            "R_step": {"type": "float", "required": False, "default": 200.0, "description": "Resistance increment (kN)."},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5, "description": "Skin friction fraction."},
            "damping_model": {"type": "str", "required": False, "default": "smith", "allowed_values": ["smith", "smith_viscous"], "description": _DAMPING_MODEL_DOC},
            **_SOIL_DOC,
            **_EMBEDMENT_DOC,
            **_HAMMER_CUSHION_DOC,
        },
        "returns": {"R_values_kN": "Ultimate resistances analysed.", "blow_counts_per_m": "Blow count per R_ult (1/set; 1,000,000 = refusal).", "permanent_sets_m": "Set per blow: maximum pile toe displacement less an average of the shaft and toe quakes (GEC-12 Vol 2 p. 158), weighted by the shares of R_ult on the shaft and at the toe: the module's assumption, since the page names no weights (choices gives the blow counts of a plain average and of the toe quake alone).", "max_comp_stresses_kPa": "Maximum compression stress in the pile per R_ult.", "max_tens_stresses_kPa": "Maximum tension stress per R_ult.", "choices": "The quake weights (when the quakes differ).", "warnings": "A diesel hammer's (it is run at its rated energy)."},
    },
    "drivability": {
        "category": "Wave Equation",
        "brief": "Drivability study: can the hammer drive the pile to target depth?",
        "parameters": {
            "hammer_name": {"type": "str", "required": False, "description": "Hammer name."},
            "pile_area": {"type": "float", "required": True, "description": "Pile area (m2)."},
            "depths": {"type": "array", "required": True, "description": "Depth stations (m)."},
            "R_at_depth": {"type": "array", "required": True, "description": "SRD at each depth (kN)."},
            "skin_fractions": {"type": "array", "required": False, "description": "Skin friction fraction per depth station."},
            "refusal_blow_count": {"type": "float", "required": False, "default": 393.7, "description": "Blow count (blows/m) above which the pile has refused. Default 393.7 (120 blows/ft): GEC-12 Vol 2 Sec. 12.4 (p. 161) chooses hammer and cushion so the maximum expected penetration resistance is under 120 blows/ft, and the drivability study of Vol 3 (p. 43) calls it practical refusal. Piles through soft soils seated on hard rock: 787.4 (240 blows/ft, p. 161), the refusal of Vol 2's Examples 3, 4 and 7."},
            "damping_model": {"type": "str", "required": False, "default": "smith", "allowed_values": ["smith", "smith_viscous"], "description": _DAMPING_MODEL_DOC},
            "pile_length": {"type": "float", "required": False, "description": "The pile's total length in the leads (m), the same at every depth: at each depth the pile is this long, embedded to the depth, the shaft resistance only below grade (GEC-12 Vol 2 p. 224: 'Length is the total pile length in the leads'; 'Penetration refers to ... final penetration for Drivability Analyses'; p. 236, the depths to be analyzed). No depth may be deeper (p. 227). Left out: the deepest depth analysed, the shortest pile that reaches it (p. 227), the same pile at every depth; the result's choices say so."},
            "shaft_distribution": _SHAFT_DISTRIBUTION_DOC,
            **_SOIL_DOC,
            **_HAMMER_CUSHION_DOC,
        },
        "returns": {"can_drive": "Whether pile can be driven.", "refusal_depth_m": "First depth whose blow count passed the refusal blow count (0: none).", "refusal_blow_count_per_m": "The refusal blow count used.", "points": "Per-depth drivability data.", "pile_length_m": "The pile's length at every depth: pile_length, else the deepest depth.", "shaft_distribution": "With 'triangular': the shaft distribution used.", "choices": "The quake weights (when the quakes differ); the pile length analysed (the deepest depth when pile_length is left out).", "warnings": "A diesel hammer's (it is run at its rated energy)."},
    },
    "list_available_hammers": {
        "category": "Wave Equation",
        "brief": "List all available pile driving hammers.",
        "parameters": {},
        "returns": {"hammers": "Dict of hammer properties."},
    },
}

# When each default applies (GM-119): a custom hammer's efficiency and type
# only when no hammer_name picks a hammer from the list.
declare_applies_when(METHOD_INFO, {
    m: {name: {"hammer_name": "unset"}
        for name in ("efficiency", "hammer_type")}
    for m in ("bearing_graph", "drivability", "single_blow")})

# What a left-out embedded length or drivability pile length means (GM-301).
declare_when_absent(METHOD_INFO, {
    "single_blow": {"embedded_length": "pile_length"},
    "bearing_graph": {"embedded_length": "pile_length"},
    "drivability": {"pile_length": (
        "the deepest depth analysed",
        "the same pile at every depth, embedded to each depth: GEC-12 Vol 2 "
        "p. 224 analyses the total pile length in the leads, and p. 227 "
        "allows no penetration deeper than the pile, so the deepest depth "
        "is the shortest pile that reaches it (give pile_length for a "
        "longer pile)")},
})

# A left-out cushion, COR or helmet: the printed driving system of the
# named hammer (GM-299; GEC-12 Vol 2 Table 12-2, p. 183), else refused.
_PRINTED_SYSTEM_CASE = (
    "the named hammer's printed driving system (hammer_name 'Vulcan 014'); "
    "any other hammer is refused",
    "GEC-12 Vol 2 Example 6, Table 12-2 (p. 183), the only driving system "
    "a held reference prints for a hammer the tool lists")
declare_when_absent(METHOD_INFO, {
    m: {name: _PRINTED_SYSTEM_CASE
        for name in ("cushion_stiffness", "cushion_cor", "helmet_weight")}
    for m in ("single_blow", "bearing_graph", "drivability")})
declare_applies_when(METHOD_INFO, {
    m: {"cushion_stiffness": {"cushion_area": "unset",
                              "driving_system": "unset"},
        "cushion_cor": {"cushion_stiffness": "unset", "cushion_area": "unset",
                        "driving_system": "unset"},
        "helmet_weight": {"cushion_stiffness": "unset",
                          "cushion_area": "unset",
                          "driving_system": "unset"}}
    for m in ("single_blow", "bearing_graph", "drivability")})
