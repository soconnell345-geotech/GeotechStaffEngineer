"""Design traffic (18-kip ESAL) accumulation per the AASHTO 1993 Guide.

Builds the design-lane cumulative W18 from either an axle-load spectrum
(per-axle load equivalency factors, LEFs) or per-vehicle truck factors,
applying traffic growth, directional distribution DD, and lane distribution
DL (Guide Part II Section 2.1.2 and Appendix D).

LEF source: prefers the full Appendix D table digitization
(``geotech_references.aashto_1993.lef`` -- all 18 printed tables D.1-D.18,
interpolated over load and SN/D; pt exactly 2.0/2.5/3.0;
single/tandem/triple); falls back to the SN=5 / D=9-in, pt=2.5 subset
curves in ``geotech_references.aashto_1993.tables`` when the full module
is not available.

UNITS: US customary (kips, 18-kip ESALs).
"""

import math

from geotech_references.aashto_1993 import tables as _tb

from .common import add_ref, midpoint_range
from .results import DesignTrafficResult

try:  # full Appendix D LEF table digitization (refs follow-on module)
    from geotech_references.aashto_1993 import lef as _lef_mod
except ImportError:  # pragma: no cover - depends on refs version
    _lef_mod = None


def growth_factor(growth_rate_pct, design_period_yr) -> dict:
    """Total traffic growth factor GF over the design period.

        GF = [(1+g)^n - 1] / g      (g > 0; GF = n for g = 0)

    Annual traffic in year 1 multiplied by GF gives the cumulative total
    over n years of compound growth (AASHTO 1993 Guide, Appendix D
    traffic-analysis procedure; e.g. 4%/yr over 20 yr -> GF = 29.78).

    Parameters
    ----------
    growth_rate_pct : float
        Annual traffic growth rate, percent (>= 0).
    design_period_yr : float
        Performance (analysis) period, years, > 0.

    Returns
    -------
    dict
        {'growth_rate_pct', 'design_period_yr', 'growth_factor',
         'equation', 'reference'}.
    """
    if design_period_yr <= 0:
        raise ValueError(f"design_period_yr must be > 0, got {design_period_yr}")
    if growth_rate_pct < 0:
        raise ValueError(f"growth_rate_pct must be >= 0, got {growth_rate_pct}")
    g = growth_rate_pct / 100.0
    if g == 0:
        gf = float(design_period_yr)
    else:
        gf = ((1.0 + g) ** design_period_yr - 1.0) / g
    return {
        "growth_rate_pct": growth_rate_pct,
        "design_period_yr": design_period_yr,
        "growth_factor": round(gf, 4),
        "equation": "GF = [(1+g)^n - 1]/g  (GF = n when g = 0)",
        "reference": ("AASHTO 1993 Guide, Appendix D traffic analysis "
                      "(growth of a uniform annual volume)"),
    }


def _single_lef(pavement_type, axle_config, load_kips, sn, d_in, pt,
                references):
    """One axle-group LEF; returns (lef, basis_string)."""
    if _lef_mod is not None:
        kw = {"sn": sn} if pavement_type == "flexible" else {"d_in": d_in}
        r = _lef_mod.load_equivalency_factor(
            pavement_type=pavement_type, axle_config=axle_config,
            axle_load_kips=load_kips, pt=pt, **kw,
        )
        add_ref(references, r.get("reference"))
        return r["lef"], "appendix_d_tables"
    # Fallback: digitized table curves (SN=5 / D=9 in, pt=2.5 only).
    table_fns = {
        ("flexible", "single"): lambda: _tb.esal_flexible_single_axle(
            load_kips, sn=sn, pt=pt),
        ("flexible", "tandem"): lambda: _tb.esal_flexible_tandem_axle(
            load_kips, sn=sn, pt=pt),
        ("rigid", "single"): lambda: _tb.esal_rigid_single_axle(
            load_kips, d_in=d_in, pt=pt),
        ("rigid", "tandem"): lambda: _tb.esal_rigid_tandem_axle(
            load_kips, d_in=d_in, pt=pt),
    }
    key = (pavement_type, axle_config)
    if key not in table_fns:
        raise NotImplementedError(
            f"LEF for axle_config='{axle_config}' requires the full "
            "Appendix D table module (geotech_references.aashto_1993.lef), "
            "which is not available in this install. Triple axles are not "
            "covered by the digitized table subset."
        )
    r = table_fns[key]()
    add_ref(references, r.get("reference"))
    return r["lef"], "digitized_table"


def compute_design_esals(
    growth_rate_pct=0.0,
    design_period_yr=20.0,
    axle_groups=None,
    vehicles=None,
    base_year_w18_two_way=None,
    pavement_type="flexible",
    sn=5.0,
    d_in=9.0,
    pt=2.5,
    directional_factor=None,
    num_lanes_per_direction=2,
    lane_factor=None,
) -> DesignTrafficResult:
    """Design-lane cumulative 18-kip ESALs over the performance period.

    Exactly one traffic description must be given:

    - ``axle_groups`` : list of dicts, each
      ``{'axle_config': 'single'|'tandem'|'triple', 'load_kips': float,
      'daily_count': float}`` (two-way daily repetitions of that axle
      group). Converted to ESALs with per-axle LEFs at the assumed
      structural capacity (``sn`` for flexible / ``d_in`` for rigid, at
      ``pt``). Because LEFs depend weakly on SN or D, re-run with the
      designed value if it differs much from the assumption (the guide's
      own practice is a single pass at an assumed SN~5 / D~9).
    - ``vehicles`` : list of dicts, each ``{'description': str,
      'daily_count': float, 'truck_factor': float}`` where truck_factor is
      ESALs per vehicle pass (agency values). No LEF lookup involved.
    - ``base_year_w18_two_way`` : first-year two-way ESALs directly.

    Growth (compound, ``growth_factor``), then directional split DD
    (default 0.5) and design-lane fraction DL (Section 2.1.2 table by
    lanes per direction; midpoint of the printed range unless
    ``lane_factor`` is given as a fraction).

    Returns a ``DesignTrafficResult`` (``.w18_design_lane`` feeds the
    design functions).
    """
    supplied = [x is not None for x in (axle_groups, vehicles,
                                        base_year_w18_two_way)]
    if sum(supplied) != 1:
        raise ValueError(
            "Provide exactly one of axle_groups, vehicles, or "
            "base_year_w18_two_way."
        )
    references = []
    notes = []
    axle_breakdown = []
    vehicle_breakdown = []
    lef_basis = "direct"

    if axle_groups is not None:
        if pavement_type not in ("flexible", "rigid"):
            raise ValueError(
                f"pavement_type must be 'flexible' or 'rigid', got "
                f"'{pavement_type}'"
            )
        base_year = 0.0
        for grp in axle_groups:
            missing = {"axle_config", "load_kips", "daily_count"} - set(grp)
            if missing:
                raise ValueError(
                    f"axle_groups entries need axle_config, load_kips, "
                    f"daily_count; missing {sorted(missing)} in {grp}"
                )
            lef, lef_basis = _single_lef(
                pavement_type, str(grp["axle_config"]).strip().lower(),
                grp["load_kips"], sn, d_in, pt, references,
            )
            annual = grp["daily_count"] * 365.0 * lef
            base_year += annual
            axle_breakdown.append({
                "axle_config": grp["axle_config"],
                "load_kips": grp["load_kips"],
                "daily_count": grp["daily_count"],
                "lef": lef,
                "esals_per_year": round(annual, 1),
            })
        if lef_basis == "digitized_table":
            notes.append(
                "LEFs from the digitized SN=5 / D=9-in, pt=2.5 table curves "
                "(full Appendix D table module not installed); valid only "
                "at that design point."
            )
    elif vehicles is not None:
        base_year = 0.0
        lef_basis = "truck_factors"
        for veh in vehicles:
            missing = {"daily_count", "truck_factor"} - set(veh)
            if missing:
                raise ValueError(
                    f"vehicles entries need daily_count and truck_factor; "
                    f"missing {sorted(missing)} in {veh}"
                )
            annual = veh["daily_count"] * 365.0 * veh["truck_factor"]
            base_year += annual
            vehicle_breakdown.append({
                "description": veh.get("description", "vehicle class"),
                "daily_count": veh["daily_count"],
                "truck_factor": veh["truck_factor"],
                "esals_per_year": round(annual, 1),
            })
    else:
        if base_year_w18_two_way <= 0:
            raise ValueError(
                f"base_year_w18_two_way must be > 0, got "
                f"{base_year_w18_two_way}"
            )
        base_year = float(base_year_w18_two_way)

    gf = growth_factor(growth_rate_pct, design_period_yr)
    add_ref(references, gf["reference"])
    two_way_total = base_year * gf["growth_factor"]

    dd_info = _tb.directional_distribution_default()
    add_ref(references, dd_info["reference"])
    if directional_factor is None:
        dd = dd_info["dd_default"]
        notes.append(f"Directional factor DD defaulted to {dd}.")
    else:
        dd = float(directional_factor)
        if not (dd_info["dd_min"] <= dd <= dd_info["dd_max"]):
            notes.append(
                f"DD = {dd} is outside the guide's typical "
                f"{dd_info['dd_min']}-{dd_info['dd_max']} range."
            )

    dl_info = _tb.lane_distribution_factor(num_lanes_per_direction)
    add_ref(references, dl_info["reference"])
    if lane_factor is None:
        dl = midpoint_range(dl_info["dl_min_pct"], dl_info["dl_max_pct"]) / 100.0
        notes.append(
            f"Lane factor DL defaulted to {dl} (midpoint of the "
            f"{dl_info['dl_min_pct']}-{dl_info['dl_max_pct']}% range for "
            f"{num_lanes_per_direction} lane(s) per direction)."
        )
    else:
        dl = float(lane_factor)
        if not (0 < dl <= 1.0):
            raise ValueError(f"lane_factor must be a fraction in (0, 1], got {dl}")

    w18_lane = two_way_total * dd * dl
    return DesignTrafficResult(
        w18_design_lane=round(w18_lane, 0),
        w18_two_way_total=round(two_way_total, 0),
        base_year_w18_two_way=round(base_year, 1),
        growth_rate_pct=growth_rate_pct,
        design_period_yr=design_period_yr,
        growth_factor=gf["growth_factor"],
        directional_factor=dd,
        lane_factor=dl,
        lef_basis=lef_basis,
        axle_breakdown=axle_breakdown,
        vehicle_breakdown=vehicle_breakdown,
        notes=notes,
        references=references,
    )
