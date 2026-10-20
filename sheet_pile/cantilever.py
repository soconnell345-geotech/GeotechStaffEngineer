"""
Cantilever sheet pile wall analysis.

Determines required embedment depth and maximum moment for a cantilever
wall using the free earth support method (simplified).

All units are SI: kN, kPa, meters.

References:
    USACE EM 1110-2-2504, Chapter 4
    USS Steel Sheet Piling Design Manual, Chapter 3
    Das, "Principles of Foundation Engineering", Chapter 9
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sheet_pile.earth_pressure import (
    rankine_Ka, rankine_Kp, coulomb_Ka, coulomb_Kp, caquot_kerisel_Kp,
    active_pressure, passive_pressure, tension_crack_depth,
)
from geotech_common.water import GAMMA_W
from geotech_common import log_spiral as _log_spiral

_LOG_SPIRAL = ("log_spiral", "caquot_kerisel", "logspiral")


def _with_log_spiral_source(pressure_method: str, source: str) -> str:
    """The pressure method the integrations take: 'log_spiral' carries the
    printed source of its passive Kp as 'log_spiral:<source>' (GM-183).
    A source other than the default with another method is refused."""
    if source not in _log_spiral.SOURCES:
        raise ValueError(f"Unknown log_spiral_source {source!r}; one of "
                         f"{list(_log_spiral.SOURCES)}")
    if pressure_method not in _LOG_SPIRAL:
        if source != _log_spiral.DEFAULT_SOURCE:
            raise ValueError(
                f"log_spiral_source {source!r} is the source of the "
                f"log-spiral passive Kp; pressure_method {pressure_method!r} "
                "does not use it (use pressure_method 'log_spiral')")
        return pressure_method
    return f"log_spiral:{source}"


def _log_spiral_choices(pressure_method: str, soil_layers, excavation_depth,
                        *, tied_back: bool = False) -> list:
    """The result's ``choices`` for the log-spiral passive Kp (GM-232): the
    printed source and, by default, the UFC 3-220-20 route, with each
    passive-side friction layer's Kp from it and from every other source.
    A sheet pile wall has a vertical face and the module takes level ground
    in front of it, so the UFC route is always Eq. 4-12. ``tied_back``
    adds UFC 3-220-20 p. 260."""
    if not pressure_method.startswith("log_spiral:"):
        return []
    from geotech_common.choices import choice, how_given
    source = pressure_method.split(":", 1)[1]
    cases, top = [], 0.0
    for i, layer in enumerate(soil_layers, 1):
        bottom = top + layer.thickness
        below = bottom > excavation_depth or i == len(soil_layers)
        top = bottom
        if below and layer.friction_angle > 0:
            cases.append((f"layer {i} (phi' {layer.friction_angle:g}, "
                          f"delta {layer.wall_friction_deg:g})",
                          layer.friction_angle, layer.wall_friction_deg))
    if not cases:
        return []

    def values(key):
        got = [(name, *_log_spiral.source_Kp(key, phi, delta))
               for name, phi, delta in cases]
        why = next((r for _, _, r in got if r), None)
        if why:
            return f"none here, {why}"
        return "; ".join(f"{name} Kp {kp:.3g}" for name, kp, _ in got)

    if source == "ufc_3_220_20":
        basis = ("UFC 3-220-20 p. 191: the log spiral method 'is commonly "
                 "used for cut walls, such as sheet pile or soldier pile "
                 "and lagging ... These walls are vertical and often have "
                 "no backslope.' The module takes a vertical face and level "
                 "ground in front of the wall, so the route is Eq. 4-12 at "
                 "each layer's wall friction: " + values("dm7_2_eq_4_12")
                 + ". " + _log_spiral.EQ_VS_CHARTS_NOTE)
        if tied_back:
            basis += ". " + _log_spiral.TIED_BACK_NOTE
        used = (source, "dm7_2_eq_4_12")
    else:
        basis = (_log_spiral.SOURCE_BASIS[source] + ": " + values(source))
        if tied_back:
            basis += ". " + _log_spiral.TIED_BACK_QUOTE
        used = (source,)
    if "dm7_2_eq_4_12" in used and any(
            _log_spiral.above_the_charts(phi, delta)
            for _, phi, delta in cases):
        basis += ". " + _log_spiral.HIGH_FRICTION_NOTE
    if tied_back:
        basis += ". " + _log_spiral.ANCHORED_BULKHEAD_QUOTE
    # each layer's wall friction is the engineer's (wall_friction_deg)
    basis += ". " + _log_spiral.wall_friction_note(
        source, left_out=(
            "The module takes each layer's wall_friction_deg, 0 when left "
            "out: a smooth wall, for which Eq. 4-12 and the 1948 chart give "
            "Rankine's Kp."))
    return [choice(
        "Log-spiral passive Kp (log_spiral_source)", source,
        how_given(source, _log_spiral.DEFAULT_SOURCE,
                  name="log_spiral_source"),
        basis,
        [f"{_log_spiral.source_label(key)}: {values(key)}"
         for key in _log_spiral.SOURCES if key not in used])]


@dataclass
class WallSoilLayer:
    """Soil layer for sheet pile wall analysis.

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    unit_weight : float
        Total unit weight (kN/m³).
    friction_angle : float
        Drained friction angle (degrees).
    cohesion : float, optional
        Cohesion (kPa). Default 0.
    description : str, optional
        Layer description.
    wall_friction_deg : float, optional
        Wall friction angle delta (degrees), used by the Coulomb and
        log-spiral pressure methods. Default 0 (smooth wall -> Coulomb, and
        the log-spiral Kp of Eq. 4-12 or the 1948 chart, reduce to Rankine).
        A nonzero value lowers the active coefficient and raises the passive
        coefficient. Must be in [0, friction_angle].
    """
    thickness: float
    unit_weight: float
    friction_angle: float = 30.0
    cohesion: float = 0.0
    description: str = ""
    wall_friction_deg: float = 0.0

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(f"Layer thickness must be positive, got {self.thickness}")
        if self.friction_angle < 0 or self.friction_angle > 50:
            raise ValueError(f"Friction angle must be 0-50, got {self.friction_angle}")
        if self.cohesion < 0:
            raise ValueError(f"Cohesion must be non-negative, got {self.cohesion}")
        if self.cohesion == 0 and self.friction_angle == 0:
            raise ValueError("Soil must have c > 0 or phi > 0")
        if self.wall_friction_deg < 0 or self.wall_friction_deg > self.friction_angle:
            raise ValueError(
                "Wall friction delta must be in [0, phi="
                f"{self.friction_angle}], got {self.wall_friction_deg}")


@dataclass
class CantileverWallResult:
    """Results from a cantilever sheet pile wall analysis.

    ``embedment_depth`` is the design embedment
    (= ``embedment_converged`` x ``embedment_increase``); the safety
    basis is recorded by ``FOS_passive`` and ``embedment_increase``.
    """
    embedment_depth: float = 0.0
    total_wall_length: float = 0.0
    max_moment: float = 0.0
    max_moment_depth: float = 0.0
    FOS_passive: float = 1.0
    excavation_depth: float = 0.0
    embedment_converged: float = 0.0
    embedment_increase: float = 1.0
    #: The moment integration's own points, top to toe: [z, p_a, p_p/FOS,
    #: net water pressure, net pressure p, shear V, moment M] (kPa, kN/m,
    #: kN-m/m). Not in to_dict; read by get_tables.
    profile: List[list] = field(default_factory=list, repr=False)
    #: What the analysis chose (GM-232: the log-spiral passive Kp's source).
    choices: list = field(default_factory=list)

    def get_tables(self) -> list:
        """The pressure diagram along the wall with the shear and bending
        moment the module integrates from it (the points its maximum moment
        is found on), and the design summary (CSV only; the calc package
        prints it). Values are this result's own, per metre of wall.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        if self.profile:
            tables.append(CalcTable(
                title="Pressure, shear and moment along the wall",
                columns=[("Depth z", "m", ".3f"),
                         ("Active pressure p_a", "kPa", ".2f"),
                         ("Passive pressure p_p/FOS", "kPa", ".2f"),
                         ("Net water pressure", "kPa", ".2f"),
                         ("Net pressure p", "kPa", ".2f"),
                         ("Shear V", "kN/m", ".2f"),
                         ("Bending moment M", "kN-m/m", ".2f")],
                rows=[list(r) for r in self.profile],
                note=(f"Free earth support, wall {self.total_wall_length:.2f}"
                      f" m long, excavation {self.excavation_depth:.2f} m; "
                      f"passive divided by FOS = {self.FOS_passive:g}; V and "
                      "M integrated from the top at the midpoint of each of "
                      f"{len(self.profile)} steps; the maximum |M| is "
                      f"{self.max_moment:.1f} kN-m/m at "
                      f"{self.max_moment_depth:.2f} m"),
                name="wall_pressures",
            ))
        tables.append(CalcTable(
            title="Wall design summary",
            columns=["Quantity", "Symbol", "Value", "Unit"],
            rows=[
                ["Excavation depth", "H", self.excavation_depth, "m"],
                ["Embedment from moment equilibrium", "D0",
                 self.embedment_converged, "m"],
                ["Embedment increase", "", self.embedment_increase, ""],
                ["Design embedment", "D", self.embedment_depth, "m"],
                ["Total wall length", "H + D", self.total_wall_length, "m"],
                ["Factor of safety on passive", "FOS", self.FOS_passive, ""],
                ["Maximum bending moment", "M_max", self.max_moment,
                 "kN-m/m"],
                ["Depth of maximum moment", "z_Mmax", self.max_moment_depth,
                 "m"],
            ],
            note="Cantilever sheet pile wall, free earth support",
            name="design_summary", printed=False,
        ))
        return tables

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CANTILEVER SHEET PILE WALL RESULTS",
            "=" * 60,
            "",
            f"  Excavation depth:   {self.excavation_depth:.2f} m",
            f"  Required embedment: {self.embedment_depth:.2f} m"
            + (f" (= {self.embedment_converged:.2f} x "
               f"{self.embedment_increase:.2f})"
               if self.embedment_increase != 1.0 else ""),
            f"  Total wall length:  {self.total_wall_length:.2f} m",
            f"  FOS on passive:     {self.FOS_passive:.2f}",
            "",
            f"  Max bending moment: {self.max_moment:.1f} kN-m/m",
            f"  at depth:           {self.max_moment_depth:.2f} m below top",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def plot_wall_diagram(self, ax=None, show=True):
        """Plot schematic wall diagram with excavation and embedment zones.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))
        H = self.excavation_depth
        D = self.embedment_depth
        # Wall line
        ax.plot([0, 0], [0, H + D], 'k-', linewidth=3, label='Wall')
        # Excavation zone
        ax.fill_betweenx([0, H], -0.5, 0, color='lightyellow', alpha=0.5)
        ax.annotate('Excavation', xy=(-0.25, H / 2), ha='center',
                    fontsize=9, fontstyle='italic')
        # Embedment zone
        ax.fill_betweenx([H, H + D], -0.5, 0.5, color='tan', alpha=0.4)
        ax.annotate('Embedment', xy=(0.25, H + D / 2), ha='center',
                    fontsize=9, fontstyle='italic')
        # Retained soil
        ax.fill_betweenx([0, H + D], 0, 0.5, color='burlywood', alpha=0.3,
                         label='Retained Soil')
        # Excavation level
        ax.axhline(y=H, color='blue', linestyle='--', linewidth=1,
                   label=f'Excavation ({H:.1f} m)')
        # Max moment location
        if self.max_moment_depth > 0:
            ax.plot(0, self.max_moment_depth, 'ro', markersize=8,
                    label=f'Max Moment ({self.max_moment:.0f} kN-m/m)')
        ax.invert_yaxis()
        setup_engineering_plot(ax, "Sheet Pile Wall Diagram",
                              "", "Depth (m)")
        ax.legend(fontsize=8, loc='lower right')
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self):
        return {
            "embedment_depth_m": round(self.embedment_depth, 3),
            "embedment_converged_m": round(self.embedment_converged, 3),
            "embedment_increase": round(self.embedment_increase, 3),
            "total_wall_length_m": round(self.total_wall_length, 3),
            "max_moment_kNm_per_m": round(self.max_moment, 1),
            "max_moment_depth_m": round(self.max_moment_depth, 2),
            "FOS_passive": round(self.FOS_passive, 2),
            "excavation_depth_m": round(self.excavation_depth, 2),
            **({"choices": self.choices} if self.choices else {}),
        }


def analyze_cantilever(
    excavation_depth: float,
    soil_layers: List[WallSoilLayer],
    gwt_depth_active: Optional[float] = None,
    gwt_depth_passive: Optional[float] = None,
    surcharge: float = 0.0,
    FOS_passive: float = 1.5,
    gamma_w: float = GAMMA_W,
    pressure_method: str = "rankine",
    embedment_increase: float = 1.0,
    log_spiral_source: str = "ufc_3_220_20",
) -> CantileverWallResult:
    """Analyze a cantilever sheet pile wall.

    Uses the simplified free earth support method:
    1. Compute active and passive pressure diagrams
    2. Sum moments about the base to find required embedment D
    3. Compute maximum moment

    Safety basis
    ------------
    The classic simplified method applies EITHER a factor of safety on
    the passive resistance (FS ~ 1.5-2.0, no depth increase) OR an
    unfactored passive diagram (FS ~ 1.0) with a 20-40% embedment
    increase — not both (USS Sheet Piling Manual; USACE EM 1110-2-2504;
    Das Ch. 9). The default here is the FOS basis: ``FOS_passive = 1.5``
    with ``embedment_increase = 1.0``. To use the depth-increase basis
    instead, pass ``FOS_passive = 1.0`` and ``embedment_increase`` of
    1.2-1.4. (Prior to v5.1 this function applied BOTH FOS 1.5 and a
    1.2x increase — a double safety margin; see SP-3 in the QC review
    and DESIGN.md.)

    Parameters
    ----------
    excavation_depth : float
        Depth of excavation H (m).
    soil_layers : list of WallSoilLayer
        Soil layers from top to bottom (both retained and embedded sides).
    gwt_depth_active : float, optional
        GWT depth on the active (retained) side (m from top of wall).
    gwt_depth_passive : float, optional
        GWT depth on the passive (excavation) side (m from top of wall).
        If None, same as active side.
    surcharge : float, optional
        Uniform surcharge on the retained side (kPa). Default 0.
    FOS_passive : float, optional
        Factor of safety applied to passive resistance. Default 1.5.
        This is the primary safety knob (see Safety basis above).
    gamma_w : float, optional
        Unit weight of water (kN/m³). Default 9.81.
    pressure_method : str, optional
        "rankine" (default), "coulomb" or "log_spiral" (Coulomb active and
        the log-spiral passive Kp).
    embedment_increase : float, optional
        Multiplier applied to the converged embedment to obtain the
        design embedment. Default 1.0 (no increase — safety carried by
        FOS_passive). Use 1.2-1.4 with FOS_passive = 1.0 for the
        depth-increase basis. Must be >= 1.0.
    log_spiral_source : str, optional
        With pressure_method "log_spiral", the printed source of the
        passive Kp (``geotech_common.log_spiral.SOURCES``). Default
        "ufc_3_220_20": UFC 3-220-20's source by the wall's geometry, here
        always Eq. 4-12 (p. 191; a vertical face and level ground, any wall
        friction). Or the Caquot-Kerisel (1948) chart, Caltrans T&S Fig.
        4-20 ("caquot_kerisel_1948"), or a UFC source by name. The result's
        ``choices`` give each layer's Kp from it and from the others.

    Returns
    -------
    CantileverWallResult
    """
    if excavation_depth <= 0:
        raise ValueError(f"Excavation depth must be positive, got {excavation_depth}")
    if embedment_increase < 1.0:
        raise ValueError(
            f"embedment_increase must be >= 1.0, got {embedment_increase}")

    if gwt_depth_passive is None:
        gwt_depth_passive = gwt_depth_active
    pressure_method = _with_log_spiral_source(pressure_method,
                                              log_spiral_source)

    # For a simple single-layer case, we use analytical formulas
    # For multi-layer, we use numerical integration
    n_points = 200
    D_max = excavation_depth * 3  # initial guess: embedment up to 3H
    total_length = excavation_depth + D_max
    dz = total_length / n_points

    # Find embedment by iteration
    D_converged = _find_embedment(
        excavation_depth, soil_layers, gwt_depth_active, gwt_depth_passive,
        surcharge, FOS_passive, gamma_w, pressure_method, n_points=500
    )

    # Design embedment. Default embedment_increase = 1.0: safety is carried
    # entirely by FOS_passive (single, explicit safety basis — see docstring).
    # A 1.2-1.4 increase is only appropriate with FOS_passive ~ 1.0.
    D_design = D_converged * embedment_increase

    total_wall = excavation_depth + D_design

    # Compute maximum moment (and keep the integration's points for the
    # calculation tables)
    profile = []
    max_moment, max_moment_depth = _compute_max_moment(
        excavation_depth, D_design, soil_layers,
        gwt_depth_active, gwt_depth_passive,
        surcharge, FOS_passive, gamma_w, pressure_method, n_points=500,
        profile=profile,
    )

    return CantileverWallResult(
        embedment_depth=D_design,
        total_wall_length=total_wall,
        max_moment=max_moment,
        max_moment_depth=max_moment_depth,
        FOS_passive=FOS_passive,
        excavation_depth=excavation_depth,
        embedment_converged=D_converged,
        embedment_increase=embedment_increase,
        profile=profile,
        choices=_log_spiral_choices(pressure_method, soil_layers,
                                    excavation_depth),
    )


def _compute_Ka_Kp(phi_deg: float, method: str = "rankine",
                   delta_deg: float = 0.0):
    """Compute Ka, Kp using the specified method.

    For method="coulomb", delta_deg is the wall friction angle (default 0 — a
    smooth wall, for which Coulomb reduces to Rankine). A nonzero delta lowers
    the active coefficient and raises the passive coefficient. Rankine ignores
    wall friction.

    method="log_spiral" (Caquot-Kerisel) uses the Coulomb active coefficient but
    the log-spiral PASSIVE coefficient (``caquot_kerisel_Kp``), which avoids
    Coulomb's over-prediction of Kp at high delta/phi. Requires delta_deg.
    "log_spiral:<source>" takes the passive Kp from that printed source
    (``geotech_common.log_spiral``; GM-183).
    """
    if method.startswith("log_spiral:"):
        return (coulomb_Ka(phi_deg, delta_deg=delta_deg),
                _log_spiral.log_spiral_Kp(phi_deg, delta_deg,
                                          source=method.split(":", 1)[1]))
    if method in _LOG_SPIRAL:
        return (coulomb_Ka(phi_deg, delta_deg=delta_deg),
                caquot_kerisel_Kp(phi_deg, delta_deg=delta_deg))
    if method == "coulomb":
        return (coulomb_Ka(phi_deg, delta_deg=delta_deg),
                coulomb_Kp(phi_deg, delta_deg=delta_deg))
    return rankine_Ka(phi_deg), rankine_Kp(phi_deg)


def _get_soil_at_depth(depth: float, soil_layers: List[WallSoilLayer]) -> WallSoilLayer:
    """Get the soil layer at a given depth from top of wall."""
    z = 0
    for layer in soil_layers:
        if z + layer.thickness > depth:
            return layer
        z += layer.thickness
    return soil_layers[-1]


def _effective_gamma(depth: float, layer: WallSoilLayer,
                     gwt_depth: Optional[float], gamma_w: float) -> float:
    """Get effective unit weight at depth."""
    if gwt_depth is not None and depth > gwt_depth:
        return layer.unit_weight - gamma_w
    return layer.unit_weight


def _find_embedment(excavation_depth, soil_layers, gwt_active, gwt_passive,
                    surcharge, FOS_passive, gamma_w,
                    pressure_method="rankine", n_points=500):
    """Find embedment depth by summing moments about the base.

    Scans a coarse grid for the first moment balance, then BISECTS the
    bracketing interval to 1 mm — the previous first-grid-crossing return
    overshot the true root by up to one grid step (~2-3 % on typical walls,
    the "grid quantization" reconciliation of the Das 9.14 sweep), and the
    0.5 m grid start acted as an artificial embedment floor for small walls.
    """
    H = excavation_depth

    def _net_moment(D_trial):
        """Resisting minus driving moment about the base for a trial D."""
        total_length = H + D_trial
        dz = total_length / n_points

        # Compute net moment about the wall base
        moment_active = 0.0
        moment_passive = 0.0

        sigma_v_active = surcharge
        sigma_v_passive = 0.0

        for i in range(n_points):
            z = (i + 0.5) * dz  # depth from top of wall
            arm = total_length - z  # moment arm from base

            layer = _get_soil_at_depth(z, soil_layers)
            Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method, layer.wall_friction_deg)

            if z <= H:
                # Above excavation: only active pressure
                gamma_eff = _effective_gamma(z, layer, gwt_active, gamma_w)
                sigma_v_at_z = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
                pa = Ka * sigma_v_at_z - 2 * layer.cohesion * math.sqrt(Ka)
                pa = max(pa, 0)  # no tension
                moment_active += pa * dz * arm

                # Water pressure on active side
                if gwt_active is not None and z > gwt_active:
                    u_active = gamma_w * (z - gwt_active)
                    moment_active += u_active * dz * arm
            else:
                # Below excavation: active + net passive
                z_below = z - H  # depth below excavation

                # Active side
                sigma_v_at_z = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
                pa = Ka * sigma_v_at_z - 2 * layer.cohesion * math.sqrt(Ka)
                pa = max(pa, 0)
                moment_active += pa * dz * arm

                # Passive side (with FOS)
                layer_passive = _get_soil_at_depth(z, soil_layers)
                sigma_v_passive_z = _cumulative_stress_passive(z_below, H, soil_layers, gwt_passive, gamma_w)
                pp = Kp * sigma_v_passive_z + 2 * layer_passive.cohesion * math.sqrt(Kp)
                pp_reduced = pp / FOS_passive
                moment_passive += pp_reduced * dz * arm

                # Differential water pressure
                if gwt_active is not None and z > gwt_active:
                    u_active = gamma_w * (z - gwt_active)
                else:
                    u_active = 0
                if gwt_passive is not None and z > gwt_passive:
                    u_passive = gamma_w * (z_below - max(0, gwt_passive - H))
                else:
                    u_passive = 0
                net_water = u_active - u_passive
                if net_water > 0:
                    moment_active += net_water * dz * arm

        return moment_passive - moment_active

    def _bisect(lo, hi):
        for _ in range(60):
            if hi - lo <= 1e-3:
                break
            mid = (lo + hi) / 2.0
            if _net_moment(mid) >= 0:
                hi = mid
            else:
                lo = mid
        return hi

    D_prev = None
    for D_trial in np.linspace(0.5, 4 * H, 200):
        if _net_moment(D_trial) >= 0:
            # Balanced at the FIRST grid point: the true root is below the
            # old 0.5 m floor — refine down instead of returning the floor.
            return _bisect(0.01 if D_prev is None else D_prev, D_trial)
        D_prev = D_trial

    warnings.warn("Embedment depth did not converge; using maximum trial depth")
    return 4 * H


def _cumulative_stress(z: float, soil_layers: List[WallSoilLayer],
                       gwt_depth: Optional[float], gamma_w: float) -> float:
    """Compute cumulative vertical effective stress at depth z."""
    sigma_v = 0.0
    depth = 0.0
    for layer in soil_layers:
        if depth >= z:
            break
        dz = min(layer.thickness, z - depth)
        if gwt_depth is not None and depth + dz > gwt_depth:
            # Part above GWT, part below
            above = max(0, gwt_depth - depth)
            below = dz - above
            sigma_v += layer.unit_weight * above
            sigma_v += (layer.unit_weight - gamma_w) * below
        else:
            sigma_v += layer.unit_weight * dz
        depth += layer.thickness
    return sigma_v


def _cumulative_stress_passive(z_below_exc: float, excavation_depth: float,
                               soil_layers: List[WallSoilLayer],
                               gwt_depth: Optional[float],
                               gamma_w: float) -> float:
    """Compute vertical effective stress on the passive side below excavation.

    Unlike _cumulative_stress which starts from the ground surface,
    this starts accumulating from the excavation depth downward using
    the soil layers that actually exist at that depth.

    Parameters
    ----------
    z_below_exc : float
        Depth below the excavation line (m).
    excavation_depth : float
        Depth of excavation from top of wall (m).
    soil_layers : list of WallSoilLayer
        Soil layers from top to bottom.
    gwt_depth : float or None
        GWT depth on the passive side (m from top of wall).
    gamma_w : float
        Unit weight of water.
    """
    # Absolute depth from the top of the wall
    z_abs = excavation_depth + z_below_exc
    # GWT depth below the excavation line (passive side reference)
    gwt_below_exc = None
    if gwt_depth is not None:
        gwt_below_exc = max(0.0, gwt_depth - excavation_depth)

    # Walk through soil layers to find where the excavation depth falls,
    # then accumulate stress from there
    sigma_v = 0.0
    depth = 0.0
    for layer in soil_layers:
        layer_top = depth
        layer_bot = depth + layer.thickness
        depth = layer_bot

        if layer_bot <= excavation_depth:
            continue  # skip layers entirely above excavation

        # The portion of this layer that is below the excavation
        start = max(layer_top, excavation_depth)
        end = min(layer_bot, z_abs)
        if start >= end:
            continue

        # Depth below excavation for this segment
        seg_top_below = start - excavation_depth
        seg_bot_below = end - excavation_depth

        dz_seg = seg_bot_below - seg_top_below
        if gwt_below_exc is not None and seg_bot_below > gwt_below_exc:
            above = max(0.0, gwt_below_exc - seg_top_below)
            below = dz_seg - above
            sigma_v += layer.unit_weight * above
            sigma_v += (layer.unit_weight - gamma_w) * below
        else:
            sigma_v += layer.unit_weight * dz_seg

        if layer_bot >= z_abs:
            break

    # If z_abs extends beyond all defined layers, use the last layer
    if depth < z_abs and soil_layers:
        last = soil_layers[-1]
        remaining = z_abs - max(depth, excavation_depth)
        if remaining > 0:
            if gwt_below_exc is not None:
                depth_below = max(depth, excavation_depth) - excavation_depth
                if depth_below < gwt_below_exc:
                    above = min(remaining, gwt_below_exc - depth_below)
                    below = remaining - above
                    sigma_v += last.unit_weight * above
                    sigma_v += (last.unit_weight - gamma_w) * below
                else:
                    sigma_v += (last.unit_weight - gamma_w) * remaining
            else:
                sigma_v += last.unit_weight * remaining

    return sigma_v


def _compute_max_moment(excavation_depth, embedment, soil_layers,
                        gwt_active, gwt_passive, surcharge,
                        FOS_passive, gamma_w,
                        pressure_method="rankine", n_points=500,
                        profile=None):
    """Compute maximum bending moment and its location.

    ``profile``, when a list is given, receives one row per integration
    point: ``[z, p_a, p_p/FOS, net water, net pressure, V, M]``."""
    H = excavation_depth
    total_length = H + embedment
    dz = total_length / n_points

    # Compute shear force along the wall; moment is max where shear = 0
    shear = 0.0
    moment = 0.0
    max_moment = 0.0
    max_moment_depth = 0.0

    for i in range(n_points):
        z = (i + 0.5) * dz
        layer = _get_soil_at_depth(z, soil_layers)
        Ka, Kp = _compute_Ka_Kp(layer.friction_angle, pressure_method, layer.wall_friction_deg)

        net_pressure = 0.0
        pp_reduced = 0.0
        water = 0.0
        if z <= H:
            sigma_v = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
            pa = Ka * sigma_v - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)
            net_pressure = pa
            if gwt_active is not None and z > gwt_active:
                water = gamma_w * (z - gwt_active)
                net_pressure += water
        else:
            z_below = z - H
            sigma_v_a = surcharge + _cumulative_stress(z, soil_layers, gwt_active, gamma_w)
            pa = Ka * sigma_v_a - 2 * layer.cohesion * math.sqrt(Ka)
            pa = max(pa, 0)

            sigma_v_p = _cumulative_stress_passive(z_below, H, soil_layers, gwt_passive, gamma_w)
            pp = Kp * sigma_v_p + 2 * layer.cohesion * math.sqrt(Kp)
            pp_reduced = pp / FOS_passive

            net_pressure = pa - pp_reduced

            # Water
            u_active = gamma_w * max(0, z - (1e10 if gwt_active is None else gwt_active))
            u_passive = gamma_w * max(0, z_below - max(0, (1e10 if gwt_passive is None else gwt_passive) - H))
            water = u_active - u_passive
            net_pressure += water

        shear += net_pressure * dz
        moment += shear * dz

        if abs(moment) > abs(max_moment):
            max_moment = abs(moment)
            max_moment_depth = z
        if profile is not None:
            profile.append([z, pa, pp_reduced, water, net_pressure, shear,
                            moment])

    return max_moment, max_moment_depth
