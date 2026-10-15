"""
Result containers for 2D FEM analysis.

Each dataclass stores analysis outputs and provides:
- summary() -> formatted string for human reading
- to_dict() -> flat dict for LLM agent consumption

All units SI: meters, kPa, kN, degrees.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


def movement_summary(nodes, u, mask=None) -> Dict[str, Any]:
    """Extremes of a nodal displacement field, with where they occur.

    Parameters
    ----------
    nodes : (n_nodes, 2) array — node coordinates (m).
    u : (>= 2*n_nodes,) array — interleaved [ux, uy] displacements (m);
        rotational DOFs beyond 2*n_nodes are ignored.
    mask : (n_nodes,) bool array, optional — nodes to consider (e.g. those
        in active elements). None = all nodes.

    Returns
    -------
    dict (y is up, so a negative uy is downward):
        settlement_m — the most negative vertical displacement (m, signed:
            negative = down; 0.0 when nothing moved down)
        heave_m — the most positive vertical displacement (m; 0.0 when
            nothing moved up)
        horizontal_m — the horizontal displacement of largest magnitude
            (m, signed: positive = +x)
        total_m — the largest displacement magnitude |u| (m)
        *_at_xy — [x, y] (m) of each (None when the value is 0)
    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    u = np.asarray(u, dtype=float)[:2 * n]
    ux, uy = u[0::2], u[1::2]
    idx = np.arange(n) if mask is None else np.where(np.asarray(mask))[0]
    out = {}

    def _at(i, value):
        if i is None or value == 0.0:
            return None
        return [round(float(nodes[i, 0]), 3), round(float(nodes[i, 1]), 3)]

    if len(idx) == 0:
        for key in ("settlement", "heave", "horizontal", "total"):
            out[f"{key}_m"] = 0.0
            out[f"{key}_at_xy"] = None
        return out
    i_down = idx[np.argmin(uy[idx])]
    i_up = idx[np.argmax(uy[idx])]
    i_h = idx[np.argmax(np.abs(ux[idx]))]
    mag = np.sqrt(ux ** 2 + uy ** 2)
    i_t = idx[np.argmax(mag[idx])]
    s = round(min(float(uy[i_down]), 0.0), 6) + 0.0
    hv = round(max(float(uy[i_up]), 0.0), 6) + 0.0
    hz = round(float(ux[i_h]), 6) + 0.0
    tt = round(float(mag[i_t]), 6) + 0.0
    out["settlement_m"] = s
    out["settlement_at_xy"] = _at(i_down, s)
    out["heave_m"] = hv
    out["heave_at_xy"] = _at(i_up, hv)
    out["horizontal_m"] = hz
    out["horizontal_at_xy"] = _at(i_h, hz)
    out["total_m"] = tt
    out["total_at_xy"] = _at(i_t, tt)
    return out


def _material_difference(a, b, rel=0.05, abs_floor=1e-4):
    """True when two displacements (m) differ by more than 5 % of the larger
    AND by more than 0.1 mm — the threshold for a reference-stage choice to
    count as changing the reported numbers."""
    a, b = float(a), float(b)
    diff = abs(a - b)
    return diff > abs_floor and diff > rel * max(abs(a), abs(b))


@dataclass
class BeamForceResult:
    """Internal forces for one beam element.

    Attributes
    ----------
    element_index : int
    node_i, node_j : int
    axial_i, shear_i, moment_i : float — forces at node i.
    axial_j, shear_j, moment_j : float — forces at node j.
    length : float — element length (m).
    """
    element_index: int = 0
    node_i: int = 0
    node_j: int = 0
    axial_i: float = 0.0
    shear_i: float = 0.0
    moment_i: float = 0.0
    axial_j: float = 0.0
    shear_j: float = 0.0
    moment_j: float = 0.0
    length: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'element_index': self.element_index,
            'node_i': self.node_i, 'node_j': self.node_j,
            'axial_i_kN': round(self.axial_i, 2),
            'shear_i_kN': round(self.shear_i, 2),
            'moment_i_kNm': round(self.moment_i, 2),
            'axial_j_kN': round(self.axial_j, 2),
            'shear_j_kN': round(self.shear_j, 2),
            'moment_j_kNm': round(self.moment_j, 2),
            'length_m': round(self.length, 3),
        }


@dataclass
class SeepageResult:
    """Results from a steady-state seepage analysis.

    Attributes
    ----------
    n_nodes : int
    n_elements : int
    max_head_m : float — maximum total head.
    min_head_m : float — minimum total head.
    max_pore_pressure_kPa : float
    max_velocity_m_per_s : float — maximum Darcy velocity magnitude.
    total_flow_m3_per_s_per_m : float — total flow through domain.
    """
    n_nodes: int = 0
    n_elements: int = 0
    max_head_m: float = 0.0
    min_head_m: float = 0.0
    max_pore_pressure_kPa: float = 0.0
    max_velocity_m_per_s: float = 0.0
    total_flow_m3_per_s_per_m: float = 0.0

    # Raw arrays (not serialized to dict)
    head: Optional[np.ndarray] = field(default=None, repr=False)
    pore_pressures: Optional[np.ndarray] = field(default=None, repr=False)
    velocity: Optional[np.ndarray] = field(default=None, repr=False)
    nodes: Optional[np.ndarray] = field(default=None, repr=False)
    elements: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  STEADY-STATE SEEPAGE RESULTS",
            "=" * 60,
            "",
            f"  Mesh: {self.n_nodes} nodes, {self.n_elements} elements",
            "",
            f"  Head range: {self.min_head_m:.3f} to {self.max_head_m:.3f} m",
            f"  Max pore pressure: {self.max_pore_pressure_kPa:.2f} kPa",
            f"  Max velocity: {self.max_velocity_m_per_s:.2e} m/s",
            f"  Total flow: {self.total_flow_m3_per_s_per_m:.2e} m³/s/m",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "max_head_m": round(self.max_head_m, 4),
            "min_head_m": round(self.min_head_m, 4),
            "max_pore_pressure_kPa": round(self.max_pore_pressure_kPa, 2),
            "max_velocity_m_per_s": float(f"{self.max_velocity_m_per_s:.4e}"),
            "total_flow_m3_per_s_per_m": float(
                f"{self.total_flow_m3_per_s_per_m:.4e}"),
        }


@dataclass
class ConsolidationResult:
    """Results from a coupled Biot consolidation analysis.

    Attributes
    ----------
    n_nodes : int
    n_elements : int
    n_time_steps : int
    times : (n_steps,) array — time since loading (s); times[0] = 0.
    max_settlement_m : float — maximum (most negative) surface settlement,
        measured from the reference stage (``displacement_reference``).
    max_excess_pore_pressure_kPa : float
    degree_of_consolidation : float — U at final time (0 to 1).
    converged : bool
    settlements / displacements : per-time surface settlement / field,
        measured from the reference stage (default: the end of the initial
        self-weight stage, i.e. the load stage alone).
    degree_of_consolidation_history : (n_steps,) array — U at every time.
    final_drained_settlement_m : float or None — the drained end state of
        the LOAD stage (monolithic), the U = 1 reference.
    scheme : str — "staggered" or "monolithic".
    notes : list of str — basis and caveats.
    stages : list of dict — per construction stage: the initial
        (self-weight) stage and the load / consolidation stage, each with
        its own movement and the cumulative movement (see to_dict).
    displacement_reference : dict — which stage the reported movements are
        measured from, and who chose it.
    judgment : dict or None — the reference-stage choice when it was a
        default AND it changes the reported numbers materially.
    initial_stage_displacements, load_stage_displacements,
    displacement_nodes : the per-stage fields (load stage per time) and the
        nodes they refer to (T6 nodes for the monolithic scheme).
    """
    n_nodes: int = 0
    n_elements: int = 0
    n_time_steps: int = 0
    times: Optional[np.ndarray] = field(default=None, repr=False)
    max_settlement_m: float = 0.0
    max_excess_pore_pressure_kPa: float = 0.0
    degree_of_consolidation: float = 0.0
    converged: bool = True

    # Time histories (displacement / pore-pressure fields are not
    # serialized; U and settlement per time are)
    displacements: Optional[np.ndarray] = field(default=None, repr=False)
    pore_pressures: Optional[np.ndarray] = field(default=None, repr=False)
    settlements: Optional[np.ndarray] = field(default=None, repr=False)
    degree_of_consolidation_history: Optional[np.ndarray] = field(
        default=None, repr=False)
    excess_pore_pressures: Optional[np.ndarray] = field(
        default=None, repr=False)
    final_drained_settlement_m: Optional[float] = None
    scheme: str = ""
    notes: list = field(default_factory=list)
    stages: list = field(default_factory=list)
    displacement_reference: Optional[Dict[str, Any]] = None
    judgment: Optional[Dict[str, Any]] = None
    initial_stage_displacements: Optional[np.ndarray] = field(
        default=None, repr=False)
    load_stage_displacements: Optional[np.ndarray] = field(
        default=None, repr=False)
    displacement_nodes: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  BIOT CONSOLIDATION RESULTS",
            "=" * 60,
            "",
            f"  Mesh: {self.n_nodes} nodes, {self.n_elements} elements",
            f"  Time steps: {self.n_time_steps}",
            f"  Converged: {self.converged}",
            "",
            f"  Max settlement: {self.max_settlement_m:.6f} m",
            f"  Max excess pore pressure: "
            f"{self.max_excess_pore_pressure_kPa:.2f} kPa",
            f"  Degree of consolidation: "
            f"{self.degree_of_consolidation:.3f}",
        ]
        if self.displacement_reference:
            lines.append(f"  Movements measured from: "
                         f"{self.displacement_reference.get('measured_from')}")
        for st in self.stages:
            lines.append(
                f"  Stage {st['stage']} ({st['name']}): this stage "
                f"{st['surface_settlement_m'] * 1000:.2f} mm, cumulative "
                f"{st['cumulative_surface_settlement_m'] * 1000:.2f} mm")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "n_time_steps": self.n_time_steps,
            "converged": self.converged,
            "max_settlement_m": round(self.max_settlement_m, 6),
            "max_excess_pore_pressure_kPa": round(
                self.max_excess_pore_pressure_kPa, 2),
            "degree_of_consolidation": round(
                self.degree_of_consolidation, 4),
        }
        # Per-time output (live smoke G6, CON-1: time_points were "for
        # output" but only the final U came back).
        if self.times is not None and self.settlements is not None:
            d["time_s"] = [float(t) for t in np.asarray(self.times)]
            d["surface_settlement_m_by_time"] = [
                round(float(s), 6) for s in np.asarray(self.settlements)]
            if self.degree_of_consolidation_history is not None:
                d["degree_of_consolidation_by_time"] = [
                    round(float(u), 4)
                    for u in np.asarray(self.degree_of_consolidation_history)]
            if self.excess_pore_pressures is not None:
                ex = np.asarray(self.excess_pore_pressures)
                d["max_excess_pore_pressure_kPa_by_time"] = [
                    round(float(v), 2) for v in np.abs(ex).max(axis=1)]
        if self.final_drained_settlement_m is not None:
            d["final_drained_settlement_m"] = round(
                self.final_drained_settlement_m, 6)
        if self.displacement_reference:
            d["displacement_reference"] = dict(self.displacement_reference)
        if self.stages:
            d["stages"] = [dict(st) for st in self.stages]
        if self.judgment:
            d["judgment"] = self.judgment
        if self.scheme:
            d["scheme"] = self.scheme
        if self.notes:
            d["notes"] = list(self.notes)
        return d

    def get_tables(self) -> list:
        """The surface settlement and the degree of consolidation at every
        time step, as this result holds them (time in seconds, settlement
        signed, negative = down, from the reference stage).

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        if self.times is None or self.settlements is None:
            return []
        times = np.asarray(self.times, dtype=float)
        sett = np.asarray(self.settlements, dtype=float)
        u_hist = (np.asarray(self.degree_of_consolidation_history,
                             dtype=float)
                  if self.degree_of_consolidation_history is not None
                  else None)
        rows = [[float(times[i]), float(sett[i]),
                 float(u_hist[i]) if u_hist is not None
                 and i < len(u_hist) else None]
                for i in range(min(len(times), len(sett)))]
        ref = (self.displacement_reference or {}).get("measured_from", "")
        return [CalcTable(
            title="Settlement and consolidation against time",
            columns=[("Time t", "s"), ("Surface settlement", "m"),
                     ("Average degree of consolidation U", "", ".3f")],
            rows=rows,
            note=(f"Coupled Biot consolidation ({self.scheme or 'fem2d'}); "
                  "settlement signed (negative = down)"
                  + (f", measured from {ref}" if ref else "")),
            name="consolidation_vs_time",
        )]


def _beam_force_table(beam_forces, nodes, title, name, stage=None,
                      printed=True):
    """Beam element forces with the coordinates of each end (from the mesh
    nodes the result carries), one row per element (per stage when the
    rows come from several stages: ``stage`` gives the label per row)."""
    from geotech_common.tables import CalcTable

    pts = np.asarray(nodes, dtype=float) if nodes is not None else None

    def xy(i):
        if pts is None or i >= len(pts):
            return [None, None]
        return [float(pts[i, 0]), float(pts[i, 1])]

    columns = [("Element", ""), ("x i", "m"), ("y i", "m"), ("x j", "m"),
               ("y j", "m"), ("Axial force N i", "kN/m"),
               ("Shear force V i", "kN/m"), ("Bending moment M i", "kN-m/m"),
               ("Axial force N j", "kN/m"), ("Shear force V j", "kN/m"),
               ("Bending moment M j", "kN-m/m")]
    if stage is not None:
        columns = ["Stage"] + columns
    rows = []
    for k, bf in enumerate(beam_forces):
        row = ([stage[k]] if stage is not None else []) + [
            bf.element_index + 1] + xy(bf.node_i) + xy(bf.node_j) + [
            bf.axial_i, bf.shear_i, bf.moment_i,
            bf.axial_j, bf.shear_j, bf.moment_j]
        rows.append(row)
    return CalcTable(
        title=title, columns=columns, rows=rows,
        note=("Euler-Bernoulli beam elements coupled to the soil mesh, "
              "forces per metre run at each end node i and j (y up)"),
        name=name, printed=printed)


@dataclass
class FEMResult:
    """Results from a 2D FEM analysis.

    Attributes
    ----------
    analysis_type : str
        Type of analysis ('elastic', 'elastoplastic', 'srm').
    n_nodes : int
    n_elements : int
    max_displacement_m : float
        Maximum displacement magnitude (m).
    max_displacement_x_m : float
    max_displacement_y_m : float
    max_sigma_xx_kPa : float
    max_sigma_yy_kPa : float
    max_tau_xy_kPa : float
    min_sigma_yy_kPa : float
        Maximum compressive vertical stress (kPa).
    FOS : float or None
        Factor of safety (SRM only).
    n_yielded_elements : int
        Number of elements that yielded (MC only).
    converged : bool
    warnings : list of str
    """
    analysis_type: str = "elastic"
    n_nodes: int = 0
    n_elements: int = 0
    max_displacement_m: float = 0.0
    max_displacement_x_m: float = 0.0
    max_displacement_y_m: float = 0.0
    max_sigma_xx_kPa: float = 0.0
    max_sigma_yy_kPa: float = 0.0
    max_tau_xy_kPa: float = 0.0
    min_sigma_yy_kPa: float = 0.0
    FOS: Optional[float] = None
    n_yielded_elements: int = 0
    converged: bool = True
    n_srf_trials: int = 0
    warnings: List[str] = field(default_factory=list)

    # Beam element results
    n_beam_elements: int = 0
    max_beam_moment_kNm_per_m: float = 0.0
    max_beam_shear_kN_per_m: float = 0.0
    beam_forces: Optional[List] = field(default=None, repr=False)

    # SRM convergence history (SRF vs displacement)
    srf_history: Optional[List[Dict]] = field(default=None, repr=False)

    # Strut results
    strut_forces: Optional[List[Dict]] = field(default=None, repr=False)

    # Staged excavation (analyze_excavation, since 2026-10-14)
    excavation_stages: Optional[List[Dict]] = field(default=None,
                                                    repr=False)
    wall_envelope: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)
    staged: Optional[Any] = field(default=None, repr=False)
    # Water in a staged excavation (GM-73, 2026-10-15)
    excavation_water: Optional[str] = None
    wall_water_pressures: Optional[List[Dict]] = field(default=None,
                                                       repr=False)

    # Raw arrays (not serialized to dict)
    nodes: Optional[np.ndarray] = field(default=None, repr=False)
    elements: Optional[np.ndarray] = field(default=None, repr=False)
    displacements: Optional[np.ndarray] = field(default=None, repr=False)
    stresses: Optional[np.ndarray] = field(default=None, repr=False)
    strains: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  2D FEM ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Analysis type: {self.analysis_type}",
            f"  Mesh: {self.n_nodes} nodes, {self.n_elements} elements",
            f"  Converged: {self.converged}",
            "",
            f"  Max displacement: {self.max_displacement_m:.4f} m",
            f"    ux_max: {self.max_displacement_x_m:.4f} m",
            f"    uy_max: {self.max_displacement_y_m:.4f} m",
            "",
            f"  Stress range:",
            f"    sigma_xx: {self.max_sigma_xx_kPa:.1f} kPa",
            f"    sigma_yy: {self.min_sigma_yy_kPa:.1f} to "
            f"{self.max_sigma_yy_kPa:.1f} kPa",
            f"    tau_xy max: {self.max_tau_xy_kPa:.1f} kPa",
        ]
        if self.FOS is not None:
            lines.extend([
                "",
                f"  Factor of Safety (SRM): {self.FOS:.3f}",
                f"  SRF trials: {self.n_srf_trials}",
            ])
        if self.n_yielded_elements > 0:
            lines.append(f"  Yielded elements: {self.n_yielded_elements}")
        if self.n_beam_elements > 0:
            lines.extend([
                "",
                f"  Beam elements: {self.n_beam_elements}",
                f"    Max moment: {self.max_beam_moment_kNm_per_m:.2f} kN*m/m",
                f"    Max shear: {self.max_beam_shear_kN_per_m:.2f} kN/m",
            ])
        if self.strut_forces:
            lines.extend(["", "  Struts:"])
            for sf in self.strut_forces:
                lines.append(
                    f"    Depth {sf['depth_m']:.1f} m: "
                    f"F = {sf['force_kN_per_m']:.2f} kN/m "
                    f"(k = {sf['stiffness_kN_per_m']:.0f} kN/m/m)")
        if self.excavation_stages:
            lines.extend(["", "  Excavation stages (u_x > 0 toward the "
                          "excavation):"])
            for st in self.excavation_stages:
                lines.append(
                    f"    Stage {st['stage']}: {st['name']}: head u_x "
                    f"{st['wall_head_ux_m'] * 1000:.1f} mm, M_max "
                    f"{st['max_beam_moment_kNm_per_m']:.1f} kN*m/m, V_max "
                    f"{st['max_beam_shear_kN_per_m']:.1f} kN/m"
                    + ("" if st['converged'] else " (NOT CONVERGED)"))
                if 'tip_pore_pressure_kPa' in st:
                    lines.append(
                        f"      water: u at the tip "
                        f"{st['tip_pore_pressure_kPa']:.1f} kPa, net at the "
                        f"excavation level "
                        f"{st['net_water_pressure_at_excavation_level_kPa']:.1f}"
                        f" kPa, upward gradient in front "
                        f"{st['upward_gradient_in_front']:.2f} (critical "
                        f"{st['critical_gradient']:.2f})")
        if self.wall_envelope:
            env = self.wall_envelope
            lines.append(
                f"    All stages: M_max "
                f"{env['max_beam_moment_kNm_per_m']:.1f} kN*m/m"
                + (f" at {env['moment_depth_m']:.2f} m depth"
                   if env.get('moment_depth_m') is not None else "")
                + f" ({env['moment_stage']}), V_max "
                f"{env['max_beam_shear_kN_per_m']:.1f} kN/m"
                + (f" at {env['shear_depth_m']:.2f} m depth"
                   if env.get('shear_depth_m') is not None else "")
                + f" ({env['shear_stage']})")
        if self.notes:
            lines.append("")
            for n in self.notes:
                lines.append(f"  NOTE: {n}")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "analysis_type": self.analysis_type,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "converged": self.converged,
            "max_displacement_m": round(self.max_displacement_m, 6),
            "max_displacement_x_m": round(self.max_displacement_x_m, 6),
            "max_displacement_y_m": round(self.max_displacement_y_m, 6),
            "max_sigma_xx_kPa": round(self.max_sigma_xx_kPa, 2),
            "max_sigma_yy_kPa": round(self.max_sigma_yy_kPa, 2),
            "min_sigma_yy_kPa": round(self.min_sigma_yy_kPa, 2),
            "max_tau_xy_kPa": round(self.max_tau_xy_kPa, 2),
            "warnings": self.warnings,
        }
        if self.FOS is not None:
            d["FOS"] = self.FOS
            d["n_srf_trials"] = self.n_srf_trials
        if self.n_yielded_elements > 0:
            d["n_yielded_elements"] = self.n_yielded_elements
        if self.n_beam_elements > 0:
            d["n_beam_elements"] = self.n_beam_elements
            d["max_beam_moment_kNm_per_m"] = round(
                self.max_beam_moment_kNm_per_m, 2)
            d["max_beam_shear_kN_per_m"] = round(
                self.max_beam_shear_kN_per_m, 2)
            if self.beam_forces:
                d["beam_forces"] = [bf.to_dict() for bf in self.beam_forces]
        if self.strut_forces:
            d["strut_forces"] = self.strut_forces
        if self.excavation_stages:
            d["excavation_stages"] = [dict(st) for st in
                                      self.excavation_stages]
        if self.wall_envelope:
            d["wall_envelope"] = dict(self.wall_envelope)
        if self.excavation_water is not None:
            d["excavation_water"] = self.excavation_water
        if self.wall_water_pressures:
            d["wall_water_pressures"] = [dict(r) for r in
                                         self.wall_water_pressures]
        if self.notes:
            d["notes"] = list(self.notes)
        return d

    def get_tables(self) -> list:
        """PLAXIS-style tables: the beam (wall) element forces with their
        positions (printed), and as CSV only (the calc package prints its
        own versions) the strength-reduction trials and the strut forces.
        Values are this result's own.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        if self.beam_forces:
            tables.append(_beam_force_table(
                self.beam_forces, self.nodes, "Beam element forces",
                "beam_forces"))
        if self.srf_history:
            tables.append(CalcTable(
                title="Strength reduction trials",
                columns=[("SRF", "", ".3f"), ("Maximum displacement", "m"),
                         ("Dimensionless displacement Eδ/(γH²)",
                          ""), "Converged", "Failed", "Iterations",
                         "Settled at the iteration ceiling",
                         ("Final residual", "")],
                rows=[[h.get("srf"), h.get("max_disp_m"),
                       h.get("dimensionless_disp"), h.get("converged"),
                       h.get("failed"), h.get("n_iter"), h.get("settled"),
                       h.get("final_residual")]
                      for h in self.srf_history],
                note=(f"Griffiths & Lane (1999) strength reduction; "
                      f"FOS = {self.FOS if self.FOS is not None else 'n/a'}"
                      "; trials in the order run"),
                name="srf_trials", printed=False,
            ))
        if self.strut_forces:
            tables.append(CalcTable(
                title="Strut forces",
                columns=[("Depth", "m"), ("Stiffness", "kN/m/m"),
                         ("Force", "kN/m"), "Node"],
                rows=[[s.get("depth_m"), s.get("stiffness_kN_per_m"),
                       s.get("force_kN_per_m"), s.get("node_id")]
                      for s in self.strut_forces],
                note=("F = k x the wall's horizontal movement since the "
                      "strut was installed, per metre run (positive = the "
                      "strut holds the wall back)"),
                name="strut_forces", printed=False,
            ))
        if self.excavation_stages:
            tables.append(CalcTable(
                title="Excavation stages",
                columns=["Stage", "Phase", ("Excavation depth", "m"),
                         "Converged", ("Wall head u_x", "m"),
                         ("Largest wall u_x", "m"),
                         ("at depth", "m"),
                         ("Largest wall moment", "kN-m/m"),
                         ("Largest wall shear", "kN/m"),
                         ("Strut forces", "kN/m")],
                rows=[[st.get("stage"), st.get("name"),
                       st.get("excavation_depth_m"), st.get("converged"),
                       st.get("wall_head_ux_m"), st.get("max_wall_ux_m"),
                       st.get("max_wall_ux_depth_m"),
                       st.get("max_beam_moment_kNm_per_m"),
                       st.get("max_beam_shear_kN_per_m"),
                       ", ".join(f"{f:.1f}" for f in
                                 st.get("strut_forces_kN_per_m") or [])
                       or "-"]
                      for st in self.excavation_stages],
                note=("Staged excavation: movements from the end of the "
                      "in-situ (K0) stage, u_x positive toward the "
                      "excavation; wall forces at the end of each stage"),
                name="excavation_stages",
            ))
            if any('tip_pore_pressure_kPa' in st
                   for st in self.excavation_stages):
                tables.append(CalcTable(
                    title="Water on the wall by stage",
                    columns=["Stage", ("Excavation depth", "m"),
                             ("Water inside, depth", "m"),
                             ("Pore pressure at the tip", "kPa"),
                             ("Total head at the tip", "m"),
                             ("Net water pressure at the excavation level",
                              "kPa"),
                             ("Upward gradient in front", ""),
                             ("Critical gradient", "")],
                    rows=[[st.get("stage"), st.get("excavation_depth_m"),
                           st.get("water_inside_depth_m"),
                           st.get("tip_pore_pressure_kPa"),
                           st.get("tip_head_m"),
                           st.get("net_water_pressure_at_excavation_level_"
                                  "kPa"),
                           st.get("upward_gradient_in_front"),
                           st.get("critical_gradient")]
                          for st in self.excavation_stages],
                    note=(f"excavation_water = {self.excavation_water!r}; "
                          "total head h = y + u / gamma_w, datum the "
                          "ground; net = retained side minus excavation "
                          "side; gradient from the wall's tip up to the "
                          "excavation level"),
                    name="excavation_water",
                ))
        if self.wall_water_pressures:
            tables.append(CalcTable(
                title="Water pressures on the wall",
                columns=[("Depth", "m"), ("Retained side", "kPa"),
                         ("Excavation side", "kPa"), ("Net", "kPa")],
                rows=[[r.get("depth_m"), r.get("retained_side_kPa"),
                       r.get("excavation_side_kPa"), r.get("net_kPa")]
                      for r in self.wall_water_pressures],
                note=("End of the last stage; net = retained minus "
                      "excavation side (positive pushes the wall toward "
                      "the excavation); the excavation side includes the "
                      "water standing in the excavation"),
                name="wall_water_pressures", printed=False,
            ))
        return tables


@dataclass
class PhaseResult:
    """Results from one phase of a staged construction analysis.

    Attributes
    ----------
    phase_name : str
    phase_index : int
    n_active_elements : int
    n_active_beams : int
    converged : bool
    max_displacement_m : float
    max_displacement_x_m : float
    max_displacement_y_m : float
    max_sigma_xx_kPa : float
    max_sigma_yy_kPa : float
    max_tau_xy_kPa : float
    min_sigma_yy_kPa : float
    n_beam_elements : int
    max_beam_moment_kNm_per_m : float
    max_beam_shear_kN_per_m : float

    Displacement bases (2026-10-09). ``max_displacement_*`` and the
    ``displacements`` array are CUMULATIVE from the start of the analysis
    (every stage, the gravity stage included) — their meaning is unchanged.
    Beside them, each stage reports three movement summaries
    (``fem2d.results.movement_summary`` dicts, over the nodes of the
    stage's active elements):
    delta_displacement : this stage's own movement (end of this stage
        minus end of the previous one; a node placed in this stage moves
        from its placement).
    cumulative_displacement : from the start of the analysis.
    displacement_since_reference : from the end of the reference stage
        (``reference_stage``), or from the start when no reset precedes this
        stage — the PLAXIS "reset displacements to zero" convention.
    The matching arrays are ``delta_displacements`` and
    ``displacements_since_reference``.

    spring_forces : per active spring (``analyze_staged(springs=...)``):
        k x its stretch since placement ('force_x_kN_per_m',
        'force_y_kN_per_m'; positive = the node moved +x / +y relative to
        ground or to ``node_b``), with 'placed_in_phase'.
    settled_steps : load steps accepted at the iteration ceiling with the
        displacements standing still (``settled_ratio``), each with its
        residual ratio; ``final_residual`` is the last step's.
    """
    phase_name: str = "Phase"
    phase_index: int = 0
    n_active_elements: int = 0
    n_active_beams: int = 0
    converged: bool = True
    max_displacement_m: float = 0.0
    max_displacement_x_m: float = 0.0
    max_displacement_y_m: float = 0.0
    max_sigma_xx_kPa: float = 0.0
    max_sigma_yy_kPa: float = 0.0
    max_tau_xy_kPa: float = 0.0
    min_sigma_yy_kPa: float = 0.0
    n_beam_elements: int = 0
    max_beam_moment_kNm_per_m: float = 0.0
    max_beam_shear_kN_per_m: float = 0.0
    beam_forces: Optional[List] = field(default=None, repr=False)
    spring_forces: Optional[List[Dict[str, Any]]] = field(
        default=None, repr=False)
    settled_steps: Optional[List[Dict[str, Any]]] = field(
        default=None, repr=False)
    final_residual: Optional[float] = None
    delta_displacement: Optional[Dict[str, Any]] = None
    cumulative_displacement: Optional[Dict[str, Any]] = None
    displacement_since_reference: Optional[Dict[str, Any]] = None
    reference_stage: str = ""
    reference_stage_index: Optional[int] = None

    # Raw arrays (not serialized to dict)
    displacements: Optional[np.ndarray] = field(default=None, repr=False)
    delta_displacements: Optional[np.ndarray] = field(
        default=None, repr=False)
    displacements_since_reference: Optional[np.ndarray] = field(
        default=None, repr=False)
    active_node_mask: Optional[np.ndarray] = field(default=None, repr=False)
    stresses: Optional[np.ndarray] = field(default=None, repr=False)
    strains: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"  Phase {self.phase_index}: {self.phase_name}",
            f"    Active elements: {self.n_active_elements}"
            f", beams: {self.n_active_beams}",
            f"    Converged: {self.converged}",
            f"    Max displacement (cumulative): "
            f"{self.max_displacement_m:.4f} m",
        ]
        if self.delta_displacement is not None:
            dd = self.delta_displacement
            lines.append(
                f"    This stage: settlement {dd['settlement_m'] * 1000:.2f}"
                f" mm, heave {dd['heave_m'] * 1000:.2f} mm, horizontal "
                f"{dd['horizontal_m'] * 1000:.2f} mm")
        if self.displacement_since_reference is not None:
            ds = self.displacement_since_reference
            lines.append(
                f"    Since {self.reference_stage}: settlement "
                f"{ds['settlement_m'] * 1000:.2f} mm, |u| "
                f"{ds['total_m'] * 1000:.2f} mm")
        lines.append(
            f"    sigma_yy range: {self.min_sigma_yy_kPa:.1f} to "
            f"{self.max_sigma_yy_kPa:.1f} kPa")
        if self.n_beam_elements > 0:
            lines.append(
                f"    Max moment: {self.max_beam_moment_kNm_per_m:.2f}"
                f" kN*m/m")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "phase_name": self.phase_name,
            "phase_index": self.phase_index,
            "n_active_elements": self.n_active_elements,
            "n_active_beams": self.n_active_beams,
            "converged": self.converged,
            "max_displacement_m": round(self.max_displacement_m, 6),
            "max_displacement_x_m": round(self.max_displacement_x_m, 6),
            "max_displacement_y_m": round(self.max_displacement_y_m, 6),
            "max_sigma_xx_kPa": round(self.max_sigma_xx_kPa, 2),
            "max_sigma_yy_kPa": round(self.max_sigma_yy_kPa, 2),
            "min_sigma_yy_kPa": round(self.min_sigma_yy_kPa, 2),
            "max_tau_xy_kPa": round(self.max_tau_xy_kPa, 2),
        }
        if self.delta_displacement is not None:
            d["delta_displacement"] = dict(self.delta_displacement)
        if self.cumulative_displacement is not None:
            d["cumulative_displacement"] = dict(self.cumulative_displacement)
        if self.displacement_since_reference is not None:
            d["displacement_since_reference"] = dict(
                self.displacement_since_reference)
            d["reference_stage"] = self.reference_stage
        if self.n_beam_elements > 0:
            d["n_beam_elements"] = self.n_beam_elements
            d["max_beam_moment_kNm_per_m"] = round(
                self.max_beam_moment_kNm_per_m, 2)
            d["max_beam_shear_kN_per_m"] = round(
                self.max_beam_shear_kN_per_m, 2)
            if self.beam_forces:
                d["beam_forces"] = [bf.to_dict() for bf in self.beam_forces]
        if self.spring_forces:
            d["spring_forces"] = [dict(s) for s in self.spring_forces]
        if self.settled_steps:
            d["settled_steps"] = [dict(s) for s in self.settled_steps]
        return d


@dataclass
class StagedConstructionResult:
    """Container for all phases of a staged construction analysis.

    Attributes
    ----------
    n_phases : int
    n_nodes : int
    n_elements : int
    converged : bool — True if all phases converged.
    phases : list of PhaseResult
    displacement_reference : dict — which stage movements are measured
        from (``reset_after_stage``, its name, and whether the user or the
        default chose it).
    judgment : dict or None — the reference-stage choice when it was a
        default AND it changes the reported numbers materially.
    notes : list of str
    """
    n_phases: int = 0
    n_nodes: int = 0
    n_elements: int = 0
    converged: bool = True
    phases: List[PhaseResult] = field(default_factory=list)
    displacement_reference: Optional[Dict[str, Any]] = None
    judgment: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)

    # Shared mesh (not serialized)
    nodes: Optional[np.ndarray] = field(default=None, repr=False)
    elements: Optional[np.ndarray] = field(default=None, repr=False)

    def get_phase(self, key):
        """Get a phase by name (str) or index (int).

        Parameters
        ----------
        key : str or int — phase name or 0-based index.

        Returns
        -------
        PhaseResult

        Raises
        ------
        KeyError / IndexError if not found.
        """
        if isinstance(key, int):
            return self.phases[key]
        for p in self.phases:
            if p.phase_name == key:
                return p
        raise KeyError(f"Phase '{key}' not found")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  STAGED CONSTRUCTION RESULTS",
            "=" * 60,
            "",
            f"  Mesh: {self.n_nodes} nodes, {self.n_elements} elements",
            f"  Phases: {self.n_phases}",
            f"  All converged: {self.converged}",
        ]
        if self.displacement_reference:
            lines.append(
                f"  Displacements reset after: "
                f"{self.displacement_reference.get('measured_from')}")
        lines.append("")
        for p in self.phases:
            lines.append(p.summary())
            lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "n_phases": self.n_phases,
            "n_nodes": self.n_nodes,
            "n_elements": self.n_elements,
            "converged": self.converged,
            "phases": [p.to_dict() for p in self.phases],
        }
        if self.displacement_reference:
            d["displacement_reference"] = dict(self.displacement_reference)
        if self.judgment:
            d["judgment"] = self.judgment
        if self.notes:
            d["notes"] = list(self.notes)
        return d

    def get_tables(self) -> list:
        """PLAXIS-style tables: the movements of every stage (this stage's
        own, cumulative, and since the reference stage; printed) and, as
        CSV, the beam element forces of every stage. Values are this
        result's own (y up: a negative settlement is downward).

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        def m(d, key):
            return d.get(key) if d else None

        rows = []
        for p in self.phases:
            dd, cd = p.delta_displacement, p.cumulative_displacement
            sd = p.displacement_since_reference
            rows.append([
                p.phase_index + 1, p.phase_name, p.converged,
                m(dd, "settlement_m"), m(dd, "heave_m"),
                m(dd, "horizontal_m"), m(dd, "total_m"),
                m(cd, "settlement_m"), m(cd, "horizontal_m"),
                m(cd, "total_m"),
                m(sd, "settlement_m"), m(sd, "total_m")])
        tables = []
        if rows:
            ref = (self.displacement_reference or {}).get(
                "measured_from", "the start of the analysis")
            tables.append(CalcTable(
                title="Movements by stage",
                columns=["Stage", "Phase", "Converged",
                         ("This stage: settlement", "m"),
                         ("This stage: heave", "m"),
                         ("This stage: horizontal", "m"),
                         ("This stage: largest |u|", "m"),
                         ("Cumulative settlement", "m"),
                         ("Cumulative horizontal", "m"),
                         ("Cumulative largest |u|", "m"),
                         ("Settlement since reference", "m"),
                         ("Largest |u| since reference", "m")],
                rows=rows,
                note=("Extremes over each stage's active nodes "
                      "(settlement = most negative vertical movement, "
                      "heave = most positive, horizontal = largest "
                      f"magnitude, signed); the reference is {ref}"),
                name="stage_movements",
            ))
        beams, labels = [], []
        for p in self.phases:
            for bf in p.beam_forces or []:
                beams.append(bf)
                labels.append(p.phase_index + 1)
        if beams:
            tables.append(_beam_force_table(
                beams, self.nodes, "Beam element forces by stage",
                "beam_forces_by_stage", stage=labels, printed=False))
        return tables
