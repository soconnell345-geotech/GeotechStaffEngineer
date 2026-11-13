"""
Rigid cap pile group analysis.

Assembles the group stiffness matrix from individual pile stiffnesses,
solves for cap displacements under applied loads, and back-calculates
individual pile forces.

Supports vertical and battered piles under 6-DOF loading. Each pile is an
axial spring along its own axis and a lateral spring across it, pinned to
the cap (no head moment or torsional stiffness of its own).

All units are SI: kN, m, radians.

Sign convention (PG-2)
----------------------
One explicit right-hand-rule convention is used end-to-end (stiffness
assembly, solution, and pile-force back-calculation):

* Global axes (x, y, z) are RIGHT-HANDED with **z positive UP**;
  x and y are the plan coordinates of the pile heads.
* For engineering convenience the vertical force and displacement are
  input/reported **positive DOWNWARD**: ``Vz > 0`` is a downward
  (compressive) load and ``dz > 0`` is settlement. These are the
  negatives of the right-handed z-components; everything else follows
  the right-hand rule about the +x, +y, +z(up) axes.
* Consequences of the convention:
  - positive ``My`` (right-hand rule about +y) adds COMPRESSION to
    piles on the +x side;
  - positive ``Mx`` (right-hand rule about +x) adds TENSION (uplift)
    to piles on the +y side;
  - positive ``Mz`` twists the cap counterclockwise in plan when
    viewed from above (+z looking down).
* Rigid-cap kinematics for a pile head at (x, y, 0):
  ``ux_pile = dx - rz*y``, ``uy_pile = dy + rz*x``,
  ``settlement_pile = dz - rx*y + ry*x``.
  The same kinematic matrix B is used to assemble K = sum(B^T k B) and
  to back-calculate pile displacements/forces, so the results are
  self-consistent (equilibrium of pile forces with the applied loads).

References:
    CPGA User's Guide (ITL-89-4, Hartman et al., 1989)
    USACE EM 1110-2-2906, Chapter 4

Provenance (VERIFIED vs the EM, 2026-07-19 wiki-verification): the Ch. 4
stiffness method (para. 4-5b, Saul/CPGA approach) as printed in the
owner-library EM — {q}_i = [B]_i{u}_i and [K] = sum([K]_i) with per-pile
stiffness transformed to global coordinates (EM p. 4-29), rigid-body load
distribution (p. 4-30), axial b33 = C33*AE/L (p. 4-31), torsional
b66 = C66*JG/L (p. 4-38), plane-remains-plane cap kinematics (p. 4-45) — is
the SAME formulation as this engine's K = sum(B^T k B) congruence-transform
notation (EQUIVALENT-NOTATION). One simplification stays: no per-pile
rotational or torsional head stiffness (the EM's b44, b55, b66), so the
piles are pinned to the cap.
Ledger: module_work/wiki_verification/wave7_em2906_pti_mo.md.

Batter (GM-319): each pile's two springs are carried to the global axes
whole, k = kl*I + (ka - kl)*l*l^T with l the pile's direction cosines, so
the terms between the horizontal and vertical head components (kxz, kyz)
are assembled with the rest. Before GM-319 they were left out, and a
battered pile's axial force did not balance the applied loads. With
kl = 0 the piles carry axial load only, UFC 3-220-20 Sec. 6-6.3's
assumption for batter piles (Fig. 6-20), and a statically determinate
group returns the forces statics gives.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from pile_group.pile_layout import GroupPile


@dataclass
class GroupLoad:
    """Applied loads on the pile cap, at the cap elevation.

    ``analyze_vertical_group_simple`` takes them at the centroid of the
    piles and measures each pile's x and y from it (GM-319 review);
    ``analyze_group_6dof`` takes them at the origin of the coordinates,
    which its rigid-cap kinematics use as given. The two are one point
    for a layout centred on the origin. See the module docstring for the
    full sign convention (right-handed axes, z up; Vz positive downward).

    Parameters
    ----------
    Vx : float
        Horizontal force in X-direction (kN).
    Vy : float
        Horizontal force in Y-direction (kN).
    Vz : float
        Vertical force (kN). Positive DOWNWARD (compression).
    Mx : float
        Moment about the +x axis (kN-m), positive per the right-hand
        rule with z UP: positive Mx puts piles on the +y side in
        TENSION (uplift).
    My : float
        Moment about the +y axis (kN-m), right-hand rule: positive My
        puts piles on the +x side in COMPRESSION.
    Mz : float
        Torsion about the +z (up) axis (kN-m), right-hand rule:
        positive Mz twists the cap counterclockwise in plan viewed
        from above.
    """
    Vx: float = 0.0
    Vy: float = 0.0
    Vz: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mz: float = 0.0


@dataclass
class PileGroupResult:
    """Results from pile group analysis.

    Attributes
    ----------
    cap_displacements : dict
        Cap displacements {dx, dy, dz, rx, ry, rz} in m and radians. Empty
        for the simplified vertical method run without pile stiffness: it
        works out no displacement (``notes`` says so).
    notes : list of str
        What the analysis took or left out, in words.
    centroid : dict or None
        Simplified vertical method only: the piles' centroid
        ``{"x": .., "y": ..}`` (m), where it takes the loads to act.
    warnings : list of str
        What the caller must read before using the numbers: from the
        simplified vertical method, a centroid more than 1 mm from the
        origin of the coordinates (the loads are taken there, not at
        the origin).
    pile_forces : list of dict
        Per-pile forces and utilization ratios.
    max_compression : float
        Maximum compression force in any pile (kN).
    max_tension : float
        Maximum tension force in any pile (kN, positive = tension).
    max_utilization : float
        Maximum utilization ratio (demand/capacity).
    pile_head_response : list of dict
        6-DOF analysis only: per pile, the head displacements from the
        rigid-cap kinematics (``ux_m``, ``uy_m``, settlement ``w_m``,
        ``axial_displacement_m``) and the head force in the global axes
        (``Fx_kN``, ``Fy_kN``, ``Fz_kN`` with z down as ``Vz``; the forces
        sum to the applied loads), plus the unrounded ``axial_kN``. Not in
        ``to_dict``; read by :meth:`get_tables`.
    """
    cap_displacements: Dict[str, float] = field(default_factory=dict)
    pile_forces: List[Dict[str, Any]] = field(default_factory=list)
    max_compression: float = 0.0
    max_tension: float = 0.0
    max_utilization: float = 0.0
    n_piles: int = 0
    pile_head_response: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    centroid: Optional[Dict[str, float]] = None
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PILE GROUP ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Number of piles: {self.n_piles}",
            "",
            "  Cap Displacements:",
        ]
        d = self.cap_displacements
        lines.extend(["    not calculated (no pile stiffness given)"]
                     if not d else [
            f"    dx = {d.get('dx', 0)*1000:.2f} mm",
            f"    dy = {d.get('dy', 0)*1000:.2f} mm",
            f"    dz = {d.get('dz', 0)*1000:.2f} mm",
            f"    rx = {d.get('rx', 0)*1000:.4f} mrad",
            f"    ry = {d.get('ry', 0)*1000:.4f} mrad",
            f"    rz = {d.get('rz', 0)*1000:.4f} mrad",
        ])
        lines.extend([
            "",
            f"  Max compression:   {self.max_compression:>10,.1f} kN",
            f"  Max tension:       {self.max_tension:>10,.1f} kN",
            f"  Max utilization:   {self.max_utilization:>10.2f}",
            "",
            "  Per-Pile Forces:",
        ])
        for pf in self.pile_forces:
            util_str = f"{pf.get('utilization', 0):.2f}" if pf.get('utilization') else "N/A"
            lines.append(
                f"    {pf['label']:>8}: axial={pf['axial_kN']:>8.1f} kN, "
                f"util={util_str}"
            )
        for note in self.notes:
            lines.extend(["", f"  Note: {note}"])
        for warning in self.warnings:
            lines.extend(["", f"  WARNING: {warning}"])
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def plot_pile_layout(self, ax=None, show=True, **kwargs):
        """Plot plan view of pile group with piles colored by utilization.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not self.pile_forces:
            raise ValueError("No pile force data available for plotting.")
        from geotech_common.plotting import get_pyplot, setup_engineering_plot
        plt = get_pyplot()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        xs = [pf['x_m'] for pf in self.pile_forces]
        ys = [pf['y_m'] for pf in self.pile_forces]
        utils = [pf.get('utilization', 0) for pf in self.pile_forces]
        labels = [pf.get('label', '') for pf in self.pile_forces]
        sc = ax.scatter(xs, ys, c=utils, cmap='RdYlGn_r', edgecolors='black',
                        linewidth=1, s=200, vmin=0, vmax=max(max(utils), 1.0),
                        **kwargs)
        plt.colorbar(sc, ax=ax, label='Utilization Ratio')
        for x, y, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (x, y), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=8)
        ax.set_aspect('equal')
        setup_engineering_plot(ax, "Pile Group Layout",
                              "X (m)", "Y (m)")
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "n_piles": self.n_piles,
            "cap_displacements": self.cap_displacements,
            "max_compression_kN": round(self.max_compression, 1),
            "max_tension_kN": round(self.max_tension, 1),
            "max_utilization": round(self.max_utilization, 3),
            "pile_forces": self.pile_forces,
            "notes": list(self.notes),
        }
        if self.centroid is not None:
            out["centroid_m"] = dict(self.centroid)
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out

    def get_tables(self) -> list:
        """GROUP-style tables: the force at each pile head, the pile-head
        displacements (6-DOF analysis), and the cap displacements (CSV
        only; the calc package prints them in its Load Distribution
        section). Values are this result's own, in its units.

        The pile-head force table is CSV-only for the simplified vertical
        method, whose per-pile axial forces the package already prints.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        tables = []
        # pile_head_response runs in the same pile order as pile_forces
        six_dof = len(self.pile_head_response) == len(self.pile_forces) > 0
        if self.pile_forces:
            columns = ["Pile", ("x", "m"), ("y", "m"),
                       ("Axial force P", "kN")]
            if six_dof:
                columns += [("Head force Fx", "kN"), ("Head force Fy", "kN"),
                            ("Head force Fz", "kN")]
            columns.append("Utilization P/capacity")
            rows = []
            for i, pf in enumerate(self.pile_forces):
                h = self.pile_head_response[i] if six_dof else None
                row = [pf["label"], pf["x_m"], pf["y_m"],
                       h["axial_kN"] if h else pf["axial_kN"]]
                if six_dof:
                    row += [h["Fx_kN"], h["Fy_kN"], h["Fz_kN"]] if h else \
                        [None, None, None]
                row.append(pf.get("utilization") or None)
                rows.append(row)
            note = ("Rigid-cap distribution (CPGA stiffness method, EM "
                    "1110-2-2906 Ch. 4); P positive in compression")
            if six_dof:
                note += ("; Fx, Fy, Fz: the force each pile takes from the "
                         "cap in the global axes, z down as Vz, summing to "
                         "the applied Vx, Vy, Vz")
            else:
                note = ("Simplified elastic distribution P = Vz/n + "
                        "My·x/Σx² - Mx·y/Σy², x and y from the piles' "
                        "centroid (with the product term Σxy where the "
                        "layout is not symmetric about its centroidal "
                        "axes); P positive in compression")
            tables.append(CalcTable(
                title="Pile-head forces", columns=columns, rows=rows,
                note=note, name="pile_head_forces", printed=six_dof,
            ))
        if six_dof:
            tables.append(CalcTable(
                title="Pile-head displacements",
                columns=["Pile", ("Displacement ux", "m"),
                         ("Displacement uy", "m"), ("Settlement w", "m"),
                         ("Axial displacement δa", "m")],
                rows=[[h["label"], h["ux_m"], h["uy_m"], h["w_m"],
                       h["axial_displacement_m"]]
                      for h in self.pile_head_response],
                note=("From the cap displacements by the rigid-cap "
                      "kinematics (ux = dx - rz·y, uy = dy + rz·x, "
                      "w = dz - rx·y + ry·x); w positive down; "
                      "δa along the pile axis, head toward toe"),
                name="pile_head_displacements",
            ))
        if self.cap_displacements:
            d = self.cap_displacements
            tables.append(CalcTable(
                title="Cap displacements",
                columns=["Component", "Value", "Unit"],
                rows=[["dx", d.get("dx", 0.0), "m"],
                      ["dy", d.get("dy", 0.0), "m"],
                      ["dz (settlement, positive down)", d.get("dz", 0.0),
                       "m"],
                      ["rx", d.get("rx", 0.0), "rad"],
                      ["ry", d.get("ry", 0.0), "rad"],
                      ["rz", d.get("rz", 0.0), "rad"]],
                note="Rigid-cap displacements; rotations by the right-hand "
                     "rule about the x, y and z (up) axes",
                name="cap_displacements", printed=False,
            ))
        return tables


#: SUM(x^2) or SUM(y^2) at or below this (m^2): every pile on that line.
_ON_LINE_M2 = 1e-12

#: The simplified method warns when the piles' centroid, where it takes the
#: loads, is more than this (m) from the origin in x or y (GM-319 review).
CENTROID_TOL_M = 1e-3


def _moment_about_pile_line(name: str, at: float = 0.0) -> str:
    """Why a moment about the one line every pile stands on is refused
    (GM-319 review): piles pinned to the cap along one line give no lever
    arm about it, so the cap turns about that line freely. ``at``: the
    line's coordinate (the piles' centroid in the simplified method)."""
    coord, axis = ("y", "x") if name == "Mx" else ("x", "y")
    at = at if abs(at) > CENTROID_TOL_M else 0.0
    lever = (f"SUM({coord}^2) = 0" if at == 0.0 else
             f"SUM(({coord} {'-' if at > 0 else '+'} {abs(at):g})^2) = 0 "
             "about the piles' centroid")
    return (f"every pile is on {coord} = {at:g}, one line along the {axis} "
            f"axis, and piles pinned to the cap on one line cannot carry a "
            f"moment about that line ({lever}, no lever arm); give piles "
            f"off that line, or leave {name} out if the cap is restrained "
            "against that rotation another way")


def centroid_load_warning(xc: float, yc: float, Vz: float) -> Optional[str]:
    """The simplified method's warning for piles whose centroid (xc, yc),
    where it takes the loads, is more than ``CENTROID_TOL_M`` from the
    origin in x or y; None when it is not (GM-319 review). Statics: a load
    at (xa, ya) is the same load at the centroid with My + Vz (xa - xc)
    and Mx - Vz (ya - yc), under the module's convention (+My compresses
    the +x side, +Mx uplifts the +y side)."""
    off_x, off_y = abs(xc) > CENTROID_TOL_M, abs(yc) > CENTROID_TOL_M
    if not (off_x or off_y):
        return None
    # rounded to the micrometre, never a "-0.000"
    xc, yc = round(xc, 6) + 0.0, round(yc, 6) + 0.0

    def minus(c):
        return f"{'-' if c >= 0 else '+'} {abs(c):.3f} m"

    general, origin = [], []
    if off_x:
        general.append(f"Vz x (xa {minus(xc)}) to My (+My compresses the "
                       "+x side)")
        origin.append(f"{Vz:g} kN x ({-xc:.3f} m) = {-Vz * xc:,.1f} kN-m "
                      "to My")
    if off_y:
        general.append(f"-Vz x (ya {minus(yc)}) to Mx (+Mx uplifts the +y "
                       "side)")
        origin.append(f"-({Vz:g} kN x {-yc:.3f} m) = {Vz * yc:,.1f} kN-m "
                      "to Mx")
    text = (f"The loads are taken at the centroid of the piles, (x, y) = "
            f"({xc:.3f}, {yc:.3f}) m, not at the origin of the coordinates "
            f"given (more than {CENTROID_TOL_M * 1000:g} mm from it). A load "
            "applied at another point (xa, ya) needs its moment about the "
            f"centroid added: {' and '.join(general)}.")
    if abs(Vz) > 1e-12:
        text += (f" For Vz = {Vz:g} kN applied at the origin, add "
                 f"{' and '.join(origin)}.")
    text += (" pile_group_6dof takes its loads at the origin of the "
             "coordinates.")
    return text


#: |SUM dx dy| at or below this fraction of sqrt(SUM dx^2 SUM dy^2), about
#: the piles' centroid, is taken as zero: the plain Eq. 8-79 (GM-319
#: re-check). Rounding in a symmetric layout is far below it.
PRODUCT_TOL = 1e-9


def simple_load_plane(xs, ys, Mx: float, My: float) -> Dict[str, Any]:
    """The rigid-cap load plane of the simplified method, P_i = Vz/n +
    a dx_i + b dy_i about the piles' centroid (GM-319 review and re-check).

    Equilibrium of a rigid cap on pinned piles of equal axial stiffness:
    SUM P dx = My and SUM P dy = -Mx (+My compresses the +x side, +Mx
    uplifts the +y side), so with Sxx = SUM dx^2, Syy = SUM dy^2,
    Sxy = SUM dx dy and D = Sxx Syy - Sxy^2,
    a = (My Syy + Mx Sxy)/D and b = -(Mx Sxx + My Sxy)/D. With Sxy = 0
    (within ``PRODUCT_TOL``) it is GEC-12 Vol I Eq. 8-79,
    a = My/Sxx and b = -Mx/Syy. Piles on one line carry no moment about
    it: a non-zero one is refused (ValueError), naming the line.

    Returns dict with xc, yc, dx, dy (arrays), sxx, syy, sxy, det, a, b,
    ``product`` (True when the product term is used) and ``form``
    ("eq_8_79", "one_line" for a line not along x or y, "product_term").
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xc, yc = float(xs.mean()), float(ys.mean())
    dx, dy = xs - xc, ys - yc
    sxx, syy = float(np.sum(dx**2)), float(np.sum(dy**2))
    sxy = float(np.sum(dx * dy))
    det = sxx * syy - sxy * sxy

    # A moment about the line every pile is on has no lever arm: refused,
    # never dropped (GM-319 review; before, its term was skipped).
    for name, moment, s2, at in (("Mx", Mx, syy, yc), ("My", My, sxx, xc)):
        if abs(moment) > 1e-12 and not s2 > _ON_LINE_M2:
            raise ValueError(f"{name} = {moment:g} kN-m cannot be carried: "
                             f"{_moment_about_pile_line(name, at)}.")
    product, form = False, "eq_8_79"
    if syy <= _ON_LINE_M2:                  # one row along x (or a point)
        a = My / sxx if sxx > _ON_LINE_M2 else 0.0
        b = 0.0
    elif sxx <= _ON_LINE_M2:                # one column along y
        a, b = 0.0, -Mx / syy
    elif abs(sxy) <= PRODUCT_TOL * math.sqrt(sxx * syy):
        a, b = My / sxx, -Mx / syy          # Eq. 8-79
    else:
        # the smaller principal second moment: at or below the one-line
        # tolerance, every pile is on one line, not along x or y
        lam = 0.5 * ((sxx + syy) - math.hypot(sxx - syy, 2.0 * sxy))
        if lam <= _ON_LINE_M2:
            s = sxx + syy
            ux, uy = math.sqrt(sxx / s), math.copysign(math.sqrt(syy / s),
                                                       sxy)
            about = Mx * ux + My * uy       # the moment about that line
            if abs(about) > 1e-9 * max(1.0, abs(Mx), abs(My)):
                raise ValueError(
                    f"Mx = {Mx:g} and My = {My:g} kN-m cannot be carried: "
                    "every pile is on one line through the piles' centroid "
                    f"(direction ({ux:.3f}, {uy:.3f}) in plan), and piles "
                    "pinned to the cap on one line cannot carry a moment "
                    f"about that line (Mx x {ux:.3f} + My x {uy:.3f} = "
                    f"{about:g} kN-m is about it, no lever arm); give piles "
                    "off that line, or a moment across it only.")
            c = (My * ux - Mx * uy) / s
            a, b = c * ux, c * uy
            form = "one_line"
        else:
            product, form = True, "product_term"
            a = (My * syy + Mx * sxy) / det
            b = -(Mx * sxx + My * sxy) / det
    return {"xc": xc, "yc": yc, "dx": dx, "dy": dy, "sxx": sxx, "syy": syy,
            "sxy": sxy, "det": det, "a": a, "b": b, "product": product,
            "form": form}


def analyze_vertical_group_simple(
    piles: List[GroupPile],
    load: GroupLoad,
) -> PileGroupResult:
    """Simplified elastic analysis for vertical piles only.

    Uses the standard pile group equation, signed per the module's
    right-hand-rule convention (z up; Vz/settlement positive down):

        Pi = Vz/n + My*xi/SUM(xi^2) - Mx*yi/SUM(yi^2)

    so positive My compresses the +x side and positive Mx uplifts the
    +y side — consistent with ``analyze_group_6dof``. GEC-12 Vol I
    Sec. 8.8 Step 3 prints it as Eq. 8-79 (p. 468), as the maximum pile
    load with the terms added (signs here by this module's convention),
    the vertical load including the cap's and the soil's weight (Wc, Ws;
    Vz here is all of it), and x and y measured from the centre of the
    column, which carries the loads.

    Where the piles are not symmetric about their centroidal axes
    (SUM dx dy not zero, beyond ``PRODUCT_TOL``: a group that has lost a
    corner pile, an L-shaped cap), Eq. 8-79 does not give back Mx and My,
    so the general rigid-cap form is used (GM-319 re-check, the lead's
    call): P = Vz/n + a dx + b dy, a = (My Syy + Mx Sxy)/D,
    b = -(Mx Sxx + My Sxy)/D, D = Sxx Syy - Sxy^2 (``simple_load_plane``);
    Eq. 8-79 is its case Sxy = 0, so a symmetric layout keeps its numbers.
    Basis: statics, the equilibrium of a rigid cap on pinned piles of equal
    axial stiffness; the result's ``notes`` say so.

    Vz, Mx and My are taken to act at the centroid of the piles (xc, yc),
    each pile's xi and yi measured from it (GM-319 review: they were
    measured from the origin of the coordinates, so a layout not centred
    on it gave forces that did not sum to Vz). Statics of the rigid cap:
    with SUM(xi - xc) = SUM(yi - yc) = 0 the forces sum to Vz, and with
    the product term where it is not zero they give back Mx and My. A
    load applied at another point (xa, ya) needs its
    moment about the centroid added, Vz (xa - xc) to My and
    -Vz (ya - yc) to Mx. A centroid more than ``CENTROID_TOL_M`` (1 mm)
    from the origin in x or y is warned of in ``warnings``; the result's
    ``centroid`` and ``notes`` give it either way. A layout centred on
    the origin gives the same forces as before.

    This method ignores lateral loads (Vx, Vy) and torsion (Mz); a
    ``UserWarning`` is issued if any of them is nonzero (PG-3). Use the
    general 6-DOF method for battered piles or combined loading.

    The formula is for equal vertical piles under a rigid cap, so (GM-319)
    a battered pile is refused, and so are piles of unequal
    ``axial_stiffness``. The forces use no stiffness. The cap's settlement
    and tilt are worked out only when every pile carries the same
    ``axial_stiffness`` ka: ry = a/ka and rx = -b/ka (My/(ka Sxx) and
    Mx/(ka Syy) where Sxy = 0), and the settlement Vz/(n ka) at the
    centroid, reported as dz at the origin of the coordinates
    (dz = Vz/(n ka) + rx yc - ry xc) so that the module's kinematics
    P_i = ka (dz - rx y_i + ry x_i), x_i and y_i as given, give the forces
    above, as ``analyze_group_6dof`` does; a note names both. Without
    stiffness ``cap_displacements`` is empty and ``notes`` says so.

    A non-zero Mx with every pile on one line along x (one y), or My with
    every pile on one line along y, or a moment about any other one line
    of piles, is refused (GM-319 review): piles on one line carry no
    moment about it, and the formula's term would be
    dropped. A zero moment runs.

    Parameters
    ----------
    piles : list of GroupPile
        Pile layout, all vertical.
    load : GroupLoad
        Applied loads on the cap.

    Returns
    -------
    PileGroupResult
    """
    n = len(piles)
    if n == 0:
        raise ValueError("At least one pile required")

    ignored = [name for name, val in
               (("Vx", load.Vx), ("Vy", load.Vy), ("Mz", load.Mz))
               if abs(val) > 1e-12]
    if ignored:
        warnings.warn(
            f"analyze_vertical_group_simple ignores {', '.join(ignored)}: "
            "lateral loads and torsion are NOT distributed to the piles by "
            "this method. Use analyze_group_6dof with lateral_stiffness "
            "(and/or battered piles) to carry them.",
            UserWarning,
        )

    battered = [p.label or f"P{i+1}" for i, p in enumerate(piles)
                if not p.is_vertical]
    if battered:
        raise ValueError(
            "The simplified method is for vertical piles: "
            f"{', '.join(battered[:6])}{' ...' if len(battered) > 6 else ''} "
            "carry a batter, which P = Vz/n + My*x/SUM(x^2) - Mx*y/SUM(y^2) "
            "does not use. Use analyze_group_6dof (the pile_group_6dof "
            "tool) for battered piles, or leave the batter out.")
    stiffs = [p.axial_stiffness for p in piles]
    given = [k for k in stiffs if k is not None]
    if given and (len(given) < n or min(given) <= 0
                  or max(given) - min(given) > 1e-9 * max(given)):
        raise ValueError(
            "The simplified method is for equal piles: axial_stiffness "
            "must be the same positive value on every pile, or on none "
            "(the forces use no stiffness). Use analyze_group_6dof (the "
            "pile_group_6dof tool) for piles of unequal stiffness.")

    xs = np.array([p.x for p in piles], dtype=float)
    ys = np.array([p.y for p in piles], dtype=float)

    # The loads act at the piles' centroid, and the formula's distances are
    # measured from it (GM-319 review: from the origin before, so a layout
    # not centred on it gave forces that did not sum to Vz). Statics of the
    # rigid cap: SUM(x - xc) = 0, so the moment terms add no net force. The
    # product term SUM dx dy is carried where it is not zero (GM-319
    # re-check); a moment about the one line every pile is on is refused.
    plane = simple_load_plane(xs, ys, load.Mx, load.My)
    xc, yc = plane["xc"], plane["yc"]
    dxs, dys = plane["dx"], plane["dy"]
    sum_x2, sum_y2 = plane["sxx"], plane["syy"]

    # Axial force in each pile
    axial = np.full(n, load.Vz / n)

    # Moment contribution (right-hand rule, z up: +My compresses +x,
    # +Mx uplifts +y)
    if plane["form"] == "eq_8_79":
        if sum_x2 > _ON_LINE_M2:
            axial += load.My * dxs / sum_x2
        if sum_y2 > _ON_LINE_M2:
            axial -= load.Mx * dys / sum_y2
    else:
        axial += plane["a"] * dxs + plane["b"] * dys

    # Results
    pile_forces = []
    max_comp = 0.0
    max_tens = 0.0
    max_util = 0.0

    for i, (pile, force) in enumerate(zip(piles, axial)):
        compression = force if force > 0 else 0
        tension = -force if force < 0 else 0

        max_comp = max(max_comp, compression)
        max_tens = max(max_tens, tension)

        util = 0.0
        if compression > 0 and pile.axial_capacity_compression:
            util = compression / pile.axial_capacity_compression
        elif tension > 0 and pile.axial_capacity_tension:
            util = tension / pile.axial_capacity_tension
        max_util = max(max_util, util)

        pile_forces.append({
            "label": pile.label or f"P{i+1}",
            "x_m": round(pile.x, 3),
            "y_m": round(pile.y, 3),
            "axial_kN": round(float(force), 1),
            "utilization": round(util, 3),
        })

    # Cap settlement and tilt: only with the piles' (equal) axial stiffness
    notes = []
    if given:
        ka = float(given[0])
        if plane["form"] == "eq_8_79":
            rx = (float(load.Mx / (ka * sum_y2))
                  if sum_y2 > _ON_LINE_M2 else 0.0)
            ry = (float(load.My / (ka * sum_x2))
                  if sum_x2 > _ON_LINE_M2 else 0.0)
        else:
            # P = ka (w_c - rx dy + ry dx) = Vz/n + a dx + b dy
            rx, ry = -plane["b"] / ka, plane["a"] / ka
        # Vz/(n ka) is the settlement at the centroid; dz is reported at
        # the origin of the coordinates, the module's kinematics' point
        w_c = float(load.Vz / (n * ka))
        cap_disp = {
            "dx": 0.0, "dy": 0.0,
            "dz": w_c + rx * yc - ry * xc,
            "rx": rx, "ry": ry, "rz": 0.0,
        }
        notes.append(
            f"dz = {cap_disp['dz'] * 1000:.3f} mm is the cap's settlement "
            "at the origin of the coordinates, (x, y) = (0, 0), the point of "
            "the rigid-cap kinematics (w = dz - rx y + ry x); under the "
            f"piles' centroid it is Vz/(n ka) = {w_c * 1000:.3f} mm.")
    else:
        cap_disp = {}
        notes.append(
            "Cap displacements not calculated: the simplified method's "
            "forces use no pile stiffness (the pile_group_simple tool takes "
            "none). The 6-DOF analysis (pile_group_6dof) gives the cap's "
            "displacements from the piles' stiffness.")
    warning = centroid_load_warning(xc, yc, load.Vz)
    # rounded to the micrometre, never a "-0.000"
    xc, yc = round(xc, 6) + 0.0, round(yc, 6) + 0.0
    notes.append(
        "Vz, Mx and My are taken to act at the centroid of the piles, "
        f"(x, y) = ({xc:.3f}, {yc:.3f}) m"
        + ("" if warning else ", the origin of the coordinates")
        + "; each pile's x and y are measured from it, so the pile forces "
        "sum to Vz (statics of the rigid cap).")
    if plane["product"]:
        notes.append(
            "The piles are not symmetric about their centroidal axes: "
            f"SUM dx dy = {plane['sxy']:.3f} m^2 about the centroid (SUM "
            f"dx^2 = {sum_x2:.3f}, SUM dy^2 = {sum_y2:.3f} m^2), so the "
            "product term is included: P = Vz/n + a dx + b dy, a = (My SUM "
            "dy^2 + Mx SUM dx dy)/D, b = -(Mx SUM dx^2 + My SUM dx dy)/D, "
            "D = SUM dx^2 SUM dy^2 - (SUM dx dy)^2. GEC-12 Vol I Eq. 8-79 "
            "(p. 468) is the case SUM dx dy = 0. Basis: statics, the "
            "equilibrium of a rigid cap on pinned piles of equal axial "
            "stiffness (the forces give back Mx and My).")

    return PileGroupResult(
        notes=notes,
        cap_displacements=cap_disp,
        pile_forces=pile_forces,
        max_compression=max_comp,
        max_tension=max_tens,
        max_utilization=max_util,
        n_piles=n,
        centroid={"x": xc, "y": yc},
        warnings=[warning] if warning else [],
    )


def analyze_group_6dof(
    piles: List[GroupPile],
    load: GroupLoad,
) -> PileGroupResult:
    """General 6-DOF rigid cap analysis for vertical and battered piles.

    Assembles a 6x6 group stiffness matrix from individual pile
    contributions (transformed to global coordinates), solves for
    cap displacements, then back-calculates individual pile axial forces.

    Sign convention (PG-2): see the module docstring — right-handed
    axes with z UP; Vz and dz positive DOWNWARD; moments per the
    right-hand rule. The stiffness matrix is assembled as
    K = sum(B^T k B) with the rigid-cap kinematic matrix B, and pile
    displacements are back-calculated with the SAME B, so pile forces
    are in equilibrium with the applied loads under the stated
    convention (positive My -> compression on the +x side; positive
    Mx -> tension on the +y side; positive Mz -> counterclockwise
    twist in plan viewed from above).

    All six loads act at the origin of the coordinates, at the cap: the
    kinematics use each pile's x and y as given (GM-319 re-check: the tool
    text did not say so, where the simplified method takes its loads at
    the piles' centroid). A load at another point (xa, ya) needs its
    moment about the origin added, Vz xa to My, -Vz ya to Mx and
    xa Vy - ya Vx to Mz; the result's ``notes`` say so, naming the piles'
    centroid.

    Each pile is an axial spring ka along its own axis and a lateral
    spring kl across it, pinned to the cap at its head. In the global axes
    its head stiffness is k = kl*I + (ka - kl)*l*l^T, l = (lx, ly, lz) the
    pile's direction cosines: the pile's stiffness "transformed to global
    coordinates" of EM 1110-2-2906 para. 4-5b(2). All of it is assembled,
    the terms a batter puts between the horizontal and vertical head
    components (kxz, kyz) included (GM-319; left out before, when a
    battered pile's axial force did not balance the applied loads). With
    kl = 0 the piles carry axial load only (UFC 3-220-20 Sec. 6-6.3,
    Fig. 6-20). Left out, and said in the result's ``notes``: any moment
    or torsional stiffness of a pile's own head (the EM's b44, b55, b66),
    so the cap's moments are carried by the piles' head forces at their
    lever arms only. If the group has no stiffness to resist an applied
    lateral force or torsion (e.g. vertical piles with no lateral_stiffness),
    or a moment about the one line every pile is on (a single row), a
    ``ValueError`` is raised rather than silently dropping that load, saying
    for each load why.

    Parameters
    ----------
    piles : list of GroupPile
        Pile layout with stiffnesses and optional batter angles.
    load : GroupLoad
        Applied 6-DOF loading on the cap.

    Returns
    -------
    PileGroupResult
    """
    n = len(piles)
    if n == 0:
        raise ValueError("At least one pile required")

    # Check all piles have stiffness
    for i, p in enumerate(piles):
        if p.axial_stiffness is None:
            raise ValueError(f"Pile {i} ({p.label}) missing axial_stiffness")

    # Build 6x6 group stiffness matrix: K = sum(B^T k B), where B maps the
    # cap DOFs U = (dx, dy, dz, rx, ry, rz) to the pile-head displacement
    # components (ux, uy, settlement) under the module sign convention
    # (right-handed, z up; dz/settlement positive DOWN):
    #   ux_p = dx - rz*y
    #   uy_p = dy + rz*x
    #   s_p  = dz - rx*y + ry*x
    # The SAME B is used for the pile-force back-calculation below, so the
    # assembly and back-calc are self-consistent (PG-2) and the pile forces
    # equilibrate the applied loads.
    K_group = np.zeros((6, 6))
    B_matrices = []
    k3_matrices = []

    for pile in piles:
        ka = pile.axial_stiffness
        kl = pile.lateral_stiffness or 0.0

        lx, ly, lz = pile.direction_cosines()

        # Pile-head stiffness in the global (ux, uy, settlement)
        # components: the axial spring ka along the pile axis l and the
        # lateral spring kl across it, k = kl*I + (ka - kl)*l*l^T. The
        # settlement component is positive down, as lz is, so the kxz and
        # kyz terms a batter gives carry the same sign as kxy (GM-319:
        # they were left at zero before).
        l_vec = np.array([lx, ly, lz])
        k3 = kl * np.eye(3) + (ka - kl) * np.outer(l_vec, l_vec)

        x = pile.x
        y = pile.y
        B = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, -y],
            [0.0, 1.0, 0.0, 0.0, 0.0, x],
            [0.0, 0.0, 1.0, -y, x, 0.0],
        ])
        B_matrices.append(B)
        k3_matrices.append(k3)

        K_group += B.T @ k3 @ B

    # Load vector
    F = np.array([load.Vx, load.Vy, load.Vz, load.Mx, load.My, load.Mz])

    # Some groups have no stiffness in certain DOFs (e.g. a purely vertical
    # group with no lateral springs has zero resistance in dx, dy and rz). Do
    # NOT silently drop loads applied to those DOFs (PG-3): identify the
    # unsupported DOFs, raise if any of them is loaded, and statically condense
    # the unloaded free DOFs (their displacement is taken as zero, which is
    # exact here because vertical piles introduce no coupling into those DOFs).
    dof_names = ['Vx', 'Vy', 'Vz', 'Mx', 'My', 'Mz']
    diag = np.abs(np.diag(K_group))
    tol = 1e-9 * max(diag.max(), 1.0)
    supported = diag > tol
    unsupported_loaded = (~supported) & (np.abs(F) > tol)
    if np.any(unsupported_loaded):
        bad = [dof_names[i] for i in range(6) if unsupported_loaded[i]]
        # Why, load by load (GM-319 review): a horizontal force or torsion
        # needs horizontal stiffness; a moment about the line every pile is
        # on has no lever arm, which no other tool can carry either.
        why = []
        for name in bad:
            if name in ("Vx", "Vy"):
                axis = name[1]
                why.append(
                    f"{name}: no pile has stiffness along {axis} (vertical "
                    "piles with no lateral_stiffness carry no horizontal "
                    f"force); give lateral_stiffness, or piles battered "
                    f"toward {axis}")
            elif name == "Mz":
                why.append(
                    "Mz: no pile has horizontal stiffness to resist torsion; "
                    "give lateral_stiffness, or battered piles")
            else:
                coord = "y" if name == "Mx" else "x"
                on_line = sum(getattr(p, coord) ** 2 for p in piles)
                why.append(f"{name}: " + (
                    _moment_about_pile_line(name) if on_line <= _ON_LINE_M2
                    else "no pile has axial stiffness at a lever arm about "
                         f"the {'x' if name == 'Mx' else 'y'} axis"))
        raise ValueError(
            f"The pile group cannot carry the applied {', '.join(bad)}. "
            + "; ".join(why) + ".")

    U = np.zeros(6)
    idx = np.where(supported)[0]
    if idx.size:
        try:
            U[idx] = np.linalg.solve(K_group[np.ix_(idx, idx)], F[idx])
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Pile group stiffness matrix is singular; check the pile "
                "layout and stiffnesses."
            ) from exc

    dx, dy, dz, rx, ry, rz = U

    # Back-calculate pile forces using the SAME kinematic matrix B as the
    # stiffness assembly (PG-2: self-consistent end-to-end).
    pile_forces = []
    head_response = []
    max_comp = 0.0
    max_tens = 0.0
    max_util = 0.0

    for i, pile in enumerate(piles):
        ka = pile.axial_stiffness
        lx, ly, lz = pile.direction_cosines()

        # Pile head displacement components (ux, uy, settlement) from the
        # rigid-cap kinematics under the module sign convention.
        u_pile = B_matrices[i] @ U
        dx_pile, dy_pile, dz_pile = u_pile
        # The head force in the global axes, as assembled into K (so the
        # forces of all piles sum to the applied loads).
        f_pile = k3_matrices[i] @ u_pile

        # Axial displacement along the pile axis (head toward tip; lz is
        # the downward component, conjugate to settlement-positive dz_pile)
        d_axial = dx_pile * lx + dy_pile * ly + dz_pile * lz

        # Axial force (positive = compression)
        axial_force = ka * d_axial

        compression = axial_force if axial_force > 0 else 0
        tension = -axial_force if axial_force < 0 else 0
        max_comp = max(max_comp, compression)
        max_tens = max(max_tens, tension)

        util = 0.0
        if compression > 0 and pile.axial_capacity_compression:
            util = compression / pile.axial_capacity_compression
        elif tension > 0 and pile.axial_capacity_tension:
            util = tension / pile.axial_capacity_tension
        max_util = max(max_util, util)

        pile_forces.append({
            "label": pile.label or f"P{i+1}",
            "x_m": round(pile.x, 3),
            "y_m": round(pile.y, 3),
            "axial_kN": round(float(axial_force), 1),
            "utilization": round(util, 3),
        })
        head_response.append({
            "label": pile.label or f"P{i+1}",
            "ux_m": float(dx_pile), "uy_m": float(dy_pile),
            "w_m": float(dz_pile), "axial_displacement_m": float(d_axial),
            "Fx_kN": float(f_pile[0]), "Fy_kN": float(f_pile[1]),
            "Fz_kN": float(f_pile[2]), "axial_kN": float(axial_force),
        })

    cap_disp = {
        "dx": float(dx), "dy": float(dy), "dz": float(dz),
        "rx": float(rx), "ry": float(ry), "rz": float(rz),
    }

    return PileGroupResult(
        cap_displacements=cap_disp,
        pile_forces=pile_forces,
        max_compression=max_comp,
        max_tension=max_tens,
        max_utilization=max_util,
        n_piles=n,
        pile_head_response=head_response,
        notes=[
            "Piles are pinned to the cap: each is an axial spring along "
            "its axis and a lateral spring across it, with no head moment "
            "or torsional stiffness of its own (EM 1110-2-2906's b44, b55, "
            "b66 are left out). Batter is carried in full, the "
            "horizontal-vertical terms included.",
            _origin_note(piles)],
    )


def _origin_note(piles: List[GroupPile]) -> str:
    """Where the 6-DOF method takes its loads (GM-319 re-check): the origin
    of the coordinates, which its kinematics use as given; the piles'
    centroid named beside it."""
    n = len(piles)
    xc = round(sum(p.x for p in piles) / n, 6) + 0.0
    yc = round(sum(p.y for p in piles) / n, 6) + 0.0
    centred = abs(xc) <= CENTROID_TOL_M and abs(yc) <= CENTROID_TOL_M
    where = ("the piles' centroid too" if centred else
             f"not at the piles' centroid, ({xc:.3f}, {yc:.3f}) m")
    return ("Vx, Vy, Vz, Mx, My and Mz are taken to act at the origin of "
            f"the coordinates, (x, y) = (0, 0), at the cap ({where}). A "
            "load applied at another point (xa, ya) needs its moment about "
            "the origin added: Vz x xa to My, -Vz x ya to Mx, and "
            "xa x Vy - ya x Vx to Mz (+My compresses the +x side, +Mx "
            "uplifts the +y side, +Mz turns the cap counterclockwise in "
            "plan).")
