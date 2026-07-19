"""
Rigid cap pile group analysis.

Assembles the group stiffness matrix from individual pile stiffnesses,
solves for cap displacements under applied loads, and back-calculates
individual pile forces.

Supports vertical and battered piles under 6-DOF loading.

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
notation (EQUIVALENT-NOTATION). The documented PG-1 simplifications (no
per-pile rotational stiffnesses / batter cross-couplings) are unchanged.
Ledger: module_work/wiki_verification/wave7_em2906_pti_mo.md.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from pile_group.pile_layout import GroupPile


@dataclass
class GroupLoad:
    """Applied loads on the pile cap.

    All loads/moments are at the centroid of the pile group at the
    cap elevation. See the module docstring for the full sign
    convention (right-handed axes, z up; Vz positive downward).

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
        Cap displacements {dx, dy, dz, rx, ry, rz} in m and radians.
    pile_forces : list of dict
        Per-pile forces and utilization ratios.
    max_compression : float
        Maximum compression force in any pile (kN).
    max_tension : float
        Maximum tension force in any pile (kN, positive = tension).
    max_utilization : float
        Maximum utilization ratio (demand/capacity).
    """
    cap_displacements: Dict[str, float] = field(default_factory=dict)
    pile_forces: List[Dict[str, Any]] = field(default_factory=list)
    max_compression: float = 0.0
    max_tension: float = 0.0
    max_utilization: float = 0.0
    n_piles: int = 0

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
        lines.extend([
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
        return {
            "n_piles": self.n_piles,
            "cap_displacements": self.cap_displacements,
            "max_compression_kN": round(self.max_compression, 1),
            "max_tension_kN": round(self.max_tension, 1),
            "max_utilization": round(self.max_utilization, 3),
            "pile_forces": self.pile_forces,
        }


def analyze_vertical_group_simple(
    piles: List[GroupPile],
    load: GroupLoad,
) -> PileGroupResult:
    """Simplified elastic analysis for vertical piles only.

    Uses the standard pile group equation, signed per the module's
    right-hand-rule convention (z up; Vz/settlement positive down):

        Pi = Vz/n + My*xi/SUM(xi^2) - Mx*yi/SUM(yi^2)

    so positive My compresses the +x side and positive Mx uplifts the
    +y side — consistent with ``analyze_group_6dof``.

    This method ignores lateral loads (Vx, Vy) and torsion (Mz); a
    ``UserWarning`` is issued if any of them is nonzero (PG-3). Use the
    general 6-DOF method for battered piles or combined loading.

    Parameters
    ----------
    piles : list of GroupPile
        Pile layout (all assumed vertical).
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

    xs = np.array([p.x for p in piles])
    ys = np.array([p.y for p in piles])

    sum_x2 = np.sum(xs**2)
    sum_y2 = np.sum(ys**2)

    # Axial force in each pile
    axial = np.full(n, load.Vz / n)

    # Moment contribution (right-hand rule, z up: +My compresses +x,
    # +Mx uplifts +y)
    if sum_x2 > 0:
        axial += load.My * xs / sum_x2
    if sum_y2 > 0:
        axial -= load.Mx * ys / sum_y2

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

    # Simple cap displacement estimate
    if piles[0].axial_stiffness and piles[0].axial_stiffness > 0:
        ka_total = n * piles[0].axial_stiffness
        dz = load.Vz / ka_total
    else:
        dz = 0.0

    cap_disp = {"dx": 0.0, "dy": 0.0, "dz": dz, "rx": 0.0, "ry": 0.0, "rz": 0.0}

    return PileGroupResult(
        cap_displacements=cap_disp,
        pile_forces=pile_forces,
        max_compression=max_comp,
        max_tension=max_tens,
        max_utilization=max_util,
        n_piles=n,
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

    The formulation models axial pile springs with eccentricity-based moment
    resistance plus in-plane lateral springs (including the in-plane kxy
    coupling and the lateral-torsion coupling that the rigid-cap kinematics
    produce). The force<->rotation coupling that batter introduces between
    the translational and moment DOFs (kxz/kyz) is not assembled, so for
    heavily battered groups under combined load a full CPGA-style coupled
    analysis is preferred. If the group has no stiffness to resist an applied
    lateral force or torsion (e.g. vertical piles with no lateral_stiffness),
    a ``ValueError`` is raised rather than silently dropping that load.

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

    for pile in piles:
        ka = pile.axial_stiffness
        kl = pile.lateral_stiffness or 0.0

        lx, ly, lz = pile.direction_cosines()

        # Pile-head stiffness in global (ux, uy, settlement) components:
        # axial spring ka along the pile axis + isotropic lateral spring kl
        # perpendicular to it.
        kxx = ka * lx**2 + kl * (1 - lx**2)
        kyy = ka * ly**2 + kl * (1 - ly**2)
        kzz = ka * lz**2 + kl * (1 - lz**2)
        kxy = (ka - kl) * lx * ly

        # NOTE (PG-1 limitation, retained): the force<->rotation coupling
        # that batter introduces between the translational DOFs and the
        # moment DOFs (the kxz/kyz terms of the projected pile stiffness)
        # is NOT assembled -- this formulation captures the
        # axial-eccentricity moment resistance, in-plane translation
        # (incl. kxy), and the rigid-cap lateral-torsion coupling. For
        # heavily battered groups under combined load, a full CPGA-style
        # coupled formulation is required.
        k3 = np.array([
            [kxx, kxy, 0.0],
            [kxy, kyy, 0.0],
            [0.0, 0.0, kzz],
        ])

        x = pile.x
        y = pile.y
        B = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, -y],
            [0.0, 1.0, 0.0, 0.0, 0.0, x],
            [0.0, 0.0, 1.0, -y, x, 0.0],
        ])
        B_matrices.append(B)

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
        raise ValueError(
            "The pile group has no stiffness to resist the applied "
            f"{', '.join(bad)}: vertical piles with no lateral_stiffness cannot "
            "carry lateral force or torsion. Provide lateral_stiffness and/or "
            "battered piles (or use analyze_vertical_group_simple for "
            "axial-only loading)."
        )

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
    max_comp = 0.0
    max_tens = 0.0
    max_util = 0.0

    for i, pile in enumerate(piles):
        ka = pile.axial_stiffness
        lx, ly, lz = pile.direction_cosines()

        # Pile head displacement components (ux, uy, settlement) from the
        # rigid-cap kinematics under the module sign convention.
        dx_pile, dy_pile, dz_pile = B_matrices[i] @ U

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
    )
