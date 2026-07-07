"""SRM mesh-refinement (mesh-consistency) study utility.

A convenience that runs ``analyze_slope_srm`` over a sequence of mesh densities
and reports the factor-of-safety convergence, so that a strength-reduction FOS
can be shown to be **mesh-converged** (and, on the cross-check benchmarks,
consistent with the limit-equilibrium answer) rather than reported at a single
arbitrary density.

Motivation (fem2d↔LE cross-check tail, v5.3 B2e). The FE strength-reduction FOS
depends on mesh density: on Griffiths & Lane (1999) Example 1 it rises toward
the published FE value 1.4 as the slope face is refined (VALIDATION.md §1), and
on the shared-geometry Bishop cross-check it falls toward the LE value as the
mesh is refined (VALIDATION.md §6). This utility makes that refinement sequence
a one call, tabulates FOS vs element count, flags whether the last step has
settled to within a tolerance, and — when the sequence is monotonic and at a
roughly constant refinement ratio — reports a Richardson-extrapolated estimate
of the mesh-independent FOS.

Nothing here changes the SRM algorithm or any default: it only drives the
existing ``analyze_slope_srm`` at several densities.

Reference: Griffiths & Lane (1999); Roache (1998) grid-convergence /
Richardson extrapolation.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from fem2d.analysis import analyze_slope_srm


@dataclass
class MeshRefinementResult:
    """Result of an SRM mesh-refinement study.

    Attributes
    ----------
    levels : list of dict
        Per-mesh record (coarse→fine order): ``nx``, ``ny``, ``n_elements``,
        ``n_nodes``, ``FOS``, ``fos_basis``, ``converged`` (bracketing),
        ``n_srf_trials``, ``wall_time_s``, ``rel_change`` (relative FOS change
        from the previous level, None for the first), and ``error_vs_published``
        (signed relative error if a published value was supplied).
    fos_finest : float
        FOS at the finest mesh — the best single-mesh estimate.
    fos_coarsest : float
        FOS at the coarsest mesh.
    fos_richardson : float or None
        Richardson-extrapolated mesh-independent FOS from the three finest
        monotonic levels, or None when the sequence is not amenable
        (non-monotonic, <3 levels, or an out-of-range observed order).
    observed_order : float or None
        Observed order of convergence p used for the extrapolation.
    converged : bool
        True when the relative FOS change between the two finest meshes is
        below ``conv_tol``.
    conv_tol : float
        Relative-change tolerance used for ``converged``.
    published : float or None
        Published/reference FOS, if supplied (for the error column).
    element_type, srm_field : str
        Echo of the analysis options used for every level.
    """
    levels: List[dict]
    fos_finest: float
    fos_coarsest: float
    fos_richardson: Optional[float]
    observed_order: Optional[float]
    converged: bool
    conv_tol: float
    published: Optional[float]
    element_type: str
    srm_field: str

    @property
    def fos_estimate(self) -> float:
        """Best FOS estimate: the Richardson value if available, else finest."""
        return self.fos_richardson if self.fos_richardson is not None \
            else self.fos_finest

    def summary(self) -> str:
        """Text convergence table."""
        lines = [
            "SRM Mesh-Refinement Study",
            "=" * 60,
            f"Element type: {self.element_type}   reduce: {self.srm_field}",
        ]
        if self.published is not None:
            lines.append(f"Published/reference FOS: {self.published:.3f}")
        lines.append("")
        header = (f"  {'nx x ny':>9}{'elements':>10}{'FOS':>8}"
                  f"{'d(FOS)':>9}{'basis':>16}")
        if self.published is not None:
            header += f"{'err':>8}"
        lines.append(header)
        for lv in self.levels:
            row = (f"  {lv['nx']:>3}x{lv['ny']:<5}{lv['n_elements']:>10}"
                   f"{lv['FOS']:>8.3f}")
            row += (f"{lv['rel_change']*100:>8.1f}%" if lv['rel_change']
                    is not None else f"{'—':>9}")
            row += f"{lv['fos_basis']:>16}"
            if self.published is not None and lv['error_vs_published'] is not None:
                row += f"{lv['error_vs_published']*100:>7.1f}%"
            lines.append(row)
        lines.append("")
        lines.append(f"Finest-mesh FOS:      {self.fos_finest:.3f}")
        if self.fos_richardson is not None:
            lines.append(f"Richardson estimate:  {self.fos_richardson:.3f} "
                         f"(observed order p={self.observed_order:.2f})")
        lines.append(f"Converged (< {self.conv_tol*100:.0f}% step): "
                     f"{self.converged}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "levels": self.levels,
            "fos_finest": self.fos_finest,
            "fos_coarsest": self.fos_coarsest,
            "fos_richardson": self.fos_richardson,
            "observed_order": self.observed_order,
            "fos_estimate": self.fos_estimate,
            "converged": self.converged,
            "conv_tol": self.conv_tol,
            "published": self.published,
            "element_type": self.element_type,
            "srm_field": self.srm_field,
        }


def _richardson(levels: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """Richardson extrapolation from the three finest levels.

    Uses characteristic element size h ∝ 1/sqrt(n_elements). For three grids
    (coarse→fine) with FOS f1,f2,f3 and refinement ratio r = h_i/h_{i+1}:

        p        = ln|(f2-f1)/(f3-f2)| / ln(r_bar)
        f_exact  = f3 + (f3 - f2) / (r_23^p - 1)

    Returns (f_exact, p) or (None, None) when the triple is non-monotonic or
    the observed order falls outside a physically sensible band.
    """
    if len(levels) < 3:
        return None, None
    f1, f2, f3 = (levels[-3]["FOS"], levels[-2]["FOS"], levels[-1]["FOS"])
    d12, d23 = f2 - f1, f3 - f2
    # need a monotonic, converging (shrinking-step) triple
    if d12 == 0 or d23 == 0 or (d12 > 0) != (d23 > 0):
        return None, None
    if abs(d23) >= abs(d12):
        return None, None  # diverging / not settling — no meaningful estimate
    n1, n2, n3 = (levels[-3]["n_elements"], levels[-2]["n_elements"],
                  levels[-1]["n_elements"])
    if n1 <= 0 or n2 <= n1 or n3 <= n2:
        return None, None
    r12 = math.sqrt(n2 / n1)
    r23 = math.sqrt(n3 / n2)
    r_bar = math.sqrt(r12 * r23)
    if r_bar <= 1.02:
        return None, None
    try:
        p = math.log(abs(d12 / d23)) / math.log(r_bar)
    except (ValueError, ZeroDivisionError):
        return None, None
    if not (0.3 <= p <= 4.0):
        return None, None
    f_exact = f3 + d23 / (r23 ** p - 1.0)
    return f_exact, p


def srm_mesh_refinement_study(surface_points, soil_layers,
                              meshes: Sequence[Tuple[int, int]], *,
                              depth=None, x_extend=None, element_type='t6',
                              srm_field='c_phi', published: Optional[float] = None,
                              conv_tol: float = 0.03,
                              **srm_kwargs) -> MeshRefinementResult:
    """Run ``analyze_slope_srm`` over a sequence of meshes and report FOS
    convergence (mesh-consistency study).

    Parameters
    ----------
    surface_points, soil_layers, depth, x_extend, element_type, srm_field :
        Passed straight through to ``analyze_slope_srm`` (identical meaning);
        held fixed across every mesh so only density varies.
    meshes : sequence of (nx, ny)
        Mesh densities to run, ordered coarse→fine. At least two; three or more
        enable the Richardson estimate.
    published : float, optional
        Reference/published FOS, used only for the per-level error column.
    conv_tol : float
        Relative-change tolerance for the ``converged`` flag (default 0.03).
    **srm_kwargs :
        Any other ``analyze_slope_srm`` keyword (e.g. ``srf_tol``, ``gwt``,
        ``blowup_factor``, ``srf_range``, ``n_gp``) applied to every run.

    Returns
    -------
    MeshRefinementResult
    """
    if len(meshes) < 2:
        raise ValueError("Provide at least two meshes for a refinement study.")

    levels: List[dict] = []
    prev_fos = None
    for nx, ny in meshes:
        t0 = time.perf_counter()
        res = analyze_slope_srm(
            surface_points=surface_points, soil_layers=soil_layers,
            depth=depth, nx=nx, ny=ny, x_extend=x_extend,
            element_type=element_type, srm_field=srm_field, **srm_kwargs)
        dt = time.perf_counter() - t0
        fos = float(res.FOS)
        rel_change = None if prev_fos is None else (fos - prev_fos) / prev_fos
        err = None if published is None else (fos - published) / published
        levels.append({
            "nx": nx, "ny": ny,
            "n_elements": int(res.n_elements),
            "n_nodes": int(res.n_nodes),
            "FOS": fos,
            "fos_basis": res.fos_basis,
            "converged": bool(res.converged),
            "n_srf_trials": int(getattr(res, "n_srf_trials", 0) or 0),
            "wall_time_s": dt,
            "rel_change": rel_change,
            "error_vs_published": err,
        })
        prev_fos = fos

    fos_richardson, observed_order = _richardson(levels)
    last_change = levels[-1]["rel_change"]
    converged = last_change is not None and abs(last_change) < conv_tol

    return MeshRefinementResult(
        levels=levels,
        fos_finest=levels[-1]["FOS"],
        fos_coarsest=levels[0]["FOS"],
        fos_richardson=fos_richardson,
        observed_order=observed_order,
        converged=converged,
        conv_tol=conv_tol,
        published=published,
        element_type=element_type,
        srm_field=srm_field,
    )
