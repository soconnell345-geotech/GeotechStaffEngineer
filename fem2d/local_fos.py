"""Pointwise local factor of safety (mobilized-strength map).

Turns a computed stress field into a per-point factor of safety against
Mohr-Coulomb shear failure:

    local_FOS = tau_available / tau_mobilized

with, at each evaluation point (element-averaged in-plane stress),

    p   = (sxx + syy) / 2                       (mean stress, tension-positive)
    tau_mob = sqrt( ((sxx - syy)/2)^2 + txy^2 )  (Mohr-circle radius = max shear)
    tau_available = c' cos(phi') - p sin(phi')   (M-C envelope shear at this p)

This is the reciprocal of PLAXIS's "relative shear stress" tau_rel =
tau_mob / tau_available. local_FOS = 1 means the Mohr circle touches the
Mohr-Coulomb envelope (yield); > 1 is elastic (safe), < 1 is inadmissible.

Sign convention matches ``fem2d.srm`` / ``materials.mc_return_mapping``
(tension-positive; the yield function there is f = tau_mob + p sin(phi) -
c cos(phi), so f >= 0 <=> local_FOS <= 1). Soil under gravity has p < 0
(compression), which raises the available shear with depth, as expected.

Applied to a **strength-reduction result at its critical SRF** (the stress
field returned by ``analyze_slope_srm``), and evaluated with the *original*
(un-reduced) c'/phi', the low-local-FOS band traces the slip surface and its
minimum is ~ the global SRM factor of safety — because at the critical SRF the
mobilized shear equals the reduced strength c'/SRF, so the un-reduced strength
is ~SRF times larger along the band. Applied to a working-stress (unfactored)
gravity field it is the ordinary pointwise margin.

Near-zero mobilization (isotropic stress, e.g. deep interior) sends the ratio to
infinity; it is capped at ``cap`` (default 10). Stress concentrations at
re-entrant corners / load edges are genuine singularities and are reported, not
smoothed.

Units SI: stresses and c' in kPa, phi' in degrees; local FOS is dimensionless.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class LocalFOSField:
    """Per-element local factor-of-safety field + nodal values for contouring.

    Attributes
    ----------
    values : (n_elem,) ndarray
        Local FOS per element (capped at ``cap``).
    nodal_values : (n_nodes,) ndarray
        Element values averaged to nodes (for smooth contour rendering).
    nodes, elements : ndarray
        The mesh (carried so the field can be plotted standalone).
    cap : float
        Upper clip applied to the ratio (near-zero mobilization).
    min_fos : float
        Minimum local FOS over all elements.
    min_location : (float, float)
        Centroid (x, y) of the minimum-FOS element.
    mean_fos, median_fos : float
        Summary statistics over elements.
    global_fos : float or None
        The SRM global FOS, if the source result carried one (for the
        band-minimum-vs-global comparison).
    frac_below_1, frac_below_1_5 : float
        Fraction of elements with local FOS below 1.0 / 1.5.
    """
    values: np.ndarray
    nodal_values: np.ndarray
    nodes: np.ndarray
    elements: np.ndarray
    cap: float
    min_fos: float
    min_location: Tuple[float, float]
    mean_fos: float
    median_fos: float
    global_fos: Optional[float] = None
    frac_below_1: float = 0.0
    frac_below_1_5: float = 0.0

    def summary(self) -> str:
        lines = [
            "Local Factor-of-Safety Field (mobilized-strength map)",
            "=" * 54,
            f"Elements:            {len(self.values)}",
            f"Minimum local FOS:   {self.min_fos:.3f} at "
            f"({self.min_location[0]:.2f}, {self.min_location[1]:.2f}) m",
            f"Mean / median:       {self.mean_fos:.3f} / {self.median_fos:.3f}",
            f"Fraction < 1.0:      {self.frac_below_1:.1%}",
            f"Fraction < 1.5:      {self.frac_below_1_5:.1%}",
            f"Cap:                 {self.cap:g}",
        ]
        if self.global_fos is not None:
            lines.append(f"Global SRM FOS:      {self.global_fos:.3f}")
            lines.append(f"min local / global:  "
                         f"{self.min_fos / self.global_fos:.3f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Summary stats only (the full per-element field is not JSON-dumped)."""
        return {
            "min_local_fos": self.min_fos,
            "min_location_xy": [float(self.min_location[0]),
                                float(self.min_location[1])],
            "mean_local_fos": self.mean_fos,
            "median_local_fos": self.median_fos,
            "global_fos": self.global_fos,
            "frac_below_1": self.frac_below_1,
            "frac_below_1_5": self.frac_below_1_5,
            "cap": self.cap,
            "n_elements": int(len(self.values)),
        }


def _corner(elements):
    return np.asarray(elements, dtype=int)[:, :3]


def local_fos_field(result, c, phi, *, cap: float = 10.0,
                    tau_floor: float = 1e-9) -> LocalFOSField:
    """Compute the pointwise local FOS field from a result's stress field.

    Parameters
    ----------
    result : object
        Any object exposing ``nodes`` (n_nodes, 2), ``elements``
        (n_elem, 3 or 6) and ``stresses`` (n_elem, 3) = element-averaged
        in-plane [sigma_xx, sigma_yy, tau_xy] (kPa). The FEMResult from
        ``analyze_slope_srm`` / ``analyze_gravity`` provides these.
    c, phi : float or (n_elem,) array-like
        Effective cohesion c' (kPa) and friction angle phi' (deg) per element
        (scalar broadcast for a homogeneous domain). Pass the *original*
        (un-reduced) strengths.
    cap : float
        Upper clip for local FOS where the mobilized shear is ~0.
    tau_floor : float
        Mobilized-shear floor (kPa) below which the point is treated as
        unmobilized (local FOS -> cap).

    Returns
    -------
    LocalFOSField
    """
    nodes = np.asarray(result.nodes, dtype=float)
    elements = np.asarray(result.elements, dtype=int)
    stresses = getattr(result, "stresses", None)
    if stresses is None:
        raise ValueError("result carries no element stresses; run an SRM or "
                         "gravity analysis first")
    stresses = np.asarray(stresses, dtype=float)
    n_elem = len(elements)
    if stresses.shape[0] != n_elem or stresses.shape[1] < 3:
        raise ValueError(
            f"expected element stresses of shape ({n_elem}, 3), got "
            f"{stresses.shape}")

    c_arr = np.broadcast_to(np.asarray(c, dtype=float), (n_elem,)).astype(float)
    phi_arr = np.broadcast_to(np.asarray(phi, dtype=float),
                              (n_elem,)).astype(float)

    sxx, syy, txy = stresses[:, 0], stresses[:, 1], stresses[:, 2]
    p = 0.5 * (sxx + syy)                                   # tension-positive
    tau_mob = np.sqrt((0.5 * (sxx - syy)) ** 2 + txy ** 2)  # Mohr radius
    phi_r = np.radians(phi_arr)
    tau_avail = c_arr * np.cos(phi_r) - p * np.sin(phi_r)
    tau_avail = np.maximum(tau_avail, 0.0)      # M-C tension cutoff on strength

    fos = np.where(tau_mob > tau_floor, tau_avail / np.maximum(tau_mob, 1e-30),
                   cap)
    fos = np.clip(fos, 0.0, cap)

    # element centroids (corner nodes)
    cent = nodes[_corner(elements)].mean(axis=1)
    imin = int(np.argmin(fos))

    # average element values to nodes for contouring
    nodal = np.full(len(nodes), np.nan)
    counts = np.zeros(len(nodes))
    acc = np.zeros(len(nodes))
    corner = _corner(elements)
    for e in range(n_elem):
        for nd in corner[e]:
            acc[nd] += fos[e]
            counts[nd] += 1.0
    good = counts > 0
    nodal[good] = acc[good] / counts[good]
    # midside nodes of a T6 mesh (not in corner list) fall back to the cap→mean
    nodal[~good] = float(np.nanmean(nodal[good])) if good.any() else cap

    global_fos = getattr(result, "FOS", None)
    global_fos = float(global_fos) if global_fos is not None else None

    return LocalFOSField(
        values=fos,
        nodal_values=nodal,
        nodes=nodes,
        elements=elements,
        cap=float(cap),
        min_fos=float(fos.min()),
        min_location=(float(cent[imin, 0]), float(cent[imin, 1])),
        mean_fos=float(fos.mean()),
        median_fos=float(np.median(fos)),
        global_fos=global_fos,
        frac_below_1=float((fos < 1.0).mean()),
        frac_below_1_5=float((fos < 1.5).mean()),
    )
