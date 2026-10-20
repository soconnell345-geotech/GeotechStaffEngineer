"""
Post-liquefaction downdrag (FHWA GEC-10 §10.7; BACKLOG GM-03).

FHWA GEC-10 (FHWA-NHI-18-024) §10.7 "Post-Liquefaction Downdrag", p. 10-52:
"Post-liquefaction reconsolidation settlement of liquefied layers due to
excess pore water pressure dissipation as well as settlement of overlying
non-liquefied layers results in drag loads on drilled shafts. ... Maximum
drag loads occur at the end of reconsolidation settlement when effective
stress has returned to a condition approximately equal to the
pre-earthquake effective stress. Drag loads imposed by previously liquefied
layers should be computed as 50 percent of the pre-earthquake side
resistance calculated using the beta method. ... For geomaterial layers
above the zone of liquefaction, drag loads should be based on side
resistance values corresponding to static conditions." The drag is then
checked "for structural strength and service limit states as described in
Section 10.6", whose neutral-plane procedure is GEC-12 §7.3.6.1, the one
:class:`downdrag.DowndragAnalysis` runs.

:func:`post_liquefaction_downdrag` runs that analysis twice on the same
inputs: before the earthquake (the static analysis as given) and after
reconsolidation, with

- the liquefied layers' unit side resistance replaced by one of the cited
  options (all evaluated side by side; ``LIQUEFIED_SIDE_RESISTANCE``);
- the soil settlement profile increased by the reconsolidation settlement
  of the liquefied layers (volumetric strain x thickness), which the soil
  above them rides down with, on top of any fill / drawdown settlement of
  the static analysis.

Every neutral-plane method of the static analysis is evaluated in both
states. DESIGN.md "Post-liquefaction downdrag" quotes the sources and works
the hand check the tests pin.

All units SI: m, kN, kPa; volumetric strain as a decimal.
"""

import math
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional, Sequence

import numpy as np

from downdrag.analysis import DowndragAnalysis
from downdrag.results import DowndragResult


#: One atmosphere (kPa). GEC-10 Eq 9-13 takes sigma'v0 and gives su,r in
#: atmospheres.
P_ATM = 101.325

#: The options for the unit side resistance of a previously liquefied
#: layer, in the order they are reported.
LIQUEFIED_SIDE_RESISTANCE = ("half_pre_earthquake", "zero",
                             "residual_strength", "ratio")

#: The default: the one rule GEC-10 §10.7 prints for the DRAG of a
#: previously liquefied layer.
DEFAULT_SIDE_RESISTANCE = "half_pre_earthquake"

#: GEC-10 §9.4.2.2, p. 9-42: "layers predicted to undergo liquefaction
#: (FSliq <= 1.0)".
DEFAULT_FS_LIQ_THRESHOLD = 1.0

#: Each option as its reference words it, where, and what it assumes.
_SIDE = {
    "half_pre_earthquake": {
        "name": "50 percent of the pre-earthquake side resistance",
        "source": ("FHWA GEC-10 §10.7, p. 10-52 (Rollins et al. 2018): "
                   "'Drag loads imposed by previously liquefied layers "
                   "should be computed as 50 percent of the pre-earthquake "
                   "side resistance calculated using the beta method.'"),
        "assumptions": (
            "At the end of reconsolidation, when the effective stress is "
            "back to about its pre-earthquake value, the liquefied layer "
            "carries half its pre-earthquake (static) unit side resistance; "
            "from instrumented deep foundations in blast-induced "
            "liquefaction (Rollins et al. 2018). GEC-10 is the drilled "
            "shaft manual; the Rollins et al. (2018) tests it cites "
            "covered 'various deep foundation types' (p. 10-51). The same "
            "unit value acts as resistance where the layer is below the "
            "neutral plane."),
    },
    "zero": {
        "name": "Zero side resistance in the liquefied layers",
        "source": ("FHWA GEC-10 §10.7, p. 10-51: resistance of liquefied "
                   "soil 'can be taken as zero for practical purposes when "
                   "evaluating the Extreme Event I limit state' (Rollins et "
                   "al. 2018: 'essentially zero side resistance'); UFC "
                   "3-220-20 Table 6-49 (p. 501): liquefaction "
                   "'temporarily eliminates the geotechnical resistance "
                   "within the liquefied zones'"),
        "assumptions": (
            "The liquefied layer carries no side resistance and no drag: "
            "the state while it is liquefied, before reconsolidation. A "
            "lower bound on the drag from that layer and on its resistance."),
    },
    "residual_strength": {
        "name": "Post-liquefaction residual strength su,r",
        "source": ("FHWA GEC-10 §10.7, p. 10-51: 'A lower bound side "
                   "resistance should be taken as equal to the "
                   "post-liquefaction residual undrained strength of the "
                   "layer, su,r'; FHWA GEC-12 §7.4.2.2, p. 383: 'assigning "
                   "residual strength properties to the liquefiable "
                   "layers'; su,r given, or from (N1)60 by GEC-10 Eq 9-13 "
                   "(p. 9-43, Kramer and Wang 2015)"),
        "assumptions": (
            "Unit side resistance = su,r in the liquefied layer, never more "
            "than its pre-earthquake value. GEC-10 gives it as the lower "
            "bound of the side resistance while excess pore pressure is "
            "high (ru close to 1)."),
    },
    "ratio": {
        "name": "A fraction of the pre-earthquake side resistance (given)",
        "source": ("liquefied_side_resistance_ratio as given; GEC-10 §10.7 "
                   "(p. 10-52) uses 0.5"),
        "assumptions": (
            "The liquefied layer carries the given fraction of its "
            "pre-earthquake unit side resistance, as drag above the neutral "
            "plane and as resistance below it."),
    },
}

_GEC10_TOE = ("FHWA GEC-10 §10.7, p. 10-51: 'In no case should a drilled "
              "shaft be designed with its tip bearing immediately above or "
              "in a layer predicted to be susceptible to liquefaction.'")


def residual_strength_kramer_wang(N1_60: float, sigma_v0: float) -> float:
    """Post-liquefaction residual undrained strength su,r (kPa), FHWA GEC-10
    Eq 9-13 (p. 9-43), the median relationship of Kramer and Wang (2015):

        su,r = exp(-8.444 + 0.109 (N1)60 + 5.379 (sigma'v0)^0.1)

    with sigma'v0 (the pre-earthquake vertical effective stress) and su,r
    in atmospheres (here converted from and to kPa, 1 atm = 101.325 kPa).
    GEC-10 prints it with "≥ drained strength"; the post-liquefaction
    analysis never lets su,r exceed the layer's pre-earthquake unit side
    resistance.

    Parameters
    ----------
    N1_60 : float
        Overburden- and energy-corrected SPT blow count (N1)60.
    sigma_v0 : float
        Pre-earthquake vertical effective stress (kPa).
    """
    s = max(float(sigma_v0), 0.0) / P_ATM
    return P_ATM * math.exp(-8.444 + 0.109 * float(N1_60)
                            + 5.379 * s ** 0.1)


@dataclass
class LiquefiedLayer:
    """A depth range that liquefies, and what it does after.

    Parameters
    ----------
    top, bottom : float
        Depths (m) below the ground surface (the pile head level).
    volumetric_strain : float
        Post-liquefaction reconsolidation volumetric strain (decimal, e.g.
        0.03 for 3 %). Its reconsolidation settlement is
        ``volumetric_strain x (bottom - top)``.
    su_r : float, optional
        Residual undrained strength (kPa), for the residual-strength option.
    N1_60 : float, optional
        (N1)60, for su,r by GEC-10 Eq 9-13 when ``su_r`` is not given.
    fs_liq : float, optional
        Factor of safety against liquefaction (reported only).
    """
    top: float
    bottom: float
    volumetric_strain: float
    su_r: Optional[float] = None
    N1_60: Optional[float] = None
    fs_liq: Optional[float] = None

    def __post_init__(self):
        for name in ("top", "bottom", "volumetric_strain"):
            v = getattr(self, name)
            if v is None or not math.isfinite(float(v)):
                raise ValueError(f"liquefied layer {name} must be a number, "
                                 f"got {v!r}")
        if self.top < 0 or self.bottom <= self.top:
            raise ValueError(
                f"liquefied layer needs 0 <= top < bottom (m), got top "
                f"{self.top}, bottom {self.bottom}")
        if not 0.0 <= self.volumetric_strain <= 0.2:
            raise ValueError(
                f"volumetric_strain is a decimal between 0 and 0.2 (0.03 = "
                f"3 %), got {self.volumetric_strain} at {self.top:g}-"
                f"{self.bottom:g} m")
        if self.su_r is not None and self.su_r < 0:
            raise ValueError(f"su_r must be non-negative, got {self.su_r}")
        if self.N1_60 is not None and self.N1_60 < 0:
            raise ValueError(f"N1_60 must be non-negative, got {self.N1_60}")

    @property
    def thickness(self) -> float:
        return self.bottom - self.top

    @property
    def settlement(self) -> float:
        """Reconsolidation settlement of the layer (m)."""
        return self.volumetric_strain * self.thickness


#: Keys a liquefied layer given as a dict may carry.
LIQUEFIED_LAYER_KEYS = ("top", "bottom", "volumetric_strain", "settlement",
                        "su_r", "N1_60", "fs_liq")


def liquefied_layer(item: Any) -> LiquefiedLayer:
    """A :class:`LiquefiedLayer` from itself or a dict with ``top``,
    ``bottom`` and ``volumetric_strain`` (decimal) or ``settlement`` (m, the
    layer's reconsolidation settlement), and optionally ``su_r`` (kPa),
    ``N1_60`` and ``fs_liq``."""
    if isinstance(item, LiquefiedLayer):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"a liquefied layer is a dict with top, bottom and "
                         f"volumetric_strain, got {item!r}")
    unknown = sorted(set(item) - set(LIQUEFIED_LAYER_KEYS))
    if unknown:
        raise ValueError(f"liquefied layer: unknown key(s) {unknown}; valid "
                         f"keys {list(LIQUEFIED_LAYER_KEYS)}")
    for k in ("top", "bottom"):
        if item.get(k) is None:
            raise ValueError(f"liquefied layer needs '{k}' (m)")
    top, bottom = float(item["top"]), float(item["bottom"])
    strain, settle = item.get("volumetric_strain"), item.get("settlement")
    if (strain is None) == (settle is None):
        raise ValueError(
            f"liquefied layer {top:g}-{bottom:g} m: give volumetric_strain "
            f"(decimal) or settlement (m), one of them: the reconsolidation "
            f"settlement that drags the pile (GEC-10 §10.7: Ishihara and "
            f"Yoshimine 1992, as Idriss and Boulanger 2008 describe it)")
    if strain is None:
        if bottom <= top:
            raise ValueError(f"liquefied layer needs top < bottom, got "
                             f"{top:g}-{bottom:g} m")
        strain = float(settle) / (bottom - top)
    return LiquefiedLayer(top=top, bottom=bottom,
                          volumetric_strain=float(strain),
                          su_r=item.get("su_r"), N1_60=item.get("N1_60"),
                          fs_liq=item.get("fs_liq"))


def _readings(result: Any):
    """``(depth, fs_liq, strain or None, N1_60 or None, bounds or None)``
    lists from a liquefaction result: the ``liquefaction`` tool's CPT
    output (``depth_profile`` rows with ``top_m``, ``bottom_m``,
    ``FOS_liq`` and ``volumetric_strain``: ``bounds`` are then their
    ``(top, bottom)``), its SPT output (``layer_results`` rows with
    ``depth_m``, ``FOS_liq`` and ``N1_60`` or ``N160``), either list of
    rows on its own, or a profile with ``depth`` and ``factor_of_safety``
    arrays (and ``volumetric_strain``), as a dict or as the CPT result
    object."""
    def get(name):
        if isinstance(result, dict):
            return result.get(name)
        return getattr(result, name, None)

    def num(v):
        return None if v is None else float(v)

    def fs_of(r):
        f = num(r.get("FOS_liq", r.get("fos")))
        return math.nan if f is None else f

    if isinstance(result, (list, tuple)) and result and all(
            isinstance(r, dict) for r in result):
        rows = list(result)
        bands = "top_m" in rows[0]
    else:
        bands = bool(get("depth_profile"))
        rows = get("depth_profile") or get("layer_results")
    if rows and bands:
        depth, fs, strain, bounds = [], [], [], []
        for r in rows:
            if r.get("top_m") is None or r.get("bottom_m") is None:
                raise ValueError(
                    "liquefaction_result: each depth_profile row needs "
                    "top_m and bottom_m")
            top, bottom = float(r["top_m"]), float(r["bottom_m"])
            bounds.append((top, bottom))
            depth.append(0.5 * (top + bottom))
            fs.append(fs_of(r))
            strain.append(num(r.get("volumetric_strain")))
        return depth, fs, strain, [None] * len(depth), bounds
    if rows:
        depth, fs, strain, n = [], [], [], []
        for r in rows:
            depth.append(float(r["depth_m"]))
            fs.append(fs_of(r))
            strain.append(num(r.get("volumetric_strain")))
            n.append(num(r.get("N1_60", r.get("N160"))))
        return depth, fs, strain, n, None
    depth = get("depth")
    fos = get("factor_of_safety")
    if fos is None:
        fos = get("FOS_liq")
    if depth is None or fos is None:
        raise ValueError(
            "liquefaction_result: expected the liquefaction tool's output "
            "(depth_profile rows with top_m, bottom_m, FOS_liq and "
            "volumetric_strain from a CPT; layer_results with depth_m and "
            "FOS_liq from SPT) or a profile with depth and "
            "factor_of_safety")
    depth = [float(x) for x in np.asarray(depth, dtype=float).ravel()]
    fos = [float(x) for x in np.asarray(fos, dtype=float).ravel()]
    if len(depth) != len(fos):
        raise ValueError("liquefaction_result: depth and factor_of_safety "
                         "are not the same length")
    vs = get("volumetric_strain")
    if vs is not None and len(np.asarray(vs).ravel()) == len(depth):
        strain = [float(x) for x in np.asarray(vs, dtype=float).ravel()]
    else:
        strain = [None] * len(depth)
    return depth, fos, strain, [None] * len(depth), None


def liquefied_layers_from_result(result: Any, volumetric_strain=None,
                                 fs_liq_threshold: float =
                                 DEFAULT_FS_LIQ_THRESHOLD,
                                 profile_depth: Optional[float] = None,
                                 water_table: Optional[float] = None):
    """The liquefied layers a liquefaction result implies.

    A reading liquefies where FS_liq <= ``fs_liq_threshold`` (GEC-10
    §9.4.2.2, p. 9-42: "layers predicted to undergo liquefaction (FSliq ≤
    1.0)"). Each reading stands for the depth halfway to its neighbours
    (the first and last half a spacing beyond their reading, the first not
    above the ground surface, the last not below ``profile_depth``), or,
    for the ``liquefaction`` tool's CPT ``depth_profile`` rows, for its
    own ``top_m`` to ``bottom_m`` (the last not below ``profile_depth``);
    and not above ``water_table``: soil above the water table does not
    liquefy. A banded row counts by its ``FOS_liq``, the band's lowest
    reading on its side of 1.0, so at a threshold other than 1.0 a band
    is all liquefied or all not.

    The reconsolidation strain is the result's own (a CPT profile from
    liquepy carries Zhang et al. 2002's) or ``volumetric_strain``: one
    decimal for every liquefied reading, or one per reading (row) in the
    result's order, which replaces the result's. The tool's SPT routes
    compute none and no reference in the library prints an SPT method
    (GEC-10 §10.7 names Ishihara and Yoshimine 1992 without printing it),
    so it must then be given.

    Returns
    -------
    (layers, left_out) : (list of LiquefiedLayer, float)
        One layer per liquefied reading, in depth order, and the
        settlement (m) the result's own strains give at readings above the
        threshold, which are left out.
    """
    if fs_liq_threshold is None or not fs_liq_threshold > 0:
        raise ValueError(f"fs_liq_threshold must be positive, got "
                         f"{fs_liq_threshold!r}")
    depth, fs, strain, n160, bounds = _readings(result)
    if bounds is None and len(depth) < 2:
        raise ValueError(
            "liquefaction_result has fewer than two readings, so the depth "
            "each stands for is not known: give liquefied_layers (top, "
            "bottom, volumetric_strain) instead")
    if not depth:
        raise ValueError("liquefaction_result has no readings")
    order = sorted(range(len(depth)), key=lambda i: depth[i])
    if volumetric_strain is not None:
        if isinstance(volumetric_strain, (list, tuple, np.ndarray)):
            given = [float(x) for x in volumetric_strain]
            if len(given) != len(depth):
                raise ValueError(
                    f"volumetric_strain has {len(given)} values for "
                    f"{len(depth)} readings: give one per reading, in the "
                    f"result's order, or a single value")
            strain = given
        else:
            # one value for the liquefied readings; nothing for the others
            one = float(volumetric_strain)
            strain = [one if math.isfinite(f) and f <= fs_liq_threshold
                      else None for f in fs]
    if bounds is not None:
        spans = [list(bounds[i]) for i in order]
        for (_, b0), (t1, _) in zip(spans[:-1], spans[1:]):
            if t1 < b0 - 1e-9:
                raise ValueError(
                    f"liquefaction_result: depth_profile rows overlap at "
                    f"{t1:g}-{b0:g} m")
        if profile_depth is not None and spans[-1][0] < float(profile_depth):
            spans[-1][1] = min(spans[-1][1], float(profile_depth))
    else:
        z = [depth[i] for i in order]
        edges = [max(0.0, z[0] - 0.5 * (z[1] - z[0]))]
        edges += [0.5 * (a + b) for a, b in zip(z[:-1], z[1:])]
        last = z[-1] + 0.5 * (z[-1] - z[-2])
        if profile_depth is not None:
            last = min(last, float(profile_depth))
        edges.append(last)
        spans = [[a, b] for a, b in zip(edges[:-1], edges[1:])]
    layers, left_out = [], 0.0
    for k, i in enumerate(order):
        top, bottom = spans[k]
        if water_table is not None:
            top = max(top, float(water_table))
        if bottom <= top:
            continue
        f = fs[i]
        eps = strain[i]
        liquefied = math.isfinite(f) and f <= fs_liq_threshold
        if not liquefied:
            if eps is not None and math.isfinite(eps) and eps > 0:
                left_out += eps * (bottom - top)
            continue
        if eps is None or not math.isfinite(eps):
            raise ValueError(
                f"the liquefaction result gives no post-liquefaction "
                f"volumetric strain at {depth[i]:g} m (FS_liq {f:.2f}); the "
                f"tool's SPT routes compute none, and no reference in the "
                f"library prints an SPT method (GEC-10 §10.7 names Ishihara "
                f"and Yoshimine (1992), as Idriss and Boulanger (2008) "
                f"describe it, without printing it). Give volumetric_strain "
                f"(decimal: one value for every liquefied reading, or one "
                f"per reading) and say where it comes from, or give "
                f"liquefied_layers; a CPT gives its own (the liquefaction "
                f"tool's depth_profile)")
        layers.append(LiquefiedLayer(top=top, bottom=bottom,
                                     volumetric_strain=eps, N1_60=n160[i],
                                     fs_liq=f))
    return layers, left_out


def _strain_basis(result: Any) -> str:
    """Where a liquefaction result's own strains come from, as far as the
    result says: the ``liquefaction`` tool's CPT route (and liquepy's CPT
    result) computes them by Zhang et al. (2002)."""
    rows = isinstance(result, dict) and result.get("depth_profile")
    if (rows or (isinstance(result, (list, tuple)) and result
                 and isinstance(result[0], dict) and "top_m" in result[0])
            or hasattr(result, "q_c1n_cs")):
        out = ("Zhang et al. (2002) (Can. Geotech. J. 39, doi "
               "10.1139/t02-047) from FS_liq and qc1Ncs, as the "
               "liquefaction tool's CPT route (liquepy) computes it; GEC-10 "
               "§10.7 names Ishihara and Yoshimine (1992), as Idriss and "
               "Boulanger (2008) describe it")
        note = isinstance(result, dict) and result.get("depth_profile_note")
        if note:
            out += (". The CPT profile came in depth bands: "
                    "each band's strain is its average")
        return out
    return "as the liquefaction result gives it"


def _zones(layers: Sequence[LiquefiedLayer]) -> List[List[LiquefiedLayer]]:
    """The liquefied layers grouped into contiguous zones, in depth
    order."""
    out: List[List[LiquefiedLayer]] = []
    for lay in sorted(layers, key=lambda s: s.top):
        if out and abs(lay.top - out[-1][-1].bottom) < 1e-9:
            out[-1].append(lay)
        else:
            out.append([lay])
    return out


def _shaft_between(z, fs, a: float, b: float, perimeter: float) -> float:
    """Shaft force (kN) of the piecewise-linear unit side resistance
    profile ``fs`` over depths ``a`` to ``b`` (a repeated depth is a
    jump)."""
    total = 0.0
    for i in range(len(z) - 1):
        z0, z1 = float(z[i]), float(z[i + 1])
        if z1 <= z0:
            continue
        lo, hi = max(a, z0), min(b, z1)
        if hi <= lo:
            continue
        slope = (fs[i + 1] - fs[i]) / (z1 - z0)
        f0 = fs[i] + slope * (lo - z0)
        f1 = fs[i] + slope * (hi - z0)
        total += 0.5 * (f0 + f1) * (hi - lo)
    return float(total * perimeter)


@dataclass
class _AfterLiquefaction(DowndragAnalysis):
    """The static analysis after reconsolidation of the liquefied layers:
    their unit side resistance by ``side_resistance`` and the soil
    settlement increased by their reconsolidation settlement."""
    liquefied: tuple = ()
    side_resistance: str = DEFAULT_SIDE_RESISTANCE
    side_ratio: float = 0.5

    def _extra_boundaries(self) -> tuple:
        out = []
        for zone in _zones(self.liquefied):
            out += [zone[0].top, zone[-1].bottom]
        return tuple(out)

    def _layer_at(self, z: float) -> Optional[LiquefiedLayer]:
        """The liquefied layer at a node's lookup depth (a repeated depth
        picks above / below as a layer boundary does); the ground surface
        and the toe belong to a layer that reaches them."""
        L = self.pile_length
        tol = 1e-9 * max(L, 1.0)
        for s in self.liquefied:
            lo = z > s.top or (s.top <= tol and z >= -tol)
            hi = z < s.bottom or (s.bottom >= L - tol and z <= L + tol)
            if lo and hi:
                return s
        # a node on the boundary between two liquefied layers of one zone
        # is inside the zone (GM-201: until then it kept its static value)
        for s in self.liquefied:
            if abs(z - s.bottom) <= tol and any(
                    abs(t.top - s.bottom) <= tol for t in self.liquefied):
                return s
        return None

    def liquefied_fs(self, s: LiquefiedLayer, fs_pre: float,
                     sigma: float) -> float:
        """Unit side resistance (kPa) after liquefaction where the
        pre-earthquake value is ``fs_pre`` at effective stress ``sigma``."""
        opt = self.side_resistance
        if opt == "half_pre_earthquake":
            return 0.5 * fs_pre
        if opt == "zero":
            return 0.0
        if opt == "ratio":
            return self.side_ratio * fs_pre
        su_r = (s.su_r if s.su_r is not None
                else residual_strength_kramer_wang(s.N1_60, sigma))
        if su_r > fs_pre + 0.05:
            # never more than before the earthquake; said in a warning
            # (0.05 kPa: Eq 9-13 gives about 0.02 kPa at zero stress)
            self.__dict__.setdefault("capped", set()).add((s.top, s.bottom))
        return min(su_r, fs_pre)

    def _compute_skin_friction(self, z_nodes: np.ndarray,
                               sigma_v: np.ndarray) -> np.ndarray:
        fs = super()._compute_skin_friction(z_nodes, sigma_v)
        for i, z in enumerate(self._lookup_depths(z_nodes)):
            s = self._layer_at(z)
            if s is not None:
                fs[i] = self.liquefied_fs(s, float(fs[i]), float(sigma_v[i]))
        return fs

    def _strain_integral(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Reconsolidation settlement (m) of each depth range [a, b]."""
        tops = np.array([s.top for s in self.liquefied])
        bots = np.array([s.bottom for s in self.liquefied])
        eps = np.array([s.volumetric_strain for s in self.liquefied])
        if eps.size == 0:
            return np.zeros(len(a))
        lo = np.maximum(a[:, None], tops[None, :])
        hi = np.minimum(b[:, None], bots[None, :])
        return (np.clip(hi - lo, 0.0, None) * eps[None, :]).sum(axis=1)

    def _compute_soil_settlement(self, z_nodes: np.ndarray, dz: float,
                                 sigma_v: np.ndarray,
                                 delta_sigma: np.ndarray) -> np.ndarray:
        """The static (fill / drawdown) settlement profile plus the
        reconsolidation settlement of the liquefied layers between each
        depth and the toe (the soil above a liquefied layer rides down with
        it)."""
        base = super()._compute_soil_settlement(z_nodes, dz, sigma_v,
                                                delta_sigma)
        sub = self._strain_integral(np.asarray(z_nodes[:-1], dtype=float),
                                    np.asarray(z_nodes[1:], dtype=float))
        add = np.zeros(len(z_nodes))
        add[:-1] = np.cumsum(sub[::-1])[::-1]
        return base + add

    def reconsolidation_below_toe(self) -> float:
        """Reconsolidation settlement (m) of the liquefied soil below the
        toe: the soil at the toe level, and the toe with it, settles that
        much."""
        L = float(self.pile_length)
        return float(sum(s.volumetric_strain
                         * max(0.0, s.bottom - max(s.top, L))
                         for s in self.liquefied))

    def _fill_settlement_below_toe(self) -> float:
        return (super()._fill_settlement_below_toe()
                + self.reconsolidation_below_toe())

    def _settling_base_depth(self, S: dict) -> Optional[float]:
        base = super()._settling_base_depth(S)
        liq = max((s.bottom for s in self.liquefied
                   if s.volumetric_strain > 0), default=None)
        found = [x for x in (base, liq) if x is not None]
        return max(found) if found else None


#: The numbers each before / after row reports.
_ROW_KEYS = ("neutral_plane_depth_m", "max_pile_load_kN", "dragload_kN",
             "drag_force_kN", "pile_settlement_mm")


@dataclass
class PostLiquefactionResult:
    """Downdrag before the earthquake and after reconsolidation of the
    liquefied layers (FHWA GEC-10 §10.7), every neutral-plane method and
    every cited side-resistance option side by side."""
    before: DowndragResult
    after: DowndragResult
    liquefied_side_resistance: str
    side_resistance_comparison: list
    liquefied_zones: list
    by_method: list
    reconsolidation_settlement: float
    reconsolidation_below_toe: float
    post_liquefaction_basis: str
    limit_states_note: str
    warnings: list = field(default_factory=list)
    choices: list = field(default_factory=list)
    fs_liq_threshold: Optional[float] = None

    @property
    def side_resistance_name(self) -> str:
        return _SIDE[self.liquefied_side_resistance]["name"]

    def summary(self) -> str:
        """Plain-text summary: the before / after numbers, the side
        resistance options, the liquefied zones and every method in both
        states."""
        b, a = self.before, self.after
        lines = [
            "Post-Liquefaction Downdrag (FHWA GEC-10 §10.7)",
            "=" * 46,
            f"Liquefied layers' side resistance: {self.side_resistance_name}",
            f"Reconsolidation settlement at the ground surface: "
            f"{self.reconsolidation_settlement * 1000:.1f} mm",
            "",
            f"{'':<28}{'Before':>12}{'After':>12}",
            f"{'Headline method':<28}  {b.headline_method}",
            f"{'':<28}  {a.headline_method}",
            f"{'Neutral plane (m)':<28}{b.neutral_plane_depth:>12.2f}"
            f"{a.neutral_plane_depth:>12.2f}",
            f"{'Q_np (kN)':<28}{b.max_pile_load:>12.1f}"
            f"{a.max_pile_load:>12.1f}",
            f"{'Drag force (kN)':<28}{b.drag_force:>12.1f}"
            f"{a.drag_force:>12.1f}",
            f"{'Pile head settlement (mm)':<28}"
            f"{b.pile_settlement * 1000:>12.1f}"
            f"{a.pile_settlement * 1000:>12.1f}",
            "",
            "--- Liquefied zones ---",
        ]
        for zrow in self.liquefied_zones:
            lines.append(
                f"  {zrow['top_m']:.2f}-{zrow['bottom_m']:.2f} m: strain "
                f"{zrow['volumetric_strain'] * 100:.2f} %, settlement "
                f"{zrow['reconsolidation_settlement_mm']:.1f} mm, side "
                f"resistance {zrow['side_resistance_before_kN']:.0f} -> "
                f"{zrow['side_resistance_after_kN']:.0f} kN; "
                f"{zrow['position']}")
        lines += ["", "--- Side resistance of the liquefied layers (each "
                  "option, same headline method) ---"]
        for r in self.side_resistance_comparison:
            mark = "* " if r["used"] else "  "
            if r["applies"]:
                lines.append(
                    f"{mark}{r['name']}: neutral plane "
                    f"{r['neutral_plane_depth_m']:.2f} m, drag force "
                    f"{r['drag_force_kN']:.0f} kN, pile settlement "
                    f"{r['pile_settlement_mm']:.1f} mm")
            else:
                lines.append(f"{mark}{r['name']}: not evaluated "
                             f"({r['reason']})")
            lines.append(f"      {r['source']}")
        lines += ["", "--- Every neutral-plane method, before -> after ---"]
        for r in self.by_method:
            def cell(state):
                if not r[f"applies_{state}"]:
                    return "n/a"
                return (f"{r[f'neutral_plane_{state}_m']:.2f} m / "
                        f"{r[f'drag_force_{state}_kN']:.0f} kN / "
                        f"{r[f'pile_settlement_{state}_mm']:.1f} mm")
            lines.append(f"  {r['name']}: {cell('before')} -> "
                         f"{cell('after')}")
        lines += ["  (neutral plane / drag force / pile head settlement)"]
        checks = []
        if a.structural_ok is not None:
            checks.append(f"Structural (UFC Eq 6-80 / GEC-12 Eq 7-70): "
                          f"{'PASS' if a.structural_ok else 'FAIL'}, demand "
                          f"{a.structural_demand:.1f} kN")
        if a.geotechnical_ok is not None:
            checks.append(f"Geotechnical: "
                          f"{'PASS' if a.geotechnical_ok else 'FAIL'}")
        if a.settlement_ok is not None:
            checks.append(f"Settlement: "
                          f"{'PASS' if a.settlement_ok else 'FAIL'}")
        if checks:
            lines += ["", "--- Limit states after liquefaction ---", *checks]
        lines += ["", self.limit_states_note, "",
                  "--- Basis ---", self.post_liquefaction_basis]
        if self.warnings:
            lines += ["", "--- Warnings ---"]
            lines += [f"* {w}" for w in self.warnings]
        from geotech_common.choices import choices_lines
        lines += choices_lines(self.choices, indent="")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        b, a = self.before, self.after
        d = {
            "post_liquefaction_basis": self.post_liquefaction_basis,
            "warnings": list(self.warnings),
            "methods_range_after": a.methods_range,
            "liquefied_side_resistance": self.liquefied_side_resistance,
            "side_resistance_name": self.side_resistance_name,
            "headline_method_before": b.headline_method,
            "headline_method_after": a.headline_method,
            "neutral_plane_before_m": b.neutral_plane_depth,
            "neutral_plane_after_m": a.neutral_plane_depth,
            "drag_force_before_kN": b.drag_force,
            "drag_force_after_kN": a.drag_force,
            "max_pile_load_before_kN": b.max_pile_load,
            "max_pile_load_after_kN": a.max_pile_load,
            "dragload_after_kN": a.dragload,
            "pile_settlement_before_mm": b.pile_settlement * 1000.0,
            "pile_settlement_after_mm": a.pile_settlement * 1000.0,
            "reconsolidation_settlement_mm":
                self.reconsolidation_settlement * 1000.0,
            "reconsolidation_below_toe_mm":
                self.reconsolidation_below_toe * 1000.0,
            "liquefied_zones": self.liquefied_zones,
            "side_resistance_comparison": self.side_resistance_comparison,
            "by_method": self.by_method,
            "limit_states_note": self.limit_states_note,
            "choices": [dict(c) for c in self.choices],
        }
        if self.fs_liq_threshold is not None:
            d["fs_liq_threshold"] = self.fs_liq_threshold
        if a.structural_ok is not None:
            d["structural_ok"] = a.structural_ok
            d["structural_demand_kN"] = a.structural_demand
        if a.geotechnical_ok is not None:
            d["geotechnical_ok"] = a.geotechnical_ok
        if a.settlement_ok is not None:
            d["settlement_ok"] = a.settlement_ok
        d["before"] = {
            "neutral_plane_depth_m": b.neutral_plane_depth,
            "headline_method": b.headline_method,
            "neutral_plane_basis": b.neutral_plane_basis,
            "methods_range": b.methods_range,
            "drag_force_kN": b.drag_force,
            "max_pile_load_kN": b.max_pile_load,
            "pile_settlement_m": b.pile_settlement,
            "warnings": list(b.warnings),
        }
        d["after"] = a.to_dict()
        return d


def _available(key: str, layers: Sequence[LiquefiedLayer],
               ratio: Optional[float]) -> str:
    """Why option ``key`` cannot be evaluated, or ''."""
    if key == "ratio" and ratio is None:
        return "needs liquefied_side_resistance_ratio"
    if key == "residual_strength":
        missing = [f"{s.top:g}-{s.bottom:g} m" for s in layers
                   if s.su_r is None and s.N1_60 is None]
        if missing:
            return ("needs su_r or N1_60 for each liquefied layer (none at "
                    + ", ".join(missing[:4])
                    + (", ..." if len(missing) > 4 else "") + ")")
    return ""


#: The neutral-plane methods whose plane follows the soil settlement.
_FOLLOW_SETTLEMENT = ("settlement_compatibility", "bearing_layer_top")

#: What each method that does not follow the settlement counts as drag.
_COUNTS_AS_DRAG = {
    "force_equilibrium": (
        "force equilibrium (UFC 3-220-20 §6-7.4) counts all the friction "
        "above the intersection of the load and resistance curves as "
        "negative"),
    "pile_toe": ("the plane at the pile toe counts the whole shaft (GEC-12 "
                 "§7.3.5.7: the upper bound)"),
    "endo": ("Endo et al. (1969) sets the plane at a fixed fraction of the "
             "embedment (CGPR #56 Table 3.1)"),
}


def _method_spread_notes(before: DowndragResult, after: DowndragResult,
                         after_analysis: DowndragAnalysis,
                         by_method: list) -> list:
    """Plain words on why the methods' drag forces differ so much: before
    the earthquake when nothing settles, and after it when the headline
    plane lies below the soil that settles."""
    notes = []
    if not np.any(np.asarray(before.soil_settlement_profile) > 0):
        rows = [r for r in by_method
                if r["method"] not in _FOLLOW_SETTLEMENT
                and r["applies_before"]
                and (r["drag_force_before_kN"] or 0.0) > 0.5]
        if rows:
            kinds = [k for k in _COUNTS_AS_DRAG
                     if any(r["method"] == k for r in rows)]
            notes.append(
                "Before the earthquake no soil settles in this model, yet "
                "these methods give a drag force: "
                + "; ".join(f"{r['name']} {r['drag_force_before_kN']:.0f} "
                            f"kN at {r['neutral_plane_before_m']:.2f} m"
                            for r in rows)
                + ". They do not follow the soil settlement: "
                + "; ".join(_COUNTS_AS_DRAG[k] for k in kinds)
                + ". The methods that follow it give no drag before the "
                "earthquake. Compare before and after within one method.")
    if after.neutral_plane_method != "force_equilibrium":
        return notes
    z_se = float(after_analysis._state()["z_se"])
    z_np = float(after.neutral_plane_depth)
    follow = [r for r in by_method
              if r["method"] in _FOLLOW_SETTLEMENT and r["applies_after"]
              and r["drag_force_after_kN"] < after.drag_force - 0.5]
    if z_np > z_se + 1e-6 and follow:
        notes.append(
            f"Why the methods differ so much after liquefaction (how each "
            f"defines the neutral plane, not an error): the headline, "
            f"{after.headline_method}, puts the plane where the load and "
            f"resistance curves meet ({z_np:.2f} m) and counts all the "
            f"friction above it as drag ({after.drag_force:.0f} kN), "
            f"{z_np - z_se:.2f} m of it below {z_se:.2f} m, where the soil "
            f"settles less than the pile: UFC 3-220-20 §6-7.4's "
            f"conservative assumption for the structural drag force. "
            + "; ".join(f"{r['name']} gives {r['drag_force_after_kN']:.0f} "
                        f"kN at {r['neutral_plane_after_m']:.2f} m"
                        for r in follow)
            + ": drag only where the soil settles more than the pile. "
            "GEC-10 §10.7 names the drag of the liquefied layers and of "
            "the layers above them; it says nothing of the soil below.")
    return notes


def post_liquefaction_downdrag(
        analysis: DowndragAnalysis,
        liquefied_layers: Optional[Sequence] = None,
        liquefaction_result: Any = None,
        volumetric_strain=None,
        fs_liq_threshold: float = DEFAULT_FS_LIQ_THRESHOLD,
        liquefied_side_resistance: str = DEFAULT_SIDE_RESISTANCE,
        liquefied_side_resistance_ratio: Optional[float] = None,
) -> PostLiquefactionResult:
    """Downdrag after liquefaction (FHWA GEC-10 §10.7), beside the static
    analysis.

    Parameters
    ----------
    analysis : DowndragAnalysis
        The static analysis (soil, pile, loads, fill / drawdown, neutral
        plane method). Run as given for "before"; "after" is the same with
        the liquefied layers' side resistance and reconsolidation.
    liquefied_layers : list, optional
        :class:`LiquefiedLayer` objects or dicts (``top``, ``bottom``,
        ``volumetric_strain`` or ``settlement``, optional ``su_r``,
        ``N1_60``, ``fs_liq``). Give this or ``liquefaction_result``.
    liquefaction_result : dict or object, optional
        The ``liquefaction`` tool's output (CPT ``depth_profile``, with its
        strains; SPT ``layer_results``) or a profile with ``depth``,
        ``factor_of_safety`` and ``volumetric_strain`` (liquepy's CPT
        result).
    volumetric_strain : float or list, optional
        Reconsolidation strain (decimal) for the readings of
        ``liquefaction_result``: one value, or one per reading. Needed for
        the SPT routes, which compute none (no reference in the library
        prints an SPT method).
    fs_liq_threshold : float
        A reading liquefies where FS_liq <= this. Default 1.0 (GEC-10
        §9.4.2.2, p. 9-42).
    liquefied_side_resistance : str
        The option that gives the headline "after" numbers; every option
        that can be evaluated is reported in ``side_resistance_comparison``:
        ``"half_pre_earthquake"`` (default; GEC-10 §10.7, p. 10-52),
        ``"zero"`` (GEC-10 p. 10-51; UFC 3-220-20 Table 6-49),
        ``"residual_strength"`` (su,r; GEC-10 p. 10-51, GEC-12 §7.4.2.2;
        needs ``su_r`` or ``N1_60`` per layer), ``"ratio"`` (needs
        ``liquefied_side_resistance_ratio``).
    liquefied_side_resistance_ratio : float, optional
        Fraction (0-1) of the pre-earthquake unit side resistance, for
        ``"ratio"``.
    """
    from geotech_common.choices import MODULE, choice, how_given

    if liquefied_side_resistance not in LIQUEFIED_SIDE_RESISTANCE:
        raise ValueError(
            f"liquefied_side_resistance must be one of "
            f"{list(LIQUEFIED_SIDE_RESISTANCE)}, got "
            f"{liquefied_side_resistance!r}")
    ratio = liquefied_side_resistance_ratio
    if ratio is not None and not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f"liquefied_side_resistance_ratio must be between "
                         f"0 and 1, got {ratio}")
    if (liquefied_layers is None) == (liquefaction_result is None):
        raise ValueError(
            "give the liquefied layers one way: liquefied_layers (top, "
            "bottom, volumetric_strain) or liquefaction_result (the "
            "liquefaction tool's output: a CPT's carries its strains, an "
            "SPT's needs volumetric_strain)")
    profile_depth = float(analysis.soil.total_depth)
    left_out = 0.0
    if liquefied_layers is not None:
        if volumetric_strain is not None:
            raise ValueError(
                "volumetric_strain goes with liquefaction_result; with "
                "liquefied_layers give each layer its own")
        layers = [liquefied_layer(x) for x in liquefied_layers]
        threshold = None
    else:
        threshold = float(fs_liq_threshold)
        layers, left_out = liquefied_layers_from_result(
            liquefaction_result, volumetric_strain, threshold,
            profile_depth, water_table=analysis.soil.gwt_depth)
    if not layers:
        raise ValueError(
            "no liquefied layer: no reading has FS_liq <= "
            f"{fs_liq_threshold:g}, so there is no post-liquefaction drag; "
            "run downdrag_analysis for the static case"
            if liquefied_layers is None else
            "liquefied_layers is empty")
    layers = sorted(layers, key=lambda s: s.top)
    for s0, s1 in zip(layers[:-1], layers[1:]):
        if s1.top < s0.bottom - 1e-9:
            raise ValueError(f"liquefied layers overlap: {s0.top:g}-"
                             f"{s0.bottom:g} m and {s1.top:g}-{s1.bottom:g} m")
    if layers[-1].bottom > profile_depth + 1e-9:
        raise ValueError(
            f"a liquefied layer reaches {layers[-1].bottom:g} m, below the "
            f"soil profile ({profile_depth:g} m): extend the profile")
    if liquefied_side_resistance == "ratio" and ratio is None:
        raise ValueError("liquefied_side_resistance='ratio' needs "
                         "liquefied_side_resistance_ratio (0-1)")

    base = {f.name: getattr(analysis, f.name) for f in fields(DowndragAnalysis)}
    before_note = ""
    try:
        before = analysis.compute()
    except ValueError as exc:
        # A method chosen for the state after liquefaction (the plane at
        # the base of the settling soil, say) may find no plane before it,
        # when nothing settles yet: before is then the UFC procedure.
        if "does not locate the neutral plane" not in str(exc):
            raise
        before = DowndragAnalysis(**{**base,
                                     "neutral_plane_method": "auto"}
                                  ).compute()
        before_note = (f"Before the earthquake neutral_plane_method="
                       f"'{analysis.neutral_plane_method}' finds no plane "
                       f"({exc}); the numbers before are the UFC 3-220-20 "
                       f"§6-7.4 procedure ('auto'). After liquefaction they "
                       f"are the method chosen.")
    chosen = liquefied_side_resistance
    keys = [k for k in LIQUEFIED_SIDE_RESISTANCE
            if k != "ratio" or ratio is not None]
    rows, runs = [], {}
    for key in keys:
        spec = _SIDE[key]
        row = {"name": spec["name"], "option": key,
               "source": spec["source"], "assumptions": spec["assumptions"],
               "applies": False, "used": key == chosen, "reason": "",
               "headline_method": None,
               **{k: None for k in _ROW_KEYS},
               "soil_settlement_at_np_mm": None}
        rows.append(row)
        why = _available(key, layers, ratio)
        if why:
            if key == chosen:
                raise ValueError(f"liquefied_side_resistance='{key}' "
                                 f"{why}.")
            row["reason"] = why
            continue
        after_analysis = _AfterLiquefaction(
            **base, liquefied=tuple(layers), side_resistance=key,
            side_ratio=0.5 if ratio is None else float(ratio))
        try:
            res = after_analysis.compute()
        except ValueError as exc:
            if key == chosen:
                raise
            row["reason"] = str(exc)
            continue
        runs[key] = (after_analysis, res)
        row.update({
            "applies": True,
            "headline_method": res.headline_method,
            "neutral_plane_depth_m": round(res.neutral_plane_depth, 3),
            "max_pile_load_kN": round(res.max_pile_load, 1),
            "dragload_kN": round(res.dragload, 1),
            "drag_force_kN": round(res.drag_force, 1),
            "pile_settlement_mm": round(res.pile_settlement * 1000.0, 2),
            "soil_settlement_at_np_mm":
                round(res.soil_settlement_at_np * 1000.0, 2),
        })
    after_analysis, after = runs[chosen]

    # The liquefied zones: settlement, side resistance before and after,
    # where they sit against the neutral plane before the earthquake
    # (FHWA GEC-12 §7.4.2.2, Fellenius and Siegel 2008).
    L = float(analysis.pile_length)
    P = float(analysis.pile_perimeter)
    z_np0 = float(before.neutral_plane_depth)
    planes_before = [(r["name"], float(r["neutral_plane_depth_m"]))
                     for r in before.comparison_rows() if r["applies"]]
    zones = []
    for zone in _zones(layers):
        top, bottom = zone[0].top, zone[-1].bottom
        settle = sum(s.settlement for s in zone)
        fs_vals = [s.fs_liq for s in zone if s.fs_liq is not None]
        if top >= L - 1e-9:
            where = "below the pile toe"
        elif bottom <= z_np0 + 1e-9:
            where = (f"above the neutral plane before the earthquake "
                     f"({z_np0:.2f} m)")
        elif top >= z_np0 - 1e-9:
            where = (f"below the neutral plane before the earthquake "
                     f"({z_np0:.2f} m)")
        else:
            where = (f"across the neutral plane before the earthquake "
                     f"({z_np0:.2f} m)")
        lo, hi = top, min(bottom, L)
        zones.append({
            "top_m": round(top, 3), "bottom_m": round(bottom, 3),
            "thickness_m": round(bottom - top, 3),
            "volumetric_strain": round(settle / (bottom - top), 5),
            "reconsolidation_settlement_mm": round(settle * 1000.0, 2),
            "fs_liq_min": (round(min(fs_vals), 3) if fs_vals else None),
            "side_resistance_before_kN": round(_shaft_between(
                before.z, before.unit_skin_friction, lo, hi, P), 1)
                if hi > lo else 0.0,
            "side_resistance_after_kN": round(_shaft_between(
                after.z, after.unit_skin_friction, lo, hi, P), 1)
                if hi > lo else 0.0,
            "position": where,
            "below_the_plane_before_of": [
                name for name, z in planes_before
                if top < L - 1e-9 and top >= z - 1e-6],
        })
    recon_total = float(sum(s.settlement for s in layers))
    below_toe = after_analysis.reconsolidation_below_toe()

    # Every neutral-plane method, before and after
    by_method = []
    old = {r["name"]: r for r in before.comparison_rows()}
    for r in after.comparison_rows():
        o = old.get(r["name"], {})
        entry = {"name": r["name"], "method": r["method"],
                 "toe_mobilization": r["toe_mobilization"],
                 "applies_before": bool(o.get("applies")),
                 "applies_after": bool(r["applies"])}
        for state, src in (("before", o), ("after", r)):
            entry[f"neutral_plane_{state}_m"] = src.get(
                "neutral_plane_depth_m")
            entry[f"drag_force_{state}_kN"] = src.get("drag_force_kN")
            entry[f"pile_settlement_{state}_mm"] = src.get(
                "pile_settlement_mm")
            if not src.get("applies"):
                entry[f"reason_{state}"] = src.get("reason", "")
        by_method.append(entry)

    # Basis, warnings, choices
    spec = _SIDE[chosen]
    static = analysis.fill_thickness > 0 or analysis.gw_drawdown > 0
    basis = (
        f"FHWA GEC-10 §10.7 'Post-Liquefaction Downdrag' (p. 10-52): the "
        f"reconsolidation settlement of the liquefied layers, and of the "
        f"soil above them, drags the pile; the drag is largest at the end "
        f"of reconsolidation, when the effective stress is back to about "
        f"its pre-earthquake value. After: the liquefied layers carry "
        f"{spec['name'].lower()} ({spec['source']}); the other layers keep "
        f"their static side resistance (GEC-10: 'For geomaterial layers "
        f"above the zone of liquefaction, drag loads should be based on "
        f"side resistance values corresponding to static conditions'). "
        f"The soil settles by the reconsolidation settlement of the "
        f"liquefied layers (volumetric strain x thickness, "
        f"{recon_total * 1000:.1f} mm in all"
        + (f", {below_toe * 1000:.1f} mm of it below the toe" if below_toe
           > 0 else "")
        + (") on top of the fill / drawdown settlement of the static "
           "analysis" if static else ")")
        + ". The neutral plane, drag force and settlement are then found as "
        "for static downdrag (GEC-10 §10.6.2, which defers to GEC-12 "
        "§7.3.6.1), every method side by side, before and after. FHWA "
        "GEC-12 §7.4.2.2 (pp. 384-385, Fellenius and Siegel 2008): "
        "'If liquefaction occurs in soil layers above the location of the "
        "neutral plane before liquefaction, the liquefaction event will "
        "have limited effect on the pile. If liquefaction occurs in soil "
        "layers located below the pre-liquefaction location of the neutral "
        "plane, it will increase the axial compression load in the pile as "
        "well as result in additional pile settlement.'")
    limit_note = (
        "Limit states: GEC-10 §10.7 (p. 10-52) checks post-liquefaction "
        "drag at the structural strength and service limit states only "
        "(§10.6), as here (structural: UFC 3-220-20 Eq 6-80 / GEC-12 Eq "
        "7-70; geotechnical without the drag; settlement). AASHTO (2017), "
        "as GEC-10 reports it, includes it in Extreme Event I with a load "
        "factor of 1.0, which GEC-10 calls misguided because "
        "reconsolidation follows the shaking by hours or days.")

    warnings = [before_note] if before_note else []
    warnings += _method_spread_notes(before, after, after_analysis,
                                     by_method)
    gwt = float(analysis.soil.gwt_depth)
    dry = [f"{s.top:g}-{s.bottom:g} m" for s in layers if s.top < gwt - 1e-9]
    if dry:
        warnings.append(
            f"Liquefied layer(s) {', '.join(dry)} reach above the water "
            f"table ({gwt:g} m); soil above the water table does not "
            f"liquefy. Check the layers or gwt_depth.")
    for s in layers:
        if abs(s.top - L) < 1e-9:
            where = "on the top of"
        elif abs(s.bottom - L) < 1e-9:
            where = "at the base of"
        elif s.top < L < s.bottom:
            where = "in"
        else:
            continue
        warnings.append(
            f"The pile toe ({L:.2f} m) is {where} a liquefied layer "
            f"({s.top:.2f}-{s.bottom:.2f} m). {_GEC10_TOE} The toe "
            f"resistance here is the static value.")
        break
    else:
        below = [s for s in layers if s.top > L]
        if below:
            s = below[0]
            warnings.append(
                f"A liquefied layer lies below the pile toe ({s.top:.2f}-"
                f"{s.bottom:.2f} m, {s.top - L:.2f} m below it); its "
                f"reconsolidation settlement takes the toe down with it. "
                f"{_GEC10_TOE}")
    under = [z for z in zones if z["below_the_plane_before_of"]]
    if under and after.drag_force <= before.drag_force + 1e-6:
        # GEC-12 §7.4.2.2: liquefaction below the plane adds load. Name
        # the methods that follow the reconsolidation settlement.
        rises = []
        for r in by_method:
            if not r["applies_after"]:
                continue
            b0 = (r["drag_force_before_kN"] if r["applies_before"]
                  else None)
            a1 = r["drag_force_after_kN"]
            if b0 is None or a1 > b0 + 0.5:
                rises.append(
                    f"{r['name']}: {a1:.0f} kN at "
                    f"{r['neutral_plane_after_m']:.2f} m"
                    + (f" (before {b0:.0f} kN)" if b0 is not None
                       else " (none before)"))
        planes = dict(planes_before)
        # One sentence for the zones below the same planes (GM-207: a
        # noisy CPT gives twenty zones, most of them below the same ones).
        groups: dict = {}
        for z in under:
            groups.setdefault(tuple(z["below_the_plane_before_of"]),
                              []).append(z)
        text = "; ".join(
            ("The liquefied zone " if len(zs) == 1
             else "The liquefied zones ")
            + ", ".join(f"{z['top_m']:g}-{z['bottom_m']:g}" for z in zs)
            + (" m lies" if len(zs) == 1 else " m lie")
            + " below the neutral plane these methods give before the "
            "earthquake: "
            + ", ".join(f"{n} ({planes[n]:.2f} m)" for n in names)
            for names, zs in groups.items()) + "."
        text += (
            f" FHWA GEC-12 §7.4.2.2 "
            f"(Fellenius and Siegel 2008): liquefaction there 'will "
            f"increase the axial compression load in the pile as well as "
            f"result in additional pile settlement'. The headline method "
            f"({after.headline_method}) gives a drag force of "
            f"{after.drag_force:.0f} kN after against "
            f"{before.drag_force:.0f} kN before"
            + (": it takes the plane from the load and resistance curves, "
               "which the reconsolidation settlement does not move."
               if after.neutral_plane_method == "force_equilibrium"
               else "."))
        if rises:
            text += (" The methods whose drag force rises: "
                     + "; ".join(rises) + ".")
        warnings.append(text)
    if chosen == "half_pre_earthquake":
        cohesive = []
        for s in layers:
            mid = 0.5 * (s.top + min(s.bottom, L))
            if s.top < L and analysis.soil.layer_at_depth(mid).soil_type \
                    == "cohesive":
                cohesive.append(f"{s.top:g}-{s.bottom:g} m")
        if cohesive:
            warnings.append(
                "GEC-10 §10.7 states the 50 percent for side resistance "
                "calculated by the beta method; the liquefied layer at "
                + ", ".join(cohesive) + " lies in a cohesive layer (alpha "
                "method), and takes 50 percent of alpha x cu.")
    if "residual_strength" in runs:
        capped = sorted(getattr(runs["residual_strength"][0], "capped",
                                set()))
        if capped:
            spans = [f"{a:g}-{b:g} m" for a, b in capped]
            warnings.append(
                "In the residual strength option su,r is more than the "
                "pre-earthquake unit side resistance in part of " + ", ".join(spans[:4])
                + (", ..." if len(spans) > 4 else "")
                + "; the pre-earthquake value is kept there (the residual "
                "strength option never adds resistance).")
    if left_out > 0:
        warnings.append(
            f"The liquefaction result gives {left_out * 1000:.1f} mm of "
            f"reconsolidation settlement at readings with FS_liq above "
            f"{threshold:g}, left out here: only the readings at or below "
            f"fs_liq_threshold count as liquefied. GEC-10 §9.4.2.2 (p. "
            f"9-42) treats 1.0 < FSliq ≲ 1.5 as generating excess pore "
            f"pressure.")
    warnings += [f"After liquefaction: {w}" for w in after.warnings]

    out_choices = [
        choice("side resistance of the liquefied layers", spec["name"],
               how_given(chosen, DEFAULT_SIDE_RESISTANCE,
                         name="liquefied_side_resistance"),
               spec["source"],
               [_SIDE[k]["name"] for k in keys if k != chosen]),
    ]
    if liquefied_layers is not None:
        out_choices.append(choice(
            "liquefied layers",
            "; ".join(f"{z['top_m']:g}-{z['bottom_m']:g} m" for z in zones),
            how_given(liquefied_layers, None, name="liquefied_layers")))
    else:
        out_choices.append(choice(
            "liquefied layers",
            "; ".join(f"{z['top_m']:g}-{z['bottom_m']:g} m" for z in zones),
            MODULE,
            f"readings of the liquefaction result with FS_liq <= "
            f"{threshold:g}, each standing for the depth halfway to its "
            f"neighbours, below the water table ({analysis.soil.gwt_depth:g} "
            f"m)"))
        out_choices.append(choice(
            "FS_liq at or below which a reading liquefies", threshold,
            how_given(threshold, DEFAULT_FS_LIQ_THRESHOLD,
                      name="fs_liq_threshold"),
            "GEC-10 §9.4.2.2, p. 9-42: 'layers predicted to undergo "
            "liquefaction (FSliq ≤ 1.0)'"))
        if volumetric_strain is not None:
            out_choices.append(choice(
                "reconsolidation volumetric strain",
                (volumetric_strain if not isinstance(
                    volumetric_strain, (list, tuple, np.ndarray))
                 else "one value per reading"),
                how_given(volumetric_strain, None, name="volumetric_strain"),
                "GEC-10 §10.7: Ishihara and Yoshimine (1992), as Idriss "
                "and Boulanger (2008) describe it"))
        else:
            out_choices.append(choice(
                "reconsolidation volumetric strain",
                "the liquefaction result's own",
                how_given(liquefaction_result, None,
                          name="liquefaction_result"),
                _strain_basis(liquefaction_result)))
    if chosen == "residual_strength":
        out_choices.append(choice(
            "residual strength su,r",
            "GEC-10 Eq 9-13 from (N1)60 and the pre-earthquake sigma'v0 "
            "where su_r is not given",
            MODULE, "GEC-10 p. 9-43 (Kramer and Wang 2015, median), never "
            "more than the pre-earthquake unit side resistance"))
    if static:
        out_choices.append(choice(
            "soil settlement after liquefaction",
            "fill / drawdown settlement + reconsolidation settlement",
            MODULE, "the soil settlement of the static analysis stays and "
            "the reconsolidation settlement adds to it; neither GEC-10 "
            "§10.7 nor GEC-12 §7.4.2.2 says how the two combine"))
    out_choices.append(choice(
        "limit states for the post-liquefaction drag",
        "structural strength and service (GEC-10 §10.7)", MODULE,
        "GEC-10 p. 10-52: 'Post-liquefaction drag loads should therefore "
        "be considered only for structural strength and service limit "
        "states'",
        ["AASHTO (2017) Extreme Event I, load factor 1.0 on the drag"]))

    after.below_toe = dict(after.below_toe)
    if below_toe > 0:
        after.below_toe["reconsolidation_below_toe_mm"] = below_toe * 1000.0
        after.below_toe["basis"] = (
            after.below_toe.get("basis", "") + f" The liquefied soil below "
            f"the toe adds {below_toe * 1000:.1f} mm of reconsolidation "
            f"settlement there (counted with the fill / drawdown "
            f"settlement above).")

    return PostLiquefactionResult(
        before=before, after=after,
        liquefied_side_resistance=chosen,
        side_resistance_comparison=rows,
        liquefied_zones=zones,
        by_method=by_method,
        reconsolidation_settlement=recon_total,
        reconsolidation_below_toe=below_toe,
        post_liquefaction_basis=basis,
        limit_states_note=limit_note,
        warnings=warnings,
        choices=out_choices,
        fs_liq_threshold=threshold,
    )
