"""
Smith soil model for wave equation analysis.

Implements the classic Smith (1960) soil resistance model with
quake and damping parameters. Distributes total soil resistance
to individual pile segments for skin friction and end bearing.

The static spring is a true elasto-plastic (kinematic) spring with
memory: it loads linearly to the quake, yields at R_ultimate, and
unloads along the same elastic slope R_ultimate/quake from the
maximum displacement, leaving a permanent (plastic) offset (GEC-12
Vol 2 Fig. 12-2, p. 157). The permanent set per blow is GEC-12's: the
maximum pile toe displacement less a weighted average of the shaft and
toe quakes (Sec. 12.3, Vol II p. 158). The page names no weights; the shares of
R_ult are this module's assumption (``SoilSetup.weighted_quake``), and
``quake_weighting_choice`` states it with the other weightings' blow
counts.

Two Smith damping variants are provided (``damping_model``), as UFC
3-220-20 (2025) Sec. 6-9.2, pp. 487-488 prints them:

* ``"smith"`` (default) — "the product of the damping factor, pile
  velocity, and mobilized static resistance": R_d = J * R_static * v,
  i.e. R = R_s * (1 + J*v). GEC-12 Vol 2 (FHWA-NHI-16-010) Sec. 12.3,
  p. 156 states the same, and Sec. 12.7.1, p. 242 makes Smith damping
  the usual setting.
* ``"smith_viscous"`` — "uses the fully-mobilized static resistance in
  place of the mobilized static resistance", so the dynamic resistance
  is proportional to the pile velocity: R_d = J * R_ultimate * v
  (GEC-12 Vol 2 p. 242: for a residual stress analysis or a vibratory
  hammer, and sometimes to match field measurements).

Typical quakes and damping factors: GEC-12 Vol 2 Sec. 12.6.7, pp.
211-212, and UFC 3-220-20 p. 487 (see SmithSoilModel).

The shaft resistance acts over the pile's embedded length, the toe
resistance at the pile toe (GM-301). ``SoilSetup.embedded_length`` is the
penetration below grade (GEC-12 Vol 2 p. 224: "Penetration refers to the
analyzed pile toe penetration below grade"; Example 6, p. 182, a 66 ft pile
with "an embedded length of 52.5 feet"); left out, the whole pile. Over it
the shaft resistance is spread by ``shaft_distribution``, as GEC-12's
examples print it: "uniform" (Example 5, Vol II p. 179: "92% uniform shaft
resistance") or "triangular", rising linearly from none at grade to its
largest at the toe (Example 1, p. 162, Fig. 12-3; Example 6, p. 182: "The
shaft resistance will be distributed triangularly along the embedded
portion of the pile shaft"; Vol 2 p. 33, "linearly increasing soil
resistance distribution (triangular)"). Each segment takes the share of
the distribution over its length.

All units are SI: kN, m, seconds.

References:
    Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    FHWA GEC-12 Vol 2, Sec. 12.3 (pp. 155-158) and 12.6.7 (pp. 210-212)
    UFC 3-220-20 (2025), Sec. 6-9.2, pp. 487-488
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_DAMPING_MODELS = ("smith", "smith_viscous")

#: How the shaft resistance is spread over the embedded length (GM-301),
#: as GEC-12 Vol 2's examples print it (pp. 162, 179, 182).
SHAFT_DISTRIBUTIONS = ("uniform", "triangular")


@dataclass
class SmithSoilModel:
    """Smith soil resistance model for a single pile segment.

    The total resistance at a segment is the elasto-plastic static
    resistance plus a velocity-dependent damping term whose form
    depends on ``damping_model`` (see module docstring).

    The static spring is STATEFUL: it remembers its plastic offset, so
    calls must be made in time order within a blow. Create fresh models
    (or call :meth:`reset_state`) for each new blow.
    ``SoilSetup.create_segment_models`` returns fresh models, and
    ``simulate_blow`` calls it once per blow, so each blow starts from
    an unloaded, zero-offset state.

    Parameters
    ----------
    R_ultimate : float
        Ultimate static resistance at this segment (kN).
    quake : float
        Elastic limit displacement (m). Shaft: 0.10 in (2.5 mm), "usually
        satisfactory" (GEC-12 Vol 2 p. 211). Toe: 0.10 in for a
        non-displacement pile (p. 171); pile diameter or width over 120
        to over 60 for a displacement pile (p. 211; D/60 in medium sand,
        D/120 in very dense soil, pp. 162, 170, 182); 0.04 in (1.0 mm)
        on hard rock (p. 212). UFC 3-220-20 p. 487: base 0.04 to 0.4 in.
    damping : float
        Smith damping factor (s/m). Shaft (GEC-12 Vol 2 p. 211): 0.05 s/ft
        (0.16 s/m) non-cohesive soils, 0.10 s/ft (0.33 s/m) silty sands,
        clayey sands and sandy silts, 0.15 s/ft (0.49 s/m) cohesive silts
        and sandy clays, 0.20 s/ft (0.66 s/m) cohesive soils. Toe: about
        0.15 s/ft (0.49 s/m) for most soils (p. 211); lower, e.g. 0.05
        s/ft, on hard rock (p. 212).
    damping_model : str
        "smith" (default): R_d = J * R_static * v (damping scales with
        the mobilized static resistance; UFC 3-220-20 p. 487).
        "smith_viscous": R_d = J * R_ultimate * v (UFC 3-220-20 p. 488).
    no_tension : bool
        If True the spring cannot develop negative (tensile) static
        resistance — use for the toe spring, which separates from the
        soil on rebound (gap) rather than pulling the pile back down
        (Smith, 1960). While the gap is open the spring carries no
        damping either. Default False (skin friction reverses on
        rebound).
    """
    R_ultimate: float
    quake: float = 0.0025
    damping: float = 0.16
    damping_model: str = "smith"
    no_tension: bool = False

    # Internal elasto-plastic state: plastic (residual) displacement of
    # the spring, and whether a no-tension spring's gap is open. Not init
    # parameters; reset per blow.
    _d_plastic: float = field(default=0.0, init=False, repr=False)
    _gap: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.damping_model not in _DAMPING_MODELS:
            raise ValueError(
                f"damping_model must be one of {_DAMPING_MODELS}, "
                f"got {self.damping_model!r}"
            )

    def reset_state(self) -> None:
        """Reset the plastic offset (start of a new blow)."""
        self._d_plastic = 0.0
        self._gap = False

    @property
    def plastic_displacement(self) -> float:
        """Current plastic (residual) displacement of the spring (m).

        The displacement at which the unloading branch reaches zero
        force. For monotonic loading to a peak d_max it equals
        max(d_max - quake, 0); for the toe spring with equal shaft and toe
        quakes that is the permanent set GEC-12 Vol 2 p. 158 prints.
        """
        return self._d_plastic

    def static_resistance(self, displacement: float) -> float:
        """Compute static soil resistance (elasto-plastic, with memory).

        Loading: R = (R_ultimate / quake) * (d - d_plastic), capped at
        +R_ultimate. On yielding, the plastic offset advances so that
        unloading follows the elastic slope R_ultimate/quake from the
        maximum displacement — a permanent offset remains (true Smith
        spring; the pre-v5.1 implementation was reversible and had no
        plastic memory).

        Reverse yielding at -R_ultimate is allowed for skin springs
        (friction reverses on rebound); a ``no_tension`` spring (toe)
        instead unloads to zero and gaps.

        NOTE: this method MUTATES the spring state (plastic offset) when
        the trial resistance exceeds the yield surface. Call in time
        order; use ``reset_state()`` between blows.

        Parameters
        ----------
        displacement : float
            Pile segment displacement (m). Positive downward.

        Returns
        -------
        float
            Static resistance force (kN). Positive opposes downward motion.
        """
        if self.R_ultimate == 0:
            return 0.0
        k = self.R_ultimate / self.quake
        R_trial = k * (displacement - self._d_plastic)
        self._gap = self.no_tension and R_trial < 0.0
        if R_trial > self.R_ultimate:
            # Plastic loading: advance the plastic offset
            self._d_plastic = displacement - self.quake
            return self.R_ultimate
        if R_trial < 0.0 and self.no_tension:
            # Toe spring: no tension — gap opens, offset retained
            return 0.0
        if R_trial < -self.R_ultimate:
            # Reverse plastic (skin friction fully reversed on rebound)
            self._d_plastic = displacement + self.quake
            return -self.R_ultimate
        return R_trial

    def total_resistance(self, displacement: float, velocity: float) -> float:
        """Compute total (static + dynamic) soil resistance.

        damping_model="smith" (default):
            R = R_static + J * |R_static| * v
        i.e. Smith's (1960) R = R_s * (1 + J*v) during loading; the
        |R_static| scaling keeps the damping term dissipative (opposing
        the motion) when the static resistance has reversed sign on
        rebound. Damping scales with the MOBILIZED static resistance
        (UFC 3-220-20 p. 487). It acts throughout the blow (loading and
        rebound) and vanishes naturally when no static resistance is
        mobilized.

        damping_model="smith_viscous":
            R = R_static + J * R_ultimate * v
        Damping proportional to the ultimate (fully mobilized) resistance,
        so proportional to the pile velocity, throughout the blow (UFC
        3-220-20 pp. 487-488). The two variants differ most early in the
        blow (small mobilized R_s, high velocity). Before GM-294 this
        variant damped only while velocity and displacement had the same
        sign, which no held reference prints.

        A no-tension (toe) spring whose gap is open carries no
        resistance, static or damping: the pile toe has left the soil.

        Parameters
        ----------
        displacement : float
            Pile segment displacement (m).
        velocity : float
            Pile segment velocity (m/s). Positive downward.

        Returns
        -------
        float
            Total resistance force (kN).
        """
        R_s = self.static_resistance(displacement)
        if self._gap:
            return 0.0

        if self.damping_model == "smith":
            # Smith (1960): damping ~ mobilized static (UFC p. 487).
            return R_s + self.damping * abs(R_s) * velocity

        # "smith_viscous": damping ~ R_ultimate (UFC pp. 487-488).
        return R_s + self.damping * self.R_ultimate * velocity


@dataclass
class SoilSetup:
    """Complete soil setup for wave equation analysis.

    Distributes ultimate resistance among pile segments as skin
    friction and end bearing.

    Parameters
    ----------
    R_ultimate : float
        Total ultimate static resistance (kN).
    skin_fraction : float
        Fraction of Rult carried by skin friction (0 to 1).
    quake_side : float
        Side quake (m). Default 0.0025 m (0.10 in, GEC-12 Vol 2 p. 211).
    quake_toe : float
        Toe quake (m). Default 0.0025 m (0.10 in), GEC-12 Vol 2's toe
        quake for a non-displacement pile (p. 171); a displacement pile
        takes D/120 to D/60 (p. 211). See SmithSoilModel.
    damping_side : float
        Side Smith damping (s/m). Default 0.16 (0.05 s/ft, non-cohesive
        soils, GEC-12 Vol 2 p. 211). See SmithSoilModel.
    damping_toe : float
        Toe Smith damping (s/m). Default 0.49 (0.15 s/ft, most soils,
        GEC-12 Vol 2 p. 211).
    damping_model : str
        "smith" (default, damping ~ mobilized R_static) or
        "smith_viscous" (damping ~ R_ultimate). See SmithSoilModel.
    shaft_distribution : str
        How the shaft resistance is spread over the embedded length
        (GM-301): "uniform" (default; GEC-12 Vol 2 Example 5, p. 179) or
        "triangular", from none at grade to its largest at the toe
        (Examples 1 and 6, pp. 162, 182). See the module docstring.
    embedded_length : float, optional
        The pile's penetration below grade (m); the shaft resistance acts
        only below grade. None (default): the whole pile. No more than the
        pile length (GEC-12 Vol 2 p. 227).
    """
    R_ultimate: float
    skin_fraction: float = 0.5
    quake_side: float = 0.0025
    quake_toe: float = 0.0025
    damping_side: float = 0.16
    damping_toe: float = 0.49
    damping_model: str = "smith"
    shaft_distribution: str = "uniform"
    embedded_length: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.skin_fraction <= 1:
            raise ValueError(
                f"skin_fraction must be 0-1, got {self.skin_fraction}"
            )
        if self.damping_model not in _DAMPING_MODELS:
            raise ValueError(
                f"damping_model must be one of {_DAMPING_MODELS}, "
                f"got {self.damping_model!r}"
            )
        if self.shaft_distribution not in SHAFT_DISTRIBUTIONS:
            raise ValueError(
                f"shaft_distribution must be one of {SHAFT_DISTRIBUTIONS}, "
                f"got {self.shaft_distribution!r}")
        if self.embedded_length is not None and not self.embedded_length > 0:
            raise ValueError(
                f"embedded_length must be more than 0 m, got "
                f"{self.embedded_length}")

    @property
    def R_skin(self) -> float:
        """Total skin friction resistance (kN)."""
        return self.R_ultimate * self.skin_fraction

    @property
    def R_toe(self) -> float:
        """Toe (end bearing) resistance (kN)."""
        return self.R_ultimate * (1.0 - self.skin_fraction)

    @property
    def weighted_quake(self) -> float:
        """The shaft and toe quakes averaged with the share of R_ult each
        carries (m): what the permanent set subtracts from the maximum
        pile toe displacement. GEC-12 Vol 2 Sec. 12.3, p. 158, prints "a
        weighted average of the shaft and toe quakes" without the
        weights; the resistance shares are this module's assumption (see
        quake_weighting_choice)."""
        f = self.skin_fraction
        return f * self.quake_side + (1.0 - f) * self.quake_toe

    def shaft_shares(self, segment_lengths) -> list:
        """Each segment's share of the shaft resistance (GM-301), from the
        pile head down: the distribution's integral over the part of the
        segment below grade, over its integral over the embedded length.
        Grade is ``embedded_length`` above the pile toe (the pile head when
        None). "uniform": the segment's length below grade over the
        embedded length L_e; "triangular", the resistance per metre rising
        linearly from none at grade: ((b - t)^2 - (max(a, t) - t)^2) / L_e^2
        for a segment from a to b below the pile head, grade at t."""
        lengths = [float(x) for x in segment_lengths]
        L = sum(lengths)
        Le = L if self.embedded_length is None else self.embedded_length
        if Le > L * (1.0 + 1e-9):
            raise ValueError(
                f"embedded_length {Le:g} m is more than the pile length "
                f"{L:g} m (GEC-12 Vol 2 p. 227: the penetration 'cannot be "
                f"greater than the pile length')")
        Le = min(Le, L)
        t = L - Le
        shares, a = [], 0.0
        for seg in lengths:
            b = a + seg
            top = max(a, t)
            if b <= t:
                shares.append(0.0)
            elif self.shaft_distribution == "uniform":
                shares.append((b - top) / Le)
            else:
                shares.append(((b - t) ** 2 - (top - t) ** 2) / Le ** 2)
            a = b
        return shares

    def create_segment_models(
        self, n_segments: int, segment_lengths=None
    ) -> tuple:
        """Create soil spring models for each pile segment.

        Spreads the skin friction over the embedded length by
        ``shaft_distribution`` (``shaft_shares``; GM-301); uniformly along
        the whole pile when neither is set, as before. Toe resistance is
        applied to the last segment only.
        The returned models are freshly constructed (zero plastic
        offset) — one set per blow.

        Parameters
        ----------
        n_segments : int
            Number of pile segments.
        segment_lengths : sequence of float, optional
            Each segment's length (m), from the pile head; needed with an
            ``embedded_length`` or a "triangular" distribution.

        Returns
        -------
        side_models : list of SmithSoilModel
            Skin friction springs for each segment.
        toe_model : SmithSoilModel
            End bearing spring at pile toe (no-tension: it gaps on
            rebound rather than pulling the pile back).
        """
        whole = self.embedded_length is None
        if not whole and segment_lengths is not None:
            # an embedded length given as the pile's own: the whole pile
            L = float(sum(segment_lengths))
            if self.embedded_length > L * (1.0 + 1e-9):
                self.shaft_shares(segment_lengths)        # refuses it
            whole = abs(self.embedded_length - L) <= 1e-9 * L
        if whole and self.shaft_distribution == "uniform":
            # Uniform skin friction over the whole pile
            R_per_segment = (self.R_skin / n_segments if n_segments > 0
                             else 0)
            R_segments = [R_per_segment] * n_segments
        else:
            if segment_lengths is None or \
                    len(segment_lengths) != n_segments:
                raise ValueError(
                    "create_segment_models needs the segments' lengths with "
                    "an embedded_length or a triangular shaft distribution")
            R_segments = [self.R_skin * s
                          for s in self.shaft_shares(segment_lengths)]

        side_models = [
            SmithSoilModel(R_seg, self.quake_side, self.damping_side,
                           damping_model=self.damping_model)
            for R_seg in R_segments
        ]

        toe_model = SmithSoilModel(
            self.R_toe, self.quake_toe, self.damping_toe,
            damping_model=self.damping_model, no_tension=True,
        )

        return side_models, toe_model


def quake_weighting_choice(quake_side: float, quake_toe: float,
                           points: list, label: str = "") -> Optional[dict]:
    """The weights of the shaft and toe quakes in the permanent set, as a
    result's ``choices`` entry (GM-294 review). GEC-12 Vol 2 Sec. 12.3,
    p. 158, subtracts "a weighted average of the shaft and toe quakes" from
    the maximum pile toe displacement and names no weights; the shares of
    R_ult (``SoilSetup.weighted_quake``) are this module's assumption. The
    alternatives, a plain average and the toe quake alone (the toe
    spring's plastic offset), carry the blow counts they give on the same
    blows. On GEC-12 Example 6 (Vol II p. 182; 0.40 in toe quake, 30 % shaft) the
    three give 92.6, 63 and 303 blows/ft, GRLWEAP 94.

    Parameters
    ----------
    quake_side, quake_toe : float
        Quakes (m).
    points : list of (float, float)
        One (skin_fraction, maximum pile toe displacement in m) per blow.
    label : str
        What the blows are at, e.g. " at R_ult 300 / 600 kN", when there
        is more than one.

    Returns
    -------
    dict or None
        None when the quakes are equal: every weighting gives the same set.
    """
    if not points or abs(quake_side - quake_toe) < 1e-12:
        return None
    from geotech_common.choices import MODULE, choice

    def blows(q_avgs):
        out = []
        for (_, d_max), q in zip(points, q_avgs):
            s = d_max - q
            out.append("refusal" if s <= 1e-6 else f"{1.0 / s:.1f}")
        return " / ".join(out)

    n = len(points)
    shares = [f * quake_side + (1.0 - f) * quake_toe for f, _ in points]
    plain = (quake_side + quake_toe) / 2.0
    fractions = {round(f, 9) for f, _ in points}
    if len(fractions) == 1:
        f = points[0][0]
        q_avg = (f"Q_avg = {f:.2f} x {quake_side * 1000:.2f} + "
                 f"{1.0 - f:.2f} x {quake_toe * 1000:.2f} = "
                 f"{shares[0] * 1000:.2f} mm")
    else:
        q_avg = "Q_avg = f_skin x Q_side + (1 - f_skin) x Q_toe"
    return choice(
        "weights of the shaft and toe quakes in the permanent set",
        f"the shares of R_ult on the shaft and at the toe, {q_avg}: "
        f"{blows(shares)} blows/m{label}",
        MODULE,
        "GEC-12 Vol 2 Sec. 12.3 (p. 158) subtracts \"a weighted average of "
        "the shaft and toe quakes\" from the maximum pile toe displacement "
        "and names no weights; the shares of R_ult are this module's "
        "assumption, not printed",
        [f"a plain average, Q_avg = {plain * 1000:.2f} mm: "
         f"{blows([plain] * n)} blows/m{label}",
         f"the toe quake alone, Q_avg = {quake_toe * 1000:.2f} mm: "
         f"{blows([quake_toe] * n)} blows/m{label}"])
