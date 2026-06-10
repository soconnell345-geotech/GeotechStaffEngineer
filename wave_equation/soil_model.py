"""
Smith soil model for wave equation analysis.

Implements the classic Smith (1960) soil resistance model with
quake and damping parameters. Distributes total soil resistance
to individual pile segments for skin friction and end bearing.

The static spring is a true elasto-plastic (kinematic) spring with
memory: it loads linearly to the quake, yields at R_ultimate, and
unloads along the same elastic slope R_ultimate/quake from the
maximum displacement, leaving a permanent (plastic) offset. The
plastic offset of the toe spring is the physical permanent set of
the pile per blow.

Two Smith damping variants are provided (``damping_model``):

* ``"smith"`` (default) — damping proportional to the *mobilized*
  static resistance: R_d = J * R_static * v. This is Smith's (1960)
  original formulation R = R_s * (1 + J*v) and the GRLWEAP default
  ("Smith damping").
* ``"smith_viscous"`` — damping proportional to the *ultimate*
  resistance: R_d = J * R_ultimate * v, applied during loading only.
  This reproduces the pre-v5.1 behavior of this module and is similar
  to GRLWEAP's "Smith viscous damping" option.

All units are SI: kN, m, seconds.

References:
    Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    WEAP87 Manual, Chapter 4
    FHWA GEC-12, Table 12-3 (typical parameters)
    GRLWEAP Manual — damping model options (Smith vs Smith-viscous)
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_DAMPING_MODELS = ("smith", "smith_viscous")


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
        Elastic limit displacement (m). Typical 0.0025 m (0.1 in).
    damping : float
        Smith damping factor (s/m). Typical: 0.16 s/m (skin, sand),
        0.65 s/m (skin, clay), 0.50 s/m (toe).
    damping_model : str
        "smith" (default): R_d = J * R_static * v (damping scales with
        the mobilized static resistance — Smith 1960 / GRLWEAP default).
        "smith_viscous": R_d = J * R_ultimate * v during loading only
        (pre-v5.1 behavior of this module).
    no_tension : bool
        If True the spring cannot develop negative (tensile) static
        resistance — use for the toe spring, which separates from the
        soil on rebound (gap) rather than pulling the pile back down
        (Smith, 1960). Default False (skin friction reverses on rebound).
    """
    R_ultimate: float
    quake: float = 0.0025
    damping: float = 0.16
    damping_model: str = "smith"
    no_tension: bool = False

    # Internal elasto-plastic state: plastic (residual) displacement of
    # the spring. Not an init parameter; reset per blow.
    _d_plastic: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self):
        if self.damping_model not in _DAMPING_MODELS:
            raise ValueError(
                f"damping_model must be one of {_DAMPING_MODELS}, "
                f"got {self.damping_model!r}"
            )

    def reset_state(self) -> None:
        """Reset the plastic offset (start of a new blow)."""
        self._d_plastic = 0.0

    @property
    def plastic_displacement(self) -> float:
        """Current plastic (residual) displacement of the spring (m).

        For the toe spring this is the physical permanent set of the
        pile point: the displacement at which the unloading branch
        reaches zero force. For monotonic loading to a peak d_max it
        equals max(d_max - quake, 0).
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
        rebound. Damping scales with the MOBILIZED static resistance,
        which is the GRLWEAP default Smith damping. It acts throughout
        the blow (loading and rebound) and vanishes naturally when no
        static resistance is mobilized.

        damping_model="smith_viscous":
            R = R_static + J * R_ultimate * v   (loading only)
        Damping proportional to the ultimate (not mobilized) resistance,
        applied only while loading (velocity and displacement in the
        same direction) — the pre-v5.1 behavior of this module, similar
        to GRLWEAP's "Smith viscous" option. The two variants differ
        most early in the blow (small mobilized R_s, high velocity).

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

        if self.damping_model == "smith":
            # Smith (1960) / GRLWEAP default: damping ~ mobilized static.
            return R_s + self.damping * abs(R_s) * velocity

        # "smith_viscous": damping ~ R_ultimate, loading-only gating
        # (pre-v5.1 behavior).
        if velocity > 0 and displacement > 0:
            R_d = self.damping * self.R_ultimate * velocity
            return R_s + R_d
        elif velocity < 0 and displacement < 0:
            R_d = self.damping * self.R_ultimate * abs(velocity)
            return R_s - R_d  # R_s is negative here; R_d adds to magnitude
        else:
            return R_s


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
        Side quake (m). Default 0.0025 m.
    quake_toe : float
        Toe quake (m). Default 0.0025 m.
    damping_side : float
        Side Smith damping (s/m). Default 0.16 (sand).
    damping_toe : float
        Toe Smith damping (s/m). Default 0.50 (sand).
    damping_model : str
        "smith" (default, damping ~ mobilized R_static) or
        "smith_viscous" (damping ~ R_ultimate, loading only —
        pre-v5.1 behavior). See SmithSoilModel.
    """
    R_ultimate: float
    skin_fraction: float = 0.5
    quake_side: float = 0.0025
    quake_toe: float = 0.0025
    damping_side: float = 0.16
    damping_toe: float = 0.50
    damping_model: str = "smith"

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

    @property
    def R_skin(self) -> float:
        """Total skin friction resistance (kN)."""
        return self.R_ultimate * self.skin_fraction

    @property
    def R_toe(self) -> float:
        """Toe (end bearing) resistance (kN)."""
        return self.R_ultimate * (1.0 - self.skin_fraction)

    def create_segment_models(
        self, n_segments: int
    ) -> tuple:
        """Create soil spring models for each pile segment.

        Distributes skin friction uniformly along the pile.
        Toe resistance is applied to the last segment only.
        The returned models are freshly constructed (zero plastic
        offset) — one set per blow.

        Parameters
        ----------
        n_segments : int
            Number of pile segments.

        Returns
        -------
        side_models : list of SmithSoilModel
            Skin friction springs for each segment.
        toe_model : SmithSoilModel
            End bearing spring at pile toe (no-tension: it gaps on
            rebound rather than pulling the pile back).
        """
        # Uniform skin friction distribution
        R_per_segment = self.R_skin / n_segments if n_segments > 0 else 0

        side_models = [
            SmithSoilModel(R_per_segment, self.quake_side, self.damping_side,
                           damping_model=self.damping_model)
            for _ in range(n_segments)
        ]

        toe_model = SmithSoilModel(
            self.R_toe, self.quake_toe, self.damping_toe,
            damping_model=self.damping_model, no_tension=True,
        )

        return side_models, toe_model
