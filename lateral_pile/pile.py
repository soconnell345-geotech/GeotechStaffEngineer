"""
Pile definition module.

Defines the Pile class for representing single piles or drilled shafts
with geometry, material properties, and flexural rigidity.

Supports solid circular, hollow circular (pipe), H-pile (HP shapes),
concrete-filled pipe (composite), and reinforced concrete sections
with moment-dependent cracked EI (Branson's equation).

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

import numpy as np


@dataclass
class PileSection:
    """A section of pile with constant properties over a depth range.

    Parameters
    ----------
    top : float
        Depth to top of section (m), measured from pile head.
    bottom : float
        Depth to bottom of section (m), measured from pile head.
    EI : float
        Flexural rigidity of this section (kN-m^2).
    """
    top: float
    bottom: float
    EI: float

    def __post_init__(self):
        if self.bottom <= self.top:
            raise ValueError(f"Section bottom ({self.bottom}) must be greater than top ({self.top})")
        if self.EI <= 0:
            raise ValueError(f"EI must be positive, got {self.EI}")


@dataclass
class Pile:
    """Single pile or drilled shaft definition.

    Supports solid circular, hollow circular (pipe), and arbitrary cross-sections.
    For pipe piles, provide diameter and thickness. For solid piles, provide
    diameter only. For arbitrary sections, provide moment_of_inertia directly.

    Parameters
    ----------
    length : float
        Embedded pile length (m).
    diameter : float
        Outer diameter of the pile (m).
    E : float
        Young's modulus of pile material (kPa). Steel ~ 200e6 kPa, concrete ~ 25e6 kPa.
    thickness : float, optional
        Wall thickness for pipe piles (m). If None, pile is treated as solid.
    moment_of_inertia : float, optional
        Moment of inertia (m^4). If None, computed from diameter and thickness.
    sections : list of PileSection, optional
        Variable EI sections along the pile. If provided, overrides uniform E*I.

    Examples
    --------
    Steel pipe pile:
    >>> pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)

    Solid concrete pile:
    >>> pile = Pile(length=15.0, diameter=0.45, E=25e6)

    Custom moment of inertia:
    >>> pile = Pile(length=20.0, diameter=0.6, E=200e6, moment_of_inertia=1.5e-4)
    """
    length: float
    diameter: float
    E: float
    thickness: Optional[float] = None
    moment_of_inertia: Optional[float] = None
    sections: Optional[List[PileSection]] = None

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"Pile length must be positive, got {self.length}")
        if self.diameter <= 0:
            raise ValueError(f"Pile diameter must be positive, got {self.diameter}")
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if self.thickness is not None:
            if self.thickness <= 0:
                raise ValueError(f"Wall thickness must be positive, got {self.thickness}")
            if self.thickness >= self.diameter / 2:
                raise ValueError(f"Wall thickness ({self.thickness}) must be less than radius ({self.diameter/2})")

        # Compute moment of inertia if not provided
        if self.moment_of_inertia is None:
            self.moment_of_inertia = self._compute_moment_of_inertia()

        # Parameter range warnings
        if self.diameter < 0.1:
            warnings.warn(f"Pile diameter {self.diameter} m is unusually small")
        if self.diameter > 5.0:
            warnings.warn(f"Pile diameter {self.diameter} m is unusually large")
        if self.length / self.diameter < 3:
            warnings.warn(f"L/D ratio {self.length/self.diameter:.1f} is very low; "
                          "method may not be applicable")

    def _compute_moment_of_inertia(self) -> float:
        """Compute moment of inertia from geometry.

        Returns
        -------
        float
            Moment of inertia (m^4).
        """
        r_outer = self.diameter / 2.0
        if self.thickness is not None:
            # Hollow circular (pipe pile)
            r_inner = r_outer - self.thickness
            I = math.pi / 4.0 * (r_outer**4 - r_inner**4)
        else:
            # Solid circular
            I = math.pi / 4.0 * r_outer**4
        return I

    @property
    def EI(self) -> float:
        """Flexural rigidity (kN-m^2) for uniform pile."""
        return self.E * self.moment_of_inertia

    @property
    def area(self) -> float:
        """Cross-sectional area (m^2)."""
        r_outer = self.diameter / 2.0
        if self.thickness is not None:
            r_inner = r_outer - self.thickness
            return math.pi * (r_outer**2 - r_inner**2)
        else:
            return math.pi * r_outer**2

    def get_EI_at_depth(self, z: float) -> float:
        """Get flexural rigidity at a given depth.

        Parameters
        ----------
        z : float
            Depth from pile head (m).

        Returns
        -------
        float
            EI at depth z (kN-m^2).
        """
        if self.sections is not None:
            for section in self.sections:
                if section.top <= z <= section.bottom:
                    return section.EI
            # If depth is outside all sections, use the nearest section
            warnings.warn(f"Depth {z} is outside defined sections, using uniform EI")
        return self.EI

    def get_EI_profile(self, depths: np.ndarray) -> np.ndarray:
        """Get EI values at an array of depths.

        Parameters
        ----------
        depths : numpy.ndarray
            Array of depths from pile head (m).

        Returns
        -------
        numpy.ndarray
            Array of EI values (kN-m^2).
        """
        if self.sections is None:
            return np.full_like(depths, self.EI)
        return np.array([self.get_EI_at_depth(z) for z in depths])

    @classmethod
    def from_h_pile(cls, designation: str, length: float,
                    axis: str = 'strong', E: float = 200e6) -> 'Pile':
        """Create a Pile from an AISC HP shape designation.

        Parameters
        ----------
        designation : str
            AISC HP shape (e.g., "HP14x117").
        length : float
            Embedded pile length (m).
        axis : str
            Bending axis: 'strong' uses Ixx, 'weak' uses Iyy. Default 'strong'.
        E : float
            Young's modulus (kPa). Default 200e6 (steel).

        Returns
        -------
        Pile

        Examples
        --------
        >>> pile = Pile.from_h_pile("HP14x117", length=20.0)
        >>> pile = Pile.from_h_pile("HP12x53", length=15.0, axis='weak')
        """
        key = designation.replace(" ", "")
        if key not in _HP_SECTIONS:
            available = ", ".join(sorted(_HP_SECTIONS.keys()))
            raise ValueError(
                f"Unknown HP shape '{designation}'. Available: {available}"
            )

        props = _HP_SECTIONS[key]
        bf = props["bf"]

        if axis == 'strong':
            I = props["Ixx"]
        elif axis == 'weak':
            I = props["Iyy"]
        else:
            raise ValueError(f"axis must be 'strong' or 'weak', got '{axis}'")

        return cls(length=length, diameter=bf, E=E, moment_of_inertia=I)

    @classmethod
    def from_filled_pipe(cls, length: float, diameter: float,
                         thickness: float, E_steel: float = 200e6,
                         fc: float = 28000.0,
                         E_concrete: Optional[float] = None) -> 'Pile':
        """Create a concrete-filled steel pipe pile.

        Computes composite EI = E_steel * I_steel + E_concrete * I_concrete.

        Parameters
        ----------
        length : float
            Embedded pile length (m).
        diameter : float
            Outside diameter (m).
        thickness : float
            Steel wall thickness (m).
        E_steel : float
            Young's modulus of steel (kPa). Default 200e6.
        fc : float
            Concrete compressive strength f'c (kPa). Default 28000 (28 MPa).
            Used to compute E_concrete per ACI 318 if E_concrete is not given.
        E_concrete : float, optional
            Young's modulus of concrete (kPa). If None, computed as
            4700 * sqrt(f'c in MPa), converted to kPa.

        Returns
        -------
        Pile

        Examples
        --------
        >>> pile = Pile.from_filled_pipe(length=20.0, diameter=0.610,
        ...                              thickness=0.0127, fc=35000.0)
        """
        if thickness <= 0:
            raise ValueError(f"Wall thickness must be positive, got {thickness}")
        if thickness >= diameter / 2:
            raise ValueError(
                f"Wall thickness ({thickness}) must be less than "
                f"radius ({diameter / 2})"
            )

        if E_concrete is None:
            fc_MPa = fc / 1000.0
            E_concrete = 4700.0 * math.sqrt(fc_MPa) * 1000.0  # back to kPa

        r_outer = diameter / 2.0
        r_inner = r_outer - thickness
        I_steel = math.pi / 4.0 * (r_outer**4 - r_inner**4)
        I_concrete = math.pi / 4.0 * r_inner**4

        EI_composite = E_steel * I_steel + E_concrete * I_concrete
        # Store as equivalent I at E_steel so EI property works correctly
        I_equiv = EI_composite / E_steel

        return cls(length=length, diameter=diameter, E=E_steel,
                   thickness=thickness, moment_of_inertia=I_equiv)

    @classmethod
    def from_rc_section(cls, length: float,
                        rc_section: 'ReinforcedConcreteSection') -> 'Pile':
        """Create a Pile from a reinforced concrete section definition.

        The pile starts with uncracked EI. When used with
        LateralPileAnalysis.solve(), the solver will automatically iterate
        on moment-dependent EI using Branson's equation if the pile has
        an rc_section attached.

        Parameters
        ----------
        length : float
            Embedded pile length (m).
        rc_section : ReinforcedConcreteSection
            Reinforced concrete section definition.

        Returns
        -------
        Pile

        Examples
        --------
        >>> rc = ReinforcedConcreteSection(
        ...     diameter=0.9, fc=35000.0, fy=420000.0,
        ...     n_bars=12, bar_diameter=0.02546, cover=0.075)
        >>> pile = Pile.from_rc_section(length=15.0, rc_section=rc)
        """
        pile = cls(
            length=length,
            diameter=rc_section.diameter,
            E=rc_section.Ec,
            moment_of_inertia=rc_section.Ig,
        )
        pile.rc_section = rc_section
        return pile


# ── AISC HP Shape Database ──────────────────────────────────────────────
# Section properties for lateral analysis (moment of inertia).
# Values from AISC Steel Construction Manual, 16th Edition.
# Ixx, Iyy in m^4; d, bf, tf, tw in m.

_HP_SECTIONS: Dict[str, Dict] = {
    # Ix/Iy from AISC 16th Ed (in^4), converted: 1 in^4 = 4.162314e-7 m^4
    "HP10x42": {
        "d": 0.2464, "bf": 0.2565, "tf": 0.01067, "tw": 0.01054,
        "Ixx": 87.41e-6, "Iyy": 29.84e-6,   # 210 / 71.7 in^4
    },
    "HP10x57": {
        "d": 0.2537, "bf": 0.2591, "tf": 0.01435, "tw": 0.01435,
        "Ixx": 122.4e-6, "Iyy": 42.04e-6,   # 294 / 101 in^4
    },
    "HP12x53": {
        "d": 0.2997, "bf": 0.3048, "tf": 0.01105, "tw": 0.01105,
        "Ixx": 163.6e-6, "Iyy": 52.86e-6,   # 393 / 127 in^4
    },
    "HP12x63": {
        "d": 0.3023, "bf": 0.3073, "tf": 0.01308, "tw": 0.01308,
        "Ixx": 196.5e-6, "Iyy": 63.68e-6,   # 472 / 153 in^4
    },
    "HP12x74": {
        "d": 0.3073, "bf": 0.3099, "tf": 0.01549, "tw": 0.01537,
        "Ixx": 236.8e-6, "Iyy": 77.42e-6,   # 569 / 186 in^4
    },
    "HP12x84": {
        "d": 0.3124, "bf": 0.3124, "tf": 0.01740, "tw": 0.01740,
        "Ixx": 270.6e-6, "Iyy": 88.66e-6,   # 650 / 213 in^4
    },
    "HP14x73": {
        "d": 0.3454, "bf": 0.3708, "tf": 0.01283, "tw": 0.01283,
        "Ixx": 303.4e-6, "Iyy": 108.6e-6,   # 729 / 261 in^4
    },
    "HP14x89": {
        "d": 0.3505, "bf": 0.3734, "tf": 0.01562, "tw": 0.01562,
        "Ixx": 376.3e-6, "Iyy": 135.7e-6,   # 904 / 326 in^4
    },
    "HP14x102": {
        "d": 0.3556, "bf": 0.3759, "tf": 0.01791, "tw": 0.01791,
        "Ixx": 437.0e-6, "Iyy": 158.2e-6,   # 1050 / 380 in^4
    },
    "HP14x117": {
        "d": 0.3607, "bf": 0.3785, "tf": 0.02045, "tw": 0.02045,
        "Ixx": 507.8e-6, "Iyy": 184.4e-6,   # 1220 / 443 in^4
    },
}


# ── Rebar Size Database ─────────────────────────────────────────────────
# Standard US rebar sizes: bar number -> diameter in meters.
# Ref: ASTM A615 / CRSI

_REBAR_SIZES: Dict[str, float] = {
    "#3": 0.009525,    # 3/8 in = 9.525 mm
    "#4": 0.012700,    # 1/2 in = 12.7 mm
    "#5": 0.015875,    # 5/8 in = 15.875 mm
    "#6": 0.019050,    # 3/4 in = 19.05 mm
    "#7": 0.022225,    # 7/8 in = 22.225 mm
    "#8": 0.025400,    # 1 in = 25.4 mm
    "#9": 0.028651,    # 1.128 in = 28.651 mm
    "#10": 0.032258,   # 1.270 in = 32.258 mm
    "#11": 0.035814,   # 1.410 in = 35.814 mm
    "#14": 0.043002,   # 1.693 in = 43.002 mm
    "#18": 0.057328,   # 2.257 in = 57.328 mm
}


def rebar_diameter(designation: str) -> float:
    """Look up rebar diameter by bar designation.

    Parameters
    ----------
    designation : str
        US rebar designation (e.g., "#8", "#11").

    Returns
    -------
    float
        Bar diameter (m).

    Raises
    ------
    ValueError
        If designation not found.
    """
    key = designation.strip()
    if key not in _REBAR_SIZES:
        available = ", ".join(sorted(_REBAR_SIZES.keys(),
                                     key=lambda s: int(s[1:])))
        raise ValueError(
            f"Unknown rebar size '{designation}'. Available: {available}"
        )
    return _REBAR_SIZES[key]


@dataclass
class ReinforcedConcreteSection:
    """Reinforced concrete circular section for moment-dependent EI.

    Computes cracking moment and effective moment of inertia using
    Branson's equation (ACI 318 / AASHTO). For use with the lateral
    pile solver's cracked-EI iteration.

    Parameters
    ----------
    diameter : float
        Shaft/pile diameter (m).
    fc : float
        Concrete compressive strength f'c (kPa).
    fy : float
        Steel yield strength (kPa). Default 420000 (420 MPa, Grade 60).
    n_bars : int
        Number of longitudinal reinforcing bars.
    bar_diameter : float
        Diameter of each bar (m). Use rebar_diameter() for standard sizes.
    cover : float
        Clear cover to center of reinforcing bars (m).
    E_steel : float
        Young's modulus of reinforcing steel (kPa). Default 200e6.

    Examples
    --------
    >>> rc = ReinforcedConcreteSection(
    ...     diameter=0.9, fc=35000.0, fy=420000.0,
    ...     n_bars=12, bar_diameter=rebar_diameter("#8"), cover=0.075)
    >>> print(f"Ec = {rc.Ec:.0f} kPa")
    >>> print(f"Mcr = {rc.Mcr:.1f} kN-m")
    """
    diameter: float
    fc: float
    fy: float = 420000.0
    n_bars: int = 12
    bar_diameter: float = 0.025400  # #8 bar default
    cover: float = 0.075
    E_steel: float = 200e6

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(f"Diameter must be positive, got {self.diameter}")
        if self.fc <= 0:
            raise ValueError(f"f'c must be positive, got {self.fc}")
        if self.n_bars < 4:
            raise ValueError(f"Need at least 4 bars, got {self.n_bars}")
        if self.bar_diameter <= 0:
            raise ValueError(
                f"Bar diameter must be positive, got {self.bar_diameter}"
            )
        if self.cover <= 0:
            raise ValueError(f"Cover must be positive, got {self.cover}")
        r = self.diameter / 2.0
        r_cage = r - self.cover
        if r_cage <= 0:
            raise ValueError(
                f"Cover ({self.cover} m) exceeds radius ({r} m)"
            )

    @property
    def Ec(self) -> float:
        """Concrete modulus of elasticity (kPa). ACI 318: 4700*sqrt(f'c MPa)."""
        fc_MPa = self.fc / 1000.0
        return 4700.0 * math.sqrt(fc_MPa) * 1000.0  # kPa

    @property
    def n_ratio(self) -> float:
        """Modular ratio Es/Ec."""
        return self.E_steel / self.Ec

    @property
    def As(self) -> float:
        """Total longitudinal steel area (m^2)."""
        return self.n_bars * math.pi / 4.0 * self.bar_diameter**2

    @property
    def Ig(self) -> float:
        """Gross moment of inertia of the circular section (m^4)."""
        r = self.diameter / 2.0
        return math.pi / 4.0 * r**4

    @property
    def fr(self) -> float:
        """Modulus of rupture (kPa). ACI 318: 0.62*sqrt(f'c MPa)."""
        fc_MPa = self.fc / 1000.0
        return 0.62 * math.sqrt(fc_MPa) * 1000.0  # kPa

    @property
    def Mcr(self) -> float:
        """Cracking moment (kN-m). Mcr = fr * Ig / yt."""
        yt = self.diameter / 2.0
        return self.fr * self.Ig / yt

    @property
    def Icr(self) -> float:
        """Cracked transformed moment of inertia (m^4).

        Computes the cracked I for a circular section with bars arranged
        in a ring. Uses an iterative approach to find the neutral axis
        depth where the cracked section is in equilibrium (compression
        concrete = tension steel), then computes I about that axis.
        """
        r = self.diameter / 2.0
        r_cage = r - self.cover
        n = self.n_ratio
        A_bar = math.pi / 4.0 * self.bar_diameter**2

        # Iterative neutral axis search: c measured from compression face
        # For circular section, use bisection on force equilibrium
        c_low = 0.01 * self.diameter
        c_high = 0.95 * self.diameter

        for _ in range(100):
            c = (c_low + c_high) / 2.0
            # Concrete compression: approximate as segment of circle
            # above neutral axis at depth c from top
            C_force, I_conc = self._concrete_compression(c, r)

            # Steel contribution
            T_force = 0.0
            for j in range(self.n_bars):
                angle = 2.0 * math.pi * j / self.n_bars
                # Bar position: distance from top of section
                y_bar = r - r_cage * math.cos(angle)
                if y_bar < c:
                    # Bar in compression zone: (n-1)*A_bar * stress
                    C_force += (n - 1) * A_bar * (c - y_bar) / c
                else:
                    # Bar in tension zone
                    T_force += n * A_bar * (y_bar - c) / c

            if C_force > T_force:
                c_high = c
            else:
                c_low = c

            if abs(C_force - T_force) < 1e-6 * max(C_force, T_force, 1e-12):
                break

        # Compute Icr about the neutral axis at depth c
        # Concrete compression zone I
        _, I_conc = self._concrete_compression(c, r)

        # Steel I about neutral axis
        I_steel = 0.0
        for j in range(self.n_bars):
            angle = 2.0 * math.pi * j / self.n_bars
            y_bar = r - r_cage * math.cos(angle)
            dist = y_bar - c
            if y_bar < c:
                I_steel += (n - 1) * A_bar * dist**2
            else:
                I_steel += n * A_bar * dist**2

        return I_conc + I_steel

    def _concrete_compression(self, c: float, r: float):
        """Compute concrete compression force ratio and I for a circular segment.

        Parameters
        ----------
        c : float
            Neutral axis depth from compression face (m).
        r : float
            Section radius (m).

        Returns
        -------
        force_ratio : float
            Proportional compression force (for equilibrium check).
        I_segment : float
            Moment of inertia of the compression zone about the NA (m^4).
        """
        # Circular segment: the compression zone is the part of the circle
        # from the top to depth c (NA at distance c from compression face).
        # Center of circle is at depth r from top.
        # d_from_center = r - c (positive means NA is above center)
        d = r - c

        if c >= 2 * r:
            # Entire section in compression
            A_comp = math.pi * r**2
            # Centroid at center (depth r from top), NA at c
            y_bar_comp = r
            I_segment = math.pi / 4.0 * r**4 + A_comp * (c - y_bar_comp)**2
            force_ratio = A_comp * 0.5  # simplified linear stress
            return force_ratio, I_segment

        if c <= 0:
            return 0.0, 0.0

        # Half-angle subtended by the chord at depth c
        # cos(theta) = d/r = (r-c)/r
        cos_theta = max(-1.0, min(1.0, d / r))
        theta = math.acos(cos_theta)

        # Area of circular segment (from top to depth c)
        A_comp = r**2 * (theta - math.sin(theta) * math.cos(theta))

        if A_comp < 1e-15:
            return 0.0, 0.0

        # Centroid of segment: distance from center of circle
        # For segment cut by chord at distance d from center:
        y_centroid_from_center = (2.0 * r * math.sin(theta)**3) / (
            3.0 * (theta - math.sin(theta) * math.cos(theta))
        )
        # Centroid depth from top = r - y_centroid_from_center
        y_centroid_from_top = r - y_centroid_from_center

        # I of segment about its own centroid (circular segment formula)
        I_seg_centroid = (r**4 / 4.0) * (
            theta - math.sin(theta) * math.cos(theta)
            + 2.0 * math.sin(theta)**3 * math.cos(theta)
        ) - A_comp * y_centroid_from_center**2

        # Transfer to NA at depth c from top
        dist_to_NA = c - y_centroid_from_top
        I_segment = I_seg_centroid + A_comp * dist_to_NA**2

        # Linear stress force: integral of (c-y)/c * dA over compression zone
        # Simplified as A_comp * (average stress ratio)
        force_ratio = A_comp * (c - y_centroid_from_top) / c

        return force_ratio, I_segment

    def get_effective_EI(self, moment: float) -> float:
        """Compute effective EI at a given moment using Branson's equation.

        Parameters
        ----------
        moment : float
            Applied bending moment (kN-m).

        Returns
        -------
        float
            Effective flexural rigidity (kN-m^2).

        Notes
        -----
        Branson's equation (ACI 318):
            Ie = Icr + (Ig - Icr) * (Mcr / Ma)^3

        where Ie is bounded by Icr <= Ie <= Ig.
        """
        Ma = abs(moment)
        Mcr = self.Mcr
        Ig = self.Ig
        Icr = self.Icr
        Ec = self.Ec

        if Ma <= Mcr:
            return Ec * Ig

        ratio = Mcr / Ma
        Ie = Icr + (Ig - Icr) * ratio**3
        # Clamp to valid range
        Ie = max(Icr, min(Ig, Ie))
        return Ec * Ie
