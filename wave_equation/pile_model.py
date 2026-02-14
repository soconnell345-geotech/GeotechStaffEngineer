"""
Pile discretization for wave equation analysis.

Divides the pile into lumped masses connected by springs for the
Smith model. Each segment has mass, spring stiffness, and a soil
resistance spring attached.

All units are SI: kN, m, kg, seconds.

References:
    Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    WEAP87 Manual, Chapter 3
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PileSegment:
    """A single pile segment for wave equation model.

    Parameters
    ----------
    length : float
        Segment length (m).
    area : float
        Cross-sectional area (m^2).
    elastic_modulus : float
        Elastic modulus (kPa).
    unit_weight_material : float
        Material unit weight (kN/m^3). Steel: 78.5, concrete: 23.6.
    """
    length: float
    area: float
    elastic_modulus: float
    unit_weight_material: float = 78.5

    @property
    def mass(self) -> float:
        """Segment mass (kg)."""
        weight_kN = self.area * self.length * self.unit_weight_material
        return weight_kN / 9.81 * 1000  # kN -> N -> kg

    @property
    def spring_stiffness(self) -> float:
        """Axial spring stiffness (kN/m) = EA/L."""
        return self.elastic_modulus * self.area / self.length

    @property
    def wave_speed(self) -> float:
        """Stress wave speed (m/s) = sqrt(E/rho)."""
        rho = self.unit_weight_material / 9.81 * 1000  # kN/m^3 -> kg/m^3
        return math.sqrt(self.elastic_modulus * 1000 / rho)  # kPa -> Pa


@dataclass
class PileModel:
    """Discretized pile model for wave equation analysis.

    Contains arrays of masses, spring stiffnesses, and segment data
    ready for time integration.

    Attributes
    ----------
    n_segments : int
        Number of pile segments.
    masses : np.ndarray
        Mass of each segment (kg). Shape (n_segments,).
    spring_stiffnesses : np.ndarray
        Spring stiffness between segments (kN/m). Shape (n_segments-1,).
    segment_lengths : np.ndarray
        Length of each segment (m).
    segment_areas : np.ndarray
        Area of each segment (m^2).
    wave_speeds : np.ndarray
        Wave speed in each segment (m/s).
    depth_at_segment : np.ndarray
        Depth to center of each segment below pile head (m).
    total_length : float
        Total pile length (m).
    impedance : float
        Pile impedance Z = EA/c (kN-s/m) at the pile head.
    """
    n_segments: int = 0
    masses: np.ndarray = field(default_factory=lambda: np.array([]))
    spring_stiffnesses: np.ndarray = field(default_factory=lambda: np.array([]))
    segment_lengths: np.ndarray = field(default_factory=lambda: np.array([]))
    segment_areas: np.ndarray = field(default_factory=lambda: np.array([]))
    wave_speeds: np.ndarray = field(default_factory=lambda: np.array([]))
    depth_at_segment: np.ndarray = field(default_factory=lambda: np.array([]))
    total_length: float = 0.0
    impedance: float = 0.0


def discretize_pile(
    length: float,
    area: float,
    elastic_modulus: float,
    segment_length: float = 1.0,
    unit_weight_material: float = 78.5,
) -> PileModel:
    """Discretize a uniform pile into mass-spring segments.

    Parameters
    ----------
    length : float
        Total pile length (m).
    area : float
        Cross-sectional area (m^2).
    elastic_modulus : float
        Elastic modulus (kPa). Steel: 200e6, concrete: 30e6.
    segment_length : float
        Target segment length (m). Actual may differ slightly to
        evenly divide the pile.
    unit_weight_material : float
        Material unit weight (kN/m^3). Default 78.5 (steel).

    Returns
    -------
    PileModel
    """
    if length <= 0:
        raise ValueError(f"Pile length must be positive, got {length}")
    if area <= 0:
        raise ValueError(f"Area must be positive, got {area}")

    n = max(1, round(length / segment_length))
    seg_len = length / n

    seg = PileSegment(seg_len, area, elastic_modulus, unit_weight_material)

    masses = np.full(n, seg.mass)
    spring_k = np.full(n - 1, seg.spring_stiffness) if n > 1 else np.array([])
    seg_lengths = np.full(n, seg_len)
    seg_areas = np.full(n, area)
    w_speeds = np.full(n, seg.wave_speed)

    # Depth to center of each segment
    depths = np.arange(n) * seg_len + seg_len / 2

    # Pile impedance at head
    rho = unit_weight_material / 9.81 * 1000  # kg/m^3
    c = math.sqrt(elastic_modulus * 1000 / rho)  # m/s
    Z = elastic_modulus * area / c  # kPa * m^2 / (m/s) = kN*s/m

    return PileModel(
        n_segments=n,
        masses=masses,
        spring_stiffnesses=spring_k,
        segment_lengths=seg_lengths,
        segment_areas=seg_areas,
        wave_speeds=w_speeds,
        depth_at_segment=depths,
        total_length=length,
        impedance=Z,
    )
