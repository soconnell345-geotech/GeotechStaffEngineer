"""
Explicit time-stepping solver for the Smith wave equation model.

Uses central difference time integration to solve the 1-D wave
equation for a hammer-cushion-pile-soil system.

All units are SI: kN, m, kg, seconds.

References:
    Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    WEAP87 Manual, Chapter 5
    FHWA GEC-12, Section 12.4
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

from wave_equation.hammer import Hammer
from wave_equation.cushion import Cushion
from wave_equation.pile_model import PileModel
from wave_equation.soil_model import SoilSetup, SmithSoilModel


@dataclass
class BlowResult:
    """Results from a single hammer blow simulation.

    Attributes
    ----------
    permanent_set : float
        Permanent pile penetration per blow (m).
    max_compression_stress : float
        Maximum compressive stress in pile (kPa).
    max_tension_stress : float
        Maximum tensile stress in pile (kPa).
    max_pile_force : float
        Maximum compressive force in pile (kN).
    max_tension_force : float
        Maximum tensile force in pile (kN).
    time : np.ndarray
        Time array (s).
    pile_head_force : np.ndarray
        Force at pile head vs time (kN).
    pile_head_velocity : np.ndarray
        Velocity at pile head vs time (m/s).
    pile_toe_displacement : np.ndarray
        Displacement at pile toe vs time (m).
    n_steps : int
        Number of time steps computed.
    R_ultimate : float
        Ultimate resistance used (kN).
    """
    permanent_set: float = 0.0
    max_compression_stress: float = 0.0
    max_tension_stress: float = 0.0
    max_pile_force: float = 0.0
    max_tension_force: float = 0.0
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    pile_head_force: np.ndarray = field(default_factory=lambda: np.array([]))
    pile_head_velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    pile_toe_displacement: np.ndarray = field(default_factory=lambda: np.array([]))
    n_steps: int = 0
    R_ultimate: float = 0.0


def compute_time_step(pile: PileModel, cushion_stiffness: float,
                      ram_mass: float) -> float:
    """Compute stable time step satisfying the Courant condition.

    dt <= min(segment_length / wave_speed) for pile segments,
    and also dt <= 2*pi*sqrt(m/k) / (2*pi) = sqrt(m/k) for the
    cushion spring.

    Parameters
    ----------
    pile : PileModel
        Discretized pile.
    cushion_stiffness : float
        Pile cushion stiffness (kN/m).
    ram_mass : float
        Ram mass (kg).

    Returns
    -------
    float
        Stable time step (s).
    """
    # Courant condition for pile segments
    if pile.n_segments > 1:
        dt_pile = np.min(pile.segment_lengths / pile.wave_speeds)
    else:
        dt_pile = pile.segment_lengths[0] / pile.wave_speeds[0]

    # Cushion spring stability
    # k in kN/m -> N/m for mass calc
    m_head = pile.masses[0]  # kg
    k_cushion_N = cushion_stiffness * 1000  # kN/m -> N/m
    if k_cushion_N > 0 and m_head > 0:
        dt_cushion = 0.5 * math.sqrt(m_head / k_cushion_N) * 2 * math.pi / 10
        # Use a fraction of the natural period for stability
        dt_cushion = min(dt_cushion, math.sqrt(ram_mass / k_cushion_N))
    else:
        dt_cushion = dt_pile

    dt = 0.8 * min(dt_pile, dt_cushion)  # Safety factor 0.8
    return dt


def simulate_blow(
    hammer: Hammer,
    cushion: Cushion,
    pile: PileModel,
    soil: SoilSetup,
    helmet_weight: float = 5.0,
    max_time: float = 0.10,
    dt: Optional[float] = None,
    store_interval: int = 10,
) -> BlowResult:
    """Simulate a single hammer blow using Smith's wave equation.

    Models the system as:
        [Ram] --cushion-- [Helmet+PileHead] --pile springs-- ... --[Toe]
                                                soil side springs    toe spring

    Parameters
    ----------
    hammer : Hammer
        Hammer properties.
    cushion : Cushion
        Pile cushion (between helmet and pile head).
    pile : PileModel
        Discretized pile model.
    soil : SoilSetup
        Soil resistance model and parameters.
    helmet_weight : float
        Helmet (drive cap) weight (kN). Default 5 kN.
    max_time : float
        Maximum simulation time (s). Default 0.10 s.
    dt : float, optional
        Time step (s). If None, computed from Courant condition.
    store_interval : int
        Store results every N time steps (to manage memory).

    Returns
    -------
    BlowResult
    """
    n = pile.n_segments

    # Create soil models
    side_models, toe_model = soil.create_segment_models(n)

    # Compute time step
    if dt is None:
        dt = compute_time_step(pile, cushion.stiffness, hammer.ram_mass)

    n_steps = int(max_time / dt) + 1

    # System: ram (index 0) + pile segments (indices 1..n)
    # Total DOFs: n + 1
    n_dof = n + 1

    # Mass array (kg)
    mass = np.zeros(n_dof)
    mass[0] = hammer.ram_mass  # Ram
    mass[1] = pile.masses[0] + helmet_weight / 9.81 * 1000  # Helmet + first pile segment
    mass[2:] = pile.masses[1:]

    # State arrays
    disp = np.zeros(n_dof)   # displacement (m)
    vel = np.zeros(n_dof)    # velocity (m/s)
    force = np.zeros(n_dof)  # net force (N)

    # Initial condition: ram at impact velocity
    vel[0] = hammer.impact_velocity

    # Pile spring stiffnesses (N/m)
    pile_k_N = np.zeros(n)
    pile_k_N[0] = cushion.stiffness * 1000  # Cushion between ram and pile head
    if n > 1:
        pile_k_N[1:] = pile.spring_stiffnesses * 1000  # kN/m -> N/m

    # Tracking
    max_comp_force = 0.0
    max_tens_force = 0.0
    max_comp_stress = 0.0
    max_tens_stress = 0.0

    # Output storage
    n_store = n_steps // store_interval + 1
    time_arr = np.zeros(n_store)
    head_force_arr = np.zeros(n_store)
    head_vel_arr = np.zeros(n_store)
    toe_disp_arr = np.zeros(n_store)
    store_idx = 0

    # Cushion state for COR
    cushion_max_compression = 0.0
    cushion_unloading = False

    for step in range(n_steps):
        t = step * dt

        # ── Compute forces on each DOF ──
        force[:] = 0.0

        # Cushion spring (between ram and pile head)
        compression = disp[0] - disp[1]
        if compression > cushion_max_compression:
            cushion_max_compression = compression
            cushion_unloading = False
        elif compression < cushion_max_compression:
            cushion_unloading = True

        if compression > 0:
            if cushion_unloading:
                # Unloading with COR
                F_cushion = cushion.stiffness * 1000 * compression * cushion.cor**2
            else:
                F_cushion = cushion.stiffness * 1000 * compression
        else:
            F_cushion = 0.0  # No tension in cushion

        force[0] -= F_cushion  # On ram (opposes downward motion)
        force[1] += F_cushion  # On pile head

        # Pile internal springs
        for i in range(n - 1):
            spring_compression = disp[i + 1] - disp[i + 2]
            F_spring = pile_k_N[i + 1] * spring_compression  # N
            force[i + 1] -= F_spring
            force[i + 2] += F_spring

            # Track pile stress
            stress_Pa = F_spring / (pile.segment_areas[i] * 1e6)  # N / m^2 -> Pa... no
            # F_spring is N, area is m^2, stress = F/A in Pa, convert to kPa
            stress_kPa = F_spring / (pile.segment_areas[i] * 1000)
            if stress_kPa > 0:
                max_comp_stress = max(max_comp_stress, stress_kPa)
                max_comp_force = max(max_comp_force, F_spring / 1000)  # N -> kN
            else:
                max_tens_stress = max(max_tens_stress, abs(stress_kPa))
                max_tens_force = max(max_tens_force, abs(F_spring) / 1000)

        # Soil side resistance
        for i in range(n):
            d_seg = disp[i + 1]  # Pile segment displacement
            v_seg = vel[i + 1]
            R_side = side_models[i].total_resistance(d_seg, v_seg)
            force[i + 1] -= R_side * 1000  # kN -> N, opposes motion

        # Soil toe resistance (on last pile segment)
        d_toe = disp[n]
        v_toe = vel[n]
        R_toe = toe_model.total_resistance(d_toe, v_toe)
        force[n] -= R_toe * 1000

        # ── Update velocities and displacements ──
        # a = F / m
        accel = np.zeros(n_dof)
        for i in range(n_dof):
            if mass[i] > 0:
                accel[i] = force[i] / mass[i]

        vel += accel * dt
        disp += vel * dt

        # Cushion check: pile head must track with head force
        # Also track the first pile spring force for stress
        if n > 0:
            F_head = F_cushion / 1000  # N -> kN
            if F_head > max_comp_force:
                max_comp_force = F_head
                stress = F_head / pile.segment_areas[0]  # kN / m^2 = kPa
                max_comp_stress = max(max_comp_stress, stress)

        # Store results at intervals
        if step % store_interval == 0 and store_idx < n_store:
            time_arr[store_idx] = t
            head_force_arr[store_idx] = F_cushion / 1000  # N -> kN
            head_vel_arr[store_idx] = vel[1]
            toe_disp_arr[store_idx] = disp[n]
            store_idx += 1

        # Check for termination: all velocities near zero and ram rebounding
        if step > 100 and np.all(vel[1:] < 0.01) and vel[0] < 0:
            break

    # Permanent set = final toe displacement minus elastic rebound
    # Elastic rebound = sum of (R_static / k) for all springs
    permanent_set = float(disp[n])
    if permanent_set < 0:
        permanent_set = 0.0

    # Trim stored arrays
    time_arr = time_arr[:store_idx]
    head_force_arr = head_force_arr[:store_idx]
    head_vel_arr = head_vel_arr[:store_idx]
    toe_disp_arr = toe_disp_arr[:store_idx]

    return BlowResult(
        permanent_set=permanent_set,
        max_compression_stress=max_comp_stress,
        max_tension_stress=max_tens_stress,
        max_pile_force=max_comp_force,
        max_tension_force=max_tens_force,
        time=time_arr,
        pile_head_force=head_force_arr,
        pile_head_velocity=head_vel_arr,
        pile_toe_displacement=toe_disp_arr,
        n_steps=step + 1,
        R_ultimate=soil.R_ultimate,
    )
