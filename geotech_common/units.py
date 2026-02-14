"""
Unit conversion utilities for geotechnical engineering.

All geotechnical modules use SI units internally:
    Length: meters (m)
    Force: kilonewtons (kN)
    Stress/Pressure: kilopascals (kPa)
    Unit weight: kN/m³
    Moment: kN·m

This module provides conversion functions to/from US Customary units.

Conversion factors reference:
    1 ft = 0.3048 m
    1 in = 0.0254 m
    1 kip = 4.44822 kN
    1 lb = 0.00444822 kN
    1 ksf = 47.8803 kPa
    1 psi = 6.89476 kPa
    1 tsf (US ton/ft²) = 95.7605 kPa
    1 pcf = 0.157087 kN/m³
"""

import functools
from typing import Callable

# ── Length ──────────────────────────────────────────────────────────────

_FT_PER_M = 1.0 / 0.3048
_M_PER_FT = 0.3048
_IN_PER_M = 1.0 / 0.0254
_M_PER_IN = 0.0254
_IN_PER_MM = 1.0 / 25.4
_MM_PER_IN = 25.4


def m_to_ft(m: float) -> float:
    """Convert meters to feet."""
    return m * _FT_PER_M


def ft_to_m(ft: float) -> float:
    """Convert feet to meters."""
    return ft * _M_PER_FT


def m_to_in(m: float) -> float:
    """Convert meters to inches."""
    return m * _IN_PER_M


def in_to_m(inches: float) -> float:
    """Convert inches to meters."""
    return inches * _M_PER_IN


def mm_to_in(mm: float) -> float:
    """Convert millimeters to inches."""
    return mm * _IN_PER_MM


def in_to_mm(inches: float) -> float:
    """Convert inches to millimeters."""
    return inches * _MM_PER_IN


# ── Force ──────────────────────────────────────────────────────────────

_KN_PER_KIP = 4.44822
_KIP_PER_KN = 1.0 / _KN_PER_KIP
_KN_PER_LB = 0.00444822
_LB_PER_KN = 1.0 / _KN_PER_LB


def kN_to_kips(kN: float) -> float:
    """Convert kilonewtons to kips."""
    return kN * _KIP_PER_KN


def kips_to_kN(kips: float) -> float:
    """Convert kips to kilonewtons."""
    return kips * _KN_PER_KIP


def kN_to_lbs(kN: float) -> float:
    """Convert kilonewtons to pounds-force."""
    return kN * _LB_PER_KN


def lbs_to_kN(lbs: float) -> float:
    """Convert pounds-force to kilonewtons."""
    return lbs * _KN_PER_LB


# ── Stress / Pressure ──────────────────────────────────────────────────

_KPA_PER_KSF = 47.8803
_KSF_PER_KPA = 1.0 / _KPA_PER_KSF
_KPA_PER_PSI = 6.89476
_PSI_PER_KPA = 1.0 / _KPA_PER_PSI
_KPA_PER_TSF = 95.7605  # 1 US ton/ft² = 2000 psf = 95.76 kPa
_TSF_PER_KPA = 1.0 / _KPA_PER_TSF


def kPa_to_ksf(kPa: float) -> float:
    """Convert kilopascals to kips per square foot."""
    return kPa * _KSF_PER_KPA


def ksf_to_kPa(ksf: float) -> float:
    """Convert kips per square foot to kilopascals."""
    return ksf * _KPA_PER_KSF


def kPa_to_psi(kPa: float) -> float:
    """Convert kilopascals to pounds per square inch."""
    return kPa * _PSI_PER_KPA


def psi_to_kPa(psi: float) -> float:
    """Convert pounds per square inch to kilopascals."""
    return psi * _KPA_PER_PSI


def kPa_to_tsf(kPa: float) -> float:
    """Convert kilopascals to US tons per square foot."""
    return kPa * _TSF_PER_KPA


def tsf_to_kPa(tsf: float) -> float:
    """Convert US tons per square foot to kilopascals."""
    return tsf * _KPA_PER_TSF


# ── Unit Weight ────────────────────────────────────────────────────────

_KNM3_PER_PCF = 0.157087
_PCF_PER_KNM3 = 1.0 / _KNM3_PER_PCF


def kNm3_to_pcf(kNm3: float) -> float:
    """Convert kN/m³ to pounds per cubic foot (pcf)."""
    return kNm3 * _PCF_PER_KNM3


def pcf_to_kNm3(pcf: float) -> float:
    """Convert pounds per cubic foot (pcf) to kN/m³."""
    return pcf * _KNM3_PER_PCF


# ── Conversion Decorator ──────────────────────────────────────────────

# Mapping of unit system names to SI conversion functions for common quantities
_TO_SI = {
    'length': ft_to_m,
    'force': kips_to_kN,
    'stress': ksf_to_kPa,
    'unit_weight': pcf_to_kNm3,
}

_FROM_SI = {
    'length': m_to_ft,
    'force': kN_to_kips,
    'stress': kPa_to_ksf,
    'unit_weight': kNm3_to_pcf,
}
