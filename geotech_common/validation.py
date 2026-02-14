"""
Common parameter validation utilities for geotechnical modules.

Provides range checking with engineering-context error messages and warnings.
"""

import warnings
from typing import Optional


def check_positive(value: float, name: str) -> None:
    """Raise ValueError if value is not positive.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Parameter name for the error message.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def check_non_negative(value: float, name: str) -> None:
    """Raise ValueError if value is negative.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Parameter name for the error message.
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def check_range(value: float, name: str, low: float, high: float,
                strict: bool = False) -> None:
    """Warn if value is outside the typical engineering range.

    Parameters
    ----------
    value : float
        Value to check.
    name : str
        Parameter name for the warning.
    low : float
        Lower bound of typical range.
    high : float
        Upper bound of typical range.
    strict : bool, optional
        If True, raise ValueError instead of warning. Default False.
    """
    if value < low or value > high:
        msg = f"{name} = {value} is outside typical range [{low}, {high}]"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)


def check_friction_angle(phi_deg: float, name: str = "Friction angle") -> None:
    """Validate friction angle is in a reasonable range.

    Parameters
    ----------
    phi_deg : float
        Friction angle in degrees.
    name : str, optional
        Parameter name for messages.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"{name} must be 0-50 degrees, got {phi_deg}")
    if phi_deg > 45:
        warnings.warn(f"{name} = {phi_deg}° is unusually high; verify with lab testing")


def check_cohesion(cu: float, name: str = "Cohesion") -> None:
    """Validate undrained shear strength / cohesion.

    Parameters
    ----------
    cu : float
        Cohesion or undrained shear strength (kPa).
    name : str, optional
        Parameter name for messages.
    """
    if cu < 0:
        raise ValueError(f"{name} must be non-negative, got {cu}")
    if cu > 500:
        warnings.warn(f"{name} = {cu} kPa is unusually high; typical soft clay < 50 kPa, "
                       "stiff clay < 200 kPa")


def check_unit_weight(gamma: float, name: str = "Unit weight") -> None:
    """Validate soil unit weight.

    Parameters
    ----------
    gamma : float
        Unit weight (kN/m³).
    name : str, optional
        Parameter name for messages.
    """
    if gamma <= 0:
        raise ValueError(f"{name} must be positive, got {gamma}")
    if gamma < 10 or gamma > 25:
        warnings.warn(f"{name} = {gamma} kN/m³ is outside typical range [10, 25]")
