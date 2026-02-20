"""
PySeismoSoil agent — nonlinear soil curve generation and Vs profile analysis.

Wraps the PySeismoSoil library for generating modulus reduction (G/Gmax)
and damping curves from MKZ and HH constitutive models, and computing
site characterization parameters (Vs30, f0, z1) from Vs profiles.

Public API
----------
generate_curves : Generate G/Gmax and damping curves.
analyze_vs_profile : Compute Vs30, f0, z1 from Vs profile.
CurveResult, VsProfileResult : Result dataclasses.
has_pyseismosoil : Check if PySeismoSoil is installed.
"""

from pyseismosoil_agent.soil_curves import generate_curves, analyze_vs_profile
from pyseismosoil_agent.results import CurveResult, VsProfileResult
from pyseismosoil_agent.pyseismosoil_utils import has_pyseismosoil

__all__ = [
    "generate_curves",
    "analyze_vs_profile",
    "CurveResult",
    "VsProfileResult",
    "has_pyseismosoil",
]
