"""
PyNite agent — elastic frame analysis wrapper.

Wraps PyNiteFEA (MIT; JWock82) for linear-elastic 2D/3D frame analysis:
general node/member models and a continuous-beam convenience method.

Units: toolkit SI — lengths m, forces kN, distributed loads kN/m,
E and G in kPa, section properties in m-based units (A m^2, I m^4).
Moments return in kN*m, deflections in m (also echoed in mm).

Public API
----------
analyze_frame : General frame analysis from nodes/members/supports/loads.
analyze_continuous_beam : Multi-span beam convenience wrapper.
FrameResult, ContinuousBeamResult : Result dataclasses.
has_pynite : Check if PyNiteFEA is installed.
"""

from pynite_agent.frame import analyze_frame
from pynite_agent.continuous_beam import analyze_continuous_beam
from pynite_agent.results import FrameResult, ContinuousBeamResult, MemberResult
from pynite_agent.pynite_utils import has_pynite

__all__ = [
    "analyze_frame",
    "analyze_continuous_beam",
    "FrameResult",
    "ContinuousBeamResult",
    "MemberResult",
    "has_pynite",
]
