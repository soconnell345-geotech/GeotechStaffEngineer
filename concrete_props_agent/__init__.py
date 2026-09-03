"""
Concrete properties agent — reinforced-concrete section analysis wrapper.

Wraps the `concreteproperties` library (MIT; Robbie van Leeuwen) for
rectangular RC section analysis: gross/cracked properties, cracking and
ultimate moment capacities, and N-M interaction diagrams.

Requires Python >= 3.12 (the library's floor); ``has_concreteproperties``
reports False below that and the adapter degrades gracefully.

Units (documented structural exception, see DESIGN.md): section dimensions
mm, material strengths MPa; moments returned in kN*m, forces in kN.

Public API
----------
analyze_rc_rectangle : Rectangular RC beam/column section analysis.
RCSectionResult : Result dataclass.
has_concreteproperties : Check if concreteproperties is installed.
"""

from concrete_props_agent.rc_section import analyze_rc_rectangle
from concrete_props_agent.results import RCSectionResult
from concrete_props_agent.concrete_utils import has_concreteproperties

__all__ = [
    "analyze_rc_rectangle",
    "RCSectionResult",
    "has_concreteproperties",
]
