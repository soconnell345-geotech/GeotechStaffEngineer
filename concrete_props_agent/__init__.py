"""
Concrete properties agent — reinforced-concrete section analysis.

Rectangular RC section analysis: gross/cracked transformed properties,
cracking and ultimate moment capacities, and N-M interaction diagrams.

The mechanics are computed natively (``rc_native``) from the ACI 318-19
material model — transformed sections, cracked-section neutral axis, and
strain compatibility with the equivalent rectangular stress block. Before
5.13 this module wrapped the `concreteproperties` library; that dependency
was removed because it (and the `sectionproperties` package it builds on)
requires `cytriangle`, a compiled wheel quarantined by the corporate package
proxy the app deploys behind — the same route taken for the groundhog removal
in 5.11.2. The library's outputs were pinned as numerical test oracles before
deletion (no code was copied); see ``module_work/structural_native/``.

Units (documented structural exception, see DESIGN.md): section dimensions
mm, material strengths MPa; moments returned in kN*m, forces in kN.

Public API
----------
analyze_rc_rectangle : Rectangular RC beam/column section analysis.
RCSectionResult : Result dataclass.
aci_beta1 : ACI 318-19 Table 22.2.2.4.3 stress-block factor.
"""

from concrete_props_agent.rc_section import aci_beta1, analyze_rc_rectangle
from concrete_props_agent.results import RCSectionResult

__all__ = [
    "analyze_rc_rectangle",
    "RCSectionResult",
    "aci_beta1",
]
