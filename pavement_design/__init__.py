"""pavement_design -- AASHTO 1993 pavement structural design.

Complete flexible (asphalt) and rigid (PCC) pavement design per the AASHTO
Guide for Design of Pavement Structures (1993), orchestrating the digitized
guide equations, tables, and charts in ``geotech_references.aashto_1993``
(every coefficient and solve carries its printed-page provenance through to
the result).

Public API
----------
design_flexible_pavement : required SN + layered D1/D2/D3 split (Figure
    3.1/3.2), or adequacy check of a given section.
design_rigid_pavement : required slab thickness D (Figure 3.7) with direct,
    simplified (MR/19.4), or full Section 3.2 composite-k; or adequacy
    check.
compute_design_esals : design-lane W18 from an axle spectrum (Appendix D
    LEFs), truck factors, or a base-year total, with growth/DD/DL.
growth_factor : compound traffic growth factor.
PavementLayer : one course of a flexible section.

UNITS: US customary (psi, pci, inches, kips, 18-kip ESALs) -- a documented
exception to the repo SI convention, because the 1993 Guide is
US-customary native (same precedent as ``geotech_references.aashto_1993``).

Scope limits (see DESIGN.md): new construction only -- overlays/
rehabilitation (Part III), swelling/frost-heave serviceability loss,
rigid joint/reinforcement design, and low-volume catalog designs are not
computed here (the reference layer carries the guide text for those).
"""

from .flexible import PavementLayer, design_flexible_pavement
from .results import (DesignTrafficResult, FlexiblePavementResult,
                      RigidPavementResult)
from .rigid import design_rigid_pavement
from .traffic import compute_design_esals, growth_factor

__all__ = [
    "PavementLayer",
    "design_flexible_pavement",
    "design_rigid_pavement",
    "compute_design_esals",
    "growth_factor",
    "DesignTrafficResult",
    "FlexiblePavementResult",
    "RigidPavementResult",
]
