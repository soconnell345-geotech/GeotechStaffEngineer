"""
Section properties agent — cross-section analysis.

Geometric, torsional and plastic properties of structural cross-sections:
parametric steel/geometric shapes and arbitrary polygons. Everything is
computed natively on numpy/scipy — the area, centroid, second moments and
plastic moduli are exact closed-form integrals over the outline
(``polygon_props``), and the torsion/warping constants come from the
published closed-form solutions (``torsion``).

Before 5.13 this module wrapped the `sectionproperties` library. That
dependency was removed because it requires `cytriangle`, a compiled wheel
quarantined by the corporate package proxy the app deploys behind — the same
route taken for the groundhog removal in 5.11.2. The library's outputs were
pinned as numerical test oracles before deletion (no code was copied); see
``module_work/structural_native/``.

Units: structural-section convention — dimensions in **mm**, results in
mm-based units (mm^2, mm^4, mm^3). This is a documented exception to the
toolkit's metre-based SI (like pavement_design's US-customary exception);
see DESIGN.md.

Public API
----------
analyze_section : Compute section properties for a parametric shape.
analyze_polygon_section : Compute section properties for an arbitrary polygon.
SectionPropertiesResult : Result dataclass.
"""

from section_props_agent.sections import (
    analyze_section,
    analyze_polygon_section,
    SECTION_SHAPES,
)
from section_props_agent.results import SectionPropertiesResult

__all__ = [
    "analyze_section",
    "analyze_polygon_section",
    "SectionPropertiesResult",
    "SECTION_SHAPES",
]
