"""
Section properties agent — cross-section analysis wrapper.

Wraps the `sectionproperties` library (MIT; Robbie van Leeuwen) for
geometric, warping, and plastic analysis of structural cross-sections:
parametric steel/geometric shapes and arbitrary polygons.

Units: structural-section convention — dimensions in **mm**, results in
mm-based units (mm^2, mm^4, mm^3). This is a documented exception to the
toolkit's metre-based SI (like pavement_design's US-customary exception);
see DESIGN.md.

Public API
----------
analyze_section : Compute section properties for a parametric shape.
analyze_polygon_section : Compute section properties for an arbitrary polygon.
SectionPropertiesResult : Result dataclass.
has_sectionproperties : Check if sectionproperties is installed.
"""

from section_props_agent.sections import (
    analyze_section,
    analyze_polygon_section,
    SECTION_SHAPES,
)
from section_props_agent.results import SectionPropertiesResult
from section_props_agent.section_utils import has_sectionproperties

__all__ = [
    "analyze_section",
    "analyze_polygon_section",
    "SectionPropertiesResult",
    "SECTION_SHAPES",
    "has_sectionproperties",
]
