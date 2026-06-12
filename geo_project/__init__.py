"""geo_project — canonical Project document for staged LE/FEM model setup.

The package the MODEL-SETUP agent edits and the human confirms:

* :mod:`geo_project.schema`    — versioned Project dataclasses + JSON round-trip
* :mod:`geo_project.validate`  — deterministic checks → Issue list
* :mod:`geo_project.builders`  — Project → SlopeGeometry / fem2d kwargs / runs
* :mod:`geo_project.templates` — parametric geometry generators
* :mod:`geo_project.ingest`    — DXF / PDF-vector / points / vision-draft → Project
* :mod:`geo_project.render`    — echo-back cross-section PNG + vertex table

Hard dependency: numpy only (matplotlib/ezdxf/PyMuPDF are optional and
lazy-imported where used).
"""

from geo_project.schema import (
    SCHEMA_VERSION,
    Anchor,
    Assumption,
    Confirmations,
    FEMAnalysis,
    GeosyntheticLayer,
    Geometry,
    Layer,
    LEAnalysis,
    LEProbabilistic,
    LESearch,
    Loads,
    Material,
    Nail,
    Project,
    ProjectMeta,
    Reinforcement,
    Surcharge,
    Water,
)
from geo_project.validate import Issue, has_errors, summarize, validate

__all__ = [
    "SCHEMA_VERSION",
    "Project", "ProjectMeta", "Geometry", "Material", "Layer", "Water",
    "Surcharge", "Loads", "Nail", "Anchor", "GeosyntheticLayer",
    "Reinforcement", "LESearch", "LEProbabilistic", "LEAnalysis",
    "FEMAnalysis", "Confirmations", "Assumption",
    "Issue", "validate", "has_errors", "summarize",
]
