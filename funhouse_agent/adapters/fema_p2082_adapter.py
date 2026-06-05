"""FEMA P-2082 reference adapter — 2020 NEHRP Recommended Seismic Provisions.

Geotech-relevant seismic-site content: Chapter 20 site classification (the
REVISED 2020 scheme with intermediate classes BC, CD, DE) and Chapter 11
seismic design criteria (SDS/SD1, two-period design spectrum, Seismic Design
Category, Risk Category). P-2082 deleted the ASCE 7-16 Fa/Fv site coefficients.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.fema_p2082 import equations, tables
    registry, info = build_lookup_registry([
        (tables, "FEMA P-2082 Tables", "FEMA P-2082"),
        (equations, "FEMA P-2082 Equations", "FEMA P-2082"),
    ])
    add_text_retrieval(registry, info, "fema_p2082", "FEMA P-2082")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
