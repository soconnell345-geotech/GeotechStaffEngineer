"""GEC-4 reference adapter — FHWA-IF-99-015 Ground Anchors and Anchored Systems (1999)."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_4 import tables
    registry, info = build_lookup_registry([
        (tables, "GEC-4 Tables", "FHWA-IF-99-015"),
    ])
    add_text_retrieval(registry, info, "gec_4", "FHWA-IF-99-015")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
