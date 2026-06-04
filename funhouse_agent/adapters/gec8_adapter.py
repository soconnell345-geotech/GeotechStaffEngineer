"""GEC-8 reference adapter — FHWA-HIF-07-03 CFA Piles (2007)."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_8 import equations, tables
    registry, info = build_lookup_registry([
        (equations, "GEC-8 Equations", "FHWA-HIF-07-03"),
        (tables, "GEC-8 Tables", "FHWA-HIF-07-03"),
    ])
    add_text_retrieval(registry, info, "gec_8", "FHWA-HIF-07-03")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
