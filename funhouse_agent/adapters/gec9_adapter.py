"""GEC-9 reference adapter — FHWA-HIF-18-031 Laterally Loaded Deep Foundations (2018)."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_9 import tables
    registry, info = build_lookup_registry([
        (tables, "GEC-9 Tables", "FHWA-HIF-18-031"),
    ])
    add_text_retrieval(registry, info, "gec_9", "FHWA-HIF-18-031")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
