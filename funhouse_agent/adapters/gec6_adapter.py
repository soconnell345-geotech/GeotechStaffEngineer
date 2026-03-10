"""GEC-6 reference adapter — FHWA-SA-02-054 Shallow Foundations."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_6 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-6 Figures", "FHWA-SA-02-054"),
        (tables, "GEC-6 Tables", "FHWA-SA-02-054"),
    ])
    add_text_retrieval(registry, info, "gec_6", "FHWA-SA-02-054")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
