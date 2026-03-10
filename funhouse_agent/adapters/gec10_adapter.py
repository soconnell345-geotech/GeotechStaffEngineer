"""GEC-10 reference adapter — FHWA-NHI-10-016 Drilled Shafts."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_10 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-10 Figures", "FHWA-NHI-10-016"),
        (tables, "GEC-10 Tables", "FHWA-NHI-10-016"),
    ])
    add_text_retrieval(registry, info, "gec_10", "FHWA-NHI-10-016")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
