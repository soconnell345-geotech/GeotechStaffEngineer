"""GEC-13 reference adapter — FHWA-NHI-16-027 Ground Modification Methods."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_13 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-13 Figures", "FHWA-NHI-16-027"),
        (tables, "GEC-13 Tables", "FHWA-NHI-16-027"),
    ])
    add_text_retrieval(registry, info, "gec_13", "FHWA-NHI-16-027")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
