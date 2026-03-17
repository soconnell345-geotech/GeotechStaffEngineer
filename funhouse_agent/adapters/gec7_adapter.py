"""GEC-7 reference adapter — FHWA-NHI-14-007 Soil Nail Walls."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_7 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-7 Figures", "FHWA-NHI-14-007"),
        (tables, "GEC-7 Tables", "FHWA-NHI-14-007"),
    ])
    add_text_retrieval(registry, info, "gec_7", "FHWA-NHI-14-007")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
