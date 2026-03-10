"""GEC-11 reference adapter — FHWA-NHI-10-024 MSE Walls & Reinforced Soil Slopes."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_11 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-11 Figures", "FHWA-NHI-10-024"),
        (tables, "GEC-11 Tables", "FHWA-NHI-10-024"),
    ])
    add_text_retrieval(registry, info, "gec_11", "FHWA-NHI-10-024")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
