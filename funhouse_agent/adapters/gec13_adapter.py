"""GEC-13 reference adapter — Ground Modification Methods, FHWA-NHI-16-027
(Volume I, chapters 1-5) and FHWA-NHI-16-028 (Volume II, chapters 6-10)."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_13 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-13 Figures", "FHWA-NHI-16-027 / -028 (GEC-13 Vols I / II)"),
        (tables, "GEC-13 Tables", "FHWA-NHI-16-027 / -028 (GEC-13 Vols I / II)"),
    ])
    add_text_retrieval(registry, info, "gec_13",
                       "FHWA-NHI-16-027 / -028 (GEC-13 Vols I / II)")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
