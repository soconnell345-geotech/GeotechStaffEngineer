"""GEC-12 reference adapter — FHWA-NHI-16-009 (Vol. I) and FHWA-NHI-16-010
(Vol. II) Driven Piles.

The figure and table lookups are Vol. I's chapter 7; the chapter text holds
both volumes, each section cited by its own volume's report number.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gec_12 import figures, tables
    registry, info = build_lookup_registry([
        (figures, "GEC-12 Figures", "FHWA-NHI-16-009"),
        (tables, "GEC-12 Tables", "FHWA-NHI-16-009"),
    ])
    add_text_retrieval(registry, info, "gec_12",
                       "FHWA-NHI-16-009 (Vol. I) / FHWA-NHI-16-010 (Vol. II)")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
