"""Micropile reference adapter — FHWA-NHI-05-039 Micropile Design & Construction."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.micropile import figures, tables
    registry, info = build_lookup_registry([
        (figures, "Micropile Figures", "FHWA-NHI-05-039"),
        (tables, "Micropile Tables", "FHWA-NHI-05-039"),
    ])
    add_text_retrieval(registry, info, "micropile", "FHWA-NHI-05-039")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
