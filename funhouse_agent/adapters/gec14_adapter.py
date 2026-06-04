"""GEC-14 reference adapter — FHWA-HIF-17-016 Assuring Quality in Geotechnical Reporting Documents (2016)."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    # GEC-14 is text-only — no Python lookup functions
    registry, info = build_lookup_registry([])
    add_text_retrieval(registry, info, "gec_14", "FHWA-HIF-17-016")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
