"""GEC-5 reference adapter — FHWA NHI-16-072 Geotechnical Site Characterization (2017)."""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    # GEC-5 is text-only — no Python lookup functions
    registry, info = build_lookup_registry([])
    add_text_retrieval(registry, info, "gec_5", "FHWA NHI-16-072")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
