"""FEMA P-2192 reference adapter — Seismic Design Category determination."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.fema_p2192 import tables
    return build_lookup_registry([
        (tables, "FEMA P-2192", "FEMA P-2192 (2024 Edition)"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
