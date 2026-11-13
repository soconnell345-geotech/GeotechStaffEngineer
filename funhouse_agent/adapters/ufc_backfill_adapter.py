"""UFC 3-220-04FA reference adapter — Backfill for Subsurface Structures."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.ufc_backfill import equations, tables
    return build_lookup_registry([
        (equations, "UFC Backfill Equations", "UFC 3-220-04FA"),
        (tables, "UFC Backfill Tables", "UFC 3-220-04FA"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
