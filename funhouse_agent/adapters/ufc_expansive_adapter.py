"""UFC 3-220-07 reference adapter — Foundations in Expansive Soils."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.ufc_expansive import equations, tables
    return build_lookup_registry([
        (equations, "UFC Expansive Equations", "UFC 3-220-07"),
        (tables, "UFC Expansive Tables", "UFC 3-220-07"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
