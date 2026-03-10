"""UFC 3-220-05 reference adapter — Dewatering and Groundwater Control."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.ufc_dewatering import equations, tables
    return build_lookup_registry([
        (equations, "UFC Dewatering Equations", "UFC 3-220-05"),
        (tables, "UFC Dewatering Tables", "UFC 3-220-05"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
