"""UFC 3-250-01 reference adapter — Pavement Design for Roads and Parking Areas (2016)."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.ufc_pavement import equations, tables
    return build_lookup_registry([
        (equations, "UFC Pavement Equations", "UFC 3-250-01"),
        (tables, "UFC Pavement Tables", "UFC 3-250-01"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
