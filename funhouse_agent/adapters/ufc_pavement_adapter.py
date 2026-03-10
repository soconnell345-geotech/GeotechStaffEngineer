"""UFC 3-260-02 reference adapter — Pavement Design for Airfields."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.ufc_pavement import equations, tables
    return build_lookup_registry([
        (equations, "UFC Pavement Equations", "UFC 3-260-02"),
        (tables, "UFC Pavement Tables", "UFC 3-260-02"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
