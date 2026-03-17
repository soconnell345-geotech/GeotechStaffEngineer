"""NOAA Frost reference adapter — frost depth equations and soil thermal properties."""

from funhouse_agent.adapters._reference_common import build_lookup_registry


def _build():
    from geotech_references.noaa_frost import equations, tables
    return build_lookup_registry([
        (equations, "NOAA Frost Equations", "Stefan/Berggren frost depth equations"),
        (tables, "NOAA Frost Tables", "Kersten/Farouki soil thermal properties"),
    ])


METHOD_REGISTRY, METHOD_INFO = _build()
