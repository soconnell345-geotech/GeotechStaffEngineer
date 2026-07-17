"""UFC 3-250-03 Standard Practice Manual for Flexible Pavements adapter.

Asphalt materials/construction practice: HMA gradation bands and Marshall/
Superpave criteria, PG-grade selection by climate, spray application rates
and temperatures, seal coats, RMP grout, plus Marshall volumetrics and the
slurry-seal design equations. Practice companion to the design UFCs.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.ufc_flexible_practice import equations, tables
    registry, info = build_lookup_registry([
        (tables, "UFC 3-250-03 Tables", "UFC 3-250-03 Flexible Pavement Practice"),
        (equations, "UFC 3-250-03 Equations", "UFC 3-250-03 Flexible Pavement Practice"),
    ])
    add_text_retrieval(registry, info, "ufc_flexible_practice",
                       "UFC 3-250-03 Standard Practice Manual for Flexible Pavements")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
