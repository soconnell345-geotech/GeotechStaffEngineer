"""UFC 3-250-11 Soil Stabilization and Modification reference adapter.

Stabilizer selection (gradation-triangle area + Table 2-3 additive guide),
strength/durability requirements, cement/lime/bituminous mix criteria,
swell potential, thickness equivalency factors, and the printed content
equations (A=100BC etc.). Feeds the pavement stabilized-layer workflow.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.ufc_stabilization import equations, tables
    registry, info = build_lookup_registry([
        (tables, "UFC 3-250-11 Tables", "UFC 3-250-11 Soil Stabilization"),
        (equations, "UFC 3-250-11 Equations", "UFC 3-250-11 Soil Stabilization"),
    ])
    add_text_retrieval(registry, info, "ufc_stabilization",
                       "UFC 3-250-11 Soil Stabilization and Modification")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
