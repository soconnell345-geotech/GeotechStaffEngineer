"""UFC 3-250-04 Standard Practice for Concrete Pavements adapter.

Concrete pavement materials/construction practice: cement/aggregate/
admixture selection tables, dowel alignment tolerances and misalignment
impacts, joint spacing, edge slump, RCC gradation, cracking causes, and
the coarseness/workability-factor equation. Practice companion to the
design UFCs.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.ufc_concrete_practice import equations, tables
    registry, info = build_lookup_registry([
        (tables, "UFC 3-250-04 Tables", "UFC 3-250-04 Concrete Pavement Practice"),
        (equations, "UFC 3-250-04 Equations", "UFC 3-250-04 Concrete Pavement Practice"),
    ])
    add_text_retrieval(registry, info, "ufc_concrete_practice",
                       "UFC 3-250-04 Standard Practice for Concrete Pavements")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
