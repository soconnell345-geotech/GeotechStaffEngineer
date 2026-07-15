"""Eurocode 7 Part 2 reference adapter (EN 1997-2:2007).

Ground investigation and testing: derived-value correlations from the
informative annexes — CPT (phi'/E from qc, oedometer alpha, Dutch pile
base/shaft factors), pressuremeter (Menard k, shape coefficients, rheological
alpha, pile compression factor), SPT (density index from N1(60), ageing,
phi' from density index), dynamic probing, weight sounding, field vane
correction mu, DMT -> Eoed, plate load test (cu, modulus, subgrade reaction),
lab-test minimum sample masses / test counts, and Eq 4.1-4.5 derived values
(cu from CPT/CPTU/FVT/DMT, Eoed from qc). SI units throughout.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.eurocode_7_2 import equations, tables
    registry, info = build_lookup_registry([
        (tables, "EC7-2 Tables", "EN 1997-2:2007"),
        (equations, "EC7-2 Equations", "EN 1997-2:2007"),
    ])
    add_text_retrieval(registry, info, "eurocode_7_2", "EN 1997-2:2007")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
