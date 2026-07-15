"""AASHTO 1993 Pavement Design Guide reference adapter.

AASHTO Guide for Design of Pavement Structures (1993): flexible design
equation (solve W18 or SN) and rigid design equation (solve W18 or slab D),
structural number composition SN = a1*D1 + a2*D2*m2 + a3*D3*m3 with the
layered minimum-thickness cascade, layer coefficients (a1 asphalt chart, a2/a3
printed regressions, cement/bituminous-treated base charts), effective roadbed
resilient modulus (seasonal relative-damage averaging), reliability (ZR table,
recommended levels by functional class, So guidance, stage compounding),
drainage coefficients mi/Cd, load transfer J, serviceability, representative
ESAL load-equivalency factors (SN=5 flexible / D=9 rigid, pt=2.5), and
aggregate-surfaced road models. NATIVE US CUSTOMARY UNITS (psi, pci, kips,
inches) per the source nomographs — units documented on every method.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.aashto_1993 import equations, tables
    registry, info = build_lookup_registry([
        (tables, "AASHTO 1993 Tables", "AASHTO 1993 Pavement Design Guide"),
        (equations, "AASHTO 1993 Equations", "AASHTO 1993 Pavement Design Guide"),
    ])
    add_text_retrieval(registry, info, "aashto_1993",
                       "AASHTO 1993 Pavement Design Guide")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
