"""EM 1110-2-2104 reference adapter (USACE, RC hydraulic structures).

Strength Design for Reinforced Concrete Hydraulic Structures (1 Nov 2023 ed.,
published 8 Jan 2025): Chapter 2 detailing (min cover, splice stagger,
temperature/shrinkage steel), Chapter 3 loads (load inventory, full load-
factor table, LRFD + earthquake load combinations) and serviceability
(service stresses, single-load-factor method, reinforcement-ratio limits,
min wall thickness), Chapter 4 + Appendix B flexure/axial INVESTIGATION
equations (singly/doubly reinforced, tension+flexure, pure flexure, Bresler
biaxial check), Appendix D-2 DESIGN equations (solve for As/As' given
Mn/Pn), and Chapter 5 shear (one-way slab/wall, special straight members,
curved members). US customary units (psi/ksi, inches, kips, pcf).
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.em_2104 import (
        design, flexure_axial, loads, reinforcement, serviceability, shear,
    )
    registry, info = build_lookup_registry([
        (reinforcement, "EM 2104 Detailing (Ch 2)", "EM 1110-2-2104"),
        (loads, "EM 2104 Loads (Ch 3)", "EM 1110-2-2104"),
        (serviceability, "EM 2104 Serviceability (Ch 3)", "EM 1110-2-2104"),
        (flexure_axial, "EM 2104 Flexure/Axial (Ch 4, App B)", "EM 1110-2-2104"),
        (design, "EM 2104 Design (App D-2)", "EM 1110-2-2104"),
        (shear, "EM 2104 Shear (Ch 5)", "EM 1110-2-2104"),
    ])
    add_text_retrieval(registry, info, "em_2104", "EM 1110-2-2104")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
