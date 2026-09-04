"""EM 1110-2-2107 reference adapter (USACE, hydraulic steel structures).

Design of Hydraulic Steel Structures (1 Aug 2022 ed.): Chapter 3 target
reliability + usual/unusual/extreme load categorization (design_basis),
Chapter 4 LRFD loads (full load-factor table, load combinations,
earthquake combinations), the Chapter 4.4 + Appendix D pseudo-dynamic
(Chopra & Tan 1989) HSS-support seismic-acceleration amplification method,
Chapter 5 fatigue screening + fracture-critical redundancy checks, Chapter 6
bolt/weld/faying-surface selection rules, and Chapter 10 + Appendix F
Tainter-gate load equations (side-seal friction, wire-rope loads,
hydrostatic load by integration/projection, trunnion friction, the full
load-combination table, anchorage shear-friction). Does NOT reprint AISC
360 member-capacity equations (members are designed to AISC 360 directly).
US customary units (kips, feet, inches, psi/ksi, pcf/kcf).
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.em_2107 import (
        connections, design_basis, fatigue_fracture, loads,
        seismic_amplification, tainter_gate_loads,
    )
    registry, info = build_lookup_registry([
        (design_basis, "EM 2107 Design Basis (Ch 3)", "EM 1110-2-2107"),
        (loads, "EM 2107 Loads (Ch 4)", "EM 1110-2-2107"),
        (seismic_amplification, "EM 2107 Seismic Amplification (Ch 4.4, App D)",
         "EM 1110-2-2107"),
        (fatigue_fracture, "EM 2107 Fatigue/Fracture (Ch 5)", "EM 1110-2-2107"),
        (connections, "EM 2107 Connections (Ch 6)", "EM 1110-2-2107"),
        (tainter_gate_loads, "EM 2107 Tainter Gate Loads (Ch 10, App F)",
         "EM 1110-2-2107"),
    ])
    add_text_retrieval(registry, info, "em_2107", "EM 1110-2-2107")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
