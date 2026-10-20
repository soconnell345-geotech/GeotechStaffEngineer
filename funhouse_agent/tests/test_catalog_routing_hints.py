"""The module catalog keeps its routing hints (BACKLOG INF-22).

``list_agents`` and the prompt's module table show each module's name beside
its brief, so a brief need not repeat the name. What a brief must keep is
where a question goes when two modules could take it. Each phrase below was
added for a routing failure or a consolidation (the commit in the comment)
and stays through any rebalance of the catalog's 7,500-character budget.
"""

import json

import pytest

from funhouse_agent.adapters import MODULE_REGISTRY
from funhouse_agent.dispatch import list_agents

HINTS = [
    # 311d2d8: one liquefaction tool; the others point at it
    ("liquefaction", "NCEER/Youd 2001"),
    ("seismic_geotech", "liquefaction triggering: 'liquefaction'"),
    ("liquepy", "routine triggering: 'liquefaction'"),
    # d4de8e4: native reliability; SORM stays in pystra
    ("reliability", "SORM: 'pystra'"),
    # limit equilibrium in one module, strength reduction in the other
    ("slope_stability", "LE methods"),
    ("fem2d", "slope SRM"),
    # downdrag apart from axial capacity
    ("downdrag", "Pile downdrag"),
    ("axial_pile", "Driven pile capacity"),
    # 0b4a0b9: worked examples before a design
    ("worked_examples", "consult before nontrivial designs"),
    # e75b516, 7701932: search to scope, then fetch; a chart is read off
    ("reference_db", "use first to scope, then reference_get"),
    ("figure_db", "read values off it with read_reference_figure"),
    # 3d1cd44: DIGGS written from the logs the agent read
    ("subsurface", "write (from logs you read)"),
    # d86308b: probabilities of totals over a footprint
    ("gstools", "conditional simulation"),
]


@pytest.mark.parametrize("module, phrase", HINTS)
def test_the_brief_keeps_its_routing_hint(module, phrase):
    assert phrase in MODULE_REGISTRY[module]["brief"]


def test_every_module_has_a_brief_under_the_budget():
    catalog = list_agents()
    assert all(str(b).strip() for b in catalog.values())
    assert len(json.dumps(catalog)) < 7500
