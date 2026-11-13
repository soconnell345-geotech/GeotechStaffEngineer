"""GeotechStaffEngineer's part of the shared core, handed in through hooks
(BACKLOG INF-32, the app split's step 11 (c); ``aec_agent_core.hooks``).

The core imports nothing of GSE (plan ``module_work/APP_SPLIT_PLAN.md``
section 2.7 B). Where a core module needs to know something only GSE knows,
it asks for a hook, and ``funhouse_agent/__init__.py`` fills it with a
function here (as ``"module:attribute"`` strings, so nothing loads until a
hook is first asked for):

- ``held_records.record_fields`` -> :func:`record_fields`: the shapes of an
  exploration and a laboratory-test record (``report_ingest.model``), which
  ``aec_agent_core.deep.held_records`` finds in a stopped helper's words;
- ``cited_documents.reference_modules`` -> :func:`reference_modules`: the
  dispatch's reference modules, whose ``call_agent`` answers are reference
  sources for ``aec_agent_core.cited_documents``;
- ``method_call.warning_packages`` -> :func:`warning_packages`: GSE's
  calculation packages, whose Python warnings raised during a method call
  reach the agent (``aec_agent_core.call_warnings``, BACKLOG GM-316).

The review suite's hooks (its tasks, documents and the page's turn) are
filled by ``funhouse_agent/review_eval/__init__.py``.
"""

from __future__ import annotations

from typing import Tuple


def record_fields() -> Tuple[frozenset, frozenset, frozenset, frozenset]:
    """(exploration fields, lab-test fields, exploration kinds, lab kinds),
    read off ``report_ingest.model`` (what ``held_records._fields`` read
    before step 11 (c)). Raises ImportError where report_ingest is missing;
    the core then holds no records, as before."""
    from typing import get_args

    from report_ingest.model import (Investigation, InvestigationKind,
                                     LabKind, LabTest)
    return (frozenset(Investigation.model_fields),
            frozenset(LabTest.model_fields),
            frozenset(get_args(InvestigationKind)),
            frozenset(get_args(LabKind)))


def reference_modules() -> frozenset:
    """The dispatch's reference modules (``funhouse_agent.dispatch.
    REFERENCE_MODULES``)."""
    from funhouse_agent.dispatch import REFERENCE_MODULES
    return frozenset(REFERENCE_MODULES)


#: GSE's calculation packages: the top-level packages of the distribution
#: (pyproject's ``[tool.setuptools.packages.find]``) that a method handler
#: runs, plus the dispatch's adapters and the reference library. Left out:
#: the app (``funhouse_agent`` apart from its adapters, ``webapp``) and
#: ``report_ingest``, which no ``call_agent`` method runs.
WARNING_PACKAGES = (
    "bearing_capacity", "settlement", "axial_pile", "sheet_pile", "soe",
    "lateral_pile", "pile_group", "wave_equation", "drilled_shaft",
    "seismic_geotech", "retaining_walls", "pavement_design",
    "ground_improvement", "slope_stability", "reliability", "downdrag",
    "geotech_common", "calc_package", "profile_figure", "opensees_agent",
    "pystrata_agent", "section_props_agent", "concrete_props_agent",
    "pynite_agent", "seismic_signals_agent", "liquepy_agent",
    "gstools_agent", "salib_agent", "pystra_agent",
    "subsurface_characterization", "dxf_import", "dxf_export", "fem2d",
    "geo_project", "funhouse_agent.adapters", "geotech_references")


def warning_packages() -> tuple:
    """The module-name prefixes whose Python warnings, raised while a method
    runs, reach the agent with its result (BACKLOG GM-316;
    ``aec_agent_core.call_warnings``)."""
    return WARNING_PACKAGES
