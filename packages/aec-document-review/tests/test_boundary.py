"""The boundary, as a plain test (BACKLOG INF-33; plan 2.7 C: neither app
imports the other). GeotechStaffEngineer's boundary test blocks its own
modules and builds the page; here there is nothing geotech to block, so the
check is direct:

1. no file of the package imports a geotechnical package or a module of
   GeotechStaffEngineer, at any depth (read from the source, so an import
   inside a function counts too);
2. building both page builds and running a turn, in a fresh interpreter,
   loads none of them.
"""

import ast
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(os.path.dirname(HERE), "aec_document_review")

#: GeotechStaffEngineer's own packages, its analysis modules and the
#: reference library (GSE's page_builds.GEOTECH_TOP_LEVEL, with its two app
#: packages): none may be imported by Document Review.
FORBIDDEN = frozenset({
    "funhouse_agent", "webapp",
    "bearing_capacity", "settlement", "axial_pile", "sheet_pile", "soe",
    "lateral_pile", "pile_group", "wave_equation", "drilled_shaft",
    "seismic_geotech", "retaining_walls", "pavement_design",
    "ground_improvement", "slope_stability", "reliability", "downdrag",
    "geotech_common", "calc_package", "profile_figure", "opensees_agent",
    "pystrata_agent", "seismic_signals_agent", "liquepy_agent",
    "gstools_agent", "salib_agent", "pystra_agent", "section_props_agent",
    "concrete_props_agent", "pynite_agent", "subsurface_characterization",
    "dxf_import", "dxf_export", "fem2d", "geo_project", "report_ingest",
    "geotech_references", "validation_examples", "foundry",
    "foundry_test_harness",
})


def _imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 \
                and node.module:
            yield node.lineno, node.module
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", "") \
                == "import_module" and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            yield node.lineno, node.args[0].value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and ":" in node.value and node.value.count(":") == 1:
            # a hook target, "module:attribute"
            mod, _, attr = node.value.partition(":")
            if mod and attr.isidentifier() and all(
                    p.isidentifier() for p in mod.split(".")):
                yield node.lineno, mod


def test_no_package_file_imports_geotech():
    found = []
    for root, _, files in os.walk(PACKAGE):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                for line, mod in _imports(path):
                    if mod.split(".")[0] in FORBIDDEN:
                        found.append(f"{os.path.relpath(path, PACKAGE)}:"
                                     f"{line}: {mod}")
    assert not found, ("Document Review imports GeotechStaffEngineer or a "
                       "geotech package:\n  " + "\n  ".join(found))


_CHILD = r"""
import json, sys
sys.path.insert(0, %(root)r)
from tests import page_builds
for name in page_builds.BUILDS:
    page_builds.run_page(name)
print(json.dumps(sorted({m.split(".")[0] for m in sys.modules})))
"""


def test_building_and_a_turn_load_nothing_geotech():
    env = dict(os.environ, MPLBACKEND="Agg", GEOTECH_NO_DISCLAIMER="1")
    out = subprocess.run(
        [sys.executable, "-c", _CHILD % {"root": os.path.dirname(HERE)}],
        capture_output=True, text=True, timeout=600, env=env,
        cwd=os.path.dirname(HERE))
    assert out.returncode == 0, out.stderr[-3000:]
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert not loaded & FORBIDDEN, sorted(loaded & FORBIDDEN)
    assert "aec_document_review" in loaded and "aec_agent_core" in loaded
