"""Importing a library package loads nothing of ``funhouse_agent``.

BACKLOG INF-30 and INF-31 (``module_work/APP_SPLIT_PLAN.md``). The app's
package door shows a one-time professional-use notice on stderr and writes
a marker in the user's home folder, so a library-only user who imports a
calculation package, or draws a module figure, must not reach it. Code that
moved into ``funhouse_agent`` is loaded on first use instead (INF-30's
review, commit 4dad2ee; INF-31's review). Each import runs in a fresh
interpreter, because this suite has already imported ``funhouse_agent``.

Beside ``test_review_boundary.py``, not in it: that file skips when the
agent stack is not installed, and this check needs none of it.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

CODE = """
import sys
import {name}
assert "funhouse_agent" not in sys.modules, (
    "importing {name} loaded funhouse_agent: "
    + ", ".join(sorted(m for m in sys.modules
                       if m.split(".")[0] == "funhouse_agent")))
print("ok")
"""


@pytest.mark.parametrize("name", [
    "profile_figure", "geotech_common", "geotech_common.plotting",
    "calc_package", "report_ingest"])
def test_importing_a_library_package_loads_nothing_of_funhouse_agent(name):
    # GEOTECH_NO_DISCLAIMER: should the check fail, the notice and its
    # marker file must not reach the home folder of whoever runs it.
    env = dict(os.environ, GEOTECH_NO_DISCLAIMER="1")
    done = subprocess.run([sys.executable, "-c", CODE.format(name=name)],
                          capture_output=True, text=True, cwd=REPO, env=env)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
