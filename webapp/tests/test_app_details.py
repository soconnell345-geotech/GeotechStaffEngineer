"""GeotechStaffEngineer hands the core its own name (INF-33 step 13), so
the run record, the Word and markup authors and the runtime messages read
as before the core stopped naming it: from each of its three doors."""

import subprocess
import sys

import pytest

DOORS = ("funhouse_agent", "webapp", "calc_package")


@pytest.mark.parametrize("door", DOORS)
def test_each_door_hands_in_gse(door):
    code = (
        f"import {door}\n"
        "from aec_agent_core import app_details, docx_renderer, "
        "document_tools, runtime_check\n"
        "from aec_agent_core.review_eval import runner\n"
        "assert app_details.details() == {'name': 'GeotechStaffEngineer', "
        "'dist': 'geotech-staff-engineer', 'package': "
        "'funhouse_agent.deep'}, app_details.details()\n"
        "assert docx_renderer.DEFAULT_AUTHOR == 'GeotechStaffEngineer'\n"
        "assert document_tools.DEFAULT_MARKUP_AUTHOR == "
        "'GeotechStaffEngineer (AI draft)'\n"
        "assert document_tools.OWN_MARK_WORDS == "
        "('GeotechStaffEngineer', '(AI draft)')\n"
        "assert runner.SUITE_AUTHOR == "
        "'Review suite via GeotechStaffEngineer (AI draft)'\n"
        "assert runner._suite_folder() == 'GeotechStaffEngineer/review_eval'\n"
        "assert runtime_check._who() == 'funhouse_agent.deep'\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
