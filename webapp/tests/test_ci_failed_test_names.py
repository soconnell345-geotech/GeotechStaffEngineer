"""The CI step that names a failed run's tests where no sign-in is needed.

`.github/scripts/name_failed_tests.py` turns the FAILED and ERROR lines of a
pytest log into GitHub error annotations; `tests.yml` calls it after a failed
step in the two jobs that run the suite on four workers. All values below are
made up.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "name_failed_tests.py"
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"

LOG = """\
........F...E
=========================== short test summary info ============================
FAILED pkg/tests/test_a.py::test_one - assert 1.25 == 1.5
FAILED pkg/tests/test_a.py::test_two[case 50%] - ValueError: no root
ERROR pkg/tests/test_b.py - ImportError: no module named made_up
FAILED pkg/tests/test_a.py::test_one - assert 1.25 == 1.5
================== 2 failed, 1 error, 40 passed in 12.34s ===================
"""


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("name_failed_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_each_failed_test_is_named_once_after_the_totals(script):
    commands = script.annotations(LOG)
    assert len(commands) == 1
    command = commands[0]
    assert command.startswith("::error title=Failed tests 1 to 3 of 3::")
    assert "\n" not in command
    assert command.split("::", 2)[2].split("%0A") == [
        "2 failed, 1 error, 40 passed in 12.34s",
        "FAILED pkg/tests/test_a.py::test_one - assert 1.25 == 1.5",
        "FAILED pkg/tests/test_a.py::test_two[case 50%25] - ValueError: no root",
        "ERROR pkg/tests/test_b.py - ImportError: no module named made_up",
    ]


def test_many_failures_are_split_and_capped(script):
    log = "\n".join("FAILED pkg/tests/test_c.py::test_%d - assert 0" % i for i in range(100))
    commands = script.annotations(log)
    assert len(commands) == script.MOST_ANNOTATIONS == 9
    assert commands[0].startswith("::error title=Failed tests 1 to 8 of 100::")
    assert commands[-1].startswith("::error title=Failed tests 65 to 72 of 100::")
    assert commands[-1].endswith("test_71 - assert 0")


def test_a_long_reason_is_cut_and_colour_codes_are_dropped(script):
    esc = chr(27)
    log = esc + "[31mFAILED pkg/tests/test_d.py::test_long - " + "x" * 500 + esc + "[0m"
    (command,) = script.annotations(log)
    body = command.split("::", 2)[2]
    assert body.startswith("FAILED pkg/tests/test_d.py::test_long - xxx")
    assert len(body) == script.LINE_LIMIT and esc not in command


def test_a_log_with_no_summary_gives_its_last_lines(script):
    (command,) = script.annotations("collected 0 items\n\nINTERNALERROR> made-up crash\n")
    assert command.startswith("::error title=No FAILED or ERROR line")
    assert command.endswith("collected 0 items%0AINTERNALERROR> made-up crash")


def test_a_missing_log_is_said_not_raised(script, tmp_path, capsys):
    assert script.main(["name_failed_tests.py", str(tmp_path / "absent.log")]) == 0
    assert capsys.readouterr().out.startswith("::error title=No pytest log")


def test_the_workflow_keeps_the_log_and_calls_the_script():
    text = WORKFLOW.read_text(encoding="utf-8")
    # The two jobs that run pytest on four workers keep its output and name
    # the failed tests after a failure.
    assert text.count("pytest -n auto --dist worksteal") == 2
    assert text.count('| tee "$RUNNER_TEMP/pytest.log"') == 2
    assert text.count('python .github/scripts/name_failed_tests.py "$RUNNER_TEMP/pytest.log"') == 2
    assert text.count("if: failure()") >= 2
