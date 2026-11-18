"""Release review of 2026-11-18 (whole change, B): ``measure`` and
``log_grid`` ran under the process's planlens host while every other
document tool of the same build ran under the build's own
(``document_tools.host_for``). Where the build's host carries an option the
process host does not (a DXF opened also as a plot, behind
``GEOTECH_REVIEW_DXF_PLOTS``), a page just looked at was then refused by
``measure`` as "damaged, incomplete or of another kind".

Made-up values only.
"""

from aec_agent_core import measure_tool
from aec_agent_core.deep.vision_kit import make_vision_tools
from planlens import review as planlens_review

SIGNER = "Made-up Reviewer via the test"


def _tools():
    return {t.name: t for t in make_vision_tools(
        engine=None, attachments={"plan.pdf": b"%PDF-1.4 made up"},
        markup_author=SIGNER)}


def test_measure_runs_under_the_builds_host(monkeypatch):
    seen = []

    def fake(args, attachments, engine, **kw):
        seen.append(planlens_review.current())
        return '{"ok": true}'

    monkeypatch.setattr(measure_tool, "run_measure", fake)
    out = _tools()["measure"].invoke(
        {"source": "plan.pdf", "page": 0, "bbox": [10, 10, 60, 60]})
    assert "ok" in out
    assert len(seen) == 1
    host = seen[0]
    signer = host.signer() if callable(host.signer) else host.signer
    assert signer == SIGNER, "measure ran under the process host"
    # and the process host is back afterwards
    assert planlens_review.current() is not host


def test_log_grid_runs_under_the_builds_host(monkeypatch):
    seen = []

    def fake(args, attachments, engine, **kw):
        seen.append(planlens_review.current())
        return '{"ok": true}'

    monkeypatch.setattr(measure_tool, "run_log_grid", fake)
    tools = _tools()
    if "log_grid" not in tools:        # a planlens without it
        import pytest
        pytest.skip("log_grid is not offered by the installed planlens")
    tools["log_grid"].invoke({"handle": "plan.pdf"})
    assert len(seen) == 1
    host = seen[0]
    signer = host.signer() if callable(host.signer) else host.signer
    assert signer == SIGNER, "log_grid ran under the process host"
