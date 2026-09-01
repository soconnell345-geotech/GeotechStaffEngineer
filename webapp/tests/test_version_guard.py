"""Runtime version-drift guard (webapp.version_guard) + diagnostics row."""

import webapp.version_guard as vg


def test_within_range_is_quiet():
    ok = {"deepagents": "0.7.11", "langchain": "1.3.18",
          "langgraph": "1.2.11", "openai": "2.52.0", "streamlit": "1.62.0"}
    assert vg.check_versions(overrides=ok) == []
    assert vg.drift_summary(overrides=ok) == "agent-stack versions: OK"


def test_newer_than_tested_warns():
    warnings = vg.check_versions(overrides={"deepagents": "0.8.1"})
    assert len(warnings) == 1
    assert "deepagents 0.8.1" in warnings[0] and "NEWER" in warnings[0]
    # The message teaches the fix, not just the fact.
    assert "pinned" in warnings[0]


def test_openai_major_bump_warns():
    warnings = vg.check_versions(overrides={"openai": "3.6.0"})
    assert len(warnings) == 1 and "openai 3.6.0" in warnings[0]


def test_older_than_floor_warns():
    warnings = vg.check_versions(overrides={"deepagents": "0.5.3"})
    assert len(warnings) == 1 and "OLDER" in warnings[0]


def test_missing_package_is_skipped():
    assert vg.check_versions(overrides={"deepagents": None}) == []


def test_suffix_versions_parse():
    assert vg._ver_tuple("1.62.0rc1") == (1, 62, 0)
    assert vg._ver_tuple("0.7.11.post1") == (0, 7, 11)


def test_guard_mirrors_pyproject_caps():
    """The guard's ceilings must stay in lockstep with the pyproject caps."""
    import pathlib, re
    py = pathlib.Path(__file__).parents[2] / "pyproject.toml"
    text = py.read_text(encoding="utf-8")
    for pkg in ("deepagents", "langchain", "langgraph", "openai"):
        cap = vg.TESTED_MAX_EXCLUSIVE[pkg][1]
        assert re.search(rf'"{pkg}[^"]*<{re.escape(cap)}"', text), \
            f"pyproject cap for {pkg} does not match guard ceiling <{cap}"


def test_diagnostics_includes_drift_row(monkeypatch):
    from webapp import diagnostics
    rows = diagnostics.run_diagnostics()
    names = [r["name"] for r in rows]
    assert "version drift" in names
    drift = next(r for r in rows if r["name"] == "version drift")
    assert drift["status"] in ("pass", "warn")
