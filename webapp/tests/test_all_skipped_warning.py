"""The root conftest warns when every test of a top-level package skipped."""
import shutil
import textwrap
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

ROOT_CONFTEST = Path(__file__).resolve().parents[2] / "conftest.py"


def _hook_source():
    src = ROOT_CONFTEST.read_text(encoding="utf-8")
    return ("import pytest\nimport re as _re\n"
            + src[src.index("_MISSING_DEP"):])


@pytest.fixture
def tree(pytester):
    pytester.makeconftest(_hook_source())
    pytester.makepyfile(**{
        "pkg_a/test_a": textwrap.dedent("""
            import pytest
            @pytest.mark.skip(reason="dll blocked")
            def test_1(): pass
            @pytest.mark.skip(reason="other")
            def test_2(): pass
        """),
        "pkg_b/test_b": textwrap.dedent("""
            import pytest
            @pytest.mark.skip(reason="dll blocked")
            def test_1(): pass
            def test_2(): pass
        """),
    })
    return pytester


def test_warns_when_a_package_all_skipped(tree):
    result = tree.runpytest("pkg_a")
    result.stdout.fnmatch_lines(
        ["*ALL SKIPPED: pkg_a (2 tests): dll blocked*"])


def test_quiet_when_one_test_passes(tree):
    result = tree.runpytest("pkg_b")
    assert "ALL SKIPPED" not in result.stdout.str()


def test_exit_code_unchanged(tree):
    # all-skipped run exits 0 (pytest's own code for no failures)
    assert tree.runpytest("pkg_a").ret == 0
    tree.makepyfile(**{"pkg_c/test_c": "def test_f(): assert False\n"})
    result = tree.runpytest("pkg_a", "pkg_c")
    assert result.ret == 1
    result.stdout.fnmatch_lines(["*ALL SKIPPED: pkg_a*"])


def test_mixed_missing_dependency_listed(pytester):
    pytester.makeconftest(_hook_source())
    pytester.makepyfile(**{
        "pkg_m/test_m": textwrap.dedent("""
            import pytest
            def test_ok(): pass
            @pytest.mark.skip(reason="openseespy not installed")
            def test_a(): pass
            @pytest.mark.skip(reason="openseespy not installed")
            def test_b(): pass
            @pytest.mark.skip(reason="needs a key")
            def test_c(): pass
        """),
    })
    result = pytester.runpytest("pkg_m")
    result.stdout.fnmatch_lines([
        "*SKIPPED, a dependency missing: openseespy not installed "
        "(2 tests: pkg_m)*"])
    out = result.stdout.str()
    assert "needs a key" not in out.split("a dependency missing")[1]
    assert "ALL SKIPPED" not in out
    assert result.ret == 0
