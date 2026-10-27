"""geotech_common.optional_packages.import_error (BACKLOG INF-54): a package
that is missing and one that is installed but fails to import are told apart.

Every package name and error text here is made up; the failing imports are
faked by the root conftest's ``fail_import`` fixture.
"""

import sys

import pytest

from geotech_common.optional_packages import (
    CANNOT_IMPORT, NOT_INSTALLED, import_error,
)


def test_an_importable_module_has_no_error():
    assert import_error("json") is None


def test_a_missing_package_is_not_installed():
    msg = import_error("inf54_madeup_pkg", "inf54-madeup")
    assert msg == (f"inf54-madeup {NOT_INSTALLED}. "
                   "Install with: pip install inf54-madeup")


def test_the_package_name_defaults_to_the_import_name():
    assert import_error("inf54_madeup_pkg").startswith(
        f"inf54_madeup_pkg {NOT_INSTALLED}")


def test_a_module_set_to_none_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "inf54_madeup_pkg", None)
    assert NOT_INSTALLED in import_error("inf54_madeup_pkg")


def test_a_missing_parent_of_a_dotted_name_is_not_installed():
    msg = import_error("inf54_madeup_pkg.engine", "inf54-madeup")
    assert msg.startswith(f"inf54-madeup {NOT_INSTALLED}")


@pytest.mark.parametrize("exc", [
    ImportError("inf54 made-up: numpy.core.multiarray failed to import"),
    AttributeError("inf54 made-up: module 'numpy' has no attribute 'long'"),
    RuntimeError("inf54 made-up: the host refused to load libmadeup.so"),
    ValueError("inf54 made-up: dtype size changed"),
])
def test_an_installed_package_whose_import_fails_says_why(fail_import, exc):
    fail_import("inf54_madeup_pkg", exc)
    msg = import_error("inf54_madeup_pkg", "inf54-madeup")
    assert msg == (f"inf54-madeup {CANNOT_IMPORT}: "
                   f"{type(exc).__name__}: {exc}")
    assert NOT_INSTALLED not in msg


def test_a_missing_dependency_is_not_the_package_missing(fail_import):
    exc = ModuleNotFoundError("No module named 'inf54_madeup_dep'",
                              name="inf54_madeup_dep")
    fail_import("inf54_madeup_pkg", exc)
    msg = import_error("inf54_madeup_pkg")
    assert msg.startswith(f"inf54_madeup_pkg {CANNOT_IMPORT}")
    assert "ModuleNotFoundError: No module named 'inf54_madeup_dep'" in msg


def test_a_missing_submodule_of_the_package_is_an_import_failure(fail_import):
    exc = ModuleNotFoundError("No module named 'inf54_madeup_pkg._native'",
                              name="inf54_madeup_pkg._native")
    fail_import("inf54_madeup_pkg", exc)
    assert CANNOT_IMPORT in import_error("inf54_madeup_pkg")


def test_a_failing_parent_of_a_dotted_name_says_why(fail_import):
    fail_import("inf54_madeup_pkg",
                RuntimeError("inf54 made-up: DLL load blocked"))
    msg = import_error("inf54_madeup_pkg.engine", "inf54-madeup")
    assert msg == (f"inf54-madeup {CANNOT_IMPORT}: "
                   "RuntimeError: inf54 made-up: DLL load blocked")
