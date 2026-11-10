"""The package itself (BACKLOG INF-33, step 13): its version, its exact pins,
the hooks it fills, its entry points and its suite."""

import os
import tomllib

import pytest

import aec_document_review

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)["project"]


def test_the_version_is_one_number():
    assert _project()["version"] == aec_document_review.__version__


def test_our_two_packages_are_pinned_exactly():
    deps = _project()["dependencies"]
    assert any(d.replace(" ", "").startswith("aec-agent-core==") for d in deps)
    assert any(d.replace(" ", "").startswith("planlens==") for d in deps)
    assert not any(d.split("[")[0].split("=")[0].split(">")[0].strip()
                   in ("geotech-staff-engineer", "geotech-references")
                   for d in deps)


def test_the_core_pin_is_the_installed_core():
    import aec_agent_core
    pin = next(d for d in _project()["dependencies"]
               if d.startswith("aec-agent-core=="))
    assert pin.split("==")[1] == aec_agent_core.__version__


def test_loading_the_package_fills_its_hooks():
    from aec_agent_core.hooks import hooks
    filled = hooks()
    for name in ("prompter.get_setting", "prompter.resolve_for",
                 "prompter.tester_keys_possible", "app.details"):
        assert name in filled, name
    from aec_agent_core import app_details
    assert app_details.details() == {"name": "Document Review",
                                     "dist": "aec-document-review",
                                     "package": "aec_document_review"}


def test_the_core_names_this_app():
    from aec_agent_core import document_tools, runtime_check
    from aec_agent_core.review_eval import runner
    assert document_tools.DEFAULT_MARKUP_AUTHOR == \
        "Document Review (AI draft)"
    assert "Document Review" in document_tools.OWN_MARK_WORDS
    assert runner.SUITE_AUTHOR == "Review suite via Document Review (AI draft)"
    assert runtime_check._who() == "aec_document_review"


def test_no_chart_lines_are_registered():
    from aec_agent_core.charts import reference_charts
    assert reference_charts() is None


def test_one_page_and_it_is_document_review():
    from aec_document_review import profiles
    assert list(profiles.PROFILES) == ["document_review"]
    assert profiles.DEFAULT is profiles.DOCUMENT_REVIEW
    assert profiles.DOCUMENT_REVIEW.version_dists == ("aec-document-review",)
    assert profiles.DOCUMENT_REVIEW.turn_checks is None
    assert profiles.DOCUMENT_REVIEW.diagnostics_checks is None


def test_the_page_shows_its_own_notice():
    from aec_document_review import core, profiles
    text = core.disclaimer_text(profiles.DOCUMENT_REVIEW)
    assert text.startswith("Document Review")
    assert "geotechnical" not in text.lower()


def test_the_entry_points_exist():
    from aec_document_review import launch, tinyapps_entry
    assert callable(tinyapps_entry.main)
    assert os.path.isfile(launch.entry_path())
    from aec_document_review import pages_entry
    assert os.path.isfile(pages_entry.app_path())
    scripts = _project()["scripts"]
    assert scripts["aec-document-review"] == "aec_document_review.launch:main"


def test_the_uploader_component_ships():
    from aec_document_review import ws_upload  # noqa: F401
    here = os.path.dirname(os.path.abspath(aec_document_review.__file__))
    assert os.path.isfile(os.path.join(here, "ws_upload_component",
                                       "index.html"))


def test_the_suite_has_its_50_tasks_all_on_this_page():
    from aec_document_review.review_eval import OPEN_TASKS
    assert len(OPEN_TASKS) == 50
    assert {t.page for t in OPEN_TASKS} == {"review"}
    assert len({t.id for t in OPEN_TASKS}) == 50


def test_the_suite_dry_run_lists_its_tasks(tmp_path):
    from aec_document_review.review_eval import score_review_suite
    out = score_review_suite(out_dir=str(tmp_path), dry_run=True)
    text = out["results_md"]
    assert text.startswith("# Dry run")
    assert "50 tasks" in text


def test_no_text_the_package_shows_names_geotechstaffengineer():
    """What a Document Review user sees or is sent says Document Review (the
    step's review fixes). Docstrings and comments may say where the code
    came from; the SharePoint folder's default keeps the owner's name for
    it."""
    import ast
    kept = {"Shared Documents/GeotechStaffEngineer"}
    found = []
    for folder, _dirs, files in os.walk(os.path.join(ROOT,
                                                     "aec_document_review")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
            docs = {id(n.body[0].value) for n in ast.walk(tree)
                    if isinstance(n, (ast.Module, ast.ClassDef,
                                      ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.body and isinstance(n.body[0], ast.Expr)
                    and isinstance(n.body[0].value, ast.Constant)}
            for n in ast.walk(tree):
                if (isinstance(n, ast.Constant) and id(n) not in docs
                        and isinstance(n.value, (str, bytes))):
                    text = (n.value.decode("utf-8", "replace")
                            if isinstance(n.value, bytes) else n.value)
                    if "GeotechStaffEngineer" in text and text not in kept:
                        found.append(f"{os.path.relpath(path, ROOT)}:"
                                     f"{n.lineno}")
    assert not found, found
