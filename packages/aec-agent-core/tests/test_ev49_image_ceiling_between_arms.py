"""BACKLOG EV-49 (2026-11-18): the review suite clears the turn's image
ceiling (``GEOTECH_TURN_MAX_IMAGE_CALLS``, TA-43) between arms, as it clears
the switches and their settings, and the turn's record names the ceiling
when an arm sets it. Offline: a made-up page host, no model, no app."""

from __future__ import annotations

import os
import types

import pytest

from aec_agent_core import hooks, review_flags
from aec_agent_core.deep import limits

ENV = limits.TURN_MAX_IMAGE_CALLS_ENV


@pytest.fixture
def clean_hooks(monkeypatch):
    """No app's hooks (restored afterwards)."""
    monkeypatch.setattr(hooks, "_HOOKS", {})
    monkeypatch.setattr(hooks, "_RESOLVED", {})


def _host(seen):
    """A page host whose turn writes down the ceiling and the switches the
    turn would record, as the page reads them while it runs."""
    host = types.SimpleNamespace(
        APP_DIST="made-up-app", APP_PACKAGE="made_up_app",
        APP_COMMIT_ENV="MADE_UP_APP_COMMIT")

    def build_agent(model, attachments, files_dir, artifacts, *, page,
                    markup_author):
        return {"files_dir": files_dir}

    def stage_upload(attachments, files_dir, name, data):
        path = os.path.join(files_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        attachments[name] = data
        return types.SimpleNamespace(path=path, name=name)

    def stream_turn(agent, history, thread, *, recursion_limit, callbacks):
        seen.append((limits.turn_max_image_calls(), review_flags.describe()))
        yield {"kind": "turn_done", "answer": "Made-up answer.",
               "turn_tokens": 1}

    host.build_agent = build_agent
    host.stage_upload = stage_upload
    host.attachment_note = lambda atts: f"[{len(atts)} attached]"
    host.assemble_user_message = lambda notes, text: " ".join(
        list(notes) + [text])
    host.orientation_request = lambda atts, page: None
    host.stream_turn = stream_turn
    host.default_recursion_limit = lambda: 7
    host.engine_record = lambda agent: {"model": "made-up"}
    host.tools_fingerprint = lambda agent: {"names": []}
    host.output_collector = lambda: None
    host.import_reported_outputs = lambda *a, **k: {}
    return host


def test_the_ceiling_is_a_later_setting_named_as_the_limits_name_it():
    assert review_flags.LATER_SETTINGS_ENVS == (ENV,)
    # apart from the two sets the geotech app's own tests pin whole (DR-51)
    assert ENV not in review_flags.SETTINGS_ENVS
    assert ENV not in review_flags.ALL_ENVS + review_flags.LATER_ENVS


def test_an_arm_that_does_not_name_the_ceiling_runs_without_one(monkeypatch):
    monkeypatch.setenv(ENV, "5")
    assert limits.turn_max_image_calls() == 5
    for arm in (review_flags.ARMS["baseline"], review_flags.ARMS["sheet_tools"],
                {"GEOTECH_REVIEW_MAX_MODEL_CALLS": "30"}):
        with review_flags.switches(arm):
            assert limits.turn_max_image_calls() == 0, arm
            assert ENV not in review_flags.describe(), arm
    # the notebook's own value is back after the arm
    assert os.environ[ENV] == "5"


def test_an_arm_that_names_the_ceiling_runs_with_it_and_says_so(monkeypatch):
    monkeypatch.setenv(ENV, "5")
    with review_flags.switches({ENV: "3"}):
        assert limits.turn_max_image_calls() == 3
        assert f"{ENV}=3" in review_flags.describe()
    assert os.environ[ENV] == "5"
    monkeypatch.delenv(ENV)
    assert ENV not in review_flags.describe()


def test_a_suite_run_clears_the_ceiling_the_environment_sets(
        tmp_path, clean_hooks, monkeypatch):
    """The done-when of EV-49, through the runner: with the ceiling set in
    the environment, a task under ``baseline`` runs with none and records
    none; under an arm that names it, the turn has it and records it."""
    from aec_agent_core.review_eval import runner
    from aec_agent_core.review_eval.tasks import Task
    seen = []
    hooks.set_hook(runner.PAGE_HOST_HOOK, lambda: _host(seen))
    monkeypatch.setenv(ENV, "5")
    task = Task(id="made-up", question="What is drawn?",
                documents=["fixture_review_document"], category="locate",
                doc_type="markup_set",
                checks=[{"type": "contains_all", "terms": ["made-up"],
                         "label": "answers"}])
    for arm, env, want in (("baseline", {}, 0),
                           ("ceiling_3", {ENV: "3"}, 3)):
        res = runner.run_task(task, model=None, arm=arm, arm_env=env,
                              docs_dir=None,
                              run_dir=str(tmp_path / arm))
        assert res["error"] is None, res.get("traceback") or res["error"]
        ceiling, recorded = seen[-1]
        assert ceiling == want, arm
        assert (f"{ENV}=3" in recorded) == bool(want), (arm, recorded)
    assert os.environ[ENV] == "5"
