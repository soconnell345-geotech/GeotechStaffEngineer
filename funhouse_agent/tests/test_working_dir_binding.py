"""The working folder is bound per turn, not read from a process-wide env var.

On a shared host (Tiny Apps) several people's turns run in one process; the
web app repoints ``GEOTECH_DEFAULT_OUTPUT_DIR`` on every rerun, so a turn that
read only the env var could write into another person's folder.
"""
import os
import threading

from funhouse_agent import _fileio
from funhouse_agent.dispatch import _outputs_into_working_folder


def test_binding_wins_over_the_env(tmp_path, monkeypatch):
    mine, other = tmp_path / "mine", tmp_path / "other"
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(other))
    with _fileio.working_dir_bound(mine):
        assert _fileio.default_output_dir() == str(mine)
        assert _fileio.host_output_dir() == str(mine)
        params, _ = _outputs_into_working_folder({"output_path": "/tmp/a.xml"})
        assert params["output_path"] == os.path.join(str(mine), "a.xml")
    assert _fileio.default_output_dir() == str(other)


def test_unbound_falls_back_to_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(tmp_path))
    assert _fileio.host_output_dir() == str(tmp_path)


def test_two_turns_at_once_keep_their_own_folders(tmp_path, monkeypatch):
    """Two worker threads, each bound to its own folder, while the env var is
    repointed underneath them: each writer lands in its own folder."""
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(tmp_path / "env"))
    start = threading.Barrier(2)
    seen = {}

    def turn(name):
        _fileio.bind_working_dir(tmp_path / name)
        start.wait()
        os.environ[_fileio.DEFAULT_OUTPUT_DIR_ENV] = str(tmp_path / "rerun")
        params, _ = _outputs_into_working_folder(
            {"output_path": f"/tmp/{name}.xml"})
        seen[name] = params["output_path"]

    threads = [threading.Thread(target=turn, args=(n,)) for n in ("ann", "bo")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen["ann"] == os.path.join(str(tmp_path / "ann"), "ann.xml")
    assert seen["bo"] == os.path.join(str(tmp_path / "bo"), "bo.xml")


def test_the_turn_worker_binds_its_conversation_folder(tmp_path, monkeypatch):
    """webapp.turn_jobs binds ctx['working_dir'] in the worker thread, so a
    tool that runs during the turn sees it even when the env var points
    elsewhere."""
    from webapp import turn_jobs
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(tmp_path / "other"))
    seen = {}

    class _Agent:
        def stream(self, *a, **k):
            seen["dir"] = _fileio.default_output_dir()
            return iter(())

    mine = tmp_path / "mine"
    mine.mkdir()
    monkeypatch.setattr(turn_jobs.core, "stream_turn",
                        lambda agent, *a, **k: (agent.stream(), iter(()))[1])
    job = turn_jobs.TurnJob("t-bind", "hi")
    ctx = {"working_dir": str(mine), "temp_dir": str(mine), "before": set(),
           "staged_inputs": [], "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": [], "prompt": "hi"}
    t = threading.Thread(target=turn_jobs._run_turn_job,
                         args=(job, _Agent(), [], "t-bind", 10, ctx))
    t.start()
    t.join(30)
    assert seen.get("dir") == str(mine)


def test_heartbeat_pump_keeps_the_turn_binding(tmp_path, monkeypatch):
    """Live smoke 2c (F47): core.with_heartbeat ran the agent stream on a bare
    thread, so the turn's folder binding was lost and two testers wrote into
    each other's folders."""
    from webapp import core
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(tmp_path / "env"))

    def gen():
        yield _fileio.default_output_dir()

    with _fileio.working_dir_bound(tmp_path / "mine"):
        items = [i for i in core.with_heartbeat(gen(), interval_s=5)
                 if not (isinstance(i, dict) and i.get("kind") == "heartbeat")]
    assert items == [str(tmp_path / "mine")]


def test_shared_host_refuses_an_unbound_folder(tmp_path, monkeypatch):
    from webapp import core
    monkeypatch.setenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, str(tmp_path / "env"))
    monkeypatch.setattr(_fileio, "_REQUIRE_BINDING", True)
    import pytest
    with pytest.raises(_fileio.UnboundWorkingDir):
        _fileio.default_output_dir()
    with _fileio.working_dir_bound(tmp_path / "mine"):
        assert _fileio.default_output_dir() == str(tmp_path / "mine")
    # the app never sets the shared env var on such a host
    core.apply_default_output_dir(str(tmp_path / "someone"))
    assert os.environ.get(_fileio.DEFAULT_OUTPUT_DIR_ENV) is None


def test_a_header_identity_turns_on_shared_host_mode(monkeypatch):
    from webapp import identity
    monkeypatch.setattr(_fileio, "_REQUIRE_BINDING", False)
    fake = identity.parse_principal(r"corp\jdoe", "header")
    if fake is None or not fake.multi_user:
        import pytest
        pytest.skip("header identity shape differs")
    monkeypatch.setattr(identity, "from_header_values", lambda *_: fake)
    identity.current_identity()
    assert _fileio.turn_binding_required()
