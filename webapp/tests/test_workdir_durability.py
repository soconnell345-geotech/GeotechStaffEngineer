"""Offline tests for the working-folder picker (A6) and durable files/links (A4).

No streamlit, no live model — exercises webapp.core directly:
* A6 — per-conversation working folder + the agent default-output-dir env hook +
  external-artifact import into the durable files/ dir.
* A4 — artifact refs persist RELATIVE to files/ and re-resolve after the whole
  conversation directory is RELOCATED (the weeks-later / app-upgrade / other-
  machine story), and a missing artifact fails soft.
"""

import os

from webapp import core


# ---------------------------------------------------------------------------
# A6 — working folder
# ---------------------------------------------------------------------------

def test_working_dir_defaults_to_files_dir(tmp_path):
    root = str(tmp_path)
    tid = "c"
    wd = core.working_dir_for(tid, root=root)
    assert wd == core.conversation_files_dir(tid, root=root)
    assert os.path.isdir(wd)


def test_set_working_dir_persists_and_resolves(tmp_path):
    root = str(tmp_path)
    tid = "c"
    custom = str(tmp_path / "project_out")
    resolved = core.set_working_dir(tid, custom, root=root)
    assert resolved == os.path.abspath(custom)
    assert os.path.isdir(resolved)
    # persisted in meta and read back by working_dir_for
    assert core.load_meta(tid, root=root)["working_dir"] == os.path.abspath(custom)
    assert core.working_dir_for(tid, root=root) == os.path.abspath(custom)


def test_set_working_dir_blank_resets_to_files_dir(tmp_path):
    root = str(tmp_path)
    tid = "c"
    core.set_working_dir(tid, str(tmp_path / "x"), root=root)
    back = core.set_working_dir(tid, "", root=root)          # blank => default
    assert back == core.conversation_files_dir(tid, root=root)
    assert core.load_meta(tid, root=root)["working_dir"] is None


def test_apply_default_output_dir_sets_and_clears_env(monkeypatch):
    monkeypatch.delenv("GEOTECH_DEFAULT_OUTPUT_DIR", raising=False)
    core.apply_default_output_dir(r"/data/proj")
    assert os.environ["GEOTECH_DEFAULT_OUTPUT_DIR"] == r"/data/proj"
    core.apply_default_output_dir(None)
    assert "GEOTECH_DEFAULT_OUTPUT_DIR" not in os.environ


def test_apply_default_output_dir_is_the_calc_package_hook(monkeypatch, tmp_path):
    """The env the app sets is exactly what the tool layer reads for the default
    save dir — so calc packages land in the working folder, not system temp."""
    from funhouse_agent import _fileio
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    core.apply_default_output_dir(str(tmp_path))
    try:
        assert _fileio.default_output_dir() == os.path.abspath(str(tmp_path))
    finally:
        core.apply_default_output_dir(None)


def test_import_external_artifacts_copies_into_files_dir(tmp_path):
    root = str(tmp_path)
    tid = "c"
    files_dir = core.conversation_files_dir(tid, root=root)
    work = str(tmp_path / "external")
    os.makedirs(work)
    before = core.snapshot_dir(work)
    # agent writes a calc package into the custom working folder
    produced = os.path.join(work, "bearing_calc.html")
    with open(produced, "w", encoding="utf-8") as fh:
        fh.write("<html>calc</html>")
    copied = core.import_external_artifacts(work, files_dir, before, input_paths=[])
    assert len(copied) == 1
    dst = copied[0]
    assert os.path.dirname(os.path.abspath(dst)) == os.path.abspath(files_dir)
    assert core.read_text(dst) == "<html>calc</html>"        # durable copy


def test_import_external_artifacts_noop_when_same_dir(tmp_path):
    root = str(tmp_path)
    files_dir = core.conversation_files_dir("c", root=root)
    assert core.import_external_artifacts(files_dir, files_dir, set(), []) == []


def test_import_external_artifacts_excludes_inputs(tmp_path):
    root = str(tmp_path)
    files_dir = core.conversation_files_dir("c", root=root)
    work = str(tmp_path / "w")
    os.makedirs(work)
    staged = os.path.join(work, "upload.pdf")
    open(staged, "w").close()
    before = core.snapshot_dir(work)                          # upload already there
    out = os.path.join(work, "out.html")
    open(out, "w").close()
    copied = core.import_external_artifacts(work, files_dir, before,
                                            input_paths=[staged])
    assert [os.path.basename(p) for p in copied] == ["out.html"]


def test_import_external_artifacts_uniquifies_collision(tmp_path):
    root = str(tmp_path)
    files_dir = core.conversation_files_dir("c", root=root)
    # a same-named file already exists in files/
    with open(os.path.join(files_dir, "plot.html"), "w") as fh:
        fh.write("OLD")
    work = str(tmp_path / "w")
    os.makedirs(work)
    before = core.snapshot_dir(work)
    with open(os.path.join(work, "plot.html"), "w") as fh:
        fh.write("NEW")
    copied = core.import_external_artifacts(work, files_dir, before, [])
    assert len(copied) == 1
    assert os.path.basename(copied[0]) == "plot_1.html"       # no clobber
    assert core.read_text(copied[0]) == "NEW"
    assert core.read_text(os.path.join(files_dir, "plot.html")) == "OLD"


# ---------------------------------------------------------------------------
# A4 — durable files + links
# ---------------------------------------------------------------------------

def test_artifact_ref_relative_and_resurvives_relocation(tmp_path):
    """The weeks-later story: an artifact under files/ is stored RELATIVE, so
    after the whole data root is relocated (upgrade / other machine) the card
    still resolves and the bytes are readable."""
    root1 = str(tmp_path / "root1")
    tid = "conv"
    files_dir = core.conversation_files_dir(tid, root=root1)
    art = os.path.join(files_dir, "slope_report.html")
    with open(art, "w", encoding="utf-8") as fh:
        fh.write("<html>report</html>")
    core.append_transcript(tid, {"role": "assistant", "text": "done",
                                 "artifacts": [art]}, root=root1)
    # stored portably (relative, no absolute leak)
    raw = open(core._conv_path(tid, "transcript.jsonl", root1)).read()
    assert '"slope_report.html"' in raw
    assert files_dir not in raw and tmp_path.name not in raw.replace(
        "slope_report.html", "")

    # RELOCATE the whole data root (simulate restart on a moved/upgraded install)
    import shutil
    root2 = str(tmp_path / "root2")
    shutil.move(root1, root2)

    loaded = core.load_transcript(tid, root=root2)
    resolved = loaded[0]["artifacts"][0]
    card = core.describe_artifact(resolved)
    assert card.exists                                        # re-resolved on disk
    assert card.kind == "html"
    assert core.read_text(resolved) == "<html>report</html>"


def test_missing_artifact_fails_soft(tmp_path):
    """A deleted/absent artifact resolves to a non-existent path whose card
    reports exists=False (the app then shows a soft note, not a crash)."""
    root = str(tmp_path)
    tid = "c"
    files_dir = core.conversation_files_dir(tid, root=root)
    gone = os.path.join(files_dir, "vanished.pdf")
    core.append_transcript(tid, {"role": "assistant", "text": "x",
                                 "artifacts": [gone]}, root=root)
    loaded = core.load_transcript(tid, root=root)
    card = core.describe_artifact(loaded[0]["artifacts"][0])
    assert card.exists is False
    assert card.size == 0


def test_external_absolute_artifact_ref_is_documented_leak(tmp_path):
    """An artifact written OUTSIDE files/ is stored ABSOLUTE (not portable) —
    this is the pre-A6 C:/tmp leak the working-folder default fixes by routing
    saves back INTO files/. import_external_artifacts is the durable bridge."""
    root = str(tmp_path)
    tid = "c"
    outside = str(tmp_path / "system_temp" / "calc.html")
    os.makedirs(os.path.dirname(outside))
    open(outside, "w").close()
    core.append_transcript(tid, {"role": "assistant", "text": "x",
                                 "artifacts": [outside]}, root=root)
    # Stored + re-resolved as an ABSOLUTE path (not portable) — the leak A6 avoids
    # by defaulting saves back into files/ (stored relative, see the test above).
    ref = core.load_transcript(tid, root=root)[0]["artifacts"][0]
    assert os.path.isabs(ref) and ref == os.path.abspath(outside)
