"""Live smoke wave 1 fixes in the web app's turn bookkeeping.

* A10 -- an orientation turn titles the conversation after the attached
  files, the first typed question retitles it, and the SharePoint folder name
  drops backticks, commas and the ellipsis.
* A13 -- a file rewritten during a turn gets a card in that turn.
* A15d -- a chart card downloads the PNG twin, and the chart card stands for
  the figure's HTML copy too.
* The tool scratch folder (``files/.scratch``) and caches make no cards, are
  not listed to the model, and stay off the SharePoint mirror.
* A6 -- a relative path a tool reports is a name in the working folder.
"""

import os
import time

import pytest

import webapp.core as core
import webapp.sharepoint_store as sp
import webapp.turn_jobs as tj


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    for env in (sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET,
                sp.ENV_TOKEN, sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT):
        monkeypatch.delenv(env, raising=False)
    tj._JOBS.clear()
    yield


def _mk_conv(tid):
    core.ensure_conversation(tid)
    return tid, core.conversation_files_dir(tid)


def _ctx(files, transcript, artifacts, prompt, **extra):
    ctx = {"prompt": prompt, "temp_dir": files,
           "before": core.snapshot_dir(files),
           "before_mtimes": core.snapshot_mtimes(files),
           "staged_inputs": set(), "working_dir": files, "before_wd": None,
           "artifacts": artifacts, "artifacts_before_len": len(artifacts),
           "transcript": transcript, "trace_on": False, "model": "m",
           "behavior": core.default_behavior()}
    ctx.update(extra)
    return ctx


def _run_turn(monkeypatch, tid, files, transcript, prompt, events=None,
              during=None, **extra):
    """One turn through the real worker with a scripted stream; ``during``
    runs inside the stream (the agent's file writes)."""
    def stream_turn(agent, messages, thread_id, recursion_limit=None, **kw):
        if during:
            during()
        yield from (events or [{"kind": "turn_done", "answer": "ok",
                                "turn_tokens": 1}])
    monkeypatch.setattr(core, "stream_turn", stream_turn)
    transcript.append({"role": "user", "text": prompt})
    job = tj.start_turn_job(object(), [{"role": "user", "content": prompt}],
                            tid, 50,
                            _ctx(files, transcript, extra.pop("artifacts", []),
                                 prompt, **extra))
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done
    return job


# ---------------------------------------------------------------------------
# A10: titles and folder names
# ---------------------------------------------------------------------------

ORIENT = ("I just attached `21.01.pdf`. Before I ask anything, give me a "
          "short orientation: open it ...")


def test_orientation_title_names_the_files():
    assert core.orientation_title(["21.01.pdf"]) == "21.01.pdf"
    assert core.orientation_title(["a.pdf", "b.pdf"]) == "a.pdf + b.pdf"
    assert core.orientation_title(["a", "b", "c", "d", "e"]) == \
        "a + b + c + 2 more"
    assert core.orientation_title([]) == "New conversation"


def test_turn_title_rules():
    # the first turn is the orientation: titled after the files
    assert core.turn_title({}, ORIENT, 1, orientation=["21.01.pdf"]) == \
        ("21.01.pdf", core.TITLE_FROM_ATTACHMENTS)
    # the first typed question replaces an attachments title
    att = {"title_source": core.TITLE_FROM_ATTACHMENTS}
    assert core.turn_title(att, "Where is the GCE tag?", 2) == \
        ("Where is the GCE tag?", core.TITLE_FROM_QUESTION)
    # a later orientation (a second upload) keeps whatever title there is
    q = {"title_source": core.TITLE_FROM_QUESTION}
    assert core.turn_title(q, ORIENT, 3, orientation=["b.pdf"]) == (None, None)
    assert core.turn_title(att, ORIENT, 2, orientation=["b.pdf"]) == \
        (None, None)
    # later typed turns keep a question title; a user's rename is never lost
    assert core.turn_title(q, "next", 3) == (None, None)
    user = {"title_source": core.TITLE_FROM_USER}
    assert core.turn_title(user, "anything", 1) == (None, None)
    # an old conversation (no source) is titled by its first turn, as before
    assert core.turn_title({}, "bearing capacity of a strip", 1) == \
        ("bearing capacity of a strip", core.TITLE_FROM_QUESTION)


def test_an_orientation_then_a_question_through_the_worker(monkeypatch):
    tid, files = _mk_conv("TT1")
    transcript = []
    _run_turn(monkeypatch, tid, files, transcript, ORIENT,
              orientation=["21.01.pdf"])
    meta = core.load_meta(tid)
    assert meta["title"] == "21.01.pdf"
    assert meta["title_source"] == core.TITLE_FROM_ATTACHMENTS
    _run_turn(monkeypatch, tid, files, transcript,
              "Which sheets carry the GCE tag?")
    meta = core.load_meta(tid)
    assert meta["title"] == "Which sheets carry the GCE tag?"
    assert meta["title_source"] == core.TITLE_FROM_QUESTION
    _run_turn(monkeypatch, tid, files, transcript, "and the FBG tag?")
    assert core.load_meta(tid)["title"] == "Which sheets carry the GCE tag?"


def test_a_renamed_conversation_keeps_its_name(monkeypatch):
    tid, files = _mk_conv("TT2")
    transcript = []
    _run_turn(monkeypatch, tid, files, transcript, ORIENT,
              orientation=["plans.pdf"])
    core.rename_conversation(tid, "Bridge 21 plans")
    _run_turn(monkeypatch, tid, files, transcript, "first question")
    assert core.load_meta(tid)["title"] == "Bridge 21 plans"


def test_folder_names_drop_backticks_commas_and_the_ellipsis():
    name = sp.sanitize_folder_name(
        "I just attached `21.01.pdf`. Before I ask anything,…")
    assert "`" not in name and "," not in name and "…" not in name
    assert name == "I_just_attached_21.01.pdf._Before_I_ask_anything"
    assert sp.sanitize_folder_name("a,b") == "a_b"
    assert sp.sanitize_folder_name("21.01.pdf + b.pdf") == "21.01.pdf_+_b.pdf"


# ---------------------------------------------------------------------------
# A13: a rewritten file gets this turn's card
# ---------------------------------------------------------------------------

def test_a_file_rewritten_in_a_turn_gets_that_turns_card(monkeypatch):
    tid, files = _mk_conv("TT3")
    memo = os.path.join(files, "memo.docx")
    artifacts = []
    save_fn = core.make_save_fn(files, artifacts)
    transcript = []
    _run_turn(monkeypatch, tid, files, transcript, "write the memo",
              during=lambda: save_fn("memo.docx", b"v1"),
              artifacts=artifacts)
    assert transcript[-1]["artifacts"] == [memo]

    def rewrite():
        time.sleep(0.01)
        save_fn("memo.docx", b"version 2, longer")
    _run_turn(monkeypatch, tid, files, transcript, "fix the memo",
              during=rewrite, artifacts=artifacts)
    assert transcript[-1]["artifacts"] == [memo]      # carded again
    assert artifacts == [memo]                        # listed once
    # a turn that touches nothing cards nothing
    _run_turn(monkeypatch, tid, files, transcript, "thanks",
              artifacts=artifacts)
    assert transcript[-1]["artifacts"] == []


def test_rewritten_files_leaves_out_staged_inputs(tmp_path):
    f = tmp_path / "upload.pdf"
    f.write_bytes(b"one")
    before = core.snapshot_mtimes(str(tmp_path))
    time.sleep(0.01)
    f.write_bytes(b"two!")
    assert core.rewritten_files(str(tmp_path), before) == [str(f)]
    assert core.rewritten_files(str(tmp_path), before, [str(f)]) == []
    assert core.rewritten_files(str(tmp_path), None) == []


# ---------------------------------------------------------------------------
# A15d: the chart card
# ---------------------------------------------------------------------------

def test_a_chart_card_downloads_the_png_and_stands_for_the_html(tmp_path):
    side = tmp_path / "pressures.plotly.json"
    side.write_text("{}")
    assert core.plotly_download_twin(str(side)) == str(side)   # no twin yet
    (tmp_path / "pressures.html").write_text("<html/>")
    assert core.plotly_download_twin(str(side)) == \
        str(tmp_path / "pressures.html")
    (tmp_path / "pressures.png").write_bytes(b"png")
    assert core.plotly_download_twin(str(side)) == \
        str(tmp_path / "pressures.png")
    assert core.plotly_download_twin(str(tmp_path / "x.pdf")) == \
        str(tmp_path / "x.pdf")
    cards = core.collect_turn_artifacts(
        [], [str(tmp_path / "pressures.html"), str(side),
             str(tmp_path / "pressures.png"), str(tmp_path / "report.html")])
    assert cards == [str(side), str(tmp_path / "report.html")]


# ---------------------------------------------------------------------------
# The scratch folder
# ---------------------------------------------------------------------------

def test_scratch_and_dot_folders_make_no_cards(tmp_path):
    work = tmp_path / "files"
    work.mkdir()
    assert ".scratch" in core.CACHE_DIRS
    before = core.snapshot_dir(str(work))
    (work / ".scratch").mkdir()
    (work / ".scratch" / "thumbs_0-29.png").write_bytes(b"x")
    (work / "sub" / ".hidden").mkdir(parents=True)
    (work / "sub" / ".hidden" / "y.png").write_bytes(b"y")
    (work / "find_like_sheet.png").write_bytes(b"z")
    new = core.new_artifacts(str(work), before, [])
    assert [os.path.basename(p) for p in new] == ["find_like_sheet.png"]
    assert core.is_cache_path(str(work / ".scratch" / "thumbs_0-29.png"),
                              str(work))
    assert core.is_cache_path(str(work / "digest" / "a" / "i.sqlite"),
                              str(work))
    assert not core.is_cache_path(str(work / "memo.docx"), str(work))
    # a save into the scratch is not recorded as a deliverable either
    arts = []
    core.make_save_fn(str(work), arts)(str(work / ".scratch" / "s.png"), b"s")
    assert arts == []


def test_scratch_files_are_not_listed_to_the_model(monkeypatch):
    tid, files = _mk_conv("TT4")
    os.makedirs(os.path.join(files, ".scratch"))
    sheet = os.path.join(files, ".scratch", "sheet.png")
    with open(sheet, "wb") as fh:
        fh.write(b"x")
    memo = os.path.join(files, "memo.docx")
    with open(memo, "wb") as fh:
        fh.write(b"x")
    tr = [{"role": "user", "text": "q"},
          {"role": "assistant", "text": "a", "artifacts": [sheet, memo]}]
    note = core.working_files_note(files, tr)
    assert "'memo.docx'" in note and "sheet.png" not in note


def test_the_mirror_leaves_out_the_scratch_and_the_cache(tmp_path):
    root = str(tmp_path / "mroot")
    tid = "MIR1"
    core.ensure_conversation(tid, root=root)
    files = core.conversation_files_dir(tid, root)
    os.makedirs(os.path.join(files, ".scratch"))
    with open(os.path.join(files, ".scratch", "x.png"), "wb") as fh:
        fh.write(b"scratch")
    os.makedirs(os.path.join(files, "digest", "abc"))
    with open(os.path.join(files, "digest", "abc", "index.sqlite"), "wb") as fh:
        fh.write(b"cache")
    with open(os.path.join(files, "memo.docx"), "wb") as fh:
        fh.write(b"deliverable")

    class FM:
        def __init__(self):
            self.uploads = []

        def create_folder(self, path):
            return True

        def upload_file(self, local, remote, overwrite=False):
            self.uploads.append(remote)
            return True

        def get_web_url(self, path):
            return f"https://sp.example/{path}"

    fm = FM()
    out = sp.SharePointStore(file_manager=fm).mirror_conversation(tid,
                                                                  root=root)
    assert not out["errors"]
    names = [r.rsplit("/", 1)[-1] for r in fm.uploads]
    assert "memo.docx" in names
    assert "x.png" not in names and "index.sqlite" not in names
    assert not any("/.scratch" in r or "/digest/" in r for r in fm.uploads)


def test_mirror_skips_only_the_working_folders_caches():
    assert core.mirror_skips_dir("files/.scratch")
    assert core.mirror_skips_dir("files/digest")
    assert core.mirror_skips_dir("files/sub/.tmp")
    assert not core.mirror_skips_dir("files")
    assert not core.mirror_skips_dir("files/sub")
    assert not core.mirror_skips_dir("files/sub/digest")   # not the cache
    assert not core.mirror_skips_dir(".")


# ---------------------------------------------------------------------------
# A6: relative reported paths
# ---------------------------------------------------------------------------

def test_a_relative_reported_output_is_a_name_in_the_working_folder(tmp_path):
    conv = tmp_path / "conv"
    files = conv / "files"
    files.mkdir(parents=True)
    work = tmp_path / "custom_work"
    work.mkdir()
    (work / "plot.png").write_bytes(b"png")
    # relative to a custom working folder: copied into files/
    copied = core.import_reported_outputs(["plot.png"], str(files),
                                          working_dir=str(work))
    assert copied == {str(work / "plot.png"): str(files / "plot.png")}
    assert (files / "plot.png").read_bytes() == b"png"
    # relative to the default working folder (files/ itself): already there
    (files / "memo.docx").write_bytes(b"d")
    assert core.import_reported_outputs(["memo.docx"], str(files)) == {}
    # an excluded relative name is excluded
    (work / "input.pdf").write_bytes(b"in")
    assert core.import_reported_outputs(
        ["input.pdf"], str(files), exclude=["input.pdf"],
        working_dir=str(work)) == {}


def test_a_relative_output_is_imported_and_carded_through_the_worker(
        monkeypatch, tmp_path):
    """A tool reports ``output_path: "fig.png"`` (relative) for a file in a
    custom working folder: the turn copies it in and cards it."""
    tid, files = _mk_conv("TT5")
    work = tmp_path / "custom"
    work.mkdir()

    class _Collector:
        def __init__(self):
            self.outputs, self.inputs = [], []

        def on_tool_end(self, *a, **k):
            pass

    import webapp.output_capture as oc

    def during():
        (work / "fig.png").write_bytes(b"png")
        made[0].outputs.append("fig.png")

    made = []

    def factory():
        c = _Collector()
        made.append(c)
        return c

    monkeypatch.setattr(oc, "OutputCollector", factory)
    transcript = []
    # before_wd=None: the working-folder bridge is off, so only the
    # reported (relative) output can bring the file in.
    _run_turn(monkeypatch, tid, files, transcript, "plot it", during=during,
              working_dir=str(work), before_wd=None)
    cards = [os.path.basename(p) for p in transcript[-1]["artifacts"]]
    assert cards == ["fig.png"]
    assert os.path.isfile(os.path.join(files, "fig.png"))
