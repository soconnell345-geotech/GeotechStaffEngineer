"""The model-first switch in the ingest graph (BACKLOG RI-87, piece 8).

``module_work/REPORT_INGEST_PLAN.md`` section 8.2.2: with the switch off,
which is the default, the graph is today's. Two tests hold it there: every
request the scripted run sends and every file it writes are the same as a
run of master's graph (a snapshot of master's run, :data:`SNAPSHOT`), and no
model-first module is imported. The rest run one made-up report through the
graph with the switch on and scripted engines.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest import mock

import pytest

pytest.importorskip("planlens.document.roles")

#: Master's run of the synthetic report, with the switch unset: one digest
#: per request and per output file (:func:`off_snapshot`), and the digest of
#: what planlens reads off the report (:func:`planlens_reading`), taken with
#: it, and the platform and library versions it was taken with
#: (``taken_with``, which the test prints and does not compare). Written
#: from master's graph by :func:`write_off_snapshot`, which is run again
#: only when today's path changes on purpose.
SNAPSHOT = Path(__file__).with_name("graph_off_snapshot.json")

#: The runs the snapshot holds: a standard run (the vision voter agreeing
#: everywhere, the report bound in at pages 15 to 18 read as its own
#: record), and the rules alone with the label review over every page.
OFF_RUNS: Tuple[str, ...] = ("standard", "rules")

#: The modules only the switch's ``on`` imports.
MODEL_FIRST_MODULES: Tuple[str, ...] = (
    "report_ingest.model_first", "report_ingest.model_first_pages",
    "report_ingest.model_first_read", "report_ingest.locate")

#: The graph's functions only the switch's ``on`` calls.
MODEL_FIRST_FUNCTIONS: Tuple[str, ...] = (
    "_model_first_run", "_placer", "_model_first_plan",
    "_model_first_units", "_read_model_first", "_still_current",
    "_part_skipped", "_read_unit", "_into_record", "_placement_qa",
    "_join_by_hole", "_ruler_slack", "_continues_log",
    "_bound_range_labels")


# ---------------------------------------------------------------------------
# the snapshot of master's run
# ---------------------------------------------------------------------------

class _Clock:
    """``time`` for the graph, standing still: a run's seconds are 0."""

    @staticmethod
    def time() -> float:
        return 1000.0


def _off_run(name: str, folder: Path
             ) -> Tuple[List[Dict[str, Any]], Dict[str, bytes],
                        Dict[bytes, str]]:
    """One scripted run of the synthetic report with the switch unset: the
    requests the engine was sent, every file under the run's folder, and
    the render call that drew each picture planlens drew for it (by the
    picture's digest): its arguments, the budget object left out."""
    from planlens.document import Document
    from report_ingest import graph
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)
    out = folder / "out"
    if name == "standard":
        engine, kwargs = FakeEngine(tg.full_script()), {}
    else:
        engine = FakeEngine(tg.rules_only_script())
        kwargs = {"label_policy": "rules", "review_mode": "all"}
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GEOTECH_INGEST_")}
    pictures: Dict[bytes, str] = {}
    drawn = Document.render
    signature = inspect.signature(drawn)

    def render(self, index, *args, **kwargs):
        made = drawn(self, index, *args, **kwargs)
        asked = signature.bind(self, index, *args, **kwargs).arguments
        # ``self`` would print its address.
        call = {k: v for k, v in asked.items() if k not in ("self", "budget")}
        pictures[hashlib.sha256(made[0]).digest()] = (
            "<render " + json.dumps(call, sort_keys=True, default=repr) + ">")
        return made

    # The record names the planlens it was read with: a stamp, not a
    # reading.
    with mock.patch.object(graph, "time", _Clock), \
            mock.patch.object(graph, "_planlens_version",
                              lambda: "<PLANLENS>"), \
            mock.patch.object(Document, "render", render), \
            mock.patch.dict(os.environ, env, clear=True):
        graph.ingest_report(pdf, engine, out_dir=out, report_id="SYN",
                            **kwargs)
    files: Dict[str, bytes] = {}
    for root, _dirs, names in os.walk(out):
        for leaf in names:
            path = os.path.join(root, leaf)
            rel = os.path.relpath(path, out).replace("\\", "/")
            files[rel] = (_db_dump(path) if leaf.endswith(".db")
                          else Path(path).read_bytes())
    return engine.calls, files, pictures


def _db_dump(path: str) -> bytes:
    """A library database as its SQL dump: the file's own bytes carry
    free pages and the order SQLite chose to write them in."""
    con = sqlite3.connect(path)
    try:
        return "\n".join(con.iterdump()).encode("utf-8")
    finally:
        con.close()


_TIME = re.compile(
    rb"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)")
_KEY_LINE = re.compile(rb'^id: "([0-9a-f]+)"', re.M)
#: A path under the run's folder once the folder is ``<RUN>``: up to the
#: quote, space or separator that ends it.
_RUN_PATH = re.compile(rb"<RUN>[\\/][^\"'\s,;)\]]*")


def _slashes(found: "re.Match[bytes]") -> bytes:
    """A path's separators as ``/``: written raw (``\\``) or inside JSON
    (``\\\\``) on Windows, ``/`` elsewhere."""
    return found.group(0).replace(b"\\\\", b"/").replace(b"\\", b"/")


def _plain(blob: bytes, folder: Path, keys: List[bytes]) -> bytes:
    """What differs between two runs of one graph, taken out: the run's
    folder, the library keys (made from the PDF's bytes, which carry the
    time it was drawn) and the library's update time; and what differs
    between platforms: the path separators under the run's folder and the
    line endings of a file written as text (CI runs on Linux, the snapshot
    was taken on Windows)."""
    for text in sorted({str(folder), str(folder).replace("\\", "/")},
                       key=len, reverse=True):
        for form in (text, json.dumps(text)[1:-1]):
            blob = blob.replace(form.encode("utf-8"), b"<RUN>")
    for key in keys:
        blob = blob.replace(key, b"<KEY>")
    blob = _RUN_PATH.sub(_slashes, blob).replace(b"\r\n", b"\n")
    return _TIME.sub(b"<TIME>", blob)


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def off_snapshot(root: Path) -> Dict[str, Any]:
    """Each off run's request and file digests, run under ``root``.

    A request is digested with each picture in it replaced by the render
    call that drew it (a picture planlens drew as it is, :func:`_off_run`:
    the page, the dots per inch, the crop), or by ``<PICTURE>`` (a contact
    sheet, or a page made smaller, whose request names its pages in its
    text): how planlens draws a page is not today's path, and differs
    between its versions, but what the graph asks it to draw is."""
    out: Dict[str, Any] = {"planlens": planlens_reading()}
    for name in OFF_RUNS:
        folder = root / name
        calls, files, pictures = _off_run(name, folder)
        keys = sorted({m for blob in files.values()
                       for m in _KEY_LINE.findall(blob)}, key=len,
                      reverse=True)

        def named(value: Any) -> str:
            if isinstance(value, (bytes, bytearray)):
                return pictures.get(hashlib.sha256(value).digest(),
                                    "<PICTURE>")
            return repr(value)

        requests = []
        for call in calls:
            text = json.dumps(call, sort_keys=True, default=named)
            requests.append({
                "tools": call["tools"],
                "system": (call["system"] or "")[:60],
                "digest": _digest(_plain(text.encode("utf-8"), folder,
                                         keys))})
        out[name] = {
            "requests": requests,
            "files": {rel: _digest(_plain(blob, folder, keys))
                      for rel, blob in sorted(files.items())}}
    return out


def planlens_reading() -> str:
    """What planlens reads off the synthetic report -- each page's rules
    label and text -- as one digest. Master's run was taken with one
    reading; with another (another planlens or PyMuPDF), the requests
    differ for that reason alone and the comparison is skipped. How it
    draws a page is not in it: the requests are compared without their
    pictures (:func:`off_snapshot`)."""
    from planlens.document import open_document
    from planlens.document.roles import page_roles
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    doc = open_document(build_narrative_report().pdf, name="SYN")
    try:
        rows = []
        for role in page_roles(doc):
            content = doc.page(role.page, tables=False)
            rows.append([role.page, role.role, round(role.confidence, 4),
                         [line.text for line in content.lines]])
    finally:
        doc.close()
    return _digest(json.dumps(rows, sort_keys=True).encode("utf-8"))


def taken_with() -> Dict[str, str]:
    """The platform and the versions of the libraries the off runs read
    with, as :data:`SNAPSHOT` records them."""
    import platform
    from importlib import metadata

    out = {"python": platform.python_version(),
           "platform": platform.system()}
    for name in ("pydantic", "pydantic-core", "PyMuPDF", "planlens"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    import planlens

    # The version of the code imported: a source checkout on the path can
    # be newer than the release the installed metadata names.
    out["planlens"] = str(getattr(planlens, "__version__", None)
                          or out["planlens"])
    return out


def write_off_snapshot(root: Path) -> Dict[str, Any]:
    """Write :data:`SNAPSHOT` from the graph this checkout has. Run it on
    master's graph only, and only when today's path changes on purpose."""
    blob = {**off_snapshot(root), "taken_with": taken_with()}
    SNAPSHOT.write_text(json.dumps(blob, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    return blob


class TestOffIsToday:

    def test_every_request_and_file_is_masters(self, tmp_path):
        """The switch unset: the scripted runs send master's requests and
        write master's files, byte for byte once the run's folder, the
        library keys and the library's clock are taken out."""
        saved = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        if planlens_reading() != saved["planlens"]:
            pytest.skip("this planlens reads the synthetic report otherwise "
                        "than the one master's snapshot was taken with")
        now = off_snapshot(tmp_path)
        # What to look at first when a digest differs: the snapshot's
        # platform and libraries beside this run's.
        print(f"snapshot taken with {saved.get('taken_with')}; "
              f"this run {taken_with()}")
        for name in OFF_RUNS:
            assert [r["tools"] for r in now[name]["requests"]] == [
                r["tools"] for r in saved[name]["requests"]], name
            for n, (got, want) in enumerate(zip(now[name]["requests"],
                                                saved[name]["requests"])):
                assert got == want, f"{name} request {n}"
            assert sorted(now[name]["files"]) == sorted(
                saved[name]["files"]), name
            for rel, digest in saved[name]["files"].items():
                assert now[name]["files"][rel] == digest, f"{name} {rel}"

    def test_the_off_runs_need_no_model_first_code(self, tmp_path):
        """The same two runs twice in one process, the second with every
        model-first function of the graph made to raise and the model-first
        modules kept from being imported: the same requests and the same
        files. This holds wherever it runs, whatever planlens reads."""
        import contextlib

        from report_ingest import graph

        def refuse(*_args, **_kwargs):
            raise AssertionError("model-first code ran with the switch off")

        first = off_snapshot(tmp_path / "first")
        with contextlib.ExitStack() as stack:
            for name in MODEL_FIRST_FUNCTIONS:
                stack.enter_context(mock.patch.object(graph, name, refuse))
            stack.enter_context(mock.patch.dict(
                sys.modules, {m: None for m in MODEL_FIRST_MODULES}))
            second = off_snapshot(tmp_path / "second")
        assert second == first

    def test_no_model_first_module_is_imported(self, tmp_path):
        """A standard run with the switch unset imports none of the
        model-first modules (checked in a fresh interpreter)."""
        script = (
            "import os, sys\n"
            "from pathlib import Path\n"
            "from report_ingest.tests.test_graph_model_first import _off_run\n"
            f"_off_run('standard', Path({str(tmp_path)!r}))\n"
            f"print([m for m in {list(MODEL_FIRST_MODULES)!r} "
            "if m in sys.modules])\n")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("GEOTECH_INGEST_")}
        root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(
            [root] + [p for p in [env.get("PYTHONPATH")] if p])
        done = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stderr[-2000:]
        assert done.stdout.strip().splitlines()[-1] == "[]"


class TestTodaysReadersOnTheirOwnEngines:
    """Off, ``stage_engines`` puts today's reader of each stage it names on
    that engine, so the two arms are measured on the same models per stage
    (the Foundry brief's D3). Unset, the snapshot above holds."""

    @staticmethod
    def _ingest(tmp_path, engine, stage_engines):
        from report_ingest import graph
        from report_ingest.tests.narrative_fixtures import (
            build_narrative_report)

        tmp_path.mkdir(parents=True, exist_ok=True)
        pdf = tmp_path / "SYN.pdf"
        pdf.write_bytes(build_narrative_report().pdf)
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("GEOTECH_INGEST_")}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.dict(sys.modules,
                                {m: None for m in MODEL_FIRST_MODULES}):
            return graph.ingest_report(
                pdf, engine, out_dir=tmp_path / "out", report_id="SYN",
                model_first="off", stage_engines=stage_engines)

    def test_each_named_stage_reads_on_its_engine(self, tmp_path):
        """The scripted standard run with the readers taken off the run's
        engine: each reader's calls go to its stage's engine, this report's
        and the bound report's alike; the triage, the vision voter and the
        identity call stay on the run's engine; the record is the same and
        counts every call; no model-first module is imported."""
        from report_ingest.tests import test_graph as tg
        from report_ingest.tests.fake_engine import FakeEngine

        main = FakeEngine([tg.triage_turn()] + tg.vision_turns()
                          + [tg.identity_turn()], name="fake-main")
        stages = {
            "narrative": FakeEngine([tg.narrative_turn(),
                                     tg.bound_narrative_turn()],
                                    name="fake-narrative"),
            "logs": FakeEngine([tg.log_turn("B-1"),
                                tg.log_turn("TP-1", "test_pit"),
                                tg.log_turn("BH-1")], name="fake-logs"),
            "lab": FakeEngine([tg.lab_turn("atterberg", 12),
                               tg.lab_turn("gradation", 13)],
                              name="fake-lab"),
            "calc": FakeEngine([tg.calc_turn()], name="fake-calc")}
        record = self._ingest(tmp_path / "split", main, stages)
        for engine in [main, *stages.values()]:
            assert engine.n_calls == len(engine.turns), engine.name
        today = self._ingest(tmp_path / "one", FakeEngine(tg.full_script()),
                             None)
        assert record.document.model_calls == today.document.model_calls
        for part in ("investigations", "lab_tests", "calculations",
                     "general", "narrative", "bound_documents"):
            assert (record.model_dump()[part]
                    == today.model_dump()[part]), part
        (note,) = [q for q in record.qa if q.where == "readers"]
        assert note.values == ["logs: fake-logs", "lab: fake-lab",
                               "calc: fake-calc",
                               "narrative: fake-narrative"]
        assert not [q for q in today.qa if q.where == "readers"]

    def test_a_callable_or_an_unknown_stage_is_refused(self, tmp_path):
        from report_ingest.tests.fake_engine import FakeEngine

        with pytest.raises(ValueError, match="is a mapping"):
            self._ingest(tmp_path, FakeEngine([]), lambda model: None)
        with pytest.raises(ValueError, match="stage_engines names"):
            self._ingest(tmp_path, FakeEngine([]),
                         {"soundings": FakeEngine([])})


# ---------------------------------------------------------------------------
# the switch's settings
# ---------------------------------------------------------------------------

class TestTheSwitch:

    @pytest.mark.parametrize("value, setting", [
        (None, "off"), ("", "off"), ("0", "off"), ("off", "off"),
        ("1", "on"), ("on", "on"), ("ON", "on"), (True, "on"),
        (False, "off")])
    def test_the_settings(self, monkeypatch, value, setting):
        from report_ingest.graph import MODEL_FIRST_ENV, model_first_setting
        monkeypatch.delenv(MODEL_FIRST_ENV, raising=False)
        assert model_first_setting(value) == setting

    def test_unset_follows_the_variable_and_a_typo_is_refused(
            self, monkeypatch):
        from report_ingest.graph import MODEL_FIRST_ENV, model_first_setting
        monkeypatch.setenv(MODEL_FIRST_ENV, "on")
        assert model_first_setting() == "on"
        assert model_first_setting("off") == "off"
        monkeypatch.setenv(MODEL_FIRST_ENV, "model first")
        with pytest.raises(ValueError, match="use off or on"):
            model_first_setting()

    def test_the_rules_alone_are_refused(self, tmp_path):
        """Model first draws its groups from the range labels."""
        from report_ingest.graph import ingest_report
        from report_ingest.tests.fake_engine import FakeEngine
        with pytest.raises(ValueError, match="range labels"):
            ingest_report(b"%PDF", FakeEngine([]), out_dir=tmp_path,
                          model_first="on", label_policy="rules")


# ---------------------------------------------------------------------------
# one made-up report, model first
# ---------------------------------------------------------------------------

#: The synthetic report's sections as the made-up range labeller gives
#: them; pages 15 to 18 are a report bound inside it.
SECTIONS = [(0, 0, "cover"), (1, 1, "toc"), (2, 4, "narrative"),
            (5, 5, "figure"), (6, 6, "other"), (7, 8, "boring_log"),
            (9, 9, "test_pit_log"), (10, 10, "photos"), (11, 11, "other"),
            (12, 13, "lab_test"), (14, 14, "other"),
            (15, 18, "appended_report"), (19, 19, "other"),
            (20, 21, "calculation")]
#: The bound report's own sections, on its own page numbers.
BOUND_SECTIONS = [(0, 0, "cover"), (1, 1, "narrative"), (2, 2, "other"),
                  (3, 3, "boring_log")]
#: What the made-up describer says each page logs and is titled.
EXPLORATIONS = {7: "B-1", 8: "B-1", 9: "TP-1", 18: "BH-1"}
TITLES = {12: "Atterberg limits", 13: "Sieve analysis"}


def _descriptions():
    from report_ingest.range_labels import PageDescription

    return [{"final": PageDescription(
        title=TITLES.get(page, f"Made-up page {page}"), kind="report page",
        names="", identifiers=[], exploration=EXPLORATIONS.get(page, ""),
        continues="", header_footer="", issuer_date="", language="English",
        summary="A made-up page.")} for page in range(22)]


def _sections(rows):
    from report_ingest.range_labels import RangeAnswer, RangeSection

    return {"final": RangeAnswer(
        sections=[RangeSection(first_page=a, last_page=b, label=label,
                               reason="made up") for a, b, label in rows],
        exceptions=[])}


def _log(hole, kind, page, layers):
    from report_ingest.model_first import LogReading

    return {"final": LogReading.model_validate({
        "investigation_id": hole, "kind": kind, "depth_unit": "m",
        "units_known": True, "total_depth": 31.5,
        "layers": [{"top": top, "bottom": bottom, "description": words,
                    "prov": {"page": page}}
                   for top, bottom, words in layers]})}


def _calc():
    from report_ingest.calc_reader import CalcReading

    return {"final": CalcReading.model_validate({
        "kind": "lateral_pile", "program": "LPILE",
        "inputs": [{"name": "Pile-head load", "value": 120.0, "unit": "kN",
                    "prov": {"page": 20}}],
        "results": []})}


def _engines(sections=None):
    """``(main, describer, labeller, stage engines)``: the scripts of one
    model-first run of the synthetic report (``sections``: the report's,
    default :data:`SECTIONS`). The laboratory window gets no structured
    answer."""
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine

    def engine(turns, name):
        made = FakeEngine(turns, name=name)
        made.accepts_pdf = True
        return made

    stages = {
        "narrative": engine([tg.narrative_turn(), tg.bound_narrative_turn()],
                            "fake-narrative"),
        "logs": engine([
            _log("B-1", "boring", 7, [
                (0.0, 3.0, "Brown sandy lean CLAY, stiff, moist"),
                (3.0, 31.5, "Made-up grey GRAVEL printed nowhere")]),
            _log("TP-1", "test_pit", 9, [(0.0, 2.0, "Made-up FILL")]),
            _log("BH-1", "boring", 18, [(0.0, 9.0, "Made-up SILT")])],
            "fake-logs"),
        "lab": engine([{"text": "Made-up words and no answer."}],
                      "fake-lab"),
        "calc": engine([_calc()], "fake-calc"),
    }
    main = FakeEngine([tg.triage_turn(), tg.identity_turn()],
                      name="fake-main")
    describer = FakeEngine(_descriptions(), name="fake-describer")
    labeller = FakeEngine([_sections(sections or SECTIONS),
                           _sections(BOUND_SECTIONS)], name="fake-labeller")
    return main, describer, labeller, stages


def _run_on(pdf, out, engines, **kwargs):
    from report_ingest.graph import ingest_report

    main, describer, labeller, stages = engines
    return ingest_report(pdf, main, out_dir=out, report_id="SYN",
                         model_first="on", describe_engine=describer,
                         range_engine=labeller, stage_engines=stages,
                         **kwargs)


def _record_in(folder: Path):
    from report_ingest.model import ReportRecord

    return ReportRecord.model_validate_json(
        (folder / "report.record.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def synthetic_pdf(tmp_path_factory):
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    path = tmp_path_factory.mktemp("mf") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


@pytest.fixture(scope="module")
def first_run(synthetic_pdf, tmp_path_factory):
    """One model-first run of the synthetic report, and its engines."""
    pytest.importorskip("report_ingest.locate")
    out = tmp_path_factory.mktemp("mf_out")
    engines = _engines()
    record = _run_on(synthetic_pdf, out, engines)
    return record, out, engines


class TestOn:

    def test_each_engine_is_asked_what_its_script_holds(self, first_run):
        """Triage and the bound report's identity on the main engine, every
        page described, the report and the bound report labelled by
        sections, and each stage's groups read on its own engine: no vision
        voter, no label review, nothing else."""
        _record, _out, (main, describer, labeller, stages) = first_run
        assert main.n_calls == 2
        assert describer.n_calls == 22
        assert labeller.n_calls == 2
        assert {stage: e.n_calls for stage, e in stages.items()} == {
            "narrative": 2, "logs": 3, "lab": 1, "calc": 1}
        # A reading call carries its pages as one PDF, then the
        # instruction, and no system prompt.
        call = stages["logs"].calls[0]
        assert call["system"] is None
        assert [b["type"] for b in call["messages"][0]["content"]] == [
            "pdf", "text"]

    def test_the_items_are_written_under_their_model_first_keys(
            self, first_run):
        _record, out, _engines = first_run
        items = sorted(p.name for p in (out / "items").iterdir())
        assert items == ["mf-calc-20.json", "mf-log-7.json", "mf-log-9.json",
                         "mf-narrative.json"]
        bound = sorted(p.name for p in
                       (out / "bound" / "bound1" / "items").iterdir())
        assert bound == ["mf-log-18.json", "mf-narrative.json"]
        assert not (out / "vision.json").exists()
        assert not (out / "review.json").exists()
        labels = json.loads((out / "labels.json").read_text("utf-8"))
        assert labels["model_first"] == "on"
        assert labels["review_mode"] == "none"
        assert labels["vision_mode"] == ""

    def test_the_record_holds_what_the_answers_read(self, first_run):
        record, _out, _engines = first_run
        assert [i.investigation_id for i in record.investigations] == [
            "B-1", "TP-1"]
        assert [c.program for c in record.calculations] == ["LPILE"]
        assert record.lab_tests == []
        assert record.general.projectName == "Rosewood Terrace Development"
        assert [b.bound_id for b in record.bound_documents] == ["bound1"]
        (note,) = [q for q in record.qa if q.where == "model_first"]
        assert "logs: fake-logs, pages as pdf" in note.values

    def test_the_bound_report_is_read_in_its_own_window(self, first_run):
        _record, out, _engines = first_run
        child = _record_in(out / "bound" / "bound1")
        assert [i.investigation_id for i in child.investigations] == ["BH-1"]
        assert child.general.projectName == \
            "Rosewood Terrace Preliminary Study"
        narrative = json.loads((out / "items" / "mf-narrative.json")
                               .read_text("utf-8"))
        assert not set(narrative["pages"]) & {15, 16, 17, 18}
        child_narrative = json.loads(
            (out / "bound" / "bound1" / "items" / "mf-narrative.json")
            .read_text("utf-8"))
        assert child_narrative["pages"] == [15, 16, 17, 18]

    def test_a_value_not_printed_on_its_page_is_a_qa_note(self, first_run):
        record, _out, _engines = first_run
        notes = [q for q in record.qa
                 if q.where == "items.boring_log.mf-log-7.placement"]
        assert len(notes) == 1
        assert "not found printed on the cited page" in notes[0].detail

    def test_a_group_with_no_answer_lists_all_its_pages(self, first_run):
        """The laboratory window got no structured answer: nothing of it is
        in the record, it is not saved, and its skipped entry names both
        its pages."""
        record, out, _engines = first_run
        assert not (out / "items" / "mf-lab-12.json").exists()
        skipped = [q for q in record.qa if q.kind == "skipped"
                   and q.where == "lab pages 12-13"]
        assert [q.pages for q in skipped] == [[12, 13]]

    def test_a_resumed_run_asks_again_only_for_what_was_not_read(
            self, synthetic_pdf, first_run, tmp_path):
        """Over the same folder, every result saved is used again and the
        laboratory window, which was not saved, is asked for again."""
        import shutil

        out = first_run[1]
        again = tmp_path / "again"
        shutil.copytree(out, again)
        main, describer, labeller, stages = _engines()
        for engine in (main, describer, labeller, *stages.values()):
            engine.turns = []
        stages["lab"].turns = [{"text": "Still no answer."}]
        record = _run_on(synthetic_pdf, again,
                         (main, describer, labeller, stages))
        assert stages["lab"].n_calls == 1
        assert sum(e.n_calls for e in (main, describer, labeller,
                                       *stages.values())) == 1
        assert [i.investigation_id for i in record.investigations] == [
            "B-1", "TP-1"]


def _hole(hole, page, top, bottom, remarks="", unit="m", kind="boring"):
    """A made-up log of one layer."""
    from report_ingest.model import Investigation, Layer, Quantity

    return Investigation(
        investigation_id=hole, kind=kind, depth_unit=unit, pages=[page],
        remarks=remarks,
        layers=[Layer(top=Quantity(value=top, unit=unit),
                      bottom=Quantity(value=bottom, unit=unit),
                      description="Made-up SAND")])


def _joined(investigations, start=0, slack=None):
    from report_ingest.graph import _join_by_hole
    from report_ingest.model import ReportRecord

    record = ReportRecord()
    record.investigations = list(investigations)
    unresolved, qa = [], []
    _join_by_hole(record, start, unresolved, qa, slack)
    return record.investigations, unresolved, qa


class TestOneHoleInTwoPlaces:

    def test_a_log_that_continues_another_is_joined_to_it(self):
        """The later log starts where the first stops: one log, the
        first's header, the later one's remarks appended and its other
        spelling listed as read, and one note. A log read before ``start``
        is left alone."""
        logs, unresolved, qa = _joined(
            [_hole("B-1", 2, 0.0, 5.0),
             _hole("B-1", 7, 0.0, 10.0, "Made-up one"),
             _hole("TP-1", 9, 0.0, 2.0),
             _hole("B 1", 20, 10.0, 20.0, "Made-up two")], start=1)
        assert [(i.investigation_id, i.pages) for i in logs] == [
            ("B-1", [2]), ("B-1", [7, 20]), ("TP-1", [9])]
        assert logs[1].remarks == "Made-up one; Made-up two"
        assert [r["what"] for r in unresolved] == [
            "log B-1: investigation_id"]
        (note,) = qa
        assert note.kind == "note" and note.pages == [7, 20]
        assert "logged in two places (page 7; page 20)" in note.detail

    def test_a_log_printed_twice_is_kept_apart_with_a_note(self):
        """The later log starts again at the top: both are kept, and one
        note names the possible duplicate. Depths in another unit are not
        a continuation either."""
        logs, unresolved, qa = _joined(
            [_hole("B-1", 7, 0.0, 10.0), _hole("B-1", 20, 0.0, 10.0),
             _hole("B-2", 30, 0.0, 10.0),
             _hole("B-2", 31, 10.0, 20.0, unit="ft")])
        assert [(i.investigation_id, i.pages) for i in logs] == [
            ("B-1", [7]), ("B-1", [20]), ("B-2", [30]), ("B-2", [31])]
        assert unresolved == []
        assert [q.pages for q in qa] == [[7, 20], [30, 31]]
        assert "does not continue it (it starts at the surface, 0)" \
            in qa[0].detail
        assert "its depths are in ft, the first's in m" in qa[1].detail

    def test_the_rulers_slack_is_allowed(self):
        """A later log whose first depth sits above the first's deepest by
        no more than the logs' ruler slack still continues it."""
        asked = []

        def slack(pages):
            asked.append(list(pages))
            return 0.5

        logs, _unresolved, qa = _joined(
            [_hole("B-1", 7, 0.0, 10.0), _hole("B-1", 20, 9.6, 20.0)],
            slack=slack)
        assert [i.pages for i in logs] == [[7, 20]]
        assert asked == [[7, 20]]
        assert "logged in two places" in qa[0].detail

    def test_identifiers_that_differ_in_leading_zeros_are_never_joined(
            self):
        logs, _unresolved, qa = _joined(
            [_hole("B-01", 7, 0.0, 10.0), _hole("B-1", 20, 10.0, 20.0)])
        assert [i.investigation_id for i in logs] == ["B-01", "B-1"]
        (note,) = qa
        assert note.pages == [7, 20]
        assert "B-01 and B-1 have identifiers that differ only in leading " \
            "zeros" in note.detail

    def test_a_log_that_starts_at_the_surface_continues_none(self):
        """HA-1, 0 to 2 ft, printed on two pages whose ruler is marked every
        5 ft: the ruler's slack reaches back past the surface, but a log
        that starts there is the hole logged again, not its continuation."""
        from types import SimpleNamespace

        from report_ingest.log_reader import _depth_slack

        ruler = SimpleNamespace(step=5.0)
        grid = SimpleNamespace(rulers={7: ruler, 20: ruler})
        assert _depth_slack(grid, [7, 20]) >= 2.0
        logs, unresolved, qa = _joined(
            [_hole("HA-1", 7, 0.0, 2.0, unit="ft", kind="hand_auger"),
             _hole("HA-1", 20, 0.0, 2.0, unit="ft", kind="hand_auger")],
            slack=lambda pages: _depth_slack(grid, pages))
        assert [i.pages for i in logs] == [[7], [20]]
        assert unresolved == []
        (note,) = qa
        assert "does not continue it (it starts at the surface, 0)" \
            in note.detail

    def test_rulers_planlens_cannot_read_give_no_slack_and_a_note(
            self, monkeypatch):
        """A planlens error reading the depth rulers does not end the run:
        the slack is 0 and one note says why."""
        from report_ingest import visual_scales
        from report_ingest.graph import _ruler_slack

        def broken(doc, pages, *args, **kwargs):
            raise RuntimeError("made-up grid failure")

        monkeypatch.setattr(visual_scales, "log_grid_for", broken)
        qa = []
        slack = _ruler_slack("made-up document", qa)
        logs, _unresolved, join_qa = _joined(
            [_hole("B-1", 7, 0.0, 10.0), _hole("B-1", 20, 9.6, 20.0)],
            slack=slack)
        assert [i.pages for i in logs] == [[7], [20]]
        assert "its depths start at 9.6, above the first's deepest, 10" \
            in join_qa[0].detail
        (note,) = qa
        assert note.pages == [7, 20]
        assert "the depth rulers on pages 7, 20 could not be read " \
            "(RuntimeError: made-up grid failure)" in note.detail


class TestTheRunsSettings:

    def _settings(self, stage_engines, page_route=None, questions=None,
                  source=b"%PDF"):
        from report_ingest.graph import _Labelling, _model_first_run, _Tally
        from report_ingest.tests.fake_engine import FakeEngine

        qa = []
        labelling = _Labelling()
        made = _model_first_run(source, FakeEngine([], name="fake-main"),
                                _Tally(), labelling, stage_engines,
                                page_route, questions, qa)
        return made, labelling, qa

    def test_a_callable_is_asked_for_each_stages_model(self):
        """``stage_engines`` as a callable is given each stage's model
        (``STAGE_MODELS``); a stage's route follows its engine."""
        from report_ingest.model_first_read import STAGE_MODELS
        from report_ingest.tests.fake_engine import FakeEngine

        asked = []

        def build(model):
            asked.append(model)
            made = FakeEngine([], name=model)
            made.accepts_pdf = model == STAGE_MODELS["lab"]
            return made

        made, labelling, qa = self._settings(build, page_route="auto")
        assert asked == list(STAGE_MODELS.values())
        assert {s: e.name for s, e in made.engines.items()} == STAGE_MODELS
        assert made.routes["lab"] == "pdf"
        assert made.routes["logs"] == "pictures"
        assert made.pdf == b"%PDF"
        assert labelling.model_first and labelling.review_mode == "none"
        assert labelling.range_arm == "plain"
        assert [q.where for q in qa] == ["model_first"]

    def test_a_stage_not_given_reads_on_the_main_engine(self):
        from report_ingest.tests.fake_engine import FakeEngine

        made, _labelling, _qa = self._settings(
            {"lab": FakeEngine([], name="fake-lab")})
        assert made.engines["lab"].name == "fake-lab"
        assert made.engines["logs"].name == "fake-main"

    def test_a_pdf_route_an_engine_cannot_send_is_refused(self):
        with pytest.raises(ValueError, match="cannot send a PDF"):
            self._settings(None, page_route="pdf")

    def test_the_callers_own_questions_are_said_not_asked(self):
        _made, _labelling, qa = self._settings(
            None, questions=["Made-up question?"])
        assert "the caller's own questions were not asked" in qa[0].detail


def test_an_appendix_file_and_the_item_ceiling_are_kept_as_today(
        synthetic_pdf, tmp_path):
    """On a file triage calls appendix material the narrative is not read
    and its prose pages are listed; the item ceiling keeps the first groups
    in page order, and the pages of those cut are listed as read by
    nothing. The bound report keeps its own workflow."""
    pytest.importorskip("report_ingest.locate")
    from report_ingest.graph import Budgets
    from report_ingest.tests import test_graph as tg

    main, describer, labeller, stages = _engines()
    main.turns = [tg.triage_turn("appendix_only"), tg.identity_turn()]
    stages["narrative"].turns = [tg.bound_narrative_turn()]
    record = _run_on(synthetic_pdf, tmp_path / "out",
                     (main, describer, labeller, stages),
                     budgets=Budgets(max_items=3))
    assert {stage: e.n_calls for stage, e in stages.items()} == {
        "narrative": 1, "logs": 3, "lab": 0, "calc": 0}
    skipped = {q.where: q.pages for q in record.qa if q.kind == "skipped"}
    assert skipped == {"items.narrative": [2, 3, 4],
                       "pages.lab_test": [12, 13],
                       "pages.calculation": [20, 21]}
    assert sorted(p.name for p in (tmp_path / "out" / "items").iterdir()) \
        == ["mf-log-7.json", "mf-log-9.json"]


def test_placement_takes_the_built_pages_as_they_stand(monkeypatch):
    """The builder has already read the answer's cites as pages or as
    positions, so locate is told to turn none of them: a built log on the
    call's pages 1 to 3 citing page 1 is not read as position 1 (page 2)
    again, and the narrative's citations go to locate's own check."""
    locate = pytest.importorskip("report_ingest.locate")
    from report_ingest.graph import _placer

    asked = []

    class Found:
        items = []

        def to_dict(self):
            return {"placed": 1}

    def recorder(name):
        def place(words, record, pages, **kwargs):
            asked.append((name, words, list(pages), kwargs))
            return Found()
        return place

    monkeypatch.setattr(locate, "PageWords", lambda doc: f"words of {doc}")
    for name in ("place_investigation", "place_lab_tests",
                 "place_calculation", "place_citations"):
        monkeypatch.setattr(locate, name, recorder(name))
    place = _placer("doc")
    assert place("logs", "log", [1, 2, 3], "doc", rulers={1: "ruler"}) == {
        "placed": 1}
    place("lab", ["test"], [4, 5], "doc")
    place("calc", "calc", [6], "doc")
    place("narrative", {"projectName": []}, [0, 1], "doc")
    assert [(name, words, pages) for name, words, pages, _ in asked] == [
        ("place_investigation", "words of doc", [1, 2, 3]),
        ("place_lab_tests", "words of doc", [4, 5]),
        ("place_calculation", "words of doc", [6]),
        ("place_citations", "words of doc", [0, 1])]
    assert asked[0][3]["rulers"] == {1: "ruler"}
    for _name, _words, pages, kwargs in asked[:3]:
        numbers = kwargs["numbers"]
        assert not numbers.positions
        assert [numbers.page(p)[0] for p in pages] == pages
    assert asked[3][3] == {}


# ---------------------------------------------------------------------------
# the review's cases (piece 8)
# ---------------------------------------------------------------------------

def _silent(engines):
    """The engines with nothing scripted: a call to any of them fails."""
    main, describer, labeller, stages = engines
    for engine in (main, describer, labeller, *stages.values()):
        engine.turns = []
    return engines


def _n_calls(engines):
    main, describer, labeller, stages = engines
    return {"main": main.n_calls, "describer": describer.n_calls,
            "labeller": labeller.n_calls,
            **{stage: e.n_calls for stage, e in stages.items()}}


class TestSavedResults:

    @pytest.mark.parametrize("change, current", [
        ({}, True), ({"pages": [7, 8, 9]}, False),
        ({"prompt_version": "mf-0"}, False), ({"route": "pictures"}, False)])
    def test_a_saved_result_is_used_only_while_current(self, change,
                                                       current):
        from report_ingest.graph import _still_current
        from report_ingest.model_first import READ_PROMPT_VERSION
        from report_ingest.model_first_pages import PageGroup

        blob = {"pages": [7, 8], "prompt_version": READ_PROMPT_VERSION,
                "route": "pdf", **change}
        parts = [PageGroup("logs", [7]), PageGroup("logs", [8])]
        assert _still_current(blob, parts, "pdf") is current

    def test_a_log_regrouped_since_is_read_again_whole(
            self, synthetic_pdf, first_run, tmp_path, monkeypatch):
        """Run 1 read B-1 as pages 7-8. Resumed after a regroup that puts
        page 9 in it, the saved 7-8 result is not used: 7-9 is read."""
        import dataclasses
        import shutil

        from report_ingest import model_first_pages

        drawn = model_first_pages.group_pages

        def regrouped(*args, **kwargs):
            groups = drawn(*args, **kwargs)
            if any(g.key == "log-7" for g in groups["logs"]):
                groups["logs"] = [
                    dataclasses.replace(g, pages=[7, 8, 9])
                    if g.key == "log-7" else g
                    for g in groups["logs"] if g.key != "log-9"]
            return groups

        monkeypatch.setattr(model_first_pages, "group_pages", regrouped)
        again = tmp_path / "again"
        shutil.copytree(first_run[1], again)
        engines = _silent(_engines())
        engines[3]["logs"].turns = [_log("B-1", "boring", 7, [
            (0.0, 3.0, "Brown sandy lean CLAY, stiff, moist")])]
        engines[3]["lab"].turns = [{"text": "Still no answer."}]
        record = _run_on(synthetic_pdf, again, engines)
        assert {k: v for k, v in _n_calls(engines).items() if v} == {
            "logs": 1, "lab": 1}
        saved = json.loads((again / "items" / "mf-log-7.json")
                           .read_text("utf-8"))
        assert saved["pages"] == [7, 8, 9]
        assert [i.investigation_id for i in record.investigations] == ["B-1"]

    def test_a_group_with_a_half_unanswered_is_not_saved(
            self, synthetic_pdf, tmp_path):
        """The laboratory window's reply is cut off; asked as two halves,
        page 12 is answered and page 13 is not. Page 12's test is in the
        record and page 13 is listed, but the result is not saved, so a
        resumed run asks for the whole window again."""
        pytest.importorskip("report_ingest.locate")
        from report_ingest.model_first import LabSheetReading

        def sheet(page):
            return {"final": LabSheetReading.model_validate({
                "depth_unit": "m",
                "tests": [{"kind": "atterberg", "investigation_id": "B-1",
                           "sample_id": "S-1", "depth_top": 1.5,
                           "ll": 41.0, "pl": 19.0, "pi": 22.0,
                           "prov": {"page": page}}]})}

        engines = _engines()
        engines[3]["lab"].turns = [
            {"text": "", "stop_reason": "max_tokens"}, sheet(12),
            {"text": "Made-up words and no answer."}]
        record = _run_on(synthetic_pdf, tmp_path / "out", engines)
        assert engines[3]["lab"].n_calls == 3
        assert [t.kind for t in record.lab_tests] == ["atterberg"]
        assert [q.pages for q in record.qa if q.kind == "skipped"
                and q.where == "lab page 13"] == [[13]]
        assert not (tmp_path / "out" / "items" / "mf-lab-12.json").exists()

        engines = _silent(_engines())
        engines[3]["lab"].turns = [sheet(12)]
        _run_on(synthetic_pdf, tmp_path / "out", engines)
        assert {k: v for k, v in _n_calls(engines).items() if v} == {
            "lab": 1}
        assert (tmp_path / "out" / "items" / "mf-lab-12.json").exists()


def test_a_bound_range_the_labels_call_narrative_is_not_listed_as_skipped(
        synthetic_pdf, tmp_path):
    """Triage alone says pages 15 to 18 are a report of their own, and the
    labeller calls them narrative: they are handed to the bound report's
    own pass, whose narrative is shown all of them, so none is listed as
    read by nothing."""
    pytest.importorskip("report_ingest.locate")
    from report_ingest.tests import test_graph as tg

    sections = [row for row in SECTIONS if row[2] != "appended_report"] + [
        (15, 18, "narrative")]
    engines = _engines(sorted(sections))
    engines[0].turns[0] = tg.triage_turn(
        "multi_document",
        bound_together=[{"kind": "appended_prior_report", "pages": "15-18",
                         "title": "Made-up Prior Study"}])
    record = _run_on(synthetic_pdf, tmp_path / "out", engines)
    assert [b.bound_id for b in record.bound_documents] == ["bound1"]
    assert [q for q in record.qa if q.where == "pages.narrative"] == []


def test_the_item_ceiling_counts_the_soundings_with_the_rest(monkeypatch):
    """One ceiling over the model-first units and today's soundings
    together, in page order; a work item of a model-first stage is
    replaced by the units, and an item no reader reads is listed and not
    counted."""
    from types import SimpleNamespace

    from report_ingest import graph

    units = [(graph._Unit("narrative", "narrative", [0, 1]), []),
             (graph._Unit("boring_log", "log-7", [7, 8]), []),
             (graph._Unit("lab_test", "lab-12", [12, 13]), [])]
    monkeypatch.setattr(graph, "_model_first_units",
                        lambda *args, **kwargs: list(units))
    cone = SimpleNamespace(kind="cpt_log", pages=[5])
    probe = SimpleNamespace(kind="dcp_log", pages=[20])
    today = SimpleNamespace(kind="boring_log", pages=[7, 8])
    figure = SimpleNamespace(kind="figure", pages=[2])
    mine, every, given, rest = graph._model_first_plan(
        None, None, {}, None, [figure, cone, today, probe], 3)
    assert [unit.id for unit, _parts in mine] == ["narrative", "log-7"]
    assert rest == [cone]
    assert given == [units[0][0], figure, cone, units[1][0]]
    assert every == [units[0][0], figure, cone, units[1][0], units[2][0],
                     probe]


def test_a_lab_test_placed_on_another_page_takes_that_page(monkeypatch):
    locate = pytest.importorskip("report_ingest.locate")
    from types import SimpleNamespace

    from report_ingest.graph import _placer

    found = locate.Placements(items=[
        locate.Placement("test 1", "test", 12, 13, locate.MOVED),
        locate.Placement("test 2", "test", 12, 12, locate.PLACED)])
    monkeypatch.setattr(locate, "PageWords", lambda doc: doc)
    monkeypatch.setattr(locate, "place_lab_tests",
                        lambda *args, **kwargs: found)
    tests = [SimpleNamespace(pages=[12]), SimpleNamespace(pages=[12])]
    _placer("doc")("lab", tests, [12, 13], "doc")
    assert [t.pages for t in tests] == [[13], [12]]
