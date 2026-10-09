"""Live smoke wave 1 fixes in the dispatcher and the file helpers.

* A6  -- results name files by their conversation-relative name; absolute
  server paths stay internal (``html_to_pdf`` still finds a named figure).
* A8  -- ``attachment_key`` resolves on the deep agent's ``call_agent``.
* A14 -- a rescue copy lands in the conversation folder under a short name,
  and the model is never told to report a server path.
* A15b -- a call whose parameters do not fit gets the method's parameters,
  not a raw TypeError.
* A15f -- ``write_diggs`` names its own file when none is given.
"""

import json
import os

import pytest

from funhouse_agent import _fileio, dispatch

SAMPLE_DIGGS = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "eval_samples", "sample_site_diggs.xml")


@pytest.fixture
def work(tmp_path, monkeypatch):
    """A bound working folder, deep enough to look like a conversation's."""
    monkeypatch.delenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, raising=False)
    folder = tmp_path / "conversations" / ("a" * 32) / "files"
    folder.mkdir(parents=True)
    with _fileio.working_dir_bound(str(folder)):
        yield folder


# ---------------------------------------------------------------------------
# A6: conversation-relative names
# ---------------------------------------------------------------------------

def test_conversation_name_and_hide_working_folder(work):
    inside = str(work / "figs" / "a.png")
    assert _fileio.conversation_name(inside) == "figs/a.png"
    assert _fileio.conversation_name(str(work)) == "."
    assert _fileio.conversation_name("/elsewhere/x.png") == "/elsewhere/x.png"
    assert _fileio.conversation_name("plain.png") == "plain.png"
    folder = str(work)
    result = {"output_path": inside, "output_dir": folder,
              "html_img_tag": f'<img src="{inside}" alt="t">',
              "figures": [{"output_path": os.path.join(folder, "b.png")}],
              "note": f"Written into '{folder}', which is attached.",
              "other": "/tmp/unrelated.png", "n": 3}
    hidden = _fileio.hide_working_folder(result)
    assert hidden["output_path"] == "figs/a.png"
    assert hidden["output_dir"] == "."
    assert hidden["html_img_tag"] == '<img src="figs/a.png" alt="t">'
    assert hidden["figures"][0]["output_path"] == "b.png"
    assert "the working folder" in hidden["note"] and folder not in \
        json.dumps(hidden)
    assert hidden["other"] == "/tmp/unrelated.png" and hidden["n"] == 3
    assert result["output_path"] == inside          # input untouched
    # a sibling folder that merely starts with the name is not inside it
    sib = folder + "_old" + os.sep + "c.png"
    assert _fileio.hide_working_folder({"p": sib})["p"] == sib


def test_without_a_host_folder_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.delenv(_fileio.DEFAULT_OUTPUT_DIR_ENV, raising=False)
    p = str(tmp_path / "x.png")
    assert _fileio.hide_working_folder({"output_path": p}) == {"output_path": p}
    assert _fileio.conversation_name(p) == p


def test_a_plot_names_its_files_and_html_to_pdf_embeds_by_name(work):
    res = dispatch.call_agent("profile_figure", "plot_data", {
        "series": [{"x": [0, 1, 2], "y": [0, 1, 4], "label": "q"}],
        "title": "Test", "output_path": "curve.png"})
    assert "error" not in res, res
    assert res["output_path"] == "curve.png"
    assert res["html_img_tag"].startswith('<img src="curve.png"')
    assert str(work) not in json.dumps(res)
    assert (work / "curve.png").is_file()
    pdf = dispatch.call_agent("calc_package", "html_to_pdf", {
        "html": f"<h1>Report</h1>{res['html_img_tag']}",
        "output_path": "report.pdf"})
    assert pdf.get("status") == "success", pdf
    assert pdf["output_path"] == "report.pdf"
    assert pdf["images_embedded"] == 1
    assert pdf["embedded_images"] == ["curve.png"]


def test_a_bare_input_name_is_found_in_the_working_folder(work):
    (work / "page.html").write_text("<p>hello</p>", encoding="utf-8")
    res = dispatch.call_agent("calc_package", "html_to_pdf",
                              {"html_path": "page.html",
                               "output_path": "page.pdf"})
    assert res.get("status") == "success", res
    assert (work / "page.pdf").is_file()


# ---------------------------------------------------------------------------
# A8: attachment_key on the deep agent's call_agent
# ---------------------------------------------------------------------------

def _deep_call_agent(attachments):
    from funhouse_agent.deep.tools import make_core_tools
    tools = {t.name: t for t in make_core_tools(attachments=attachments)}
    return tools["call_agent"]


@pytest.mark.skipif(not os.path.isfile(SAMPLE_DIGGS), reason="no sample")
def test_parse_diggs_by_attachment_key_on_the_deep_agent(work):
    with open(SAMPLE_DIGGS, "rb") as fh:
        data = fh.read()
    tool = _deep_call_agent({"sample_site_diggs.xml": data})
    out = json.loads(tool.invoke({
        "agent_name": "subsurface", "method": "parse_diggs",
        "parameters": {"attachment_key": "sample_site_diggs.xml"}}))
    assert "error" not in out, out
    assert out.get("site_key") and out.get("n_investigations")


@pytest.mark.skipif(not os.path.isfile(SAMPLE_DIGGS), reason="no sample")
def test_an_attachment_key_naming_a_working_folder_file_resolves(work):
    with open(SAMPLE_DIGGS, "rb") as fh:
        (work / "site.xml").write_bytes(fh.read())
    tool = _deep_call_agent({})                  # nothing in the dict
    out = json.loads(tool.invoke({
        "agent_name": "subsurface", "method": "parse_diggs",
        "parameters": {"attachment_key": "site.xml"}}))
    assert "error" not in out, out


def test_an_unknown_attachment_key_says_what_is_attached(work):
    out = dispatch.call_agent("subsurface", "parse_diggs",
                              {"attachment_key": "nope.xml"},
                              attachments={"a.pdf": b"x"})
    assert "nope.xml" in out["error"] and "a.pdf" in out["error"]
    assert "provide 'file_path'" not in out["error"]


def test_a_file_reading_method_gets_the_file_not_its_text(work):
    (work / "log.ags").write_text("x", encoding="utf-8")
    params = dispatch._resolve_attachment(
        {"attachment_key": "log.ags"}, {"log.ags": b"x"},
        {"file_path": {"required": True}})
    assert params == {"file_path": str(work / "log.ags")}
    params = dispatch._resolve_attachment(
        {"attachment_key": "log.ags"}, {"log.ags": b"x"},
        {"file_path": {}, "content": {}})
    assert params == {"content": "x"}


# ---------------------------------------------------------------------------
# A15b: parameters that do not fit
# ---------------------------------------------------------------------------

def test_a_missing_parameter_names_the_methods_parameters():
    out = dispatch.call_agent("gec6", "table_5_1_bearing_capacity_factors", {})
    assert "phi" in out["error"]
    assert "phi" in out.get("required_parameters", [])
    assert "TypeError" not in out["error"]
    assert "describe_method" in out["directive"]


def test_an_unknown_keyword_is_named_without_internals():
    out = dispatch.call_agent("reference_db", "reference_get",
                              {"id": "dm7_1-4-2"})
    text = json.dumps(out)
    assert "'id' is not a parameter of reference_get" in out["error"]
    assert "<locals>" not in text and "_build" not in text
    assert "reference" in out["required_parameters"]
    assert "section_id" in out["required_parameters"]


# ---------------------------------------------------------------------------
# A15f: write_diggs names its own file
# ---------------------------------------------------------------------------

def test_write_diggs_without_output_path_names_the_file(work):
    from funhouse_agent.adapters.subsurface_adapter import _diggs_default_name

    class _I:
        def __init__(self, i):
            self.investigation_id = i

    assert _diggs_default_name([_I("B-1")]) == "B-1.diggs.xml"
    assert _diggs_default_name([_I("B-1"), _I("B-2")]) == "B-1_B-2.diggs.xml"
    assert _diggs_default_name([_I(f"B-{i}") for i in range(1, 13)]) == \
        "B-1_to_B-12.diggs.xml"
    assert _diggs_default_name([]) == "site.diggs.xml"
    out = dispatch.call_agent("subsurface", "write_diggs", {
        "investigations": [{"investigation_id": "B-1", "kind": "boring"}]})
    assert "error" not in out, out
    assert out["output_path"] == "B-1.diggs.xml"
    assert (work / "B-1.diggs.xml").is_file()


# ---------------------------------------------------------------------------
# A14: rescues
# ---------------------------------------------------------------------------

def test_a_rescue_lands_in_the_working_folder_under_a_short_name(work):
    long_name = "x" * 300 + ".png"
    rescue = _fileio.rescue_write(str(work / "deep" / long_name), b"PNG")
    assert rescue is not None
    assert os.path.dirname(rescue) == str(work)
    assert len(rescue) <= _fileio.MAX_RESCUE_PATH
    note = _fileio.rescue_note(rescue)
    assert str(work) not in note
    assert os.path.basename(rescue) in note
    assert "report THAT path" not in note


def test_a_failed_save_names_the_rescue_and_never_asks_for_a_path(
        work, monkeypatch):
    def broken(path, content):
        raise OSError("path too long")
    monkeypatch.setattr(_fileio, "_local_write", broken)
    out = _fileio.save_verified(str(work / "fig.png"), b"PNGDATA")
    assert out.get("rescue_path")
    assert os.path.dirname(out["rescue_path"]) == str(work)
    assert "report THAT path" not in out["error"]
    assert "in the working folder as 'fig_1.png'" in out["error"]


def test_generated_figure_names_are_short(tmp_path):
    from funhouse_agent.adapters.calc_package import (_MAX_FIGURE_PATH,
                                                      _figure_file_name)
    out_dir = str(tmp_path / ("d" * 40))
    name = _figure_file_name(out_dir, "slope_stability", 1,
                             "Slope cross section with the critical surface "
                             "and every trial surface", "221735")
    assert name.startswith("slope_stability_f1_") and name.endswith(
        "_221735.png")
    assert len(os.path.join(out_dir, name)) <= _MAX_FIGURE_PATH
    deep = str(tmp_path / ("d" * 205))
    assert len(os.path.join(deep, _figure_file_name(
        deep, "slope_stability", 2, "t", "1"))) <= max(
        _MAX_FIGURE_PATH, len(deep) + 12)
