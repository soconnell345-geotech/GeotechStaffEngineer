"""read_text_file: a REAL text file from disk (field feedback 2026-09-15, N1).

The calc sub-agent had no tool that could open the HTML report it wrote a turn
earlier, so a "rebuild with the figures embedded" came back with every number
replaced by "per prior analysis".
"""

import json

from funhouse_agent.vision_tools import dispatch_extended_tool


def _call(args):
    return json.loads(dispatch_extended_tool("read_text_file", args,
                                             engine=None, attachments={}))


def test_reads_text_in_pages(tmp_path):
    f = tmp_path / "SOE_sensitivity_review.html"
    body = "<tr><td>F</td><td>26</td><td>14.4</td><td>43.522</td></tr>" * 300
    f.write_text(body, encoding="utf-8")
    first = _call({"path": str(f)})
    assert first["text"] == body[:6000]
    assert first["truncated"] is True and first["next_offset"] == 6000
    assert first["chars_total"] == len(body)
    second = _call({"path": str(f), "offset": first["next_offset"]})
    assert second["text"] == body[6000:12000]


def test_output_stays_whole_json_under_the_reference_cap(tmp_path):
    f = tmp_path / "quotes.html"
    f.write_text('<img src="a.png" alt="x">\n' * 2000, encoding="utf-8")
    raw = dispatch_extended_tool("read_text_file", {"path": str(f)},
                                 engine=None, attachments={})
    assert len(raw) < 16000
    json.loads(raw)


def test_bare_name_is_found_in_the_working_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "soe_sensitivity_summary.txt").write_text("Case F: phi=26, q=14.4")
    assert "Case F" in _call({"path": "soe_sensitivity_summary.txt"})["text"]


def test_binary_and_missing_files_say_what_to_use(tmp_path):
    pdf = tmp_path / "package.pdf"
    pdf.write_bytes(b"%PDF-1.7\x00\x01\x02")
    assert "read_pdf_text" in _call({"path": str(pdf)})["error"]
    assert "list_files" in _call({"path": str(tmp_path / "nope.txt")})["error"]
    assert "required" in _call({})["error"]


def test_deep_tool_is_built_by_default():
    from funhouse_agent.deep.tools import make_vision_tools
    assert "read_text_file" in [t.name for t in make_vision_tools()]
