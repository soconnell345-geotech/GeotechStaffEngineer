"""Deliverables written outside the conversation folder, and files only read.

Field feedback 2026-09-15 (Nairobi SOE re-run): the calc package, figures and
plot were written to /tmp, so no download card, no inline image and no
SharePoint copy (N4/N6); the SharePoint download the agent only READ was shown
as if produced (N8).
"""

import json
import os

from langchain_core.messages import ToolMessage

from webapp import core
from webapp.output_capture import OutputCollector, paths_in


def test_paths_in_json_repr_and_sharepoint_results():
    js = json.dumps({"status": "success", "output_path": "/tmp/SOE review.pdf",
                     "figures": [{"output_path": "C:\\tmp\\fig 1.png"}]})
    outs, ins = paths_in(js)
    assert outs == ["/tmp/SOE review.pdf", "C:\\tmp\\fig 1.png"]
    assert ins == []
    outs, _ = paths_in("{'saved': '/tmp/pywall.png', 'file_exists': True}")
    assert outs == ["/tmp/pywall.png"]
    dl = ("Downloaded Shared Documents/General/GSE_app/uploaded references/a b.pdf "
          "-> /root/.geotech_webapp/conversations/t/files/a b.pdf (23,006,108 "
          "bytes). The file is now in the working folder.")
    outs, ins = paths_in(dl)
    assert outs == [] and ins == ["/root/.geotech_webapp/conversations/t/files/a b.pdf"]
    up = "Uploaded /tmp/SOE.pdf -> Shared Documents/General/GSE_app/x/SOE.pdf."
    assert paths_in(up)[0] == ["/tmp/SOE.pdf"]


def test_collector_reads_tool_messages_and_dedupes():
    c = OutputCollector()
    msg = ToolMessage(content=json.dumps({"saved": "/tmp/a.html"}),
                      tool_call_id="1")
    c.on_tool_end(msg)
    c.on_tool_end(json.dumps({"output_path": "/tmp/a.html"}))
    c.on_tool_end(object())                       # never raises
    assert c.outputs == ["/tmp/a.html"]


def test_import_copies_outside_files_in_and_dedupes(tmp_path):
    conv = tmp_path / "conversations" / "t1"
    files = conv / "files"
    files.mkdir(parents=True)
    outside = tmp_path / "tmp"
    outside.mkdir()
    pdf = outside / "SOE_sensitivity_review_embedded.pdf"
    pdf.write_bytes(b"%PDF-1.4 numbers")
    inside = files / "already_here.png"
    inside.write_bytes(b"png")
    staged = outside / "upload.pdf"
    staged.write_bytes(b"%PDF upload")

    copied = core.import_reported_outputs(
        [str(pdf), str(pdf), str(inside), str(outside / "gone.png"), str(staged)],
        str(files), exclude=[str(staged)])
    assert list(copied) == [os.path.abspath(pdf)]
    dst = copied[os.path.abspath(pdf)]
    assert os.path.dirname(dst) == os.path.abspath(files)
    assert open(dst, "rb").read() == b"%PDF-1.4 numbers"

    # same content again: reuse; changed content: a new name, history kept
    assert core.import_reported_outputs([str(pdf)], str(files))[
        os.path.abspath(pdf)] == dst
    # Rebuilt with DIFFERENT content of the SAME size, and the same
    # timestamp: filecmp.cmp caches by (size, mtime) and would call these
    # equal, leaving the conversation showing the old version.
    stamp = os.stat(pdf).st_mtime
    pdf.write_bytes(b"%PDF-1.4 rebuilt")
    assert len(b"%PDF-1.4 rebuilt") == len(b"%PDF-1.4 numbers")
    os.utime(pdf, (stamp, stamp))
    dst2 = core.import_reported_outputs([str(pdf)], str(files))[os.path.abspath(pdf)]
    assert dst2 != dst and dst2.endswith("_1.pdf")
    assert open(dst2, "rb").read() == b"%PDF-1.4 rebuilt"
    assert open(dst, "rb").read() == b"%PDF-1.4 numbers"


def test_displayable_markdown_points_local_images_at_the_card():
    text = ("Plot:\n\n![PYWall lateral earth pressures](/tmp/pywall_lateral_pressures.png)"
            "\n\n![web](https://example.com/a.png)\n![x](/tmp/other.png)")
    out = core.displayable_markdown(
        text, ["/root/c/files/pywall_lateral_pressures.png"])
    assert "*(PYWall lateral earth pressures — shown below)*" in out
    assert "![web](https://example.com/a.png)" in out
    assert "not viewable in chat" in out and "/tmp/other.png" in out
