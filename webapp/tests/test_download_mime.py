"""Download cards hand the browser the right content type.

A file served as ``application/octet-stream`` is saved by name and opened by
guess; the Office formats are what that costs, because a .docx saved without
its type reads to the browser as a zip.

``app.py`` is executed in a SUBPROCESS. Running the script in this process
leaves Streamlit's own globals part-way through the sidebar's form, and every
later ``AppTest`` in the session then fails with "st.button() can't be used in
an st.form()" — a fresh interpreter is the cheap way to read one table out of a
script without wrecking the suite that follows it.
"""

import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("streamlit")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
_APP = os.path.join(_ROOT, "webapp", "app.py")

_PROBE = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("webapp_app_for_mime", %r)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
names = json.loads(sys.argv[1])
print("<<<" + json.dumps({n: module._mime_for(n) for n in names}) + ">>>")
""" % _APP

_EXPECTED = {
    "review.pdf": "application/pdf",
    "calc.html": "text/html",
    "notes.md": "text/markdown",
    "profile.png": "image/png",
    "section.dxf": "application/dxf",
    "memo.docx": ("application/vnd.openxmlformats-officedocument"
                  ".wordprocessingml.document"),
    "MEMO.DOCX": ("application/vnd.openxmlformats-officedocument"
                  ".wordprocessingml.document"),
    "table.xlsx": ("application/vnd.openxmlformats-officedocument"
                   ".spreadsheetml.sheet"),
    "deck.pptx": ("application/vnd.openxmlformats-officedocument"
                  ".presentationml.presentation"),
    "archive.tar.gz": "application/octet-stream",
}


def test_the_formats_the_agent_writes_all_have_a_type(tmp_path):
    env = dict(os.environ, GEOTECH_WEBAPP_DATA=str(tmp_path))
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(sorted(_EXPECTED))],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert "<<<" in proc.stdout, proc.stdout[-2000:] + proc.stderr[-2000:]
    payload = proc.stdout.split("<<<", 1)[1].split(">>>", 1)[0]
    assert json.loads(payload) == _EXPECTED
