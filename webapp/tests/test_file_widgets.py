"""Downloads and the PDF preview in the Foundry app frame (INF-37).

In the published Foundry app ``st.download_button`` saves an HTML error page
(``<id>.htm``, "File wasn't available on site") and a ``data:`` PDF frame is
blocked; a file the page builds in the browser saves, and images display
(the AI FDE's run and the owner's checks, 2026-10-20). ``webapp.file_widgets``
draws the in-browser download on Foundry and Streamlit's button elsewhere,
and the PDF preview as page images on Foundry.
"""

import ast
import base64
import json
import os
import re

import pytest

pytest.importorskip("streamlit")
fitz = pytest.importorskip("fitz")

import webapp.core as core
from webapp import file_widgets
from webapp.tests.test_app_smoke import _mk_at

WEBAPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_no_bare_download_button_outside_the_helper():
    """Every download in webapp goes through file_widgets.download_button:
    a bare ``st.download_button`` is broken in the Foundry frame."""
    bare = []
    for folder, dirs, files in os.walk(WEBAPP):
        dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
        for name in files:
            path = os.path.join(folder, name)
            if not name.endswith(".py") or \
                    os.path.samefile(path, file_widgets.__file__):
                continue
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and \
                        node.attr == "download_button" and not (
                            isinstance(node.value, ast.Name)
                            and node.value.id == "file_widgets"):
                    bare.append(f"{os.path.relpath(path, WEBAPP)}:"
                                f"{node.lineno}")
                if isinstance(node, ast.ImportFrom) and any(
                        a.name == "download_button" for a in node.names) \
                        and node.module != "webapp.file_widgets":
                    bare.append(f"{os.path.relpath(path, WEBAPP)}:"
                                f"{node.lineno}")
    assert not bare, bare


def _foundry(monkeypatch):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")


def _record(monkeypatch):
    seen = {"component": [], "native": [], "caption": []}
    monkeypatch.setattr(file_widgets, "_component",
                        lambda page, height: seen["component"].append(page))
    monkeypatch.setattr(file_widgets.st, "download_button",
                        lambda *a, **k: seen["native"].append((a, k)))
    monkeypatch.setattr(file_widgets.st, "caption",
                        lambda text, *a, **k: seen["caption"].append(text))
    return seen


def test_the_page_carries_the_bytes_and_the_real_name():
    data = bytes(range(256)) * 3
    name = 'calc "package" </script><b>.pdf'
    page = file_widgets.browser_download_html(data, name, "application/pdf",
                                              "Download")
    assert base64.b64encode(data).decode() in page
    assert "Open in new tab" in page and "URL.createObjectURL" in page
    # the name is a JS string that cannot close the script, and HTML-escaped
    # where it is text
    script = page.split("<script>", 1)[1]
    assert "</script>" not in script.rsplit("</script>", 1)[0]
    literal = re.search(r"const NAME = (.*);", script).group(1)
    assert json.loads(literal) == name
    assert "<b>" not in page.split("<script>", 1)[0]


@pytest.mark.parametrize("name", [
    "<!--<script>.pdf",                       # would hold the element open
    '"><img src=x onerror=alert(1)>.png',
    "a b & c > d.csv",
])
def test_no_name_reaches_the_page_as_markup(name):
    """The name is markup nowhere: escaped in the attribute and the button,
    and a JS string with no ``<``, ``>`` or ``&`` in the script, so the
    script element ends where it should and the page still runs."""
    page = file_widgets.browser_download_html(b"x", name, "a</script>b",
                                              f"Download {name}")
    head, script = page.split("<script>", 1)
    assert "<img" not in head and "<!--" not in head
    body, tail = script.split("</script>", 1)
    assert tail == "</body></html>"
    for const, value in (("NAME", name), ("MIME", "a</script>b")):
        literal = re.search(rf"const {const} = (.*);", body).group(1)
        assert not set("<>& ") & set(literal)
        assert json.loads(literal) == value


def test_the_frame_is_st_iframe_where_streamlit_has_it(monkeypatch):
    """``components.v1.html`` is deprecated for ``st.iframe``, which draws
    the same frame."""
    seen = []
    monkeypatch.setattr(file_widgets.st, "iframe",
                        lambda page, height: seen.append((page, height)),
                        raising=False)
    file_widgets._component("<p>x</p>", 44)
    assert seen == [("<p>x</p>", 44)]


def test_on_foundry_the_download_is_made_in_the_browser(monkeypatch):
    _foundry(monkeypatch)
    seen = _record(monkeypatch)
    file_widgets.download_button("Download", b"a,b\n1,2\n",
                                 file_name="table.csv", mime="text/csv",
                                 key="k1")
    assert not seen["native"]
    assert len(seen["component"]) == 1
    assert base64.b64encode(b"a,b\n1,2\n").decode() in seen["component"][0]
    # text data is sent as its UTF-8 bytes
    file_widgets.download_button("notes.md", "é", file_name="notes.md")
    assert base64.b64encode("é".encode()).decode() in seen["component"][1]


def test_on_foundry_a_file_over_the_cap_gets_a_message(monkeypatch):
    _foundry(monkeypatch)
    monkeypatch.setenv(file_widgets.DOWNLOAD_MAX_MB_ENV, "0.001")
    seen = _record(monkeypatch)
    file_widgets.download_button("Download", b"x" * 5000,
                                 file_name="big.pdf", mime="application/pdf")
    assert not seen["component"] and not seen["native"]
    assert "big.pdf" in seen["caption"][0]
    assert file_widgets.DOWNLOAD_MAX_MB_ENV in seen["caption"][0]


def test_elsewhere_it_is_streamlits_own_button(monkeypatch):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    seen = _record(monkeypatch)
    file_widgets.download_button("Download", b"abc", file_name="a.txt",
                                 mime="text/plain", key="k2")
    assert not seen["component"]
    (args, kwargs), = seen["native"]
    assert args == ("Download",)
    assert kwargs == {"data": b"abc", "file_name": "a.txt",
                      "mime": "text/plain", "key": "k2"}


# -- INF-44: large files on request, HTML in a sandbox --------------------


def _buttons(monkeypatch, pressed):
    """``st.button`` answering ``pressed`` (a list the test edits), with the
    labels shown recorded; session state a plain dict."""
    shown = []
    monkeypatch.setattr(file_widgets.st, "session_state", {})
    monkeypatch.setattr(file_widgets.st, "button",
                        lambda label, **k: shown.append(label)
                        or bool(pressed and pressed[0]))
    return shown


def test_on_foundry_a_large_file_waits_for_prepare(monkeypatch, tmp_path):
    _foundry(monkeypatch)
    monkeypatch.setenv(file_widgets.DOWNLOAD_PREPARE_MB_ENV, "0.01")
    seen = _record(monkeypatch)
    pressed = [False]
    shown = _buttons(monkeypatch, pressed)
    reads = []
    real = core.artifact_bytes
    monkeypatch.setattr(core, "artifact_bytes",
                        lambda p: reads.append(p) or real(p))
    big = tmp_path / "model.csv"
    big.write_bytes(b"1,2\n" * 5000)                       # 20 kB
    small = tmp_path / "small.csv"
    small.write_bytes(b"1,2\n")

    def both():
        file_widgets.download_file("Download", str(big), mime="text/csv",
                                   key="card")
        file_widgets.download_file("small.csv", str(small), key="side")

    both()                       # the large one: a button, and not read
    assert shown == ["Prepare download (0.0 MB)"]
    assert reads == [str(small)] and len(seen["component"]) == 1
    pressed[0] = True
    both()                       # pressed: its bytes go into the page
    assert reads[-2:] == [str(big), str(small)]
    assert base64.b64encode(big.read_bytes()).decode() in seen["component"][1]
    pressed[0] = False
    shown.clear()
    both()                       # a rerun keeps it, with no button
    assert shown == [] and len(seen["component"]) == 5
    file_widgets.forget_prepared()           # the conversation moved on
    both()
    assert shown == ["Prepare download (0.0 MB)"]


def test_on_foundry_bytes_made_on_request_are_not_asked_again(monkeypatch):
    _foundry(monkeypatch)
    monkeypatch.setenv(file_widgets.DOWNLOAD_PREPARE_MB_ENV, "0.001")
    seen = _record(monkeypatch)
    shown = _buttons(monkeypatch, [False])
    file_widgets.download_button("Download chat.zip", b"z" * 5000,
                                 file_name="chat.zip", prepare=False)
    assert not shown and len(seen["component"]) == 1
    file_widgets.download_button("chat.zip", b"z" * 5000,
                                 file_name="chat.zip", key="k")
    assert shown == ["Prepare chat.zip (0.0 MB)"]


def test_elsewhere_a_file_is_streamlits_own_button(monkeypatch, tmp_path):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    monkeypatch.setenv(file_widgets.DOWNLOAD_PREPARE_MB_ENV, "0")
    seen = _record(monkeypatch)
    f = tmp_path / "a.txt"
    f.write_bytes(b"abc")
    file_widgets.download_file("a.txt", str(f), mime="text/plain", key="k")
    (args, kwargs), = seen["native"]
    assert kwargs == {"data": b"abc", "file_name": "a.txt",
                      "mime": "text/plain", "key": "k"}
    with pytest.raises(OSError):
        file_widgets.download_file("x", str(tmp_path / "gone.txt"))


def test_the_sandbox_never_allows_the_apps_origin():
    flags = file_widgets.HTML_SANDBOX.split()
    assert "allow-scripts" in flags
    assert not {"allow-same-origin", "allow-top-navigation"} & set(flags)


class _Frames:
    def __init__(self, page):
        from html.parser import HTMLParser
        self.frames = []
        outer = self

        class P(HTMLParser):
            def handle_starttag(self, tag, attrs):
                if tag == "iframe":
                    outer.frames.append(dict(attrs))
        P().feed(page)


@pytest.mark.parametrize("text", [
    '<p onclick="x()">a & b</p><script>parent.document.title="x"</script>',
    '"></iframe><script>alert(1)</script>',
])
def test_a_preview_page_is_the_html_in_a_sandboxed_frame(text):
    page = file_widgets.sandboxed_html_page(text)
    (frame,) = _Frames(page).frames
    assert frame["sandbox"] == file_widgets.HTML_SANDBOX
    assert frame["srcdoc"] == text           # all of it, and only in there
    assert "<script" not in page


def test_open_in_new_tab_frames_html_and_svg(monkeypatch):
    page = file_widgets.browser_download_html(b"<p>x</p>", "r.html",
                                              "text/html", "Download")
    script = page.split("<script>", 1)[1]
    sandbox = re.search(r"const SANDBOX = (.*);", script).group(1)
    assert json.loads(sandbox) == file_widgets.HTML_SANDBOX
    active = re.search(r"const ACTIVE = JSON.parse\((.*)\);", script).group(1)
    assert set(json.loads(json.loads(active))) == {
        "text/html", "application/xhtml+xml", "image/svg+xml"}
    assert "window.open(openUrl()" in script
    assert 'frame.setAttribute("sandbox", SANDBOX)' in script


def _pdf(path, pages):
    doc = fitz.open()
    for i in range(pages):
        doc.new_page(width=300, height=400).insert_text((40, 60),
                                                         f"page {i + 1}")
    doc.save(str(path))
    doc.close()


def test_pdf_pages_are_drawn_once_per_file_version(tmp_path, monkeypatch):
    pdf = tmp_path / "calc.pdf"
    _pdf(pdf, 3)
    pngs, count = core.pdf_page_pngs(str(pdf), max_pages=2)
    assert count == 3 and len(pngs) == 2
    assert all(p.startswith(b"\x89PNG") for p in pngs)
    # cached: a rerun does not draw the pages again
    monkeypatch.setattr(fitz, "open", lambda *_a, **_k: 1 / 0)
    assert core.pdf_page_pngs(str(pdf), max_pages=2) == (pngs, count)
    assert core.pdf_page_pngs(str(tmp_path / "missing.pdf")) == ([], 0)


def _stream_files(_agent, _messages, thread_id, **_kw):
    files = core.conversation_files_dir(thread_id)
    os.makedirs(files, exist_ok=True)
    _pdf(os.path.join(files, "calc_package.pdf"), 3)
    with open(os.path.join(files, "inputs.csv"), "w", encoding="utf-8") as fh:
        fh.write("phi,30\n")
    yield {"kind": "token", "text": "Done."}
    yield {"kind": "turn_done", "answer": "Done.", "turn_tokens": 2}


def _all_protos(at) -> str:
    out = []

    def walk(node):
        if getattr(node, "proto", None) is not None:
            out.append(str(node.proto))
        children = getattr(node, "children", None)
        for child in (children.values() if isinstance(children, dict)
                      else []):
            walk(child)

    walk(at._tree)
    return "\n".join(out)


def _run_turn(monkeypatch, tmp_path):
    at = _mk_at(monkeypatch, tmp_path, _stream_files).run()
    at.chat_input[0].set_value("make the package").run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_foundry_app_has_no_native_download_and_previews_pages(monkeypatch,
                                                               tmp_path):
    """The whole app on Foundry: no Streamlit download button anywhere, the
    PDF card previews its pages as images once its switch is on (not
    before: every rerun draws every card), and no data: PDF frame."""
    _foundry(monkeypatch)
    pages = []
    monkeypatch.setattr(file_widgets, "_component",
                        lambda page, height: pages.append(page))
    drawn = []
    real_pngs = core.pdf_page_pngs
    monkeypatch.setattr(core, "pdf_page_pngs",
                        lambda path, *a, **k: drawn.append(path)
                        or real_pngs(path, *a, **k))
    at = _run_turn(monkeypatch, tmp_path)
    assert not at.get("download_button")
    names = {re.search(r"const NAME = (.*);", p).group(1) for p in pages}
    assert {'"calc_package.pdf"', '"inputs.csv"'} <= names
    images = " ".join(str(i.proto) for i in at.get("image"))
    assert "Page 1 of" not in images and not drawn
    (switch,) = [t for t in at.toggle
                 if str(t.key or "").startswith("pdfpages_")]
    switch.set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    images = " ".join(str(i.proto) for i in at.get("image"))
    assert "Page 1 of 3" in images and "Page 3 of 3" in images
    assert "data:application/pdf" not in _all_protos(at)


def test_elsewhere_the_app_keeps_its_buttons_and_pdf_frame(monkeypatch,
                                                          tmp_path):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    at = _run_turn(monkeypatch, tmp_path)
    labels = [b.proto.label for b in at.get("download_button")]
    assert "Download" in labels and "calc_package.pdf" in labels
    assert "Page 1 of 3" not in " ".join(str(i.proto)
                                         for i in at.get("image"))
    assert "data:application/pdf" in _all_protos(at)


def _stream_html(_agent, _messages, thread_id, **_kw):
    files = core.conversation_files_dir(thread_id)
    with open(os.path.join(files, "report.html"), "w", encoding="utf-8") as fh:
        fh.write("<h1>Report</h1><script>parent.document.title='x'</script>")
    yield {"kind": "token", "text": "Done."}
    yield {"kind": "turn_done", "answer": "Done.", "turn_tokens": 2}


def test_foundry_app_prepares_large_files_until_the_next_turn(monkeypatch,
                                                              tmp_path):
    """The whole app on Foundry: the PDF over the inline size waits for its
    Prepare button (sidebar and card alike), the small CSV is one click; a
    press puts the PDF in the page until the next turn."""
    _foundry(monkeypatch)
    monkeypatch.setenv(file_widgets.DOWNLOAD_PREPARE_MB_ENV, "0.0001")
    pages = []
    monkeypatch.setattr(file_widgets, "_component",
                        lambda page, height: pages.append(page))

    def names():
        out = {json.loads(re.search(r"const NAME = (.*);", p).group(1))
               for p in pages}
        pages.clear()
        return out

    at = _run_turn(monkeypatch, tmp_path)
    assert names() == {"inputs.csv"}
    prepare = [b for b in at.button if b.label.startswith("Prepare ")
               and b.label.endswith(" MB)")]
    assert {b.label.split(" (")[0] for b in prepare} == {
        "Prepare download", "Prepare calc_package.pdf"}
    (card,) = [b for b in prepare if b.label.startswith("Prepare download (")]
    card.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert names() == {"inputs.csv", "calc_package.pdf"}
    at.run()                                       # kept on a rerun
    assert names() == {"inputs.csv", "calc_package.pdf"}
    at.chat_input[0].set_value("and again").run()  # the next turn drops it
    assert not at.exception, [e.value for e in at.exception]
    names()
    at.run()
    assert names() == {"inputs.csv"}


def test_the_html_preview_is_sandboxed_on_every_host(monkeypatch, tmp_path):
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    at = _mk_at(monkeypatch, tmp_path, _stream_html).run()
    at.chat_input[0].set_value("write a report").run()
    assert not at.exception, [e.value for e in at.exception]
    protos = _all_protos(at)
    assert "<h1>Report</h1>" not in protos           # never the page itself
    assert "&lt;h1&gt;Report&lt;/h1&gt;" in protos   # only inside srcdoc
    assert f'sandbox=\\"{file_widgets.HTML_SANDBOX}\\"' in protos \
        or f'sandbox="{file_widgets.HTML_SANDBOX}"' in protos
