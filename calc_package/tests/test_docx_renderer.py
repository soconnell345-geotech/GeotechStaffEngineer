"""Markdown -> .docx, checked by reopening the file python-docx wrote.

Every assertion here is a round trip: the sample below exercises one of each
construct the renderer claims, and the document is read back with python-docx
rather than inspected as it is built.
"""

import os

import pytest

docx = pytest.importorskip("docx")
fitz = pytest.importorskip("fitz")
pytest.importorskip("markdown_it")

from calc_package.docx_renderer import markdown_to_docx  # noqa: E402

SAMPLE = """\
# Subsurface Review

Borings were advanced to **25 m** and the profile is *interpreted*, not
measured; the driver is `analyze_bearing_capacity`.

## Findings

### Stratigraphy

#### Notes

- Fill over residual soil
- Groundwater at 3.2 m
  - Measured 24 hours after drilling
  - Not a stabilized level
- Rock at 18 m

1. Confirm the bearing elevation
2. Re-survey the benchmark
   1. Tie to the state plane control

| Boring | Depth (m) | N |
|---|---|---|
| B-1 | 25.0 | 14 |
| B-2 | 18.5 | 31 |

*Table 1 - Summary of the borings.*

> The report states a 2.5 m frost depth, which the local code does not.

```python
fos = capacity / demand
print(fos)
```

![Interpreted profile](profile.png)

![Missing figure](nowhere.png)

---

## Recommendations

See the [design guide](https://example.invalid/guide) for the method.
"""


def _png(path):
    """A real 16x16 PNG, so the renderer has an image it can actually embed."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 16), False)
    pix.set_rect(pix.irect, (200, 120, 40))
    path.write_bytes(pix.tobytes("png"))
    return str(path)


@pytest.fixture
def rendered(tmp_path):
    _png(tmp_path / "profile.png")
    out = str(tmp_path / "review.docx")
    warnings = []
    path = markdown_to_docx(SAMPLE, out, base_dir=str(tmp_path),
                            title="Geotechnical Review", warnings=warnings)
    return docx.Document(path), warnings, path


def test_the_file_is_written_where_it_was_asked_for(rendered, tmp_path):
    _doc, _warnings, path = rendered
    assert path == os.path.abspath(str(tmp_path / "review.docx"))
    assert os.path.getsize(path) > 0


def test_headings_keep_their_level_and_their_words(rendered):
    doc, _w, _p = rendered
    heads = [(p.style.name, p.text) for p in doc.paragraphs
             if p.style.name.startswith(("Heading", "Title"))]
    assert ("Title", "Geotechnical Review") in heads
    assert ("Heading 1", "Subsurface Review") in heads
    assert ("Heading 2", "Findings") in heads
    assert ("Heading 3", "Stratigraphy") in heads
    assert ("Heading 4", "Notes") in heads


def test_bold_italic_and_inline_code_survive_as_runs(rendered):
    doc, _w, _p = rendered
    par = next(p for p in doc.paragraphs if p.text.startswith("Borings were"))
    assert any(r.bold and "25 m" in r.text for r in par.runs)
    assert any(r.italic and "interpreted" in r.text for r in par.runs)
    assert any(r.font.name == "Consolas" and "analyze_bearing" in r.text
               for r in par.runs)
    # A soft line break inside a paragraph stays one paragraph.
    assert "not\nmeasured" not in par.text and "not measured" in par.text


def test_lists_use_word_list_styles_with_one_level_of_nesting(rendered):
    doc, _w, _p = rendered
    styles = {p.text: p.style.name for p in doc.paragraphs if p.text}
    assert styles["Fill over residual soil"] == "List Bullet"
    assert styles["Measured 24 hours after drilling"] == "List Bullet 2"
    assert styles["Confirm the bearing elevation"] == "List Number"
    assert styles["Tie to the state plane control"] == "List Number 2"


def test_the_table_is_a_grid_with_a_bold_header_and_a_caption(rendered):
    doc, _w, _p = rendered
    assert len(doc.tables) == 1
    table = doc.tables[0]
    assert table.style.name == "Table Grid"
    assert [c.text for c in table.rows[0].cells] == ["Boring", "Depth (m)", "N"]
    assert [c.text for c in table.rows[1].cells] == ["B-1", "25.0", "14"]
    assert [c.text for c in table.rows[2].cells] == ["B-2", "18.5", "31"]
    header_runs = [r for c in table.rows[0].cells
                   for p in c.paragraphs for r in p.runs]
    assert header_runs and all(r.bold for r in header_runs)
    # The italic line under the table becomes its caption, not a paragraph.
    caption = next(p for p in doc.paragraphs
                   if p.text.startswith("Table 1 - Summary"))
    assert caption.style.name == "Caption"


def test_a_block_quote_is_indented_and_italic(rendered):
    doc, _w, _p = rendered
    quote = next(p for p in doc.paragraphs
                 if p.text.startswith("The report states"))
    assert quote.paragraph_format.left_indent is not None
    assert quote.paragraph_format.left_indent > 0
    assert all(r.italic for r in quote.runs if r.text.strip())


def test_a_fenced_block_is_one_monospaced_paragraph(rendered):
    doc, _w, _p = rendered
    code = next(p for p in doc.paragraphs if "fos = capacity" in p.text)
    assert all(r.font.name == "Consolas" for r in code.runs)
    assert "print(fos)" in code.text          # both lines, one paragraph


def test_the_image_is_embedded_and_captioned_from_its_alt_text(rendered):
    doc, _w, _p = rendered
    assert len(doc.inline_shapes) == 1
    shape = doc.inline_shapes[0]
    section = doc.sections[0]
    text_width = (section.page_width - section.left_margin
                  - section.right_margin)
    assert 0 < shape.width <= text_width
    caption = next(p for p in doc.paragraphs
                   if p.text == "Interpreted profile")
    assert caption.style.name == "Caption"


def test_a_missing_image_is_a_warning_and_a_note_not_an_exception(rendered):
    doc, warnings, _p = rendered
    assert any("nowhere.png" in w for w in warnings)
    assert any(p.text == "[image not found: nowhere.png]"
               for p in doc.paragraphs)


def test_a_horizontal_rule_becomes_a_page_break(rendered):
    doc, _w, _p = rendered
    breaks = [p for p in doc.paragraphs if 'w:type="page"' in p._p.xml]
    assert len(breaks) == 1


def test_a_link_keeps_its_words_and_its_address(rendered):
    doc, _w, _p = rendered
    par = next(p for p in doc.paragraphs if "design guide" in p.text)
    assert "https://example.invalid/guide" in par.text


# -- degrading rather than raising ---------------------------------------------

def test_unknown_constructs_become_plain_paragraphs(tmp_path):
    warnings = []
    path = markdown_to_docx(
        "<div class='x'>raw html</div>\n\n"
        "###### Six deep\n\n"
        "Ordinary text.\n",
        str(tmp_path / "odd.docx"), warnings=warnings)
    doc = docx.Document(path)
    texts = [p.text for p in doc.paragraphs]
    assert any("raw html" in t for t in texts)
    assert any("Six deep" in t for t in texts)
    assert any("raw HTML" in w for w in warnings)
    assert any("heading level 6" in w for w in warnings)


def test_empty_markdown_still_writes_a_document(tmp_path):
    path = markdown_to_docx("", str(tmp_path / "empty.docx"), title="Nothing")
    doc = docx.Document(path)
    assert [p.text for p in doc.paragraphs if p.text] == ["Nothing"]


def test_a_relative_image_resolves_against_base_dir(tmp_path):
    figures = tmp_path / "files"
    figures.mkdir()
    _png(figures / "fig.png")
    warnings = []
    path = markdown_to_docx("![](fig.png)\n", str(tmp_path / "rel.docx"),
                            base_dir=str(figures), warnings=warnings)
    assert warnings == []
    assert len(docx.Document(path).inline_shapes) == 1


def test_missing_python_docx_says_how_to_install_it(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def no_docx(name, *args, **kwargs):
        if name == "docx" or name.startswith("docx."):
            raise ImportError("No module named 'docx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_docx)
    with pytest.raises(ImportError, match="pip install python-docx"):
        markdown_to_docx("# x", str(tmp_path / "none.docx"))
