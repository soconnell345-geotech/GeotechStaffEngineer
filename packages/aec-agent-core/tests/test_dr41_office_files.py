"""DR-41: what the file tools lose without saying (made-up files, built here).

Each test fails on the code before DR-41 and passes after it.
"""
import io
import zipfile

import pytest

docx = pytest.importorskip("docx")
openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("markdown_it")

from aec_agent_core.docx_renderer import docx_to_markdown, markdown_to_docx
from aec_agent_core.office_text import read_office


def _png(path, w, h):
    from PIL import Image
    Image.new("RGB", (w, h), (200, 30, 30)).save(path)


# -- Word out --------------------------------------------------------------

def test_sup_sub_become_runs(tmp_path):
    out = tmp_path / "a.docx"
    markdown_to_docx("Area is 12 m<sup>2</sup> and CO<sub>2</sub> too.", str(out))
    d = docx.Document(str(out))
    p = d.paragraphs[-1]
    assert "<sup>" not in p.text and "<sub>" not in p.text
    assert p.text == "Area is 12 m2 and CO2 too."
    sup = [r.text for r in p.runs if r.font.superscript]
    sub = [r.text for r in p.runs if r.font.subscript]
    assert sup == ["2"] and sub == ["2"]


def test_sup_in_table_cell(tmp_path):
    out = tmp_path / "t.docx"
    markdown_to_docx("| a | b |\n|---|---|\n| 5 m<sup>3</sup> | x |\n", str(out))
    cell = docx.Document(str(out)).tables[0].rows[1].cells[0]
    assert "<sup>" not in cell.text
    assert any(r.font.superscript for r in cell.paragraphs[0].runs)


def test_strike_is_a_strike_run(tmp_path):
    out = tmp_path / "s.docx"
    markdown_to_docx("keep ~~gone~~ keep", str(out))
    p = docx.Document(str(out)).paragraphs[-1]
    assert "~" not in p.text
    assert [r.text for r in p.runs if r.font.strike] == ["gone"]


def test_tall_image_fits_the_page(tmp_path):
    img = tmp_path / "tall.png"
    _png(img, 100, 2000)
    out = tmp_path / "i.docx"
    markdown_to_docx("![tall](tall.png)", str(out), base_dir=str(tmp_path))
    d = docx.Document(str(out))
    sec = d.sections[0]
    tw = sec.page_width - sec.left_margin - sec.right_margin
    th = sec.page_height - sec.top_margin - sec.bottom_margin
    shape = d.inline_shapes[0]
    assert shape.width <= tw and shape.height <= th
    assert abs(shape.height / shape.width - 20) < 0.5      # aspect kept


# -- Excel in --------------------------------------------------------------

def test_merged_label_fills_every_row(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Schedule"
    ws.append(["Level", "Item", "Qty"])
    ws.append(["Level 1", "Footing", 4])
    ws.append([None, "Pier", 2])
    ws.append([None, "Beam", 9])
    ws.append(["Level 2", "Footing", 3])
    ws.append([None, "Pier", 1])
    ws.merge_cells("A2:A4")
    ws.merge_cells("A5:A6")
    path = tmp_path / "s.xlsx"
    wb.save(path)
    md, _ = read_office(path)
    assert "| 3 | Level 1 | Pier | 2 |" in md
    assert "| 4 | Level 1 | Beam | 9 |" in md
    assert "| 6 | Level 2 | Pier | 1 |" in md
    assert "A2:A4" in md and "A5:A6" in md and "merged" in md.lower()


def test_merged_across_columns_and_no_merges_unchanged(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Zone", "Zone", "Qty"])
    ws.append(["North", None, 1])
    ws.merge_cells("A2:B2")
    p1 = tmp_path / "m.xlsx"
    wb.save(p1)
    md, _ = read_office(p1)
    assert "| 2 | North | North | 1 |" in md
    wb2 = openpyxl.Workbook()
    wb2.active.append(["a", "b"])
    wb2.active.append([1, 2])
    p2 = tmp_path / "n.xlsx"
    wb2.save(p2)
    md2, _ = read_office(p2)
    assert "Merged ranges" not in md2


# -- Word in ---------------------------------------------------------------

def _add_part(path, name, xml, rel_type, content_type):
    """Add a part to a .docx by hand (made-up comments / notes)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(path) as zin, \
            zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(
                    b"</Types>", (f'<Override PartName="/word/{name}" '
                                  f'ContentType="{content_type}"/></Types>'
                                  ).encode())
            if item.filename == "word/_rels/document.xml.rels":
                data = data.replace(
                    b"</Relationships>",
                    (f'<Relationship Id="rIdX{name[:3]}" Type="{rel_type}" '
                     f'Target="{name}"/></Relationships>').encode())
            zout.writestr(item, data)
        zout.writestr("word/" + name, xml)
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"


def test_comments_notes_headers_and_text_boxes_are_read(tmp_path):
    path = tmp_path / "r.docx"
    d = docx.Document()
    d.sections[0].header.paragraphs[0].text = "Project Alpha memo header"
    d.sections[0].footer.paragraphs[0].text = "Confidential footer"
    d.add_paragraph("Intro paragraph.")
    p = d.add_paragraph()
    from docx.oxml import parse_xml
    p._p.append(parse_xml(f'<w:commentRangeStart {W} w:id="0"/>'))
    p.add_run("The footing is 2 m wide")
    p._p.append(parse_xml(f'<w:commentRangeEnd {W} w:id="0"/>'))
    p.add_run(" and bears on clay.")
    # a text box in a run
    box = parse_xml(
        f'<w:r {W}><w:pict><v:shape xmlns:v="urn:schemas-microsoft-com:vml">'
        f'<v:textbox><w:txbxContent><w:p><w:r><w:t>Boxed note text'
        f'</w:t></w:r></w:p></w:txbxContent></v:textbox></v:shape></w:pict>'
        f'</w:r>')
    p._p.append(box)
    d.save(str(path))
    _add_part(str(path), "comments.xml",
              f'<?xml version="1.0"?><w:comments {W}><w:comment w:id="0" '
              f'w:author="J. Doe" w:date="2026-05-01T10:00:00Z"><w:p><w:r>'
              f'<w:t>Check the bearing value.</w:t></w:r></w:p></w:comment>'
              f'</w:comments>', REL + "comments",
              "application/vnd.openxmlformats-officedocument."
              "wordprocessingml.comments+xml")
    _add_part(str(path), "footnotes.xml",
              f'<?xml version="1.0"?><w:footnotes {W}>'
              f'<w:footnote w:type="separator" w:id="-1"><w:p/></w:footnote>'
              f'<w:footnote w:id="1"><w:p><w:r><w:t>Per the boring log.'
              f'</w:t></w:r></w:p></w:footnote></w:footnotes>',
              REL + "footnotes",
              "application/vnd.openxmlformats-officedocument."
              "wordprocessingml.footnotes+xml")
    md, fields = read_office(path)
    assert "Intro paragraph." in md
    assert "J. Doe" in md and "Check the bearing value." in md
    assert 'on "The footing is 2 m wide"' in md
    assert "Per the boring log." in md
    assert "Project Alpha memo header" in md and "Confidential footer" in md
    assert "Boxed note text" in md
    assert "picture" not in md                 # a text box is not a picture
    assert "read-only" not in md
    assert "Comments in the Word file" in fields["note"]
    assert set(fields) <= {"format", "read_as", "note", "title", "pictures",
                           "pictures_not_extracted"}


def test_chart_is_named_as_not_read(tmp_path):
    path = tmp_path / "c.docx"
    d = docx.Document()
    p = d.add_paragraph("Before the chart.")
    from docx.oxml import parse_xml
    p._p.append(parse_xml(
        f'<w:r {W} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/'
        f'main"><w:drawing><a:graphic><a:graphicData uri="http://schemas.'
        f'openxmlformats.org/drawingml/2006/chart"/></a:graphic></w:drawing>'
        f'</w:r>'))
    d.save(str(path))
    md, _ = read_office(path)
    assert "chart" in md and "not read" in md


def test_plain_document_has_no_extras(tmp_path):
    path = tmp_path / "p.docx"
    markdown_to_docx("# T\n\nBody text.\n", str(path))
    r = docx_to_markdown(str(path))
    assert r["extras"] == []
    assert "in the Word file" not in r["markdown"]
