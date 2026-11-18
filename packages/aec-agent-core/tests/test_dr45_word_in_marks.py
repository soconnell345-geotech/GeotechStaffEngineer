"""DR-45: Word in says what it did not read and gives tracked deletions.

Equations in Word's own format and symbols in a font with no table here are
counted in the not-read section; a Symbol-font ``w:sym``, and text typed in
the Symbol font, are read as the characters printed, and a run set in a
symbol font with no table keeps its letters and is counted; a file with tracked changes says so and gives the deleted text in
a section of its own after the body on the read_office path the model reads,
while docx_to_markdown's ``markdown`` stays the body alone (the writer's
inverse) and the section comes as ``tracked_changes``. Made-up files. Each
test fails on the code before DR-45 and passes after it, except the last,
which pins that a file with none of these reads exactly as before.
"""
import pytest

docx = pytest.importorskip("docx")
pytest.importorskip("markdown_it")

from docx.oxml import parse_xml  # noqa: E402

from aec_agent_core.docx_renderer import docx_to_markdown  # noqa: E402
from aec_agent_core.office_text import read_office  # noqa: E402

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
M = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
PHI = chr(0x3C6)          # GREEK SMALL LETTER PHI
GAMMA = chr(0x3B3)        # GREEK SMALL LETTER GAMMA
DEGREE = chr(0xB0)        # DEGREE SIGN


def _sym(font, char):
    return parse_xml(f'<w:r {W}><w:sym w:font="{font}" w:char="{char}"/>'
                     f'</w:r>')


def _tracked(kind, text, cid, author="J. Doe"):
    inner = "w:delText" if kind == "del" else "w:t"
    return parse_xml(
        f'<w:{kind} {W} w:id="{cid}" w:author="{author}" '
        f'w:date="2026-05-01T10:00:00Z"><w:r><{inner} xml:space="preserve">'
        f'{text}</{inner}></w:r></w:{kind}>')


def test_symbol_font_characters_are_read(tmp_path):
    d = docx.Document()
    p = d.add_paragraph("Friction angle ")
    p._p.append(_sym("Symbol", "F066"))
    p.add_run(" = 32")
    p._p.append(_sym("Symbol", "F0B0"))
    p.add_run(", unit weight ")
    p._p.append(_sym("Symbol", "0067"))          # the code without 0xF000
    p.add_run(" = 19 kN/m3.")
    path = tmp_path / "s.docx"
    d.save(str(path))
    r = docx_to_markdown(str(path))
    assert (f"Friction angle {PHI} = 32{DEGREE}, unit weight {GAMMA} = 19 "
            f"kN/m3.") in r["markdown"]
    assert "Not read in the Word file" not in r["markdown"]


def test_unknown_symbols_and_equations_are_counted_as_not_read(tmp_path):
    d = docx.Document()
    p = d.add_paragraph("Checked ")
    p._p.append(_sym("Wingdings", "F0FC"))       # no table here for it
    p._p.append(_sym("Symbol", "F0E2"))          # private-use only: no char
    p.add_run(" and the bearing equation ")
    p._p.append(parse_xml(f'<m:oMath {M}><m:r><m:t>q=cN</m:t></m:r>'
                          f'</m:oMath>'))
    p.add_run(" holds.")
    shown = d.add_paragraph()
    shown._p.append(parse_xml(
        f'<m:oMathPara {M}><m:oMath><m:r><m:t>a</m:t></m:r></m:oMath>'
        f'<m:oMath><m:r><m:t>b</m:t></m:r></m:oMath></m:oMathPara>'))
    path = tmp_path / "e.docx"
    d.save(str(path))
    md, fields = read_office(path)
    assert "Checked  and the bearing equation  holds." in md
    assert "- 3 x equation (Word's own equation format): present" in md
    assert "- 1 x symbol character in the Wingdings font: present" in md
    assert "- 1 x symbol character in the Symbol font: present" in md
    assert "Not read in the Word file" in fields["note"]


def _font_run(text, font):
    return parse_xml(f'<w:r {W}><w:rPr><w:rFonts w:ascii="{font}" '
                     f'w:hAnsi="{font}"/></w:rPr><w:t xml:space="preserve">'
                     f'{text}</w:t></w:r>')


def test_text_set_in_the_symbol_font_reads_as_printed(tmp_path):
    from docx.enum.style import WD_STYLE_TYPE
    d = docx.Document()
    greek = d.styles.add_style("Greek letters", WD_STYLE_TYPE.CHARACTER)
    greek.font.name = "Symbol"
    p = d.add_paragraph("Friction angle ")
    p._p.append(_font_run("f", "Symbol"))                 # typed "f"
    p.add_run(" = 30, ")
    p._p.append(_font_run(chr(0xF067) + " ", "Symbol"))   # 0xF000 + "g"
    p.add_run("= 18 kN/m3, ")
    p.add_run("s").style = greek                          # by its style
    p.add_run(" = 50 kPa, ")
    p._p.append(_font_run("f", "Times New Roman"))         # an ordinary f
    p.add_run(" = 2.")
    q = d.add_paragraph("Check: ")
    q._p.append(_font_run("f =", "Symbol"))   # a non-breaking space
    q.add_run(" 1.")
    path = tmp_path / "f.docx"
    d.save(str(path))
    r = docx_to_markdown(str(path))
    assert (f"Friction angle {PHI} = 30, {GAMMA} = 18 kN/m3, "
            f"{chr(0x3C3)} = 50 kPa, f = 2.") in r["markdown"]
    assert f"Check: {PHI} = 1." in r["markdown"]       # white space
    assert "Not read in the Word file" not in r["markdown"]


def test_symbol_font_runs_without_a_table_are_kept_and_counted(tmp_path):
    d = docx.Document()
    p = d.add_paragraph("Marks: ")
    odd = "q" + chr(0xF0E2)              # 0xE2 has only a private-use value
    p._p.append(_font_run(odd, "Symbol"))
    p.add_run(" and ")
    ding = chr(0xFC) + chr(0xFB)          # letters set in Wingdings
    p._p.append(_font_run(ding, "Wingdings"))
    p.add_run(" end.")
    path = tmp_path / "w.docx"
    d.save(str(path))
    md, fields = read_office(path)
    assert f"Marks: {odd} and {ding} end." in md        # left as they are
    assert ("- 2 x run set in a symbol font (the letters shown are not the "
            "characters printed): present in the file, not read") in md
    assert "Not read in the Word file" in fields["note"]


def test_tracked_changes_are_said_and_deletions_given_apart(tmp_path):
    d = docx.Document()
    p = d.add_paragraph("The footing bears on ")
    p._p.append(_tracked("del", "so", 1))
    p._p.append(_tracked("del", "ft ", 2))       # one deletion, two runs
    p._p.append(_tracked("ins", "stiff ", 3))
    p.add_run("clay at 2 m.")
    gone = d.add_paragraph()
    gone._p.append(_tracked("del", "Option B was rejected.", 4, "A. Roe"))
    d.add_paragraph("Settlement is 25 mm.")
    path = tmp_path / "t.docx"
    d.save(str(path))
    md, fields = read_office(path)
    body, _, section = md.partition("## Tracked changes in the Word file")
    # The body is Word's "accept all" view, as before.
    assert "The footing bears on stiff clay at 2 m." in body
    assert "soft" not in body and "Option B" not in body
    assert "The file holds tracked changes." in section
    assert ('- J. Doe, 2026-05-01: deleted "soft" after "The footing bears '
            'on"') in section
    assert ('- A. Roe, 2026-05-01: deleted "Option B was rejected." (its '
            'paragraph has no other text)') in section
    assert section.count("deleted \"") == 2
    assert "Tracked changes in the Word file" in fields["note"]
    # docx_to_markdown's body stays the writer's inverse: no deleted text,
    # no section; the section comes under its own key.
    r = docx_to_markdown(str(path))
    assert "soft" not in r["markdown"] and "Option B" not in r["markdown"]
    assert "Tracked changes" not in r["markdown"] and r["extras"] == []
    assert r["tracked_changes"].startswith(
        "## Tracked changes in the Word file")
    assert r["tracked_changes"].strip() in md


def test_insertions_only_are_said_and_the_body_is_as_before(tmp_path):
    def build(tracked):
        d = docx.Document()
        p = d.add_paragraph("Groundwater at ")
        if tracked:
            p._p.append(_tracked("ins", "3.5 m", 1))
        else:
            p.add_run("3.5 m")
        p.add_run(" below grade.")
        path = tmp_path / ("i.docx" if tracked else "plain.docx")
        d.save(str(path))
        return path
    plain = docx_to_markdown(str(build(False)))
    tracked_path = build(True)
    tracked = docx_to_markdown(str(tracked_path))
    assert tracked["markdown"] == plain["markdown"]
    assert tracked["extras"] == plain["extras"] == []
    assert plain["tracked_changes"] == ""
    assert "The file holds tracked changes." in tracked["tracked_changes"]
    assert 'deleted "' not in tracked["tracked_changes"]
    md, fields = read_office(tracked_path)
    assert md == plain["markdown"] + "\n" + tracked["tracked_changes"]
    assert "Tracked changes in the Word file" in fields["note"]


def test_a_few_short_deletions_read_exactly_as_before_the_bounds(tmp_path):
    d = docx.Document()
    p = d.add_paragraph("The footing bears on ")
    p._p.append(_tracked("del", "so", 1))
    p._p.append(_tracked("del", "ft ", 2))
    p._p.append(_tracked("ins", "stiff ", 3))
    p.add_run("clay at 2 m.")
    gone = d.add_paragraph()
    gone._p.append(_tracked("del", "Option B was rejected.", 4, "A. Roe"))
    q = d.add_paragraph()
    q._p.append(_tracked("del", "Early ", 5))
    q.add_run("results are given in Table 2.")
    path = tmp_path / "few.docx"
    d.save(str(path))
    assert docx_to_markdown(str(path))["tracked_changes"] == (
        "## Tracked changes in the Word file\n\n"
        "The file holds tracked changes. The text above reads them as "
        "accepted: insertions in, deletions out. The deleted text, which is "
        "not part of the document's current text:\n\n"
        '- J. Doe, 2026-05-01: deleted "soft" after "The footing bears on"\n'
        '- A. Roe, 2026-05-01: deleted "Option B was rejected." (its '
        "paragraph has no other text)\n"
        '- J. Doe, 2026-05-01: deleted "Early" before "results are given in '
        'Table 2."\n')


def test_a_long_deletion_is_clipped_with_its_place_kept(tmp_path):
    from aec_agent_core.docx_renderer import _DELETION_MAX_CHARS
    words = " ".join(f"clause{k}" for k in range(1500))   # 15,389 chars
    d = docx.Document()
    p = d.add_paragraph("The design basis is ")
    p._p.append(_tracked("del", words + " ", 1))
    p.add_run("as stated in section 3.")
    path = tmp_path / "long.docx"
    d.save(str(path))
    md, _fields = read_office(path)
    body, _, section = md.partition("## Tracked changes in the Word file")
    assert "clause" not in body
    (line,) = [x for x in section.splitlines() if 'deleted "' in x]
    assert line.startswith('- J. Doe, 2026-05-01: deleted "clause0 clause1 ')
    assert line.endswith(' clause1498 clause1499" after "The design basis '
                         'is"')
    quoted = line.split('deleted "', 1)[1].rsplit('" after', 1)[0]
    head, marker, tail = quoted.partition(" [... ")
    left = int(tail.split(" characters not shown ...] ", 1)[0]
               .replace(",", ""))
    shown = tail.split(" characters not shown ...] ", 1)[1]
    assert len(head) + left + len(shown) == len(words)       # exact count
    assert words.startswith(head) and words.endswith(shown)
    assert len(head) + len(shown) <= _DELETION_MAX_CHARS
    assert len(line) < _DELETION_MAX_CHARS + 150


def test_many_deletions_are_bounded_and_the_rest_counted(tmp_path):
    import re
    from aec_agent_core.docx_renderer import _DELETIONS_MAX_CHARS
    d = docx.Document()
    for k in range(200):
        p = d.add_paragraph(f"Boring B-{k} log, sheet {k}: ")
        p._p.append(_tracked("del", f"sample {k} was lost in transit ", k))
        p.add_run(f"refusal at {k}.5 m.")
    path = tmp_path / "many.docx"
    d.save(str(path))
    section = docx_to_markdown(str(path))["tracked_changes"]
    lines = [x for x in section.splitlines() if 'deleted "' in x]
    assert 20 < len(lines) < 200
    assert sum(len(x) + 1 for x in lines) <= _DELETIONS_MAX_CHARS
    for k, line in enumerate(lines):          # in order, each in its place
        assert line == (f'- J. Doe, 2026-05-01: deleted "sample {k} was '
                        f'lost in transit" after "Boring B-{k} log, sheet '
                        f'{k}:"')
    more = re.findall(r"^\((\d+) more deletions are not shown", section,
                      re.M)
    assert [int(n) for n in more] == [200 - len(lines)]
    assert section.endswith(" more deletions are not shown here: this "
                            "list stops at about "
                            f"{_DELETIONS_MAX_CHARS:,} characters.)\n")


def test_a_file_with_none_of_these_reads_exactly_as_before(tmp_path):
    from aec_agent_core.docx_renderer import markdown_to_docx
    path = tmp_path / "m.docx"
    markdown_to_docx("# Memo\n\nBody with **bold** and phi written out.\n\n"
                     "- one\n- two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
                     str(path))
    r = docx_to_markdown(str(path))
    assert r["extras"] == [] and r["tracked_changes"] == ""
    assert read_office(path)[0] == r["markdown"]
    assert r["markdown"] == ("# Memo\n\nBody with **bold** and phi written "
                             "out.\n\n- one\n- two\n\n| a | b |\n"
                             "| --- | --- |\n| 1 | 2 |\n")
