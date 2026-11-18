"""DR-47: write_docx stores photographs as JPEG at print size (made-up images).

Photo crops were embedded as the PNG files they arrived in, about 1 MB a
photo, and a 20-photo report passed a common 20 MB mail limit. A photograph
now goes into the .docx as JPEG, no wider than its printed size at about
200 dpi; charts, line work and anything the rule doubts stay as they were.
Each test fails on the code before DR-47 and passes after it, except the
"left alone" ones, which pin today's behaviour for those pictures.
"""
import hashlib
import io
import os

import pytest

docx = pytest.importorskip("docx")
np = pytest.importorskip("numpy")
PIL_Image = pytest.importorskip("PIL.Image")
pytest.importorskip("markdown_it")

from aec_agent_core.docx_renderer import (PHOTO_MAX_DPI,  # noqa: E402
                                          markdown_to_docx)

EMU_PER_INCH = 914400


def _photo(w, h, seed=0, noise=3.0):
    """A made-up "photograph": smooth colour fields with sensor-like noise."""
    rng = np.random.default_rng(seed)
    Image = PIL_Image
    coarse = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    base = np.asarray(Image.fromarray(coarse).resize((w, h), Image.BICUBIC),
                      dtype=float)
    detail = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    base += (np.asarray(Image.fromarray(detail).resize(
        (w, h), Image.BILINEAR), dtype=float) - 128) * 0.25
    base += rng.normal(0, noise, (h, w, 3))
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _chart(path):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot()
    x = np.linspace(0, 10, 200)
    for k in range(4):
        ax.plot(x, np.sin(x + k), label=f"case {k}")
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Settlement (mm)")
    fig.savefig(path)


def _drawing(path):
    from PIL import ImageDraw
    img = PIL_Image.new("RGB", (1600, 1100), "white")
    d = ImageDraw.Draw(img)
    for k in range(12):
        d.line((100, 80 + 80 * k, 1500, 120 + 70 * k), fill=(0, 0, 0),
               width=3)
        d.text((120, 90 + 80 * k), f"B-{k + 1}  SAND  N={10 + k}",
               fill=(200, 0, 0))
    d.rectangle((900, 300, 1400, 800), outline=(0, 0, 255), width=4)
    img.save(path)


def _pictures(path):
    """``(content type, blob)`` of each placed picture, in document order."""
    d = docx.Document(str(path))
    out = []
    for shape in d.inline_shapes:
        rid = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        part = d.part.related_parts[rid]
        out.append((part.content_type, part.blob))
    return d, out


def _text_width_in(d):
    sec = d.sections[0]
    return (sec.page_width - sec.left_margin - sec.right_margin) / EMU_PER_INCH


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_photo_is_stored_as_jpeg_no_wider_than_the_page_at_200_dpi(tmp_path):
    src = tmp_path / "site.png"
    _photo(1500, 1000).save(src)
    before = _sha(src)
    out = tmp_path / "p.docx"
    markdown_to_docx("![Site photo](site.png)", str(out),
                     base_dir=str(tmp_path))
    d, pics = _pictures(out)
    assert [ct for ct, _ in pics] == ["image/jpeg"]
    img = PIL_Image.open(io.BytesIO(pics[0][1]))
    assert img.format == "JPEG"
    assert img.width <= round(_text_width_in(d) * PHOTO_MAX_DPI) + 1
    shape = d.inline_shapes[0]
    sec = d.sections[0]
    # Printed as before: the full text width, the aspect kept.
    assert shape.width == sec.page_width - sec.left_margin - sec.right_margin
    assert abs(shape.width / shape.height - 1.5) < 0.01
    assert _sha(src) == before                      # the source is untouched
    assert len(pics[0][1]) < os.path.getsize(src)


def test_photo_with_its_own_dpi_prints_at_the_same_size(tmp_path):
    src = tmp_path / "close.png"
    _photo(1500, 1000, seed=1).save(src, dpi=(300, 300))    # 5 in x 3.33 in
    out = tmp_path / "q.docx"
    markdown_to_docx("![](close.png)", str(out), base_dir=str(tmp_path))
    d, pics = _pictures(out)
    assert pics[0][0] == "image/jpeg"
    shape = d.inline_shapes[0]
    assert abs(shape.width - 5 * EMU_PER_INCH) < EMU_PER_INCH / 100
    img = PIL_Image.open(io.BytesIO(pics[0][1]))
    assert abs(img.width - 5 * PHOTO_MAX_DPI) <= 1
    # Read back and written again, it still prints 5 inches wide.
    from aec_agent_core.docx_renderer import docx_to_markdown
    read = docx_to_markdown(str(out), image_dir=str(tmp_path / "img"),
                            image_ref="img")
    again = tmp_path / "again.docx"
    markdown_to_docx(read["markdown"], str(again), base_dir=str(tmp_path))
    d2 = docx.Document(str(again))
    assert abs(d2.inline_shapes[0].width - 5 * EMU_PER_INCH) \
        < EMU_PER_INCH / 50


def test_chart_drawing_and_doubtful_pictures_are_left_as_they_are(tmp_path):
    _chart(tmp_path / "chart.png")
    _drawing(tmp_path / "plan.png")
    _photo(900, 600, seed=2).convert("L").save(tmp_path / "grey.png")
    rgba = _photo(900, 600, seed=3).convert("RGBA")
    rgba.putpixel((0, 0), (0, 0, 0, 0))
    rgba.save(tmp_path / "cutout.png")
    _photo(800, 533, seed=4).save(tmp_path / "small.jpg", quality=95)
    names = ["chart.png", "plan.png", "grey.png", "cutout.png", "small.jpg"]
    before = {n: (tmp_path / n).read_bytes() for n in names}
    out = tmp_path / "c.docx"
    markdown_to_docx("\n\n".join(f"![{n}]({n})" for n in names), str(out),
                     base_dir=str(tmp_path))
    _d, pics = _pictures(out)
    assert [ct for ct, _ in pics] == ["image/png"] * 4 + ["image/jpeg"]
    for name, (_ct, blob) in zip(names, pics):
        assert blob == before[name], name           # byte for byte


def test_order_kept_and_an_unreadable_picture_is_noted_as_before(tmp_path):
    _photo(1400, 900, seed=5).save(tmp_path / "a.png")
    _chart(tmp_path / "b.png")
    _photo(1400, 900, seed=6).save(tmp_path / "c.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    warnings = []
    out = tmp_path / "o.docx"
    markdown_to_docx("![](a.png)\n\n![](broken.png)\n\n![](b.png)\n\n"
                     "![](c.png)\n", str(out), base_dir=str(tmp_path),
                     warnings=warnings)
    d, pics = _pictures(out)
    assert [ct for ct, _ in pics] == ["image/jpeg", "image/png", "image/jpeg"]
    assert any("broken.png" in w for w in warnings)
    assert any(p.text == "[image not found: broken.png]"
               for p in d.paragraphs)


def test_a_twenty_photo_report_stays_under_3_mb(tmp_path):
    base = np.asarray(_photo(1400, 933, seed=7))
    lines = ["# Site visit", ""]
    for k in range(20):
        # Twenty different photographs (python-docx stores a repeat once).
        img = np.roll(base, 53 * k, axis=1)
        img = np.roll(img, 31 * k, axis=0)
        PIL_Image.fromarray(img).save(tmp_path / f"photo{k}.png",
                                      compress_level=0)
        lines += [f"![Photo {k + 1}](photo{k}.png)", ""]
    out = tmp_path / "report.docx"
    markdown_to_docx("\n".join(lines), str(out), base_dir=str(tmp_path))
    d, pics = _pictures(out)
    assert len(pics) == 20                          # every photo kept
    assert all(ct == "image/jpeg" for ct, _ in pics)
    assert os.path.getsize(out) < 3 * 1024 * 1024
