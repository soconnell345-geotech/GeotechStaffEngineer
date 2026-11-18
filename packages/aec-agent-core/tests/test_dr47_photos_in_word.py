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


def test_a_twenty_photo_report_keeps_every_photo_and_shrinks(tmp_path):
    # Made-up photographs are smoother than real ones and store smaller, so
    # the 5 MB bound below is theirs. Real photographs at 1200 x 800 come to
    # about 180 to 430 KB each at quality 85, more at full colour
    # resolution: a real 20-photograph report is about 4 to 13 MB. The aim
    # that holds is well under a 20 MB mail limit with every photograph kept.
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
    assert os.path.getsize(out) < 5 * 1024 * 1024


def test_photo_is_stored_at_full_colour_resolution(tmp_path):
    """4:4:4, not the 4:2:0 a JPEG writer uses by default: a chart or a
    drawing crop that is photograph-like in its pixels (a shaded heat map,
    line work over an aerial photograph) converts too, and its coloured
    lettering and symbols must not bleed."""
    from PIL.JpegImagePlugin import get_sampling
    _photo(1500, 1000, seed=8).save(tmp_path / "s.png")
    out = tmp_path / "s.docx"
    markdown_to_docx("![](s.png)", str(out), base_dir=str(tmp_path))
    _d, pics = _pictures(out)
    assert pics[0][0] == "image/jpeg"
    img = PIL_Image.open(io.BytesIO(pics[0][1]))
    assert get_sampling(img) == 0                   # 4:4:4


# Paper tones at and across the edges of 32-level cells (224 is one, 208
# and 240 are halfway), and flat coloured zones whose tones sit in the
# middle of a cell: the case where a single grid of cells split the paper
# into up to eight fragments and took the zone for the paper.
_PAPER = [(224, 224, 224), (225, 223, 222), (223, 226, 224),
          (240, 238, 236), (224, 208, 224)]
_ZONES = [(144, 208, 240), (240, 240, 144), (240, 176, 208)]


def _scan(tone, noise, zone, zone_rgb, seed):
    """A made-up colour scan of a drawing: paper of one tone, a flat
    coloured zone over ``zone`` of its width, line work and lettering, a
    scanner's slight blur, and noise of standard deviation ``noise``."""
    from PIL import ImageDraw, ImageFilter
    rng = np.random.default_rng(seed)
    w, h = 1275, 825
    img = PIL_Image.new("RGB", (w, h), tone)
    d = ImageDraw.Draw(img)
    zw = int(w * zone)
    x0 = int(rng.integers(0, w - zw))
    d.rectangle((x0, 0, x0 + zw, h), fill=zone_rgb)
    for k in range(11):
        y = 45 + 70 * k
        d.line((30, y, w - 30, y + int(rng.integers(-25, 25))),
               fill=(40, 40, 40), width=2)
    for k in range(8):
        x = 60 + 150 * k
        d.line((x, 30, x + int(rng.integers(-30, 30)), h - 30),
               fill=(60, 60, 60), width=3)
    for k in range(30):
        d.text((int(rng.integers(30, w - 200)), int(rng.integers(20, h - 30))),
               f"B-{k} SAND N={k + 7} EL 12.{k}", fill=(20, 20, 20))
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    a = np.asarray(img, dtype=float) + rng.normal(0, noise, (h, w, 3))
    return PIL_Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _sky_photo(w, h, seed):
    """A made-up photograph with a smooth sky over its top 45 %."""
    a = np.asarray(_photo(w, h, seed=seed), dtype=float)
    top = int(h * 0.45)
    t = np.linspace(0.0, 1.0, top)[:, None, None]
    sky = (1 - t) * np.array([100, 150, 215]) + t * np.array([195, 215, 240])
    rng = np.random.default_rng(seed)
    a[:top] = sky + rng.normal(0, 2.0, (top, w, 3))
    return PIL_Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def test_colour_scans_stay_and_photographs_convert_whatever_the_paper_tone(
        tmp_path):
    import itertools
    from aec_agent_core.docx_renderer import _is_photograph
    judged = []
    cases = list(itertools.product(_PAPER, (2, 3, 5), (0.15, 0.25)))
    for i, (tone, noise, zone) in enumerate(cases):
        if _is_photograph(_scan(tone, noise, zone, _ZONES[i % 3], seed=i)):
            judged.append((tone, noise, zone))
    assert judged == [], f"scans taken for photographs: {judged}"
    photos = [_photo(1500, 1000, seed=20), _photo(900, 600, seed=21),
              _photo(1200, 800, seed=22, noise=1.5),
              _sky_photo(1500, 1000, 23), _sky_photo(1200, 800, 24)]
    assert [_is_photograph(p) for p in photos] == [True] * len(photos)
    # End to end, the first of the scans the single grid took for a
    # photograph stays PNG, byte for byte.
    _scan((224, 224, 224), 3, 0.25, _ZONES[1], seed=1).save(
        tmp_path / "scan.png")
    before = (tmp_path / "scan.png").read_bytes()
    out = tmp_path / "scan.docx"
    markdown_to_docx("![Scanned plan](scan.png)", str(out),
                     base_dir=str(tmp_path))
    _d, pics = _pictures(out)
    assert pics == [("image/png", before)]
