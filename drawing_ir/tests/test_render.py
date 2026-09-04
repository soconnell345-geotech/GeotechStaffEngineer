"""Tests for drawing_ir.render — the region-snip vision primitive (B2)."""

import io

import pytest

fitz = pytest.importorskip("fitz")

from drawing_ir.render import (
    DEFAULT_MARK_RADIUS, MIN_CROP_SIZE, clip_rect_for_bbox, render_region,
)


def _decode(png_bytes):
    """Decode PNG bytes -> a PyMuPDF Pixmap (no new dependency for tests)."""
    return fitz.Pixmap(io.BytesIO(png_bytes))


@pytest.fixture
def sheet_pdf(tmp_path):
    """A 400x300 pt page with a small black square at (100,100)-(120,120)."""
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    sh = page.new_shape()
    sh.draw_rect(fitz.Rect(100, 100, 120, 120))
    sh.finish(color=(0, 0, 0), fill=(0, 0, 0))
    sh.commit()
    path = tmp_path / "sheet.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


class TestClipRectForBbox:
    def test_none_bbox_is_full_page(self):
        assert clip_rect_for_bbox(None, (0, 0, 400, 300)) == (0, 0, 400, 300)

    def test_padding_applied(self):
        # bbox 20x20 -> span=max(20,20,MIN_CROP_SIZE)=20 -> pad=0.15*20=3.
        clip = clip_rect_for_bbox((100, 100, 120, 120), (0, 0, 400, 300),
                                  pad_frac=0.15)
        assert clip == pytest.approx((97.0, 97.0, 123.0, 123.0))

    def test_degenerate_bbox_uses_min_crop_floor(self):
        # A point-like bbox: span floors to MIN_CROP_SIZE, not 0.
        clip = clip_rect_for_bbox((50, 50, 50, 50), (0, 0, 400, 300),
                                  pad_frac=0.5)
        pad = 0.5 * MIN_CROP_SIZE
        assert clip == pytest.approx((50 - pad, 50 - pad, 50 + pad, 50 + pad))

    def test_clamped_to_page(self):
        clip = clip_rect_for_bbox((-50, -50, 10, 10), (0, 0, 400, 300),
                                  pad_frac=0.0)
        assert clip[0] == 0.0 and clip[1] == 0.0

    def test_normalizes_reversed_bbox(self):
        a = clip_rect_for_bbox((120, 120, 100, 100), (0, 0, 400, 300))
        b = clip_rect_for_bbox((100, 100, 120, 120), (0, 0, 400, 300))
        assert a == b


class TestRenderRegion:
    def test_full_page_pixel_size(self, sheet_pdf):
        png = render_region(filepath=sheet_pdf, page=0, bbox=None, dpi=150)
        pix = _decode(png)
        assert pix.width == pytest.approx(round(400 * 150 / 72), abs=1)
        assert pix.height == pytest.approx(round(300 * 150 / 72), abs=1)

    def test_png_magic_bytes(self, sheet_pdf):
        png = render_region(filepath=sheet_pdf, page=0, bbox=(100, 100, 120, 120))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_bbox_crop_matches_predicted_size(self, sheet_pdf):
        bbox = (100, 100, 120, 120)
        dpi = 300
        clip = clip_rect_for_bbox(bbox, (0, 0, 400, 300), pad_frac=0.15)
        png = render_region(filepath=sheet_pdf, page=0, bbox=bbox, dpi=dpi,
                            pad_frac=0.15)
        pix = _decode(png)
        expected_w = round((clip[2] - clip[0]) * dpi / 72)
        expected_h = round((clip[3] - clip[1]) * dpi / 72)
        assert pix.width == pytest.approx(expected_w, abs=1)
        assert pix.height == pytest.approx(expected_h, abs=1)

    def test_crop_smaller_than_full_page(self, sheet_pdf):
        full = _decode(render_region(filepath=sheet_pdf, page=0, bbox=None, dpi=150))
        crop = _decode(render_region(filepath=sheet_pdf, page=0,
                                     bbox=(100, 100, 120, 120), dpi=150))
        assert crop.width < full.width and crop.height < full.height

    def test_content_bytes_input(self, sheet_pdf):
        with open(sheet_pdf, "rb") as f:
            data = f.read()
        png = render_region(content=data, page=0, bbox=(100, 100, 120, 120))
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_page_out_of_range(self, sheet_pdf):
        with pytest.raises(ValueError, match="out of range"):
            render_region(filepath=sheet_pdf, page=5)

    def test_no_source_raises(self):
        with pytest.raises(ValueError):
            render_region()

    def test_higher_dpi_yields_larger_image(self, sheet_pdf):
        lo = _decode(render_region(filepath=sheet_pdf, page=0,
                                   bbox=(100, 100, 120, 120), dpi=150))
        hi = _decode(render_region(filepath=sheet_pdf, page=0,
                                   bbox=(100, 100, 120, 120), dpi=300))
        assert hi.width > lo.width and hi.height > lo.height


class TestMarks:
    def _window_has_nonwhite(self, pix, x_pt, y_pt, clip, dpi, half=None):
        """True if any pixel in a window around (x_pt, y_pt) is non-white.

        A window (not one exact pixel) because a mark is drawn as a circle
        OUTLINE + label text, not a filled dot — the dead-center pixel itself
        can legitimately stay white. The window must be wide enough to reach
        the ring at ``DEFAULT_MARK_RADIUS`` (in device pixels at ``dpi``) plus
        a margin for the label text drawn just outside it.
        """
        if half is None:
            half = int(DEFAULT_MARK_RADIUS * dpi / 72) + 8
        cx = round((x_pt - clip[0]) * dpi / 72)
        cy = round((y_pt - clip[1]) * dpi / 72)
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                px, py = cx + dx, cy + dy
                if 0 <= px < pix.width and 0 <= py < pix.height:
                    if pix.pixel(px, py) != (255, 255, 255):
                        return True
        return False

    def test_mark_changes_pixels_at_its_location(self, sheet_pdf):
        # An empty region of the page (no geometry), so any change there is
        # unambiguously the mark, not incidental drawing content.
        bbox = (200, 200, 220, 220)
        dpi = 300
        clip = clip_rect_for_bbox(bbox, (0, 0, 400, 300), pad_frac=0.15)
        mark_xy = (210, 210)

        plain = _decode(render_region(filepath=sheet_pdf, page=0, bbox=bbox, dpi=dpi))
        marked = _decode(render_region(filepath=sheet_pdf, page=0, bbox=bbox, dpi=dpi,
                                       marks=[(mark_xy[0], mark_xy[1], "1")]))
        assert plain.width == marked.width and plain.height == marked.height

        assert not self._window_has_nonwhite(plain, *mark_xy, clip, dpi)
        assert self._window_has_nonwhite(marked, *mark_xy, clip, dpi)

    def test_multiple_marks_all_drawn(self, sheet_pdf):
        bbox = (150, 150, 250, 250)
        dpi = 300
        clip = clip_rect_for_bbox(bbox, (0, 0, 400, 300), pad_frac=0.15)
        marks = [(170, 170, "1"), (230, 230, "2")]
        png = render_region(filepath=sheet_pdf, page=0, bbox=bbox, dpi=dpi,
                            marks=marks)
        pix = _decode(png)
        for x, y, _ in marks:
            assert self._window_has_nonwhite(pix, x, y, clip, dpi)

    def test_default_mark_radius_is_positive(self):
        assert DEFAULT_MARK_RADIUS > 0

    def test_no_marks_does_not_mutate_source(self, sheet_pdf):
        # Rendering with marks must not persist onto the source file (doc.close()
        # discards edits) — render the SAME region twice, second call plain.
        bbox = (200, 200, 220, 220)
        render_region(filepath=sheet_pdf, page=0, bbox=bbox,
                      marks=[(210, 210, "1")])
        after = _decode(render_region(filepath=sheet_pdf, page=0, bbox=bbox))
        clip = clip_rect_for_bbox(bbox, (0, 0, 400, 300), pad_frac=0.15)
        px = round((210 - clip[0]) * 300 / 72)
        py = round((210 - clip[1]) * 300 / 72)
        assert after.pixel(min(px, after.width - 1), min(py, after.height - 1)) == (255, 255, 255)
