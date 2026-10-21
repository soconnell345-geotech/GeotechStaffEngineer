"""How an image FILE reaches the vision model: an upload, a pasted
screenshot, a photo of a log, a scan saved as PNG or JPEG (BACKLOG VIS-09).

A PDF page is re-drawn from the PDF at whatever size a look needs
(:mod:`funhouse_agent.vision_view`). An image file has only the pixels it
came with, so every look at one is CUT from the original here and sized to
the vision model's budget, and the rest of the vision harness treats it like
a page:

* **The frame is the image's own pixels.** A look's ``view`` is a rect in
  pixels of the ORIGINAL image as a viewer shows it (a photo's EXIF
  orientation applied), top-left origin, x to the right, y down;
  ``analyze_image``'s view is the whole image, ``[0, 0, W, H]``. Boxes come
  back exactly as on a page: the vision call answers in pixels of the image
  it was SENT, the tool converts them in code to 0-999 boxes on the view
  (``vision_view.boxes_to_grid``), and ``render_region(view=, image_box=)``
  turns the two back into the original's pixels
  (``vision_view.image_box_to_page``, which is units-free).
* **Never a re-crop of a shrunk copy.** Every crop is cut from the original
  at full resolution (:func:`render`), then sized to the budget
  (:func:`sent_size`): shrunk when it is larger than the model reads in one
  image, as a page is; and, for a zoom or a tile, ENLARGED by a whole number
  of times (at most :data:`MAX_UPSCALE`) when the budget holds that much -
  no new detail, but lettering a few pixels tall arrives large enough to
  read. Whole numbers only: an enlargement by 1.03-1.2 times blurs one-pixel
  strokes into a beat pattern (measured on 1-px bars, 2026-10-14: 7 of 9
  bars left at 1.03x, all 9 at 2x, 3x and 4x).
* **The whole image goes as it is when it fits.** When it does not, it is
  shrunk to fit and, under the robust policy, ALSO read in tiles close to
  its own resolution (:func:`tile_count`), as a page with small lettering
  is; the tiles reuse ``vision_view.tile_boxes``.

Pillow does the decoding and the resizing; without it ``analyze_image``
sends the file as it always did and a zoom on an image says it needs Pillow.
"""

from __future__ import annotations

import io
import math
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

#: A zoom or a tile is enlarged at most this many times. A crop of a few
#: dozen pixels would otherwise be blown up fifty times into a few blurred
#: blocks; four times puts a 9 px screen font at 36 px.
MAX_UPSCALE = 4

#: The whole image is read in one look when it reaches the model at least
#: this fraction of its own size; smaller, its lettering has shrunk past
#: reading (a 12 px screen font under 11 px) and ``tiles="auto"`` also reads
#: it in tiles. A 1920 x 1080 screenshot fits whole; a 2560 x 1440 one goes
#: at 0.8 and is tiled 2 x 2; a 4032 x 3024 phone photo, 3 x 3.
WHOLE_MIN_SCALE = 0.9

#: A zoom window is never narrower than this many of the image's pixels on
#: either side (widened about its centre), so a box round one character
#: still shows the letters beside it.
MIN_WINDOW_PX = 32

#: A JPEG photo's crop is sent as JPEG at this quality when the engine labels
#: bytes by their real type; everything else goes as PNG.
JPEG_QUALITY = 90

#: The EXIF tag that says which way a photo is turned.
_EXIF_ORIENTATION = 0x0112

#: PIL's transposes that turn an image ``n`` degrees CLOCKWISE (PIL's own
#: names count counter-clockwise).
_CLOCKWISE = {90: "ROTATE_270", 180: "ROTATE_180", 270: "ROTATE_90"}


class Picture(NamedTuple):
    """An image file, decoded once: ``image`` is the PIL image oriented as a
    viewer shows it (all of its pixels), ``width`` x ``height`` its size,
    ``kind`` the file type, ``data`` the file's bytes, ``note`` what had to
    be done to read it (``None`` when nothing), and ``as_is`` whether the
    file can be sent unchanged when the whole image fits."""
    image: Any
    width: int
    height: int
    kind: str
    data: bytes
    note: Optional[str]
    as_is: bool


def open_picture(data: bytes, kind: str) -> Picture:
    """Decode an image file of ``kind`` (``png``, ``jpeg``, ``webp``, ``gif``,
    ``tiff``, ``bmp``): its first frame, turned upright as its EXIF
    orientation says, in a mode the resizer and the encoders take. Raises
    ``ImportError`` without Pillow and ``ValueError`` for a damaged file."""
    from PIL import Image, ImageOps
    try:
        im = Image.open(io.BytesIO(bytes(data)))
        frames = getattr(im, "n_frames", 1) or 1
        im.seek(0)
        im.load()
        try:
            orientation = int(im.getexif().get(_EXIF_ORIENTATION) or 1)
        except Exception:  # noqa: BLE001 - no EXIF is upright
            orientation = 1
        if orientation != 1:
            im = ImageOps.exif_transpose(im)
        if im.mode == "1":
            im = im.convert("L")
        elif im.mode.startswith("I") or im.mode == "F":
            im = _deep_grey_to_l(im)
        elif im.mode not in ("L", "LA", "RGB", "RGBA"):
            im = im.convert("RGBA" if im.mode in ("P", "PA") else "RGB")
    except Exception as exc:  # noqa: BLE001 - said to the agent as damage
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc
    notes = []
    if kind not in ("png", "jpeg", "webp"):
        notes.append(f"converted from {kind.upper()} to PNG")
        if frames > 1:
            notes[-1] += f" (page 1 of {frames}: only the first was sent)"
    if orientation != 1:
        notes.append("turned upright as the photo's EXIF orientation says "
                     "(as a viewer shows it); its pixels are counted that "
                     "way")
    return Picture(im, int(im.size[0]), int(im.size[1]), kind, bytes(data),
                   "; ".join(notes) or None,
                   kind in ("png", "jpeg", "webp") and orientation == 1)


def _deep_grey_to_l(im):
    """A 16-bit or 32-bit greyscale image (some scanners write 16-bit PNG)
    as 8-bit greyscale, its own range stretched to 0-255: Pillow's plain
    conversion clips everything over 255 to white."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return Image.new("L", im.size, 255 if hi > 0 else 0)
    a = (a - lo) * (255.0 / (hi - lo))
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8), "L")


def _fit(width: int, height: int, engine=None) -> Tuple[int, int]:
    """The largest size of this shape the vision budget holds (larger or
    smaller than ``width`` x ``height``) - planlens' ``fit_size`` on
    ``vision_view.budget``, or the image edge cap when no budget applies."""
    from funhouse_agent import vision_view
    bud = vision_view.budget(engine)
    if bud is not None:
        try:
            from planlens.document.budget import fit_size
            return fit_size(width, height, bud)
        except Exception:  # noqa: BLE001 - an older planlens: the edge cap
            pass
    cap = vision_view.max_px(engine) or vision_view.DEFAULT_MAX_PX
    s = float(cap) / max(width, height)
    return max(1, int(width * s)), max(1, int(height * s))


def sent_size(width: int, height: int, engine=None,
              enlarge: bool = False) -> Tuple[int, int]:
    """The size a ``width`` x ``height`` crop is sent at: shrunk to fit the
    budget when it is larger; as it is when it fits; and, with ``enlarge``
    (a zoom, a tile), as many whole times larger as the budget holds, at
    most :data:`MAX_UPSCALE`."""
    fw, fh = _fit(width, height, engine)
    scale = min(fw / float(width), fh / float(height))
    if scale < 1.0:
        return fw, fh
    k = max(1, min(MAX_UPSCALE, int(math.floor(scale + 1e-9)))) \
        if enlarge else 1
    return width * k, height * k


def whole_scale(picture: Picture, engine=None) -> float:
    """How large the whole image reaches the model, as a fraction of its own
    size (1 when it fits)."""
    sw, sh = sent_size(picture.width, picture.height, engine)
    return min(sw / float(picture.width), sh / float(picture.height))


def tile_count(picture: Picture, engine=None) -> int:
    """Tiles per side for ``tiles="auto"``: 1 when the whole image reaches
    the model at :data:`WHOLE_MIN_SCALE` of its size or more, else enough
    that each tile goes at about its own resolution (at most
    ``vision_view.MAX_TILES_PER_SIDE``)."""
    from funhouse_agent import vision_view
    scale = whole_scale(picture, engine)
    if scale >= WHOLE_MIN_SCALE:
        return 1
    return max(2, min(vision_view.MAX_TILES_PER_SIDE,
                      int(math.ceil(1.0 / max(scale, 1e-6) - 1e-9))))


def tile_boxes(picture: Picture, n: int) -> List[List[int]]:
    """``n`` x ``n`` overlapping tiles over the image (``vision_view.tile_boxes``),
    in whole pixels of the original, row-major from the top-left."""
    from funhouse_agent import vision_view
    out = []
    for b in vision_view.tile_boxes((0, 0, picture.width, picture.height), n):
        out.append(whole_box(b, picture))
    return out


def whole_box(box: Sequence[float], picture: Picture) -> List[int]:
    """``box`` (the image's pixels) as whole pixels, rounded OUT and held to
    the image; at least one pixel each way."""
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    ix0 = max(0, min(picture.width - 1, int(math.floor(x0))))
    iy0 = max(0, min(picture.height - 1, int(math.floor(y0))))
    ix1 = max(ix0 + 1, min(picture.width, int(math.ceil(x1))))
    iy1 = max(iy0 + 1, min(picture.height, int(math.ceil(y1))))
    return [ix0, iy0, ix1, iy1]


def zoom_box(picture: Picture, window: Optional[Sequence[float]],
             pad_frac: float = 0.0) -> Tuple[List[int], bool]:
    """``(crop, outside)``: the crop a zoom on ``window`` (the image's
    pixels; ``None`` = the whole image) cuts - padded by ``pad_frac`` the way
    planlens pads a PDF zoom (``planlens.ir.render.clip_rect_for_bbox``),
    held to the image, at least :data:`MIN_WINDOW_PX` each way - and
    whether the window missed the image on an axis (that axis then spans
    the whole image)."""
    W, H = picture.width, picture.height
    if window is None:
        return [0, 0, W, H], False
    x0, y0, x1, y1 = (float(v) for v in window)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    outside = x1 <= 0 or x0 >= W or y1 <= 0 or y0 >= H
    from planlens.ir.render import clip_rect_for_bbox
    cx0, cy0, cx1, cy1 = clip_rect_for_bbox((x0, y0, x1, y1),
                                            (0.0, 0.0, float(W), float(H)),
                                            float(pad_frac))

    def at_least(a, b, size):
        if b - a >= min(MIN_WINDOW_PX, size):
            return a, b
        mid, half = (a + b) / 2.0, min(MIN_WINDOW_PX, size) / 2.0
        a, b = mid - half, mid + half
        if a < 0:
            a, b = 0.0, b - a
        if b > size:
            a, b = a - (b - size), float(size)
        return max(0.0, a), min(float(size), b)

    cx0, cx1 = at_least(cx0, cx1, W)
    cy0, cy1 = at_least(cy0, cy1, H)
    return whole_box((cx0, cy0, cx1, cy1), picture), outside


def render(picture: Picture, box: Optional[Sequence[int]] = None, engine=None,
           enlarge: bool = False, turn: Optional[int] = None,
           turn_from: str = "asked", marks: Optional[Sequence] = None,
           allow_jpeg: bool = False) -> Tuple[bytes, Dict[str, Any]]:
    """``(image bytes, info)`` for a look at ``box`` (whole pixels of the
    original; ``None`` = the whole image): cut from the original, sized by
    :func:`sent_size` (``enlarge`` for a zoom or a tile), with ``marks``
    ([x, y, label] in the original's pixels) drawn as numbered circles, and
    turned ``turn`` degrees clockwise. ``info`` is shaped like
    ``vision_view.render_view``'s, so every helper that reads a page look
    reads this one: ``clip`` (the view, in the original's pixels),
    ``width_px`` / ``height_px`` (the image AS SENT), ``format``,
    ``budget``, ``turn`` / ``turn_from``, and ``scale`` (sent pixels per
    original pixel, before any turn). The whole image, when it fits and
    nothing is drawn on it, is the file's own bytes."""
    from PIL import Image
    from funhouse_agent import vision_view
    W, H = picture.width, picture.height
    x0, y0, x1, y1 = [0, 0, W, H] if box is None else [int(v) for v in box]
    cw, ch = x1 - x0, y1 - y0
    sw, sh = sent_size(cw, ch, engine, enlarge=enlarge)
    quarter = int(turn or 0) % 360
    quarter = quarter if quarter in _CLOCKWISE else 0
    whole = (x0, y0, x1, y1) == (0, 0, W, H)
    if whole and (sw, sh) == (W, H) and picture.as_is and not quarter \
            and not marks:
        data, fmt = picture.data, picture.kind
        out_w, out_h = W, H
    else:
        im = picture.image if whole else picture.image.crop((x0, y0, x1, y1))
        if (sw, sh) != (cw, ch):
            im = im.resize((sw, sh), getattr(getattr(Image, "Resampling",
                                                     Image), "LANCZOS"))
        if marks:
            im = _draw_marks(im, marks, (x0, y0), (sw / float(cw),
                                                   sh / float(ch)))
        if quarter:
            im = im.transpose(getattr(getattr(Image, "Transpose", Image),
                                      _CLOCKWISE[quarter]))
        data, fmt = _encode(im, picture.kind, allow_jpeg)
        out_w, out_h = im.size
    info: Dict[str, Any] = {"clip": [x0, y0, x1, y1], "width_px": int(out_w),
                            "height_px": int(out_h), "format": fmt,
                            "scale": round(min(sw / float(cw),
                                               sh / float(ch)), 4)}
    if vision_view.budget(engine) is not None:
        info["budget"] = vision_view.budget_name(engine)
    if quarter:
        info["turn"], info["turn_from"] = quarter, turn_from
    return data, info


def _encode(im, kind: str, allow_jpeg: bool) -> Tuple[bytes, str]:
    """A photo as JPEG where the engine takes it, everything else as PNG."""
    buf = io.BytesIO()
    if kind == "jpeg" and allow_jpeg:
        if im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        return buf.getvalue(), "jpeg"
    im.save(buf, format="PNG")
    return buf.getvalue(), "png"


def _draw_marks(im, marks: Sequence, origin: Tuple[int, int],
                scale: Tuple[float, float]):
    """Numbered red circles at ``marks`` ([x, y, label], the original's
    pixels) on a crop whose top-left is ``origin`` and which was resized by
    ``scale`` (x, y) - the set-of-marks prompt planlens draws on a page."""
    from PIL import ImageDraw, ImageFont
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.mode else "RGB")
    draw = ImageDraw.Draw(im)
    r = max(6, int(round(0.012 * max(im.size))))
    try:
        font = ImageFont.load_default(size=max(10, int(round(r * 1.6))))
    except TypeError:                 # Pillow older than 10.1
        font = ImageFont.load_default()
    for i, m in enumerate(marks):
        try:
            x, y = float(m[0]), float(m[1])
        except (TypeError, ValueError, IndexError):
            continue
        label = str(m[2]) if len(m) >= 3 else str(i + 1)
        cx = (x - origin[0]) * scale[0]
        cy = (y - origin[1]) * scale[1]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0),
                     width=max(2, r // 4))
        draw.text((cx + r + 2, cy - r - 2), label, fill=(255, 0, 0),
                  font=font)
    return im


#: The longest page side, in points, every PDF viewer opens (Acrobat's
#: limit, 200 inches). An image longer than this in pixels goes on its page
#: at less than one point a pixel (:func:`page_scale`).
MAX_PAGE_PT = 14400.0


def shown_size(data: bytes) -> Optional[Tuple[int, int]]:
    """``(width, height)`` of an image file as a viewer shows it (its EXIF
    orientation applied), read from its header without decoding it;
    ``None`` when Pillow cannot read it."""
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(bytes(data)))
        w, h = (int(v) for v in im.size)
        try:
            orientation = int(im.getexif().get(_EXIF_ORIENTATION) or 1)
        except Exception:  # noqa: BLE001 - no EXIF is upright
            orientation = 1
    except Exception:  # noqa: BLE001 - not readable here
        return None
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def page_scale(picture: Picture) -> float:
    """Points a pixel on the page :func:`as_pdf` makes of ``picture``: one,
    unless the image is longer than :data:`MAX_PAGE_PT` pixels."""
    side = max(picture.width, picture.height)
    return 1.0 if side <= MAX_PAGE_PT else MAX_PAGE_PT / float(side)


def as_pdf(picture: Picture) -> bytes:
    """A one-page PDF of the image as a viewer shows it (EXIF orientation
    applied), its page the image at :func:`page_scale` points a pixel --
    one point a pixel, so a position in the image's own pixels is the same
    number on the page (BACKLOG VIS-26).

    PyMuPDF opens an image file as a page sized by the resolution the file
    declares: none declared is 96 dpi, 0.75 pt a pixel; a 300 dpi scan is
    0.24 pt a pixel. A box in the image's pixels put on that page lands at
    that fraction of the spot, or off the page (live wave w4, F39: a ring
    for a phone photo drawn past the page's right edge, twice). The image
    goes in as the file's own bytes when it can (a PNG or JPEG that is not
    turned), else re-encoded upright: a JPEG photo as JPEG, anything else as
    PNG."""
    import pymupdf
    s = page_scale(picture)
    if picture.as_is and picture.kind in ("png", "jpeg"):
        stream = picture.data
    else:
        buf = io.BytesIO()
        im = picture.image
        if picture.kind == "jpeg":
            if im.mode not in ("L", "RGB"):
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=95)
        else:
            im.save(buf, format="PNG")
        stream = buf.getvalue()
    doc = pymupdf.open()
    try:
        page = doc.new_page(width=picture.width * s,
                            height=picture.height * s)
        page.insert_image(page.rect, stream=stream, keep_proportion=False)
        return doc.tobytes(garbage=1, deflate=True)
    finally:
        doc.close()


#: Said on every image look about the frame its boxes are in.
FRAME_NOTE = ("view, bbox and every box converted from an image_box are in "
              "this image's own pixels (top-left origin, x right, y down); "
              "render_region takes them with attachment_key = this image")


def precision_note(view: Sequence[float], scale: float) -> str:
    """How far a box read off this look of an image can be trusted, and
    that a box is not a measurement (``vision_view.NOT_A_MEASUREMENT``)."""
    from funhouse_agent import vision_view
    x0, y0, x1, y1 = (float(v) for v in view)
    vw, vh = x1 - x0, y1 - y0
    if scale >= 1.0:
        return (f"a box read off this {vw:.0f} x {vh:.0f} px view, sent at "
                f"full resolution or larger, is good to a few of the "
                f"image's pixels. {vision_view.NOT_A_MEASUREMENT}")
    ex, ey = vision_view.location_error(view)
    return (f"a box read off this {vw:.0f} x {vh:.0f} px view, sent shrunk "
            f"to {scale:.2f} of its size, can be up to ~{ex:.0f} x {ey:.0f} "
            f"px from the thing: good for finding where to zoom "
            f"(render_region with this view + image_box pads its window by "
            f"that much). {vision_view.NOT_A_MEASUREMENT}")


def view_payload(info: Dict[str, Any], picture: Picture,
                 engine=None) -> Dict[str, Any]:
    """The fields an image look carries so the agent can zoom on it: the
    page look's (``vision_view.view_payload``), with the precision and turn
    said for an image, the image's own size, the frame, and the scale it
    was sent at when it was not its own size."""
    from funhouse_agent import vision_view
    out = vision_view.view_payload(info, engine)
    out["view"] = [int(v) for v in info["clip"]]
    out["image_px"] = [picture.width, picture.height]
    out["frame"] = FRAME_NOTE
    scale = float(info.get("scale") or 1.0)
    out["precision"] = precision_note(info["clip"], scale)
    if scale != 1.0:
        out["sent_scale"] = round(scale, 3)
    quarter = int(info.get("turn") or 0)
    if quarter:
        out["turned"] = vision_view.turned_note(quarter, info.get("turn_from"),
                                                what="image")
    return out


def frame_of(picture: Picture):
    """A text-layer stand-in that carries only the image's size: what the
    turn memory weighs a look by (its share of the image)."""
    from funhouse_agent import vision_view
    return vision_view.TextLayer([], float(picture.width),
                                 float(picture.height))


__all__ = ["MAX_UPSCALE", "WHOLE_MIN_SCALE", "MIN_WINDOW_PX", "JPEG_QUALITY",
           "Picture", "open_picture", "sent_size", "whole_scale",
           "tile_count", "tile_boxes", "whole_box", "zoom_box", "render",
           "MAX_PAGE_PT", "shown_size", "page_scale", "as_pdf",
           "FRAME_NOTE", "precision_note", "view_payload", "frame_of"]
