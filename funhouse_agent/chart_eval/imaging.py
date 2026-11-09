"""Image sizing and the zoom crop for the chart suite's direct arms.

The zoom pattern is ported from Anthropic's Claude Cookbook recipe "Giving
Claude a Zoom Tool for Reading Fine Image Detail"
(``multimodal/crop_tool.ipynb``, github.com/anthropics/claude-cookbooks):

* the image the model is shown is resized HERE, to exactly the size the model
  would shrink it to, so the pixel coordinates the model gives back refer to
  the image we hold;
* a zoom crops the region from the full-resolution ORIGINAL and scales the
  crop UP to the largest size the model's image limits allow;
* zooms go back as JPEG (quality 92, no chroma subsampling) so a long
  conversation of crops stays small.

What differs from the recipe: the limits are chosen by model family (Claude's
high-resolution tier, or a GPT model's edge cap), because on Foundry the same
suite runs on both.

The cookbook is MIT-licensed:

    MIT License

    Copyright (c) 2023 Anthropic

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the
    following conditions:

    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union

#: JPEG quality for zoomed crops (the cookbook's value).
JPEG_QUALITY = 92


@dataclass(frozen=True)
class ImageLimits:
    """The largest image a model reads without shrinking it: the longest
    edge in pixels, and the most patches (``patch`` x ``patch`` pixels each)
    an image may cost."""
    name: str
    max_edge: int
    max_tokens: int
    patch: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


#: Claude's high-resolution tier (Opus 5.5, Sonnet 5.5, Fable 5.x): 2576 px
#: and 4784 visual tokens of 28 x 28 px (the cookbook's values).
CLAUDE_HIRES = ImageLimits("claude-hires", 2576, 4784, 28)
#: Claude's standard tier: 1568 px and 1568 visual tokens.
CLAUDE_STANDARD = ImageLimits("claude", 1568, 1568, 28)
#: A GPT model at ``detail="original"``: 6000 px and 10,000 patches of 32 px
#: (OpenAI's published limits), held to the app's host cap (2048 px, the
#: largest edge every host the app runs on delivers unshrunk; see
#: ``funhouse_agent.vision_view.DEFAULT_MAX_PX``). Raise it with
#: ``image_limits=`` where the host is known to deliver more.
OPENAI_ORIGINAL = ImageLimits("openai-original", 6000, 10000, 32)

NAMED_LIMITS = {lim.name: lim for lim in (CLAUDE_HIRES, CLAUDE_STANDARD,
                                          OPENAI_ORIGINAL)}

_CLAUDE_WORDS = ("claude", "opus", "sonnet", "haiku", "fable", "anthropic")


def family_of(model: Any) -> str:
    """``"anthropic"`` or ``"openai"`` for a LangChain chat model: by its
    class name, else the model name it carries (a Foundry RID or API name
    states it). Anything not recognisably Claude is treated as OpenAI-shaped,
    as the app's own routing does."""
    names = [type(model).__name__]
    for attr in ("model", "model_name", "model_id", "model_api_name",
                 "deployment_name"):
        val = getattr(model, attr, None)
        if isinstance(val, str):
            names.append(val)
    text = " ".join(names).lower()
    return "anthropic" if any(w in text for w in _CLAUDE_WORDS) else "openai"


def host_max_edge() -> int:
    """The app's host cap on an image's long edge (``GEOTECH_VISION_MAX_PX``,
    default 2048)."""
    try:
        from funhouse_agent import vision_view
        raw = os.environ.get(vision_view.MAX_PX_ENV, "")
        if raw.strip().lower() in ("none", "off", "0"):
            return OPENAI_ORIGINAL.max_edge
        if raw.strip():
            return max(256, int(float(raw)))
        return int(vision_view.DEFAULT_MAX_PX)
    except Exception:                                     # noqa: BLE001
        return 2048


def limits_for(model: Any, override: Union[None, str, ImageLimits,
                                           Tuple[int, int, int]] = None
               ) -> ImageLimits:
    """The image limits to size for: ``override`` (a name in
    :data:`NAMED_LIMITS`, an :class:`ImageLimits`, or ``(max_edge,
    max_tokens, patch)``), else by the model's family."""
    if isinstance(override, ImageLimits):
        return override
    if isinstance(override, str) and override.strip():
        if override not in NAMED_LIMITS:
            raise ValueError(f"unknown image limits {override!r}; known: "
                             f"{sorted(NAMED_LIMITS)}")
        lim = NAMED_LIMITS[override]
    elif isinstance(override, (tuple, list)) and len(override) == 3:
        edge, tokens, patch = (int(v) for v in override)
        return ImageLimits("custom", edge, tokens, patch)
    elif family_of(model) == "anthropic":
        return CLAUDE_HIRES
    else:
        lim = OPENAI_ORIGINAL
    if lim.name == OPENAI_ORIGINAL.name:
        cap = host_max_edge()
        if cap < lim.max_edge:
            lim = ImageLimits(f"{lim.name}@{cap}px", cap, lim.max_tokens,
                              lim.patch)
    return lim


def count_image_tokens(width: int, height: int, patch: int = 28) -> int:
    """Patches an image costs: one per ``patch`` x ``patch`` pixels."""
    return math.ceil(width / patch) * math.ceil(height / patch)


def resized_size(width: int, height: int,
                 limits: ImageLimits = CLAUDE_HIRES) -> Tuple[int, int]:
    """The size the model resizes an image to (unchanged if it fits): the
    largest aspect-preserving size within ``limits`` (the cookbook's
    ``resized_size``)."""
    edge, tokens, patch = limits.max_edge, limits.max_tokens, limits.patch

    def fits(w: int, h: int) -> bool:
        return (math.ceil(w / patch) * patch <= edge
                and math.ceil(h / patch) * patch <= edge
                and count_image_tokens(w, h, patch) <= tokens)

    if fits(width, height):
        return (width, height)
    if height > width:
        rh, rw = resized_size(height, width, limits)
        return (rw, rh)
    aspect = width / height
    lo, hi = 1, width                       # lo always fits; hi never fits
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return (lo, max(round(lo / aspect), 1))


def zoom_size(width: int, height: int,
              limits: ImageLimits = CLAUDE_HIRES) -> Tuple[int, int]:
    """The largest aspect-preserving size within ``limits``: the upscale
    target for a zoomed crop."""
    return resized_size(width * 10_000, height * 10_000, limits)


def open_image(path_or_bytes: Union[str, bytes]):
    """A PIL image, flattened onto white (a transparent chart background
    would otherwise turn black)."""
    from PIL import Image
    if isinstance(path_or_bytes, (bytes, bytearray)):
        img = Image.open(io.BytesIO(bytes(path_or_bytes)))
    else:
        img = Image.open(path_or_bytes)
    img.load()
    return flatten(img)


def flatten(img):
    from PIL import Image
    if img.mode == "RGB":
        return img
    rgba = img.convert("RGBA")
    background = Image.new("RGB", rgba.size, "white")
    background.paste(rgba, mask=rgba.getchannel("A"))
    return background


def prepare_image(img, limits: ImageLimits = CLAUDE_HIRES):
    """``img`` resized to exactly the size the model will see."""
    from PIL import Image
    target = resized_size(img.width, img.height, limits)
    if (img.width, img.height) == target:
        return img
    return img.resize(target, Image.Resampling.LANCZOS)


def png_bytes(img) -> bytes:
    buf = io.BytesIO()
    flatten(img).save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(img, quality: int = JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    flatten(img).save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def data_uri(data: bytes) -> str:
    kind = "image/jpeg" if data[:3] == b"\xff\xd8\xff" else "image/png"
    return f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"


@dataclass
class ZoomResult:
    """What one zoom call gives back."""
    ok: bool
    text: str
    image: Optional[bytes] = None
    crop_px: Optional[Tuple[int, int, int, int]] = None
    zoomed_px: Optional[Tuple[int, int]] = None


def zoom(original, view_size: Tuple[int, int], x1: Any, y1: Any, x2: Any,
         y2: Any, limits: ImageLimits = CLAUDE_HIRES) -> ZoomResult:
    """One zoom: ``(x1, y1)-(x2, y2)`` in pixels of the image the model saw
    (``view_size``), cropped from ``original`` and magnified to fill
    ``limits`` (the cookbook's ``handle_zoom``)."""
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in (x1, y1, x2, y2))
    except (TypeError, ValueError):
        return ZoomResult(False, "Error: x1, y1, x2 and y2 must be numbers "
                                 "(pixels)")
    view_w, view_h = view_size
    x1, x2 = max(0, min(x1, view_w)), max(0, min(x2, view_w))
    y1, y2 = max(0, min(y1, view_h)), max(0, min(y2, view_h))
    if x1 >= x2 or y1 >= y2:
        return ZoomResult(False, "Error: invalid region (need x1 < x2 and "
                                 "y1 < y2, within the image)")
    # The resize kept the aspect ratio, so one scale covers both axes.
    scale = original.width / view_w
    box = (round(x1 * scale), round(y1 * scale), round(x2 * scale),
           round(y2 * scale))
    box = (box[0], box[1], max(box[2], box[0] + 1), max(box[3], box[1] + 1))
    from PIL import Image
    cropped = original.crop(box)
    zoomed = cropped.resize(zoom_size(cropped.width, cropped.height, limits),
                            Image.Resampling.LANCZOS)
    text = (f"Zoomed into ({x1},{y1})-({x2},{y2}) of the image you see at "
            f"{view_w}x{view_h}px: a {cropped.width}x{cropped.height}px "
            f"region of the original, returned magnified to "
            f"{zoomed.width}x{zoomed.height}px.")
    return ZoomResult(True, text, jpeg_bytes(zoomed), box,
                      (zoomed.width, zoomed.height))


__all__ = ["ImageLimits", "CLAUDE_HIRES", "CLAUDE_STANDARD",
           "OPENAI_ORIGINAL", "NAMED_LIMITS", "family_of", "limits_for",
           "count_image_tokens", "resized_size", "zoom_size", "open_image",
           "prepare_image", "png_bytes", "jpeg_bytes", "data_uri", "zoom",
           "ZoomResult"]
