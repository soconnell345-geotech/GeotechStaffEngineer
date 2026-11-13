"""
The lettering of a drawing sheet: one font the code always has.

ezdxf's drawing add-on takes fonts from the system's font folders, and the
app's hosts may have none. matplotlib (a required package) ships DejaVu
Sans as plain TTF files, so those files are registered with ezdxf's font
manager, in this process only, before anything is measured or rendered:
the sheet's text styles name ``DejaVuSans.ttf`` / ``DejaVuSans-Bold.ttf``
and resolve to matplotlib's copies whatever the host holds. Nothing is
written: ezdxf's own font cache file (``~/.cache/ezdxf``) is neither
needed nor touched by this step, and ezdxf already carries on when it
cannot write that file (a read-only home folder).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

#: The sheet's fonts (file names, as a DXF text style names them).
REGULAR = "DejaVuSans.ttf"
BOLD = "DejaVuSans-Bold.ttf"

#: The text styles' width factor: lettering a little narrower than the
#: font draws it, as drawing lettering is.
WIDTH_FACTOR = 0.85

_registered = False


def font_folder() -> Path:
    """The folder of TTF fonts matplotlib ships."""
    import matplotlib
    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def ensure_fonts() -> None:
    """Register matplotlib's DejaVu Sans with ezdxf's font manager (once a
    process). Raises ``RuntimeError`` if the files are not there."""
    global _registered
    if _registered:
        return
    from ezdxf.fonts import fonts
    from ezdxf.fonts.font_manager import get_ttf_font_face

    manager = fonts.font_manager
    folder = font_folder()
    for name in (REGULAR, BOLD):
        path = folder / name
        if not path.is_file():
            raise RuntimeError(
                f"The drawing font {name} was not found in matplotlib's "
                f"font folder ({folder}); a sheet cannot be lettered.")
        # Our copy wins over a system copy of the same file name, so the
        # same description draws the same sheet on every host. The font
        # manager has no public call that adds one file; add_entry on its
        # cache is what its own folder scan calls per file.
        try:
            manager._font_cache.add_entry(path, get_ttf_font_face(path))
            manager._loaded_ttf_fonts.pop(name, None)
        except AttributeError:          # a later ezdxf: scan the folder
            manager.scan_folder(folder)
    _registered = True


@lru_cache(maxsize=8)
def _font(name: str):
    ensure_fonts()
    from ezdxf.fonts import fonts
    return fonts.make_font(name, 1.0)


def text_width(text: str, height: float, bold: bool = False) -> float:
    """The width of ``text`` set at cap height ``height`` in the sheet's
    style, in the units of ``height``."""
    if not text:
        return 0.0
    return (_font(BOLD if bold else REGULAR).text_width(text) * height
            * WIDTH_FACTOR)


def wrap(text: str, width: float, height: float,
         bold: bool = False) -> List[str]:
    """``text`` broken at spaces into lines no wider than ``width``; a
    single word wider than that is broken where it reaches the width (it
    used to keep a line of its own and run past its column, and a long
    enough one slid the whole sheet off the page: GM-313 review A1)."""
    lines: List[str] = []
    line = ""
    for word in text.split():
        while text_width(word, height, bold) > width and len(word) > 1:
            if line:
                lines.append(line)
                line = ""
            low, high = 1, len(word) - 1    # the longest head that fits
            while low < high:
                mid = (low + high + 1) // 2
                if text_width(word[:mid], height, bold) <= width:
                    low = mid
                else:
                    high = mid - 1
            cut = low
            lines.append(word[:cut])
            word = word[cut:]
        trial = f"{line} {word}" if line else word
        if line and text_width(trial, height, bold) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    return lines or [""]


def fit(text: str, width: float, height: float, bold: bool = False) -> str:
    """``text``, cut and ended with "..." if wider than ``width``."""
    if text_width(text, height, bold) <= width:
        return text
    while text and text_width(text + "...", height, bold) > width:
        text = text[:-1]
    return text.rstrip() + "..."
