"""Notes on a COPY of a DXF drawing, written by a CAD library (live smoke
wave 2c, E4).

F40: asked to "put a note at B-2" on a DXF, the model had no tool for it, so
it retyped the whole 19 KB file through ``save_file`` -- twice, 70 s and
66 s -- and the first copy silently wrote group code `` 43`` for ``143``. A
megabyte drawing cannot be done that way at all. :func:`add_notes` reads the
drawing with ezdxf, adds each note as TEXT (MTEXT for several lines) on a
markup layer of its own, with an optional leader to the point it is about,
writes the drawing back in its own format and line endings, and reads the
result again to check that every original entity is still there.
"""

from __future__ import annotations

import io
import os
import statistics
import tempfile
from collections import Counter
from typing import Any, Dict, List, Tuple

#: The layer notes go on unless the caller names another.
DEFAULT_LAYER = "REVIEW"
#: Its colour (ACI 1 = red), so the notes stand out from the drawing.
LAYER_COLOR = 1
#: Notes one call adds at most.
MAX_NOTES = 200
#: Characters per note at most.
MAX_NOTE_CHARS = 2000


class DxfNoteError(ValueError):
    """A note that cannot be added, or a drawing that cannot be read or
    written, in words fit to show."""


def available() -> bool:
    try:
        import ezdxf  # noqa: F401
    except ImportError:
        return False
    return True


def _read(data: bytes):
    """The drawing in ``data`` (ASCII or binary DXF, any encoding the file
    declares), read by ezdxf from a temporary file."""
    import ezdxf
    fd, tmp = tempfile.mkstemp(suffix=".dxf", prefix="gse_dxf_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return ezdxf.readfile(tmp)
    except Exception as exc:  # noqa: BLE001 - ezdxf's DXFStructureError etc.
        raise DxfNoteError(f"the file could not be read as a DXF drawing: "
                           f"{type(exc).__name__}: {exc}") from exc
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _counts(doc) -> Counter:
    """Entity types in every layout (model space and the paper spaces)."""
    c: Counter = Counter()
    for layout in doc.layouts:
        c.update(e.dxftype() for e in layout)
    return c


def _number(v, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise DxfNoteError(f"{what} must be a number, got {v!r}") from None
    if f != f or f in (float("inf"), float("-inf")):
        raise DxfNoteError(f"{what} must be a finite number")
    return f


def _point(v, what: str) -> Tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        raise DxfNoteError(f"{what} must be [x, y] in drawing units")
    return _number(v[0], f"{what}[0]"), _number(v[1], f"{what}[1]")


def _parse(notes) -> List[Dict[str, Any]]:
    if not isinstance(notes, list) or not notes:
        raise DxfNoteError("notes must be a non-empty list of "
                           "{text, x, y} (drawing units)")
    if len(notes) > MAX_NOTES:
        raise DxfNoteError(f"at most {MAX_NOTES} notes a call")
    out = []
    for i, n in enumerate(notes):
        if not isinstance(n, dict):
            raise DxfNoteError(f"note {i} must be an object with text, x, y")
        text = str(n.get("text") or n.get("comment") or "").strip()
        if not text:
            raise DxfNoteError(f"note {i} has no text")
        if len(text) > MAX_NOTE_CHARS:
            raise DxfNoteError(f"note {i} is longer than {MAX_NOTE_CHARS} "
                               f"characters")
        if n.get("at") is not None:
            x, y = _point(n["at"], f"note {i} at")
        else:
            x = _number(n.get("x"), f"note {i} x")
            y = _number(n.get("y"), f"note {i} y")
        leader = (_point(n["leader_to"], f"note {i} leader_to")
                  if n.get("leader_to") is not None else None)
        height = (_number(n["height"], f"note {i} height")
                  if n.get("height") not in (None, "") else None)
        if height is not None and height <= 0:
            raise DxfNoteError(f"note {i} height must be above 0")
        out.append({"text": text, "x": x, "y": y, "leader_to": leader,
                    "height": height})
    return out


def _typical_height(doc) -> Tuple[float, str]:
    """The drawing's usual lettering height, and how it was found."""
    hs = []
    for e in doc.modelspace():
        t = e.dxftype()
        try:
            if t == "TEXT":
                hs.append(float(e.dxf.height))
            elif t == "MTEXT":
                hs.append(float(e.dxf.char_height))
        except Exception:  # noqa: BLE001 - an entity without the attribute
            continue
    hs = [h for h in hs if h > 0]
    if hs:
        return float(statistics.median(hs)), "the drawing's usual text height"
    try:
        lo, hi = doc.header["$EXTMIN"], doc.header["$EXTMAX"]
        diag = ((hi[0] - lo[0]) ** 2 + (hi[1] - lo[1]) ** 2) ** 0.5
        if diag > 0 and diag < 1e12:
            return diag / 150.0, "1/150 of the drawing's extent"
    except Exception:  # noqa: BLE001
        pass
    return 2.5, "a default (the drawing has no text or extent to go by)"


def _write(doc, original: bytes) -> bytes:
    """The drawing as bytes in the original's format and line endings."""
    if original[:22].startswith(b"AutoCAD Binary DXF"):
        bio = io.BytesIO()
        doc.write(bio, fmt="bin")
        return bio.getvalue()
    sio = io.StringIO()
    doc.write(sio)
    text = sio.getvalue()
    if b"\r\n" in original[:4096]:
        text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    return doc.encode(text)


def add_notes(data: bytes, notes, layer: str = DEFAULT_LAYER
              ) -> Tuple[bytes, Dict[str, Any]]:
    """``(new_bytes, report)``: the drawing in ``data`` with ``notes`` added
    on ``layer``. Each note is ``{text, x, y}`` (or ``at: [x, y]``) in the
    drawing's own units and coordinates, with optional ``leader_to: [x, y]``
    (a leader from the note to that point) and ``height``. Raises
    :class:`DxfNoteError`."""
    parsed = _parse(notes)
    layer = str(layer or DEFAULT_LAYER).strip() or DEFAULT_LAYER
    doc = _read(bytes(data))
    before = _counts(doc)
    msp = doc.modelspace()
    if not doc.layers.has_entry(layer):
        doc.layers.add(layer, color=LAYER_COLOR)
    typical, how = _typical_height(doc)
    dimstyle = next((s for s in ("Standard", "STANDARD", "EZDXF")
                     if doc.dimstyles.has_entry(s)), None)
    added: Counter = Counter()
    for n in parsed:
        h = n["height"] or typical
        attribs = {"layer": layer, "insert": (n["x"], n["y"])}
        if "\n" in n["text"]:
            m = msp.add_mtext(n["text"].replace("\r\n", "\n")
                              .replace("\n", "\\P"), dxfattribs=attribs)
            m.dxf.char_height = h
            added["MTEXT"] += 1
        else:
            msp.add_text(n["text"], height=h, dxfattribs=attribs)
            added["TEXT"] += 1
        if n["leader_to"] is not None:
            tip = n["leader_to"]
            if dimstyle is not None:
                msp.add_leader([tip, (n["x"], n["y"])], dimstyle=dimstyle,
                               dxfattribs={"layer": layer})
                added["LEADER"] += 1
            else:
                msp.add_line(tip, (n["x"], n["y"]),
                             dxfattribs={"layer": layer})
                added["LINE"] += 1
    try:
        out = _write(doc, bytes(data))
    except Exception as exc:  # noqa: BLE001
        raise DxfNoteError(f"the marked copy could not be written: "
                           f"{type(exc).__name__}: {exc}") from exc
    # Read the copy back: every original entity is still there, and the
    # notes are on their layer.
    check = _read(out)
    after = _counts(check)
    expected = before + added
    lost = {k: expected[k] - after.get(k, 0) for k in expected
            if after.get(k, 0) < expected[k]}
    on_layer = [e for e in check.modelspace()
                if e.dxftype() in ("TEXT", "MTEXT") and e.dxf.layer == layer]
    if lost or len(on_layer) < added["TEXT"] + added["MTEXT"]:
        raise DxfNoteError(
            f"the marked copy did not read back whole (missing: "
            f"{lost or 'notes on ' + layer}); nothing was saved")
    report = {
        "notes_added": len(parsed), "layer": layer,
        "entities_added": dict(added),
        "entities_before": sum(before.values()),
        "entities_after": sum(after.values()),
        "text_height": round(typical, 4) if not any(
            n["height"] for n in parsed) else "as given",
        "check": (f"read back with a CAD library: all "
                  f"{sum(before.values())} original entities are kept, "
                  f"and the {len(parsed)} note(s) are on layer '{layer}'"),
    }
    if not any(n["height"] for n in parsed):
        report["text_height_from"] = how
    if any(n["leader_to"] for n in parsed) and dimstyle is None:
        report["leader_note"] = ("the drawing has no dimension style for a "
                                 "leader, so each pointer is a plain line")
    return out, report
