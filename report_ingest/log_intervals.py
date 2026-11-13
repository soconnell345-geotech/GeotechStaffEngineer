"""Every drawn line an interval, written changes as notes, stratum labels
as printed (BACKLOG RI-103, parts 1 and 2).

A boring log's soil column is cut by LINES the draughtsman drew, and between
two lines is one interval. Most firms draw a line at every change of
description; some draw lines only at their strata and WRITE the smaller
changes inside the interval ("gray below 3.5 ft") with no line. Some print a
stratum or unit label against an interval; most do not. What the record
keeps under this module is what the log shows, and nothing the reader
decided for it:

* every drawn line is an interval boundary, and two intervals are never
  joined, whatever their descriptions or classifications;
* a change written inside an interval with no line is a NOTE on that
  interval (:class:`report_ingest.model.LayerNote`), with its depth where
  one is written; it is never an interval, as a drawn line is never a note;
* the kind of an interval's top line is kept only where the log's legend or
  the line's own label says it (never from the look of the line alone), and
  whether the line runs the full width of the column;
* a stratum label is kept only where the log prints one, as printed, with
  where it was printed, and marked ``carried`` where the log prints it once
  and plainly runs it down over the intervals below.

Mapping each interval to the narrative's units (RI-103 part 3) is a later
model step and is not here. Each interval stays addressable for it: an
interval is one entry of ``Investigation.layers``, in the log's order, with
its own top depth, and nothing here joins or reorders them.

ALL OF IT SITS BEHIND ONE SWITCH, :data:`LOG_INTERVALS_ENV`, off by default,
on the model-first log path only (:func:`report_ingest.model_first_read.
read_logs`). Off, the log instruction and the log answer schema are byte for
byte what they were and :data:`report_ingest.model_first.READ_PROMPT_VERSION`
is not raised. On, ``model_first.prompt_version("logs")`` carries
:data:`VERSION_SUFFIX`, so a resumed run never takes a log answer read with
the switch the other way as current. Today's log reader and the floor are
not changed.

MODEL-FACING: :data:`MODEL_FIRST_PARAGRAPH` (added to the model-first log
instruction) and the answer schema (:class:`LogReading`, with
:class:`ReadLayer` and :class:`ReadIntervalNote`). The classes keep the
names the model reads in the answer tool ("Your answer, as a LogReading").

THE RECORD: :func:`apply_interval_additions` puts what the reading adds onto
the layers today's builder made (the builder reads the fields it knows and
ignores the rest, as with ``model_first.apply_log_additions``), and
:func:`carried_label_entries` lists a label marked carried that no interval
above prints, once the log's calls are joined.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest import log_reader as _log
from report_ingest import model_first as _mf
from report_ingest.model import (Investigation, Layer, LayerNote, Provenance,
                                 Quantity)

__all__ = [
    "LOG_INTERVALS_ENV", "log_intervals_enabled", "use_log_intervals",
    "VERSION_SUFFIX", "MODEL_FIRST_PARAGRAPH", "ReadIntervalNote",
    "ReadLayer", "LogReading", "ADDED_LAYER_FIELDS",
    "apply_interval_additions", "carried_label_entries", "same_label",
]

# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

#: Off by default: the model-first log reading asks for every drawn line as
#: an interval, written changes as notes and stratum labels as printed.
LOG_INTERVALS_ENV = "GEOTECH_INGEST_LOG_INTERVALS"
_FORCED: ContextVar[Optional[bool]] = ContextVar(
    "ingest_log_intervals", default=None)

#: What ``model_first.prompt_version("logs")`` carries while the switch is
#: on.
VERSION_SUFFIX = "+li"


def log_intervals_enabled() -> bool:
    """Whether the model-first log reading is asked for intervals, notes and
    stratum labels (default: off)."""
    forced = _FORCED.get()
    if forced is not None:
        return forced
    return (os.environ.get(LOG_INTERVALS_ENV) or "").strip().lower() in (
        "1", "on", "true", "yes")


@contextmanager
def use_log_intervals(on: bool = True) -> Iterator[None]:
    """Turn the interval reading on (or off) inside the block."""
    token = _FORCED.set(bool(on))
    try:
        yield
    finally:
        _FORCED.reset(token)


# ---------------------------------------------------------------------------
# what the model is told
# ---------------------------------------------------------------------------

#: Added to the model-first log instruction, after "say which unit.", when
#: the switch is on. Written for any form, in the terms a log legend uses:
#: what a drawn line is, what a written change is, and that neither is ever
#: joined to or made into the other.
MODEL_FIRST_PARAGRAPH = (
    "A layer in this answer is an INTERVAL: the stretch of the soil column "
    "between two lines the log draws across it. Give one layer for every "
    "interval, from one drawn line to the next, with its depths, the "
    "elevation of its top where the log "
    "prints one, and its description and classification as printed; never "
    "join two intervals, even where their descriptions or symbols are the "
    "same. A change the log WRITES inside an interval without drawing a "
    "line (\"wet below 7 ft\", \"as above, but brown\") is a note on that "
    "interval, with its depth where one is written and its words as "
    "printed: a written change is never a layer, and a drawn line is never "
    "a note. Fill top_line only where the log's legend or the line's own "
    "label names the kind of line (solid, dashed, approximate, inferred, "
    "gradational), never from the look of the line alone. Give a "
    "stratum_label only where the log prints one (a stratum or unit column, "
    "a geologic symbol or formation name, a model-layer number, a tag such "
    "as FILL set apart from the description), as printed, with where it is "
    "printed. A label printed once that a bracket, an arrow or a merged "
    "cell plainly runs down over the intervals below goes on each of them, "
    "marked carried; where how far it runs is not plain, only on the "
    "interval it is printed against.")


# ---------------------------------------------------------------------------
# the answer schema (MODEL-FACING)
# ---------------------------------------------------------------------------

class ReadIntervalNote(BaseModel):
    """A change the log writes inside an interval without drawing a line."""

    model_config = ConfigDict(extra="forbid")

    depth: Optional[float] = Field(
        default=None,
        description="the depth the note is written for, in the log's own "
                    "unit ('wet below 7 ft' is 7); null when the note "
                    "states none ('as above, but brown')")
    text: str = Field(description="the note's words, as printed")


class ReadLayer(_log.ReadLayer):
    """One interval of the soil column, between two lines the log draws."""

    model_config = ConfigDict(extra="forbid")

    top_elevation: Optional[float] = Field(
        default=None,
        description="the elevation the log prints for the top of this "
                    "interval, in the unit of the header's elevation; null "
                    "if the log prints none. Never worked out from the "
                    "depth")
    notes: List[ReadIntervalNote] = Field(
        default_factory=list,
        description="every change the log writes inside this interval "
                    "without drawing a line; empty if none")
    top_line: str = Field(
        default="",
        description="the kind of the line at the top of this interval, in "
                    "the words of the log's legend or of the line's own "
                    "label (solid, dashed, approximate, inferred, "
                    "gradational). Empty when the log says nothing of its "
                    "lines: never from the look of the line alone")
    top_line_full_width: Optional[bool] = Field(
        default=None,
        description="true when the line at the top of this interval runs "
                    "the full width of the soil column, false when it runs "
                    "part of the way; null when you cannot tell or there is "
                    "no line (the top of the hole)")
    stratum_label: str = Field(
        default="",
        description="the stratum or unit label the log PRINTS for this "
                    "interval, as printed. Empty when it prints none. Never "
                    "work one out from the description")
    stratum_label_source: str = Field(
        default="",
        description="where the label is printed: the heading of its "
                    "column, as printed, or 'in the description'; empty "
                    "with no label")
    stratum_label_carried: bool = Field(
        default=False,
        description="true when the label is printed once above and a "
                    "bracket, an arrow or a merged cell plainly runs it "
                    "down over this interval; false when it is printed "
                    "against this interval itself")


class LogReading(_mf.LogReading):
    """One log as the model read it."""

    model_config = ConfigDict(extra="forbid")

    layers: List[ReadLayer] = Field(
        default_factory=list,
        description="one entry for every interval between two lines the "
                    "log draws across its soil column, top to bottom")


#: The fields :class:`ReadLayer` adds to today's: what the model is asked
#: under the switch that it is not asked without it.
ADDED_LAYER_FIELDS: Tuple[str, ...] = (
    "top_elevation", "notes", "top_line", "top_line_full_width",
    "stratum_label", "stratum_label_source", "stratum_label_carried")


# ---------------------------------------------------------------------------
# where the new fields go in the record
# ---------------------------------------------------------------------------

def _plain(text: Any) -> str:
    return " ".join(str(text or "").split())


def same_label(a: Any, b: Any) -> bool:
    """Whether two stratum labels are one as printed: exact after whitespace
    and case (the log scorer's rule for a label, too)."""
    return _plain(a).casefold() == _plain(b).casefold()


def _pairs(read: List[Any], built: List[Layer]) -> List[Optional[int]]:
    """For each built layer, the index of the reading's layer it was built
    from, or None.

    The builder makes one layer per layer read, in order, and leaves out
    only one whose top it could not keep, so the built layers are the read
    ones in order with some missing: the same number pairs by position;
    otherwise each built layer takes the next one read with its description
    and its top (a top just above the ground is recorded as 0, so 0 pairs
    with any top the description agrees on).
    """
    if len(read) == len(built):
        return list(range(len(built)))
    out: List[Optional[int]] = []
    cursor = 0
    for layer in built:
        found = None
        for k in range(cursor, len(read)):
            if read[k].description != layer.description:
                continue
            if float(read[k].top) == layer.top.value or layer.top.value == 0.0:
                found = k
                break
        if found is not None:
            cursor = found + 1
        out.append(found)
    return out


def _note_prov(layer: Layer) -> Optional[Provenance]:
    """A note's provenance: its interval's page and how the interval was
    read (the model gives a note no box of its own)."""
    if layer.prov is None:
        return None
    return Provenance(page=layer.prov.page, method=layer.prov.method,
                      confidence=layer.prov.confidence)


def apply_interval_additions(reading: Any, investigation: Investigation
                             ) -> Tuple[Investigation, List[Dict[str, Any]]]:
    """``investigation`` (today's log builder's record of ``reading``) with
    what :class:`ReadLayer` adds on each layer, and what could not go in.

    * ``top_elevation`` becomes a Quantity in the header's elevation unit
      (empty when the log prints none, which leaves the unit unknown, as the
      record's rule is);
    * each note becomes a :class:`~report_ingest.model.LayerNote`: its words
      as printed, its depth a Quantity in the log's depth unit with the
      interval's page. A note with no words is left out, listed. A negative
      depth is left off its note, listed with the number as read. A depth
      outside its interval is kept and listed (the note may belong to
      another interval), its confidence lowered;
    * ``top_line`` and ``top_line_full_width`` as read;
    * the stratum label as printed, with its source and its carried mark. A
      source or a carried mark with no label is dropped: there is nothing
      for it to describe.

    Nothing here joins, drops or reorders a layer. A layer of the reading
    the builder kept out (an impossible top) takes its additions with it:
    the builder's own entry keeps the whole layer as read. Returns a copy;
    ``investigation`` is not changed.
    """
    from report_ingest.checks import FLAG_CONFIDENCE

    out = investigation.model_copy(deep=True)
    unresolved: List[Dict[str, Any]] = []
    read = list(getattr(reading, "layers", None) or [])
    unit = out.depth_unit
    elevation_unit = _plain(getattr(reading, "elevation_unit", ""))
    for layer, k in zip(out.layers, _pairs(read, out.layers)):
        if k is None:
            continue
        source = read[k]
        page = layer.prov.page if layer.prov is not None else None
        what = (f"interval at {layer.top.value:g} {layer.top.unit}".strip())

        elevation = getattr(source, "top_elevation", None)
        if elevation is not None and math.isfinite(float(elevation)):
            layer.top_elevation = Quantity(
                value=float(elevation), unit=elevation_unit,
                prov=_note_prov(layer))

        notes: List[LayerNote] = []
        for note in getattr(source, "notes", None) or []:
            text = _plain(note.text)
            depth = note.depth
            if not text:
                unresolved.append({
                    "what": f"{what}: a note", "page": page,
                    "value": depth,
                    "why": "the note has no words; left out",
                    "refused_by": "python"})
                continue
            prov = _note_prov(layer)
            quantity = None
            if depth is not None:
                depth = float(depth)
                if not math.isfinite(depth) or depth < 0.0:
                    unresolved.append({
                        "what": f"{what}: note {text!r} depth",
                        "page": page, "value": depth,
                        "why": "a depth cannot be negative; the note is "
                               "kept without its depth",
                        "refused_by": "python"})
                else:
                    quantity = Quantity(value=depth, unit=unit,
                                        prov=_note_prov(layer))
                    top = layer.top.value
                    bottom = (layer.bottom.value
                              if layer.bottom is not None else None)
                    if depth < top or (bottom is not None and depth > bottom):
                        span = (f"{top:g} to {bottom:g}" if bottom is not None
                                else f"from {top:g}")
                        unresolved.append({
                            "what": f"{what}: note {text!r} depth",
                            "page": page, "value": depth,
                            "why": f"the note's depth is outside its "
                                   f"interval ({span} {unit}); kept as read: "
                                   f"it may belong to another interval",
                            "flag": "interval_note_depth"})
                        if prov is not None:
                            prov.confidence = min(prov.confidence,
                                                  FLAG_CONFIDENCE)
            notes.append(LayerNote(depth=quantity, text=text, prov=prov))
        layer.notes = notes

        layer.top_line = _plain(getattr(source, "top_line", ""))
        width = getattr(source, "top_line_full_width", None)
        layer.top_line_full_width = None if width is None else bool(width)

        label = _plain(getattr(source, "stratum_label", ""))
        layer.stratum_label = label
        layer.stratum_label_source = (
            _plain(getattr(source, "stratum_label_source", ""))
            if label else "")
        layer.stratum_label_carried = bool(
            label and getattr(source, "stratum_label_carried", False))
    return out, unresolved


def carried_label_entries(investigation: Investigation
                          ) -> List[Dict[str, Any]]:
    """One entry for each interval whose stratum label is marked carried
    though no interval above it in the log has that label: the mark says the
    label was printed once above, and nothing above prints it. The label and
    its mark are kept as read; the entry is for a person to look.

    Asked of the whole log, after its calls are joined, because a label
    printed on one sheet can run down onto the next.
    """
    rows: List[Dict[str, Any]] = []
    above: List[str] = []
    for layer in investigation.layers:
        label = layer.stratum_label
        if label and layer.stratum_label_carried and not any(
                same_label(label, seen) for seen in above):
            rows.append({
                "what": (f"interval at {layer.top.value:g} "
                         f"{layer.top.unit}".strip() + ": stratum label"),
                "page": layer.prov.page if layer.prov is not None else None,
                "read": label,
                "why": f"the label {label!r} is marked as carried down "
                       f"from an interval above, and no interval above has "
                       f"it; kept as read",
                "flag": "label_carried"})
        if label:
            above.append(label)
    return rows
