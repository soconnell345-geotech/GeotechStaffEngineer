"""The 38-report corpus, by ID, with its Azure Document Intelligence results.

PRIVACY. Everything this module reads lives under ``RAW_DIR``, which is
gitignored (``module_work/field_feedback/**/raw/``). A report is identified by
its ID (R01-R38) everywhere. :attr:`ReportInfo.private_name` holds the
manifest's source-file stem because the label-sheet matching needs it, and it
is the one field nothing here prints unless asked.

WHAT ``text_source`` ACTUALLY DOES (read before trusting ``di="all"``).
``planlens.document.Document._page_text`` asks the text source
``covers(index)`` and, when the answer is yes, takes ITS lines, blocks and
tables and ignores the PDF text layer completely. There is no merge and no
"only where the PDF has no text" rule: an :class:`AzureLayout` that covers
every page REPLACES the text on every page, including the ones whose embedded
text is exact. Two things follow from a page's source: ``classify_page`` is
told ``text_is_optical``, so page kinds can move; and ``needs_ocr`` is derived
from whatever text the page ended up with, so a page read by DI never reports
that it needs OCR.

Hence three modes, of which ``auto`` is the useful one:

``di="none"``
    PDF text layer only.
``di="all"``
    Attach the DI result for every page it covers.
``di="auto"`` (default)
    Open the PDF once with no text source, take ``pages_needing_ocr(doc)``
    (image-only pages, plus pages whose text layer is present but unreliable),
    and re-open with the DI result RESTRICTED to those pages
    (:class:`_PagesOf`). Pages with good embedded text keep it.

Measured 2026-09-16 on this corpus, counting pages whose text source came out
``azure_di`` rather than ``pdf_text``:

| report | pages | DI covers | read by DI, ``auto`` | ``all`` | s, ``auto`` | s, ``all`` |
|---|---|---|---|---|---|---|
| R28 | 455 | 455 | 28 | 455 | 11 | 63 |
| R15 | 202 | 202 | 78 | 202 | 9 | 13 |
| R13 | 197 | 197 | 197 | 197 | 9 | 9 |

R28 is the case the table exists for: 28 of its 455 pages need help (6 scanned
pages and 22 calc printouts whose fonts carry no Unicode map), and ``all``
would throw away the exact embedded text on the other 427 to buy optical text
for those 28 -- at six times the wall clock. R13 is the other end: a pure
scan, where every page needs DI and the two modes agree.
"""

from __future__ import annotations

import gzip
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The gitignored corpus root. Nothing under here may be committed.
RAW_DIR = (Path(__file__).resolve().parents[1]
           / "field_feedback" / "2026-09-16_report-ingest" / "raw")
CORPUS_DIR = RAW_DIR / "corpus"
DI_DIR = RAW_DIR / "di"
#: Where the harness writes renders for a human to look at.
CHECKS_DIR = RAW_DIR / "checks"
MANIFEST = CORPUS_DIR / "MANIFEST.md"
DI_MANIFEST = DI_DIR / "MANIFEST.md"
LABELS_XLSX = CORPUS_DIR / "trial_pages_working_r2.xlsx"
#: Runtime cache of the label-sheet-to-ID matching (derived, stays private).
LABELS_MAP_JSON = RAW_DIR / "labels_map.json"

_ID_RE = re.compile(r"^R\d\d$")

DI_MODES = ("auto", "all", "none")


@dataclass(frozen=True)
class ReportInfo:
    """One row of the corpus manifest.

    ``private_name`` is the source file's stem. It is loaded so the hand-label
    sheets can be matched to IDs at runtime; it must never reach a tracked
    file. Every other field is safe to publish.
    """

    id: str
    private_name: str
    origin: str
    pages: int
    text_pages: int
    image_only_pages: int
    blank_pages: int
    text_over_image_pages: int
    avg_chars: int
    sizes: str
    size_counts: Tuple[Tuple[float, float, int], ...] = field(default=())

    @property
    def is_public(self) -> bool:
        return self.origin.startswith("public")

    def __repr__(self) -> str:                    # never leak the file name
        return (f"ReportInfo({self.id}, pages={self.pages}, "
                f"text_pages={self.text_pages}, "
                f"image_only={self.image_only_pages})")


def raw_available() -> bool:
    """Is the private corpus present on this machine?"""
    return MANIFEST.is_file() and CORPUS_DIR.is_dir()


def _require_raw() -> None:
    if not raw_available():
        raise FileNotFoundError(
            f"the report-ingest corpus is not on this machine ({RAW_DIR} is "
            f"gitignored and exists only where the owner put it)")


def _split_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _int(cell: str, default: int = 0) -> int:
    cell = cell.strip().strip("`")
    try:
        return int(cell)
    except ValueError:
        return default


def _parse_sizes(cell: str) -> Tuple[Tuple[float, float, int], ...]:
    out: List[Tuple[float, float, int]] = []
    for part in cell.split(","):
        m = re.match(r"\s*([\d.]+)x([\d.]+)\s*[x×]\s*(\d+)", part)
        if m:
            out.append((float(m.group(1)), float(m.group(2)), int(m.group(3))))
    return tuple(out)


_REPORTS: Optional[List[ReportInfo]] = None


def list_reports(refresh: bool = False) -> List[ReportInfo]:
    """Every corpus report, in ID order, parsed from the private manifest."""
    global _REPORTS
    if _REPORTS is not None and not refresh:
        return list(_REPORTS)
    _require_raw()
    rows: List[ReportInfo] = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = _split_row(line)
        if len(cells) < 10 or not _ID_RE.match(cells[0]):
            continue
        name = cells[1].strip().strip("`")
        rows.append(ReportInfo(
            id=cells[0],
            private_name=Path(name).stem,
            origin=cells[2],
            pages=_int(cells[3]),
            text_pages=_int(cells[4]),
            image_only_pages=_int(cells[5]),
            blank_pages=_int(cells[6]),
            text_over_image_pages=_int(cells[7]),
            avg_chars=_int(cells[8]),
            sizes=cells[9],
            size_counts=_parse_sizes(cells[9]),
        ))
    rows.sort(key=lambda r: r.id)
    _REPORTS = rows
    return list(rows)


def report(rid: str) -> ReportInfo:
    """The manifest row for one ID."""
    for r in list_reports():
        if r.id == rid:
            return r
    raise KeyError(f"{rid} is not in the corpus manifest")


def pdf_path(rid: str) -> Path:
    """The PDF for one ID (always ``<ID>.pdf`` inside ``raw/corpus``)."""
    _require_raw()
    p = CORPUS_DIR / f"{rid}.pdf"
    if not p.is_file():
        raise FileNotFoundError(f"no PDF for {rid} at {p}")
    return p


# -- Azure Document Intelligence results ------------------------------------

_DI_MANIFEST: Optional[Dict[str, Dict[str, Any]]] = None


def di_manifest() -> Dict[str, Dict[str, Any]]:
    """ID to {di_pages, di_tables, di_paragraphs, ok} from the DI manifest.

    Reading the manifest is how the harness answers "how many pages does the
    DI result cover" without inflating a 380 MB JSON. ``ok`` is False for a
    result whose export was truncated.
    """
    global _DI_MANIFEST
    if _DI_MANIFEST is not None:
        return _DI_MANIFEST
    out: Dict[str, Dict[str, Any]] = {}
    if DI_MANIFEST.is_file():
        for line in DI_MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = _split_row(line)
            if len(cells) < 6 or not _ID_RE.match(cells[0]):
                continue
            pages_cell = cells[3]
            ok = pages_cell.isdigit() and di_file(cells[0]).is_file()
            out[cells[0]] = {
                "di_pages": _int(pages_cell, -1),
                "di_tables": _int(cells[4], -1),
                "di_paragraphs": _int(cells[5], -1),
                "ok": ok,
            }
    _DI_MANIFEST = out
    return out


def di_file(rid: str) -> Path:
    return DI_DIR / f"{rid}.json.gz"


def has_di(rid: str) -> bool:
    """Is a usable DI result on disk for this ID?

    False for an ID with no result at all, and for one the manifest records as
    a truncated export.
    """
    if not di_file(rid).is_file():
        return False
    row = di_manifest().get(rid)
    return True if row is None else bool(row["ok"])


def di_page_count(rid: str) -> Optional[int]:
    """Pages the DI result covers, from the manifest (no JSON is read)."""
    row = di_manifest().get(rid)
    if row is None or not row["ok"]:
        return None
    return row["di_pages"] or None


def load_di(rid: str) -> Optional[dict]:
    """The gunzipped DI result for one ID, or None with a warning.

    None means: no result on disk, or the export is truncated and the JSON
    will not parse (R17 at the time of writing). Never an exception, because
    every caller's honest answer to a missing DI result is "read the text
    layer instead".
    """
    path = di_file(rid)
    if not path.is_file():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"{rid}: the Azure DI result is unusable ({type(exc).__name__}: "
            f"{exc}); treating it as unavailable",
            RuntimeWarning, stacklevel=2)
        return None


class _PagesOf:
    """An ``AzureLayout`` narrowed to a set of pages.

    The text-source protocol planlens asks for is three members: ``name``,
    ``covers(index)`` and ``extract(page, index, words=...)``. Narrowing
    ``covers`` is the whole mechanism behind ``di="auto"`` -- the pages left
    out keep their PDF text.
    """

    def __init__(self, layout, pages: Sequence[int]):
        self._layout = layout
        self.pages = sorted({int(p) for p in pages if layout.covers(int(p))})
        self._set = set(self.pages)
        self.name = layout.name

    def covers(self, index: int) -> bool:
        return index in self._set

    def extract(self, page, index: int, words: bool = False):
        return self._layout.extract(page, index, words=words)


def open_report(rid: str, di: str = "auto", *, warn: bool = True,
                ocr_pages: Optional[Sequence[int]] = None):
    """Open one corpus report as a planlens ``Document``.

    ``di`` is ``"auto"`` (default), ``"all"`` or ``"none"``; the module
    docstring says what each does to a page's text source. ``auto`` and
    ``all`` fall back to ``none``, with a warning, when the ID has no usable
    DI result.

    ``ocr_pages`` short-circuits the probe pass ``auto`` otherwise runs to
    find the pages that need help. Pass it only when the list came from
    ``pages_needing_ocr`` on this same PDF with NO text source attached; a
    caller that has already built that document (a measurement script
    reporting the before-and-after) saves a full page map by handing it over.

    The caller owns the document and should close it (it is a context
    manager).
    """
    if di not in DI_MODES:
        raise ValueError(f"di must be one of {DI_MODES}, not {di!r}")
    from planlens.document import open_document
    from planlens.document.azure_di import AzureLayout, pages_needing_ocr

    path = pdf_path(rid)
    if di == "none":
        return open_document(str(path), name=rid)

    if not has_di(rid):
        if warn:
            warnings.warn(
                f"{rid}: no usable Azure DI result; opening with the PDF text "
                f"layer only", RuntimeWarning, stacklevel=2)
        return open_document(str(path), name=rid)

    wanted: List[int] = []
    if di == "auto":
        if ocr_pages is None:
            # Which pages need help is a question about the PDF's OWN text,
            # so it has to be asked of a document with no text source.
            with open_document(str(path), name=rid) as probe:
                wanted = pages_needing_ocr(probe)
        else:
            wanted = [int(p) for p in ocr_pages]
        if not wanted:
            return open_document(str(path), name=rid)

    result = load_di(rid)
    if result is None:
        return open_document(str(path), name=rid)
    layout: Any = AzureLayout(result)
    if di == "auto":
        layout = _PagesOf(layout, wanted)
    return open_document(str(path), text_source=layout, name=rid)


def text_source_counts(doc) -> Dict[str, int]:
    """How many pages ended up reading from each text source.

    ``page_map`` records a non-default source in the page's evidence, so a
    page with no ``text_source`` key read the PDF's own text layer.
    """
    out: Dict[str, int] = {}
    for s in doc.page_map():
        name = s.evidence.get("text_source", "pdf_text")
        out[name] = out.get(name, 0) + 1
    return out
