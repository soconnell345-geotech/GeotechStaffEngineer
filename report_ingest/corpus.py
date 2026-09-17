"""A folder of reports, by ID, wherever that folder happens to be.

The WP0 harness read the corpus from one hard-coded place inside the repo.
The scoring now has to run on the cluster, where the corpus is a Unity
Catalog Volume or a synced SharePoint folder and the repo is not there at
all, so the loading lives here -- in the shipped package, taking directories
as arguments -- and the harness binds it to the repo's own paths.

WHAT A CORPUS FOLDER LOOKS LIKE::

    <corpus_dir>/R01.pdf ... R38.pdf
    <corpus_dir>/MANIFEST.md          one table row per report
    <di_dir>/R01.json.gz ...          Azure Document Intelligence results
    <labels_xlsx>                     the hand-labelled page types

Only the PDFs are required. A missing manifest costs the page counts, a
missing DI folder means the text layer is read on its own, and a missing
spreadsheet means a run measures but does not score.

PRIVACY. A report is an ID everywhere. :attr:`ReportInfo.private_name` holds
the manifest's source-file stem because the label sheets have to be matched
to IDs, and it is the one field ``repr`` leaves out.

WHAT ``di`` DOES, and why ``auto`` is the useful setting. planlens'
``text_source`` REPLACES the PDF's text on every page it covers -- there is
no merge and no "only where the PDF has none" rule. A DI result covering
every page therefore throws away exact embedded text to buy optical text.
``auto`` opens the PDF once with no text source, asks ``pages_needing_ocr``
which pages cannot be read, and attaches DI to only those. Measured on the
corpus: 28 pages of 455 on the longest report, at a sixth of the wall clock
of attaching it to all of them.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Corpus", "ReportInfo", "PageLabel", "SheetMatch", "LABELS",
    "RAW_TO_LABEL", "UnknownPageType", "to_label", "DI_MODES",
]

#: The page-label vocabulary. It is planlens' ``ROLES`` plus nothing: a label
#: the scoring can produce but planlens cannot, or the other way round, would
#: read as a miss on every page carrying it.
LABELS: Tuple[str, ...] = (
    "narrative", "figure", "plan", "profile", "boring_log", "test_pit_log",
    "cpt_log", "dcp_log", "lab_test", "field_test", "calculation",
    "appended_report", "photos", "divider", "cover", "letter", "toc", "other",
)

#: Every ``page_type`` string the hand-label spreadsheet uses, mapped into
#: :data:`LABELS`. Deliberately TOTAL over those strings: an unknown one
#: raises rather than being dropped, because a silently ignored label is a
#: silently wrong score. Change a row here, never in the spreadsheet.
RAW_TO_LABEL: Dict[str, str] = {
    "main report narrative": "narrative",
    "boring log": "boring_log",
    "test pit log": "test_pit_log",
    "cpt log": "cpt_log",
    "dcp log": "dcp_log",
    "lab testing": "lab_test",
    "subsurface drainage testing": "field_test",
    "calculation": "calculation",
    "appended report": "appended_report",
    "field photos": "photos",
    "appendix cover page": "divider",
    "figures or tables cover page": "divider",
    "figures or tables cover sheet": "divider",
    "figures or reports cover page": "divider",
    "cover page": "cover",
    "cover letter": "letter",
    "table of contents": "toc",
    "figure": "figure",
    "test location plan": "plan",
    "subsurface profile": "profile",
    "informational appendix content": "other",
    "other or unknown": "other",
    "other appendix table": "other",
}

DI_MODES = ("auto", "all", "none")
_ID_RE = re.compile(r"^R\d\d$")


class UnknownPageType(ValueError):
    """A ``page_type`` the mapping does not cover."""


def to_label(raw: str) -> str:
    """Map one raw ``page_type`` string into :data:`LABELS`."""
    key = " ".join(str(raw).strip().lower().split())
    try:
        return RAW_TO_LABEL[key]
    except KeyError:
        raise UnknownPageType(
            f"page_type {key!r} is not in RAW_TO_LABEL; add a row rather than "
            f"letting the page fall out of the score") from None


@dataclass(frozen=True)
class ReportInfo:
    """One row of the corpus manifest."""

    id: str
    private_name: str
    origin: str = ""
    pages: int = 0
    text_pages: int = 0
    image_only_pages: int = 0
    blank_pages: int = 0
    text_over_image_pages: int = 0
    avg_chars: int = 0
    sizes: str = ""
    size_counts: Tuple[Tuple[float, float, int], ...] = field(default=())

    @property
    def is_public(self) -> bool:
        return self.origin.startswith("public")

    def __repr__(self) -> str:                    # never leak the file name
        return (f"ReportInfo({self.id}, pages={self.pages}, "
                f"text_pages={self.text_pages}, "
                f"image_only={self.image_only_pages})")


@dataclass(frozen=True)
class PageLabel:
    """One hand-labelled page. ``page0`` is 0-based; the sheet is 1-based."""

    page0: int
    raw: str
    label: str


@dataclass(frozen=True)
class SheetMatch:
    """How one label sheet was matched to a corpus ID. Safe to print whole."""

    index: int
    rows: int
    rid: Optional[str]
    method: str                      # "name", "pages" or "unmatched"
    manifest_pages: Optional[int] = None

    @property
    def rows_match(self) -> Optional[bool]:
        if self.manifest_pages is None:
            return None
        return self.rows == self.manifest_pages

    def describe(self) -> str:
        if self.rid is None:
            return (f"sheet {self.index}: NO MATCH ({self.rows} label rows) "
                    f"-- no corpus PDF matches its report name, and nothing "
                    f"with that page count shares a word with it")
        verdict = "rows == pages" if self.rows_match else (
            f"MISMATCH: {self.rows} label rows vs {self.manifest_pages} pages")
        return (f"sheet {self.index}: {self.rid} "
                f"({self.rows} rows, matched by {self.method}) -- {verdict}")


def _norm(name: str) -> str:
    """Case, punctuation and accent-insensitive form of a file stem."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    if text.endswith(".pdf"):
        text = text[:-4]
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _tokens(name: str) -> List[str]:
    """Distinctive words of a file stem: three characters or more."""
    return [t for t in _norm(name).split() if len(t) >= 3]


class _PagesOf:
    """An ``AzureLayout`` narrowed to a set of pages.

    planlens asks a text source for three things: ``name``, ``covers(index)``
    and ``extract(page, index, words=...)``. Narrowing ``covers`` is the
    whole mechanism behind ``di="auto"``; the pages left out keep their own
    text.
    """

    def __init__(self, layout: Any, pages: Sequence[int]) -> None:
        self._layout = layout
        self.pages = sorted({int(p) for p in pages if layout.covers(int(p))})
        self._set = set(self.pages)
        self.name = layout.name

    def covers(self, index: int) -> bool:
        return index in self._set

    def extract(self, page: Any, index: int, words: bool = False) -> Any:
        return self._layout.extract(page, index, words=words)


class Corpus:
    """The reports, their DI results and their hand labels, in one folder.

    ``corpus_dir`` holds ``R01.pdf`` ... and ``MANIFEST.md``. ``di_dir``
    holds ``R01.json.gz`` ...; it defaults to ``corpus_dir/../di`` and then
    to ``corpus_dir/di``, so both the repo's layout and a flat cluster folder
    work without an argument. ``labels_xlsx`` is the hand-label spreadsheet.
    ``cache_dir`` is where the derived sheet-to-ID map is written; it
    defaults beside the corpus and is skipped silently if unwritable, which
    is what a read-only Volume needs.
    """

    def __init__(self, corpus_dir: Any, *, di_dir: Any = None,
                 labels_xlsx: Any = None, cache_dir: Any = None) -> None:
        self.corpus_dir = Path(corpus_dir)
        if di_dir is not None:
            self.di_dir = Path(di_dir)
        elif (self.corpus_dir.parent / "di").is_dir():
            self.di_dir = self.corpus_dir.parent / "di"
        else:
            self.di_dir = self.corpus_dir / "di"
        self.labels_xlsx = Path(labels_xlsx) if labels_xlsx else None
        self.cache_dir = Path(cache_dir) if cache_dir else self.corpus_dir.parent
        self.manifest = self.corpus_dir / "MANIFEST.md"
        self.di_manifest = self.di_dir / "MANIFEST.md"
        self._reports: Optional[List[ReportInfo]] = None
        self._di_rows: Optional[Dict[str, Dict[str, Any]]] = None
        self._sheets: Optional[Tuple[List[SheetMatch], Dict[int, str]]] = None

    # -- is it here --------------------------------------------------------
    @property
    def available(self) -> bool:
        """Is there a corpus at this path at all?"""
        return self.corpus_dir.is_dir() and bool(
            list(self.corpus_dir.glob("R[0-9][0-9].pdf")))

    @property
    def labels_available(self) -> bool:
        return bool(self.labels_xlsx and self.labels_xlsx.is_file())

    def _require(self) -> None:
        if not self.available:
            raise FileNotFoundError(
                f"no corpus at {self.corpus_dir}: expected R01.pdf ... "
                f"R38.pdf and MANIFEST.md there")

    # -- the reports -------------------------------------------------------
    def list_reports(self, refresh: bool = False) -> List[ReportInfo]:
        """Every report, in ID order, from the manifest or from the files."""
        if self._reports is not None and not refresh:
            return list(self._reports)
        self._require()
        rows: List[ReportInfo] = []
        if self.manifest.is_file():
            rows = self._parse_manifest()
        if not rows:
            # No manifest: the PDFs themselves are the list. Page counts stay
            # zero rather than being guessed; nothing here needs them.
            rows = [ReportInfo(id=p.stem, private_name=p.stem)
                    for p in sorted(self.corpus_dir.glob("R[0-9][0-9].pdf"))]
        rows.sort(key=lambda r: r.id)
        self._reports = rows
        return list(rows)

    def _parse_manifest(self) -> List[ReportInfo]:
        def cells_of(line: str) -> List[str]:
            return [c.strip() for c in line.strip().strip("|").split("|")]

        def as_int(cell: str, default: int = 0) -> int:
            try:
                return int(cell.strip().strip("`"))
            except ValueError:
                return default

        def sizes_of(cell: str) -> Tuple[Tuple[float, float, int], ...]:
            out: List[Tuple[float, float, int]] = []
            for part in cell.split(","):
                m = re.match(r"\s*([\d.]+)x([\d.]+)\s*[x×]\s*(\d+)", part)
                if m:
                    out.append((float(m.group(1)), float(m.group(2)),
                                int(m.group(3))))
            return tuple(out)

        rows: List[ReportInfo] = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if not line.lstrip().startswith("|"):
                continue
            cells = cells_of(line)
            if len(cells) < 10 or not _ID_RE.match(cells[0]):
                continue
            rows.append(ReportInfo(
                id=cells[0],
                private_name=Path(cells[1].strip().strip("`")).stem,
                origin=cells[2],
                pages=as_int(cells[3]),
                text_pages=as_int(cells[4]),
                image_only_pages=as_int(cells[5]),
                blank_pages=as_int(cells[6]),
                text_over_image_pages=as_int(cells[7]),
                avg_chars=as_int(cells[8]),
                sizes=cells[9],
                size_counts=sizes_of(cells[9]),
            ))
        return rows

    def report(self, rid: str) -> ReportInfo:
        for r in self.list_reports():
            if r.id == rid:
                return r
        raise KeyError(f"{rid} is not in the corpus manifest")

    def ids(self) -> List[str]:
        return [r.id for r in self.list_reports()]

    def pdf_path(self, rid: str) -> Path:
        self._require()
        path = self.corpus_dir / f"{rid}.pdf"
        if not path.is_file():
            raise FileNotFoundError(f"no PDF for {rid} at {path}")
        return path

    # -- Azure Document Intelligence ---------------------------------------
    def di_manifest_rows(self) -> Dict[str, Dict[str, Any]]:
        """ID to the DI manifest's row, without opening a 380 MB JSON."""
        if self._di_rows is not None:
            return self._di_rows
        out: Dict[str, Dict[str, Any]] = {}
        if self.di_manifest.is_file():
            for line in self.di_manifest.read_text(
                    encoding="utf-8").splitlines():
                if not line.lstrip().startswith("|"):
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if len(cells) < 6 or not _ID_RE.match(cells[0]):
                    continue
                pages = cells[3]
                out[cells[0]] = {
                    "di_pages": int(pages) if pages.isdigit() else -1,
                    "ok": pages.isdigit() and self.di_file(cells[0]).is_file(),
                }
        self._di_rows = out
        return out

    def di_file(self, rid: str) -> Path:
        return self.di_dir / f"{rid}.json.gz"

    def has_di(self, rid: str) -> bool:
        """Is a usable DI result on disk? False for a truncated export."""
        if not self.di_file(rid).is_file():
            return False
        row = self.di_manifest_rows().get(rid)
        return True if row is None else bool(row["ok"])

    def load_di(self, rid: str) -> Optional[dict]:
        """The DI result, or None with a warning.

        Never an exception: every caller's honest answer to a missing or
        truncated result is "read the text layer instead".
        """
        path = self.di_file(rid)
        if not path.is_file():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, EOFError, json.JSONDecodeError) as exc:
            warnings.warn(
                f"{rid}: the Azure DI result is unusable "
                f"({type(exc).__name__}: {exc}); treating it as unavailable",
                RuntimeWarning, stacklevel=2)
            return None

    # -- opening -----------------------------------------------------------
    def open_report(self, rid: str, di: str = "auto", *, warn: bool = True,
                    ocr_pages: Optional[Sequence[int]] = None) -> Any:
        """Open one report as a planlens ``Document``.

        The caller owns the document and should close it; it is a context
        manager. ``ocr_pages`` short-circuits the probe pass ``auto``
        otherwise runs, and must have come from ``pages_needing_ocr`` on this
        same PDF with NO text source attached.
        """
        if di not in DI_MODES:
            raise ValueError(f"di must be one of {DI_MODES}, not {di!r}")
        from planlens.document import open_document
        from planlens.document.azure_di import AzureLayout, pages_needing_ocr

        path = self.pdf_path(rid)
        if di == "none":
            return open_document(str(path), name=rid)
        if not self.has_di(rid):
            if warn:
                warnings.warn(
                    f"{rid}: no usable Azure DI result; opening with the PDF "
                    f"text layer only", RuntimeWarning, stacklevel=2)
            return open_document(str(path), name=rid)

        wanted: List[int] = []
        if di == "auto":
            if ocr_pages is None:
                # Which pages need help is a question about the PDF's OWN
                # text, so it has to be asked of a document with no source.
                with open_document(str(path), name=rid) as probe:
                    wanted = pages_needing_ocr(probe)
            else:
                wanted = [int(p) for p in ocr_pages]
            if not wanted:
                return open_document(str(path), name=rid)

        result = self.load_di(rid)
        if result is None:
            return open_document(str(path), name=rid)
        layout: Any = AzureLayout(result)
        if di == "auto":
            layout = _PagesOf(layout, wanted)
        return open_document(str(path), text_source=layout, name=rid)

    # -- the hand labels ---------------------------------------------------
    def _workbook(self) -> Any:
        try:
            import openpyxl
        except ImportError as exc:                   # pragma: no cover
            raise ImportError(
                "reading the hand labels needs openpyxl") from exc
        if not self.labels_available:
            raise FileNotFoundError(
                f"no hand-label spreadsheet ({self.labels_xlsx})")
        return openpyxl.load_workbook(self.labels_xlsx, read_only=True,
                                      data_only=True)

    @staticmethod
    def _sheet_rows(ws: Any) -> List[dict]:
        out: List[dict] = []
        header: Optional[List[str]] = None
        for row in ws.iter_rows(values_only=True):
            if header is None:
                header = [(" ".join(str(c).strip().lower().split()) if c else "")
                          for c in row]
                continue
            if all(c is None for c in row):
                continue
            out.append(dict(zip(header, row)))
        return out

    @property
    def _labels_map_path(self) -> Path:
        return self.cache_dir / "labels_map.json"

    def sheet_map(self, refresh: bool = False) -> List[SheetMatch]:
        """Match every label sheet to a corpus ID.

        The cache holds the sheet NAMES, which are private, so it is written
        beside the corpus and never returned.
        """
        if self._sheets is not None and not refresh:
            return list(self._sheets[0])
        cache = self._labels_map_path
        if not refresh and cache.is_file():
            try:
                blob = json.loads(cache.read_text(encoding="utf-8"))
                matches = [SheetMatch(**{k: v for k, v in row.items()
                                         if k != "sheet_name"})
                           for row in blob["sheets"]]
                names = {int(row["index"]): row["sheet_name"]
                         for row in blob["sheets"] if row.get("sheet_name")}
                self._sheets = (matches, names)
                return list(matches)
            except (OSError, ValueError, KeyError, TypeError):
                pass                                 # rebuild below
        matches, names = self._build_sheet_map()
        self._sheets = (matches, names)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({
                "note": "derived at runtime; PRIVATE (sheet names)",
                "sheets": [dict(index=m.index, rows=m.rows, rid=m.rid,
                                method=m.method,
                                manifest_pages=m.manifest_pages,
                                sheet_name=names.get(m.index))
                           for m in matches],
            }, indent=2), encoding="utf-8")
        except OSError:                              # a read-only Volume
            pass
        return list(matches)

    def _build_sheet_map(self) -> Tuple[List[SheetMatch], Dict[int, str]]:
        wb = self._workbook()
        sheet_names = list(wb.sheetnames)
        summary: Dict[str, str] = {}
        for row in self._sheet_rows(wb[sheet_names[0]]):
            sheet, report_name = row.get("sheet"), row.get("report name")
            if sheet and report_name:
                summary[str(sheet)] = str(report_name)

        by_norm: Dict[str, str] = {}
        for info in self.list_reports():
            by_norm.setdefault(_norm(info.private_name), info.id)

        matches: List[SheetMatch] = []
        names: Dict[int, str] = {}
        taken: set = set()
        pending: List[Tuple[int, str, int]] = []

        for index, sheet_name in enumerate(sheet_names):
            if index == 0:
                continue
            rows = len(self._sheet_rows(wb[sheet_name]))
            rid = by_norm.get(_norm(summary.get(sheet_name, "")))
            if rid is not None and rid not in taken:
                taken.add(rid)
                names[index] = sheet_name
                matches.append(SheetMatch(index, rows, rid, "name",
                                          self.report(rid).pages))
            else:
                pending.append((index, sheet_name, rows))

        # Fallback: a unique report with exactly that many pages AND a
        # distinctive word in common. The token guard is not fussiness --
        # without it the one sheet whose report was never copied into the
        # corpus matched an unrelated report of the same length on the page
        # count alone, and 153 hand labels would have scored the wrong
        # document. A page count is a weak signal; two that agree are worth
        # acting on, one is not.
        for index, sheet_name, rows in pending:
            wanted = set(_tokens(summary.get(sheet_name, "")))
            candidates = [r for r in self.list_reports()
                          if r.pages == rows and r.id not in taken]
            shared = [r for r in candidates
                      if wanted & set(_tokens(r.private_name))]
            if len(candidates) == 1 and len(shared) == 1:
                rid = shared[0].id
                taken.add(rid)
                names[index] = sheet_name
                matches.append(SheetMatch(index, rows, rid, "pages",
                                          self.report(rid).pages))
            else:
                matches.append(SheetMatch(index, rows, None, "unmatched"))

        matches.sort(key=lambda m: m.index)
        wb.close()
        return matches, names

    def mapped_ids(self) -> List[str]:
        """Corpus IDs that have hand labels, in ID order."""
        return sorted(m.rid for m in self.sheet_map() if m.rid)

    def _sheet_name_for(self, rid: str) -> str:
        self.sheet_map()
        assert self._sheets is not None
        for index, name in self._sheets[1].items():
            for m in self._sheets[0]:
                if m.index == index and m.rid == rid:
                    return name
        raise KeyError(f"{rid} has no hand-label sheet")

    def labels_for(self, rid: str) -> List[PageLabel]:
        """The hand labels for one report, 0-based, in page order."""
        name = self._sheet_name_for(rid)
        wb = self._workbook()
        try:
            rows = self._sheet_rows(wb[name])
        finally:
            wb.close()
        out: List[PageLabel] = []
        for row in rows:
            page, raw = row.get("page"), row.get("page_type")
            if page is None or raw is None:
                continue
            out.append(PageLabel(
                page0=int(page) - 1,
                raw=" ".join(str(raw).strip().lower().split()),
                label=to_label(raw)))
        out.sort(key=lambda p: p.page0)
        return out

    def label_counts(self) -> Dict[str, int]:
        counts = {label: 0 for label in LABELS}
        for rid in self.mapped_ids():
            for pl in self.labels_for(rid):
                counts[pl.label] += 1
        return counts


def text_source_counts(doc: Any) -> Dict[str, int]:
    """How many pages ended up reading from each text source."""
    out: Dict[str, int] = {}
    for s in doc.page_map():
        name = s.evidence.get("text_source", "pdf_text")
        out[name] = out.get(name, 0) + 1
    return out
