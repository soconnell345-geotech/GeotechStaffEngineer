"""The shared finding format for document reviews (plan S1.5).

WHY. ``module_work/REVIEW_ARCHITECTURE.md`` settles on ONE finding format for
every review shape: a small review's agent, a large review's lead, a
discipline reviewer of a mega-review. Findings in one format can be compared,
de-duplicated and checked by code, and every deliverable (the Word comment
log, the marked-up PDF) is rendered FROM them instead of being written by
hand. Findings are also work saved as a file: between turns the app keeps only
the user's messages and the agent's answers, so a review split across turns
forgets what it found unless it wrote it down.

WHAT.

* :class:`Citation` and :class:`Finding`, with ``to_dict`` / ``from_dict``
  and validation that says what is wrong (a bad severity, an empty
  statement, a finding with no citation).
* :func:`verify_quotes`: every quoted citation is checked against the text
  of the page it cites and the text of its review markups (rapidfuzz
  ``partial_ratio`` after whitespace and case are normalised). A finding
  whose quotes are all missing from their pages is marked
  ``confidence="low"`` - never deleted - unless it was SEEN (lettering read
  by looking is often not in the text layer). A page with no text layer, or
  one whose text layer planlens marks unreliable, cannot be checked, and
  says so rather than failing the quote. :func:`planlens_page_text` reads
  the pages with planlens.
* :class:`FindingsLedger`: ``findings.json`` in the conversation's working
  folder (:func:`funhouse_agent._fileio.default_output_dir`), ids ``F1``,
  ``F2``, ...
* :func:`to_markdown` (a comment log for ``write_docx``) and
  :func:`to_markups` (the input ``annotate_document`` takes).

The agent-facing tools are in :mod:`funhouse_agent.deep.findings_tools`,
offered on the lean review agent with ``GEOTECH_REVIEW_FINDINGS=1``.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Mapping, Optional,
                    Sequence, Union)

SEVERITIES = ("info", "minor", "major", "critical")
CONFIDENCES = ("high", "medium", "low")
EVIDENCE_KINDS = ("read", "seen", "computed")
STATUSES = ("draft", "confirmed", "rejected")

#: The lowest partial-ratio score (0-100) at which a quote counts as found.
DEFAULT_MIN_SCORE = 85
#: The ledger's file name in the working folder.
LEDGER_NAME = "findings.json"
LEDGER_VERSION = 1

#: Where a note goes on a page when a citation has neither quote nor box.
NOTE_POINT = [24.0, 24.0]


# ---------------------------------------------------------------------------
# The format
# ---------------------------------------------------------------------------

def _enum(value: Any, allowed: Sequence[str], what: str) -> str:
    text = str(value or "").strip().lower()
    if text not in allowed:
        raise ValueError(f"{what} must be one of {', '.join(allowed)}; "
                         f"got {value!r}")
    return text


def _opt_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: Any, what: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{what} must be a list of strings; got {value!r}")
    return [str(v).strip() for v in value if str(v).strip()]


@dataclass
class Citation:
    """Where a finding is on the page.

    ``page`` is 0-based (as every tool numbers pages); ``pdf_page`` is always
    ``page + 1``, the number a PDF viewer shows. ``bbox`` is
    ``[x0, y0, x1, y1]`` in displayed-page PDF points, top-left origin (the
    frame of ``read_document(with_locations=true)`` and ``render_region``).
    ``quote`` is the exact words on that page the finding rests on.
    """
    document: str
    page: int
    pdf_page: Optional[int] = None
    printed_page: Optional[str] = None
    sheet: Optional[str] = None
    bbox: Optional[List[float]] = None
    quote: Optional[str] = None

    def __post_init__(self) -> None:
        self.document = str(self.document or "").strip()
        if not self.document:
            raise ValueError("a citation needs the document it cites (the "
                             "attachment name or path you opened)")
        try:
            page = int(self.page)
        except (TypeError, ValueError):
            raise ValueError(f"a citation's page must be a 0-based integer; "
                             f"got {self.page!r}")
        if isinstance(self.page, bool) or page < 0:
            raise ValueError(f"a citation's page must be a 0-based integer; "
                             f"got {self.page!r}")
        self.page = page
        self.pdf_page = page + 1
        self.printed_page = _opt_text(self.printed_page)
        self.sheet = _opt_text(self.sheet)
        self.quote = _opt_text(self.quote)
        if self.bbox is not None:
            try:
                box = [float(v) for v in self.bbox]
            except (TypeError, ValueError):
                box = []
            if len(box) != 4:
                raise ValueError("a citation's bbox must be [x0, y0, x1, y1] "
                                 "in PDF points (top-left origin); got "
                                 f"{self.bbox!r}")
            self.bbox = box

    def to_dict(self) -> Dict[str, Any]:
        return {"document": self.document, "page": self.page,
                "pdf_page": self.pdf_page, "printed_page": self.printed_page,
                "sheet": self.sheet, "bbox": self.bbox, "quote": self.quote}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Citation":
        if isinstance(data, Citation):
            return data
        if not isinstance(data, Mapping):
            raise ValueError(f"a citation must be an object with document "
                             f"and page; got {data!r}")
        page = data.get("page")
        if page is None and data.get("pdf_page") is not None:
            try:
                page = int(data["pdf_page"]) - 1     # a viewer page given alone
            except (TypeError, ValueError):
                page = data.get("pdf_page")
        return cls(document=data.get("document"), page=page,
                   printed_page=data.get("printed_page"),
                   sheet=data.get("sheet"), bbox=data.get("bbox"),
                   quote=data.get("quote"))

    def where(self) -> str:
        """How a reader finds it: the sheet, else the printed page, else the
        viewer page (never the 0-based index)."""
        if self.sheet:
            return f"Sheet {self.sheet}"
        if self.printed_page:
            return f"p. {self.printed_page}"
        return f"PDF p. {self.pdf_page}"


@dataclass
class Finding:
    """One review finding in the shared format (see the module docstring).

    ``severity`` is info / minor / major / critical; ``confidence`` high /
    medium / low; ``evidence`` how it is known - read (from the text), seen
    (by looking at the page) or computed; ``status`` draft / confirmed /
    rejected (by the user). ``source`` names the worker or reviewer that
    wrote it. ``quote_verified`` and ``quote_note`` are set by
    :func:`verify_quotes`.
    """
    id: str = ""
    statement: str = ""
    severity: str = ""
    confidence: str = ""
    disciplines: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    evidence: str = ""
    status: str = "draft"
    related: List[str] = field(default_factory=list)
    source: str = ""
    quote_verified: Optional[bool] = None
    quote_note: Optional[str] = None

    def __post_init__(self) -> None:
        self.id = str(self.id or "").strip()
        self.statement = str(self.statement or "").strip()
        if not self.statement:
            raise ValueError("a finding needs a statement: what is wrong or "
                             "worth noting, in a sentence or two")
        self.severity = _enum(self.severity, SEVERITIES, "severity")
        self.confidence = _enum(self.confidence, CONFIDENCES, "confidence")
        self.evidence = _enum(self.evidence, EVIDENCE_KINDS, "evidence")
        self.status = _enum(self.status or "draft", STATUSES, "status")
        self.disciplines = _str_list(self.disciplines, "disciplines")
        self.related = _str_list(self.related, "related")
        self.source = str(self.source or "").strip()
        cites = self.citations
        if isinstance(cites, (Mapping, Citation)):
            cites = [cites]
        if not isinstance(cites, (list, tuple)):
            raise ValueError("citations must be a list of {document, page, "
                             "quote or bbox}")
        self.citations = [Citation.from_dict(c) for c in cites]
        if not self.citations:
            raise ValueError("a finding needs at least one citation: the "
                             "document and 0-based page it is on, with the "
                             "quote it rests on or a bbox")
        if self.quote_verified is not None:
            self.quote_verified = bool(self.quote_verified)
        self.quote_note = _opt_text(self.quote_note)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "statement": self.statement,
                "severity": self.severity, "confidence": self.confidence,
                "disciplines": list(self.disciplines),
                "citations": [c.to_dict() for c in self.citations],
                "evidence": self.evidence, "status": self.status,
                "related": list(self.related), "source": self.source,
                "quote_verified": self.quote_verified,
                "quote_note": self.quote_note}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Finding":
        if isinstance(data, Finding):
            return data
        if not isinstance(data, Mapping):
            raise ValueError(f"a finding must be an object; got {data!r}")
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in known})

    def where(self) -> str:
        """Every citation's place, for a one-line summary."""
        return "; ".join(f"{c.where()} ({c.document})" for c in self.citations)


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

_QUOTE_CHARS = {"‘": "'", "’": "'", "‚": "'", "‛": "'",
                "“": '"', "”": '"', "„": '"', "′": "'",
                "″": '"', "‐": "-", "‑": "-", "‒": "-",
                "–": "-", "—": "-", "−": "-", " ": " "}
_SPACES = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    """Case, whitespace, typographic quotes and dashes flattened, so a quote
    copied across a line break or retyped with straight quotes still
    matches the page."""
    t = unicodedata.normalize("NFKC", str(text or ""))
    t = "".join(_QUOTE_CHARS.get(ch, ch) for ch in t)
    return _SPACES.sub(" ", t).strip().lower()


def _difflib_partial_ratio(short: str, long: str) -> float:
    """rapidfuzz's ``partial_ratio`` where rapidfuzz is not installed: the best
    ratio of ``short`` against any window of ``long`` its matching blocks
    point at."""
    from difflib import SequenceMatcher
    best = 0.0
    blocks = SequenceMatcher(None, short, long, autojunk=False
                             ).get_matching_blocks()
    for a, b, _size in blocks:
        start = max(0, b - a)
        window = long[start:start + len(short)]
        if not window:
            continue
        best = max(best, SequenceMatcher(None, short, window,
                                         autojunk=False).ratio())
        if best >= 0.999:
            break
    return 100.0 * best


def quote_score(quote: Any, page_text: Any) -> float:
    """0-100: how well ``quote`` appears in ``page_text`` (100 = verbatim
    after normalising). A page text SHORTER than the quote is compared whole,
    so a short page cannot "contain" a long quote."""
    q = normalize_text(quote)
    t = normalize_text(page_text)
    if not q or not t:
        return 0.0
    if q in t:
        return 100.0
    try:
        from rapidfuzz import fuzz
    except ImportError:          # planlens brings rapidfuzz; this is a floor
        fuzz = None
    if len(t) < len(q):
        if fuzz is not None:
            return float(fuzz.ratio(q, t))
        from difflib import SequenceMatcher
        return 100.0 * SequenceMatcher(None, q, t, autojunk=False).ratio()
    if fuzz is not None:
        return float(fuzz.partial_ratio(q, t))
    return _difflib_partial_ratio(q, t)


class CitedPage(str):
    """What a ``page_text`` callable may return: the page's text layer (it IS
    the string, so a plain ``str`` works the same) plus what the page's
    review markups say (their comments and the words a stamp draws) and
    whether the text layer is what the page shows (planlens'
    ``text_reliable``: False when its fonts carry no usable Unicode map)."""

    def __new__(cls, text: Any = "", markup_text: Any = "",
                text_reliable: bool = True) -> "CitedPage":
        obj = super().__new__(cls, str(text or ""))
        obj.markup_text = str(markup_text or "")
        obj.text_reliable = bool(text_reliable)
        return obj


PageText = Callable[[str, int], str]


def _check_one(quote: str, page: Any, min_score: float):
    """``(verdict, note)`` for one quote against one page: ``True`` found,
    ``False`` not found, ``None`` could not be checked."""
    text = str(page or "")
    marks = str(getattr(page, "markup_text", "") or "")
    reliable = bool(getattr(page, "text_reliable", True))
    has_text = bool(normalize_text(text))
    has_marks = bool(normalize_text(marks))
    in_text = quote_score(quote, text) if has_text and reliable else 0.0
    in_marks = quote_score(quote, marks) if has_marks else 0.0
    if max(in_text, in_marks) >= float(min_score):
        if in_text >= in_marks:
            return True, f"quote found on the page (score {in_text:.0f})"
        return True, (f"quote found in a review markup on the page (score "
                      f"{in_marks:.0f})")
    if has_text and not reliable:
        return None, ("the page's text layer is not what the page shows (its "
                      "fonts carry no usable character map), so the quote "
                      "could not be checked - look at the page")
    if not has_text:
        return None, ("the page has no text layer, so the quote could not be "
                      "checked")
    return False, (f"quote NOT found on the page (score "
                   f"{max(in_text, in_marks):.0f})")


def verify_quotes(findings: Iterable[Finding], page_text: PageText,
                  min_score: float = DEFAULT_MIN_SCORE) -> List[Finding]:
    """Check each quoted citation against the text of the page it cites.

    ``page_text(document, page)`` returns the page's text (0-based page) - a
    ``str``, or a :class:`CitedPage` that also carries the page's markup
    text and whether its text layer is reliable; it raises ``KeyError`` for
    a document it does not know and ``IndexError`` for a page the document
    does not have - both mean the quote is NOT on the page cited. A quote
    counts as found in the text layer or in a review markup's text. A page
    with no text (a scan, lettering drawn as lines), a page whose text layer
    planlens marks unreliable, or one that cannot be read leaves that quote
    unchecked.

    Sets ``quote_verified`` (``True`` when every checked quote was found,
    ``False`` when any was not, ``None`` when nothing could be checked) and
    ``quote_note``. A finding none of whose checked quotes was found drops to
    ``confidence="low"`` - unless its evidence is ``seen``: lettering read by
    looking at a page is often not in its text layer, so such a finding
    keeps its confidence and says why. Nothing is ever removed. Returns the
    findings.
    """
    out = list(findings)
    for f in out:
        notes: List[str] = []
        checked: List[bool] = []
        for c in f.citations:
            if not c.quote:
                continue
            where = f"{c.where()} of {c.document}"
            try:
                page = page_text(c.document, c.page)
            except KeyError as exc:
                checked.append(False)
                notes.append(f"{where}: the cited document was not found "
                             f"({_msg(exc)})")
                continue
            except IndexError as exc:
                checked.append(False)
                notes.append(f"{where}: that page is not in the document "
                             f"({_msg(exc)})")
                continue
            except Exception as exc:  # noqa: BLE001 - unchecked, not failed
                notes.append(f"{where}: could not read the page to check the "
                             f"quote ({type(exc).__name__})")
                continue
            verdict, note = _check_one(c.quote, page, min_score)
            if verdict is not None:
                checked.append(verdict)
            notes.append(f"{where}: {note}")
        if not checked:
            f.quote_verified = None
        else:
            f.quote_verified = all(checked)
            if not any(checked):
                if f.evidence == "seen":
                    notes.append("confidence kept: the finding was SEEN on "
                                 "the page, and lettering read by looking is "
                                 "often not in the page's text layer")
                elif f.confidence != "low":
                    f.confidence = "low"
                    notes.append("confidence lowered to low: no quote was "
                                  "found on the page it cites")
        f.quote_note = "; ".join(notes) or None
    return out


def _msg(exc: BaseException) -> str:
    text = exc.args[0] if exc.args else str(exc)
    return str(text)


def _same_document(a: str, b: str) -> bool:
    if a == b:
        return True
    base = lambda s: os.path.basename(str(s).replace("\\", "/")).lower()  # noqa: E731
    return bool(a) and bool(b) and base(a) == base(b)


class PlanlensPageText:
    """``page_text(document, page)`` over planlens, for :func:`verify_quotes`.

    ``sources`` maps a document name to its bytes or a path; a name is also
    matched by its file name alone (``plans.pdf`` finds ``/tmp/x/plans.pdf``).
    Documents are opened once and kept until :meth:`close`.
    """

    def __init__(self, sources: Optional[Mapping[str, Union[bytes, str]]]
                 = None) -> None:
        self.sources: Dict[str, Union[bytes, str]] = dict(sources or {})
        self._docs: Dict[str, Any] = {}

    def _key(self, document: str) -> str:
        if document in self.sources:
            return document
        for key in self.sources:
            if _same_document(key, document):
                return key
        raise KeyError(f"'{document}' is not among "
                       f"{sorted(self.sources) or 'no documents'}")

    def _open(self, key: str):
        if key not in self._docs:
            from planlens.document import Document
            src = self.sources[key]
            if isinstance(src, (bytes, bytearray)):
                self._docs[key] = Document(content=bytes(src), name=key)
            else:
                self._docs[key] = Document(filepath=os.fspath(src))
        return self._docs[key]

    def __call__(self, document: str, page: int) -> "CitedPage":
        doc = self._open(self._key(document))
        n = int(doc.n_pages)
        if not 0 <= int(page) < n:
            raise IndexError(f"page {page} is outside {document}, which has "
                             f"pages 0-{n - 1}")
        content = doc.page(int(page), tables=False)
        marks: List[str] = []
        for m in getattr(content, "markups", None) or []:
            for said in (getattr(m, "text", None),
                         getattr(m, "appearance_text", None)):
                if said and str(said).strip():
                    marks.append(str(said))
        try:
            reliable = bool(getattr(doc.summary(int(page)), "text_reliable",
                                    True))
        except Exception:  # noqa: BLE001 - an older planlens, an odd page
            reliable = True
        return CitedPage(content.text(), markup_text="\n".join(marks),
                         text_reliable=reliable)

    def close(self) -> None:
        for doc in self._docs.values():
            try:
                doc.close()
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass
        self._docs.clear()

    def __enter__(self) -> "PlanlensPageText":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def planlens_page_text(sources: Optional[Mapping[str, Union[bytes, str]]]
                       ) -> PlanlensPageText:
    """A ``page_text`` callable reading ``sources`` (name -> bytes or path)
    with ``planlens.document.Document(...).page(i).text()``. Close it (or use
    it as a context manager) when done."""
    return PlanlensPageText(sources)


# ---------------------------------------------------------------------------
# The ledger: findings.json in the working folder
# ---------------------------------------------------------------------------

_LEDGER_LOCK = threading.RLock()
_ID = re.compile(r"^F(\d+)$", re.IGNORECASE)


def default_ledger_path() -> str:
    """``findings.json`` in the conversation's working folder."""
    from funhouse_agent._fileio import default_output_dir
    return os.path.abspath(os.path.join(default_output_dir() or ".",
                                        LEDGER_NAME))


def ledger_path_in(working_dir: str) -> str:
    """``findings.json`` in ``working_dir`` (a working folder bound when the
    agent was built)."""
    return os.path.abspath(os.path.join(os.fspath(working_dir), LEDGER_NAME))


class FindingsLedger:
    """The conversation's findings, kept as ``findings.json``.

    With no ``path`` the file is looked up in the working folder at EACH call,
    so one ledger object follows the conversation the host points the tools
    at. A missing file is an empty ledger; one that is not valid JSON is
    moved aside (``findings.unreadable.json``) rather than written over. A
    file that cannot be READ right now (a lock, a permission, a flaky share)
    raises ``OSError``: it is not corrupt, and a fresh ledger started over it
    would lose the findings. Safe to use from several threads of one process.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path

    @property
    def path(self) -> str:
        return (os.path.abspath(os.fspath(self._path)) if self._path
                else default_ledger_path())

    def load(self) -> List[Finding]:
        with _LEDGER_LOCK:
            return self._read(self.path)

    def _read(self, path: str) -> List[Finding]:
        if not os.path.isfile(path):
            return []
        # An OSError (the file is there but cannot be read just now) is
        # raised to the caller: only content that is not JSON is corrupt.
        with open(path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except ValueError:           # includes a UnicodeDecodeError
                corrupt = True
            else:
                corrupt = False
        if corrupt:
            self._set_aside(path)
            return []
        rows = data.get("findings", []) if isinstance(data, dict) else data
        out: List[Finding] = []
        for row in rows if isinstance(rows, list) else []:
            try:
                out.append(Finding.from_dict(row))
            except (ValueError, TypeError):
                continue
        return out

    @staticmethod
    def _set_aside(path: str) -> None:
        stem, ext = os.path.splitext(path)
        dest = f"{stem}.unreadable{ext}"
        n = 1
        while os.path.exists(dest):
            dest = f"{stem}.unreadable_{n}{ext}"
            n += 1
        try:
            os.replace(path, dest)
        except OSError:
            pass

    def save(self, findings: Iterable[Finding]) -> str:
        with _LEDGER_LOCK:
            return self._write(self.path, list(findings))

    @staticmethod
    def _write(path: str, findings: List[Finding]) -> str:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {"version": LEDGER_VERSION,
                   "findings": [f.to_dict() for f in findings]}
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    @staticmethod
    def _next_id(findings: Sequence[Finding]) -> str:
        nums = [int(m.group(1)) for m in (_ID.match(f.id) for f in findings)
                if m]
        return f"F{max(nums, default=0) + 1}"

    def add(self, finding: Union[Finding, Mapping[str, Any]]) -> Finding:
        """Keep ``finding`` under the next free id (``F1``, ``F2``, ...)."""
        finding = Finding.from_dict(finding)
        with _LEDGER_LOCK:
            path = self.path
            current = self._read(path)
            if not finding.id or any(f.id == finding.id for f in current):
                finding.id = self._next_id(current)
            current.append(finding)
            self._write(path, current)
        return finding

    def get(self, finding_id: str) -> Optional[Finding]:
        want = str(finding_id or "").strip().upper()
        return next((f for f in self.load() if f.id.upper() == want), None)

    def update(self, finding_id: str, **fields: Any) -> Finding:
        """Change fields of one finding (validated like a new one). Raises
        ``KeyError`` for an unknown id and ``ValueError`` for a bad value."""
        want = str(finding_id or "").strip().upper()
        fields.pop("id", None)
        with _LEDGER_LOCK:
            path = self.path
            current = self._read(path)
            for i, f in enumerate(current):
                if f.id.upper() == want:
                    data = f.to_dict()
                    data.update(fields)
                    current[i] = Finding.from_dict(data)
                    self._write(path, current)
                    return current[i]
        raise KeyError(f"no finding {finding_id!r}")

    def list(self, status: Any = None, severity: Any = None,
             discipline: Any = None) -> List[Finding]:
        """The findings, optionally only those with one of the given
        statuses / severities / disciplines (each a string or a list)."""
        def wanted(value: Any) -> Optional[set]:
            items = _str_list(value, "filter") if value not in (None, "") \
                else []
            return {i.lower() for i in items} or None

        st, sv, dp = wanted(status), wanted(severity), wanted(discipline)
        out = []
        for f in self.load():
            if st and f.status not in st:
                continue
            if sv and f.severity not in sv:
                continue
            if dp and not {d.lower() for d in f.disciplines} & dp:
                continue
            out.append(f)
        return out


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _cell(text: Any) -> str:
    return (str(text or "").replace("\r", " ").replace("\n", " ")
            .replace("|", "\\|").strip())


def _severity_rank(f: Finding) -> int:
    return -SEVERITIES.index(f.severity)


def _id_number(f: Finding) -> int:
    m = _ID.match(f.id)
    return int(m.group(1)) if m else 0


def sort_findings(findings: Iterable[Finding]) -> List[Finding]:
    """Most severe first, then in the order recorded."""
    return sorted(findings, key=lambda f: (_severity_rank(f), _id_number(f)))


def to_markdown(findings: Iterable[Finding]) -> str:
    """A comment log as a Markdown table (No., Severity, Finding, Where,
    Evidence), ready for ``write_docx``. Where cites the sheet, else the
    printed page, else the PDF viewer page; the document is named when the
    log covers more than one."""
    rows = list(findings)
    if not rows:
        return "_No findings recorded._\n"
    docs = {c.document for f in rows for c in f.citations}
    lines = ["| No. | Severity | Finding | Where | Evidence |",
             "|---|---|---|---|---|"]
    for f in rows:
        where = "; ".join(
            c.where() + (f" ({c.document})" if len(docs) > 1 else "")
            for c in f.citations)
        evidence = f"{f.evidence}, {f.confidence} confidence"
        if f.quote_verified is False:
            evidence += "; quote not found on the cited page"
        if f.status != "draft":
            evidence += f"; {f.status}"
        lines.append(f"| {_cell(f.id)} | {_cell(f.severity)} | "
                     f"{_cell(f.statement)} | {_cell(where)} | "
                     f"{_cell(evidence)} |")
    n = len(rows)
    lines.append("")
    lines.append(f"*Comment log: {n} finding{'s' if n != 1 else ''}. Drafts "
                 f"unless marked confirmed.*")
    return "\n".join(lines) + "\n"


def to_markups(findings: Iterable[Finding], document: str
               ) -> List[Dict[str, Any]]:
    """One markup per citation of ``document``, in the input shape planlens'
    ``annotate_document`` takes: a highlight on the quote, else a box on the
    bbox (also used when the finding's quotes were not found), else a note
    at the page's top-left corner. ``page`` is 0-based."""
    out: List[Dict[str, Any]] = []
    for f in findings:
        comment = f"{f.id} ({f.severity}): {f.statement}".strip()
        if f.confidence == "low":
            comment += " [low confidence]"
        for c in f.citations:
            if not _same_document(c.document, document):
                continue
            mark: Dict[str, Any] = {"page": c.page, "comment": comment}
            use_quote = bool(c.quote) and not (f.quote_verified is False
                                               and c.bbox)
            if use_quote:
                mark.update(kind="highlight", quote=c.quote)
            elif c.bbox:
                mark.update(kind="box", bbox=list(c.bbox))
            else:
                mark.update(kind="note", point=list(NOTE_POINT))
            out.append(mark)
    return out


__all__ = ["Citation", "Finding", "FindingsLedger", "PlanlensPageText",
           "CitedPage", "SEVERITIES", "CONFIDENCES", "EVIDENCE_KINDS",
           "STATUSES", "DEFAULT_MIN_SCORE", "LEDGER_NAME",
           "default_ledger_path", "ledger_path_in", "normalize_text",
           "quote_score", "verify_quotes", "planlens_page_text",
           "sort_findings", "to_markdown", "to_markups"]
