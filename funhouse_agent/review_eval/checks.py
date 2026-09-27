"""Deterministic checks on a review answer and the files it produced.

Each check is a dict with a ``type`` and its parameters; :func:`run_check`
returns ``(passed, detail)``. A check with ``"info": true`` is reported but not
scored. No check calls a model, so a score means the same thing on every run.

Terms. Wherever a check takes ``terms``, each term is one of:

* a string — found if it appears in the answer (case-insensitive, after
  :func:`normalize`: curly quotes, primes and dashes made ASCII, runs of
  whitespace collapsed);
* a list of strings/regexes — ALTERNATIVES, found if any one is;
* ``{"re": "..."}`` — a regular expression searched in the normalized answer
  (case-insensitive).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "′": "'", "´": "'",
    "“": '"', "”": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "−": "-", " ": " ", " ": " ",
})


def normalize(text: str) -> str:
    """Lower-case ASCII-ish text for matching: quotes/primes/dashes made
    ASCII, markdown emphasis and backticks dropped, whitespace collapsed."""
    t = str(text or "").translate(_TRANSLATE).lower()
    t = t.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", t)


def _one(term: Any, text: str) -> bool:
    if isinstance(term, dict) and "re" in term:
        return re.search(term["re"], text, flags=re.IGNORECASE) is not None
    return normalize(str(term)) in text


def term_found(term: Any, text: str) -> bool:
    """Whether ``term`` (string, alternatives list or regex) is in ``text``
    (already normalized)."""
    if isinstance(term, (list, tuple)):
        return any(_one(t, text) for t in term)
    return _one(term, text)


def _label(term: Any) -> str:
    if isinstance(term, (list, tuple)):
        return " | ".join(_label(t) for t in term)
    if isinstance(term, dict) and "re" in term:
        return f"/{term['re']}/"
    return str(term)


# ---------------------------------------------------------------------------
# Answer checks
# ---------------------------------------------------------------------------

def check_contains_all(answer: str, terms: Sequence[Any], **_) -> Tuple[bool, str]:
    text = normalize(answer)
    missing = [_label(t) for t in terms if not term_found(t, text)]
    return (not missing,
            "all present" if not missing else f"missing: {missing}")


def check_contains_any(answer: str, terms: Sequence[Any], **_) -> Tuple[bool, str]:
    text = normalize(answer)
    hit = [_label(t) for t in terms if term_found(t, text)]
    return (bool(hit), f"found: {hit}" if hit else
            f"none of: {[_label(t) for t in terms]}")


def check_not_contains(answer: str, terms: Sequence[Any], **_) -> Tuple[bool, str]:
    text = normalize(answer)
    hit = [_label(t) for t in terms if term_found(t, text)]
    return (not hit, "clean" if not hit else f"contains: {hit}")


def _mentions(text: str, item: str, aliases: Dict[str, Sequence[str]]) -> bool:
    forms = list(aliases.get(item) or [item])
    for f in forms:
        pat = re.escape(normalize(f))
        # An id must stand alone: "20.00a" must not match inside "120.00a",
        # and "11.01" must not match "11.015".
        if re.search(rf"(?<![\w.]){pat}(?![\w]|\.\d)", text):
            return True
    return False


def check_set_match(answer: str, vocabulary: Sequence[str],
                    expected: Sequence[str],
                    aliases: Optional[Dict[str, Sequence[str]]] = None,
                    min_recall: float = 1.0, min_precision: float = 0.0,
                    **_) -> Tuple[bool, str]:
    """The answer's mentions of ``vocabulary`` items, scored against the
    ``expected`` subset: recall and precision must both reach their floor.

    Mentions are counted over the whole answer, so an answer that names a
    sheet to rule it out ("30.01 prints 3600 cu ft, not concrete") is charged
    a false positive; ``min_precision`` is set with that in mind.
    """
    text = normalize(answer)
    aliases = aliases or {}
    said = {v for v in vocabulary if _mentions(text, v, aliases)}
    exp = set(expected)
    tp = len(said & exp)
    recall = tp / len(exp) if exp else 1.0
    precision = tp / len(said) if said else (1.0 if not exp else 0.0)
    ok = recall >= min_recall - 1e-9 and precision >= min_precision - 1e-9
    return ok, (f"recall {recall:.2f} (>= {min_recall}), precision "
                f"{precision:.2f} (>= {min_precision}); missing "
                f"{sorted(exp - said)}; extra {sorted(said - exp)}")


#: "page"/"p."/"pp." as a word of its own ("step. 12" and "mpg 30" are not
#: page citations).
_PAGE_WORD = r"(?<![a-z])(?:pdf\s+)?(?:pages?|pgs?\.?|pp\.|p\.)"
#: What separates the numbers in "pages 3, 5, and 7" / "pp. 3-7".
_PAGE_SEP = r"(?:,\s*and|,|and|&|-|to)"


def check_cites(answer: str, pages: Sequence[Any] = (),
                printed: Sequence[str] = (), sheets: Sequence[str] = (),
                **_) -> Tuple[bool, str]:
    """The answer cites the right place: a viewer page number (``pages``,
    1-based, written after "page"/"p."), a printed page number (``printed``,
    e.g. "5-2", anywhere) or a sheet id (``sheets``, anywhere)."""
    text = normalize(answer)
    for p in pages:
        if re.search(rf"{_PAGE_WORD}\s*(?:\d+\s*{_PAGE_SEP}\s*)*{int(p)}\b",
                     text):
            return True, f"cites page {p}"
    for p in printed:
        if re.search(rf"(?<![\w.-]){re.escape(normalize(p))}(?![\w])", text):
            return True, f"cites printed page {p}"
    for s in sheets:
        if _mentions(text, s, {}):
            return True, f"cites sheet {s}"
    return False, (f"no citation of pages {list(pages)}, printed "
                   f"{list(printed)} or sheets {list(sheets)}")


# ---------------------------------------------------------------------------
# File checks (on the files the agent wrote into its working folder)
# ---------------------------------------------------------------------------

def _files(files: Iterable[str], ext: str = "",
           name_contains: str = "") -> List[str]:
    out = []
    for f in files or ():
        base = os.path.basename(f).lower()
        if ext and not base.endswith(ext.lower()):
            continue
        if name_contains and name_contains.lower() not in base:
            continue
        out.append(f)
    return out


def check_file_produced(answer: str, files: Sequence[str] = (), ext: str = "",
                        name_contains: str = "", **_) -> Tuple[bool, str]:
    hits = _files(files, ext, name_contains)
    return (bool(hits), f"produced: {[os.path.basename(h) for h in hits]}"
            if hits else f"no {ext or 'file'} "
            f"{'named *' + name_contains + '* ' if name_contains else ''}"
            f"among {[os.path.basename(f) for f in files or ()]}")


def check_pdf_markups(answer: str, files: Sequence[str] = (), min: int = 1,
                      pages: Sequence[int] = (), text_contains: str = "",
                      author_contains: str = "", **_) -> Tuple[bool, str]:
    """A produced PDF carries at least ``min`` markups (on ``pages``, 0-based,
    if given; saying ``text_contains`` and signed ``author_contains`` if
    given), read back with planlens' own markup reader."""
    pdfs = _files(files, ".pdf")
    if not pdfs:
        return False, "no PDF produced"
    try:
        from planlens.document import Document
    except ImportError as exc:                       # pragma: no cover
        return False, f"planlens unavailable: {exc}"
    best = 0
    for path in pdfs:
        try:
            doc = Document(filepath=path)
            try:
                marks = doc.markups()
            finally:
                doc.close()
        except Exception as exc:                      # a broken PDF is a fail
            return False, f"{os.path.basename(path)} unreadable: {exc}"
        n = 0
        for m in marks:
            if pages and m.page not in pages:
                continue
            said = normalize(" ".join(str(x or "") for x in (
                getattr(m, "text", ""), getattr(m, "appearance_text", ""))))
            if text_contains and normalize(text_contains) not in said:
                continue
            if author_contains and normalize(author_contains) not in normalize(
                    getattr(m, "author", "") or ""):
                continue
            n += 1
        best = max(best, n)
    return best >= min, f"{best} matching markup(s) (need {min})"


def check_docx_contains(answer: str, files: Sequence[str] = (),
                        terms: Sequence[Any] = (), **_) -> Tuple[bool, str]:
    docs = _files(files, ".docx")
    if not docs:
        return False, "no .docx produced"
    try:
        import docx
    except ImportError as exc:                       # pragma: no cover
        return False, f"python-docx unavailable: {exc}"
    texts = []
    for path in docs:
        try:
            d = docx.Document(path)
        except Exception as exc:
            return False, f"{os.path.basename(path)} unreadable: {exc}"
        texts += [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                texts += [c.text for c in row.cells]
    ok, detail = check_contains_all("\n".join(texts), terms)
    return ok, detail


# ---------------------------------------------------------------------------
# Process checks (informational by default: outcomes are what is scored)
# ---------------------------------------------------------------------------

def check_tool_used(answer: str, tool_calls: Sequence[Dict[str, Any]] = (),
                    any: Sequence[str] = (), min_calls: int = 1,
                    **_) -> Tuple[bool, str]:
    names = [c.get("name") for c in tool_calls or ()]
    n = sum(1 for x in names if x in set(any))
    return n >= min_calls, f"{n} call(s) of {list(any)}"


CHECKS = {
    "contains_all": check_contains_all,
    "contains_any": check_contains_any,
    "not_contains": check_not_contains,
    "set_match": check_set_match,
    "cites": check_cites,
    "file_produced": check_file_produced,
    "pdf_markups": check_pdf_markups,
    "docx_contains": check_docx_contains,
    "tool_used": check_tool_used,
}


def run_check(check: Dict[str, Any], answer: str,
              files: Sequence[str] = (),
              tool_calls: Sequence[Dict[str, Any]] = ()) -> Dict[str, Any]:
    """Run one check; never raises (a broken check is a failed check)."""
    kind = check.get("type")
    fn = CHECKS.get(kind)
    params = {k: v for k, v in check.items()
              if k not in ("type", "label", "info")}
    label = check.get("label") or kind
    if fn is None:
        return {"type": kind, "label": label, "passed": False,
                "info": bool(check.get("info")),
                "detail": f"unknown check type {kind!r}"}
    try:
        ok, detail = fn(answer or "", files=files, tool_calls=tool_calls,
                        **params)
    except Exception as exc:
        ok, detail = False, f"check error: {type(exc).__name__}: {exc}"
    return {"type": kind, "label": label, "passed": bool(ok),
            "info": bool(check.get("info")), "detail": detail}


def score(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Scored checks passed / total, and whether the task passed (all did)."""
    scored = [r for r in results if not r.get("info")]
    passed = sum(1 for r in scored if r["passed"])
    return {"checks_passed": passed, "checks_total": len(scored),
            "passed": bool(scored) and passed == len(scored)}


__all__ = ["normalize", "term_found", "run_check", "score", "CHECKS"]
