"""Page numbers a tool takes: 0-based ``page`` or 1-based ``pdf_page``, and a
clear error -- never a raise -- for one the document does not have.

Live smoke wave 2c (the F47 re-run): ``analyze_pdf_page`` RAISED
``IndexError: page(s) [5] out of range: this document has 5 pages, numbered
0-4`` when the model meant PDF page 5, the number a viewer shows. The tools
count pages from 0 (planlens does), the results already carry the viewer's
1-based twin (``pdf_page``, wave 2b C9), and a model reading "page 5 of 5"
off a viewer -- or off the user -- passes 5.

* :func:`resolve_page` takes ``page`` (0-based) or ``pdf_page`` (1-based),
  refuses the two naming different pages, and never guesses which was
  meant.
* :func:`page_error` is the JSON-ready error for a page out of range, with
  the hint "pages are 0-based here; PDF page 5 is page 4" when the page is
  exactly one past the end (the usual off-by-one).
* :func:`range_hint` adds that hint to an out-of-range error text another
  layer wrote (planlens' document tools, ``find_like``).
* :func:`pdf_pages_to_pages` turns a 1-based page spec ("1-3,6") into the
  0-based one the document tools take ("0-2,5").

Pure Python, no third-party import.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

#: What every page-taking tool's description can say about the two numbers.
PAGE_ARGS_NOTE = ("page is 0-based (the first page is 0); or give pdf_page "
                  "instead, 1-based, the number a PDF viewer shows")


def _whole(value: Any, name: str) -> Tuple[Optional[int], Optional[str]]:
    if isinstance(value, bool):
        return None, f"{name} must be a whole number, not {value!r}"
    try:
        n = int(str(value).strip()) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a whole number, not {value!r}"
    if isinstance(value, float) and value != n:
        return None, f"{name} must be a whole number, not {value!r}"
    return n, None


def resolve_page(page: Any = None, pdf_page: Any = None,
                 default: int = 0) -> Tuple[Optional[int], Optional[Dict]]:
    """``(page, None)`` -- the 0-based page meant -- or ``(None, error)``.

    ``pdf_page`` (1-based) wins when it is the only one given; both given
    must name the same page, or the call is refused naming both (no silent
    choice between them). Neither given is ``default``."""
    p = None
    if page is not None and page != "":
        p, problem = _whole(page, "page")
        if problem:
            return None, {"error": problem, "hint": PAGE_ARGS_NOTE}
    if pdf_page is None or pdf_page == "":
        return (default if p is None else p), None
    pp, problem = _whole(pdf_page, "pdf_page")
    if problem:
        return None, {"error": problem, "hint": PAGE_ARGS_NOTE}
    if pp < 1:
        return None, {"error": f"pdf_page {pp} is not a page: pdf_page "
                               f"counts from 1, as a PDF viewer does",
                      "hint": PAGE_ARGS_NOTE}
    if p is not None and p != pp - 1:
        return None, {
            "error": (f"page {p} (0-based) and pdf_page {pp} (1-based) name "
                      f"different pages (pdf_page {pp} is page {pp - 1})"),
            "hint": "give one of them: " + PAGE_ARGS_NOTE}
    return pp - 1, None


def page_error(page: int, n_pages: int,
               as_pdf_page: bool = False) -> Optional[Dict[str, str]]:
    """The error for a 0-based ``page`` this ``n_pages``-page document does
    not have, else ``None``. ``as_pdf_page`` words it in the 1-based number
    the caller gave."""
    if n_pages is None or 0 <= page < n_pages:
        return None
    if n_pages <= 0:
        return {"error": "this document has no pages"}
    if as_pdf_page:
        err = (f"pdf_page {page + 1} is out of range: this document has "
               f"{n_pages} pages, pdf_page 1-{n_pages}")
        hint = f"pdf_page counts from 1 here; the last page is {n_pages}"
    else:
        err = (f"page {page} is out of range: this document has {n_pages} "
               f"pages, numbered 0-{n_pages - 1} here (pdf_page "
               f"1-{n_pages})")
        hint = (f"pages are 0-based here; PDF page {page} is page "
                f"{page - 1}" if page == n_pages else
                f"pass page 0-{n_pages - 1}, or pdf_page 1-{n_pages} (the "
                f"number a PDF viewer shows)")
    return {"error": err, "hint": hint}


#: planlens' and PyMuPDF's out-of-range wordings.
_RANGE_PATTERNS = (
    # planlens.document: "page(s) [5] out of range: this document has 5
    # pages, numbered 0-4"
    re.compile(r"page\(s\) \[([\d,\s]+)\] out of range: (?:this|the) "
               r"document has (\d+) pages", re.IGNORECASE),
    # planlens.pdf / ir: "Page 5 out of range (document has 5 pages)"
    re.compile(r"page (\d+) (?:is )?out of range[^\d]*?(\d+) pages",
               re.IGNORECASE),
    # planlens markup writer: "page 5 is outside this document, which has
    # pages 0-4"
    re.compile(r"page (\d+) is outside this document, which has pages "
               r"0-(\d+)", re.IGNORECASE),
)


def _bad_and_count(text: str) -> Tuple[List[int], Optional[int]]:
    for i, pattern in enumerate(_RANGE_PATTERNS):
        m = pattern.search(text or "")
        if not m:
            continue
        bad = [int(x) for x in re.findall(r"\d+", m.group(1))]
        n = int(m.group(2)) + (1 if i == 2 else 0)
        return bad, n
    return [], None


def range_hint(text: str, one_based: bool = False) -> Optional[str]:
    """A hint for an out-of-range error text, or ``None`` when ``text`` is
    not one: "pages are 0-based here; PDF page 5 is page 4" when a page is
    exactly one past the end; with ``one_based`` (the caller gave
    ``pdf_pages``) the viewer's range instead."""
    bad, n = _bad_and_count(text)
    if n is None:
        return None
    if one_based:
        return (f"you gave 1-based pdf_pages: this document's pages are "
                f"pdf_page 1-{n}")
    if n in bad:
        return f"pages are 0-based here; PDF page {n} is page {n - 1}"
    return (f"pages are 0-based here: 0-{n - 1} for this document (pdf_page "
            f"1-{n})")


_SPEC = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def pdf_pages_to_pages(spec: Any) -> Tuple[Any, Optional[Dict[str, str]]]:
    """A 1-based page spec -- an int, a list of ints, or "1-3,6" -- as the
    0-based spec the document tools take; ``(None, error)`` for a spec with
    a page 0 or below, or one that is not a page spec."""
    def bad(why):
        return None, {"error": f"pdf_pages {spec!r}: {why}",
                      "hint": ("pdf_pages is 1-based, as a viewer shows: an "
                               "int, a list, or a range like '1-3,6'")}
    if isinstance(spec, bool):
        return bad("not a page number")
    if isinstance(spec, int):
        return (spec - 1, None) if spec >= 1 else bad("pages count from 1")
    if isinstance(spec, (list, tuple)):
        out = []
        for v in spec:
            n, problem = _whole(v, "pdf_pages")
            if problem or n < 1:
                return bad("pages count from 1")
            out.append(n - 1)
        return out, None
    parts = []
    for part in str(spec).split(","):
        if not part.strip():
            continue
        m = _SPEC.match(part)
        if not m:
            return bad(f"'{part.strip()}' is not a page or a range")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else None
        if a < 1 or (b is not None and b < 1):
            return bad("pages count from 1")
        parts.append(f"{a - 1}" if b is None else f"{a - 1}-{b - 1}")
    if not parts:
        return bad("no pages named")
    return ",".join(parts), None


__all__ = ["resolve_page", "page_error", "range_hint", "pdf_pages_to_pages",
           "PAGE_ARGS_NOTE"]
