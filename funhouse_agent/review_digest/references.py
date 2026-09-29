"""Cross-references in a document's text, and the sheet labels they point at.

A reviewer's coverage questions often turn on what one page points at: the
detail "3/S-501", the county standard "STD. NO. 10.17", the specification
section "03 30 00", "Table 5-1", "Appendix B". This module finds those in the
exact text of a page, with where they sit, and says which cited sheets and
standards a set of uploads does not contain.

EVERY PATTERN NEEDS ITS KEYWORD (or, for a detail callout, its slash): a bare
"5-1" is a printed page number, a paragraph or a range as often as a table,
so it is never read as one. Each pattern is written to generalise over how
the profession prints a kind of reference and is held down where it would
fire on something else:

* a sheet, detail or standard number followed by a unit ("SHEET 5 mm",
  "standard 1.5 in") is a size, not a reference;
* "SHEET 1 OF 12" is a sheet count;
* "#4" is a reinforcing bar, so an agency's "#" form ("MCLDS #10.35B") needs
  a dotted number, and "STANDARD" written out needs one too unless "NO." or
  "#" says a number follows;
* a two-letter English word ("APPENDIX TO", "DETAIL AS SHOWN") is not an id;
* a MasterFormat number printed without the word "Section" ("03 30 00")
  needs a cue before it ("per", "see", "spec") or its title after it, so a
  row of three two-digit numbers in a table is not a specification.

A table, figure or appendix that STARTS a line and is followed by a title (or
nothing) is its CAPTION - the place it is, not a pointer to it - and is kept
with ``role="caption"`` and its title; the same line in a contents list (a
dotted leader) is ``role="listed"``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

#: Kinds of reference, in the order results are grouped.
KINDS = ("sheet", "detail", "standard", "spec_section", "section", "table",
         "figure", "appendix", "attachment")

#: Kinds whose target is a SHEET of a set: what the missing-sheet check uses.
SHEET_KINDS = ("sheet", "detail", "standard")

#: Two-letter words that are never an appendix, detail or issuer id.
_STOP_WORDS = frozenset(
    "AS IS OF ON IN AT TO BY OR NO IF BE IT AN UP SO DO WE US MY HE ME GO AM "
    "PM OK RE".split())

#: Words before "#"/"STD" that are not an issuing agency.
_NOT_ISSUER = frozenset(
    "SEE PER AND THE WITH USE FOR TO OF IN ON AS AT BY OR NO NOT ALL ANY "
    "EACH FROM INTO ONTO SHALL WILL MAY MUST BE IS ARE REF REFER NOTE NOTES "
    "TYP TYPICAL SIM SIMILAR DETAIL DETAILS SHEET STD STANDARD NEW EXISTING "
    "USING BAR BARS REBAR".split())

#: A unit right after a number makes it a size, not a reference.
_UNIT_AFTER = (r"(?!\s*(?:%|°|\"|'|(?i:in\b|in\.|inch|inches|ft\b|feet|foot|"
               r"mm\b|cm\b|m\b|km\b|kn\b|kpa|mpa|psi|psf|pcf|ksi|ksf|lbs?\b|"
               r"kips?\b|deg\b|degrees)))")

#: Sheet ids as sets number them: S-501, C3.01, A101, M-1, E2.1A.
SHEET_ID = r"[A-Z]{1,3}[-.]?\d{1,4}(?:\.\d{1,3})?[A-Z]?"
#: A standard or detail number: 10.17, 10.35B, 840.54, 6.60.
_STD_ID = r"\d{1,4}(?:\.\d{1,3}){0,2}[A-Z]?"
#: A table or figure number: 5-1, A-2, 12.2-1, 4-1(a), IV, B.
_TF_ID = (r"(?:(?:[A-Z]{1,2}-?)?\d{1,3}(?:[.-]\d{1,3}){0,2}"
          r"(?:\([a-z0-9]{1,2}\))?[a-z]?(?![\w])|[IVX]{2,4}(?![\w])"
          r"|[A-Z](?![\w]))")
#: After an id: not more of an id. A full stop ends a sentence ("DETAIL
#: 11.51."), a full stop and a digit continues a number ("11.51.3").
_ID_END = r"(?![\w]|\.\d)"
_APX_ID = r"(?:[IVX]{2,4}(?![\w])|[A-Z]{1,2}(?![\w])|\d{1,2}" + _ID_END + ")"

_NO = r"(?:(?i:no\.?|nos\.?|number)\s*|#\s*)?"

# A reference's keyword (or keywords) up to where its id starts. Case: the
# keyword is read in any case, the id only as printed (an id is capitals or
# digits), which is what keeps "table of contents" from being "Table OF".
_HEADS: Dict[str, "re.Pattern[str]"] = {
    "sheet": re.compile(r"\b(?i:sheets?|shts?\.?|dwgs?\.?|drawings?)"
                        r"(?![A-Za-z])\.?\s*" + _NO),
    "detail": re.compile(r"\b(?i:details?|dets?\.)(?![A-Za-z])\s*" + _NO),
    "standard": re.compile(
        r"(?:\b(?P<issuer>[A-Z][A-Z0-9&]{1,9})\s+)?"
        r"\b(?P<kw>STDS?|(?i:standards?))(?![A-Za-z])\.?\s*"
        r"(?:(?i:dwg|detail|drawing)\.?\s*)?(?P<no>" + _NO + r")#?\s*"),
    "section": re.compile(r"(?:\b(?i:sections?|sect?\.)(?![A-Za-z])|§)"
                          r"\s*(?i:no\.?\s*)?"),
    "table": re.compile(r"\b(?i:tables?|tbls?\.)(?![A-Za-z])\s*"
                        r"(?i:no\.?\s*)?"),
    "figure": re.compile(r"\b(?i:figures?|figs?\.?)(?![A-Za-z])\s*"
                         r"(?i:no\.?\s*)?"),
    "appendix": re.compile(r"\b(?i:appendix|appendices|appendixes|appx\.?)"
                           r"(?![A-Za-z])\s*"),
    "attachment": re.compile(r"\b(?i:attachments?|exhibits?|annex(?:es)?|"
                             r"enclosures?)(?![A-Za-z])\s*"),
}

_IDS: Dict[str, "re.Pattern[str]"] = {
    "sheet": re.compile(r"(?:" + SHEET_ID + r"(?![\w])|\d{1,3}" + _ID_END
                        + ")"
                        + _UNIT_AFTER),
    "detail": re.compile(r"(?:\d{1,4}(?:\.\d{1,3})?[A-Z]?" + _ID_END
                         + _UNIT_AFTER + r"|[A-Z]{1,2}\d{0,2}(?![\w]))"
                         r"(?:\s*/\s*(?P<on>" + SHEET_ID + r")(?![\w/]))?"),
    "standard": re.compile(_STD_ID + r"(?![\w])" + _UNIT_AFTER),
    "section": re.compile(
        r"(?P<mf>[0-4]\d[ \u00a0]\d{2}[ \u00a0]\d{2}(?:\.\d{2})?(?![\d])"
        r"|\d{6}(?:\.\d{2})?(?![\d])|\d{5}(?![\d]))"
        r"|\d{1,4}(?:[.-]\d{1,3}){0,4}(?:\([a-z0-9]{1,2}\))?[A-Za-z]?(?![\w])"
        + _UNIT_AFTER),
    "table": re.compile(_TF_ID),
    "figure": re.compile(_TF_ID),
    "appendix": re.compile(_APX_ID),
    "attachment": re.compile(_APX_ID),
}

#: What joins the ids of a list: "Tables 8-3, 8-4, or 8-5", "A through N".
_CONT = re.compile(r"\s*(?:,\s*(?:and|or|&)?|and|or|&|through|thru|to)\s*",
                   re.I)

#: An agency's own number: "MCLDS #10.35B", "NCESCPDM #6.60". Dotted only:
#: "#4" is a reinforcing bar.
_AGENCY = re.compile(r"\b(?P<issuer>[A-Z][A-Z0-9&]{1,9})\s*#\s*"
                     r"(?P<id>\d{1,4}\.\d{1,3}(?:\.\d{1,3})?[A-Z]?)(?![\w])"
                     + _UNIT_AFTER)

#: A detail callout: detail 3 on sheet S-501 printed "3/S-501". Numbered
#: details only: "F/A-18" is an aircraft, so a lettered detail needs the
#: word DETAIL ("DETAIL A/S-501").
_ON_SHEET = re.compile(r"(?<![\w/.])(?P<d>\d{1,2})\s*/\s*"
                       r"(?P<s>[A-Z]{1,3}[-.]?\d{1,4}(?:\.\d{1,3})?[A-Z]?)"
                       r"(?![\w/])")

#: A MasterFormat number with no "Section" before it.
_BARE_MF = re.compile(r"(?<![\d.,:/-])(?P<id>[0-4]\d[ \u00a0]\d{2}[ \u00a0]"
                      r"\d{2}(?:\.\d{2})?)(?![\d,:/]|\.\d)")
_MF_CUE_BEFORE = re.compile(
    r"(?i:\b(?:spec(?:ification)?s?|sections?|division|per|see|refer(?:\s+to)?"
    r"|in\s+accordance\s+with|conform(?:ing|s)?\s+to|under))\W{0,3}$")
_MF_TITLE_AFTER = re.compile(r"^\s*[-\u2013\u2014:]?\s*(?P<w>[A-Z][A-Z-]{3,})")
_UNIT_WORDS = frozenset(
    "FEET FOOT INCH INCHES YARD YARDS METER METERS METRE METRES MILE MILES "
    "POUNDS KIPS TONS DAYS HOURS YEARS MONTHS WEEKS PERCENT DEGREES PSI"
    .split())

#: A contents-list line: a dotted leader before a page number.
_LEADER = re.compile(r"(?:\.\s?){4,}|\u2026{2,}")


def _shape(ident: str) -> Tuple[bool, bool, bool]:
    """What a list's later ids must look like to belong to it: "8-3, 8-4" is
    a list, "Table 5-1 and 50" is not."""
    return ("-" in ident, "." in ident, ident.isalpha())


def _ok_word(ident: str) -> bool:
    return ident.upper() not in _STOP_WORDS


def _accept(kind: str, ident: str, head: "re.Match[str]") -> bool:
    if kind in ("appendix", "attachment", "detail", "table", "figure"):
        core = ident.split("/")[0].strip()
        if core.isalpha() and not _ok_word(core):
            return False
    if kind == "standard":
        has_no = bool((head.group("no") or "").strip())
        dotted = "." in ident
        kw = head.group("kw")
        if not has_no:
            if kw.upper().startswith("STANDARD") and not dotted:
                return False           # "standard 2" in prose
            if kw.startswith("STD") and not dotted and len(ident) < 2:
                return False
    return True


def _issuer(match: "re.Match[str]") -> Optional[str]:
    try:
        issuer = match.group("issuer")
    except IndexError:
        return None
    if issuer and issuer.upper() not in _NOT_ISSUER:
        return issuer
    return None


def normalize_id(kind: str, ident: str) -> str:
    """The id as it is shown: capitals, single spaces."""
    ident = " ".join(str(ident).replace("\u00a0", " ").split())
    if kind in ("sheet", "standard", "detail"):
        ident = re.sub(r"\s*/\s*", "/", ident.upper())
    elif kind == "spec_section":
        digits = re.sub(r"\D", "", ident.split(".")[0])
        tail = ident.split(".")[1] if "." in ident else ""
        if len(digits) == 6:
            ident = f"{digits[:2]} {digits[2:4]} {digits[4:]}"
        if tail:
            ident += "." + tail
    elif kind in ("appendix", "attachment"):
        ident = ident.upper()
    return ident


def sheet_key(ident: str) -> str:
    """A sheet or standard id reduced for comparison: "S-501", "S 501" and
    "s501" are one sheet; "10.35B" stays "10.35B"."""
    s = re.sub(r"[\s#]", "", str(ident).upper())
    return re.sub(r"^([A-Z]{1,3})[-.]?(?=\d)", r"\1", s)


def target_of(kind: str, ident: str, drawing_no: bool = True
              ) -> Optional[str]:
    """The sheet a reference points at, for the missing-sheet check, or
    ``None`` when it does not name one a set would be labelled with.

    A detail "3/S-501" points at S-501; "DETAIL 11.51" (a standard detail
    numbered like a sheet) at 11.51; "DETAIL 3" or "DETAIL A" at a drawing
    on the same sheet - nothing to look for. A bare "SHEET 5" names a
    position in a set, not a label, and is left out too. A standard counts
    when it is numbered like a standard DRAWING (``drawing_no``: a dotted
    number, or "NO."/"#" before it): "STD. NO. 10.17" is a sheet, "MIL-STD
    3007" is a whole document.
    """
    if kind not in SHEET_KINDS:
        return None
    if "/" in ident:
        return ident.split("/", 1)[1]
    if kind == "standard":
        return ident if drawing_no else None
    if re.fullmatch(r"\d{1,4}\.\d{1,3}[A-Z]?", ident) or re.fullmatch(
            SHEET_ID, ident) and re.search(r"[A-Z]", ident) and re.search(
            r"\d", ident) and (re.search(r"[-.]", ident)
                               or len(re.sub(r"\D", "", ident)) >= 2):
        return ident
    return None


# ---------------------------------------------------------------------------
# Finding them
# ---------------------------------------------------------------------------

def _candidates(text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for kind, head_rx in _HEADS.items():
        id_rx = _IDS[kind]
        for head in head_rx.finditer(text):
            pos = head.end()
            m = id_rx.match(text, pos)
            if not m or not m.group(0).strip():
                continue
            ident = m.group(0)
            if kind == "sheet" and re.match(r"\s+(?i:of)\s+\d",
                                            text[m.end():]):
                continue                           # "SHEET 1 OF 12"
            this_kind = kind
            if kind == "section" and m.groupdict().get("mf"):
                this_kind = "spec_section"
            if not _accept(kind, ident, head):
                continue
            # A standard's issuer word is part of the printed reference.
            start = head.start()
            if kind == "standard" and head.group("issuer") \
                    and not _issuer(head):
                start = head.start("kw")
            drawing_no = True
            if kind == "standard":
                drawing_no = ("." in ident
                              or bool((head.group("no") or "").strip()))
            out.append({"kind": this_kind, "id": ident, "start": start,
                        "end": m.end(), "issuer": _issuer(head)
                        if kind == "standard" else None,
                        "drawing_no": drawing_no})
            # A list: "Tables 8-3, 8-4, or 8-5", "Appendices A through N".
            shape = _shape(ident)
            end = m.end()
            while True:
                c = _CONT.match(text, end)
                if not c:
                    break
                n = id_rx.match(text, c.end())
                if not n or _shape(n.group(0)) != shape:
                    break
                nxt = n.group(0)
                if not _accept(kind, nxt, head):
                    break
                nk = kind
                if kind == "section" and n.groupdict().get("mf"):
                    nk = "spec_section"
                out.append({"kind": nk, "id": nxt, "start": n.start(),
                            "end": n.end(), "issuer": None,
                            "drawing_no": drawing_no})
                end = n.end()
    for m in _AGENCY.finditer(text):
        if _issuer(m):
            out.append({"kind": "standard", "id": m.group("id"),
                        "start": m.start(), "end": m.end(),
                        "issuer": m.group("issuer")})
    for m in _ON_SHEET.finditer(text):
        sheet = m.group("s")
        digits = re.sub(r"\D", "", sheet)
        if not (re.search(r"[-.]", sheet) or len(digits) >= 2):
            continue
        out.append({"kind": "detail", "id": f"{m.group('d')}/{sheet}",
                    "start": m.start(), "end": m.end(), "issuer": None})
    for m in _BARE_MF.finditer(text):
        before = text[max(0, m.start() - 30):m.start()]
        after = _MF_TITLE_AFTER.match(text[m.end():])
        if _MF_CUE_BEFORE.search(before) or (
                after and after.group("w").upper() not in _UNIT_WORDS):
            out.append({"kind": "spec_section", "id": m.group("id"),
                        "start": m.start(), "end": m.end(), "issuer": None})
    return out


def _dedupe(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Where two patterns claim overlapping text, the longer claim wins
    ("DETAIL 3/S-501" over its bare "3/S-501")."""
    cands.sort(key=lambda c: (c["start"], -(c["end"] - c["start"])))
    kept: List[Dict[str, Any]] = []
    for c in cands:
        clash = None
        for k in kept:
            if c["start"] < k["end"] and k["start"] < c["end"]:
                clash = k
                break
        if clash is None:
            kept.append(c)
        elif (c["end"] - c["start"]) > (clash["end"] - clash["start"]) and (
                c["start"] <= clash["start"] and c["end"] >= clash["end"]):
            kept[kept.index(clash)] = c
    return kept


#: Words a caption's title does not use and a sentence about the table does
#: ("Table 4-1 presents ...", "Table 3-1, ..., must be used in lieu of").
_SENTENCE_WORDS = frozenset(
    "is are was were be been shall should will would may must can could "
    "show shows shown present presents presented give gives given list lists "
    "listed illustrate illustrates provide provides contain contains "
    "summarize summarizes summarise summarises indicate indicates "
    "include includes apply applies used use see".split())


def _role(kind: str, text: str, ref: Dict[str, Any],
          line_starts: Sequence[int]) -> Tuple[str, Optional[str]]:
    """``(role, title)``: "caption" when a table, figure or appendix starts
    its line and a title (or nothing) follows; "listed" when that line is a
    contents entry; else "ref".

    A line that starts with "Table 12-3." can still be the middle of a
    sentence that wrapped ("... listed in / Table 12-3.  Divide the ..."),
    so a line whose block runs into it without ending a sentence is a
    reference, and so is one whose words read as a sentence about it."""
    if kind not in ("table", "figure", "appendix", "attachment"):
        return "ref", None
    if ref["start"] not in line_starts:
        return "ref", None
    idx = line_starts.index(ref["start"])
    if idx > 0:
        prev = text[line_starts[idx - 1]:ref["start"]].rstrip()
        if prev and not prev.endswith((".", ":", ";", "!", "?")):
            return "ref", None
    line_end = (line_starts[idx + 1] - 1 if idx + 1 < len(line_starts)
                else len(text))
    line = text[ref["start"]:line_end]
    rest = text[ref["end"]:line_end]
    if _LEADER.search(line):
        title = _LEADER.split(rest)[0]
        return "listed", _clean_title(title)
    stripped = rest.lstrip()
    if stripped[:1] in (".", ":", "|", "\u2013", "\u2014", "-", ","):
        stripped = stripped[1:].lstrip()
    if not stripped:
        return "caption", None
    if stripped[0].islower():
        return "ref", None
    words = [w.lower() for w in
             re.findall(r"[A-Za-z][A-Za-z'\u2019-]*", stripped)[:10]]
    if any(w in _SENTENCE_WORDS for w in words):
        return "ref", None
    return "caption", _clean_title(stripped)


def _clean_title(title: str) -> Optional[str]:
    title = " ".join(str(title or "").split()).strip(" .:-\u2013\u2014|")
    if not title:
        return None
    return title if len(title) <= 120 else title[:117] + "..."


def find_references(text: str, line_starts: Iterable[int] = (0,)
                    ) -> List[Dict[str, Any]]:
    """Every reference in ``text`` (one text block, lines joined by a space),
    in reading order: ``kind``, ``id`` (as shown), ``text`` (as printed),
    ``start``/``end`` offsets, ``role`` ("ref", "caption", "listed"), and
    ``title`` for a caption, ``issuer`` for an agency's standard and
    ``target`` for a reference to a sheet of a set. ``line_starts`` are the
    offsets where the block's lines begin (captions start a line)."""
    starts = sorted(set(int(s) for s in line_starts)) or [0]
    out = []
    for c in _dedupe(_candidates(text)):
        kind = c["kind"]
        ident = normalize_id(kind, c["id"])
        role, title = _role(kind, text, c, starts)
        ref: Dict[str, Any] = {"kind": kind, "id": ident,
                               "text": " ".join(text[c["start"]:c["end"]]
                                                .split())[:80],
                               "start": c["start"], "end": c["end"],
                               "role": role}
        if title:
            ref["title"] = title
        if c.get("issuer"):
            ref["issuer"] = c["issuer"]
        tgt = target_of(kind, ident, c.get("drawing_no", True))
        if tgt:
            ref["target"] = tgt
        out.append(ref)
    return out


# ---------------------------------------------------------------------------
# Sheet labels and the missing-sheet rule
# ---------------------------------------------------------------------------

#: A sheet label as a set prints it in its title block.
SHEET_LABEL = re.compile(
    r"(?:[A-Z]{1,3}[-.]?\d{1,4}(?:\.\d{1,3})?[A-Z]?"
    r"|\d{1,4}(?:\.\d{1,3})+[A-Z]?|\d{1,3}[A-Z]?)")
#: The title-block word that labels a sheet's number.
LABEL_WORD = re.compile(
    r"^\s*(?:SHEET|SHT|DWG|DRAWING|STD|STANDARD|DETAIL|PLATE)\.?\s*"
    r"(?:NO\.?|NUMBER|#)?\s*[:#]?\s*$", re.I)
LABEL_INLINE = re.compile(
    r"^\s*(?:SHEET|SHT|DWG|DRAWING|STD|STANDARD)\.?\s*(?:NO\.?|NUMBER|#)\s*"
    r"[:#]?\s*(?P<v>\S+)\s*$", re.I)


def is_sheet_label(text: str, allow_plain: bool = False) -> bool:
    """Whether ``text`` reads as a sheet label. A plain number ("5") only
    beside a label word (``allow_plain``)."""
    t = str(text or "").strip().rstrip(".").upper()
    if not SHEET_LABEL.fullmatch(t):
        return False
    if not allow_plain and re.fullmatch(r"\d{1,3}[A-Z]?", t):
        return False
    return True


def label_matches(ref_target: str, labels: Iterable[str]) -> Optional[str]:
    """The sheet label that satisfies a reference, or ``None``.

    Equal ids match ("S-501" = "S501"). And one more rule: a STANDARD is
    drawn on LETTERED sheets - county and state standard details print
    10.17A, 10.17B for the sheets of standard 10.17 - so a reference to the
    standard ("SEE STD. NO. 10.17") is satisfied by any one of its lettered
    sheets. The reverse is not assumed: a reference to sheet 10.35B is not
    satisfied by a sheet labelled 10.35, which says nothing about sheet B.
    Only a single trailing letter after a digit counts, so "10.1" is not
    "10.17" and "S-50" is not "S-501".
    """
    key = sheet_key(ref_target)
    keys = {sheet_key(l): l for l in labels}
    if key in keys:
        return keys[key]
    if key and key[-1].isdigit():
        for k, label in keys.items():
            if len(k) == len(key) + 1 and k.startswith(key) \
                    and k[-1].isalpha():
                return label
    return None


__all__ = ["KINDS", "SHEET_KINDS", "SHEET_ID", "find_references",
           "normalize_id", "sheet_key", "target_of", "is_sheet_label",
           "label_matches", "LABEL_WORD", "LABEL_INLINE", "SHEET_LABEL"]
