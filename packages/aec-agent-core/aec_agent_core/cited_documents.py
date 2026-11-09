"""Published documents an answer or a written file cites, checked against
what this conversation actually looked up (BACKLOG EV-21).

WHY. In live smoke wave w8b (``module_work/live_smoke/runs/w8b-plot-code/
TRACE_REVIEW.md`` §4.2, finding 1) the main agent told the calc helper to
"cite Rankine (1857) and GEC-11 / AASHTO LRFD 3.11.5". The user named
neither GEC-11 nor AASHTO, and no references tool ran in the turn. The PDF
printed the citation "as given by the project, not independently verified",
and the answer told the user the citations were "as you stated them". GEC-11
is FHWA's manual for MSE walls, not cantilever walls. Nothing compared a
cited document with what had been looked up, as EV-18 (:mod:`funhouse_agent.
cited_pages`) does for pages.

WHAT IS CHECKED, AND WHERE

* The ANSWER: after each turn every document the reply cites is checked
  (:func:`check_text`). The result goes into the turn's record
  (``activity.jsonl``, event :data:`RECORD_EVENT`) whenever the reply cites
  one. Behind ``GEOTECH_CITED_DOCUMENTS`` (:func:`funhouse_agent.
  review_flags.cited_documents`, OFF by default): ``note`` appends a short
  note to the answer naming the citations nothing looked up; ``retry`` tells
  the model once, before the answer goes out, and the reply after that note
  is the answer (with the note appended if it still cites them). The
  :class:`TurnCitationCheck` object carries both for ``webapp.core.
  stream_turn``, beside EV-18's page check.
* DELIVERABLES the turn writes: ``calc_package`` ``html_to_pdf`` and the
  ``*_package`` methods (``dispatch.call_agent``), ``write_docx`` and a text
  report through ``save_file`` (``vision_tools.dispatch_extended_tool``) and
  the notes of ``annotate_document`` (``document_tools.write_marked_copy``):
  the text the MODEL gave the writer is checked (:func:`check_deliverable`),
  never the app's own template text. With the switch on (``note`` or
  ``retry``) the tool result gains ``cited_not_looked_up`` and one plain
  ``citation_note``. The file is written all the same: the check never
  refuses a deliverable. Whatever the switch, a writer's check that finds a
  citation is recorded in the activity log as a :data:`RECORD_EVENT` event
  (``where`` names the writer) when the call runs inside a turn.

WHAT COUNTS AS A CITATION (:func:`find_citations`). Generous in what is
found, precise in how it is named:

* a published code, manual, standard or report by its designation: FHWA
  GEC-n (also "Geotechnical Engineering Circular No. n") and FHWA report
  numbers (FHWA-NHI-10-024; the GECs' numbers are the same document as the
  GEC, as the reference modules' own briefs and chapter files name them:
  GEC-11 = FHWA-NHI-10-024 / -025), AASHTO LRFD with or without an article
  number, the AASHTO 1993 pavement guide and AASHTO test standards, UFC
  n-nnn-nn and UFS n-nnn-nn (one key: UFS 3-220-10, 2026, supersedes UFC
  3-220-10), NAVFAC DM 7 / 7.01 / 7.02 (the same documents as UFS 3-220-10 /
  UFC 3-220-20, which replaced them and which the app's ``dm7`` module
  serves), USACE EM / ETL / ER / EP and TM numbers, ASTM, ACI, ASCE 7, AISC, API RP, IBC,
  the Caltrans manuals, NCHRP and CGPR reports, Eurocodes and EN 1997, FEMA
  P- numbers and NEHRP, ISO / BS standards, PTI DC35.1, the GSA Alternate
  Path guidelines and the Wood Handbook;
* a work by author and year: "Rankine (1857)", "Duncan & Wright (2014)",
  "Seed et al. (1985)", "(Bowles 1996; Das 2010)", and an agency as author,
  "FHWA (2010)". A name with a year in running text without brackets
  ("drilled in March 2021", "Load 2000 kN") is not one.

A citation is checked at the level of the DOCUMENT: "AASHTO LRFD 3.11.5" is
looked up when AASHTO LRFD is, and the article is shown, not checked (a
references result seldom prints every article it rests on). A page cited in
a published document is EV-18's to leave alone and this module's to check.

WHAT "LOOKED UP" MEANS (:func:`looked_up`). The record is the app's own,
never a model's account: the conversation's ``activity.jsonl``, its
``transcript.jsonl`` and the user's messages the turn binds
(:func:`user_words_bound`). A document counts as looked up when one of these
names it:

* the USER's own messages, any turn: a document the user named is theirs to
  cite and is never flagged;
* a REFERENCES result: anything the ``references`` helper's tools returned
  and its hand-back, ``consult_references``, ``read_reference_figure``,
  ``view_worked_example_source``, and ``call_agent`` on a reference module
  (``gec11``, ``dm7``, ``ufc_*``... - which looks up that module's own
  document even when its result does not print the name);
* a WORKED EXAMPLE fetched (``worked_examples``: each carries its source);
* a DOCUMENT the conversation opened (``read_document``, ``read_pdf_text``,
  ``search_document``, ``analyze_pdf_page``, ``digest_references``...): a
  report that itself cites AASHTO makes AASHTO citable - unless the document
  is one the conversation WROTE (a PDF it built and read back would launder
  its own citation);
* a CALCULATION's own result or a method's own documentation
  (``call_agent`` on an analysis module, ``describe_method``): the method
  the app ran names its basis (downdrag's UFC 3-220-20). The parts of a
  result that only repeat the call's own arguments do not count.

Not sources: the model's own words anywhere - briefs to helpers, a helper's
hand-back (other than the references helper's), earlier answers, a scratch
file it wrote. The module CATALOG (``list_agents``, ``list_methods``) never
makes a DOCUMENT citable: in w8b the ``retaining_walls`` method list named
GEC-11 for its MSE-wall method, which is no basis for citing it on a
cantilever wall. It names the methods the app has, so it does count as
naming an AUTHOR (below).

AUTHOR AND YEAR, AND RANKINE. A work is looked up when a source names the
author with that year (a reference list, a narrative citation). When a
source names the author but not that year - most often because the author's
name IS the method the user asked for, the calculation ran or the module
catalog lists ("Use Rankine active pressure"; ``"theory": "rankine"``;
"Schmertmann (1978) strain influence factor method") - the citation is
recorded under
``year_not_checked`` and NOT flagged: the record confirms the work's
author, so the citation says where the method came from; only the year is
the model's, and a theory's originator's year is common knowledge. A work
whose author nothing in the record names (a "Duncan & Wright (2014)" or a
"Kulhawy & Mayne (1990)" from memory) is flagged like a document. So in the
w8b case GEC-11 and AASHTO LRFD 3.11.5 are flagged and "Rankine (1857)" is
not: the user asked for Rankine pressure and the module ran it.

Without the activity log nothing knows what was looked up, so nothing is
checked and the result says so (as EV-18).
"""

from __future__ import annotations

import contextlib
import contextvars
import html as _html
import json
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Set, Tuple)

#: The activity-log event a check is recorded under.
RECORD_EVENT = "cited_documents"
#: How the feedback note to the model opens (``GEOTECH_CITED_DOCUMENTS=retry``).
NOTE_PREFIX = "[Citation check]"
#: The status line shown while the answer is written again.
RETRY_STATUS = ("citation check: the answer cites documents nothing in this "
                "conversation looked up; it is being written again")
#: The fields a writer's result gains (switch on, something not looked up).
RESULT_LIST = "cited_not_looked_up"
RESULT_NOTE = "citation_note"

ACTIVITY_FILE = "activity.jsonl"
TRANSCRIPT_FILE = "transcript.jsonl"

#: Most citations / rows kept in one result.
MAX_LISTED = 40
#: A sidecar holding a whole tool result is read up to this size.
MAX_SIDECAR_BYTES = 4 * 1024 * 1024

#: Where a source came from, best first (a citation names the best).
USER = "user"
REFERENCES = "references"
WORKED_EXAMPLE = "worked example"
DOCUMENT = "document"
CALCULATION = "calculation"
#: The module catalog (``list_methods``): it names an author's METHOD the
#: app has ("Schmertmann (1978) strain influence factor method"), never a
#: document's basis, so it backs a work's author only.
CATALOG = "catalog"
_RANK = {USER: 0, REFERENCES: 1, WORKED_EXAMPLE: 2, DOCUMENT: 3,
         CALCULATION: 4, CATALOG: 5}
#: How a source is named to the user.
SOURCE_WORDS = {USER: "your messages", REFERENCES: "a references result",
                WORKED_EXAMPLE: "a worked example",
                DOCUMENT: "a document opened",
                CALCULATION: "a calculation's own result",
                CATALOG: "the module catalog"}


# ---------------------------------------------------------------------------
# The documents the app knows by more than one name
# ---------------------------------------------------------------------------

#: Other designations of one document -> its key. The GECs' FHWA report
#: numbers are the reference modules' own (``funhouse_agent/adapters/
#: __init__.py`` briefs, ``geotech_references/*/text/*.json``
#: ``reference_id``, ``funhouse_agent/worked_examples.json`` sources).
ALIASES: Dict[str, str] = {
    "fhwa-if-99-015": "gec-4",
    "fhwa-nhi-16-072": "gec-5",
    "fhwa-sa-02-054": "gec-6",
    "fhwa-nhi-14-007": "gec-7",
    "fhwa-hif-07-03": "gec-8",
    "fhwa-hif-18-031": "gec-9",
    "fhwa-nhi-18-024": "gec-10",
    "fhwa-nhi-10-016": "gec-10",
    "fhwa-nhi-10-024": "gec-11",
    "fhwa-nhi-10-025": "gec-11",
    "fhwa-nhi-16-009": "gec-12",
    "fhwa-nhi-16-010": "gec-12",
    "fhwa-nhi-16-064": "gec-12",
    "fhwa-nhi-16-027": "gec-13",
    "fhwa-nhi-16-028": "gec-13",
    "fhwa-hif-17-016": "gec-14",
    "fpl-gtr-282": "wood-handbook",
    "fpl-gtr-190": "wood-handbook",
}

#: A name that covers several documents -> the documents. A citation of the
#: family is looked up by any member, and a member by the family's name
#: ("DM7" in a references result covers "UFC 3-220-10").
FAMILIES: Dict[str, Set[str]] = {
    "dm7": {"ufc-3-220-10", "ufc-3-220-20", "navfac-dm-7.03"},
    "en-1997": {"en-1997-1", "en-1997-2"},
    "caltrans": {"caltrans-trenching-shoring", "caltrans-bds", "caltrans-sdc",
                 "caltrans-foundation-manual", "caltrans-geotechnical-manual",
                 "caltrans-hdm", "caltrans-standard-specifications"},
    "nehrp": {"fema-p-2082", "fema-p-1050", "fema-p-750"},
}

#: The documents each reference module serves (``dispatch.
#: REFERENCE_MODULES``; the reference library's own ids too, as a references
#: result may name them). ``call_agent`` on one looks up its documents.
MODULE_DOCUMENTS: Dict[str, Set[str]] = {
    "dm7": {"dm7", "ufc-3-220-10", "ufc-3-220-20"},
    "dm7_1": {"ufc-3-220-10"},
    "dm7_2": {"ufc-3-220-20"},
    "em_2104": {"em-1110-2-2104"},
    "em_2107": {"em-1110-2-2107"},
    "gec4": {"gec-4"}, "gec5": {"gec-5"}, "gec6": {"gec-6"},
    "gec7": {"gec-7"}, "gec8": {"gec-8"}, "gec9": {"gec-9"},
    "gec10": {"gec-10"}, "gec11": {"gec-11"}, "gec12": {"gec-12"},
    "gec13": {"gec-13"}, "gec14": {"gec-14"},
    "micropile": {"fhwa-nhi-05-039"},
    "ufc_backfill": {"ufc-3-220-04"},
    "ufc_expansive": {"ufc-3-220-07"},
    "ufc_dewatering": {"ufc-3-220-05"},
    "ufc_pavement": {"ufc-3-250-01"},
    "ufc_stabilization": {"ufc-3-250-11"},
    "ufc_flexible_practice": {"ufc-3-250-03"},
    "ufc_concrete_practice": {"ufc-3-250-04"},
    "ufc_structural": {"ufc-3-301-01"},
    "ufc_collapse": {"ufc-4-023-03"},
    "gsa_collapse": {"gsa-alternate-path"},
    "wood_handbook": {"wood-handbook"},
    "fema_p2082": {"fema-p-2082", "nehrp"},
    "fema_p2192": {"fema-p-2192"},
    "california_trenching": {"caltrans-trenching-shoring"},
    "fhwa_pavements": {"fhwa-nhi-05-037"},
    "fhwa_soils_foundations": {"fhwa-nhi-06-088", "fhwa-nhi-06-089"},
    "eurocode_7_1": {"en-1997-1"},
    "eurocode_7_2": {"en-1997-2"},
    "aashto_1993": {"aashto-1993"},
}


def module_documents(name: Any) -> Set[str]:
    """The documents a reference module (or the library's own id for it:
    ``gec_11``, ``dm7_2``) serves; empty for any other module."""
    key = str(name or "").strip().lower().replace("-", "_")
    if key in MODULE_DOCUMENTS:
        return set(MODULE_DOCUMENTS[key])
    m = re.fullmatch(r"gec_(\d{1,2})", key)
    if m:
        return set(MODULE_DOCUMENTS.get(f"gec{int(m.group(1))}", ()))
    return set()


def canonical(key: str) -> str:
    """One key per document (an FHWA number of a GEC is the GEC)."""
    return ALIASES.get(key, key)


def covered(key: str, found: Iterable[str]) -> bool:
    """Whether the documents ``found`` include the one ``key`` names, or a
    family it belongs to, or (for a family) one of its members."""
    have = set(found)
    if key in have:
        return True
    if key in FAMILIES and FAMILIES[key] & have:
        return True
    return any(key in members and fam in have
               for fam, members in FAMILIES.items())


# ---------------------------------------------------------------------------
# Citations in a text
# ---------------------------------------------------------------------------

_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-",
                         "–": "-", "—": "-", "−": "-",
                         " ": " ", " ": " ", " ": " "})
_TAG = re.compile(r"</?[A-Za-z][^>]*>")


def plain_text(text: Any) -> str:
    """``text`` with HTML tags and entities taken out (a writer is often
    given HTML) and dashes and odd spaces made plain."""
    t = str(text or "")
    if "<" in t and _TAG.search(t):
        t = re.sub(r"(?is)<(script|style)\b.*?</\1\s*>", " ", t)
        t = _TAG.sub(" ", t)
    if "&" in t:
        t = _html.unescape(t)
    return t.translate(_DASHES)


@dataclass
class DocCited:
    """One citation of a published document or work."""
    #: As shown to the reader: "GEC-11", "AASHTO LRFD 3.11.5",
    #: "Rankine (1857)".
    shown: str
    #: One key per document ("gec-11"); for a work, ``work:<author>:<year>``.
    key: str
    #: ``document`` (a designation) or ``work`` (an author and year).
    kind: str = "document"
    #: An article or section number cited with it, shown and not checked.
    section: str = ""
    start: int = 0
    end: int = 0
    #: A work's first author, folded to lower case without accents.
    author: str = ""
    year: str = ""


def _fold(word: str) -> str:
    text = unicodedata.normalize("NFKD", str(word))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _name(word: str) -> str:
    """A name as compared: folded, without edge quotes or a possessive."""
    name = _fold(word).strip("'’-")
    for tail in ("'s", "’s"):
        if name.endswith(tail):
            name = name[:-len(tail)]
    return name


#: "Section 4.3", "Article 3.11.5", "§6-7.4", "Ch. 5": a section cited with
#: a document (for AASHTO LRFD a dotted number alone is one: "LRFD 3.11.5").
_SECTION_WORD = (r"(?P<word>Article|Art\.|Section|Sec\.|§+|Clause|"
                 r"Chapter|Ch\.)")
_SECTION_WORDED = re.compile(
    r"[ \t]*,?[ \t]*" + _SECTION_WORD + r"[ \t]*"
    r"(?P<sec>\d{1,2}(?:[.-]\d{1,3}){0,6})(?!\d)(?![.-]\d)")
_SECTION_BARE = re.compile(
    r"[ \t]*,?[ \t]*(?:\(?\d{4}\)?[ \t]*,?[ \t]*)?(?:" + _SECTION_WORD
    + r"[ \t]*)?(?P<sec>\d{1,2}(?:\.\d{1,3}){1,5})(?!\d)(?!\.\d)")
_WORDS_SHOWN = {"article": "Art.", "art.": "Art.", "section": "Sec.",
                "sec.": "Sec.", "clause": "Clause", "chapter": "Ch.",
                "ch.": "Ch."}


def _section(text: str, pos: int, bare: bool = False) -> Tuple[str, int]:
    """``(section as shown, end)`` of a section cited at ``pos``."""
    m = (_SECTION_BARE if bare else _SECTION_WORDED).match(text, pos)
    if not m:
        return "", pos
    word = m.group("word") or ""
    if word.startswith("§"):
        return f"§{m.group('sec')}", m.end()
    shown = _WORDS_SHOWN.get(word.lower(), "")
    return (f"{shown} {m.group('sec')}" if shown else m.group("sec")), m.end()


_Found = Tuple[int, int, str, str, str]          # start, end, key, shown, sec

_GEC = re.compile(
    r"(?i)(?<![\w-])(?:FHWA[ \t-]*)?(?:GEC|Geotechnical[ \t]+Engineering[ \t]+"
    r"Circular)[ \t#-]*(?:No\.?[ \t]*|Number[ \t]*)?(?P<n>\d{1,2})"
    r"(?!\d)(?!\.\d)")
_FHWA_NO = re.compile(
    r"(?i)(?<![\w-])(?:FHWA[ \t-]*(?P<p>NHI|SA|IF|HIF|HRT|RD|ED|TS|IP|CFL|"
    r"FLH|PL|HI|EP)|(?P<q>NHI))[ \t-]*(?P<a>\d{2})[ \t]*-[ \t]*(?P<b>\d{2,3})"
    r"(?!\d)")
_AASHTO = re.compile(r"(?i)(?<![\w-])AASHTO(?![\w-])")
_EDITION = (r"(?:[ \t,]*(?:\(?(?:19|20)\d{2}\)?|\d{1,2}(?:st|nd|rd|th)[ \t]+"
            r"ed(?:ition|\.)?))*")
_AASHTO_LRFD = re.compile(
    r"(?i)" + _EDITION + r"[ \t,]*(?:LRFD(?:[ \t]+Bridge[ \t]+Design[ \t]+"
    r"Specifications?|[ \t]+BDS|[ \t]+Specifications?)?|Bridge[ \t]+Design"
    r"[ \t]+Specifications?|BDS)(?![\w-])" + _EDITION)
_AASHTO_1993 = re.compile(
    r"(?i)[ \t,]*(?:\(?1993\)?(?![\d])(?:[ \t]+(?:Pavement[ \t]+)?(?:Design"
    r"[ \t]+)?Guide)?|Guide[ \t]+for[ \t]+(?:the[ \t]+)?Design[ \t]+of[ \t]+"
    r"Pavement[ \t]+Structures)")
_AASHTO_STD = re.compile(r"[ \t]*(?P<l>[TMR])[ \t-]?(?P<n>\d{1,3})(?!\d)")
_UFC = re.compile(      # UFS: a Unified Facilities Supplement (DM 7.1, 2026)
    r"(?i)(?<![\w-])(?P<p>UF[CS])\)?[ \t-]*(?P<a>\d)-(?P<b>\d{3})-(?P<c>\d{2})"
    r"(?P<s>[A-Z]{0,2})(?![\w-])")
_DM = re.compile(
    r"(?<![\w-])(?:NAVFAC[ \t-]*(?:DM|Design[ \t]+Manual)|NAVDOCKS[ \t-]*DM|DM)"
    r"[ \t-]*7(?:\.0?(?P<v>[1-3]))?(?!\d)(?!\.\d)")
_USACE = re.compile(
    r"(?i)(?<![\w-])(?:USACE[ \t-]*)?(?P<t>EM|ETL|ER|EP)[ \t-]*(?P<a>\d{3,4})"
    r"-(?P<b>\d{1,2})-(?P<c>\d{1,4})(?!\d)")
_TM = re.compile(
    r"(?<![\w-])TM[ \t-]*(?P<a>\d{1,2})-(?P<b>\d{3})-(?P<c>\d{1,2})(?!\d)")
_ASTM = re.compile(
    r"(?i)(?<![\w-])ASTM[ \t-]*(?:International[ \t,]*)?(?P<l>[A-G])[ \t-]?"
    r"(?P<n>\d{1,5})(?:[ \t]*/[ \t]*[A-G][ \t]?\d{1,5}M?)?"
    r"(?:-\d{2}(?:[a-z]\d?)?(?:\(\d{4}\))?)?(?!\d)")
_ACI = re.compile(
    r"(?<![\w-])ACI[ \t-]*(?P<n>\d{3}(?:\.\d{1,2})?R?)(?:-\d{2,4})?(?!\d)")
#: "ASCE 7-22"; not a journal volume ("J. SMFD ASCE 99(SM1)").
_ASCE = re.compile(
    r"(?<![\w-])ASCE(?:/SEI|/COPRI|/GI)?[ \t-]*(?P<n>\d{1,3})(?:-\d{2})?"
    r"(?!\d)(?!\.\d)(?!\()")
_JOURNAL = re.compile(r"(?i)(?:\bJ\.|\bJournal\b|\bProc\.|\bProceedings\b)"
                      r"[^.;\n]{0,55}$")
_AISC = re.compile(r"(?<![\w-])AISC[ \t-]*(?P<n>\d{3})(?!\d)")
_API = re.compile(
    r"(?<![\w-])API[ \t]+(?:RP|Std\.?|Spec\.?|Recommended[ \t]+Practice)"
    r"[ \t-]*(?P<n>\d[0-9A-Z]{0,5})(?![\w])")
_IBC = re.compile(
    r"(?<![\w-])(?:(?:19|20)\d{2}[ \t]+)?IBC(?![\w-])"
    r"|(?i:International[ \t]+Building[ \t]+Code)")
_CALTRANS = re.compile(
    r"(?i)(?<![\w-])(?:Caltrans|California[ \t]+Department[ \t]+of[ \t]+"
    r"Transportation)(?![\w-])")
_CALTRANS_KIND = (
    ("caltrans-trenching-shoring", "Caltrans Trenching and Shoring Manual",
     r"Trenching|Shoring|T[ \t]*&[ \t]*S\b"),
    ("caltrans-bds", "Caltrans Bridge Design Specifications",
     r"Bridge[ \t]+Design[ \t]+Spec|BDS\b"),
    ("caltrans-sdc", "Caltrans Seismic Design Criteria",
     r"Seismic[ \t]+Design[ \t]+Criteria|SDC\b"),
    ("caltrans-foundation-manual", "Caltrans Foundation Manual",
     r"Foundation[ \t]+Manual"),
    ("caltrans-geotechnical-manual", "Caltrans Geotechnical Manual",
     r"Geotechnical[ \t]+Manual"),
    ("caltrans-hdm", "Caltrans Highway Design Manual",
     r"Highway[ \t]+Design[ \t]+Manual|HDM\b"),
    ("caltrans-standard-specifications", "Caltrans Standard Specifications",
     r"Standard[ \t]+Specifications"),
    ("caltrans", "Caltrans",
     r"Example|Manual|Method|Procedure|Specification|Guideline|Memo"),
)
_TRENCHING = re.compile(r"(?i)Trenching[ \t]+and[ \t]+Shoring[ \t]+Manual")
_NCHRP = re.compile(
    r"(?i)(?<![\w-])NCHRP[ \t-]+(?P<t>Report|Synthesis|Project|Web[ \t-]*Only"
    r"[ \t]+Document|Research[ \t]+Results[ \t]+Digest|RRD)?[ \t]*(?:No\.?"
    r"[ \t]*|#[ \t]*)?(?P<n>\d{1,4}(?:-\d{1,3})?)(?!\d)")
_CGPR = re.compile(
    r"(?i)(?<![\w-])CGPR[ \t-]*(?:Report[ \t]*)?(?:No\.?[ \t]*|#[ \t]*)?"
    r"(?P<n>\d{1,3})(?!\d)")
_EN = re.compile(
    r"(?<![\w-])(?:(?:BS|NF|DIN|I\.?S\.?)[ \t]+)?EN[ \t-]*(?P<n>199\d)"
    r"(?:[ \t]*-[ \t]*(?P<p>\d{1,2}))?(?!\d)")
_EUROCODE = re.compile(
    r"(?i)(?<![\w-])Eurocode[ \t-]*(?P<n>\d)(?:[ \t]*(?:-|,|Part)[ \t]*"
    r"(?:Part[ \t]*)?(?P<p>\d)(?!\d))?(?!\d)")
_EC = re.compile(r"(?<![\w-])EC[ \t-]?(?P<n>[78])(?:-(?P<p>\d))?(?!\d)(?!\.\d)")
_FEMA = re.compile(
    r"(?i)(?<![\w-])FEMA[ \t-]*(?P<p>P[ \t-]?)?(?P<n>\d{2,4})(?!\d)")
_NEHRP = re.compile(r"(?<![\w-])NEHRP(?![\w-])")
_GSA = re.compile(
    r"(?<![\w-])GSA(?![\w-])[^.;\n]{0,40}?(?i:Alternate[ \t]+Path|"
    r"Progressive[ \t]+Collapse)")
_WOOD = re.compile(
    r"(?i)(?<![\w-])(?:(?:USDA[ \t]+)?Wood[ \t]+Handbook|FPL[ \t-]*GTR[ \t-]*"
    r"(?P<n>\d{2,3}))")
_ISO = re.compile(
    r"(?<![\w-])(?P<b>BS[ \t]+EN[ \t]+ISO|EN[ \t]+ISO|BS[ \t]+ISO|ISO|BS)"
    r"[ \t-]*(?P<n>\d{3,5})(?:[ \t]*-[ \t]*(?P<p>\d{1,2}))?(?::\d{4})?(?!\d)")
_PTI = re.compile(r"(?<![\w-])PTI[ \t-]*(?P<n>DC[ \t]?\d{2}(?:\.\d)?)")
#: The reference library's own ids, as a references result may print them
#: (sources only: nobody cites "gec_11" in prose).
_INTERNAL = re.compile(
    r"(?<![\w-])(dm7_[12]|gec_\d{1,2}|ufc_[a-z]+(?:_[a-z]+)*|fema_p2082|"
    r"fema_p2192|california_trenching|fhwa_pavements|fhwa_soils_foundations|"
    r"eurocode_7_[12]|"
    r"aashto_1993|em_210[47]|gsa_collapse|wood_handbook)(?![\w])")


def _documents(t: str, internal: bool = False) -> List[_Found]:
    """Every document designation in ``t`` (already :func:`plain_text`)."""
    out: List[_Found] = []

    def add(start, end, key, shown, sec=""):
        key = canonical(key)
        out.append((start, end, key, f"{shown} {sec}" if sec else shown, sec))

    for m in _GEC.finditer(t):
        n = int(m.group("n"))
        sec, end = _section(t, m.end())
        add(m.start(), end, f"gec-{n}", f"GEC-{n}", sec)
    for m in _FHWA_NO.finditer(t):
        p = (m.group("p") or m.group("q")).upper()
        num = f"{m.group('a')}-{m.group('b')}"
        sec, end = _section(t, m.end())
        add(m.start(), end, f"fhwa-{p.lower()}-{num}", f"FHWA-{p}-{num}", sec)
    for m in _AASHTO.finditer(t):
        tail = t[m.end():m.end() + 200]
        got = _AASHTO_1993.match(tail)
        if got:
            add(m.start(), m.end() + got.end(), "aashto-1993", "AASHTO 1993")
            continue
        got = _AASHTO_LRFD.match(tail)
        if got:
            sec, end = _section(t, m.end() + got.end(), bare=True)
            add(m.start(), end, "aashto-lrfd", "AASHTO LRFD", sec)
            continue
        got = _AASHTO_STD.match(tail)
        if got:
            name = f"{got.group('l')}{int(got.group('n'))}"
            add(m.start(), m.end() + got.end(), f"aashto-{name.lower()}",
                f"AASHTO {got.group('l')} {int(got.group('n'))}")
            continue
        sec, end = _section(t, m.end(), bare=True)
        if sec:                     # "AASHTO 3.11.5": an LRFD article
            add(m.start(), end, "aashto-lrfd", "AASHTO LRFD", sec)
    for m in _UFC.finditer(t):
        num = f"{m.group('a')}-{m.group('b')}-{m.group('c')}"
        sec, end = _section(t, m.end())
        add(m.start(), end, f"ufc-{num}",
            f"{m.group('p').upper()} {num}{(m.group('s') or '').upper()}",
            sec)
    for m in _DM.finditer(t):
        v = m.group("v")
        key, shown = {"1": ("ufc-3-220-10", "NAVFAC DM 7.01"),
                      "2": ("ufc-3-220-20", "NAVFAC DM 7.02"),
                      "3": ("navfac-dm-7.03", "NAVFAC DM 7.3")}.get(
                          v or "", ("dm7", "NAVFAC DM 7"))
        sec, end = _section(t, m.end())
        add(m.start(), end, key, shown, sec)
    for m in _USACE.finditer(t):
        kind = m.group("t").upper()
        num = f"{m.group('a')}-{m.group('b')}-{m.group('c')}"
        sec, end = _section(t, m.end())
        add(m.start(), end, f"{kind.lower()}-{num}", f"{kind} {num}", sec)
    for m in _TM.finditer(t):
        num = f"{m.group('a')}-{m.group('b')}-{m.group('c')}"
        add(m.start(), m.end(), f"tm-{num}", f"TM {num}")
    for m in _ASTM.finditer(t):
        name = f"{m.group('l').upper()}{int(m.group('n'))}"
        add(m.start(), m.end(), f"astm-{name.lower()}", f"ASTM {name}")
    for m in _ACI.finditer(t):
        n = m.group("n")
        add(m.start(), m.end(), f"aci-{n.lower()}", f"ACI {n}")
    for m in _ASCE.finditer(t):
        if _JOURNAL.search(t[max(0, m.start() - 70):m.start()]):
            continue                 # a journal's volume, not a standard
        n = int(m.group("n"))
        sec, end = _section(t, m.end())
        add(m.start(), end, f"asce-{n}", f"ASCE {n}", sec)
    for m in _AISC.finditer(t):
        add(m.start(), m.end(), f"aisc-{m.group('n')}", f"AISC {m.group('n')}")
    for m in _API.finditer(t):
        n = m.group("n").upper()
        add(m.start(), m.end(), f"api-{n.lower()}", f"API RP {n}")
    for m in _IBC.finditer(t):
        sec, end = _section(t, m.end())
        add(m.start(), end, "ibc", "IBC", sec)
    for m in _CALTRANS.finditer(t):
        tail = re.split(r"[.;\n]", t[m.end():m.end() + 70], maxsplit=1)[0]
        for key, shown, words in _CALTRANS_KIND:
            got = re.search(rf"(?i)\b(?:{words})", tail)
            if got:
                add(m.start(), m.end() + got.end(), key, shown)
                break
    for m in _TRENCHING.finditer(t):
        add(m.start(), m.end(), "caltrans-trenching-shoring",
            "Caltrans Trenching and Shoring Manual")
    for m in _NCHRP.finditer(t):
        if not m.group("t") and re.fullmatch(r"(?:19|20)\d{2}", m.group("n")):
            continue                 # "NCHRP 1997a": an author and year
        kind = (m.group("t") or "Report").title()
        add(m.start(), m.end(), f"nchrp-{m.group('n')}",
            f"NCHRP {kind} {m.group('n')}")
    for m in _CGPR.finditer(t):
        n = int(m.group("n"))
        add(m.start(), m.end(), f"cgpr-{n}", f"CGPR {n}")
    for rx in (_EN, _EUROCODE, _EC):
        for m in rx.finditer(t):
            n = m.group("n")
            base = n if len(n) == 4 else f"199{n}"
            p = m.group("p")
            key = f"en-{base}-{p}" if p else f"en-{base}"
            sec, end = _section(t, m.end())
            add(m.start(), end, key, f"EN {base}-{p}" if p else f"EN {base}",
                sec)
    for m in _FEMA.finditer(t):
        n = m.group("n")
        if not m.group("p") and len(n) == 4 and n[:2] in ("19", "20"):
            continue                 # "FEMA 2020": a year, not a number
        key = f"fema-p-{int(n)}" if m.group("p") else f"fema-{int(n)}"
        add(m.start(), m.end(), key,
            f"FEMA P-{int(n)}" if m.group("p") else f"FEMA {int(n)}")
    for m in _NEHRP.finditer(t):
        add(m.start(), m.end(), "nehrp", "NEHRP")
    for m in _GSA.finditer(t):
        add(m.start(), m.end(), "gsa-alternate-path", "GSA Alternate Path")
    for m in _WOOD.finditer(t):
        n = m.group("n")
        key = f"fpl-gtr-{int(n)}" if n else "wood-handbook"
        add(m.start(), m.end(), key, "Wood Handbook" if not n
            or canonical(key) == "wood-handbook" else f"FPL-GTR-{int(n)}")
    for m in _ISO.finditer(t):
        body = "iso" if "ISO" in m.group("b") else "bs"
        num = m.group("n") + (f"-{m.group('p')}" if m.group("p") else "")
        add(m.start(), m.end(), f"{body}-{num}", f"{body.upper()} {num}")
    for m in _PTI.finditer(t):
        n = m.group("n").replace(" ", "")
        add(m.start(), m.end(), f"pti-{n.lower()}", f"PTI {n}")
    if internal:
        for m in _INTERNAL.finditer(t):
            for key in sorted(module_documents(m.group(1))):
                add(m.start(), m.end(), key, m.group(1))
    return out


# -- works by author and year -------------------------------------------------

#: A capital (with the accented ones of Latin-1 and Latin Extended-A:
#: "Vesić", "Šuklje"), and any letter that is not an ASCII or Latin-1
#: capital.
_UP = "A-ZÀ-ÖØ-ÞĀ-ſ"
_LO = r"[^\W\d_A-ZÀ-ÖØ-Þ]"
_SURNAME = (rf"[{_UP}]{_LO}+(?:[{_UP}]{_LO}+)?"
            rf"(?:[-'’][{_UP}]?{_LO}+)*")
#: Agencies that stand as an author ("FHWA (2010)").
_AGENCY = (r"(?:FHWA|USACE|NAVFAC|AASHTO|USBR|NRCS|ASCE|ACI|FEMA|NCHRP|TRB|"
           r"USDA|USGS|NAVDOCKS|ASTM|Caltrans|WSDOT|TxDOT|FDOT|NYSDOT|"
           r"PennDOT|MnDOT)")
_AUTHORS = (rf"(?P<authors>(?P<who>{_SURNAME}|{_AGENCY})"
            rf"(?:[ \t]*,[ \t]*{_SURNAME})*"
            rf"(?:[ \t]*,?[ \t]+(?:and|&)[ \t]+{_SURNAME}"
            rf"|[ \t]+et[ \t]+al\.?)?)")
_YEAR = r"(?P<year>(?:1[6-9]|20)\d{2})(?P<suffix>[a-z])?"
#: "Rankine (1857)", "Duncan and Wright (2014, p. 12)", "Rankine's (1857)".
_NARRATIVE = re.compile(
    rf"(?<![\w-]){_AUTHORS}(?:'s|’s)?[ \t]*\([ \t]*{_YEAR}[ \t]*[),;:]")
#: One bracketed group: "(Rankine 1857; see GEC-11)", "(Bowles, 1996)".
_PAREN = re.compile(r"\(([^()\n]{1,240})\)")
_PAREN_PIECE = re.compile(
    rf"[ \t]*(?:(?:e\.g\.|i\.e\.|see(?:[ \t]+also)?|cf\.|after|from|per|"
    rf"following|adapted[ \t]+from)[,:]?[ \t]+)?{_AUTHORS}[ \t]*,?[ \t]*"
    rf"{_YEAR}(?!\d)")
#: Capitalised words that are not authors before a bracketed year.
_NOT_AUTHORS = frozenset(_fold(w) for w in (
    "January February March April May June July August September October "
    "November December Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec "
    "Spring Summer Fall Autumn Winter In On At By Since From Until Before "
    "After During Through Circa The This That These Those And Or Per See "
    "Also Note Notes Table Tables Figure Figures Fig Page Pages Section "
    "Article Chapter Appendix Annex Vol Volume Edition Ed Eq Equation Rev "
    "Revision Version Draft Final Report Study Project Phase Stage Sheet "
    "Boring Borings Sample Test Tests Station Elevation El Datum Grid Zone "
    "Plate Exhibit Attachment Item Step Case Run Load Unit Layer Area Site "
    "Lot Parcel Route Mile Highway Interstate Bridge Pier Abutment Contract "
    "Task Order Proposal Year Years Dated Updated Issued Published "
    "Copyright Change Built Constructed Installed Drilled Tested Sampled "
    "Completed Started Total Mean Average Max Min Engineering Engineers "
    "Consultants Associates Inc Corporation Company Geotechnical "
    "Investigation Survey Department Division Office Agency Authority "
    "Institute University Laboratory Standard Standards Code Codes Manual "
    "Guide Specification Specifications Provisions Design Method Model "
    "Analysis Value Values Result Results Data Calc Calculation Note "
    "Memo Letter Field Lab Laboratory").split())


def _works(t: str) -> List[Tuple[int, int, str, str, str, str]]:
    """``(start, end, key, shown, author, year)`` of every author-and-year
    citation in ``t``."""
    out = []

    def take(m, base):
        who = _name(m.group("who"))
        if who in _NOT_AUTHORS:
            return
        authors = re.sub(r"\s+", " ", m.group("authors")).strip()
        authors = re.sub(r"(?:'s|’s)$", "", authors)
        year = m.group("year")
        shown = f"{authors} ({year}{m.group('suffix') or ''})"
        out.append((base + m.start("authors"), base + m.end(),
                    f"work:{who}:{year}", shown, who, year))

    for m in _NARRATIVE.finditer(t):
        take(m, 0)
    for g in _PAREN.finditer(t):
        body, at = g.group(1), g.start(1)
        pos = 0
        for piece in body.split(";"):
            m = _PAREN_PIECE.match(piece)
            if m:
                take(m, at + pos)
            pos += len(piece) + 1
    return out


def find_citations(text: Any, internal: bool = False,
                   every: bool = False) -> List[DocCited]:
    """Every document and work ``text`` cites, in order, one per document
    (the first time it is cited, with the first section cited). HTML is
    read as its text. ``internal`` also reads the reference library's own
    ids ("gec_11"), as a references result may print them. ``every``
    gives each citation where it stands, with its own section, not just
    each document's first (BACKLOG EV-30: a deliverable that names a
    document first without a section and cites its sections later)."""
    t = plain_text(text)
    docs = _documents(t, internal=internal)
    spans = [(s, e) for s, e, *_ in docs]
    found: List[DocCited] = [DocCited(shown=shown, key=key, section=sec,
                                      start=s, end=e)
                             for s, e, key, shown, sec in docs]
    for s, e, key, shown, who, year in _works(t):
        # A year printed with a designation is the document's edition.
        if any(a < e and s < b for a, b in spans):
            continue
        found.append(DocCited(shown=shown, key=key, kind="work", start=s,
                              end=e, author=who, year=year))
    seen: Set[str] = set()
    out: List[DocCited] = []
    for c in sorted(found, key=lambda c: (c.start, -c.end)):
        if c.key in seen and not every:
            continue
        seen.add(c.key)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# What a source names
# ---------------------------------------------------------------------------

_YEAR_ANY = re.compile(r"(?<!\d)(1[6-9]\d\d|20[0-3]\d)(?!\d)")
_CAP_WORD = re.compile(rf"[{_UP}](?:[^\W\d_]|['’-])+")
_ANY_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])+")


@dataclass
class Marks:
    """What a set of sources names: documents, authors with years, and
    names; each with the best kind of source that named it."""
    keys: Dict[str, str] = field(default_factory=dict)
    pairs: Dict[Tuple[str, str], str] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _put(store: Dict[Any, str], item: Any, kind: str) -> None:
        old = store.get(item)
        if old is None or _RANK.get(kind, 9) < _RANK.get(old, 9):
            store[item] = kind

    def add_text(self, text: Any, kind: str, internal: bool = False) -> None:
        """Note what ``text`` names, as a source of ``kind``. A user's words
        and a calculation's result name a method in any case ("use
        rankine", ``"theory": "rankine"``); elsewhere a name is capitalised."""
        t = plain_text(text)
        if not t.strip():
            return
        words = _ANY_WORD if kind in (USER, CALCULATION) else _CAP_WORD
        for w in set(words.findall(t)):
            self._put(self.names, _name(w), kind)
        if kind == CATALOG:
            return                     # names only (see CATALOG)
        for c in find_citations(t, internal=internal):
            self._put(self.keys, c.key, kind)
        for y in _YEAR_ANY.finditer(t):
            before = t[max(0, y.start() - 90):y.start()]
            before = before[before.rfind("\n") + 1:]
            for w in set(words.findall(before)):
                self._put(self.pairs, (_name(w), y.group(1)), kind)

    def add_keys(self, keys: Iterable[str], kind: str) -> None:
        for key in keys:
            self._put(self.keys, canonical(key), kind)

    def merge(self, other: "Marks") -> None:
        for store, theirs in ((self.keys, other.keys),
                              (self.pairs, other.pairs),
                              (self.names, other.names)):
            for item, kind in theirs.items():
                self._put(store, item, kind)

    def document_kind(self, key: str) -> Optional[str]:
        """The best source naming document ``key`` (or its family)."""
        kinds = [kind for k, kind in self.keys.items() if covered(key, [k])]
        return min(kinds, key=lambda k: _RANK.get(k, 9)) if kinds else None


# ---------------------------------------------------------------------------
# The record: what this conversation looked up
# ---------------------------------------------------------------------------

#: Tools whose results are the TEXT of a document the conversation opened.
DOCUMENT_TOOLS = frozenset({
    "read_document", "read_pdf_text", "read_text_file", "search_document",
    "find_quantities", "analyze_pdf_page", "render_region", "analyze_image",
    "open_document", "document_structure", "document_page_map",
    "document_markups", "log_grid", "find_like", "kept_readings",
    "digest_search", "digest_pages", "digest_references",
    "document_inventory", "sweep_pages", "report_ingest", "report_library",
    "title_block", "drawing_callouts"})
#: Tools whose results are a reference read.
REFERENCE_TOOLS = frozenset({"consult_references", "read_reference_figure",
                             "view_worked_example_source"})
#: Tools that write a file: what they write is the conversation's own, and
#: reading it back is not a source.
WRITING_TOOLS = frozenset({"save_file", "write_docx", "write_xlsx",
                           "annotate_document", "annotate_dxf", "plot_data",
                           "plot_code", "call_agent"})
#: The helper whose whole work is a references read.
REFERENCES_HELPER = "references"
#: Keys of a result or arguments that name a file written.
_OUTPUT_KEYS = ("output_path", "path", "saved_path", "file", "files",
                "images", "figures", "png", "pdf", "output_files")
#: The same for ``call_agent``, whose results may also name a file READ
#: ("file", "path"): only the keys that name what a method wrote.
_MODULE_OUTPUT_KEYS = ("output_path", "files", "images", "figures",
                       "output_files")
#: Arguments naming the file a reading tool reads.
_TARGET_KEYS = ("source", "path", "attachment_key", "file", "document",
                "file_path", "pdf", "image")


def _json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _args_dict(args: Any) -> dict:
    data = _json(args)
    return data if isinstance(data, dict) else {}


def _base(name: Any) -> str:
    text = str(name or "").strip().replace("\\", "/").rstrip("/")
    return os.path.basename(text).lower() if text else ""


def _strings(obj: Any, out: List[str], depth: int = 0,
             skip: Sequence[str] = ()) -> List[str]:
    if depth > 8:
        return out
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) not in skip:
                _strings(v, out, depth + 1, skip)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _strings(v, out, depth + 1, skip)
    return out


def _files_named(obj: Any, out: Set[str], keys: Sequence[str] = _OUTPUT_KEYS,
                 depth: int = 0) -> None:
    """The base names of the files a writer's result or arguments name
    under ``keys``."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in keys or (str(k) in ("file", "output_path")
                                  and depth > 0):
                if isinstance(v, str) and "." in v[-6:]:
                    out.add(_base(v))
                elif isinstance(v, (list, dict)):
                    _files_named(v, out, keys, depth + 1)
            elif str(k) == "parameters" and isinstance(v, dict):
                _files_named(v, out, keys, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, str) and "." in v[-6:]:
                out.add(_base(v))
            else:
                _files_named(v, out, keys, depth + 1)


def _without_echo(result: str, args: dict) -> str:
    """A calculation's result without the strings it was GIVEN (a title, a
    note): those are the model's words, not the method's basis."""
    for s in sorted(set(_strings(args, [])), key=len, reverse=True):
        if len(s) >= 8 and s in result:
            result = result.replace(s, " ")
        quoted = json.dumps(s)[1:-1]
        if len(quoted) >= 8 and quoted != s and quoted in result:
            result = result.replace(quoted, " ")
    return result


class _LookupState:
    """What one ``activity.jsonl`` says so far: it is append-only, so each
    read starts where the last one stopped."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.size_seen = 0
        self.starts: Dict[str, Tuple[str, Any, str]] = {}
        self.marks = Marks()
        self.outputs: Set[str] = set()
        self.handle_names: Dict[str, str] = {}


_STATES: "OrderedDict[str, _LookupState]" = OrderedDict()
_STATES_LOCK = threading.Lock()
_STATES_KEEP = 32


def _state_for(path: str) -> _LookupState:
    key = os.path.normcase(os.path.abspath(path))
    with _STATES_LOCK:
        st = _STATES.get(key)
        if st is None:
            st = _STATES[key] = _LookupState()
        _STATES.move_to_end(key)
        while len(_STATES) > _STATES_KEEP:
            _STATES.popitem(last=False)
        return st


def clear_cache() -> None:
    """Forget what the activity logs said (tests; a host that rewrites one)."""
    with _STATES_LOCK:
        _STATES.clear()


def _whole_result(rec: dict, base: str) -> str:
    """A tool result as text, from its sidecar when the record cut it."""
    result = rec.get("result")
    if rec.get("truncated") and rec.get("full"):
        path = os.path.join(base, str(rec["full"]))
        try:
            if os.path.getsize(path) <= MAX_SIDECAR_BYTES:
                with open(path, encoding="utf-8", errors="replace") as fh:
                    return fh.read()
        except OSError:
            pass
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str) if result is not None else ""


def source_kind(name: str, args: dict, agent: str = "") -> Optional[str]:
    """What kind of source one finished tool call is (``None``: not one).
    See the module docstring."""
    if agent == REFERENCES_HELPER:
        return REFERENCES
    if name == "task":
        return (REFERENCES if str(args.get("subagent_type") or "")
                == REFERENCES_HELPER else None)
    if name in REFERENCE_TOOLS:
        return REFERENCES
    if name == "describe_method":
        return CALCULATION
    if name in ("list_methods", "list_agents"):
        return CATALOG
    if name == "call_agent":
        module = str(args.get("agent_name") or "").strip().lower()
        try:
            from funhouse_agent.dispatch import REFERENCE_MODULES
        except Exception:  # noqa: BLE001
            REFERENCE_MODULES = frozenset(MODULE_DOCUMENTS)
        if module in REFERENCE_MODULES:
            return REFERENCES
        if module == "worked_examples":
            return WORKED_EXAMPLE
        if module == "calc_package":
            return None              # the app's own writer
        return CALCULATION
    if name in DOCUMENT_TOOLS:
        return DOCUMENT
    return None


def _target(args: dict, st: _LookupState) -> str:
    for k in _TARGET_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v
    handle = str(args.get("handle") or "")
    return st.handle_names.get(handle, handle)


def _feed(st: _LookupState, path: str) -> None:
    """Read what was added to ``path`` since the last call."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return
    if size < st.size_seen:            # rewritten: start over
        st.reset()
    if size == st.offset:
        st.size_seen = size
        return
    base = os.path.dirname(path)
    with open(path, "rb") as fh:
        fh.seek(st.offset)
        chunk = fh.read(size - st.offset)
    cut = chunk.rfind(b"\n")
    if cut < 0:
        return                         # half a line: wait for the rest
    st.offset += cut + 1
    st.size_seen = size
    for raw in chunk[:cut + 1].splitlines():
        try:
            rec = json.loads(raw.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        ev = rec.get("event")
        if ev == "turn_start":
            st.marks.add_text(rec.get("prompt"), USER)
            continue
        rid = str(rec.get("run_id") or "")
        if ev == "tool_start" and rid:
            st.starts[rid] = (str(rec.get("name") or ""), rec.get("args"),
                              str(rec.get("agent") or ""))
            continue
        if ev not in ("tool_end", "tool_error") or not rid:
            continue
        start = st.starts.pop(rid, None)
        if ev == "tool_error" or start is None:
            continue
        name, args, agent = start
        args = _args_dict(args)
        text = _whole_result(rec, base)
        data = _json(text)
        if isinstance(data, dict) and data.get("error"):
            continue
        _note_call(st, name, args, agent, text, data)


def _note_call(st: _LookupState, name: str, args: dict, agent: str,
               text: str, data: Any) -> None:
    """One finished tool call into the record."""
    if name == "open_document" and isinstance(data, dict):
        handle = str(data.get("handle") or "")
        shown = str(data.get("name") or data.get("source")
                    or args.get("source") or "")
        if handle and shown:
            st.handle_names[handle] = shown
    if name in WRITING_TOOLS:
        wrote: Set[str] = set()
        result = data if isinstance(data, (dict, list)) else {}
        if name != "call_agent":
            _files_named(result, wrote)
            _files_named(args, wrote)
        else:
            _files_named(result, wrote, _MODULE_OUTPUT_KEYS)
            if isinstance(data, dict) and (data.get("file_exists")
                                           or data.get("output_path")):
                _files_named(args, wrote, ("output_path",))
        st.outputs |= {w for w in wrote if w}
    kind = source_kind(name, args, agent)
    if kind is None:
        return
    if kind == DOCUMENT and _base(_target(args, st)) in st.outputs:
        return                         # a file this conversation wrote
    if name == "call_agent" and kind == REFERENCES:
        st.marks.add_keys(module_documents(args.get("agent_name")), kind)
    if name == "read_reference_figure":
        st.marks.add_keys(module_documents(args.get("reference")), kind)
    if kind == CALCULATION:
        text = _without_echo(text, args)
    st.marks.add_text(text, kind, internal=(kind == REFERENCES))


# -- the user's own words -----------------------------------------------------

_USER_WORDS: "contextvars.ContextVar[Optional[Tuple[str, ...]]]" = \
    contextvars.ContextVar("geotech_cited_documents_user_words", default=None)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    return ""


def user_texts(messages: Any) -> List[str]:
    """The text of every USER message in ``messages`` (dicts with a
    ``role`` or LangChain messages)."""
    out = []
    for m in messages or []:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type")
            content = m.get("content")
        else:
            role = getattr(m, "type", "") or getattr(m, "role", "")
            content = getattr(m, "content", None)
        if str(role) in ("user", "human"):
            text = _content_text(content)
            if text.strip():
                out.append(text)
    return out


@contextlib.contextmanager
def user_words_bound(texts: Optional[Sequence[str]]):
    """``with user_words_bound(texts):`` - the user's own messages for the
    calls made in this block (the turn binds them; a writer's check reads
    them). A no-op for ``None``."""
    if texts is None:
        yield
        return
    token = _USER_WORDS.set(tuple(str(t) for t in texts))
    try:
        yield
    finally:
        try:
            _USER_WORDS.reset(token)
        except (ValueError, LookupError, RuntimeError):
            pass


def _transcript_user_texts(folder: str) -> List[str]:
    path = os.path.join(folder, TRANSCRIPT_FILE)
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("role") == "user":
                    out.append(str(row.get("text") or ""))
    except OSError:
        pass
    return out


@dataclass
class LookedUp:
    """What one conversation looked up, from the app's own record."""
    marks: Marks = field(default_factory=Marks)
    #: The records read ("activity log", "your messages"...).
    records: List[str] = field(default_factory=list)
    #: Whether the activity log was found; without it nothing is checked.
    knows: bool = False


def _record_folder(record_folder: Optional[str]) -> Optional[str]:
    if record_folder:
        return record_folder
    try:
        from funhouse_agent.cited_pages import record_dir
        return record_dir()
    except Exception:  # noqa: BLE001
        return None


def looked_up(record_folder: Optional[str] = None,
              user_words: Optional[Sequence[str]] = None) -> LookedUp:
    """What this conversation looked up and what its user said, from the
    app's records (never from what a model says it read).

    ``record_folder`` holds ``activity.jsonl`` / ``transcript.jsonl`` (the
    one the turn bound, or the working folder's, when ``None``);
    ``user_words`` are the user's messages (those the turn bound when
    ``None``)."""
    out = LookedUp()
    base = _record_folder(record_folder)
    if base:
        path = os.path.join(base, ACTIVITY_FILE)
        if os.path.isfile(path):
            st = _state_for(path)
            with st.lock:
                try:
                    _feed(st, path)
                except Exception:  # noqa: BLE001 - a record problem
                    pass
                out.marks.merge(st.marks)
            out.records.append("activity log")
            out.knows = True
        said = _transcript_user_texts(base)
        for text in said:
            out.marks.add_text(text, USER)
        if said:
            out.records.append("transcript")
    words = user_words if user_words is not None else _USER_WORDS.get()
    if words:
        for text in words:
            out.marks.add_text(text, USER)
        out.records.append("your messages")
    return out


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def check_text(text: Any, looked: Optional[LookedUp] = None,
               record_folder: Optional[str] = None,
               user_words: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Every document and work ``text`` cites, against what the
    conversation looked up.

    Returns ``{"checked", "n_cited", "cited", "not_looked_up", "user_named",
    "looked_up", "year_not_checked", "against"}`` (``checked`` False, with
    ``why``, when there is no record to check against). Each listed
    citation is ``{"cited": as shown, ...}``."""
    cites = find_citations(text)
    if not cites:
        return {"checked": True, "n_cited": 0}
    result: Dict[str, Any] = {"n_cited": len(cites),
                              "cited": [c.shown for c in cites][:MAX_LISTED]}
    if looked is None:
        looked = looked_up(record_folder, user_words)
    if not looked.knows:
        result.update(checked=False,
                      why="this host keeps no record of what was looked up "
                          "(no activity log), so cited documents were not "
                          "checked")
        return result
    result["checked"] = True
    result["against"] = ", ".join(looked.records)
    marks = looked.marks
    lists: Dict[str, List[Dict[str, Any]]] = {
        "not_looked_up": [], "user_named": [], "looked_up": [],
        "year_not_checked": []}
    for c in cites:
        row: Dict[str, Any] = {"cited": c.shown}
        if c.kind == "work":
            by = marks.pairs.get((c.author, c.year))
            if by is None:
                named = marks.names.get(c.author)
                if named is not None:
                    row["named_by"] = SOURCE_WORDS.get(named, named)
                    lists["year_not_checked"].append(row)
                    continue
        else:
            by = marks.document_kind(c.key)
            row["document"] = c.key
        if by is None:
            lists["not_looked_up"].append(row)
        elif by == USER:
            lists["user_named"].append(row)
        else:
            row["by"] = SOURCE_WORDS.get(by, by)
            lists["looked_up"].append(row)
    for k, v in lists.items():
        if v:
            result[k] = v[:MAX_LISTED]
            if len(v) > MAX_LISTED:
                result[f"n_{k}"] = len(v)
    return result


def not_looked_up(result: Dict[str, Any]) -> List[str]:
    """The citations a check found cited but never looked up, as shown."""
    return [str(r.get("cited") or "") for r in
            result.get("not_looked_up") or []]


def _listing(names: Sequence[str], limit: int = 12) -> str:
    listed = ", ".join(names[:limit])
    more = len(names) - limit
    return listed + (f", and {more} more" if more > 0 else "")


def note_for_user(result: Dict[str, Any]) -> Optional[str]:
    """The note appended to an answer, or ``None``."""
    names = not_looked_up(result)
    if not names:
        return None
    return ("**Citation check (by the app).** This answer cites "
            f"{_listing(names)}, which nothing in this conversation looked "
            "up: no references result, no document opened and no message of "
            "yours names them. Treat those citations as unchecked until they "
            "are looked up.")


def note_for_model(result: Dict[str, Any]) -> Optional[str]:
    """The note the model is given once (``GEOTECH_CITED_DOCUMENTS=retry``)."""
    names = not_looked_up(result)
    if not names:
        return None
    lines = "\n".join(f"- {n}" for n in names[:20])
    return (f"{NOTE_PREFIX} Before your answer goes to the user, the app "
            "checked every code, manual, standard and author-and-year work "
            "it cites against what this conversation actually looked up: "
            "references results, documents opened, worked examples, the "
            "calculations' own results and the user's own messages. These "
            "are cited but were never looked up:\n" + lines + "\n"
            "The answer you wrote is held back: the user will not see it. "
            "Look each one up in the references and check what you say about "
            "it, or drop the citation, or say plainly that it was not looked "
            "up. A citation is the user's only when the user's own words "
            "name it. This note comes once. Then write your whole answer to "
            "the user's request again, as they will read it: it replaces the "
            "one held back.")


def result_note(names: Sequence[str]) -> str:
    """The one line a writer's result gains."""
    return (f"cited but never looked up in this conversation: "
            f"{_listing(list(names))} — consult the references (the "
            "references helper, or a reference module), or drop the "
            "citation; a citation is the user's only when the user's own "
            "words name it. The file was written as given.")


# ---------------------------------------------------------------------------
# A writer's check (calc_package, write_docx, save_file, annotate_document)
# ---------------------------------------------------------------------------

#: Keys of a writer's arguments that are not text written into the file.
_NOT_TEXT_KEYS = ("output_path", "output_dir", "path", "format", "encoding",
                  "attachment_key", "file_path", "handle", "source",
                  "renderer", "page", "pdf_page", "bbox", "box", "point",
                  "points_at", "view", "image_box", "anchor", "quote",
                  "reply_to", "id", "color", "colour", "kind", "type",
                  "shape", "remove", "_author", "name_prefix", "package")
#: ``save_file`` files that are reports a reader opens (a CSV or JSON data
#: file is not checked).
REPORT_EXTENSIONS = (".md", ".markdown", ".html", ".htm", ".txt", ".tex",
                     ".rst")


def deliverable_texts(arguments: Any) -> List[str]:
    """The text a writer was GIVEN to put in its file (its string arguments
    other than paths, formats and anchors)."""
    return [s for s in _strings(arguments, [], skip=_NOT_TEXT_KEYS)
            if s.strip()]


def _event(data: Dict[str, Any]) -> None:
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(RECORD_EVENT, data)
    except Exception:  # noqa: BLE001 - outside a run, or no logger
        pass


def check_deliverable(texts: Iterable[str], where: str,
                      file: str = "", mode: Optional[str] = None
                      ) -> Dict[str, Any]:
    """The fields a writer's result gains for the citations in ``texts``:
    ``{cited_not_looked_up, citation_note}`` with the switch on and
    something cited that nothing looked up, else ``{}``. Whatever the
    switch, a check that finds a citation is recorded (inside a turn) as a
    :data:`RECORD_EVENT` event. Never raises; never refuses the file."""
    try:
        if mode is None:
            from funhouse_agent import review_flags
            mode = review_flags.cited_documents()
        joined = "\n\n".join(str(t) for t in texts if t)
        if not joined.strip() or not find_citations(joined):
            return {}
        if not _record_folder(None):
            return {}
        result = check_text(joined)
        if not result.get("checked"):
            return {}
        names = not_looked_up(result)
        add: Dict[str, Any] = {}
        if names and mode in ("note", "retry"):
            add = {RESULT_LIST: names, RESULT_NOTE: result_note(names)}
        _event({**result, "where": where, "file": _base(file) or None,
                "mode": mode, "action": "noted" if add else "none"})
        return add
    except Exception:  # noqa: BLE001 - never the writer's result
        return {}


def add_to_result(result: Any, texts: Iterable[str], where: str,
                  file: str = "") -> Any:
    """``result`` (a dict, or a JSON string of one) with the writer's check
    added (:func:`check_deliverable`); unchanged when the write failed or
    there is nothing to add."""
    try:
        data = result if isinstance(result, dict) else _json(result)
        if not isinstance(data, dict) or data.get("error") \
                or data.get("status") == "error":
            return result
        name = file or str(data.get("output_path") or data.get("path")
                           or data.get("saved_path") or "")
        add = check_deliverable(texts, where, name)
        if not add:
            return result
        if isinstance(result, dict):
            out = dict(result)
            out.update(add)
            return out
        return json.dumps({**data, **add})
    except Exception:  # noqa: BLE001
        return result


# ---------------------------------------------------------------------------
# One turn's check (webapp.core.stream_turn)
# ---------------------------------------------------------------------------

class TurnCitationCheck:
    """The citation check of one turn's answer: ``feedback`` before the
    answer goes out (``retry`` mode, once), ``finish`` when it does
    (records the check and, per the mode, notes the answer). Never raises.
    ``user_words`` are the user's messages of the conversation (``bound``
    gives them to the turn's tools for the writers' check)."""

    retry_status = RETRY_STATUS

    def __init__(self, record_folder: Optional[str] = None,
                 user_words: Optional[Sequence[str]] = None,
                 log: Optional[Callable[[Dict[str, Any]], None]] = None,
                 mode: Optional[str] = None):
        self.record_folder = record_folder
        self.user_words = list(user_words) if user_words is not None \
            else None
        self.log = log
        if mode is None:
            try:
                from funhouse_agent import review_flags
                mode = review_flags.cited_documents()
            except Exception:  # noqa: BLE001
                mode = "off"
        self.mode = mode
        self.fed_back = False

    def bound(self):
        """The user's words bound for the turn's tools."""
        return user_words_bound(self.user_words)

    def _check(self, answer: str) -> Dict[str, Any]:
        try:
            if not self.record_folder:
                return {"checked": False, "n_cited": len(find_citations(
                    answer)), "why": "no record folder for this turn"}
            return check_text(answer, record_folder=self.record_folder,
                              user_words=self.user_words or ())
        except Exception as exc:  # noqa: BLE001 - never the turn
            return {"checked": False, "n_cited": 0,
                    "why": f"the check failed: {type(exc).__name__}: {exc}"}

    def _record(self, result: Dict[str, Any], action: str) -> None:
        if self.log is None or not result.get("n_cited"):
            return
        try:
            self.log({**result, "where": "answer", "mode": self.mode,
                      "action": action})
        except Exception:  # noqa: BLE001
            pass

    def feedback(self, answer: str) -> Optional[str]:
        """The note to give the model before this answer goes out, once a
        turn, in ``retry`` mode; else ``None``."""
        if self.mode != "retry" or self.fed_back or not str(answer).strip():
            return None
        result = self._check(answer)
        note = note_for_model(result)
        if note:
            self.fed_back = True
            self._record(result, "fed_back")
        return note

    def finish(self, answer: str) -> str:
        """``answer`` as the user gets it: checked, recorded, and in
        ``note`` mode (or after a ``retry`` that still cites documents
        nothing looked up) with the note appended."""
        try:
            if not str(answer or "").strip():
                return answer
            result = self._check(answer)
            note = (note_for_user(result)
                    if self.mode in ("note", "retry") else None)
            self._record(result, "noted" if note else "none")
            if note:
                return f"{str(answer).rstrip()}\n\n---\n{note}"
            return answer
        except Exception:  # noqa: BLE001 - never the turn
            return answer


__all__ = ["RECORD_EVENT", "NOTE_PREFIX", "RETRY_STATUS", "RESULT_LIST",
           "RESULT_NOTE", "ALIASES", "FAMILIES", "MODULE_DOCUMENTS",
           "DocCited", "Marks", "LookedUp", "TurnCitationCheck",
           "module_documents", "canonical", "covered", "plain_text",
           "find_citations", "source_kind", "user_texts", "user_words_bound",
           "looked_up", "check_text", "not_looked_up", "note_for_user",
           "note_for_model", "result_note", "deliverable_texts",
           "check_deliverable", "add_to_result", "clear_cache"]
