"""Model-first report ingest (BACKLOG RI-87; the plan's section 8 in
``module_work/REPORT_INGEST_PLAN.md``).

Under ``GEOTECH_INGEST_MODEL_FIRST=on`` the model reads each item's pages in
one call and its answer becomes the record. This module grows piece by piece
(section 8.9). Today it holds the probe of section 8.7, which says whether a
route carries a PDF to the model whole.

THE PROBE (:func:`probe_pdf_route`, wave MF-0). Two calls over a two-page
PDF made in memory. Page 2 holds two made-up code words: the TEXT word in
its text layer alone (invisible text, as a scanned page's OCR layer is),
the PICTURE word in a picture alone, with no text under it. Page 1 and the
questions hold neither. A route that hands the model the whole PDF, text
layer and page pictures both, brings both words back; one that sends the
text layer alone brings back the text word only; one that drops the PDF,
or keeps its first page only, brings back neither.

1. The PDF call sends the PDF (:func:`~report_ingest.engine.pdf_block`)
   and asks for every code word on page 2. ``use_pdf`` is true only when
   BOTH words come back.
2. The pictures call is the control: the two pages as pictures, asking for
   the picture word. A model that reads it there but not from the PDF was
   let down by the route, not by its eyes.

Set ``GEOTECH_FOUNDRY_CLAUDE_PDF=1`` only if the PDF call brought both
words back on the Foundry Claude proxy; the switch does nothing on any
other route. The PDF call goes out even where the engine would not send a
PDF (:attr:`~report_ingest.engine.ChatModelEngine.accepts_pdf` off),
because finding out is the point. A call that raises ``ValueError`` was
refused before anything was sent (the ``palantir_models`` Claude class
refuses a PDF); any other error is a failed call, and the probe is run
again. Every probe draws fresh code words, so a cached reply to an earlier
probe cannot pass for this one.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.engine import (FOUNDRY_CLAUDE_PDF_ENV,
                                  FOUNDRY_PROXY_LLM_TYPE, ChatModelEngine,
                                  engine_for, image_block, pdf_block,
                                  text_block, user)

#: Made-up words a probe's code words are drawn from, with four digits after.
CODE_WORDS = ("AMBERWICK", "COPPERWREN", "MARIGOLD", "QUILLBANK",
              "TALLOWMERE", "WILLOWCREST")

#: What the PDF call asks. Neither code word is in it.
PDF_ASK = ("Page 2 of this document holds two code words. Reply with every "
           "code word you find on page 2, exactly as printed, separated by a "
           "space, and nothing else. If you find none, reply NONE.")

#: What the pictures call asks: the picture word, the only one it can see.
PICTURES_ASK = ("Page 2 shows a code word in a picture. Reply with that code "
                "word exactly as printed and nothing else. If you cannot see "
                "page 2, reply NONE.")

#: The resolution the pictures call renders the probe's pages at.
PICTURE_DPI = 100


def fresh_code_words() -> Tuple[str, str]:
    """Two different made-up code words no model has seen, each a word and
    four digits: ``(text word, picture word)``."""
    pick = random.SystemRandom()
    first, second = pick.sample(CODE_WORDS, 2)
    return (f"{first}-{pick.randint(1000, 9999)}",
            f"{second}-{pick.randint(1000, 9999)}")


def _word_picture(line: str) -> bytes:
    """``line`` drawn as a PNG, with no text layer anywhere."""
    import pymupdf

    scratch = pymupdf.open()
    try:
        page = scratch.new_page(width=460, height=48)
        page.insert_text((8, 32), line, fontsize=20)
        return page.get_pixmap(dpi=150).tobytes("png")
    finally:
        scratch.close()


def probe_document(text_word: str, picture_word: str) -> bytes:
    """The probe's two-page PDF, made in memory. Page 2 prints
    ``text_word`` in its text layer alone (invisible text) and shows
    ``picture_word`` in a picture alone; page 1 holds neither."""
    import pymupdf

    doc = pymupdf.open()
    try:
        first = doc.new_page()
        first.insert_text((72, 120), "Page 1 of 2", fontsize=20)
        first.insert_text((72, 156), "The code words are on the next page.",
                          fontsize=20)
        second = doc.new_page()
        second.insert_text((72, 120), "Page 2 of 2", fontsize=20)
        second.insert_text((72, 156), f"The text code word is {text_word}.",
                           fontsize=20, render_mode=3)
        second.insert_image(
            pymupdf.Rect(72, 200, 532, 248),
            stream=_word_picture(f"The picture code word is {picture_word}."))
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


def _pictures(pdf: bytes) -> List[bytes]:
    """Each page of ``pdf`` as a PNG."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_pixmap(dpi=PICTURE_DPI).tobytes("png")
                for page in doc]


def _squashed(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def _ask(engine: Any, blocks: Sequence[Any],
         words: Tuple[str, str]) -> Dict[str, Any]:
    """One probe call: which code words its reply carries, what came back,
    or what the call raised (``outcome`` reply, refused or failed)."""
    started = time.time()
    try:
        reply = engine.complete([user(*blocks)])
    except Exception as exc:                            # noqa: BLE001
        return {"outcome": "refused" if isinstance(exc, ValueError)
                else "failed",
                "text_read": False, "picture_read": False, "reply": "",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "seconds": round(time.time() - started, 1)}
    said = _squashed(reply.text)
    return {"outcome": "reply",
            "text_read": _squashed(words[0]) in said,
            "picture_read": _squashed(words[1]) in said,
            "reply": reply.text[:200], "error": None,
            "stop_reason": reply.stop_reason, "served_by": reply.model,
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
            "seconds": round(time.time() - started, 1)}


def _on_the_proxy(engine: Any) -> bool:
    """Whether the engine's model is the Foundry Claude proxy's class."""
    try:
        return getattr(getattr(engine, "chat_model", None), "_llm_type",
                       None) == FOUNDRY_PROXY_LLM_TYPE
    except Exception:                                   # noqa: BLE001
        return False


def _verdict(name: str, proxy: bool, accepts: Optional[bool],
             by_pdf: Dict[str, Any], by_pictures: Dict[str, Any]) -> str:
    if by_pdf["outcome"] == "failed":
        return (f"{name}: the PDF call failed ({by_pdf['error'][:200]}); "
                f"run the probe again.")
    if by_pdf["text_read"] and by_pdf["picture_read"]:
        whole = (f"{name}: the PDF reached the model whole: its text word "
                 f"and its picture word both came back.")
        if proxy:
            return f"{whole} Set {FOUNDRY_CLAUDE_PDF_ENV}=1."
        return (f"{whole} {FOUNDRY_CLAUDE_PDF_ENV} applies only to the "
                f"Foundry Claude proxy and does nothing for this route; "
                + ("the engine already sends a PDF here." if accepts else
                   "a PDF goes out here only from an engine built with "
                   "accepts_pdf=True."))
    if by_pdf["outcome"] == "refused":
        why = (f"the PDF was refused before it was sent "
               f"({by_pdf['error'][:200]})")
    elif by_pdf["text_read"]:
        why = ("only the PDF's text word came back, not its picture word: "
               "the route sends the text layer without the page pictures")
    elif by_pdf["picture_read"]:
        why = ("only the PDF's picture word came back, not its text word: "
               "the route sends the page pictures without the text layer")
    else:
        why = "the PDF reply carried neither code word"
    if by_pictures["picture_read"]:
        return (f"{name}: {why}. The pictures call read page 2, so leave "
                f"{FOUNDRY_CLAUDE_PDF_ENV} unset and send the pages as "
                f"pictures.")
    if by_pictures["outcome"] == "failed":
        return (f"{name}: {why}, and the pictures call failed "
                f"({by_pictures['error'][:200]}); run the probe again.")
    return (f"{name}: {why}, and the pictures call did not read page 2 "
            f"either, so the probe says nothing about the PDF route. Read "
            f"the replies.")


def probe_pdf_route(model: Any, *, words: Optional[Tuple[str, str]] = None
                    ) -> Dict[str, Any]:
    """Whether a PDF reaches the model whole on this route: two calls.

    ``model`` is an ingest engine or what :func:`engine_for` turns into one
    (a LangChain chat model above all). ``words`` (text word, picture word)
    is for tests; a probe draws its own. Nothing raises from a call: a
    refusal or an error is recorded on that call.

    Returns ``{"model", "words", "proxy", "accepts_pdf", "pdf",
    "pictures", "use_pdf", "says"}``: ``proxy`` is whether the model is the
    Foundry Claude proxy's class, ``accepts_pdf`` what the engine said
    before the probe, ``pdf`` and ``pictures`` the two calls (``outcome``,
    ``text_read``, ``picture_read``, ``reply``, ``error``, the tokens and
    seconds), ``use_pdf`` whether the PDF call brought both words back, and
    ``says`` the verdict in a sentence.
    """
    engine = engine_for(model)
    if engine is None:
        raise ValueError(f"no ingest engine for a {type(model).__name__}")
    words = tuple(words) if words else fresh_code_words()
    pdf = probe_document(*words)
    accepts = getattr(engine, "accepts_pdf", None)
    sender = engine
    if isinstance(engine, ChatModelEngine) and not engine.accepts_pdf:
        sender = ChatModelEngine(engine.chat_model, meter=engine.meter,
                                 max_tokens=engine.max_tokens,
                                 name=engine.name, accepts_pdf=True)
    by_pdf = _ask(sender, [pdf_block(pdf), text_block(PDF_ASK)], words)
    first, second = _pictures(pdf)
    by_pictures = _ask(engine, [text_block("Page 1:"), image_block(first),
                                text_block("Page 2:"), image_block(second),
                                text_block(PICTURES_ASK)], words)
    name = str(getattr(engine, "name", "") or type(engine).__name__)
    proxy = _on_the_proxy(engine)
    return {"model": name, "words": list(words), "proxy": proxy,
            "accepts_pdf": accepts, "pdf": by_pdf, "pictures": by_pictures,
            "use_pdf": bool(by_pdf["text_read"] and by_pdf["picture_read"]),
            "says": _verdict(name, proxy, accepts, by_pdf, by_pictures)}
