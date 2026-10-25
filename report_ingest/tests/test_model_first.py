"""Model-first ingest (BACKLOG RI-87), offline: the PDF route probe.

:func:`~report_ingest.model_first.probe_pdf_route` runs on Foundry as wave
MF-0. Here a scripted chat model stands in for each route
(:mod:`report_ingest.tests.chat_fixtures`): one that hands the model the
whole PDF, one that hands it the text layer alone (pymupdf's ``get_text``),
one that keeps the first page, one whose class refuses a PDF, one that
fails. No key, no network, made-up words only.
"""

from __future__ import annotations

import base64
import re

import pytest

pytest.importorskip("langchain_core")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.messages import AIMessage  # noqa: E402

from report_ingest.engine import (  # noqa: E402
    FOUNDRY_CLAUDE_PDF_ENV, FOUNDRY_PROXY_LLM_TYPE, ChatModelEngine,
)
from report_ingest.model_first import (  # noqa: E402
    PDF_ASK, PICTURES_ASK, fresh_code_words, probe_document, probe_pdf_route,
)
from report_ingest.tests.chat_fixtures import ScriptedChatModel  # noqa: E402

TEXT_WORD, PICTURE_WORD = WORDS = ("AMBERWICK-1357", "MARIGOLD-2468")


def _said(text):
    return AIMessage(content=text,
                     response_metadata={"finish_reason": "stop"},
                     usage_metadata={"input_tokens": 900, "output_tokens": 6,
                                     "total_tokens": 906})


def _blocks(messages):
    """The last user turn's content blocks."""
    return list(messages[-1].content)


def _pdf_in(messages):
    """The bytes of the PDF the turn carries, or None."""
    files = [b for b in _blocks(messages)
             if isinstance(b, dict) and b.get("type") == "file"]
    return base64.b64decode(files[0]["base64"]) if files else None


def _page_texts(pdf):
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_text() for page in doc]


def _words_in(text):
    return " ".join(re.findall(r"[A-Z]+-\d{4}", text)) or "NONE"


def _text_layer_only(messages):
    """A route that hands the model the PDF's text layer and nothing else:
    every code word pymupdf's ``get_text`` finds."""
    return _said(_words_in(" ".join(_page_texts(_pdf_in(messages)))))


def _whole_pdf(messages):
    """A route that hands the model the text layer AND the page pictures:
    it reads the text word off the text and sees the picture on page 2."""
    pdf = _pdf_in(messages)
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        assert doc[1].get_images(), "page 2 carries the picture"
    return _said(f"{_words_in(_page_texts(pdf)[1])} {PICTURE_WORD}")


def _first_page_only(messages):
    return _said(_words_in(_page_texts(_pdf_in(messages))[0]))


def _sees_the_pictures(messages):
    """The control: two pictures, no PDF, no code word in any text, and the
    question asks for the picture word."""
    blocks = _blocks(messages)
    assert _pdf_in(messages) is None
    assert sum(b.get("type") == "image_url" for b in blocks) == 2
    assert blocks[-1] == {"type": "text", "text": PICTURES_ASK}
    assert not any(w in str(b.get("text", "")) for b in blocks for w in WORDS)
    return _said(f"The code word is {PICTURE_WORD.lower()}.")


class _RefusesAPdf(ScriptedChatModel):
    """A class that carries text and pictures only and refuses a PDF before
    sending anything, as the ``palantir_models`` Claude class does."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if _pdf_in(messages) is not None:
            raise ValueError("a content block of type 'file' "
                             "(application/pdf) cannot be sent")
        return super()._generate(messages, stop, run_manager, **kwargs)


class _OnTheProxy(ScriptedChatModel):
    @property
    def _llm_type(self) -> str:
        return FOUNDRY_PROXY_LLM_TYPE


# ---------------------------------------------------------------------------
# the probe's PDF
# ---------------------------------------------------------------------------

def test_page_2_holds_the_text_word_in_its_text_alone():
    pdf = probe_document(*WORDS)
    first, second = _page_texts(pdf)
    assert TEXT_WORD in second and TEXT_WORD not in first
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        # Invisible text: in the text layer, never drawn.
        hidden = "".join(chr(c[0]) for t in doc[1].get_texttrace()
                         if t["type"] == 3 for c in t["chars"])
        assert TEXT_WORD in hidden
        assert len(doc[1].get_images()) == 1 and not doc[0].get_images()
    other = probe_document("QUILLBANK-9999", PICTURE_WORD)
    with pymupdf.open(stream=pdf, filetype="pdf") as a, \
            pymupdf.open(stream=other, filetype="pdf") as b:
        assert a[1].get_pixmap(dpi=72).samples == \
            b[1].get_pixmap(dpi=72).samples, "the text word draws nothing"


def test_page_2_holds_the_picture_word_in_its_picture_alone():
    pdf = probe_document(*WORDS)
    assert PICTURE_WORD not in " ".join(_page_texts(pdf))
    other = probe_document(TEXT_WORD, "QUILLBANK-9999")
    with pymupdf.open(stream=pdf, filetype="pdf") as a, \
            pymupdf.open(stream=other, filetype="pdf") as b:
        assert a[1].get_pixmap(dpi=72).samples != \
            b[1].get_pixmap(dpi=72).samples, "the picture word is drawn"


def test_no_question_holds_a_word():
    assert not any(w in ask for w in WORDS for ask in (PDF_ASK, PICTURES_ASK))


def test_every_probe_draws_two_made_up_words():
    text, picture = fresh_code_words()
    assert re.fullmatch(r"[A-Z]+-\d{4}", text)
    assert re.fullmatch(r"[A-Z]+-\d{4}", picture)
    assert text.split("-")[0] != picture.split("-")[0]


# ---------------------------------------------------------------------------
# the verdicts
# ---------------------------------------------------------------------------

def test_a_route_that_hands_over_the_whole_pdf_passes():
    model = ScriptedChatModel(replies=[_whole_pdf, _sees_the_pictures])
    engine = ChatModelEngine(model)
    out = probe_pdf_route(engine, words=WORDS)

    assert out["use_pdf"] is True
    assert out["pdf"]["text_read"] and out["pdf"]["picture_read"]
    assert out["pdf"]["outcome"] == "reply" and out["pdf"]["error"] is None
    assert out["pictures"]["picture_read"]
    assert out["words"] == list(WORDS) and out["model"] == "scripted-model"
    # The PDF goes first in its turn, the question after it.
    first, ask = _blocks(model.requests[0]["messages"])
    assert first["type"] == "file" and ask == {"type": "text",
                                               "text": PDF_ASK}
    # Sent although the engine would send none by itself (a scripted model
    # is no Claude): finding out is the probe's point. The engine itself is
    # left as it was, and both calls are on its meter.
    assert out["accepts_pdf"] is False and engine.accepts_pdf is False
    assert engine.meter.calls == 2


def test_a_route_that_passes_the_text_layer_alone_does_not_pass():
    model = ScriptedChatModel(replies=[_text_layer_only, _sees_the_pictures])
    out = probe_pdf_route(model, words=WORDS)
    assert out["pdf"]["reply"] == TEXT_WORD
    assert out["pdf"]["text_read"] and not out["pdf"]["picture_read"]
    assert out["use_pdf"] is False
    assert "only the PDF's text word came back" in out["says"]
    assert "leave GEOTECH_FOUNDRY_CLAUDE_PDF unset" in out["says"]


def test_a_route_that_keeps_the_first_page_only_does_not_pass():
    model = ScriptedChatModel(replies=[_first_page_only, _sees_the_pictures])
    out = probe_pdf_route(model, words=WORDS)
    assert out["use_pdf"] is False and out["pdf"]["reply"] == "NONE"
    assert "carried neither code word" in out["says"]
    assert "send the pages as pictures" in out["says"]


def test_a_refusal_before_sending_is_reported_as_refused():
    model = _RefusesAPdf(replies=[_sees_the_pictures])
    out = probe_pdf_route(model, words=WORDS)
    assert out["use_pdf"] is False and out["pdf"]["outcome"] == "refused"
    assert "type 'file'" in out["pdf"]["error"]
    assert len(model.requests) == 1, "the pictures call alone went out"
    assert "refused before it was sent" in out["says"]


def test_any_other_error_is_a_failed_call_to_run_again():
    model = ScriptedChatModel(replies=[TimeoutError("the request timed out"),
                                       _sees_the_pictures])
    out = probe_pdf_route(model, words=WORDS)
    assert out["use_pdf"] is False and out["pdf"]["outcome"] == "failed"
    assert out["says"].endswith("run the probe again.")


def test_neither_call_reading_page_2_says_the_probe_tells_nothing():
    model = ScriptedChatModel(replies=[_said("NONE"), _said("NONE")])
    out = probe_pdf_route(model, words=WORDS)
    assert out["use_pdf"] is False and not out["pictures"]["picture_read"]
    assert "says nothing about the PDF route" in out["says"]


def test_on_the_foundry_proxy_a_pass_sets_the_switch():
    out = probe_pdf_route(_OnTheProxy(replies=[_whole_pdf,
                                               _sees_the_pictures]),
                          words=WORDS)
    assert out["use_pdf"] is True and out["proxy"] is True
    assert out["says"].endswith(f"Set {FOUNDRY_CLAUDE_PDF_ENV}=1.")


def test_elsewhere_a_pass_says_the_switch_does_nothing_there():
    out = probe_pdf_route(ScriptedChatModel(
        replies=[_whole_pdf, _sees_the_pictures]), words=WORDS)
    assert out["use_pdf"] is True and out["proxy"] is False
    assert ("applies only to the Foundry Claude proxy and does nothing for "
            "this route") in out["says"]
    assert "accepts_pdf=True" in out["says"]
    assert f"Set {FOUNDRY_CLAUDE_PDF_ENV}=1" not in out["says"]


def test_nothing_to_probe_is_said():
    with pytest.raises(ValueError, match="no ingest engine"):
        probe_pdf_route(object())
