"""Show an uploaded image to the page's main model itself
(``GEOTECH_VISION_SHOW_UPLOADS``, BACKLOG DR-38).

On the chart suite (EV-35) every loss of the Document Review page traced to
the text a vision side call wrote about an uploaded chart: the main model
never saw the chart. ``GEOTECH_VISION_SHOW_ME`` lets the model ask to see an
image (``show_me=true``), but in the runs that offered it no call asked. So
behind this switch the choice is not the model's: on the page's main agent,
``analyze_image`` of an uploaded image FILE runs no vision call; the image is
shown to the main model at its next step, the way ``show_me`` and the
``minimal`` build show one (:mod:`aec_agent_core.deep.inline_images`: a user
message after the tool results, in the request only, the newest
``GEOTECH_VISION_INLINE_KEEP`` images). The image goes as the same
``image_url`` block the vision side calls send
(:class:`aec_agent_core.deep.vision_engine.LangChainVisionEngine`), which
each engine family's chat model already translates for its service; the
``detail`` is the engine's (:func:`aec_agent_core.vision_view.detail`).

What stays as it was:

* a PDF, a working-folder file opened by its path, a contact sheet: the
  switch is for the user's uploaded image files only;
* the reading helper and every other helper: only the main agent's
  ``analyze_image`` calls are marked (:class:`UploadImageMiddleware`), so a
  helper's look is today's vision call;
* ``render_region`` on the upload: a zoom is still read by a vision call;
* the geotech page and the ``minimal`` build: they never carry the
  middleware, so nothing here runs on them whatever the switch says.

The limits of a look hold: the image is first shrunk by the rule a
whole-image look uses (:func:`aec_agent_core.image_view.render`, to the
model's budget); one shown image is one image call on the turn's count
(``image_calls`` in the record) and is refused past the turn's ceiling
(``GEOTECH_TURN_MAX_IMAGE_CALLS``) with the ceiling's own words; an image
still over :data:`aec_agent_core.vision_tools.INLINE_IMAGE_MAX_BYTES` after
shrinking is read by a vision call as before, and the result says why it
was not shown.

A main model that takes no images is told apart when the page is built
(:func:`takes_images`): the page then gets today's behaviour and the turn's
record says why (``build.show_uploads``).
"""

from __future__ import annotations

import contextvars
import json
import os
from typing import Any, Mapping, Optional, Tuple

from aec_agent_core.deep.inline_images import (
    InlineImageMiddleware, _ID, _text, newest_image_ids)

#: The looking tool whose uploaded images are shown.
TOOL = "analyze_image"

#: True while the main agent's ``analyze_image`` runs under the switch
#: (set by :class:`UploadImageMiddleware`, read by the vision tools).
_ON: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "aec_agent_core_show_upload", default=False)

#: What the model is told when an upload is shown to it; ``{in_view}`` says
#: how long (as the other inline notes do).
UPLOAD_NOTE = (
    "The uploaded image is shown to you with your next step {in_view}: look "
    "at it yourself now. No vision call read it and the prompt was not "
    "used. To have a part of it read out (small lettering, a table), zoom "
    "with render_region(attachment_key=<this image>, view=<this view>, "
    "image_box=<a 0-999 box on it>).")

#: Added when ``tiles`` or ``turn`` was asked for: neither applies to an
#: image the model looks at itself.
UNUSED_NOTE = ("{what} not used when the image is shown to you: it is "
               "shown whole, as the image shows it")

#: Added to a vision call's result when the upload was too large to show.
TOO_LARGE_NOTE = (
    "this upload was not shown to you: after shrinking to {w} x {h} px it is "
    "{mb:.1f} MB, over the {cap_mb:.0f} MB an image shown to you may be (a "
    "request has a size limit on some gateways), so a vision call read it "
    "as before")


#: Added to the image message when one step looked at more uploads than a
#: model call carries (``GEOTECH_VISION_INLINE_KEEP``): each result said
#: its image would be shown, and the oldest were not.
NOT_SHOWN_NOTE = (
    "[NOT shown to you: {names}. Only the newest {keep} image{s} can be "
    "shown at one step, so {this} never in front of you: say nothing about "
    "{it} from sight. To see {it}, call analyze_image on {it} again, at "
    "most {keep} image{s} in one step.]")


#: Put in an upload's result, in the request, when its stored image was
#: pushed out before the model call that would have shown it.
GONE_NOTE = (
    "{name} was NOT shown to you: this turn stored more than {most} images "
    "after it before your next step, so its image was no longer held when "
    "its turn came. It was never in front of you: say nothing about it "
    "from sight. To see it, call analyze_image on it again.")


def tell_not_shown(request):
    """``request`` with every upload result of this turn whose stored image
    was pushed out before it was shown saying so in place of its note (in
    the request only, at every later call of the turn); the names go to
    the turn's record (``uploads_not_shown``)."""
    from langchain_core.messages import HumanMessage, ToolMessage
    from aec_agent_core import inline_store
    from aec_agent_core.deep import keep_work
    messages, names = list(request.messages or []), []
    for at in range(len(messages) - 1, -1, -1):
        m = messages[at]
        if isinstance(m, HumanMessage):
            break
        for image_id in _ID.findall(_text(m.content)) \
                if isinstance(m, ToolMessage) else ():
            meta = inline_store.pushed_out(image_id) or {}
            if not meta.get("upload") or meta.get("shown"):
                continue
            name = os.path.basename(str(meta.get("source") or image_id))
            note = GONE_NOTE.format(name=name, most=inline_store.MAX_IMAGES)
            try:
                data = json.loads(_text(m.content))
                data.pop("note", None)
                text = json.dumps({"not_shown": note, **data})
            except (TypeError, ValueError, AttributeError):
                text = f"[not shown] {note}\n\n{_text(m.content)}"
            messages[at] = m.model_copy(update={"content": text})
            names.append(name)
    if not names:
        return request
    scope = keep_work.current_turn()
    if scope is not None:
        told = scope.__dict__.setdefault("uploads_not_shown", [])
        told.extend(n for n in reversed(names) if n not in told)
    return request.override(messages=messages)


def left_out(messages, shown) -> list:
    """The names of the uploads the newest step's ``analyze_image`` results
    stored (the tool results that end ``messages``) whose images are not
    among ``shown``, the ids this model call carries; oldest first."""
    from langchain_core.messages import ToolMessage
    from aec_agent_core import inline_store
    names: list = []
    for m in reversed(list(messages or [])):
        if not isinstance(m, ToolMessage):
            break
        for image_id in reversed(_ID.findall(_text(m.content))):
            got = inline_store.get(image_id)
            if got is None or image_id in shown or not got[1].get("upload"):
                continue
            name = os.path.basename(str(got[1].get("source") or image_id))
            if name not in names:
                names.append(name)
    return list(reversed(names))


def active() -> bool:
    """Whether the running tool call is the main agent's ``analyze_image``
    under the switch."""
    return bool(_ON.get())


def takes_images(model: Any, engine: Any) -> Tuple[bool, str]:
    """``(yes, why)``: whether the page's main ``model`` is known to take
    images, decided before any image is sent.

    No when the model's own profile says it takes none (LangChain's
    ``profile["image_inputs"]``), when the page has no vision engine, or
    when its vision engine reads images with another model (then nothing
    shows this one takes them). Yes when the page's looking tools already
    send their images to this same model object, as the page builds them
    by default."""
    profile = getattr(model, "profile", None)
    if isinstance(profile, Mapping) and profile.get("image_inputs") is False:
        return False, "the main model's own profile says it takes no images"
    if engine is None:
        return False, "the page has no vision engine, so it reads no image"
    if getattr(engine, "model", None) is not model:
        return False, ("the page's images are read by another model than "
                       "its main model, so the main model is not known to "
                       "take images")
    return True, ("the page's looking tools already send their images to "
                  "its main model")


class UploadImageMiddleware(InlineImageMiddleware):
    """The page's main agent under ``GEOTECH_VISION_SHOW_UPLOADS``: its
    ``analyze_image`` calls are marked (:func:`active`), and the newest
    stored images are shown to its model at each call, as
    :class:`InlineImageMiddleware` shows them (a ``show_me`` look's too)."""

    @staticmethod
    def _is_look(request) -> bool:
        call = getattr(request, "tool_call", None)
        return isinstance(call, dict) and call.get("name") == TOOL

    def _with_images(self, request):
        """The request with the newest images, and, straight after a step
        that looked at more uploads than a call carries, a line naming the
        ones not shown (their results said they would be)."""
        request = tell_not_shown(request)
        out = super()._with_images(request)
        shown = newest_image_ids(request.messages, self.keep)
        from aec_agent_core import inline_store
        for image_id in shown:
            # an image pushed out later is then known to have been seen
            (inline_store.get(image_id) or (b"", {}))[1]["shown"] = True
        names = left_out(request.messages, shown) if shown else []
        if not names:
            return out
        from langchain_core.messages import HumanMessage
        one = len(names) == 1
        note = NOT_SHOWN_NOTE.format(
            names=", ".join(names), keep=self.keep,
            s="" if self.keep == 1 else "s",
            this="it was" if one else "they were",
            it="it" if one else "them")
        *before, last = out.messages
        return out.override(messages=[*before, HumanMessage(
            content=[*last.content, {"type": "text", "text": note}])])

    def wrap_tool_call(self, request, handler):
        if not self._is_look(request):
            return handler(request)
        token = _ON.set(True)
        try:
            return handler(request)
        finally:
            _ON.reset(token)

    async def awrap_tool_call(self, request, handler):
        if not self._is_look(request):
            return await handler(request)
        token = _ON.set(True)
        try:
            return await handler(request)
        finally:
            _ON.reset(token)


def _today(arguments, engine, attachments, note: Optional[str] = None) -> str:
    """Today's ``analyze_image`` result (a vision call, or its refusal),
    with ``note`` added when one is given."""
    from aec_agent_core import vision_tools
    token = _ON.set(False)
    try:
        # the call that came here already wrote this read to the run
        # record (``image_read``): the second pass does not write it again
        raw = vision_tools._dispatch_analyze_image(
            {**arguments, "_image_read_noted": True}, engine, attachments)
    finally:
        _ON.reset(token)
    if not note:
        return raw
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        data = None
    if isinstance(data, dict):
        data["not_shown"] = note
        return json.dumps(data)
    return f"[not shown] {note}.\n\n{raw}"


def show_upload(arguments: dict, engine: Any, attachments: Mapping,
                data: bytes, key: str, tools=None) -> str:
    """``analyze_image`` of the uploaded file ``key`` (its bytes ``data``)
    on the page's main agent under the switch: the image stored and shown
    to the model at its next step, or today's result where it is not an
    image, cannot be decoded here, or is too large to show."""
    from aec_agent_core import image_view, inline_store, vision_tools
    picture, refusal = vision_tools._open_image(data, key, tools)
    if refusal is not None or picture is None:
        # a PDF or another file that is not an image, or a damaged image:
        # today's answer, word for word
        return _today(arguments, engine, attachments)
    try:
        shown, info = image_view.render(
            picture, None, engine, turn=0, turn_from="asked",
            allow_jpeg=getattr(engine, "accepts_jpeg", False))
    except Exception:  # noqa: BLE001 - today's look says what went wrong
        return _today(arguments, engine, attachments)
    cap = vision_tools.INLINE_IMAGE_MAX_BYTES
    if len(shown) > cap:
        return _today(arguments, engine, attachments, TOO_LARGE_NOTE.format(
            w=info["width_px"], h=info["height_px"],
            mb=len(shown) / 1048576.0, cap_mb=cap / 1048576.0))
    from aec_agent_core.deep import keep_work
    try:
        # one image shown = one image call on the turn (TA-43)
        keep_work.take_image_call()
    except keep_work.ImageBudgetSpent as exc:
        return json.dumps({"error": str(exc)})
    clip = [int(v) for v in info["clip"]]
    name = os.path.basename(str(key))
    image_id = inline_store.put(shown, {
        "label": f"the uploaded image {name}, view {clip} in its own pixels",
        "view": clip, "source": str(key), "upload": True})
    out = {"image_id": image_id, "source": str(key),
           **image_view.view_payload(info, picture, engine),
           "note": UPLOAD_NOTE.format(in_view=vision_tools._in_view())}
    out.pop("zoom_hint", None)
    if picture.note:
        # what today's read says of the file (only the first page of a
        # TIFF or GIF, a photo turned as its EXIF says) holds for the image
        # shown as well
        out["image_note"] = picture.note
    unused = [w for w, given in (
        ("tiles is", arguments.get("tiles") not in (None, "", "auto")),
        ("turn is", bool(arguments.get("turn")))) if given]
    if unused:
        out["unused"] = "; ".join(UNUSED_NOTE.format(what=w) for w in unused)
    return json.dumps(out)


__all__ = ["TOOL", "UPLOAD_NOTE", "UNUSED_NOTE", "TOO_LARGE_NOTE",
           "NOT_SHOWN_NOTE", "GONE_NOTE", "tell_not_shown", "left_out",
           "active",
           "takes_images", "UploadImageMiddleware", "show_upload"]
