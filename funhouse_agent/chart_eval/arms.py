"""The chart suite's arms: how a model is set up to read a chart.

Direct arms (one LangChain chat model, no app around it):

``none``
    The chart and the question in one message, no tools: the shape of
    Surge's own harness (``solver=generate()``), and of the "without tools"
    numbers in the Opus 5.5 system card.
``zoom``
    The Claude Cookbook's zoom tool (``multimodal/crop_tool.ipynb``): the
    chart shown at exactly the size the model reads, pixel coordinates in, a
    crop of the full-resolution original magnified to the model's image
    limits out, as many zooms as the model wants (up to ``max_steps``).
``zoom_code``
    The zoom tool plus ``run_python``: code the model writes, run on the
    full-resolution chart in the app's locked plotting box
    (:mod:`funhouse_agent.plot_box`: numpy, scipy, matplotlib, no network,
    no processes). The nearest this app comes to the system card's "with
    tools" setup (a container with the image and standard libraries, plus an
    image-cropping tool); the box takes no PIL or OpenCV import, so pixel
    work goes through ``matplotlib.pyplot.imread`` and numpy/scipy.

The page arms (``page:<arm>``, run by :mod:`.runner`) put the chart through
the app's own Document Review page instead, exactly as a user's upload goes,
with the switches of :data:`funhouse_agent.review_flags.ARMS` ``<arm>``:
``page:baseline`` is what users get today (the main model never sees the
image; one-shot vision side calls describe it and ``render_region`` zooms),
``page:minimal`` the looking-only agent that sees page images itself.

Images after a tool call. Both model families get a zoom's image the same
way: the tool result carries the text, and a user message right after it
carries the image (the app's inline-image route does the same). The cookbook
puts the image inside Claude's tool result; a user message beside it is the
form every Foundry route takes, OpenAI's chat format included.

Every direct run writes its whole record: ``transcript.jsonl`` (each model
call's text, reasoning, tool calls, usage and time; each tool call's
arguments and result) and every image the model was shown under
``images/``.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from funhouse_agent.chart_eval import imaging

DIRECT_ARMS = ("none", "zoom", "zoom_code")

#: The cookbook's zoom tool description, kept as it is.
ZOOM_DESCRIPTION = (
    "Zoom into a rectangular region of an image to see it magnified.\n\n"
    "Use this whenever text, numbers, lines, or details are too small to "
    "read confidently in the full view. The region is cropped from the "
    "image's full-resolution original and scaled up to fill the image "
    "budget, so small elements become much easier to read. Coordinates are "
    "absolute pixels in the image as you see it, origin at the top-left "
    "corner. Zoom again on the result for even finer detail.")

RUN_PYTHON_DESCRIPTION = (
    "Run Python code on the chart at full resolution, in a locked box (no "
    "network, no files outside the working folder, a time limit). The chart "
    "is the file chart.png in the working folder ({w}x{h} pixels, the "
    "full-resolution original; the image you see is it resized to "
    "{vw}x{vh}). Read it with:\n"
    "    import matplotlib.pyplot as plt\n"
    "    img = plt.imread('chart.png')   # numpy array, rows x cols x "
    "channels, values 0..1\n"
    "Allowed imports: numpy, scipy, matplotlib, pandas, math, statistics "
    "and the standard library's maths and text modules (no PIL, no OpenCV, "
    "no os). print() what you need; it comes back to you. To look at "
    "something you made, draw it with matplotlib and call "
    "save_figure(fig, 'name'): the saved image is shown to you. Use it to "
    "measure pixels (axis tick positions, a line's height at a given x, "
    "colours), not to guess.")

#: Retry waits (seconds) for a busy model (rate limit, overload).
BUSY_WAITS = (15, 30, 60, 120, 120)


class ArmError(RuntimeError):
    """The arm could not run (not the model's own fault)."""


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------

class Recorder:
    """Writes one run's transcript and the images the model was shown."""

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        self.images_dir = os.path.join(run_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self._path = os.path.join(run_dir, "transcript.jsonl")
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._n_images = 0
        open(self._path, "w", encoding="utf-8").close()

    def event(self, kind: str, **fields: Any) -> None:
        rec = {"t": round(time.monotonic() - self._t0, 2), "event": kind}
        rec.update(fields)
        with self._lock, open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def image(self, data: bytes, label: str) -> str:
        """Save an image the model is shown; its path relative to the run."""
        self._n_images += 1
        ext = ".jpg" if data[:3] == b"\xff\xd8\xff" else ".png"
        name = f"{self._n_images:02d}_{label}{ext}"
        with open(os.path.join(self.images_dir, name), "wb") as fh:
            fh.write(data)
        return f"images/{name}"


# ---------------------------------------------------------------------------
# Talking to the model
# ---------------------------------------------------------------------------

def _text_and_reasoning(content: Any):
    if isinstance(content, str):
        return content, ""
    text, reasoning = [], []
    for b in content or []:
        if isinstance(b, str):
            text.append(b)
        elif isinstance(b, dict):
            kind = b.get("type")
            if kind in ("text", "output_text"):
                text.append(str(b.get("text") or ""))
            elif kind in ("thinking", "reasoning"):
                reasoning.append(str(b.get("thinking") or b.get("reasoning")
                                     or b.get("text") or ""))
                for s in b.get("summary") or []:
                    if isinstance(s, dict):
                        reasoning.append(str(s.get("text") or ""))
    return "".join(text), "\n".join(r for r in reasoning if r)


def _usage(msg: Any) -> Dict[str, Any]:
    usage = getattr(msg, "usage_metadata", None) or {}
    out = {k: usage.get(k) for k in ("input_tokens", "output_tokens",
                                     "total_tokens") if usage.get(k)
           is not None}
    for k in ("input_token_details", "output_token_details"):
        if usage.get(k):
            out[k] = dict(usage[k])
    return out


def _is_busy(exc: Exception) -> bool:
    try:
        from funhouse_agent.deep.vision_engine import busy_kind
        return busy_kind(exc) is not None
    except Exception:                                     # noqa: BLE001
        text = str(exc).lower()
        return any(w in text for w in ("rate limit", "429", "overloaded",
                                       "529", "503", "too many requests"))


def _refused_detail(exc: Exception) -> bool:
    try:
        from funhouse_agent.deep.vision_engine import refused_detail
        return refused_detail(exc)
    except Exception:                                     # noqa: BLE001
        return False


def _strip_detail(messages: List[Any]) -> List[Any]:
    from langchain_core.messages import HumanMessage
    out = []
    for m in messages:
        if isinstance(m, HumanMessage) and isinstance(m.content, list):
            blocks = []
            for b in m.content:
                if (isinstance(b, dict) and b.get("type") == "image_url"
                        and isinstance(b.get("image_url"), dict)):
                    b = {"type": "image_url",
                         "image_url": {"url": b["image_url"]["url"]}}
                blocks.append(b)
            m = HumanMessage(content=blocks)
        out.append(m)
    return out


class Caller:
    """One run's model calls: retries a busy model, drops an image
    ``detail`` the model refuses (once, for the rest of the run), and
    records every call."""

    def __init__(self, model: Any, rec: Recorder,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.model = model
        self.rec = rec
        self.sleep = sleep
        self.calls = 0
        self.no_detail = False
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def invoke(self, runnable: Any, messages: List[Any], step: int):
        msgs = _strip_detail(messages) if self.no_detail else messages
        waits = list(BUSY_WAITS)
        while True:
            t0 = time.monotonic()
            try:
                reply = runnable.invoke(msgs)
                break
            except Exception as exc:                      # noqa: BLE001
                if not self.no_detail and _refused_detail(exc):
                    self.no_detail = True
                    self.rec.event("detail_refused", step=step,
                                   error=str(exc)[:300])
                    msgs = _strip_detail(messages)
                    continue
                if _is_busy(exc) and waits:
                    wait = waits.pop(0)
                    self.rec.event("busy_retry", step=step, wait_s=wait,
                                   error=str(exc)[:300])
                    self.sleep(wait)
                    continue
                self.rec.event("model_error", step=step,
                               error=f"{type(exc).__name__}: {exc}"[:2000])
                raise
        self.calls += 1
        text, reasoning = _text_and_reasoning(getattr(reply, "content", ""))
        usage = _usage(reply)
        for k in ("input_tokens", "output_tokens"):
            if isinstance(usage.get(k), (int, float)):
                self.usage[k] += int(usage[k])
        meta = getattr(reply, "response_metadata", None) or {}
        self.rec.event(
            "model_call", step=step, seconds=round(time.monotonic() - t0, 2),
            messages_sent=len(msgs), text=text, reasoning=reasoning or None,
            tool_calls=[{"name": c.get("name"), "args": c.get("args"),
                         "id": c.get("id")}
                        for c in getattr(reply, "tool_calls", None) or []],
            usage=usage,
            stop=meta.get("stop_reason") or meta.get("finish_reason"),
            model=meta.get("model") or meta.get("model_name"))
        return reply


def _image_block(data: bytes, detail: Optional[str]) -> Dict[str, Any]:
    image_url: Dict[str, Any] = {"url": imaging.data_uri(data)}
    if detail:
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def image_detail(model: Any) -> Optional[str]:
    """The ``detail`` to send: ``original`` for a GPT model (full size; the
    app's Foundry SDK route sends it as AUTO), none for Claude."""
    return None if imaging.family_of(model) == "anthropic" else "original"


# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

class Chart:
    """The chart as the arms use it: the original, and the view the model
    is shown (resized to exactly what the model reads)."""

    def __init__(self, path: str, limits: imaging.ImageLimits) -> None:
        self.path = path
        self.original = imaging.open_image(path)
        self.limits = limits
        self.view = imaging.prepare_image(self.original, limits)
        self.view_png = imaging.png_bytes(self.view)

    def describe(self) -> Dict[str, Any]:
        return {"original_px": [self.original.width, self.original.height],
                "view_px": [self.view.width, self.view.height],
                "limits": self.limits.to_dict()}


def run_none(model: Any, question: str, chart: Chart, run_dir: str,
             sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """The chart and the question, one call, no tools."""
    from langchain_core.messages import HumanMessage
    rec = Recorder(run_dir)
    detail = image_detail(model)
    rec.event("image_shown", file=rec.image(chart.view_png, "chart"),
              px=[chart.view.width, chart.view.height], detail=detail)
    caller = Caller(model, rec, sleep)
    messages = [HumanMessage(content=[
        {"type": "text", "text": question},
        _image_block(chart.view_png, detail)])]
    reply = caller.invoke(model, messages, step=1)
    text, _ = _text_and_reasoning(getattr(reply, "content", ""))
    return {"answer": text, "stop": "answered", "model_calls": caller.calls,
            "tool_counts": {}, "usage": caller.usage,
            "detail_dropped": caller.no_detail, "chart": chart.describe()}


def _zoom_tools(chart: Chart, with_code: bool, box_dir: Optional[str]):
    """The LangChain tools and their handlers (handlers return
    ``(text, [image bytes], is_error)``)."""
    from pydantic import BaseModel, Field
    from langchain_core.tools import StructuredTool

    class ZoomArgs(BaseModel):
        x1: int = Field(description="Left edge of the region in pixels "
                                    "(origin is the image's top-left "
                                    "corner).")
        y1: int = Field(description="Top edge of the region in pixels.")
        x2: int = Field(description="Right edge of the region in pixels "
                                    "(must be greater than x1).")
        y2: int = Field(description="Bottom edge of the region in pixels "
                                    "(must be greater than y1).")

    def zoom_handler(args: Dict[str, Any]):
        got = imaging.zoom(chart.original,
                           (chart.view.width, chart.view.height),
                           args.get("x1"), args.get("y1"), args.get("x2"),
                           args.get("y2"), chart.limits)
        if not got.ok:
            return got.text, [], True
        return got.text, [got.image], False

    def _never(**_: Any) -> str:          # the loop runs the handlers itself
        return ""

    tools = [StructuredTool.from_function(
        func=_never, name="zoom", description=ZOOM_DESCRIPTION,
        args_schema=ZoomArgs)]
    handlers = {"zoom": zoom_handler}

    if with_code:
        class CodeArgs(BaseModel):
            code: str = Field(description="The Python code to run.")

        def code_handler(args: Dict[str, Any]):
            return run_chart_code(str(args.get("code") or ""), box_dir,
                                  chart.limits)

        tools.append(StructuredTool.from_function(
            func=_never, name="run_python",
            description=RUN_PYTHON_DESCRIPTION.format(
                w=chart.original.width, h=chart.original.height,
                vw=chart.view.width, vh=chart.view.height),
            args_schema=CodeArgs))
        handlers["run_python"] = code_handler
    return tools, handlers


def run_chart_code(code: str, box_dir: Optional[str],
                   limits: imaging.ImageLimits):
    """``code`` in the locked box, in ``box_dir`` (which holds chart.png):
    ``(text, [images], is_error)``."""
    from funhouse_agent import plot_box
    if not box_dir or not plot_box.available():
        return "run_python is not available on this host.", [], True
    res = plot_box.run_plot_code(code, root=box_dir, name="run")
    err = res.get("error") or ""
    # The box is a figure tool: a script that only prints is a success here.
    if res.get("status") == "error" and err.startswith(
            "The code ran but saved no figure"):
        err = ""
    parts = []
    if res.get("printed"):
        parts.append("Printed:\n" + str(res["printed"]))
    if res.get("warnings"):
        parts.append("Warnings:\n" + str(res["warnings"])[-1500:])
    if err:
        parts.append("Error:\n" + str(err))
    images = []
    for entry in res.get("images") or []:
        path = entry.get("output_path") if isinstance(entry, dict) else None
        if path and os.path.isfile(path):
            try:
                img = imaging.prepare_image(imaging.open_image(path), limits)
                images.append(imaging.png_bytes(img))
                parts.append(f"Saved and shown to you: {entry.get('file')}")
            except Exception as exc:                      # noqa: BLE001
                parts.append(f"Saved {entry.get('file')} but it could not be "
                             f"shown ({type(exc).__name__}).")
    if not parts:
        parts.append("The code ran and printed nothing.")
    return "\n\n".join(parts), images, bool(err)


def run_zoom(model: Any, question: str, chart: Chart, run_dir: str,
             with_code: bool = False, max_steps: int = 20,
             sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """The cookbook's zoom loop (and ``run_python`` with ``with_code``)."""
    from langchain_core.messages import HumanMessage, ToolMessage
    rec = Recorder(run_dir)
    detail = image_detail(model)
    box_dir = None
    if with_code:
        box_dir = os.path.join(run_dir, "box")
        os.makedirs(box_dir, exist_ok=True)
        target = os.path.join(box_dir, "chart.png")
        if chart.path.lower().endswith(".png"):
            shutil.copyfile(chart.path, target)
        else:
            chart.original.save(target, format="PNG")
    tools, handlers = _zoom_tools(chart, with_code, box_dir)
    bound = model.bind_tools(tools)
    caller = Caller(model, rec, sleep)
    w, h = chart.view.width, chart.view.height
    rec.event("image_shown", file=rec.image(chart.view_png, "chart"),
              px=[w, h], detail=detail)
    closing = ("\nPixel coordinates start at (0, 0) in the top-left corner, "
               "with x increasing to the right and y increasing downward. "
               "Use the zoom tool to examine small details like legends, "
               "axis labels, and closely spaced values.")
    if with_code:
        closing += (" You can also run Python on the full-resolution chart "
                    "with run_python (the file chart.png, "
                    f"{chart.original.width}x{chart.original.height} "
                    "pixels) to measure what you see.")
    messages: List[Any] = [HumanMessage(content=[
        {"type": "text", "text": "Answer the following question about the "
                                 "image below.\n\nThe question is: "
                                 + question},
        {"type": "text", "text": f"\nThe image is {w}x{h} pixels:"},
        _image_block(chart.view_png, detail),
        {"type": "text", "text": closing}])]
    counts: Dict[str, int] = {}
    stop, answer, nudged = "answered", "", False
    for step in range(1, int(max_steps) + 2):
        reply = caller.invoke(bound, messages, step=step)
        messages.append(reply)
        calls = list(getattr(reply, "tool_calls", None) or [])
        text, _ = _text_and_reasoning(getattr(reply, "content", ""))
        if not calls:
            if text.strip() or nudged:
                answer = text
                break
            # a turn with neither a tool call nor text: one plain turn to
            # answer (the cookbook's manual loop does the same)
            nudged = True
            messages.append(HumanMessage(content="Please give your final "
                                                 "answer to the question."))
            continue
        if step > max_steps:
            # the budget is spent and it still asks for tools: take its text
            answer, stop = text, "step_cap"
            break
        shown: List[Any] = []
        for call in calls:
            name = call.get("name")
            counts[name] = counts.get(name, 0) + 1
            handler = handlers.get(name)
            if handler is None:
                result, images, is_error = (f"Unknown tool: {name}", [], True)
            else:
                try:
                    result, images, is_error = handler(call.get("args") or {})
                except Exception as exc:                  # noqa: BLE001
                    result, images, is_error = (
                        f"Error: {type(exc).__name__}: {exc}", [], True)
            files = [rec.image(img, name) for img in images]
            rec.event("tool", name=name, args=call.get("args"),
                      result=result, images=files, error=is_error)
            messages.append(ToolMessage(
                content=result + (f"\n[{len(images)} image(s) follow in the "
                                  "next message.]" if images else ""),
                tool_call_id=call.get("id") or name,
                status="error" if is_error else "success"))
            for img in images:
                shown.append({"type": "text",
                              "text": f"Result of {name} ({call.get('id')}):"})
                shown.append(_image_block(img, detail))
        if shown:
            messages.append(HumanMessage(content=shown))
        if step == max_steps:
            messages.append(HumanMessage(content=(
                "That was the last tool call this run allows. Give your "
                "final answer to the question now.")))
    if not answer.strip() and stop == "answered":
        stop = "no_answer"
    return {"answer": answer, "stop": stop, "model_calls": caller.calls,
            "tool_counts": counts, "usage": caller.usage,
            "detail_dropped": caller.no_detail, "chart": chart.describe()}


def run_direct(arm: str, model: Any, question: str, image_path: str,
               run_dir: str, limits: Any = None, max_steps: int = 20,
               sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """One direct arm on one chart."""
    if arm not in DIRECT_ARMS:
        raise ArmError(f"unknown direct arm {arm!r}; known: {DIRECT_ARMS}")
    lim = imaging.limits_for(model, limits)
    chart = Chart(image_path, lim)
    if arm == "none":
        return run_none(model, question, chart, run_dir, sleep=sleep)
    if arm == "zoom_code":
        from funhouse_agent import plot_box
        if not plot_box.available():
            raise ArmError("zoom_code needs the plotting box (matplotlib); "
                           "it is not available here")
    return run_zoom(model, question, chart, run_dir,
                    with_code=(arm == "zoom_code"), max_steps=max_steps,
                    sleep=sleep)


__all__ = ["DIRECT_ARMS", "run_direct", "run_none", "run_zoom",
           "run_chart_code", "Chart", "Recorder", "ArmError",
           "image_detail"]
