"""Error text fit to show: no server paths, and a tool error the turn survives.

Live smoke wave 2b (C3): ``annotate_document`` raised MuPDF's
``FzErrorSystem: cannot remove file 'C:\\...\\users\\livesmoke__tester\\...
\\review_set_marked.pdf'``. Nothing between the tool and the graph caught it,
so the whole turn ended on "(no answer text)" and the raw text -- the server
path with the tester's folder in it -- was shown to the tester. On Tiny Apps
the same text would show ``/home/data/geotech_webapp/users/<domain>__<user>/``.

* :func:`scrub_paths` takes every absolute path out of a text: the host's
  working folder becomes the file's conversation-relative name
  (``_fileio.hide_working_folder``), and any other absolute path -- Windows,
  UNC or POSIX, quoted or not -- becomes its file name. URLs are left alone.
* :func:`tool_error` is the JSON a tool returns instead of raising: what
  failed, said plainly, and that the turn goes on.
* :func:`budget_exhausted` tells an AI budget or quota that is USED UP
  apart from a model that is merely busy (live smoke wave 2c, D2): asking
  again in a minute fixes a rate limit, never a spent budget, so the two
  are said differently and only the first is retried.

Pure Python, no third-party import: the document tools, the agent's tool
middleware and the web app all use it.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

#: A quoted absolute path (Windows drive, UNC or POSIX): its quotes kept.
_QUOTED = re.compile(
    r"""(?P<q>['"])(?P<p>(?:[A-Za-z]:[\\/]|\\\\|/)[^'"\n]*?)(?P=q)""")
#: An unquoted Windows or UNC path (up to whitespace or a closing bracket).
_WIN = re.compile(r"(?<![\w/\\])(?:[A-Za-z]:[\\/]|\\\\)[^\s'\"<>|()\[\]{}]+")
#: An unquoted POSIX path of two or more parts, not part of a URL
#: ("https://h/x" has a ':' or '/' before each of its slashes).
_POSIX = re.compile(
    r"(?<![\w:/.~\\])/(?:[^\s/'\"<>|()\[\]{}]+/)+[^\s/'\"<>|()\[\]{}]*")


def _name_of(path: str) -> str:
    """The last part of ``path``; a folder's own name for a trailing slash."""
    parts = [p for p in re.split(r"[\\/]+", path.strip()) if p]
    return parts[-1] if parts else "(a folder)"


def scrub_paths(text: Any) -> str:
    """``text`` with every absolute path replaced by its file name.

    The host's working folder goes first (a path inside it keeps its
    conversation-relative name, e.g. ``figs/x.png``); then any other
    absolute path, quoted or not, becomes its last part. Never raises."""
    s = "" if text is None else str(text)
    if not s:
        return s
    try:
        from funhouse_agent._fileio import hide_working_folder
        hidden = hide_working_folder(s)
        if isinstance(hidden, str):
            s = hidden
    except Exception:  # noqa: BLE001 - a scrub never fails the caller
        pass
    try:
        s = _QUOTED.sub(lambda m: f"{m.group('q')}{_name_of(m.group('p'))}"
                                  f"{m.group('q')}", s)
        s = _WIN.sub(lambda m: _name_of(m.group(0)), s)
        s = _POSIX.sub(lambda m: _name_of(m.group(0)), s)
    except Exception:  # noqa: BLE001
        pass
    return s


def error_line(exc: BaseException, limit: int = 300) -> str:
    """``"<Type>: <message>"`` on one line, paths scrubbed, at most
    ``limit`` characters of message. An AI budget that is used up is said in
    plain words instead (:func:`budget_exhausted`): its raw text is provider
    JSON with a request id in it, which a tester should never be shown."""
    stop = budget_exhausted(exc)
    if stop is not None:
        return budget_tool_note(stop)
    msg = " ".join(scrub_paths(str(exc)).split())
    if len(msg) > limit:
        msg = msg[:limit] + " …"
    name = type(exc).__name__
    return f"{name}: {msg}" if msg else name


def tool_error(tool: str, exc: BaseException,
               hint: Optional[str] = None) -> Dict[str, Any]:
    """The result a tool gives instead of raising ``exc``: what failed (no
    server path in it), and that the conversation goes on."""
    if budget_exhausted(exc) is not None:
        hint = BUDGET_HINT
    out: Dict[str, Any] = {
        "error": f"{tool or 'the tool'} failed: {error_line(exc)}",
        "hint": hint or (
            "The tool raised an error instead of answering; nothing it was "
            "doing was finished. Try it again once if the cause looks "
            "passing, or another way; tell the user plainly what could not "
            "be done rather than stopping."),
    }
    return out


def tool_error_json(tool: str, exc: BaseException,
                    hint: Optional[str] = None) -> str:
    return json.dumps(tool_error(tool, exc, hint), ensure_ascii=False)


# ---------------------------------------------------------------------------
# An AI budget or quota that is used up (live smoke wave 2c, D2)
# ---------------------------------------------------------------------------
# Live smoke 2c: the API account reached its usage limit and every turn
# failed in under a second. The tester saw "ask again, or ask the agent to
# continue" over Anthropic's raw 400 -- {'type': 'error', 'error': {'type':
# 'invalid_request_error', 'message': 'You have reached your specified API
# usage limits. You will regain access on 2026-11-01 at 00:00 UTC.'},
# 'request_id': ...}. Only the Funhouse SDK's BudgetExceededError was known,
# by its class name. On Tiny Apps the shared Prompter key ($50 a month) WILL
# run out, and an OpenAI-style gateway says so with a 429 whose code is
# insufficient_quota -- which the busy rule called a rate limit, retried, and
# told the tester to "wait a minute".
#
# Read off the error's type name, HTTP status and body (its type/code, then
# its message), never off the words of an arbitrary exception: only the
# phrases below, which providers use for a spent account and nothing else.

#: Body ``type`` / ``code`` values that mean the money or quota is spent.
BUDGET_CODES = frozenset({
    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
    "quota_exceeded", "budget_exceeded", "credit_limit_exceeded",
    "spend_limit_exceeded",
})

#: Words in an error's message that mean the same (lower case).
BUDGET_PHRASES = (
    "usage limit",                    # Anthropic: "...API usage limits..."
    "credit balance is too low",      # Anthropic billing
    "exceeded your current quota",    # OpenAI insufficient_quota
    "insufficient_quota",
    "out of call volume quota",       # Azure API Management quota policy
    "quota will be replenished",
    "budget exceeded", "budget has been exceeded", "budget is exhausted",
    "budget exhausted", "budget is used up",
    "spend limit", "spending limit",
)

#: Class names that mean the same (lower case, without "error").
BUDGET_TYPE_NAMES = ("budgetexceeded", "insufficientquota", "quotaexceeded")

#: Body ``type`` / ``code`` values that mean a RATE limit: never a budget,
#: whatever the message says.
_RATE_CODES = frozenset({"rate_limit_exceeded", "rate_limit_error"})


@dataclass(frozen=True)
class BudgetStop:
    """An AI budget or quota that is used up: ``until`` is when it comes
    back, as the provider said it (``None`` when it did not), and ``why``
    what said so (a status, a code or a phrase) for the record."""

    until: Optional[str] = None
    why: str = ""


def _status_of(exc) -> Optional[int]:
    for holder in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status", "http_status"):
            v = getattr(holder, attr, None)
            if isinstance(v, int) and 100 <= v < 600:
                return v
    return None


def _body_parts(body) -> tuple:
    """``(codes, message)`` of an error body: the ``type`` / ``code`` values
    (lower case) and the message, from ``{"error": {...}}`` or the inner
    object itself."""
    codes, message = set(), ""
    if not isinstance(body, dict):
        return codes, message
    for holder in (body, body.get("error")):
        if not isinstance(holder, dict):
            continue
        for key in ("type", "code"):
            v = holder.get(key)
            if isinstance(v, str) and v.strip():
                codes.add(v.strip().lower())
        msg = holder.get("message")
        if isinstance(msg, str) and msg.strip():
            message = msg
    return codes, message


def _chain(exc):
    """``exc`` and the errors it was raised FROM (an explicit cause only)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__


#: Where a date the error gives ends: a full stop (then a space, the end, or
#: the quote / bracket of the JSON it sits in), a new line, or the end.
_END = r"(?=\.(?:\s|$|['\"\]}])|\n|$)"
_UNTIL_PATTERNS = (
    re.compile(r"regain access (?:on|at) ([^.\n'\"]{4,60}?)" + _END,
               re.IGNORECASE),
    re.compile(r"(?:resets?|renews?|is renewed|is reset) (?:on|at) "
               r"([^.\n'\"]{4,60}?)" + _END, re.IGNORECASE),
    re.compile(r"until (\d{4}-\d{2}-\d{2}[^.\n'\"]{0,40}?)" + _END,
               re.IGNORECASE),
)
#: Azure API Management: "Quota will be replenished in 2.04:12:23."
_REPLENISHED = re.compile(
    r"replenished in (?:(\d+)\.)?(\d{1,2}):(\d{2}):(\d{2})", re.IGNORECASE)


def _until(text: str, now: Optional[_dt.datetime] = None) -> Optional[str]:
    """When a spent budget comes back, as the error says it; ``None``."""
    if not text:
        return None
    m = _REPLENISHED.search(text)
    if m:
        days, h, mi, s = (int(m.group(1) or 0), int(m.group(2)),
                          int(m.group(3)), int(m.group(4)))
        base = now or _dt.datetime.now(_dt.timezone.utc)
        when = base + _dt.timedelta(days=days, hours=h, minutes=mi, seconds=s)
        return when.strftime("%Y-%m-%d %H:%M UTC")
    for pattern in _UNTIL_PATTERNS:
        m = pattern.search(text)
        if m:
            return " ".join(m.group(1).split()).rstrip(" ,;")
    return None


def budget_signal(status: Optional[int] = None, body: Any = None,
                  text: str = "", type_name: str = ""
                  ) -> Optional[BudgetStop]:
    """A :class:`BudgetStop` when an HTTP refusal says the AI budget or
    quota is used up, else ``None``. ``body`` is the parsed error body (a
    dict), ``text`` the raw message, ``type_name`` the exception's class
    name. A body coded as a rate limit is never a budget."""
    codes, message = _body_parts(body)
    name = (type_name or "").lower().replace("error", "")
    if codes & _RATE_CODES:
        return None
    why = ""
    if any(n in name for n in BUDGET_TYPE_NAMES):
        why = f"type {type_name}"
    elif codes & BUDGET_CODES:
        why = f"code {sorted(codes & BUDGET_CODES)[0]}"
    elif status == 402:
        why = "status 402"
    else:
        for words in (message, text):
            low = (words or "").lower()
            hit = next((p for p in BUDGET_PHRASES if p in low), None)
            if hit is not None:
                why = f"'{hit}'" + (f" (status {status})" if status else "")
                break
    if not why:
        return None
    return BudgetStop(until=_until(message) or _until(text), why=why)


def budget_exhausted(exc) -> Optional[BudgetStop]:
    """A :class:`BudgetStop` when ``exc`` (or an error it was raised from)
    says the AI budget or quota is USED UP -- the Funhouse SDK's
    ``BudgetExceededError``, an OpenAI / Azure ``insufficient_quota`` (a
    429), an API Management quota (a 403), Anthropic's 400 "You have reached
    your specified API usage limits", any 402 -- else ``None``. A plain rate
    limit (a 429 coded ``rate_limit_exceeded``, or saying nothing of a
    budget) is not one: that is a busy model, asked again later. Never
    raises."""
    try:
        for e in _chain(exc):
            stop = budget_signal(_status_of(e), getattr(e, "body", None),
                                 str(e), type(e).__name__)
            if stop is None:
                code = getattr(e, "code", None)
                if isinstance(code, str) and code.lower() in BUDGET_CODES:
                    stop = BudgetStop(until=_until(str(e)),
                                      why=f"code {code.lower()}")
            if stop is not None:
                return stop
    except Exception:  # noqa: BLE001 - classifying never fails the caller
        return None
    return None


def budget_message(stop: Optional[BudgetStop] = None) -> str:
    """What a TESTER is shown when the AI budget is used up: one plain
    line, the date it comes back when the provider gave one, no provider
    text and no "ask again"."""
    until = f" (until {stop.until})" if stop is not None and stop.until \
        else ""
    return (f"The AI budget for this app is used up{until}. Your "
            "conversation and its files are kept, but nothing more can run "
            "until the budget is renewed. Tell the app owner.")


def budget_tool_note(stop: Optional[BudgetStop] = None) -> str:
    """What a TOOL result says when a model call it made hit a spent
    budget: what happened, and that retrying cannot help."""
    until = f" (until {stop.until})" if stop is not None and stop.until \
        else ""
    return (f"the AI budget for this app is used up{until}, so this model "
            "call was refused and nothing was read")


#: The hint a tool result carries with :func:`budget_tool_note`.
BUDGET_HINT = ("No model call can run until the budget is renewed, so do NOT "
               "retry this or any other tool that looks at a page. Tell the "
               "user plainly that the app's AI budget is used up and that "
               "they should tell the app owner.")


__all__ = ["scrub_paths", "error_line", "tool_error", "tool_error_json",
           "BudgetStop", "budget_signal", "budget_exhausted",
           "budget_message", "budget_tool_note", "BUDGET_HINT",
           "BUDGET_CODES", "BUDGET_PHRASES"]
