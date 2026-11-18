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
* :func:`error_line` (the line :func:`tool_error` is built on) masks a
  secret the exception's text names or plainly holds
  (:func:`mask_error_secrets`): a failed download names its signed URL, and
  the model can quote the error into an answer that is saved.
* :func:`budget_exhausted` tells an AI budget or quota that is USED UP
  apart from a model that is merely busy (live smoke wave 2c, D2): asking
  again in a minute fixes a rate limit, never a spent budget, so the two
  are said differently and only the first is retried.
* :func:`rate_limit_named` reads a rate limit the error NAMES, as Foundry's
  model service sends one: a 400 whose ``errorName`` is
  ``RateLimitsExceeded`` or a QoS exception (BACKLOG RI-169, INF-62). Every
  route's busy rule asks it, so there is one copy of the names.

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
        from aec_agent_core._fileio import hide_working_folder
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
    """``"<Type>: <message>"`` on one line, paths scrubbed, secrets masked
    (:func:`mask_error_secrets`, before the cut), at most ``limit``
    characters of message. An AI budget that is used up is said in plain
    words instead (:func:`budget_exhausted`): its raw text is provider JSON
    with a request id in it, which a tester should never be shown. A message
    with no secret in it comes back as it did before the masking, character
    for character; if masking itself failed, only the type's name."""
    stop = budget_exhausted(exc)
    if stop is not None:
        return budget_tool_note(stop)
    msg = mask_error_secrets(" ".join(scrub_paths(str(exc)).split()))
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
    text and no "ask again". It is THEIR budget: on Tiny Apps each tester
    has a key, and so a budget, of their own (per-tester keys, 2026-10-09);
    on Funhouse the budget is per user too."""
    until = f" (until {stop.until})" if stop is not None and stop.until \
        else ""
    return (f"Your AI budget is used up{until}. Your "
            "conversation and its files are kept, but nothing more can run "
            "until the budget is renewed. Tell the app owner.")


# ---------------------------------------------------------------------------
# A rate limit the error NAMES (Foundry; BACKLOG RI-169, INF-62)
# ---------------------------------------------------------------------------
# Foundry's model service sends a rate limit as an HTTP 400 that names it --
# ``errorName: 'LanguageModelService:RateLimitsExceeded'``, or a QoS
# exception -- never as a 429 (the AI FDE's table of wave 2's 221 rate
# limits, 2026-10-27), and the AI FDE says its Anthropic proxy does the
# same. Read off the NAME only: a 400 that names neither is a refused
# request like any other, and no other words in it count.

#: The error names that mean a rate limit (lower case), matched inside the
#: full name ("LanguageModelService:RateLimitsExceeded",
#: "AzurePortalQosException", "HubQosException").
RATE_LIMIT_NAMES = ("ratelimitsexceeded", "qosexception")

#: Where an error body names its error: Foundry's own key, and the
#: ``code`` / ``type`` an OpenAI- or Anthropic-style body uses.
_NAME_KEYS = ("errorName", "error_name", "code", "type")

#: An error's name written out in text: the Foundry SDK's "ErrorName:
#: 'X'", an SDK message carrying the body ("'errorName': 'X'"), a body as
#: JSON text ('"errorName":"X"', escaped or not). Group 1 is the name.
_NAMED_IN_TEXT = re.compile(
    r"error_?name\\?['\"]?\s*[:=]\s*\\?['\"]?([\w:.\-]+)", re.IGNORECASE)


def _is_rate_name(value: Any) -> bool:
    return isinstance(value, str) and any(
        n in value.lower() for n in RATE_LIMIT_NAMES)


def _text_names_rate(text: Any) -> bool:
    return isinstance(text, str) and any(
        _is_rate_name(m.group(1)) for m in _NAMED_IN_TEXT.finditer(text))


def _body_names_rate(body: Any) -> bool:
    """Whether an error body names a rate limit: :data:`_NAME_KEYS` at the
    top or under ``error`` (one level down again for a proxy that wraps
    one), ``error`` itself where it is the name, a ``message`` that writes
    the name out, or a body that is text."""
    if isinstance(body, str):
        return _text_names_rate(body)
    holders = [body]
    for _ in range(2):
        inner = holders[-1].get("error") if isinstance(holders[-1], dict) \
            else None
        if isinstance(inner, str) and _is_rate_name(inner):
            return True
        if not isinstance(inner, dict):
            break
        holders.append(inner)
    for holder in holders:
        if not isinstance(holder, dict):
            continue
        if any(_is_rate_name(holder.get(k)) for k in _NAME_KEYS):
            return True
        if _text_names_rate(holder.get("message")):
            return True
    return False


def _response_body(response: Any) -> Any:
    """A response's parsed JSON, else its text, else ``None`` (an unread
    stream, a fake response): reading it never raises."""
    try:
        parse = getattr(response, "json", None)
        if callable(parse):
            body = parse()
            if isinstance(body, (dict, str)):
                return body
    except Exception:  # noqa: BLE001 - not JSON, or not read
        pass
    try:
        text = getattr(response, "text", None)
        return text if isinstance(text, str) else None
    except Exception:  # noqa: BLE001
        return None


def rate_limit_named(exc) -> bool:
    """Whether ``exc`` (or an error it was raised from) NAMES a rate limit
    (:data:`RATE_LIMIT_NAMES`), whatever its HTTP status -- on Foundry a
    400. Read off the error's ``error_name`` (the Foundry SDK's errors),
    its body (an Anthropic SDK error's whole body; an OpenAI SDK error's
    ``error`` object), its response's JSON, and its message where the name
    is written out there ("ErrorName: '...RateLimitsExceeded'"). A spent
    budget is told apart by :func:`budget_exhausted`, which callers ask
    first. Never raises."""
    try:
        for e in _chain(exc):
            if any(_is_rate_name(getattr(e, attr, None))
                   for attr in ("error_name", "_error_name")):
                return True
            if _body_names_rate(getattr(e, "body", None)):
                return True
            response = getattr(e, "response", None)
            if response is not None and _body_names_rate(
                    _response_body(response)):
                return True
            if _text_names_rate(str(e)):
                return True
    except Exception:  # noqa: BLE001 - classifying never fails the caller
        return False
    return False


# ---------------------------------------------------------------------------
# A model call refused at sign-in (HTTP 401; BACKLOG TA-01)
# ---------------------------------------------------------------------------
#
# Field session 2026-09-09 (an SOE session): the agent's own model call to the
# Funhouse proxy came back as an IIS "401 - Unauthorized" HTML page, the turn's
# error showed that page raw, and it read as a SharePoint permissions failure.
# It was transient: the next turn worked seconds later on the same
# credentials. On Tiny Apps a key that is wrong, expired or not enabled for
# the model is refused the same way.

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_TAGS = re.compile(r"<[^>]+>")
_HTML_PAGE = re.compile(r"<(?:!doctype|html|head|body|title)\b", re.IGNORECASE)
#: Anything shaped like a key or a bearer token, never shown.
_SECRETS = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_*\-]{6,}|\bBearer\s+\S+|\b[A-Za-z0-9_\-]{32,}\b")

# ---------------------------------------------------------------------------
# Masking the text of an error before it is kept (the run record)
# ---------------------------------------------------------------------------
# An exception from a model client or an HTTP library can carry a key, a
# bearer token or a signed URL (``requests`` names the URL it was refused:
# "403 Client Error: Forbidden for url: https://...?...&tempauth=..."), and
# the run record is what a tester downloads and sends on. The rules below
# (:func:`mask_error_secrets`) are general shapes, not one vendor's: a
# secret is masked where it is NAMED (a header, a query parameter, a
# ``name=value`` or ``"name": value`` pair, a URL's password) or where its
# form says it (a private-key block, a JSON web token, an ``sk-`` /
# ``pk-`` / ``rk-`` key, a bearer token). A run of characters with no name
# beside it is kept, however long: a request id, a run id or a hash is
# what someone needs to trace a failure, and an unnamed key cannot be told
# from one. So an error with no secret in it is kept as it was.
# :func:`mask_secrets` keeps its own, wider rule (any run of 32 or more
# token characters) for the places that used it before.

#: A private key in PEM form (a certificate, being public, is kept). A
#: block with no END line (an error cut short) is masked to the text's end:
#: the key is not shown, and the search ends at the first BEGIN (without
#: that, every BEGIN line searched to the end of the text for its END, and
#: a long text with many of them took minutes).
_PEM_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?(?:-----END [A-Z0-9 ]*PRIVATE "
    r"KEY-----|\Z)", re.DOTALL)
#: Every value below stops at a quote, a backslash, a space or a closing
#: bracket, so a secret inside JSON text (a tool's error result, where a
#: quote is written ``\"``) is masked and the JSON stays whole; a full
#: stop at a value's end ends the sentence, and is kept.
#: A password in a URL (``scheme://user:password@host``): the user is kept.
#: The password runs to the LAST ``@`` before the host (one written with an
#: ``@`` in it, unencoded, was shown from that ``@`` on).
_URL_PASSWORD = re.compile(
    r"(?<=://)([^\s/:@'\"<>\\]+):[^\s/'\"<>\\]+@")
#: Between a name and its value: ``=``, ``: ``, ``": "``, ``\": \"``.
_SEP = r"(\\?[\"']?\s*[:=]\s*\\?[\"']?)"
#: An ``Authorization`` header's value, with its scheme.
_AUTH_HEADER = re.compile(
    r"(?i)\b((?:proxy-)?authorization)" + _SEP +
    r"(?!\[removed\])(?:(?:bearer|basic|token|sharedkey|negotiate)\s+)?"
    r"([^\s\"',;})\]\\]+)(?<!\.)")
#: A JSON web token (an access token, a signed link's ``tempauth``).
_JWT = re.compile(
    r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?:\.[A-Za-z0-9_-]*)?")
#: Names that say their value is a secret wherever they stand: a query
#: parameter, a form field, ``name=value``, ``name: value``, a quoted key.
_SECRET_NAMES = (
    r"api[-_ ]?key(?: provided)?|x-api-key|subscription[-_]?key|"
    r"ocp-apim-subscription-key|"
    r"access[-_]?token|refresh[-_]?token|id[-_]?token|auth[-_]?token|"
    r"session[-_]?token|security[-_]?token|client[-_]?secret|"
    r"client[-_]?assertion|password|passwd|tempauth|x-amz-signature|"
    r"x-amz-credential|x-amz-security-token|x-goog-signature|"
    r"x-goog-credential|account[-_]?key|shared[-_]?access[-_]?key|"
    r"shared[-_]?access[-_]?signature|secret[-_]?access[-_]?key|"
    r"secret[-_]?key|private[-_]?key|passphrase|x-auth-token")
#: The name may be the end of a longer one joined by ``_`` or ``-``, as a
#: setting is written (``GRAPH_CLIENT_SECRET=``, ``PROMPTER_API_KEY=``,
#: ``aws_secret_access_key=``), and may carry a per-user suffix after
#: ``--`` (``PROMPTER-API-KEY--<NAME>=``).
_NAMED_SECRET = re.compile(
    r"(?i)(?<![A-Za-z0-9])((?:" + _SECRET_NAMES + r")(?:--[\w.]+)?)" + _SEP +
    r"(?!\[removed\])([^\s&\"',;}\])#<>\\]+)(?<!\.)")
#: Names that are a secret only as a URL's query parameter (``?sig=``,
#: ``&token=``, ``?code=``, ``?key=``): in prose they are ordinary words.
_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:sig|signature|token|secret|code|key)=)"
    r"([^&\s\"'#<>)\\]+)(?<!\.)")
#: ``key``, ``token``, ``secret``, ``signature``, ``sig`` and ``code``
#: outside a URL's query (a JSON body's ``"token": "..."``, a dict's
#: ``'key': '...'``, a form body that starts ``token=...``): masked only
#: when the value has the form of a generated one (:func:`_generated`),
#: because these names also head ordinary values (``"key": "sheet_index"``,
#: ``code=context_length_exceeded``).
_LOOSE_SECRET = re.compile(
    r"(?i)(?<![A-Za-z0-9?&])(key|token|secret|signature|sig|code)" + _SEP +
    r"([^\s&\"',;}\])#<>\\]+)(?<!\.)")
#: A cookie header's pairs (``Cookie: a=1; b=2``, ``Set-Cookie: a=1``).
_COOKIE = re.compile(
    r"(?i)\b((?:set-)?cookie)" + _SEP +
    r"(?!…)[^\s;=\"'\\]+=[^\s;\"'\\]*"
    r"(?:;\s*[^\s;=\"'\\]+=[^\s;\"'\\]*)*")
#: The same names inside a URL that is itself a parameter's value, where
#: ``?``, ``&`` and ``=`` are written ``%3F``, ``%26`` and ``%3D`` (a
#: sign-in's ``redirect_uri=https%3A%2F%2F...%3Faccess_token%3D...``).
#: The value ends at the next encoded ``&`` or ``#``.
_ENCODED_SECRET = re.compile(
    r"(?i)((?:(?:(?<![A-Za-z0-9])|(?<=%[0-9a-f]{2}))"
    r"(?:" + _SECRET_NAMES + r")"
    r"|%(?:3F|26)(?:sig|signature|token|secret|code|key))"
    r"%3D)((?:(?!%2[36])[^\s&\"'#<>)\\])+)")
#: A value that says nothing secret: a missing or already-masked one.
_PLACEHOLDERS = frozenset({"none", "null", "nil", "true", "false", "unset",
                           "undefined", "empty", "missing", "required",
                           "redacted", "hidden", "[removed]"})
#: ``sk-`` / ``pk-`` / ``rk-`` keys (a provider's own refusal quotes one
#: part-starred: "Incorrect API key provided: sk-...****abcd").
#: Not a document's file name that starts the same way (a sketch sheet
#: saved as ``sk-101-foundation-plan.pdf``): a tool's error that names the
#: file it could not open went to the model as ``'….pdf'``. The run is
#: taken whole (possessive), then kept only where a document extension
#: follows it; a key before a full stop that ends a sentence is masked.
_FILE_EXTENSION = (r"\.(?i:pdf|dxf|dwg|docx?|xlsx?|pptx?|csv|txt|md|json|"
                   r"html?|png|jpe?g|tiff?|gif|bmp|zip)(?![A-Za-z0-9])")
_PREFIXED_KEY = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_*\-]{6,}+(?!" + _FILE_EXTENSION + r")")
_BEARER = re.compile(r"\bBearer\s+[^\s\"',;})\]<>\\]+")


def _keeps(value: str) -> bool:
    """Whether a named value is no secret: a placeholder, stars or dots, or
    under four characters."""
    v = value.strip("\"'")
    return (len(v) < 4 or v.lower() in _PLACEHOLDERS
            or set(v) <= set("*.…xX"))


def _generated(value: str) -> bool:
    """Whether a value has the form of a generated secret: 16 characters or
    more with a letter and a digit among them (``sheet_index`` and
    ``context_length_exceeded`` have not)."""
    return (len(value) >= 16 and any(c.isdigit() for c in value)
            and any(c.isalpha() for c in value))


def _mask(text: str) -> str:
    s = _PEM_KEY.sub("…", text)
    s = _URL_PASSWORD.sub(lambda m: f"{m.group(1)}:…@", s)
    s = _AUTH_HEADER.sub(
        lambda m: m.group(0) if _keeps(m.group(3))
        else f"{m.group(1)}{m.group(2)}…", s)
    s = _JWT.sub("…", s)
    s = _NAMED_SECRET.sub(
        lambda m: m.group(0) if _keeps(m.group(3))
        else f"{m.group(1)}{m.group(2)}…", s)
    s = _QUERY_SECRET.sub(
        lambda m: m.group(0) if _keeps(m.group(2)) else f"{m.group(1)}…", s)
    low = s.lower()
    if "%3d" in low:
        s = _ENCODED_SECRET.sub(
            lambda m: m.group(0) if _keeps(m.group(2))
            else f"{m.group(1)}…", s)
    s = _LOOSE_SECRET.sub(
        lambda m: f"{m.group(1)}{m.group(2)}…" if _generated(m.group(3))
        else m.group(0), s)
    if "cookie" in low:
        s = _COOKIE.sub(lambda m: f"{m.group(1)}{m.group(2)}…", s)
    s = _PREFIXED_KEY.sub("…", s)
    return _BEARER.sub("…", s)


def mask_secrets(text: Any) -> str:
    """``text`` with anything shaped like a key or a bearer token replaced
    by an ellipsis (the rule :func:`html_gist` applies). Never raises."""
    try:
        return _SECRETS.sub("…", "" if text is None else str(text))
    except Exception:  # noqa: BLE001
        return ""


def mask_error_secrets(text: Any) -> str:
    """``text`` -- the text of an error that is kept or shown -- with each
    secret replaced by an ellipsis where its NAME or its FORM says it is
    one: a private-key block, a URL's password, an ``Authorization``
    header's value, a JSON web token, the value of a parameter, header or
    field whose name says it is a secret (``api_key``, ``x-api-key``,
    ``access_token``, ``client_secret``, ``password``,
    ``x-amz-signature``, ...; ``sig``, ``token``, ``code``, ``key`` as a
    URL's query parameter), an ``sk-`` / ``pk-`` / ``rk-`` key, and a
    bearer token. Narrower than :func:`mask_secrets`: a run of characters
    with no name beside it (a request id, a run id, a hash, a UUID) is
    kept, however long, so the error can still be traced. Text with none
    of these comes back unchanged, character for character. Never raises
    (``""`` if masking itself failed)."""
    try:
        return _mask("" if text is None else str(text))
    except Exception:  # noqa: BLE001
        return ""


#: What stands for a text that could not be checked for secrets (the
#: masker itself failed): never the text as it came.
TEXT_NOT_CHECKED = ("(not shown: this text could not be checked for "
                    "secrets)")


def masked_error_text(text: Any, limit: Optional[int] = None) -> str:
    """The text of an error, masked WHERE IT IS PRODUCED for a field the
    record does not mask by its name (a nested key, a profile, a check's
    detail): :func:`mask_error_secrets`, then cut to ``limit`` characters
    (masked before the cut, so a secret the cut would split is still
    masked). A text with no secret in it comes back character for
    character. If the masker failed, :data:`TEXT_NOT_CHECKED` stands for a
    text that was not empty -- never the text as it came. Never raises."""
    try:
        raw = "" if text is None else str(text)
    except Exception:  # noqa: BLE001
        return TEXT_NOT_CHECKED
    masked = mask_error_secrets(raw)
    if raw and not masked:
        masked = TEXT_NOT_CHECKED
    return masked[:limit] if limit else masked


def mask_error_result(text: Any) -> Any:
    """A tool's result with :func:`mask_error_secrets` applied when the
    result is the tool's ERROR -- a JSON object with an ``"error"`` string
    at its top, the shape :func:`tool_error` and the tools' own error
    results take -- and unchanged otherwise (a tool's ordinary output is
    kept as the model got it). Never raises."""
    if not isinstance(text, str) or '"error"' not in text \
            or not text.lstrip().startswith("{"):
        return text
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict) and isinstance(data.get("error"), str):
        return mask_error_secrets(text)
    return text


def auth_refused(exc) -> bool:
    """Whether ``exc`` (or an error it was raised from) is the AI service
    refusing a call's sign-in: OpenAI's ``AuthenticationError`` or any HTTP
    401. A spent budget (:func:`budget_exhausted`) is never one. Never
    raises."""
    try:
        if budget_exhausted(exc) is not None:
            return False
        for e in _chain(exc):
            if type(e).__name__ == "AuthenticationError" or _status_of(e) == 401:
                return True
            low = str(e).lower()
            if "error code: 401" in low or "401 - unauthorized" in low:
                return True
    except Exception:  # noqa: BLE001 - classifying never fails the caller
        return False
    return False


def html_gist(text: Any, limit: int = 200) -> str:
    """Error text fit for one line: an HTML error page cut to its title
    (else its text without tags), anything shaped like a key masked, the
    spacing collapsed, at most ``limit`` characters."""
    import html as _html
    s = str(text or "")
    if _HTML_PAGE.search(s):
        m = _TITLE.search(s)
        s = _html.unescape(m.group(1) if m else _TAGS.sub(" ", s))
    s = _SECRETS.sub("…", s)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[:limit] + " …"


def auth_message(exc) -> str:
    """What a user is shown when the AI service refused a call's sign-in:
    the service named, its page cut to its title, what to do."""
    gist = html_gist(scrub_paths(str(exc)))
    said = f": {gist}" if gist else ""
    return ("The AI model service refused this app's sign-in (HTTP 401"
            f"{said}), so this turn stopped. It is not a problem with your "
            "request or your documents, and nothing is lost: files saved or "
            "downloaded this turn are still in the conversation. Ask again in "
            "a moment - a one-off refusal has cleared by the next turn "
            "before. If it keeps happening, tell the app owner: the AI key "
            "this app uses for you may be missing, expired or not enabled for "
            "this model.")


def budget_tool_note(stop: Optional[BudgetStop] = None) -> str:
    """What a TOOL result says when a model call it made hit a spent
    budget: what happened, and that retrying cannot help."""
    until = f" (until {stop.until})" if stop is not None and stop.until \
        else ""
    return (f"the user's AI budget is used up{until}, so this model "
            "call was refused and nothing was read")


#: The hint a tool result carries with :func:`budget_tool_note`.
BUDGET_HINT = ("No model call can run until the budget is renewed, so do NOT "
               "retry this or any other tool that looks at a page. Tell the "
               "user plainly that their AI budget is used up and that "
               "they should tell the app owner.")


__all__ = ["scrub_paths", "error_line", "tool_error", "tool_error_json",
           "BudgetStop", "budget_signal", "budget_exhausted",
           "budget_message", "budget_tool_note", "BUDGET_HINT",
           "BUDGET_CODES", "BUDGET_PHRASES", "auth_refused", "auth_message",
           "html_gist", "mask_secrets", "mask_error_secrets",
           "mask_error_result", "rate_limit_named",
           "RATE_LIMIT_NAMES"]
