"""Per-user AI budget status for the web app sidebar (Funhouse SDK backed).

The Funhouse SDK meters each user's AI spend (``FunhouseBudget``, SQLite
``meter_usage_fact``) against a monthly cap from the Funhouse config key
``budget.monthly_budget`` (default $50). This module reads both so the app
can show "AI budget: $12.34 of $50.00 used this month" next to the token
line — and warn when the budget is nearly exhausted (a metered call past the
cap raises ``BudgetExceededError`` mid-turn, which reads as a crash; see
``core.friendly_turn_error`` for the matching turn-error advice).

Design rules (house style, mirrors :mod:`aec_document_review.sharepoint_store`):

* **Best-effort, never raises** — ``get_budget_status()`` returns ``None`` on
  ANY failure (SDK absent, config absent, backend error) and the sidebar just
  omits the line.
* **Lazy SDK imports** — importable and testable without the Funhouse SDK.
* **Expensive read** — ``FunhouseBudget.get_current_spend()`` scans the
  month's usage table; call it at most once per session. The app caches the
  returned status in ``st.session_state`` and offers a small refresh button
  (see the sidebar wiring in ``aec_document_review/app.py``).
* **Streamlit-free** — no ``st`` usage here.
"""

from __future__ import annotations

from typing import Optional


def available() -> bool:
    """True when the Funhouse budget SDK is importable (off-Funhouse installs
    simply have no budget line)."""
    try:
        import funhouse.admin.budget  # noqa: F401
        return True
    except Exception:
        return False


def get_budget_status() -> Optional[dict]:
    """Current-month AI spend vs the monthly cap: ``{"spent", "cap"}`` (USD).

    Returns ``None`` on ANY failure — SDK not installed, config unavailable,
    metering backend error — so callers can simply skip the display.

    NOTE: ``get_current_spend()`` is an expensive full-month rollup; cache the
    result (the app keeps it in ``st.session_state``) rather than calling this
    every rerun.
    """
    try:
        from funhouse.admin.budget import FunhouseBudget
        from funhouse.config import FunhouseConfig
        cfg = FunhouseConfig.get_instance()
        budget = FunhouseBudget(config=cfg)
        spent = float(budget.get_current_spend())
        cap = float(cfg.get("budget.monthly_budget", default=50.0))
        return {"spent": spent, "cap": cap}
    except Exception:
        return None


def format_line(status: dict) -> str:
    """One caption line for the sidebar, e.g.
    ``AI budget: $12.34 of $50.00 used this month``."""
    return (f"AI budget: ${status['spent']:,.2f} of ${status['cap']:,.2f} "
            "used this month")


# ---------------------------------------------------------------------------
# An estimate without the SDK (BACKLOG TA-05), behind GEOTECH_SPEND_ESTIMATE
# ---------------------------------------------------------------------------
#
# A tester on a host with no Funhouse SDK (Tiny Apps: a key with a monthly
# budget) sees nothing of what their turns cost. With the setting on, the
# sidebar shows the caller's own spend this calendar month, ESTIMATED from
# the token counts their own conversations' records hold (each turn's
# ``turn_end`` in ``activity.jsonl``) and priced by two settings. Only the
# caller's own conversation folder is read: never another tester's, and not
# the shared ``.trash`` folder (so a conversation the caller deleted no
# longer counts; the line says it is an estimate).

#: On (1/true/yes/on) = show the estimate where the SDK is absent. Off unless
#: set.
SPEND_ESTIMATE_ENV = "GEOTECH_SPEND_ESTIMATE"
#: US dollars per million input tokens, and per million output tokens.
SPEND_INPUT_PRICE_ENV = "GEOTECH_SPEND_INPUT_USD_PER_MTOK"
SPEND_OUTPUT_PRICE_ENV = "GEOTECH_SPEND_OUTPUT_USD_PER_MTOK"
#: The monthly budget in US dollars (optional).
SPEND_BUDGET_ENV = "GEOTECH_SPEND_MONTHLY_BUDGET_USD"


def estimate_on() -> bool:
    """Whether ``GEOTECH_SPEND_ESTIMATE`` is on (read each time)."""
    import os
    return (os.environ.get(SPEND_ESTIMATE_ENV) or "").strip().lower() in (
        "1", "true", "yes", "on")


def _price(name: str) -> Optional[float]:
    import os
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v >= 0 else None


def _month_start(now: Optional[float] = None) -> float:
    """The first instant of the current calendar month, UTC."""
    import calendar
    import time
    t = time.gmtime(time.time() if now is None else now)
    return float(calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0)))


def month_tokens(root: Optional[str], now: Optional[float] = None) -> dict:
    """Input and output tokens over every turn this calendar month (UTC) in
    the conversations under ``root`` (the caller's own conversation root,
    ``<root>/conversations/<id>/activity.jsonl``). A turn whose record has
    no input/output split (written before the split was kept) counts its
    total as input. Never raises."""
    import json
    import os
    out = {"input_tokens": 0, "output_tokens": 0, "turns": 0}
    if not root:
        return out
    start = _month_start(now)
    base = os.path.join(root, "conversations")
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for name in names:
        if name.startswith("."):
            continue
        path = os.path.join(base, name, "activity.jsonl")
        try:
            if os.path.getmtime(path) < start:
                continue
            fh = open(path, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                if '"turn_end"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("event") != "turn_end" or \
                        float(rec.get("ts") or 0) < start:
                    continue
                usage = rec.get("usage") if isinstance(rec.get("usage"),
                                                       dict) else None
                try:
                    if usage and ("input_tokens" in usage
                                  or "output_tokens" in usage):
                        out["input_tokens"] += int(usage.get("input_tokens")
                                                   or 0)
                        out["output_tokens"] += int(
                            usage.get("output_tokens") or 0)
                    else:
                        out["input_tokens"] += int(rec.get("turn_tokens")
                                                   or 0)
                except (TypeError, ValueError):
                    continue
                out["turns"] += 1
    return out


def estimate_status(root: Optional[str],
                    now: Optional[float] = None) -> Optional[dict]:
    """The caller's estimated spend this month: ``{"spent", "cap",
    "input_tokens", "output_tokens", "turns", "estimate": True}``.
    ``spent`` is ``None`` when either price setting is missing (the line
    then gives the tokens alone); ``cap`` is ``None`` without a budget.
    ``None`` on any failure."""
    try:
        toks = month_tokens(root, now)
        pin, pout = _price(SPEND_INPUT_PRICE_ENV), _price(SPEND_OUTPUT_PRICE_ENV)
        spent = (None if pin is None or pout is None else
                 (toks["input_tokens"] * pin
                  + toks["output_tokens"] * pout) / 1_000_000)
        return {"spent": spent, "cap": _price(SPEND_BUDGET_ENV),
                "estimate": True, **toks}
    except Exception:
        return None


def format_estimate_line(status: dict) -> str:
    """One caption line, labelled as an estimate, e.g. ``AI spend this
    month (estimate): $3.21 of $50.00, from 1.2M input and 40k output
    tokens in your conversations``."""
    def _n(v: int) -> str:
        return (f"{v / 1e6:.1f}M" if v >= 1_000_000 else
                f"{v / 1e3:.0f}k" if v >= 1_000 else str(v))
    toks = (f"{_n(status.get('input_tokens', 0))} input and "
            f"{_n(status.get('output_tokens', 0))} output tokens in your "
            "conversations")
    spent, cap = status.get("spent"), status.get("cap")
    if spent is None:
        return f"AI use this month (estimate): {toks}; no token prices set"
    money = f"${spent:,.2f}" + (f" of ${cap:,.2f}" if cap else "")
    return f"AI spend this month (estimate): {money}, from {toks}"


__all__ = ["available", "get_budget_status", "format_line", "estimate_on",
           "month_tokens", "estimate_status", "format_estimate_line",
           "SPEND_ESTIMATE_ENV", "SPEND_INPUT_PRICE_ENV",
           "SPEND_OUTPUT_PRICE_ENV", "SPEND_BUDGET_ENV"]
