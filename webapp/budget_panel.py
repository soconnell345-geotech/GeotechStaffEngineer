"""Per-user AI budget status for the web app sidebar (Funhouse SDK backed).

The Funhouse SDK meters each user's AI spend (``FunhouseBudget``, SQLite
``meter_usage_fact``) against a monthly cap from the Funhouse config key
``budget.monthly_budget`` (default $50). This module reads both so the app
can show "AI budget: $12.34 of $50.00 used this month" next to the token
line — and warn when the budget is nearly exhausted (a metered call past the
cap raises ``BudgetExceededError`` mid-turn, which reads as a crash; see
``core.friendly_turn_error`` for the matching turn-error advice).

Design rules (house style, mirrors :mod:`webapp.sharepoint_store`):

* **Best-effort, never raises** — ``get_budget_status()`` returns ``None`` on
  ANY failure (SDK absent, config absent, backend error) and the sidebar just
  omits the line.
* **Lazy SDK imports** — importable and testable without the Funhouse SDK.
* **Expensive read** — ``FunhouseBudget.get_current_spend()`` scans the
  month's usage table; call it at most once per session. The app caches the
  returned status in ``st.session_state`` and offers a small refresh button
  (see the sidebar wiring in ``webapp/app.py``).
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


__all__ = ["available", "get_budget_status", "format_line"]
