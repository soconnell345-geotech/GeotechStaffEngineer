"""Spend control: prices, the cumulative ledger, wave caps, the per-call meter.

The owner's hard total is :data:`HARD_TOTAL_USD` across every live run.

* :class:`SpendLedger` keeps cumulative spend in ``spend_ledger.json``
  (gitignored), per wave. It is written after EVERY metered model call, so a
  crash loses nothing.
* A wave has a cap. :meth:`SpendLedger.begin_wave` refuses to start a wave
  whose remaining cap would carry cumulative spend past the hard total.
* :class:`SpendMeter` is a LangChain callback attached to the MODEL OBJECT
  (``ChatAnthropic(callbacks=[meter])``), so every call through that model is
  metered whatever run config it was made under -- the primary agent, the
  deepagents sub-agents, the vision side calls (``LangChainVisionEngine``
  calls ``model.invoke`` directly) and the vision probe all share the one
  model object the app was handed. Before each call it checks the wave cap
  and the hard total and RAISES :class:`WaveCapReached` (``raise_error`` is
  True for that one hook), which ends the turn cleanly as a turn error; the
  stop is recorded on the ledger.

Prices are USD per million tokens, from
platform.claude.com/docs/en/about-claude/pricing (read 2026-10-08). The
usage comes from ``AIMessage.usage_metadata`` as ``langchain_anthropic``
1.4.4 builds it: ``input_tokens`` INCLUDES the cache tokens, and
``input_token_details`` carries ``cache_read`` and either ``cache_creation``
or the per-TTL ``ephemeral_5m_input_tokens`` / ``ephemeral_1h_input_tokens``.
A 1-hour cache write is priced at 2x input (Anthropic's published multiplier;
the owner's table lists the 5-minute rate only). Where a response carries no
usage metadata the raw ``llm_output["usage"]`` is used; where neither
exists the call is counted as UNMETERED and reported.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - langchain is always present here
    BaseCallbackHandler = object  # type: ignore[misc,assignment]

#: The owner's hard total across all live runs (USD).
HARD_TOTAL_USD = 80.0

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEDGER = os.path.join(HERE, "spend_ledger.json")

#: Haiku 5.5 is priced in two tiers by prompt size (total input tokens).
HAIKU_TIER_TOKENS = 100_000

#: USD per million tokens. ``cache_write`` is the 5-minute write rate.
PRICES: Dict[str, Any] = {
    "claude-haiku-5-5": {
        "tiers": [
            (HAIKU_TIER_TOKENS, {"input": 0.10, "output": 0.50,
                                 "cache_read": 0.01, "cache_write": 0.125}),
            (None, {"input": 0.50, "output": 2.50,
                    "cache_read": 0.05, "cache_write": 0.625}),
        ]},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00,
                          "cache_read": 0.10, "cache_write": 2.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00,
                        "cache_read": 0.20, "cache_write": 5.00},
}

#: A 1-hour cache write costs this multiple of the input rate.
CACHE_WRITE_1H_MULT = 2.0


class WaveCapReached(RuntimeError):
    """The wave's cap (or the hard total) is spent; no further model call."""


class RefuseToStart(RuntimeError):
    """A wave may not start: it could carry spend past the hard total."""


class UnpricedModel(ValueError):
    """A live model with no price on file (never run unmetered)."""


def rates_for(model_id: str, prompt_tokens: int = 0,
              prices: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """The per-million rates for ``model_id`` at this prompt size."""
    table = PRICES if prices is None else prices
    spec = table.get(model_id)
    if spec is None:
        raise UnpricedModel(f"no price on file for {model_id!r}; known: "
                            f"{sorted(table)}")
    if "tiers" in spec:
        for limit, rates in spec["tiers"]:
            if limit is None or prompt_tokens <= limit:
                return dict(rates)
        return dict(spec["tiers"][-1][1])
    return dict(spec)


def split_usage(usage: Optional[dict]) -> Optional[Dict[str, int]]:
    """``usage_metadata`` (LangChain) -> the billable token classes.

    Returns ``{"prompt", "uncached", "cache_read", "cache_write_5m",
    "cache_write_1h", "output", "cache_split"}`` or ``None`` when there is no
    usage at all. ``cache_split`` is False when the metadata carried no cache
    breakdown (then every input token is priced as uncached input)."""
    if not usage:
        return None
    total_in = int(usage.get("input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)
    det = usage.get("input_token_details") or {}
    split = isinstance(det, dict) and any(
        k in det for k in ("cache_read", "cache_creation",
                           "ephemeral_5m_input_tokens",
                           "ephemeral_1h_input_tokens"))
    det = det if isinstance(det, dict) else {}
    cache_read = int(det.get("cache_read") or 0)
    w5 = int(det.get("ephemeral_5m_input_tokens") or 0)
    w1 = int(det.get("ephemeral_1h_input_tokens") or 0)
    if not (w5 or w1):
        w5 = int(det.get("cache_creation") or 0)
    uncached = max(0, total_in - cache_read - w5 - w1)
    return {"prompt": total_in, "uncached": uncached, "cache_read": cache_read,
            "cache_write_5m": w5, "cache_write_1h": w1, "output": out,
            "cache_split": bool(split)}


def usage_from_raw(raw: Optional[dict]) -> Optional[dict]:
    """The Anthropic API's own ``usage`` block -> LangChain-shaped usage
    (``input_tokens`` there EXCLUDES the cache tokens)."""
    if not isinstance(raw, dict) or not raw:
        return None
    base = int(raw.get("input_tokens") or 0)
    cr = int(raw.get("cache_read_input_tokens") or 0)
    cw = int(raw.get("cache_creation_input_tokens") or 0)
    return {"input_tokens": base + cr + cw,
            "output_tokens": int(raw.get("output_tokens") or 0),
            "input_token_details": {"cache_read": cr, "cache_creation": cw}}


def cost_of(model_id: str, usage: Optional[dict],
            prices: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    """``{"usd", "tokens": split_usage(...), "rates"}`` for one call, or
    ``None`` when there is no usage to price."""
    tok = split_usage(usage)
    if tok is None:
        return None
    r = rates_for(model_id, tok["prompt"], prices)
    usd = (tok["uncached"] * r["input"]
           + tok["cache_read"] * r["cache_read"]
           + tok["cache_write_5m"] * r["cache_write"]
           + tok["cache_write_1h"] * r["input"] * CACHE_WRITE_1H_MULT
           + tok["output"] * r["output"]) / 1_000_000.0
    return {"usd": usd, "tokens": tok, "rates": r}


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def _empty_ledger() -> dict:
    return {"hard_total_usd": HARD_TOTAL_USD, "cumulative_usd": 0.0,
            "waves": {}, "updated": None}


class SpendLedger:
    """Cumulative spend across runs, persisted to ``path`` after each call."""

    def __init__(self, path: str = DEFAULT_LEDGER,
                 hard_total: float = HARD_TOTAL_USD):
        self.path = os.path.abspath(path)
        self.hard_total = float(hard_total)
        self._lock = threading.RLock()
        self.data = self._load()

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "waves" in data:
                data.setdefault("cumulative_usd", 0.0)
                return data
        except (OSError, ValueError):
            pass
        return _empty_ledger()

    def save(self) -> None:
        with self._lock:
            self.data["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.data["hard_total_usd"] = self.hard_total
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=1)
            os.replace(tmp, self.path)

    @property
    def cumulative(self) -> float:
        return float(self.data.get("cumulative_usd") or 0.0)

    def wave(self, name: str) -> dict:
        return self.data["waves"].setdefault(name, {
            "cap_usd": 0.0, "spent_usd": 0.0, "calls": 0,
            "unmetered_calls": 0, "tokens": {}, "stopped_by_cap": False,
            "stop_reason": None, "sessions": [], "scenarios": {}})

    def begin_wave(self, name: str, cap_usd: float, *, model: str,
                   mode: str) -> dict:
        """Open (or resume) wave ``name`` with ``cap_usd``. Raises
        :class:`RefuseToStart` when the wave's remaining cap would carry
        cumulative spend past the hard total."""
        with self._lock:
            w = self.data["waves"].get(name)
            spent = float((w or {}).get("spent_usd") or 0.0)
            remaining = max(0.0, float(cap_usd) - spent)
            if self.cumulative + remaining > self.hard_total + 1e-9:
                raise RefuseToStart(
                    f"wave {name!r}: cumulative ${self.cumulative:.4f} + "
                    f"remaining cap ${remaining:.4f} would pass the hard "
                    f"total ${self.hard_total:.2f}")
            if spent >= float(cap_usd) - 1e-12 and cap_usd > 0 and w:
                raise RefuseToStart(
                    f"wave {name!r} already spent ${spent:.4f} of its "
                    f"${cap_usd:.2f} cap; raise --cap or use a new wave name")
            w = self.wave(name)
            w["cap_usd"] = float(cap_usd)
            w["stopped_by_cap"] = False
            w["stop_reason"] = None
            w["sessions"].append({"started": time.strftime(
                "%Y-%m-%dT%H:%M:%S"), "model": model, "mode": mode})
            self.save()
            return w

    def end_wave(self, name: str) -> None:
        with self._lock:
            w = self.wave(name)
            if w["sessions"]:
                w["sessions"][-1]["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.save()

    def add(self, wave: str, usd: float, tokens: Optional[dict],
            scenario: Optional[str] = None) -> None:
        with self._lock:
            w = self.wave(wave)
            w["spent_usd"] = float(w["spent_usd"]) + float(usd)
            w["calls"] = int(w["calls"]) + 1
            self.data["cumulative_usd"] = self.cumulative + float(usd)
            if tokens:
                for k, v in tokens.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        w["tokens"][k] = int(w["tokens"].get(k, 0)) + int(v)
            if scenario:
                s = w["scenarios"].setdefault(scenario, {"usd": 0.0,
                                                         "calls": 0})
                s["usd"] = float(s["usd"]) + float(usd)
                s["calls"] = int(s["calls"]) + 1
            self.save()

    def add_unmetered(self, wave: str) -> None:
        with self._lock:
            w = self.wave(wave)
            w["unmetered_calls"] = int(w["unmetered_calls"]) + 1
            self.save()

    def record_stop(self, wave: str, reason: str) -> None:
        with self._lock:
            w = self.wave(wave)
            w["stopped_by_cap"] = True
            w["stop_reason"] = reason
            self.save()

    def wave_spent(self, wave: str) -> float:
        return float(self.wave(wave).get("spent_usd") or 0.0)


# ---------------------------------------------------------------------------
# The per-call meter
# ---------------------------------------------------------------------------

class SpendMeter(BaseCallbackHandler):
    """Meter every call of ONE model object and enforce the wave cap.

    Attach it to the model (``ChatAnthropic(callbacks=[meter])``). Only the
    cap check raises (``raise_error``); the accounting never does.
    """

    raise_error = True
    run_inline = True

    def __init__(self, ledger: SpendLedger, wave: str, cap_usd: float,
                 model_id: str, prices: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.ledger = ledger
        self.wave = wave
        self.cap = float(cap_usd)
        self.model_id = model_id
        self.prices = prices
        self.scenario: Optional[str] = None
        self.calls: List[dict] = []
        self.stopped: Optional[str] = None
        self.refused_calls = 0
        self.accounting_errors: List[str] = []
        self._lock = threading.Lock()
        self._starts: Dict[str, float] = {}
        # A live model must have a price before the first call.
        rates_for(model_id, 0, prices)

    # -- the guard ----------------------------------------------------------
    def check(self) -> None:
        """Raise :class:`WaveCapReached` when the cap or the hard total is
        spent. Records the stop on the ledger once."""
        spent = self.ledger.wave_spent(self.wave)
        reason = None
        if spent >= self.cap:
            reason = (f"wave cap reached: ${spent:.4f} of ${self.cap:.2f} "
                      f"spent in wave {self.wave!r}")
        elif self.ledger.cumulative >= self.ledger.hard_total:
            reason = (f"hard total reached: ${self.ledger.cumulative:.4f} of "
                      f"${self.ledger.hard_total:.2f}")
        if reason:
            with self._lock:
                self.refused_calls += 1
                first = self.stopped is None
                self.stopped = self.stopped or reason
            if first:
                self.ledger.record_stop(self.wave, reason)
            raise WaveCapReached(reason)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.check()
        self._starts[str(run_id)] = time.time()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self.check()
        self._starts[str(run_id)] = time.time()

    # -- the accounting -----------------------------------------------------
    def on_llm_end(self, response, *, run_id, **kwargs):
        try:
            usage, source = None, None
            for row in getattr(response, "generations", None) or []:
                for g in row or []:
                    um = getattr(getattr(g, "message", None),
                                 "usage_metadata", None)
                    if um:
                        usage, source = dict(um), "usage_metadata"
                        break
                if usage:
                    break
            if usage is None:
                lo = getattr(response, "llm_output", None) or {}
                usage = usage_from_raw(lo.get("usage"))
                source = "llm_output" if usage else None
            t0 = self._starts.pop(str(run_id), None)
            rec = {"ts": time.time(), "scenario": self.scenario,
                   "model": self.model_id, "usage_source": source,
                   "seconds": round(time.time() - t0, 2) if t0 else None}
            priced = cost_of(self.model_id, usage, self.prices)
            if priced is None:
                rec.update(usd=0.0, unmetered=True)
                self.ledger.add_unmetered(self.wave)
            else:
                rec.update(usd=priced["usd"], tokens=priced["tokens"],
                           unmetered=False)
                self.ledger.add(self.wave, priced["usd"], {
                    k: v for k, v in priced["tokens"].items()
                    if k != "cache_split"}, self.scenario)
            with self._lock:
                self.calls.append(rec)
        except Exception as exc:  # noqa: BLE001 - accounting never fails a call
            self.accounting_errors.append(f"{type(exc).__name__}: {exc}")

    def on_llm_error(self, error, *, run_id, **kwargs):
        try:
            self._starts.pop(str(run_id), None)
            with self._lock:
                self.calls.append({"ts": time.time(), "scenario": self.scenario,
                                   "model": self.model_id, "usd": 0.0,
                                   "error": f"{type(error).__name__}: "
                                            f"{str(error)[:300]}"})
        except Exception:  # noqa: BLE001
            pass

    # -- summaries ----------------------------------------------------------
    def summary(self, calls: Optional[List[dict]] = None) -> dict:
        """Totals over ``calls`` (default: every call so far)."""
        calls = self.calls if calls is None else calls
        tot = {"usd": 0.0, "calls": 0, "errors": 0, "unmetered": 0,
               "prompt": 0, "uncached": 0, "cache_read": 0,
               "cache_write_5m": 0, "cache_write_1h": 0, "output": 0,
               "cache_split_seen": False}
        for c in calls:
            if c.get("error"):
                tot["errors"] += 1
                continue
            tot["calls"] += 1
            tot["usd"] += float(c.get("usd") or 0.0)
            if c.get("unmetered"):
                tot["unmetered"] += 1
            for k, v in (c.get("tokens") or {}).items():
                if k == "cache_split":
                    tot["cache_split_seen"] = tot["cache_split_seen"] or bool(v)
                elif k in tot:
                    tot[k] += int(v)
        tot["usd"] = round(tot["usd"], 6)
        return tot


def spend_line(ledger: SpendLedger, wave: str, cap: float,
               scenario_usd: float, label: str) -> str:
    """The one line printed after every scenario."""
    return (f"[spend] {label}: ${scenario_usd:.4f} | wave {wave!r} "
            f"${ledger.wave_spent(wave):.4f} of ${cap:.2f} | cumulative "
            f"${ledger.cumulative:.4f} of ${ledger.hard_total:.2f}")


__all__ = ["HARD_TOTAL_USD", "PRICES", "SpendLedger", "SpendMeter",
           "WaveCapReached", "RefuseToStart", "UnpricedModel", "rates_for",
           "cost_of", "split_usage", "usage_from_raw", "spend_line",
           "DEFAULT_LEDGER"]
