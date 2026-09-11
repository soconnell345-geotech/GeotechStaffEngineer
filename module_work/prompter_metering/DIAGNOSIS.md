# Why Funhouse's usage counter does not see our app's Prompter usage

Read-only investigation. Nothing in either repo was edited, staged or committed.
Scratch files: `C:\Users\socon\.claude\jobs\69be0e95\tmp\wrap_probe.py`, `config_probe.py`.

Every claim below is tagged **[EXECUTED]** (I ran it on this machine) or **[STATIC]**
(established by reading code; the cluster was not reachable).

---

## 0. Headline

The metering wrapper is **not** the problem — it works, and our app's calls do reach it.
The problem is one line further down: `FunhouseLogger.meter_log()` **discards** the record
because the launched app process has no meter storage configured.

`meter_log` is not an HTTP call to a service. It is an append to a **local SQLite file**
whose path comes from the in-process `FunhouseConfig` singleton. The org's setup notebook
puts that path into the config **in memory** (`fh_config.set(...)`), and our app runs in a
**separate OS process** that never inherits it. So every Prompter call the app makes is
priced, logged and then dropped on the floor — silently, because the one diagnostic line
is emitted at INFO and the default stdout level is WARNING.

**The gap is TOTAL** for app traffic: main-agent turns, sub-agent turns, tool-call turns,
streaming and non-streaming alike. The only Prompter usage that *is* metered is what the
owner does **in the notebook kernel itself**, which is why the counter reads "far too
little" rather than exactly zero.

**We can fix it entirely on our side** — one block in `webapp/databricks_launcher.py`.
No SDK change and no admin action is required to make the tokens land. (One *residual*
attribution defect is genuinely an SDK bug; §5 explains why it does not block the fix.)

---

## 1. The call path, established precisely

| Step | Where | What happens |
|---|---|---|
| 1 | `webapp/databricks_launcher.py:695` | `_popen([python_exe, script_path], env=env)` — the Streamlit app is a **fresh OS process**, env built by `_child_env(os.environ, ...)` at `:374-416`. |
| 2 | `webapp/databricks_launcher.py:253-281` | In that child, `_register_prompter()` builds `PrompterAPI(backend="prompter", username=…, password=…, base_url=…, chat_model=MODEL)` — **no `config=`, no `config_path=`**. |
| 3 | `funhouse/services/prompter/prompter_api.py:485-491` | `PrompterAPI.__init__` therefore calls `FunhouseConfig.get_instance()` with **no path**. |
| 4 | `funhouse/config/funhouse_config.py:479-489` | `_load_config(None)` merges **only** `DEFAULTS` + `FUNHOUSE_*` environment variables. No file is read. |
| 5 | `prompter_api.py:530` | `self.logger = FunhouseLogger.get_instance()` → `funhouse_logger.py:27` `self._config = FunhouseConfig.get_instance()` — the same DEFAULTS-only singleton. |
| 6 | `prompter_api.py:712` | `wrap_all_openai_methods(self.client, self.logger)` wraps the OpenAI client. |
| 7 | `funhouse_agent/deep/databricks_bridge.py:397` | Our `PrompterChatModel._create_with_param_fallback` does `create = self.prompter.client.chat.completions.create` → **the wrapped method**. |
| 8 | `prompter_api.py:118-124` | The wrapper sees `response.usage`, flattens it and calls `logger.meter_log(service="Prompter", operation="ChatCompletion", metrics=…)`. **This fires.** |
| 9 | `funhouse/logging/funhouse_logger.py:131-136` | `if not should_write_sqlite_for_user(self._config, uid): self.info("No meter SQLite path configured; stdout only: …"); return` ← **the record dies here.** |

Step 8 was verified by execution; step 9 was verified by execution. See §2 and §3.

Every LLM call the app makes funnels through step 7. Confirmed by grep over the app:
`funhouse_agent/deep/databricks_bridge.py:397` (deep agent, all turns and sub-agents),
`funhouse_agent/engine.py:255,293` (v1 engine), `funhouse_agent/agent.py:458`.
There is no `langchain_openai.ChatOpenAI`, no async OpenAI client, and the SDK's own
`funhouse/utils/langchain_prompter_chat.PrompterChatModel` is **not** used by the app.

---

## 2. Lead 1 and Lead 3 — DISPROVEN by execution

Your strongest structural hypothesis (LangChain building its own client, or grabbing a
stale bound-method reference, so the wrapping never applies) is **not what is happening**.

**[EXECUTED]** `wrap_probe.py` copies `wrap_openai_method` / `wrap_all_openai_methods`
verbatim from `prompter_api.py:94-143`, applies them to a real `openai.OpenAI` client
(openai 2.46.0, Python 3.14.5, `httpx.MockTransport` returning a canned completion):

```
=== AFTER WRAP ===
create._is_wrapped: True
has __wrapped__: True
identity changed: True

=== NON-STREAMING CALL through wrapped client ===
response.usage: CompletionUsage(completion_tokens=3, prompt_tokens=11, total_tokens=14, …)
METER_CALLS: [('Prompter', 'ChatCompletion',
   {'completion_tokens': 3, 'prompt_tokens': 11, 'total_tokens': 14,
    'completion_tokens_details': None, 'prompt_tokens_details': None,
    'model': 'funhouse-gpt-high', 'object': 'ChatCompletion'})]
```

So: the recursive wrapper **does** reach `client.chat.completions.create`
(`wrap_all_openai_methods` descends `client` → `chat` → `completions` because those are
non-callable objects with a `__dict__`, then `setattr`s the wrapped `create` onto the
`Completions` instance), and a non-streaming call **does** produce a `meter_log`.

Our `_create_with_param_fallback` re-resolves the attribute on every call
(`databricks_bridge.py:397`), so it always picks up the wrapped version — it never
caches a pre-wrap reference. LangChain is not in the way: our `PrompterChatModel` is a
thin `BaseChatModel` that calls the client directly.

**Conclusion: the wrapper is healthy. Rule out leads 1 and 3.**

---

## 3. THE CAUSE — sufficient on its own, TOTAL gap

### 3a. The meter's storage config never crosses the process boundary

The org setup notebook every user `%run`s configures the meter **in memory only**:

`funhouse-sdk-python/scripts/setup/python_prod_funhousesdk.py:123-137`
```python
DATA_ROOT = f"/Workspace/Users/{user}"                 # user from dbutils…userName()
METER_SQLITE_DIR = os.path.join(DATA_ROOT, ".funhouse_meter")
fh_config.set("budget.storage_backend", "sqlite")
fh_config.set("budget.sqlite_directory", METER_SQLITE_DIR)
fh_config.set("budget.sqlite_filename", "funhouse_meter.sqlite")
fh_config.set("budget.cost_summary_enabled", True)
fh_config.set("activity.enabled", True)
```

`FunhouseConfig` is a **per-process singleton** (`funhouse_config.py:406-417`). These
values live in the kernel's heap. They are not written to a file and not exported to the
environment. Our app is `subprocess.Popen`'d (`databricks_launcher.py:695`), so it starts
with `DEFAULTS` + `FUNHOUSE_*` env only — and `DEFAULTS["budget"]` contains
`storage_backend`, `monthly_budget`, `sqlite_filename` and the pricing table but **no
`sqlite_path` and no `sqlite_directory`** (`funhouse_config.py:139-145`).

**[EXECUTED]** fresh process, no funhouse config file, no `FUNHOUSE_*` env:

```
'budget.storage_backend'            -> 'sqlite'
'budget.sqlite_path'                -> None
'budget.sqlite_directory'           -> None
'budget.sqlite_filename'            -> 'funhouse_meter.sqlite'
'budget.sqlite_path_template'       -> None
'budget.sqlite_directory_template'  -> None
'budget.sqlite_path_per_user'       -> None
'budget.sqlite_directory_per_user'  -> None

resolve_sqlite_database_path(cfg)          -> None
resolve_sqlite_database_path_for_user(...) -> None
should_write_sqlite_for_user(cfg, uid)     -> False
get_budget_sqlite_store(cfg, uid)          -> None
```

`should_write_sqlite_for_user` (`funhouse/admin/budget/budget_sqlite_store.py:710-717`)
returns the OR of the two resolvers; both need `budget.sqlite_path` or
`budget.sqlite_directory` (or a per-user/template variant) — all `None`.

Therefore `funhouse_logger.py:131` short-circuits and `meter_log` **returns without
writing anything**. Fail-open by design (`:139-141`, `:161-162`), so nothing ever raises.

### 3b. …and it fails silently

The single diagnostic is `self.info("No meter SQLite path configured; stdout only: …")`
(`funhouse_logger.py:133-135`). `FunhouseLogger._setup_stdout_logging` attaches one
handler at `logging.stdout_level`, whose default is `WARNING`
(`funhouse_logger.py:83-96`; `DEFAULTS["logging"] == {'stdout_level': 'WARNING'}`).

**[EXECUTED]**
```
--- calling lg.info() now ---
--- calling lg.warning() now ---
2026-09-10 … - funhouse - WARNING - WARNING-LEVEL CONTROL LINE
```
The INFO line is not emitted. There is no trace of the dropped meter record anywhere in
the app log — which is exactly why this has gone unnoticed.

### 3c. Why the notebook's own usage *does* show up

Same SDK, same wrapper — but the notebook kernel's `FunhouseConfig` singleton has the
budget paths, so `meter_log` writes. That asymmetry is the whole symptom: the counter is
not empty, it is missing everything the app did.

---

## 4. Lead 2 — streaming: a dead SDK branch, correctly routed around by us

**[EXECUTED]** Calling the **wrapped** client with `stream=True`:

```
=== STREAMING CALL through wrapped client ===
RAISED: TypeError Completions.create() got an unexpected keyword argument 'collect_usage'
last request body keys: ['messages', 'model']
collect_usage in body: False
```

`prompter_api.py:113-114` injects `kwargs["collect_usage"] = True` whenever `stream` is
truthy. The OpenAI Python SDK has no such parameter, so the call raises before any HTTP
request is made. **`collect_usage` never survives into the request, and the SDK's
streaming metering branch can never fire for anybody.** That is an SDK defect worth
reporting upstream, but it is not our gap.

Our app already documented and routed around this
(`databricks_bridge.py:223-245` class docstring, "Landmine 1"):

* `_streaming_create()` (`:418-452`) builds a **clean, unwrapped** `OpenAI` client reusing
  `prompter.http_client` (so NTLM identity still rides along), `base_url`, `api_key`,
  `max_retries` — mirroring `PrompterAPI`'s own construction at `prompter_api.py:707-718`.
* `_stream()` sends `stream_options={"include_usage": True}` (`:514`) to get the final
  usage chunk.
* `_meter_streamed_usage()` (`:616-645`) re-implements the wrapper's metering call:
  `logger.meter_log(service="Prompter", operation=obj_type, metrics=<flattened usage>)`.

So with `GEOTECH_PROMPTER_STREAMING=1` the tokens still reach `meter_log` — and still die
at `funhouse_logger.py:131`. **Streaming is not a separate cause; it lands in the same
hole.** Our compensation code is correct and starts working the moment §3 is fixed.

One label defect in that path: `operation` is `type(raw_chunk).__name__` →
`"ChatCompletionChunk"` (`:546`, `:618`), where the non-streaming wrapper writes
`"ChatCompletion"`. Cost is unaffected (pricing keys on Service + Model, not Action —
`funhouse_budget.py:306-345`), but `meter_event.action` / `meter_usage_fact.action` will
split our streamed usage into a second bucket in any action-grouped report. One-word fix:
pass `operation="ChatCompletion"`.

---

## 5. Lead 4 and Lead 5 — checked, and what they actually are

**Lead 4 — transport.** `funhouse/services/prompter/prompter_usage_tracker.py` is a
**read-side** client. It GETs the server's `/api/supported-models/usage/models` and
`/azure-costs` endpoints (`:78-103`, `:220-238`) and returns a DataFrame. It **never
writes** anything and is not the transport for our usage. The write path is
`meter_log` → `BudgetSqliteStore.insert_sqlite_meter_facts`
(`budget_sqlite_store.py:370-460`) → local SQLite (`meter_event` + priced
`meter_usage_fact` rows). Every step is fail-open; nothing surfaces an error.

This gives you a **free cross-check**: because `PrompterUsageTracker` reads what the
*server* recorded, and our app authenticates with the notebook's NTLM identity
(`GEOTECH_FH_USERNAME` threaded at `databricks_launcher.py:407`), the app's tokens
**should** appear there. If the server-side tracker shows them but the local SQLite meter
does not, that pins the gap to the client-side meter exactly as diagnosed.

**Lead 5 — `is_logging_active`.** Not a contributor.
`prompter_api.py:109-127` sets the ContextVar and resets it in a `finally`, so it cannot
leak across turns; a ContextVar is per-context/per-thread, and LangGraph's
`copy_context()` usage does not defeat the reset. In our path no wrapped method calls
another wrapped method: only `chat.completions.create` is invoked, and everything it
calls internally (`_post`, `_client`, …) is underscore-prefixed and skipped by
`wrap_all_openai_methods:135-136`. For the record, **if** nesting did occur, the **inner**
record would be lost — the outer holds the token, the inner returns early at `:109-110`.

**Bonus finding, separate from metering: budget enforcement is bypassed entirely.**
`@check_budget()` decorates `PrompterAPI.chat` (`prompter_api.py:744`), `get_embedding`
(`:1087`) and six other methods. Our app calls `client.chat.completions.create` directly
and **never calls `PrompterAPI.chat`**, so the monthly-cap check never runs on app
traffic. Even in the notebook the cap could not fire on app spend, because the spend rows
are never written. Worth telling whoever owns the budget policy.

---

## 6. What would make the numbers WRONG rather than missing

Ranked by how much they matter once §3 is fixed.

1. **`unknown_user` attribution (high).** `meter_log` stamps `user_name` from
   `self.current_user_name` → `FunhouseLogger.session_service.user_name`
   (`funhouse_logger.py:52-56`) → the `CURRENT_USER_NAME` env var
   (`funhouse_session_service.py:120-122`). In a Databricks **subprocess**,
   `import databricks.sdk.runtime` succeeds, so the Databricks branch runs
   (`:36-55`), but `_get_user_name_from_dbutils` (`:78-87`) needs the **notebook** Py4J
   context, which a bare subprocess does not have — it swallows the failure and returns
   `"unknown_user"`, then `os.environ["CURRENT_USER_NAME"] = "unknown_user"` by **direct
   assignment** at `:48`, clobbering anything we inherited. (Only if the `dbutils`
   *import* itself raises `ImportError` does it fall to `_set_default_env_vars()` at
   `:64-76`, which uses `setdefault` and would respect our value.) **[STATIC]** — which
   branch the cluster takes must be confirmed by the check in §8.
   Related SDK bug: `FunhouseConfig._fetch_session_variables` guards with
   `session_value != f"unknown_{config_key}"` → it compares against `"unknown_user_name"`
   while the service returns `"unknown_user"`, so the guard never fires and
   `session.user_name` is set to `"unknown_user"` (`funhouse_config.py:499-513`).

2. **Model name → cost 0.00 (high, and easy to miss).** `process_costs` matches the model
   with **exact equality** against `budget.pricing.Prompter.models`
   (`funhouse_budget.py:332`, `df['Model'] == model`), and the
   `default_cost_per_million_tokens_input/output` fallback applies **only** to rows whose
   model is literally `'NA'` (`:370-385`). `_infer_model` extracts whatever the metrics
   say (`budget_sqlite_store.py:364-368`), so it is never `'NA'` when a model is present.
   **[EXECUTED]** the pricing table's model keys are:
   `gpt-3.5-turbo, text-embedding-ada-002, embedding-3-small, text-embedding-3-{small,large},
   gpt-4o-mini-2024-07-18, gpt-4o-2024-{05-13,08-06,11-20}, gpt-4.1-2025-04-14,
   gpt-5.4-2026-03-05, gpt-4.1-mini-2025-04-14, gpt-5.1-2025-11-13, gpt-5-mini-2025-08-07,
   gpt-4-0613, gpt-4, o1-mini-…, o1-preview-…, o1-2024-12-17, o3-mini(-2025-01-31),
   gpt-4o-mini, grok-*, funhouse-gpt-image, gpt-image-1(.5)`.
   There is **no `funhouse-gpt-high`, `funhouse-gpt-medium` or
   `funhouse-gpt-4o-mini-20240718`**. If the proxy echoes a `funhouse-*` alias in
   `response.model`, every row prices at **Cost = 0.0** — token counts right, spend zero.
   Check what `model=` actually says (§8 step 3); if it is an alias, the fix is to add the
   alias to `budget.pricing.Prompter.models` (we can set that via env too, though a
   nested dict via `FUNHOUSE_*` is awkward — an admin/config change is cleaner).

3. **Under-count on client retries (low).** The OpenAI client's own `max_retries`
   (PrompterAPI default 1, `prompter_api.py:707,716`) retries *inside* one wrapped
   invocation, so N billed HTTP attempts produce one meter row. Same shape in
   `_create_with_param_fallback` (`databricks_bridge.py:394-402`) — though there the first
   attempt failed before returning usage, so dropping it is correct.

4. **Streamed-call operation label** = `"ChatCompletionChunk"` (see §4).

5. **`source_job_id` (cosmetic).** `_resolve_meter_source_job_id`
   (`funhouse_logger.py:100-114`) reads `DATABRICKS_JOB_ID` / `DATABRICKS_JOB_RUN_ID` from
   the environment; the child inherits whatever the driver had, so app rows can be
   labelled as belonging to a job run.

6. **No double counting found.** The two metering call sites in our path are mutually
   exclusive: `_meter_streamed_usage` fires only on the unwrapped streaming client;
   the `_generate` fallback (`databricks_bridge.py:455-484`) meters only through the
   wrapper. The SDK's `chat()` also has explicit `meter_log` calls at
   `prompter_api.py:899` / `:991` — but those are the **Grok** and **Azure-OpenAI**
   backends, which do not touch the wrapped client, so they do not double-count either.
   Whisper (`:1653`, `:1685`) uses the raw `http_client`, likewise single-counted.

---

## 7. The fix — ours, entirely

`FunhouseConfig._load_env_vars` (`funhouse_config.py:518-524`) maps
`FUNHOUSE_<A>__<B>` → `a.b`. So threading the notebook's live meter config into the child
environment turns the meter on with no SDK change.

**[EXECUTED]** with `FUNHOUSE_BUDGET__SQLITE_DIRECTORY` and `CURRENT_USER_NAME` set in a
fresh process:

```
budget.sqlite_directory -> C:/…/tmp/meterdir
session.user_name       -> oconnells@state.gov
should_write            -> True
resolved path           -> C:/…/tmp/meterdir\funhouse_meter.sqlite
```

### Where to put it

`webapp/databricks_launcher.py`, function `_child_env` (`:374-416`) — right beside the
existing `GEOTECH_FH_*` credential threading, which already reads the notebook's live
objects at launch time. Sketch (NOT applied — this investigation was read-only):

```python
# Metering obligation: the notebook's meter config lives only in the kernel's
# FunhouseConfig singleton (scripts/setup/python_prod_funhousesdk.py sets it with
# fh_config.set(...)), so a Popen'd child sees DEFAULTS only and meter_log silently
# drops every record. FunhouseConfig._load_env_vars maps FUNHOUSE_A__B -> a.b.
try:
    from funhouse.config.funhouse_config import FunhouseConfig
    _c = FunhouseConfig.get_instance()
    for cfg_key, env_key in (
        ("budget.storage_backend",  "FUNHOUSE_BUDGET__STORAGE_BACKEND"),
        ("budget.sqlite_directory", "FUNHOUSE_BUDGET__SQLITE_DIRECTORY"),
        ("budget.sqlite_path",      "FUNHOUSE_BUDGET__SQLITE_PATH"),
        ("budget.sqlite_filename",  "FUNHOUSE_BUDGET__SQLITE_FILENAME"),
        ("budget.monthly_budget",   "FUNHOUSE_BUDGET__MONTHLY_BUDGET"),
        ("session.user_name",       "FUNHOUSE_SESSION__USER_NAME"),
    ):
        v = _c.get(cfg_key)
        if v not in (None, ""):
            env.setdefault(env_key, str(v))
    _u = str(_c.get("session.user_name") or "").strip()
    if _u:
        env.setdefault("CURRENT_USER_NAME", _u)
except Exception:
    pass          # never block a launch on metering setup
```

### Attribution: two layers, because of §6.1

* **Layer 1 (does the work).** The notebook's `budget.sqlite_directory` is
  `/Workspace/Users/<user email>/.funhouse_meter` — **the identity is in the path**. Even
  if the `user_name` column says `unknown_user`, the rows land in the right person's meter
  file, which is what the admin cross-user report keys on
  (`discover_meter_sqlite_users`, `budget_sqlite_store.py:720-736`; the `sqlite_path`
  column in `examples_python/22. Admin/00a. Funhouse Budgets/2. Admin cross-user SQLite
  reporting.py:100`). This alone closes the compliance gap.
* **Layer 2 (makes the column right).** Passing `CURRENT_USER_NAME` may be clobbered by
  `FunhouseSessionService` (§6.1). If §8's check shows `unknown_user`, the deterministic
  app-side patch is to pin the logger's cached session service once, in
  `_register_prompter()` right after the `PrompterAPI` is built:

  ```python
  from funhouse.logging import FunhouseLogger
  _lg = FunhouseLogger.get_instance()
  if _lg.current_user_name in ("unknown_user", "") and _os.environ.get("GEOTECH_FH_USER"):
      _svc = _lg.session_service                      # force the cached instance
      _svc._current_user_name = _os.environ["GEOTECH_FH_USER"]
  ```

  That touches a private attribute, so flag it as a workaround and file the SDK bug:
  `FunhouseSessionService` should honour an explicit `CURRENT_USER_NAME` /
  `FUNHOUSE_SESSION__USER_NAME` instead of overwriting it with `unknown_user`
  (`funhouse_session_service.py:44-55`), and the `unknown_{config_key}` guard in
  `funhouse_config.py:503-513` should compare against `unknown_user` / `unknown_cluster`.

**Needs SDK/admin, not us:** (a) the `collect_usage` streaming bug (§4) — but we already
route around it; (b) `funhouse-*` model aliases missing from the pricing table (§6.2), if
that turns out to be what the proxy echoes.

---

## 8. Verification the owner can run on the cluster

### Step 1 — confirm the diagnosis (≈30 s, no app launch needed)

In the notebook, after the usual `%run` setup:

```python
CHECK = r'''
from funhouse.config import FunhouseConfig
from funhouse.logging import FunhouseLogger
from funhouse.admin.budget.budget_sqlite_store import (
    should_write_sqlite_for_user, resolve_sqlite_database_path_for_user)
c  = FunhouseConfig.get_instance()
lg = FunhouseLogger.get_instance()
u  = lg.current_user_name
print("user          :", u)
print("sqlite_dir    :", c.get("budget.sqlite_directory"))
print("resolved path :", resolve_sqlite_database_path_for_user(c, u))
print("WILL METER    :", should_write_sqlite_for_user(c, u))
'''
print("=== NOTEBOOK KERNEL ===");  exec(CHECK)

import subprocess, sys, os
print("=== CHILD PROCESS (what the app sees) ===")
r = subprocess.run([sys.executable, "-c", CHECK], env=os.environ.copy(),
                   capture_output=True, text=True)
print(r.stdout or r.stderr)
```

**Expected if the diagnosis is right:** the kernel prints a real user, a real
`/Workspace/Users/…/.funhouse_meter` directory and `WILL METER: True`; the child prints
`sqlite_dir: None`, `resolved path: None`, **`WILL METER: False`** — and very likely
`user: unknown_user`, which also confirms or refutes §6.1 in the same breath.

That one contrast is the entire diagnosis.

### Step 2 — prove the meter is silent for real app traffic

Launch the app as usual, ask it **one** question that definitely calls the model, then in
the notebook:

```python
import sqlite3, pandas as pd
p = resolve_sqlite_database_path_for_user(c, lg.current_user_name)
con = sqlite3.connect(p)
print(pd.read_sql("SELECT user_name, service, action, COUNT(*) n, MAX(event_ts) last_ts "
                  "FROM meter_event GROUP BY 1,2,3 ORDER BY last_ts DESC", con))
```

**Expected now:** no row whose `last_ts` matches the app turn (only notebook-originated
rows). **Expected after the fix:** a fresh `Prompter / ChatCompletion` row at that
timestamp.

### Step 3 — after the fix, confirm the money is right too

```python
print(pd.read_sql("SELECT user_name, model, metric_key, quantity, cost, rate, action "
                  "FROM meter_usage_fact ORDER BY id DESC LIMIT 10", con))
```

Look for three things:
* `user_name` is the owner, not `unknown_user` → §6.1 clear.
* `model` matches a key in `budget.pricing.Prompter.models` and **`cost > 0`**. If `cost`
  is `0.0` with a non-null model, that is §6.2 — note the exact `model` string and add it
  to the pricing table.
* `action` is `ChatCompletion` (streamed turns will read `ChatCompletionChunk` until the
  one-word fix in §4 lands).

### Optional cross-check — the server's own numbers

```python
from funhouse.services.prompter.prompter_usage_tracker import PrompterUsageTracker
PrompterUsageTracker(config=c).fetch_this_month(granularity="daily")
```

This reads the **service side**, so it should already include the app's tokens even today.
Server-side present + local meter absent = the diagnosis, confirmed from both ends.

---

## 9. Summary table

| # | Cause | Sufficient alone? | Scope | Who fixes |
|---|---|---|---|---|
| A | App subprocess has no `budget.sqlite_*` config → `meter_log` early-returns (`funhouse_logger.py:131-136`) | **Yes** | **TOTAL** — every app call | **Us** (`databricks_launcher.py` `_child_env`) |
| B | `CURRENT_USER_NAME` → `unknown_user` in the child (`funhouse_session_service.py:44-55,78-87`) | Yes, for *per-user* attribution | All app rows | Us (mitigate via per-user path + pinned session service); SDK for the clean cure |
| C | `collect_usage` injection breaks streaming through the wrapper (`prompter_api.py:113-114`) | Would be, for streamed calls | Streaming only | Already routed around by us; SDK should fix |
| D | `funhouse-*` model alias absent from `budget.pricing.Prompter.models` → `cost = 0.0` | Makes spend wrong, not missing | All rows, if the alias is echoed | Admin/config |
| E | `@check_budget()` never runs (app bypasses `PrompterAPI.chat`) | n/a — enforcement, not metering | All app calls | Us, if enforcement is wanted |

**Not causes:** the wrapper's reach (executed, works), LangChain owning its own client
(we don't use `langchain_openai`), an async client (none), `is_logging_active` leaking
(reset in `finally`, no nesting in our path), `PrompterUsageTracker` (read-only).
