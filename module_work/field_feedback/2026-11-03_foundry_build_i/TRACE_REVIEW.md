# Trace review: Foundry build i, the first GSE turn with storage on (Rankine Ka, Opus 5.5)

Reviewed 2026-11-03 with no model calls. **This file is PUBLIC.** It holds counts, field names, timings and scores only. It has no resource IDs, dataset or branch names, host names, people's names or token values. The few private details (the model and dataset identifiers, the workspace commits, the conversation's thread ID) are in the gitignored `raw/TRACE_NOTES_private.md`.

**What ran:**
- Build i: app `5.33.0+foundry.20261103.2`, which is master `4779f31` plus a version-only commit; references and planlens are build h's wheels.
- The published app on Foundry, Claude through the Foundry Anthropic proxy (`FoundryChatAnthropic`): main agent Opus 5.5, references helper Sonnet 5.5.
- Conversation storage on a raw-files dataset through the workspace dataset library (INF-38).
- The owner asked one question, then sent one note from the sidebar Feedback box.

**What was read, in full:**
- the AI FDE's `REPORT.md` and `diagnostics_a7_b4.txt`;
- the conversation folder: `activity.jsonl` (17 records), `transcript.jsonl`, `messages.json`, `trace.jsonl`, `meta.json`, `FEEDBACK.md`, `feedback.jsonl`.

**Code read to explain the record:**
- `webapp/engine_config.py`: `FOUNDRY_PROMPT_CACHE_ENV`, `FoundryChatAnthropic` and `sanitize_foundry_anthropic_payload`;
- `webapp/activity_log.py`: `_usage_from_response`;
- `webapp/turn_jobs.py` and `webapp/sharepoint_store.py`: the save after a turn;
- `webapp/foundry_dataset_store.py`;
- deepagents 0.7.13 and langchain-anthropic 1.7.6, the versions Foundry runs, read from their wheels and not run.

**One local check (no model):** the geotech page's real agent build (`core.build_agent`, the same code as build i) sent one turn to the stand-in Foundry proxy that `webapp/tests/test_foundry_model_family.py` uses, and the raw request bodies were read (section 2.3).

---

## Verdict

- **The answer is right.** Ka = 0.333 from the module's Rankine route, given with the formula, the design use and what would change it.
- **The turn did what the prompt tells it to do.** It listed the module's methods, described the method before first use, called it and answered. There was no wasted call, no error and no retry.
- **146,036 tokens is the gross count this build spends on any one-method question.** Each of the four model calls re-sends a ~33.7k-token prefix: the 40 tool definitions are about 72 % of it and the system prompt about 28 %. That repeated prefix is 92.9 % of the turn's input tokens.
- **The record cannot say whether the prompt cache was read, so it cannot say what the turn cost.**
  - `GEOTECH_FOUNDRY_PROMPT_CACHE=unset` means caching is ON. The cache marks go out on the last system block and the last message block, and the proxy accepted them (no 400).
  - `activity.jsonl` keeps only `input_tokens`, `output_tokens` and `total_tokens`. langchain-anthropic's `input_tokens` already counts cache reads and writes, so 146,036 reads the same with or without the cache.
  - At Anthropic's published Opus 5.5 rates, the turn costs about $0.60 with no cache read and about $0.23 with the cache read as marked.
- **Storage worked.**
  - The five record files landed 6.8 s after the turn ended, in one save.
  - The feedback note's two files landed 5.3 s after it was sent, without a turn.
  - Restart, the gone check and restore were not run, so INF-38 stays open on them.

## 1. The turn, step by step

The whole question: "Rankine active earth pressure coefficient Ka for phi' = 30 degrees, level backfill, vertical smooth wall." It was turn 1 of a new conversation, 42 min after the app's process started, beginning at 20:40:25.046Z.

**What the model was shown on its first call (`n_messages` 2):**
- the system prompt, 27,603 chars, with one fingerprint across all four calls;
- 40 tools (`tools_offered`; `HideTools` hides `delete` and `execute`);
- the question.

**The prompt rules this turn followed:**
- The "Tool Discipline" section says: "Before using any method for the first time, call `describe_method`" and "If unsure which methods a module exposes, call `list_methods` first".
- The "CRITICAL" section says to use `call_agent` for any quantitative result, earth pressure included.

| call | starts (s) | model s | messages | tokens in | out | what it did | tool result |
|---|---:|---:|---:|---:|---:|---|---|
| 1 | 0.03 | 4.15 | 2 | 33,710 | 115 | `list_methods(retaining_walls)` | 1,180 chars in 0.011 s: four groups, with `earth_pressure_coefficient` under Earth Pressure |
| 2 | 4.20 | 3.59 | 4 | 34,347 | 87 | `describe_method(retaining_walls, earth_pressure_coefficient)` | 8,624 chars in 0.001 s: nine parameters with defaults (theory `rankine`, state `active`, beta 0, alpha 90) |
| 3 | 7.81 | 3.83 | 6 | 38,407 | 145 | `call_agent(retaining_walls.earth_pressure_coefficient, phi_deg 30, theory rankine, state active, beta_deg 0)` | 141 chars in 0.001 s: `K` 0.3333, `coefficient` Ka, `delta_deg` 0.0, `alpha_deg` 90.0 |
| 4 | 11.65 | 8.13 | 8 | 38,638 | 587 | the answer, 1,346 chars, no tool | - |
| **turn** | | **19.70 of 19.78** | | **145,102** | **934** | 3 tool calls, 12 graph steps of 50 | `turn_tokens` 146,036 |

- Model time is 99.6 % of the turn. The three tools took 0.013 s together.
- `trace.jsonl` gives 19.819 s because its clock starts about 0.04 s before the activity log's. It is the same turn.
- Only call 4 has the model's own words. Calls 1-3 have no text and no `reasoning`, because Opus 5.5 returns its thinking empty unless asked. The steps follow the prompt rules above exactly, so the "why" of each step is clear here.
- Call 3 passed the inputs the question states and nothing else. Delta was left to the Rankine value of 0, which is what "smooth wall" means.

**The answer checked:**
- Ka = tan²(45° − φ'/2) = tan²(30°) = (1 − sin 30°)/(1 + sin 30°) = 0.5/1.5 = 1/3 = 0.3333. The module returned 0.3333 and the answer says 0.333 (exactly 1/3). Right.
- The rest is right too:
  - Pa = ½ Ka γ H², acting at H/3, for dry cohesionless backfill;
  - below the water table, Ka applies to the effective vertical stress and the water pressure is added separately;
  - backfill rising away from the wall raises Ka;
  - Coulomb with δ > 0 lowers Ka a little (about 0.30 at φ' 30°, δ 20°).
- K0 = 1 − sin φ' = 0.50 (Jaky, OCR 1) is right, but the model stated it without a tool call. That goes against the prompt's "NEVER do arithmetic in your head" (noted line).
- "There are no judgment calls in the number itself" (the judgment-calls switch is on) is fair.

## 2. Where the 146,036 tokens went

### 2.1 The arithmetic

- 33,825 + 34,434 + 38,552 + 39,225 = 146,036, the sum of each call's `total_tokens`: 145,102 in and 934 out.
- Call 1's 33,710 tokens are the fixed prefix: system prompt, tools and question. Each later call re-sends that prefix plus the conversation so far:
  - call 2 adds 637 (call 1's tool call and the 1,180-char method list);
  - call 3 adds 4,060 (call 2's tool call and the 8,624-char description);
  - call 4 adds 231 (call 3's tool call and the 141-char result).
- 4 × 33,710 = 134,840 of the 145,102 tokens in (92.9 %) is the prefix sent again. The conversation itself adds 10,262. The method description alone is about 4k tokens, and calls 3 and 4 both carry it.

### 2.2 What is in the 33.7k-token prefix

- The record gives the system prompt's size (27,603 chars), but for the tools only their names and a fingerprint, not their size.
- Measured on a local build of the same code, which has the same 40 tool names: 69,091 chars of tool definitions.
- By character share that is about 72 % tools (about 24k tokens) and 28 % system prompt (about 9.6k tokens), at this turn's own 2.9 chars a token.
- This split is an estimate. The local system text differs (37,115 chars on deepagents 0.6.8, against Foundry's 0.7.13), and deepagents' own tools may be worded differently on Foundry.
- The four largest tool definitions:
  - `task`: 11,087 chars (deepagents' helper spawner, with all five helpers described);
  - `plot_data`: 8,139;
  - `annotate_document`: 6,879;
  - `write_todos`: 4,335.
- The 18 document and drawing tools take 29,064 chars, which is 42 % of the tool text and about 10k tokens on every call, even on a turn with no document. They are `open_document`, `read_document`, `search_document`, `document_*`, `render_*`, `analyze_*`, `find_like`, `find_quantities`, `log_grid`, `measure`, `kept_readings`, `annotate_document`, `annotate_dxf` and `read_pdf_text`.

### 2.3 The prompt cache: on, marked and accepted, but not recorded

**How the marks get onto the request:**
- With `GEOTECH_FOUNDRY_PROMPT_CACHE` unset, caching is ON. Only `0`, `off`, `false` or `no` strips the marks (`FoundryChatAnthropic._get_request_payload` in `webapp/engine_config.py`).
- deepagents 0.7.13 always adds langchain-anthropic's `AnthropicPromptCachingMiddleware` (`deepagents/middleware/_prompt_caching.py`).
- `FoundryChatAnthropic` names a non-direct `_llm_type`, so langchain-anthropic 1.7.6 moves the top-level mark onto the last message block. Then `sanitize_foundry_anthropic_payload` drops the tool mark.

**The local check:** the real build through the stand-in proxy, on deepagents 0.6.8 and langchain-anthropic 1.4.4.
- Both requests carried exactly two marks. One was on the last of the 4 system blocks. The other was on the last message block: the question in the first request, the tool result in the second.
- No mark was at the top level or on a tool.
- The API reads tools, then system, then messages, so the system mark also covers the tools.

**The proxy took the marks.** There was no 400 and no retry (the record has no `http` fields). The 2026-10-20 probe cited in the docstring of `sanitize_foundry_anthropic_payload` found such marks "accepted and cached" on this proxy.

**The record cannot show whether this turn read the cache:**
- `_usage_from_response` (`webapp/activity_log.py`) keeps three keys of `usage_metadata` and drops `input_token_details` (`cache_read`, `cache_creation`).
- langchain-anthropic's `_create_usage_metadata` adds cache reads and writes into `input_tokens` (in 1.4.4 locally and in 1.7.6 on Foundry), so `turn_tokens` 146,036 is the same either way.
- On 2026-10-27, EV-16 parked "keep the cache-token split in `usage`" because no question needed it. This turn asks that question.
- The timings do not settle it. Call 1 (cold) took 4.15 s and calls 2 and 3 took 3.59 s and 3.83 s, but the thinking tokens inside each call's output are unknown.

**What the cache is worth** at Anthropic's published Opus 5.5 rates ($4 in and $20 out per million; a 5-minute cache write at 1.25×; a cache read at $0.20). What Foundry charges for the proxy is not in the record.

| | tokens | cost |
|---|---|---:|
| no cache read | 145,102 in, 934 out | about $0.60 |
| cache read as marked | 38,638 written (33,710 + 637 + 4,060 + 231), 106,464 read, 934 out | about $0.23 (−61 %) |

Even with the cache, a turn's first call writes the whole prefix again when the last call was more than 5 minutes earlier. So a short question asked later costs about the same as this one.

### 2.4 Were the calls needed?

- Yes, under the prompt's rule. Two of the four calls are discovery (`list_methods`, `describe_method`), which the Tool Discipline section asks for before a method's first use. A later question to the same method in the same conversation would skip them.
- The method description is 8,624 chars, mostly about the log-spiral passive sources, which a Rankine active call does not use (noted line).

## 3. Storage

| event | time (UTC) | in the dataset | gap |
|---|---|---|---:|
| turn ends (`turn_end`; `trace.jsonl` written) | 20:40:44.824 | 5 files updated 20:40:51.608-.744: transcript 1,589 B, meta 2,748 B, trace 364 B, activity 18,998 B, messages 1,845 B | 6.8 s |
| feedback note saved | 21:07:56.799 | `FEEDBACK.md` 518 B and `feedback.jsonl` 387 B at 21:08:02.081-.155; `meta.json` unchanged | 5.3 s |

**What the save shows:**
- The five files in the hand-back are the same sizes as the dataset files the FDE listed.
- They were all updated within 136 ms, which fits one `upload_directory` transaction. The FDE had no tool to read the transaction history and confirm it.
- The 6.8 s matches the 5-7 s the code expects. `turn_jobs.py` queues the save to run after the turn, so the chat does not wait for it. The FDE did not notice whether the chat waited, and nothing in the record can show it.
- The feedback save wrote the two feedback files and left `meta.json` unchanged. `meta.json` is the only other file the FDE checked.
- "Sync now" after the turn's save: 0 uploaded, 5 up to date, 0.001 s.

**`activity.jsonl` does not record the background save, and cannot record its own.** The save that carries a turn's record runs after the record is closed. The save's result (time, files, errors) is kept only in memory for the sidebar and is lost at a restart. The FDE saw no save line after the turn; INF-65 already asks the sidebar to show the save's result.

**Not run:** restart, the gone check, restore and the later turn, which are INF-38's round trip and still open. This turn cannot say whether `upload_directory` skips names that start with `_` or `.`, because none of the five files has such a name.

The empty dataset's 400 before the placeholder upload is INF-65.

## 4. Gaps in the record

1. **No cache split in `usage`** (section 2.3). The record cannot say what a turn cost. See Fix 1.
2. **No tool-definition size.** The engine record names the 42 tools and gives a fingerprint. To split a call's tokens between system and tools, a reader has to rebuild the agent locally. One number (`tools_chars`) would do (noted).
3. **No model reasoning.** Opus 5.5 returns its thinking empty by default, so calls 1-3 have no words at all. Their output tokens (115, 87, 145) cannot be split between thinking and the tool call. Here the prompt rules explain each step; on a turn that goes wrong they might not (noted).
4. **No save record.** When the background save ran, what it sent, how long it took and whether it failed are not kept anywhere that survives a restart (section 3; noted, beside INF-65).
5. **Whether the chat waited for the save.** The code says no, but the record cannot confirm it.

## 5. The feedback entry

The note arrived.
- What it says: "Can you see this feedback?"
- Where it came from: the sidebar box (`source` user, `kind` user_feedback), at 21:07:56Z on turn 1, with the app version and model in its context.
- Where it went: `FEEDBACK.md` and `feedback.jsonl`, and the dataset 5.3 s later without a turn or Sync.

The answer to its question is yes: this review read it.

## Fixes

A fix is listed only if it changes what a live run returns or costs.

1. **Prompt cache on the Foundry proxy: record the cache split, and confirm the cache is read** (cost).
   - *Evidence:* section 2.3.
     - Every call re-sends the ~33.7k-token prefix, which is 92.9 % of the turn's input.
     - The marks go out and are accepted, but the record cannot show reads.
     - With reads, the turn costs about 61 % less than without them. If the proxy does not read them, every turn pays full price and the cause has to be found.
   - *Change:* keep `input_token_details.cache_read` and `cache_creation` in each `model_end`'s `usage`, and sum them on `turn_end`, as report ingest already counts them (RI-171). This takes that piece out of EV-16's parked list.
   - *File:* `webapp/activity_log.py` (`_usage_from_response` and the `turn_end` record).
   - *Check:* on the next Foundry turn shaped like this one, calls 2-4 should read about 33.7k, 34.3k and 38.4k tokens from the cache.
2. **Measure first: the document tools on turns with no document** (cost).
   - *Evidence:* section 2.2. The 18 document and drawing tools are about 10k tokens of every call's prefix on any geotech turn, with no document in the conversation. That is about a quarter of a short turn's cost whether or not the cache is read, because with the cache the first call writes them at 1.25×.
   - *Change, if measurement shows it is worth it:* keep them off the model's list until the conversation holds a document. The tools stay registered, as `HideTools` already does for `delete` and `execute`. Showing them later changes the cached prefix once.
   - *Risk to measure:* a turn that fetches a document and must open it in the same turn.
   - *Files:* `funhouse_agent/deep/agent.py` (the geotech build's middleware) and `funhouse_agent/deep/limits.py` (`HideTools`); `webapp/tests/page_fingerprints.json` re-recorded in its own commit.

## Noted, not fixed

- K0 = 0.50 was given without a tool call, against the prompt's rule; the value is right.
- `describe_method` for `earth_pressure_coefficient` is 8,624 chars, mostly the log-spiral passive sources; a Rankine active call carries about 4k tokens it does not use.
- `task`'s definition is 11,087 chars on every call. That is deepagents' own text plus the five helper descriptions.
- The engine record gives no tool-definition size (gap 2).
- The record has no model reasoning (gap 3). Asking the model to return a summary of its thinking would fill it, but the proxy has not been seen to accept that field.
- The background save leaves no record that survives a restart (gap 4); the sidebar half is INF-65.
- INF-38's restart, gone check and restore were not run in this build.
