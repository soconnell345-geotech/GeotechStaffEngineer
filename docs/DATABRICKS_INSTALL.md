# Installing on the Funhouse / Databricks cluster — and reading the log

**Why this file exists.** Every release the owner pastes a `%pip install` log and
asks "any concerns?". Three times running, an agent re-derived the same answers
from scratch — which packages cause the numpy cascade, whether the conflict
warnings matter, whether a new major version is safe. This file is the standing
answer so that review is a two-minute check against a list, not an
investigation. **Read this BEFORE triaging an install log.**

Last verified against a real cluster log: **5.14.0, 2026-09-11** (clean).

---

## 1. The install

```python
%pip install "geotech-staff-engineer==5.20.3"
%restart_python
```

No extras, no pins, no patch cell. `[deep,full]` has not been needed since
5.11.x — if you see it in any doc, that doc is stale.

The restart is not optional: pip installs into a notebook-scoped environment
that the already-running Python process cannot see.

---

## 2. What "good" looks like

Three lines decide it. Find them and you're done:

1. **`Successfully installed ... geotech-staff-engineer-5.20.0 ...`** — with
   `planlens-0.6.0` and `geotech-references-1.4.0` in the same list.
   `planlens-0.6.0` brings no package that `planlens-0.4.0` did not, so
   nothing new should appear beside it.
2. **No `403` and no `quarantined` anywhere.**
3. The final `ERROR: pip's dependency resolver ...` block lists **only** the
   conflicts in §3. That block is a *warning printed as ERROR* — pip says
   "ERROR" and then installs successfully anyway. It is not an install failure.

If all three hold, the install is good regardless of how alarming the middle of
the log looks.

---

## 3. Expected, benign, and already explained

These appear on every install. None of them is a problem. The **"changes if"**
column is what would make one worth a second look.

| Log line | Why it's benign | Changes if |
|---|---|---|
| `tensorflow 2.19.0 requires numpy<2.2.0` | numpy cascade, §4. Only bites if *that kernel* also runs TensorFlow. | You start using TF in the same notebook |
| `mosaicml-streaming 0.12.0 requires numpy<2.2.0` | same | same |
| `ydata-profiling 4.16.1 requires numpy<2.2 / numba<=0.61` | same (two lines, one cause) | same |
| `langchain-text-splitters 0.3.8 requires langchain-core<1.0.0` | Ours — we upgrade `langchain-core`. But **we never import text-splitters** (verified: zero references in the repo). It is a stale Databricks-runtime package complaining about a library it doesn't need to talk to. | We ever add a text-splitters import |
| `datasets 3.5.0 requires fsspec<=2024.12.0` | **Not ours.** fsspec was already at 2026.7.0 before our install; we don't touch it. | — |
| `pmdarima 2.1.1 requires statsmodels>=0.14.5` | **Not ours.** Pre-existing runtime mismatch. | — |
| `Can't uninstall 'X'. No files were found to uninstall.` | Normal notebook-scoped behavior: the new copy **shadows** the cluster-wide one rather than replacing it. Appears for numpy, numba, openai, langchain, etc. | — |
| `INFO: pip is looking at multiple versions ... This is taking longer than usual` followed by dozens of `httpx2` / `uvicorn` downloads | Resolver backtracking, §5. Slow, not broken. | It ends in `ResolutionImpossible` |
| Agent-stack versions higher than the gate ran on | §6 — expected by design. | §6's red flag fires |

---

## 4. The numpy cascade — settled, don't re-derive it

**`PyNiteFEA` requires `numpy>=2.4.0`.** That is the entire cause. Confirmed
from the published metadata of the exact wheel the cluster installs
(`PyNiteFEA 3.1.0`), not inferred from the log.

The chain:

- Databricks boots with **numpy 2.1.3**.
- PyNiteFEA forces **numpy ≥ 2.4.0**; `numba` caps **numpy < 2.5** → the
  resolver lands on exactly **numpy 2.4.6**, every time.
- numpy 2.4.6 breaks the cluster's **numba 0.61** (caps numpy<2.2) → numba
  **0.61 → 0.67**, llvmlite **0.44 → 0.49**.
- That, and only that, produces the four numpy/numba conflict lines in §3.

**Nothing else in the tree needs numpy above 2.1.3.** So moving PyNiteFEA to an
optional extra would remove *all four* warnings in one line — it is the whole
fix, not part of one. The cost is that frame analysis would then need an extra
on install. **Owner's call; not urgent.**

**numba 0.67 is verified safe, not assumed.** pystrata JIT-compiles its wave
propagation core, and the gate had only ever run numba 0.65.1. A live cluster-
vs-gate comparison (30 m Vs=180 PI=15 over Vs=760 rock) matched to 4 decimal
places on EQL surface PGA, amplification and max strain. Numerics unchanged.

### Re-checking it yourself

If you ever want to confirm which package is forcing numpy up, this prints it
directly from installed metadata — no guessing from the log:

```python
from importlib.metadata import distributions
from packaging.requirements import Requirement
from packaging.version import Version
target = Version("2.1.3")          # what Databricks boots with
for d in distributions():
    for r in (d.requires or []):
        try: req = Requirement(r)
        except Exception: continue
        if req.name.lower() != "numpy": continue
        if req.marker and not req.marker.evaluate(): continue
        if str(req.specifier) and not req.specifier.contains(target, prereleases=True):
            print("FORCES UPGRADE:", d.metadata["Name"], d.version, req.specifier)
```

---

## 5. The backtracking storm (why the install is slow)

`httpx2` walks 2.12 → 2.0 and `uvicorn` walks 0.52 → 0.30, downloading each.
This is the resolver searching, not failing.

Cause: `langsmith 0.12.4` (a **deepagents** dependency) pulls the `httpx2` /
`httpcore2` fork stack. The `openai<3` pin in `pyproject.toml` was written
specifically to keep that stack off the cluster and **no longer achieves it** —
httpx2 now arrives by a different road. Nothing is broken (the Prompter's NTLM
path still runs on the untouched httpx 0.28.1), but **the pin's rationale
comment is wrong about what protects what.** Backlog item, not a blocker.

---

## 6. Version drift: the gate runs the floor, the cluster runs the ceiling

Our agent-stack pins are ranges (`deepagents>=0.6.8,<0.8`). The local gate
resolves to whatever is already installed — often the **floor**. The cluster
resolves fresh against the proxy and gets the **ceiling**. On 5.14.0 that was
deepagents **0.6.8** (gate) vs **0.7.13** (cluster).

This is the exact shape of the **5.10.2 outage**, when deepagents 0.7.11
stopped auto-attaching the todo middleware while our prompt still told the model
to call `write_todos` — every question failed with a recursion error.

**It is safe now, by design rather than by luck.** `build_deep_agent`
(`funhouse_agent/deep/agent.py`) inspects the **compiled agent** for a
`write_todos` tool and re-attaches `TodoListMiddleware` if it's missing, instead
of sniffing version numbers. That self-corrects on deepagents versions that did
not exist when the guard was written.

**Red flag:** the agent answering every question with a recursion error right
after an install. That means a new deepagents changed something the compiled-
agent check doesn't cover. Everything else about version drift is expected.

---

## 7. Major-version watch list

A dependency arriving at a **new major version** is the one class of drift that
deserves a real check, because our pins mostly have no upper bound. Currently
verified:

| Package | Pin | Cluster has | Status |
|---|---|---|---|
| `opencv-python-headless` (via `planlens`) | `>=4.8` — **no ceiling** | **5.0.0.93** | **Verified safe.** Full planlens suite (792 passed / 1 skipped, the suite as it stood on 2026-09-11; it is 1,307 at planlens 0.6.0 and has NOT been re-run against OpenCV 5 since) run against OpenCV 5.0.0. The cv2 surface we use is small and long-stable (`imdecode`, `cvtColor`, `Canny`, `HoughLinesP`, `HoughCircles`, `findContours`, `approxPolyDP`, `threshold`, `medianBlur`, `rotate`), and `findContours` is already unpacked version-agnostically (`found[0] if len(found)==2 else found[1]`). |
| `rapidfuzz` (via `planlens`, new at 5.18.0) | `>=3.9` — **no ceiling** | **3.14.6** expected | **New at 5.18.0.** Arrives with planlens 0.4.0, which moved it into its core; it is not an app dependency. Cleared the Nexus probe 2026-09-16 at 3.14.6. MIT, and it pulls **no dependencies of its own** — a C++ extension with prebuilt wheels, so it cannot disturb the numpy or streamlit resolution. |
| `numpy` | `>=2.0` | 2.4.6 | §4 |
| `streamlit` | `>=1.39` | 1.63.0 | fine |
| `openai` | `<3` | 2.54.0 | bounded |
| agent stack | ranges `<0.8` / `<1.4` / `<1.3` | ceilings | §6 |

**planlens 0.6.0 (the 5.20.0 pin) adds no third-party package at all**, and
neither did 0.5.0. Every one of them declares exactly what 0.4.0 declared —
numpy, ezdxf, PyMuPDF, opencv-python-headless, rapidfuzz, with `ocr` and
`mcp` as extras we do not install — so the 5.20.0 log should read like
5.18.0's with `planlens-0.6.0` in place of `planlens-0.4.0` and nothing else
moved. The app's own `report_ingest` package is pure Python over what is
already installed: `anthropic` is optional and deliberately absent here, and
`openpyxl` already arrives as a hard requirement of `python-ags4`.

**OpenCV 5 arrived unannounced and unbounded** — we only found out by reading
an install log. It happens to be fine. Optional hardening: bound planlens'
OpenCV dependency to `>=4.8,<6` so the *next* major version is a deliberate
decision instead of a discovery. That needs a planlens point release, so it is
the owner's call. (Since planlens 0.4.0 that dependency is in planlens' CORE,
not the `raster` extra — `[raster]` and `[text]` are now empty alias extras
kept only so older pins keep resolving.)

### Checking a major bump yourself

```bash
# does the library still work on the new major version?
python -m venv scratch && scratch/Scripts/python -m pip install \
    "opencv-python-headless==<new version>" numpy ezdxf PyMuPDF pytest
scratch/Scripts/python -m pip install --no-deps <path to planlens>
cd <planlens> && ../scratch/Scripts/python -m pytest -q
```

---

## 8. What is deliberately NOT installed on the cluster

**OCR.** The app pins plain `planlens`, whose core brings OpenCV and rapidfuzz
— **not** `rapidocr-onnxruntime`. So scanned sheets and SHX-stroked lettering
cannot be read optically on the cluster. (`fuzzy=true` on `search_document`
does NOT change this: it forgives errors in text that was already read, and
there is no OCR on the cluster to produce that text from a scan.)

This is intentional: every rapidocr distribution hard-requires the full **GUI**
`opencv-python`, which collides in the `cv2` namespace with the headless build
a server needs, and pip cannot express "either variant". Headless deploys use
the `--no-deps` recipe in the planlens README instead.

Failure mode is graceful and self-explaining — the tool returns
*"OCR support needs the optional extra: pip install planlens[ocr]"* rather than
crashing the turn. Vector-text sheets (the common case) are unaffected: they are
read exactly, and OCR is skipped on them deliberately.

---

## 9. Post-install verification (cluster only)

Two things can only be checked on the cluster:

- **Prompter metering** — usage must reach Funhouse's meter (a compliance
  obligation). Snippet: `module_work/prompter_metering/DIAGNOSIS.md` §8.
- **DXF block explosion** — in a drawing result, `blocks_exploded: false`
  should be **gone** and `n_block_entities` should **appear**.
- **Reference PDFs for chart read-off** — not part of the install at all: the
  PDFs live in SharePoint `GSE_app/primary_references`, and each is downloaded
  the first time a chart from it is needed (a local folder named by
  `GEOTECH_REFERENCES_DOCS` is tried first when set). The sidebar's Connection
  diagnostics row "reference PDFs (chart read-off)" says how many it can
  reach. Setup: `webapp/README.md` §3 "Reference PDFs".

---

## 10. Actual red flags

Stop and investigate only for these:

- `403` / `Requested item is quarantined` — the Nexus malware-defense proxy
  blocked a package. No waivers are available for this project; the remedy is
  to remove the dependency and re-implement natively (the **groundhog /
  cytriangle precedent**: pin the retired library's outputs as numerical
  oracles *before* deleting it, then rebuild from published sources — no code
  copied). Ledger: `module_work/structural_native/`.
- `ResolutionImpossible`, or backtracking that ends in an error rather than
  `Successfully installed`.
- A package we depend on **failing to build** a wheel.
- `geotech-staff-engineer`, `planlens`, or `geotech-references` resolving to an
  **older version than requested**.
- After install: the agent failing **every** question with a recursion error
  (§6).
- A dependency at a **new major version** not in §7's table.

---

## 11. History

| Version | Date | Install outcome |
|---|---|---|
| 5.12.0 | 2026-09-05 | Clean. |
| 5.13.0 | 2026-09-10 | Clean. numpy cascade fired as predicted; numba 0.67 verified numerically identical. |
| 5.14.0 | 2026-09-11 | Clean. OpenCV 5.0.0.93 arrived and was verified safe; deepagents 0.7.13 vs gate's 0.6.8 held via the compiled-agent guard; app confirmed working live by the owner. |
| 5.15.0 | — | Not yet installed (app feedback train: no dependency changes, so the same log is expected). Replace this row when the cluster install is confirmed. |
| 5.16.0 | — | Not yet installed (document-review train: pin `planlens[raster]>=0.3`; planlens 0.3.0 adds pure-Python subpackages over PyMuPDF, no new third-party packages, so the same log plus `planlens-0.3.0` is expected). Replace this row when the cluster install is confirmed. |
| 5.17.0 | — | Superseded by 5.17.1 the same day (deliverable-import fix); install 5.17.1 instead. |
| 5.17.1 | — | Not yet installed (Nairobi SOE fix train + one follow-up fix: **no dependency changes**, so the log should read exactly like 5.16.0's). Replace this row when the cluster install is confirmed. |
| 5.18.0 | — | Not yet installed — pin `planlens>=0.4`; planlens 0.4.0 brings `rapidfuzz` (new on the cluster; it cleared the Nexus firewall 2026-09-16 as 3.14.6, MIT, no dependencies of its own) and keeps `opencv-python-headless`, now in planlens' core rather than the `raster` extra; **no new direct app deps**. Expect the 5.16.0 log plus `planlens-0.4.0` and `rapidfuzz-3.14.6`. Release gate 11,773 passed / 33 skipped / 0 failed. Replace this row when the cluster install is confirmed. |
| 5.19.0 | — | Not yet installed — pin `planlens>=0.5`. planlens 0.5.0 declares **exactly** the dependencies 0.4.0 did (numpy, ezdxf, PyMuPDF, opencv-python-headless, rapidfuzz), so expect the 5.18.0 log with `planlens-0.5.0` in place of `planlens-0.4.0` and **no other package moved**. The new `report_ingest` package is pure Python over what is already there: `anthropic` stays optional and must NOT appear, and `openpyxl` (the hand-label reader, imported lazily) already arrives with `python-ags4`. Release gate 11,890 passed / 33 skipped / 0 failed. Replace this row when the cluster install is confirmed. |
| 5.20.0 | 2026-09-18 | **Installed and imported on the cluster**: the owner's first scoring run proved the install, the `planlens-0.6.0` pin and the corpus paths, then stopped on a model parameter (fixed in 5.20.1, shimmed meanwhile). Pin `planlens>=0.6`. planlens 0.6.0 declares **exactly** the dependencies 0.4.0 and 0.5.0 did (numpy, ezdxf, PyMuPDF, opencv-python-headless, rapidfuzz), so expect the 5.19.0 log with `planlens-0.6.0` in place of `planlens-0.5.0` and **no other package moved**. What is new is all pure Python inside `report_ingest` (the record, three readers, the reconciler, the writers, the DIGGS 2.6 writer, the sub-agent, five-stage cluster scoring including the vision-first experiment): `anthropic` stays optional and must NOT appear, and `openpyxl` still arrives with `python-ags4`. Release gate 12,381 passed / 33 skipped / 0 failed. Replace this row when the cluster install is confirmed. |
| 5.20.1 | — | On PyPI 2026-09-18; **reaches the cluster a day or two later** (the Nexus mirror lag — never assume a release is installable the day it is cut). Changes no dependency: 5.20.0 plus the Prompter parameter fix in `report_ingest/engine.py` (the triage tier's model refused `max_tokens`; now adapted and remembered). Expect the 5.20.0 log exactly. Until it arrives, the 2026-09-18 scoring runs use **5.20.0 + a notebook shim** (a wrapper on the Prompter client renaming `max_tokens`, `prefer_chat` off), and their results are labelled that way. |
| 5.20.2 | — | On PyPI 2026-09-18, same mirror lag. No dependency change: 5.20.1 plus the resume fix in `report_ingest/cluster_scoring.py` (a run file that recorded a failed model call is retried, not skipped as done). Expect the 5.20.0 log exactly. |
| 5.20.3 | — | On PyPI 2026-09-18, same mirror lag. No dependency change: 5.20.2 plus the strict-schema fix in `report_ingest/engine.py` (pydantic's `default`/`minItems`/`maxItems` stripped from every structured-output schema; OpenAI refused the readers' schemas before any token was spent). Expect the 5.20.0 log exactly. **Install this one** when it arrives; 5.20.1 and 5.20.2 are superseded. |

Earlier, 5.12.0's **first** attempt failed: pip 403 on `cytriangle`, pulled in by
`sectionproperties` **and** `concreteproperties`. Both were removed and rebuilt
natively for 5.13.0. That is why §10's first bullet is written the way it is.
