# HANDOFF — GeotechStaffEngineer (current state, read this first)

**Last updated: 2026-09-08.** This is the authoritative handoff for a fresh LLM
session (any model). The older `HANDOFF_2026-06-14.md` is kept only for the
detailed Phase-E history; this file supersedes it.

---

## 0a-current. PICKUP LIST (2026-09-08, supersedes everything below)

### 5.14.0 RELEASED with planlens 0.2.0 (2026-09-11) — both trees COMMITTED

The publish order finally ran as designed: **planlens 0.2.0** (`f0a2b9d`, tag
`v0.2.0`, trusted-publisher workflow, live on PyPI) then **app 5.14.0**
restoring `planlens[raster]>=0.2`. Gate on the release tree 11,594 / 33
skipped / 0 failed; real PyPI resolve of the built wheel: 149 packages,
planlens 0.2.0, refs 1.4.0, no cytriangle chain anywhere.

**Cluster:** `%pip install "geotech-staff-engineer==5.14.0"` then
`%restart_python`. Carries, on top of 5.13.0: the Prompter METERING fix
(`_child_env` threads the kernel's meter config to the app subprocess — see
the metering section; verify with §8 of
`module_work/prompter_metering/DIAGNOSIS.md`), the OCR render cap (planlens)
and OCR-skipped-when-the-page-already-has-text (adapter, `ocr_force` escape
hatch), the SharePoint base-folder de-dup, and DXF block explosion back via
planlens 0.2.0. The PyNiteFEA numpy>=2.4 cascade is unchanged (owner's
call to keep it in core; site response verified numerically identical on
numba 0.67).

**Both working trees are now clean of release work.** The app tree still has
the owner's untracked scratch (the `*_copy.md` plans, docs PDFs, the .bat,
`bamako_agent_cell.py`, `suite_trial.json`) — deliberately never staged.
planlens has the one stray screenshot, never staged.

**Open, in priority order:**
1. Cluster verification of the metering fix (§8 snippet) and of DXF block
   explosion (`blocks_exploded` no longer appears; `n_block_entities` does).
2. The Prompter 401 — send the revised admin note; run the confirming test
   when convenient. Two app-side items still PLANNED: retry-once on
   `AuthenticationError`, `friendly_turn_error` auth case.
3. Owner decision: PyNiteFEA -> extra to undo the numpy cascade (TF /
   mosaicml / ydata-profiling broken in that kernel). CONFIRMED from published
   metadata that PyNiteFEA 3.1.0 requires numpy>=2.4.0 and is the SOLE driver,
   so this one line removes all four warnings. `docs/DATABRICKS_INSTALL.md` §4.
4. The `openai<3` pin comment now lies — httpx2/httpcore2 arrive via
   langsmith; fix the comment (and decide whether the pin still earns its
   place). This is also what makes the install slow (resolver backtracking):
   `docs/DATABRICKS_INSTALL.md` §5.
4b. Optional: bound planlens' raster extra to `opencv-python-headless>=4.8,<6`
   so the next OpenCV major is a decision rather than a discovery (§7).
5. planlens findings 5 (60-deg+ triangles) and 6 (concave dart), documented
   not fixed; a synthetic-terminator fixture set alongside the corpus is the
   standing lesson of rounds 4-6 (`round4_repro.py`, `round5_attack.py`).
6. Chunked websocket upload for files past the 25 MB cap.

### 5.14.0 INSTALLED AND VERIFIED ON THE CLUSTER (2026-09-11) — install-log triage now has a home

`%pip install "geotech-staff-engineer==5.14.0"` **succeeds and the app runs**
(owner confirmed live). No 403, no quarantine, no cytriangle chain. planlens
0.2.0 and refs 1.4.0 both landed.

**READ `docs/DATABRICKS_INSTALL.md` BEFORE TRIAGING ANY INSTALL LOG.** Three
releases running, an agent re-derived the same answers from scratch — which
package drives the numpy cascade, whether the seven conflict warnings matter,
whether a new major version is safe. That file is now the standing answer:
what "good" looks like in three lines, every expected-and-benign log line with
what would change its verdict, the real red flags, and copy-paste snippets to
re-check a numpy driver or a major bump. Keep its §11 history table current.

Two findings from the 5.14.0 log that were NOT previously recorded:

**OpenCV arrived at a new MAJOR version, unbounded and unannounced.** The
cluster installed `opencv-python-headless 5.0.0.93` because planlens pins
`>=4.8` with no ceiling. **Verified safe**: the full planlens suite (792
passed / 1 skipped) was run against OpenCV 5.0.0, and the local gate machine
turns out to have been on cv2 5.0.0 already — so it was tested, but by
accident rather than by design. The cv2 surface we use is small and
long-stable, and `findContours` is already unpacked version-agnostically.
Optional hardening: bound the planlens raster extra to `>=4.8,<6` so the next
major version is a decision, not a discovery — needs a planlens point release,
so it is the owner's call. Watch table: `docs/DATABRICKS_INSTALL.md` §7.

**The gate runs the FLOOR of our agent-stack ranges; the cluster runs the
CEILING.** 5.14.0 shipped with the gate on deepagents **0.6.8** and the cluster
resolving **0.7.13** — the exact shape of the 5.10.2 outage. It held, and by
design rather than luck: `build_deep_agent` inspects the **compiled agent** for
`write_todos` and re-attaches `TodoListMiddleware` when missing, instead of
sniffing versions, so it self-corrects on deepagents releases that did not
exist when the guard was written. The red flag to watch for is the agent
failing *every* question with a recursion error right after an install.

**PyNiteFEA's numpy floor is now CONFIRMED, not inferred**: `PyNiteFEA 3.1.0`
declares `numpy>=2.4.0` in its published metadata. With numba capping
numpy<2.5 the resolver lands on exactly 2.4.6 every time, and nothing else in
the tree wants numpy above 2.1.3 — so moving PyNiteFEA to an extra removes
**all four** numpy/numba conflict lines, not merely some of them. The watch
item written into `pyproject.toml` at the 5.13.0 pin is hereby closed.

### FUNHOUSE METERING WAS SILENTLY DROPPING EVERY APP RECORD (2026-09-10)

The owner noticed Funhouse's token counter was not picking up app usage.
Diagnosed with executed proof; full report committed at
`module_work/prompter_metering/DIAGNOSIS.md`.

**Two plausible theories were DISPROVEN by execution before the real one was
found** — worth knowing, because both are the kind of thing a reader would
accept on reasoning alone. LangChain is NOT building its own unwrapped OpenAI
client (we use our own `PrompterChatModel`,
`funhouse_agent/deep/databricks_bridge.py:397`, which calls the SDK's wrapped
client directly), and the `is_logging_active` ContextVar guard is innocent.
`wrap_all_openai_methods` really does fire `meter_log` with real usage.

**The cause is one layer below the wrapper.** `meter_log` writes to a LOCAL
SQLite file whose path exists only in the notebook kernel's in-memory
`FunhouseConfig` singleton. The app is `Popen`'d as a fresh process
(`databricks_launcher.py:695`) and builds `PrompterAPI` with no `config=`, so
it sees DEFAULTS: `should_write_sqlite_for_user -> False`,
`get_budget_sqlite_store -> None`, every record dropped. The one diagnostic it
emits is INFO, below the default WARNING stdout level, so nothing ever said
so. Gap is TOTAL for app traffic (every turn, sub-agent, tool call, streamed
or not); the notebook kernel's own usage still meters, which is why the
counter read low rather than empty.

**Fixed ours-side, no SDK or admin change:** `_child_env` in
`webapp/databricks_launcher.py` now threads the kernel's meter config to the
child as `FUNHOUSE_BUDGET__*` / `FUNHOUSE_SESSION__USER_NAME` /
`CURRENT_USER_NAME` (`FunhouseConfig._load_env_vars` maps `FUNHOUSE_A__B` ->
`a.b`). Five tests in `webapp/tests/test_databricks_launcher.py`, including
that an explicit override wins, a broken kernel config never blocks a launch,
and that it is a no-op off-cluster where no SDK exists.

**Deliberately NOT done, and why.** The `user_name` column may read
`unknown_user` in a subprocess; the report's fix for that pokes a private SDK
attribute. Skipped — the user's identity is in the sqlite PATH
(`/Workspace/Users/<user>/.funhouse_meter`), which is what the admin
cross-user report keys on, so compliance is satisfied without touching
internals. Only worth revisiting if the column itself matters.

**Still theirs, not ours:** the pricing table matches models by exact string
and carries no `funhouse-*` chat aliases, so rows may meter tokens at cost
0.00 — numbers WRONG rather than missing. Raise it once tokens are landing.
Also the SDK's `collect_usage` streaming injection raises TypeError before the
wire (we already route around it).

**Verify on the cluster:** §8 of the diagnosis is a 30-second notebook snippet
contrasting kernel vs child, plus a post-fix check that the money column is
right too.

### 5.13.0 INSTALLED ON THE CLUSTER — what the numpy cascade actually did (2026-09-10)

`%pip install "geotech-staff-engineer==5.13.0"` **succeeds**. No cytriangle, no
waiver, planlens 0.1.0 as designed. App boots and answers.

**The PyNiteFEA numpy watch item fired, exactly as predicted.** numpy went
2.1.3 -> 2.4.6, and because numba 0.61 caps numpy<2.2 it cascaded: numba
0.61 -> 0.67, llvmlite 0.44 -> 0.49. Three Databricks runtime packages are now
unsatisfied IN THAT KERNEL: `tensorflow 2.19` , `mosaicml-streaming 0.12`,
`ydata-profiling 4.16` (all want numpy<2.2; ydata also numba<=0.61). Harmless
unless that kernel needs them. PyNiteFEA is the SOLE cause — nothing else in
core wants numpy above 2.1.3 — so moving it to an extra reverts the whole
cascade in one line if the owner ever wants the runtime left alone.

**numba 0.67 is VERIFIED SAFE, not assumed.** pystrata JITs its wave
propagation core (`propagation.py:67`, `@numba.jit(nopython=True)`), and the
gate had only ever run on numba 0.65.1. Live comparison on the cluster vs the
gate machine, same profile (30 m Vs=180 PI=15 over Vs=760 rock,
`synthetic_long`):

| | cluster (numba 0.67) | local (numba 0.65.1) |
|---|---|---|
| EQL surface PGA / amp / max strain | 0.1736 g / 1.157 / 0.2218% | identical to 4 dp |
| Linear surface PGA / amp | 0.5564 g / 3.709 | identical once the same 2% linear damping is used |

The linear case first looked like a mismatch (3.709 vs 3.988) — it was the
agent substituting `linear, damping=2%` for the requested Darendeli PI=15,
which it disclosed in its answer. Re-running locally with that substitution
reproduces 0.5564/3.709 exactly. Numerics are unchanged across the numba
upgrade.

**Still open from that install:** `httpx2`/`httpcore2` reached the cluster via
`langsmith 0.12.4` (a deepagents dependency), NOT via openai. The `openai<3`
pin was written specifically to keep that fork stack off the cluster and no
longer achieves it. Not currently breaking anything — the gate machine runs the
same stack, and the Prompter's NTLM path still uses the untouched httpx 0.28.1
— but the pin's rationale comment is now wrong about what protects what.

### FIELD FEEDBACK 2026-09-09 — Nairobi SoE: the 401 was the MODEL, not SharePoint

Drop triaged into `module_work/field_feedback/2026-09-09_nairobi-soe_v5.11.2/`
(FINDINGS.md committed, raw/ gitignored). The owner's read was that the agent
"did some search it didn't have credentials for". The evidence says otherwise
and the correction matters: `AuthenticationError` is the **openai** exception
class, every SharePoint tool catches and stringifies its own errors (so none
can kill a turn), and the file was fetched successfully at its true
23,006,108 bytes. The agent's own model call to the Funhouse proxy came back
as an **IIS 401 HTML page**, which `friendly_turn_error` then dumped raw into
the chat — and that raw page is what made a model-proxy failure look like a
SharePoint permissions failure.

Transient, not mis-permissioned: turn 2 succeeded seconds later on the same
credentials. Working hypothesis is NTLM's per-connection auth going stale
during the 23 MB download that preceded the model call.

Two planned fixes, neither landed (mid-release, and neither is verifiable off
the cluster): (1) retry the main model call once on `AuthenticationError` with
a fresh client — there is currently NO retry on that path, so one 401 ends the
turn; (2) give `friendly_turn_error` an auth case that names the failing
surface and strips an HTML body to its title. Backlog: `_render_pdf_page` has
no output-size cap (measured 2.13 MB base64 for a D-size sheet at 220 dpi —
fine here, but unbounded by design).

---

### THE 22 MB UPLOAD LOOP — root-caused and fixed (2026-09-09)

Live on 5.11.2: uploading a 22 MB file put the app into a permanent
websocket reconnect loop, sockets dying every ~14 s (owner's devtools trace:
`stream` 101 at 14.37 / 13.48 / 17.30 / 14.28 s, each followed by the stock
`health` + `host-config` refetch). Everything was stable before the upload --
this is NOT the old ping/TTL family, which 5.11.2 closed.

**Root cause: a component value is WIDGET STATE, and widget state is re-sent
by the browser on every rerun and every reconnect.** Confirmed in Streamlit's
own source, not inferred:

* frontend bundle: `createWidgetStatesMsg(){ let e=new be; return
  this.widgetStates.forEach(t=>e.widgets.push(t)), e }` -- the WHOLE map, no
  delta, no filtering.
* `runtime/app_session.py`: each rerun BackMsg is read as
  `widget_states=client_state.widget_states`.

So the ws uploader (`webapp/ws_upload.py`, the proxy-safe attach path the
Databricks launcher forces via `GEOTECH_UPLOAD_MODE=ws`) leaves the file's
base64 in its component value, and the browser re-uploads it forever. 22 MB
inflates to ~29 MB of base64; that cannot cross the driver proxy inside one
socket lifetime, so the socket dies mid-send, the client reconnects, re-sends
the same 29 MB, and dies again. Unbreakable by design -- the payload rides
the very reconnect that is supposed to recover it.

Note the asymmetry that hid this: the DOWNWARD direction is protected.
`runtime/forward_msg_cache.py` (`populate_hash_if_needed` /
`create_reference_msg` / `global.minCachedMessageSize`) re-sends a large
repeated ForwardMsg as a hash reference, which is why `st.download_button`
with big artifact bytes has never behaved this way. There is no equivalent
cache browser -> server.

**Fix (in the tree, unreleased):** retire the uploader's widget id the moment
its bytes are in hand. `ws_upload.next_upload_key(thread_id, epoch)` +
`ss.upload_epoch`, bumped in `app.py` whenever the ws uploader yields pairs
(including a duplicate filename, which is still 29 MB sitting in state). A
new widget id makes the old one inactive, and the client's
`WidgetStateManager.removeInactive` deletes its state -- so the file crosses
the wire exactly once. Guarded by 6 tests in
`webapp/tests/test_ws_upload.py::TestOneShotWidgetKey`, one of which asserts
the bump still exists in `app.py` (the component cannot be driven through
AppTest, and losing that one line brings the loop straight back).

**Immediate workaround on the running 5.11.2, no release needed:** reload the
app tab. Widget state lives in the page's JS memory, so a reload starts an
empty state map and the loop stops. Until the fix ships, keep ws-mode
attachments small -- the 25 MB cap is about a single crossing, and anything
over roughly 10 MB is re-sent on every interaction.

**Still worth doing later:** chunk the ws upload (send the file in N
messages, reassemble server-side) so the per-file cap stops being a function
of what survives one socket lifetime.

---

### 5.13.0 — the quarantined structural libraries are GONE (2026-09-08)

`%pip install "geotech-staff-engineer==5.12.0"` died on the cluster with a
Nexus **403 "Requested item is quarantined"** on `cytriangle`. That package is
a transitive dep of BOTH `sectionproperties` and `concreteproperties`
(verified from installed metadata -- not concreteproperties-only, which the
pip traceback makes it look like), and both were core deps, so one blocked
wheel failed the WHOLE install: no agent stack, no planlens, nothing.

**Owner's call:** no security waiver is available for a project this size, so
the two libraries were REMOVED and their capability re-implemented natively --
the groundhog precedent from 5.11.2. A 5.12.1 hotfix that merely moved them to
an opt-in extra was built and verified first, but the owner chose to wait for
the rebuild rather than release it (branch `hotfix/5.12.1-structural-extra`
survives as a fallback; it is cut from the v5.12.0 tag so it can still ship
without planlens 0.2.0).

**What was built (all native, numpy/scipy only):**

* `section_props_agent/polygon_props.py` -- exact Green's-theorem integration
  of area, centroid, Ixx/Iyy/Ixy, principal axes, elastic moduli; plastic
  moduli by half-plane clipping + area-halving bisection; native
  self-intersection check replacing shapely.
* `section_props_agent/torsion.py` -- published closed forms (exact St.
  Venant series for rectangles, polar for circular, Bredt for closed boxes,
  El Darwish & Johnston/AISC for I-sections) plus a finite-difference Prandtl
  solve, Richardson-extrapolated, for arbitrary outlines.
* `concrete_props_agent/rc_native.py` -- transformed and cracked sections,
  cracking moment, ultimate capacity by ACI strain compatibility, and an
  N-M interaction diagram sampled uniformly in axial force.

**Verification.** The libraries' outputs were pinned as oracles BEFORE
deletion (`module_work/structural_native/pin_oracles.py` -> `oracles.json`,
23 cases); no library code was copied. Measured against them: every RC scalar
within **0.037%**; all section geometry within **0.5%**; polygon torsion
within **0.5%** (and within 0.013% of the exact series where one exists).
Two deliberate differences are declared in the oracle tests with reasons --
the warping constant is no longer reported for solid rectangles, closed boxes
or arbitrary polygons (no defensible closed form; design practice neglects
it), and the interaction curve differs above 70% of the squash load, which is
above ACI's own 0.80*P0 cap. Structural suites went 58 -> **150 tests** and
got ~17x faster (no meshing).

**Packaging.** `sectionproperties` and `concreteproperties` are gone from
`pyproject.toml`; `[structural]` is an empty alias like every other retired
extra name. **PyNiteFEA stays in core** (owner's call -- it is on no block
list and installs through the proxy fine). Watch one thing on the first
cluster install: PyNite 3.x requires `numpy>=2.4` and the runtime boots with
2.1.3, so pip will upgrade numpy under the kernel -- the failure mode that
SIGTERMed the REPL via Pygments in 5.10.0. If that bites, moving PyNiteFEA to
an extra is the one-line fix.

**Still open:** version is bumped to 5.13.0 in the working tree but NOT
released, and the tree still carries the uncommitted planlens Phase-3.2 work
plus the `planlens[raster]>=0.2` pin -- so the publish order (planlens 0.2.0
first, then the app) still applies. Ask the DT Nexus admins to release
`cytriangle` from quarantine anyway; it is a Cython wrapper around Shewchuk's
Triangle and the flag looks like the usual binary-wheel false positive.

---

### The staging trap (read first)

Six planlens paths MUST enter the same commit or the SOURCE tree breaks --
the tracked-and-modified shims `planlens/ir/tests/{leader,construct}_fixtures.py`
already delegate to them, and the guard that would catch the mistake is
itself in the untracked set:

    planlens/testing/{__init__,leader_fixtures,construct_fixtures}.py
    planlens/tests/{test_packaging,test_readme_claims}.py
    planlens/ir/tests/test_text_bearing_scenes.py

Measured consequence of missing them: planlens 412 -> 369 collected + 4
collection errors; the app repo 58 -> **0 collected**. Also untracked and
wanted: `planlens/ir/{measure,spatial}.py` + their tests (new work, below),
and app-side `module_work/code_review/2026-09-06_planlens_phase32/FINDINGS.md`
+ `module_work/drawing_ground_truth/doc_claims_check.py`.
Do NOT stage the stray `Screenshot 2026-09-07 204229.png` in planlens.

### Publish order: DECOUPLED (owner's call, 2026-09-09)

The app no longer waits on planlens. The pin is back to
`planlens[raster]>=0.1` and **app 5.13.0 publishes alone**, because its two
fixes -- the cytriangle install failure and the 22 MB websocket loop -- are
urgent, unrelated to planlens, and the cluster is broken in both ways today.
planlens 0.2.0 is HELD pending the round-4 repair (see the section above).

Verified before relaxing the pin, so this is not a hope: every planlens symbol
the app imports at RUNTIME exists in the published 0.1.0 wheel (checked by
parsing the wheel, not by installing it), and `planlens.testing` is imported
only by tests, which the wheel does not ship. The single live cost is that DXF
block line-work is absent, which `_dxf_supports_explode`
(`adapters/drawing_ir_adapter.py:118`) already handles by falling back to an
unexploded ingest.

**The follow-up release must restore `planlens[raster]>=0.2`** once 0.2.0
publishes. The comment at that pin in `pyproject.toml` says so too.

The `geotech-references` pointer still moves to `d8ff52e` (1.4.0, already on
PyPI) in the same commit.

### Round 6 — independently verified SHIP (2026-09-11 morning); release sequence in progress

Round 6 (same Fable builder, resumed after a session-limit reset on a tree
verified byte-identical to its freeze) answered every round-5 defect, and a
fresh pass by the same independent verifier measured it: **SHIP.** Reports:
`ROUND6_BUILDER_REPORT.md`, `ROUND6_VERIFICATION.md` beside the ledger.

- **D1 closed the way the tip had it right.** A fill cluster seats at its
  NEAREST MEMBER (was centroid); between the two sound tiers the better-seated
  candidate wins outright, tie -> directional; `_SEAT_DOMINANCE` REMOVED (a
  test asserts its absence). Verifier: cluster keeps its end in all 42
  shift x seat cells; the leader owning the foreign chevron survives with the
  CORRECT reading at every shift. Rationale: at shift 2.0 the cluster is
  0.5 pt off and the chevron 0.94, so any ratio > 1.9 fails and a scale
  threshold would need a 0.01 pt margin — nearer-wins has no constant.
- **D3:** oriented now also requires arrow scale (`_MIN_ARROW_SIZE_SCALE`
  0.5x, one home with the open-3 gate) AND a taper toward the end in the
  shape's own PCA frame. Every rectangle is blunt however turned (closes the
  scale-bar hole); every former "oriented" corpus member was a sub-scale
  fragment. **Oriented corpus population 0/0/0, so the README blunt figure is
  BACK at 19/17/0 = round 4 exactly; round 5's 18/16/0 is superseded.**
  `doc_claims_check.py` now prints the ORIENTED row too; both pinned.
- **D2 became a large win:** cProfile put 64% of the round-5 cost in the
  witness search building ~1,050 rounded dicts per tip to keep 50;
  `_ending_near_from_grid(limit=50)` is proven output-inert (vcheck
  byte-identical) and takes 3001 from 3.6 s to 0.68 s. Verifier's own
  timing vs round 4: -54% / -35% / -10% on the three totals (round 5 was
  +34/+15/+16). Exact prune kept, now bound against the best SOUND seat with
  the cluster's REAL radius (a zero radius had been pruning real clusters at
  shift 1-2 — caught by the verifier's D1 table).
- **D4:** tie-break `(seat, -alignment, d)`, order-independent, pinned.
- **Corpus:** every acceptance number, all 41 residuals, the 322-row called
  set AND the 1662-row leader set byte-identical to 1f6551c. Suites 798 / 61
  / 11. Observational churn vs round 4: +57 at <=0.25 and 15 rows at 0.3, all
  0.45 junk >= 30 pt from truth, per-row accounted in the builder report.
- **R1, the one accepted residual (documented, pinned, not a code change):**
  a stipple splash centred on a shaft end takes the end from a drawn arrow at
  ANY non-zero crookedness (the README had said "> ~9.6 deg"; measured 0.5 deg
  already loses, publishing 94.3 for 100 at 0.947). Corpus-inert, identical to
  the tip, a regression vs round 4 only for 0.3-9.6 deg arrows. Shipped as-is
  because a splash on the end IS a real stipple arrowhead's anatomy and the
  scene needs two terminators at one end; README sentence corrected and the
  0.5/1.0/2.4/5.0 deg rows pinned as the accepted residual
  (`TestTheAcceptedResidualAtItsRealThreshold`). Two low observations added to
  the README sharp edges (size floor is a sheet statistic; mirror-image ties
  resolve by grid order).
- Findings 5 and 6 remain documented, not fixed.

Final tree: `queries.py` md5 `e2b2b3de5820fe12f8400348108e098b` (the verdict's
file, unchanged through the doc pass); README `83685e33…`,
`test_end_ownership.py` `c40e863c…`. Release sequence from here: lead's own
planlens full suite + app drawing tests (61 green) -> commit planlens (six
staging-trap paths in ONE commit, screenshot excluded) -> tag v0.2.0 -> push
(trusted-publisher workflow) -> app: restore `planlens[raster]>=0.2`, version
bump, final gate, commit, tag, push.

### Round 5 (the repair) was independently verified: DO NOT SHIP as-is — round 6 in flight (2026-09-10 evening)

Builder (Fable, fresh) implemented Change A + Change B; a fresh Fable verifier
then measured the frozen tree (`queries.py` md5 `c0c67b34…`). Both reports are
committed beside the ledger: `ROUND5_BUILDER_REPORT.md`,
`ROUND5_VERIFICATION.md`, `round5_attack.py` (three-tree probes) in
`module_work/code_review/2026-09-06_planlens_phase32/`.

**What verified sound (RUN by the verifier):** every headline number
reproduces; findings 1, 3, 4 fixed on the fixtures; the restored pre-prune is
proven exact (triangle inequality) and measured inert on all ten sheets; 41
residuals and the 322-row called set byte-identical; suites 724 / 61 green.

**D1 — BLOCKER.** Finding 2 (the cluster anatomy, the highest-risk line) is
fixed only inside a 0.05-0.4 pt window. The 10x seat-dominance ratio needs
`seat_cluster*10 < seat_chevron`; real clusters seat 0.5-2 pt and an attached
foreign chevron seats up to ~4.6 pt, so at every realistic offset the foreign
chevron still takes the end and `exclude_dimensions` still deletes the owning
leader. Tip 1f6551c gets every shift right; round 4 none; the repair moved the
boundary from "never" to "sub-half-point". Corpus-inert — the corpus has no
such case, the same non-evidence three rounds already recorded. The builder
had flagged this exact judgement call (ratio vs absolute) as unsure; the
measurement settled it. Fix shape (verifier + lead agree): a cluster's seat
should be its REACH on the ray (`_reach_on_ray` exists), i.e. 0.0 whenever the
end lies inside the splash's axial extent, so real clusters always win and
round 4's splash-5-pt-off case stays capped.

**D3 — policy.** The "oriented" rank's real corpus population is 6/6 sub-2-pt
SHX glyph fragments, 0/6 terminators; with other rungs absent it CALLS a
[true arrow + 0.8 pt oblong fragment] at 1.0 and a graphic SCALE BAR with 2:1
end blocks as a dimension at 0.944 (tip: no proposal). Decision: oriented must
be commensurate with the arrowhead scale; scale-bar anatomy pinned as not a
dimension; fallback is clamp-only with all tipless capped.

**D2 cost** (+34% @0.0, +15-16% at the default, driven by 3001; 4.3 s absolute
max — the builder never timed the shipped tree). **D4** entity-order tie
(2.5 pt swing). **D5** docs: "+60 from the prune" wrong (60 removed / 120
added by seat ordering un-colliding the DISTINCT rule), "6 rows" is 8, the
"six oriented" README figure has no command.

Round 6 routed to the builder with all of the above and the verifier's drop-in
pytest; verifier re-runs after. Ledger correction applied: the two stale "316"
test counts now read 350.

### The remediation train: the caveat is CLOSED, and it found two regressions

The round-4 adversarial pass was run (2026-09-09, independent verifier, no
build context). Full report + runnable repro fixtures for all three trees:
`module_work/code_review/2026-09-06_planlens_phase32/ROUND4_VERIFICATION.md`
and `round4_repro.py` alongside it.

**Verdict: DO NOT SHIP as 0.2.0** until findings 1 and 2 are fixed.

What held up: **every measured number in the ledger reproduces** (632 / 58 /
25/25 / 16/16 / 23/25 / FPs 2·10·6·19·23·20·38 / precision 5·10·12·10 / blunt
19·17·0 / layer-0 557·158·1696), no README figure disagrees with
`doc_claims_check.py`, and — the strongest evidence for the tree, which the
ledger never claimed — **all 41 matched corpus defpoint residuals are identical
to `1f6551c` to three decimals.** On the corpus the remediation is provably
inert.

What broke, in six-line fixtures, on terminator styles the corpus does not
contain:

1. **Box-like (diamond / trapezoid) terminators**: `_arrow_geometry` computes
   the apex correctly and the blunt branch then discards it for a centroid
   projection — length 100.0 -> **94.0**, confidence 0.952 -> **0.45**, never
   called. Clean on v0.1.0 AND on the tip: a regression from the uncommitted
   work.
2. **A fill-cluster arrowhead loses its end to a farther foreign chevron** (the
   cluster is 0.04 pt away, the chevron 3.24 pt) because distance is only
   consulted *within* a tier. No cap, no flag, confidence stays 0.944 — and the
   foreign arrowhead lands in `arrowhead_ids`, so `exclude_dimensions=True`
   deletes whichever leader owns it. That is the arbitration steal round 4's
   headline claims to have closed, reached by a route its fixture does not
   cover. **On a corpus whose real arrowheads ARE clusters, this is the
   highest-risk line in round 4.** Also clean on 0.1.0 and the tip.

Findings 3-5 are regressions vs the published 0.1.0 but pre-existing at the
tip; 6 is pre-existing everywhere (a concave dart is deleted at every
threshold — document it, do not fix it); 7-10 are docs/staging, minutes each.
Finding 7: `"632 passed (from 316)"` — the tip actually collects **350**.

**Repair is TWO changes, not four (~1.5 days):**
- **Change A — finding 1.** Independent and corpus-inert (0 blunt proposals at
  0.5). One-line coordinate substitution at `queries.py:1275` using the
  primitive the cluster leg already has (`_reach_on_ray`, which returns
  `(100,0)` exactly on the diamond), plus a policy call on the unconditional
  0.45 blunt cap. Half a day. Land it first.
- **Change B — findings 2, 3 and 4 together.** Three halves of one question:
  who wins the end (2), what a win costs (4), whether the winner's coordinate
  is publishable (3). They pull against each other — repairing 3 makes 4 worse;
  repairing 4 re-opens the steal unless 2's distance-dominance rule lands with
  it. Four sites: `_end_tier` `:1360-1366`, the per-end comparison
  `:2111-2130`, the continuous apex publication `:2227`, the cap ladder
  `:2264-2272`. A day, with fixtures pinning BOTH directions plus a corpus and
  residual re-run.

**The structural lesson, which outlives these fixes:** the corpus cannot
falsify a change to a terminator style it does not contain, and its 18 pt match
tolerance would hide a 9 pt error. Five of the six code findings live in
exactly that blind spot. Any future gate on this code needs synthetic
terminator fixtures alongside the corpus — `round4_repro.py` is the start of
that set.

### Published numbers now have a command

Two rounds running, prose was edited with figures no run reproduced. The fix:
`module_work/drawing_ground_truth/doc_claims_check.py` regenerates every
corpus figure the planlens docs publish, and ten guards in
`planlens/tests/test_readme_claims.py` pin them. **If a number disagrees with
that script, the DOCUMENT is wrong.** Current measured values: blunt
terminators fire on 19 proposals @0.0, 17 @0.3, **0 @0.5** (only sheets 11.01
and 21.01); layer-0 inheritance re-homes 557/158/1696 entities on
10.17a/11.01/5003 -- and note the `n_layers` METADATA field counts something
different and does NOT follow the inheritance.

### NEW DIRECTION (owner correction, approved plan)

Recognition of CAD constructs was never the goal -- the goal is engineers
reviewing design and construction documents, and an uploaded artifact is
usually a REPORT (thirty pages of narrative, then tables, then figures), not
a sheet. The user must not have to say whether they want a language review or
a visual review.

**Plan of record: `C:/Users/socon/.claude/plans/delightful-swinging-sky.md`.**
Driving example: *"what is the average spacing of the borings in this plan?"*
over a Langan Subsurface Investigation Plan. Surveyed verdict: today that
fails as a plausible wrong number rather than a refusal.

Owner decisions taken: source documents are **PDF nearly always**; build the
**vertical slice** (one question end to end); output shape is "whatever the
reviewer asked for", so findings are data, not a fixed deliverable.

Started and green (new, untracked): `planlens/ir/measure.py` (the `Quantity`
envelope -- units mandatory, confidence composes by `min` not product, and a
page-point value REFUSES to become feet without a resolved scale) and
`planlens/ir/spatial.py` (point-pattern maths, numpy only, no scipy). 37
tests. Notable measured result: on a perfectly regular 3x3 grid the three
spacing conventions differ by **2.45x**, which is why the API deliberately
has no key named `average_spacing` at any depth.

**Blocked on the owner:** the Langan sheet is in NEITHER repo. Real ground
truth needs it plus 2-4 more subsurface investigation plans.

Next in the plan's own order: the two ingest representation fixes (PDF fill
is absent from the IR entirely, and multi-subpath drawings are fused into one
polyline) -- both are load-bearing for symbol work and both move every
published corpus number, so they are a re-baselining milestone, not a patch.

### Still open from before

Owner's live 5.12.0 cluster shakedown; SoM A/B live run (recipe in
`som_ab_check.py` docstring); planlens submodule wiring in the app repo;
TinyApps onboarding; cluster OCR needs the planlens README headless recipe;
master CI Tests should have self-healed now planlens is on PyPI (VERIFY).

## 0a-prev-c. PICKUP LIST (2026-09-05) [HISTORICAL]

**5.12.0 RELEASED TO PYPI 2026-09-05, VERIFIED LIVE** (with planlens
0.1.0 + geotech-references 1.4.0, both first published the same hour;
full gate 11,468/0 beforehand). Cluster install:
`%pip install "geotech-staff-engineer==5.12.0"` — the launcher
baseUrlPath fix ships, so NO notebook patch cell. Release gotcha for
the future: PyPI enforces a 512-char cap on the pyproject description
SERVER-SIDE only (twine check passes longer) — the 5.12.0 publish
400'd twice on a 513-char description. OPEN after release: owner's
live 5.12.0 cluster shakedown; SoM A/B live run (Funhouse recipe in
som_ab_check.py docstring); drawing Phase 3.2 list (design memo
banner); planlens submodule wiring; master CI Tests should self-heal
now that planlens is on PyPI (was red only for that reason — VERIFY);
TinyApps onboarding; cluster OCR needs the planlens README headless
recipe.

## 0a-prev-b. PICKUP LIST (2026-09-04 EVENING) [HISTORICAL]

Current release: **5.11.2** (PyPI; refs 1.3.3). Master carries the big
UNRELEASED 5.12.0-candidate train: security hardening, geophysics
excision, structural stack, drawing Phases 1+2 (verified ship-as-is),
ground-truth harvest, the **planlens split**, and the launcher
baseUrlPath fix. CLAUDE.md "Post-5.11.2" + "PLANLENS SPLIT" = summary.

DONE TODAY (2026-09-04): drawing Phase 2 COMPLETE + independently
verified; Mecklenburg ground truth harvested (10 DWG+PDF pairs + truth
JSON; SHX no-text-layer finding); 5.11.2 live shakedown PASSED after
the "Not Found" root cause (launcher baseUrlPath vs prefix-stripping
proxy — fixed b6683a6; live via notebook template patch) — heartbeat
holds, uploads work; **planlens split executed** (owner-named).

1. **planlens finish-out:** owner creates
   github.com/soconnell345-geotech/planlens → push the local repo
   (C:/Users/socon/OneDrive/dev/planlens, branch main) → `git submodule
   add https://github.com/soconnell345-geotech/planlens.git planlens`
   in the app repo (refs pattern). **5.12.0 CANNOT ship until planlens
   0.1.0 is on PyPI** (app now depends on planlens[raster]>=0.1;
   publish owner-gated). Dev environments: `pip install -e ../planlens`
   (already done in .venv).
2. **Drawing Phase 3 (owner GO 2026-09-04, build in planlens):**
   fill-cluster arrowhead detection (micro-dot fills — THE real-sheet
   leader/dimension recall gap) + B7 raster/OCR leg (REQUIRED even for
   vector PDFs: SHX sheets have no text layer); then B3 set-of-marks
   A/B + B1 DXF-native LEADER/DIMENSION ingest + set-level IR caching.
   Ground truth stays app-side: module_work/drawing_ground_truth/.
3. **Next release (5.12.0)** — owner decision 2026-09-04: HOLD until
   Phase 3 is built and tested. Order: planlens 0.1.0 to PyPI → refs
   release (1.4.0?) + pin bump → 5.12.0. Until then the cluster needs
   the notebook baseUrlPath patch cell each fresh kernel.
4. **TinyApps onboarding** (tinyapps/TINYAPPS.md): MOU → GitHub license
   → wrapper repo; planlens is the pitch's banner app (OBO branding at
   app-setup time, NOT in the package).
5. **FEMA P-2192**: fema.gov 502s persist — owner browser fallback;
   it's a worked-examples corpus play (discuss usage before building).
6. **FIELD FEEDBACK — ALL SIX FIXED 2026-09-05 (owner ordered
   execution; commits + details in the FINDINGS.md disposition table:
   module_work/field_feedback/2026-09-04_praia-downdrag_v5.11.2/).
   Ships with 5.12.0. Original items for reference:**
   (a) false "sources unavailable" in calc reports — calc sub-agent
       isolation likely drops working-folder visibility (TRUST issue);
   (b) subsurface `profile_figure` tool (matplotlib schematic: layers +
       GWT + pile/wall overlay → PNG) + prompt default "calc packages
       include a profile viz" + html_to_pdf fails LOUDLY on
       un-embeddable figures (model shipped "[image]" placeholders and
       a color-table fake figure);
   (c) CGPR #56 downdrag method family — downdrag.py has ONLY Fellenius
       neutral plane; implement the report's other methods w/ citations
       (CHECK CGPR distribution terms before any digitization);
   (d) conversations-tab rename/delete icons lost (streamlit 1.63
       glyph regression — repro, fix, consider version cap);
   (e) attachment awareness — model disclaims "can't attach in chat"
       while writing into the conversation files dir (which IS the
       attach surface): prompt + tool-result note;
   (f) SharePoint mirror folders named by custom conversation name +
       date stamp instead of thread-id hex.
7. Old backlog unchanged: axial_pile cohesive_phi trap; SCDOT
   onboarding; 108-Q eval rerun; auto-continue taxonomy; λ-method;
   Highter-Anders; 6.0 restructure (dedicated session); foundry/
   cleanup; Monday drift canary; pandas-2.x DT-sweep watch; deferred
   structural: pelicun + USGS design-maps route (FUTURE_IDEAS).

## 0a-prev. PICKUP LIST (2026-09-03) [HISTORICAL]

Current release: **5.11.2** (websocket saga CLOSED + upload workarounds +
detached turns + groundhog removal; refs 1.3.3). **TINYAPPS PILOT AWARDED
2026-09-03** — the strategic hosting path is live; plan + wrapper-repo
skeleton + office-hours question list = `tinyapps/TINYAPPS.md`. The
Funhouse/Databricks app stays as fast tester + backup (owner directive).

**The websocket saga, final ledger (do not relitigate):** three stacked
causes, all fixed — (1) the proxy swallows ws ping/pong control frames;
(2) streamlit's server pings 30 s / hangs up 60 s (code 1011) when pongs
never return; (3) the launcher's `bootstrap.run(flag_options)` NEVER
APPLIED flags — `load_config_options()` was the missing call (76e21ff), so
the 5.11.1 ping fix was inert and EVERY launcher flag had been ignored
since birth. Lab probe rig: aiohttp `ws_connect(autoping=False)` against
`/_stcore/stream` (session tmp scripts; recipe in CLAUDE.md). Plus
`webapp/turn_jobs.py`: turns now run in a socket-independent worker and
reattach across reconnects — belt and braces. "Hard proxy TTL" was an
interim theory, RETIRED.

1. **TinyApps onboarding (owner + agents):** MOU → GitHub Enterprise
   license → init wrapper repo (thin: app.py/packages.txt/run.sh from
   tinyapps/wrapper_repo/, package via Nexus). Blockers to resolve at
   office hours: key-auth Prompter client, SharePoint-from-App-Service
   auth, boundary/CAB status. `webapp/tinyapps_entry.py` stub exists;
   engine wiring lands after answer #1.
2. **5.11.2 live shakedown on Funhouse:** the 60 s flap should now be
   GONE (probe: PING should never arrive). Then: ws uploader, diagnostics
   upload-probe verdict, model picker, budget line, email_file,
   SharePoint link, streaming opt-in.
3. **Old backlog (unchanged):** axial_pile beta cohesive_phi trap; SCDOT
   examples onboarding; GPT-5.x 108-Q eval rerun; auto-continue taxonomy
   port; Document Intelligence bake-off; λ-method; Highter-Anders;
   6.0 restructure (dedicated session); foundry/ cleanup; Monday drift
   canary reports; pandas-2.x DT-sweep watch.

Standing rules: releases/tags ONLY on owner word (v* tag auto-publishes);
additive default-preserving changes; validate vs published values; never
unpinned installs on the cluster; owner is a practicing engineer, not a
developer — actionable results, no lectures; use Fable subagents freely
for build work (owner directive, Max plan).

## 0a-prev. PICKUP LIST (2026-09-01) [HISTORICAL]

Current release: **5.11.0** (PyPI, verified live; refs 1.3.3). Master ==
released (`ae126e8`). Gate green (1,778 on the release chunk; full gate needs
detached >10-min chunks). CLAUDE.md release sections + auto-memory carry the
narrative; `module_work/FUTURE_IDEAS.md` holds the **Funhouse SDK survey
ledger** (9 ranked findings — streaming landmines, Tiny Apps, auto-continue
taxonomy, Document Intelligence, environment facts) and the forward specs.

**OWNER PIVOT (2026-07-30): Foundry is RETIRED** (their Palantir team shut it
down) — **Databricks/Funhouse is THE deployment target.** Ignore the
Foundry-401 items in the old §0a; Foundry code ships dormant/parked.

1. **5.11.0 live shakedown (owner's next cluster session):** pinned install
   `%pip install "geotech-staff-engineer==5.11.0"` (no extra pins — openai<3
   is in core; mirror lags PyPI 1–2 days). Verify: heartbeat survives
   follow-up questions (the "Connecting" flap), SharePoint sidebar link now
   clickable, model picker (funhouse-gpt-medium), budget line, email_file,
   opt-in streaming (`GEOTECH_PROMPTER_STREAMING=1`), and the localhost PUT
   probe for the still-open browser-upload 403 (SharePoint-fetch = fallback).
2. **Funhouse office hours follow-ups (owner):** Tiny Apps intake status
   (form submitted >1 month ago; Tiny Apps = sanctioned durable tier,
   SUPPORTS Streamlit), the vanished `Tiny App Development/` example folder,
   streaming/metering blessing, 30-min cluster idle timeout.
3. **Monday drift canary:** `geotech-drift-canary` cloud routine (Mondays
   10:00 UTC) force-upgrades the agent stack past the pyproject caps in a
   scratch venv and runs the deep+webapp gate — read its report before
   raising any cap.
4. **axial_pile beta cohesive_phi trap** — still item #1 in the ledger's
   Das-sweep ergonomics section (per-layer fix, global fallback).
5. **Sample-calc detector next targets:** SCDOT design examples, USACE EM
   appendices, more NHI manuals (doctrine in FUTURE_IDEAS header; ledger =
   module_work/wiki_verification/TIER_A_LEDGER.md).
6. **Eval:** GPT-5.x live rerun of the 108-Q suite (`eval_harness --ids PAV`
   for the pavement subset).
7. **Parked/major:** 6.0.0 restructure (module_work/V6.0_RESTRUCTURE_PLAN.md,
   dedicated session); auto-continue gap-taxonomy port; Document Intelligence
   OCR bake-off (needs live tenant); λ-method; Highter-Anders; foundry/ dir
   cleanup (NOT quick — 9 test suites import it).

Standing rules (unchanged, restated in the old §0a below): releases/tags ONLY
on the owner's word (v* tag push auto-publishes); additive default-preserving
changes; validate against published values, never tune; never run unpinned
installs on the cluster; owner is a practicing geotech engineer, not a
developer — actionable results, no git lectures.

## 0a. FOR THE NEXT AGENT (written 2026-07-20 at the Fable→Opus handoff) — prioritized pickup list [HISTORICAL — Foundry items superseded]

Current release: **5.9.1** (PyPI; refs 1.3.3). Master == released. Full gate
green (~10k tests). Auto-memory carries the running narrative; CLAUDE.md
release sections carry per-release detail; `module_work/wiki_verification/
TIER_A_LEDGER.md` is the verification/defect master ledger.

1. **Foundry 401 follow-up (owner-blocking):** the published app awaits the
   enclave admins' answer on LLM-proxy access (ticket text in docs/FOUNDRY.md
   troubleshooting). When it arrives, the fix is likely ONE
   `GEOTECH_FOUNDRY_HOST` line in the owner's app file. Then: upgrade app to
   5.9.1 (`maestro env pip install` → Publish and sync), run the sidebar
   Connection diagnostics, then the pavement shakedown prompts.
2. **axial_pile beta cohesive_phi trap** (nearest-to-defect ergonomics item,
   +12.4% naive-user error; doc claims tip-only, behavior is all cohesive
   layers) — fix per-layer, keep global as fallback; details in the ledger's
   Das-sweep section.
3. **Continue the sample-calc detector** (owner standing directive; doctrine +
   next targets in module_work/FUTURE_IDEAS.md header): SCDOT design examples
   next; then USACE EM appendices; more NHI manuals live in the owner's
   OneDrive Lib (search the WikiLLM at C:\Users\socon\OneDrive\Lib\_WikiLLM,
   library.db — copy WITHOUT its stale journal to read).
4. **Ergonomics backlog items 2-6** (ledger): Schmertmann gamma_soil +
   Iz subdivision; anchored_wall embedment-grid refinement + waterfront mode;
   two-gamma GWT bearing; Highter-Anders option; lambda method.
5. **Eval:** owner's GPT-5.x live rerun of the 108-Q suite (PAV questions
   new; `eval_harness --ids PAV` for the subset); triage vs the 68 keys.
6. **Parked/major:** 6.0.0 single-namespace restructure — now FULLY PLANNED
   with a measured surface (2,795 imports/517 files + enumerated string/path
   traps) and an owner-approved step-by-step:
   **module_work/V6.0_RESTRUCTURE_PLAN.md** (one dedicated session); worked-examples phase-2 (owner's firm reports, private);
   playbooks + recompute-from-report QC specs; Databricks launcher
   live-verify; foundry/ dir cleanup (NOT quick — 9 test suites import it).

Standing rules that must survive the handoff: releases/tags ONLY on the
owner's word (a v* tag push auto-publishes via OIDC); additive
default-preserving changes; validate against published values, never tune;
verify against the printed page before changing chart-derived code; ONE
agent per wave (parallel fan-outs burn the session limit); owner is a
practicing geotech engineer, not a developer — actionable results, no git
lectures; wiki/Das content = internal anchors only, never shipped.

## 0. Delta since the table below (2026-07-15 → 07-18; table not yet rewritten)

Everything here is also in CLAUDE.md's release sections (read those) and the
auto-memory. Headlines:

- **Releases 5.7.0 → 5.8.0 → 5.8.1 → 5.8.2** (all owner-OK'd, all on PyPI, tags
  `v5.7.0`..`v5.8.2`; refs 1.3.2 + 1.3.3 released too): the complete pavement
  stack (AASHTO 1993 module + UFC 3-250-01 alternative method + pavement
  specialist agent), then two same-night Foundry-deployment patch releases
  (custom-RID clobber fix; Connection-diagnostics panel + persistent errors +
  `max_completion_tokens` for GPT-5 RIDs; extras folded into the core install —
  plain `pip install geotech-staff-engineer` now brings everything).
- **Owner's Foundry app IS PUBLISHED and running** (State gov enclave,
  stateobo.palantirgov.com, PDCS Sandbox / geotechStaffChatbot) but **blocked on
  a 401 from the LLM proxy = enrollment/permissions**, NOT our code. Full field
  notes + the admin ticket text: `docs/FOUNDRY.md` troubleshooting section. When
  the admin answers, the fix is likely one `GEOTECH_FOUNDRY_HOST` line in the
  owner's app file.
- **UNRELEASED on master (5.9.0 candidate, owner-gated):** the
  `worked_examples` system — 17 verified-by-execution exemplar calculations
  from real published design reports (`funhouse_agent/worked_examples.json`,
  adapter `find_worked_examples`/`get_worked_example`, prompt wiring, gate
  tests); webapp/tests added to the pytest gate; FOUNDRY.md troubleshooting;
  `module_work/FUTURE_IDEAS.md` (8 forward specs incl. phase-2 harvesting of
  the owner's own reports, playbooks, recompute-from-report QC).
- **Eval suite is now 108 questions / 68 keyed** (PAV-1..PAV-8 added, ground
  truth run on v5.8.0) and `eval_harness --ids PAV` runs a subset.
- **Do NOT casually delete `foundry/`** — attempted 2026-07-18, reverted: 9
  agent-wrapper test suites import it throughout (scope in CLAUDE.md).
- Gate at handoff: 9,169 + 844 (slope/fem2d) + wrapper suites green, ~9,950
  tests + 48-ish skips.

---

## 1. One-screen status

| Item | Value |
|------|-------|
| Repo | github.com/soconnell345-geotech/GeotechStaffEngineer (private) |
| **master HEAD** | **v5.6.0 release** (2026-07-15, owner OK'd) = 5.5.2 + the app A-workstream (A1–A8, `module_work/APP_PLAN.md`) + html_to_pdf + retaining_walls base-interface overrides (double-2/3 verdict) + the Foundry deployment path (`docs/FOUNDRY.md`). Prior tags: `b81ddaa` = v5.5.2 | — PUBLISHED to PyPI 2026-07-13. Line: 5.5.0 (post-5.4.1 train: correlated pairs, Bray-Travasarou, Lowe-Karafiath, aniso su, slope_report_package, inline Plotly) → 5.5.1 (provenance-audit doc fixes, persistent conversations, model picker, save hardening) → 5.5.2 (Databricks Prompter launcher run_on_databricks, eval-suite fixes: self-contained questions + DIR-1 path + discoverability aliases, fem2d schema backfill + positive-depth guards). Eval: owner ran the 100-Q suite live (GPT-5.1 driver, docs/geotech_eval_20260713.json) — 72% graded pass, 0 exceptions, all misses triaged (2 real fem2d issues FIXED, 1 suite bug FIXED, 5 questions rewritten, rest = weak-model behavior) |
| **Branch `v5.4`** | tip `eb38615`, PUSHED, fully MERGED into master (`aad984e`); stays checked out in the worktree for the next train |
| Submodule `geotech-references` | `3b25e0e` = **v1.3.1 on PyPI** (owner OK'd 2026-07-08; ufc_expansive figures complete, 42/42 page-accurate); parent pin `>=1.3.1` |
| Version string | `5.6.0` in `pyproject.toml` (master) |
| Validation suite | `validation_examples/` — 191+ passed (offline; V-001..V-054) |
| Full repo suite | **8557 passed / 48 skipped** (5.5.2 release gate, 2026-07-13) |
| **Publish status** | 5.5.2 + refs 1.3.1 PUBLISHED. Owner-gated next: GPT-5.4 eval rerun (quota); TinyApp hosting feedback (form submitted 2026-07-10) → fold env answers into webapp/; APP WORKSTREAM (plan of record: module_work/APP_PLAN.md, A1-A8); reference-wiki integration PARKED (owner 2026-07-13). Deferred: toe-circle search under-sampling, steep-phi Kc sensitivity, E2 default stays fellenius (owner). |

**⚠️ Release gate (still applies):** a `v*` git tag push **auto-publishes to
PyPI** via `.github/workflows/publish.yml` (OIDC trusted publishing). Do **not**
push a tag, bump the version, or merge `v5.4` to master without the owner's
explicit OK.

---

## 1b. Post-5.5.2 app workstream (2026-07-13/14, on master, UNRELEASED)

Owner pivoted to app-heavy work (wiki integration PARKED). **All A-items built +
deployed locally** — plan of record + close-out: `module_work/APP_PLAN.md`
(owner copy `APP_PLAN_copy.md` at repo root). Headlines: calc sub-agent context
isolation (default ON, −84% measured, `module_work/A2_CONTEXT_DESIGN.md`),
crash-proof turn persistence, Agent/Analysis-depth/model pickers, per-conversation
working folder (`GEOTECH_DEFAULT_OUTPUT_DIR`), optional tracing (`GEOTECH_TRACE=1`
+ LangSmith envs, webapp/README §Tracing), industry memo
(`module_work/A7_INDUSTRY_MEMO.md`). Post-A owner-session fixes: bounded
auto-continue for mid-turn "Let me…" stops, download-button MIME types
(.md-as-.bin), recursion-cap visibility. Owner decisions: summarization backstop
SKIPPED; durable checkpointer PARKED; API thinking layer DEFERRED (per-model
gating needed); "Analysis depth" naming reserved "Thinking" for future API
control. Open backlog (next cycle): retaining_walls sliding-check convention vs
free-body discrepancy (owner wall session); CI mock eval subset. Webapp tests:
105 (`pytest webapp/tests -q`). Launch: `"Start Geotech App.bat"` /
`streamlit run webapp\app.py --server.headless true`.

---

## 2. What's on master (done + durable)

Chronological, all merged and pushed. Full detail: `docs/V5.1_SUMMARY.html`,
the per-module `DESIGN.md`/`VALIDATION.md`/`UPGRADE_PLAN.md`, and the memory
note `project_le_fem_modernization` (auto-loaded).

- **v5.1 line** (the bulk of the work): v5.1 TODO sweep; LE+FEM modernization
  (rigorous GLE/Morgenstern-Price `slope_stability`; T6 + 3D-MC-return + GL99
  SRM `fem2d`); the `reliability/` module (FOSM/PEM/MC/FORM + COV database);
  calc-package figures/tables + plotly viewers (`calc_package/interactive.py`);
  the staged model-setup agent (`geo_project/` + `deep/setup_agent.py`, OFF by
  default). Intentional behavior changes vs 5.0 are listed in the summary page.
- **Post-v5.0 field-failure fixes** (from a real Funhouse session): the
  lateral-pile calc-package bug (package re-ran the analysis; an invented
  `E_GPa` was silently dropped → steel-default; the deep agent's virtual FS
  couldn't see real files) → adapter-ergonomics sweep across ~30 adapters
  (`require_params`/`reject_unknown_params`, documented params, self-verifying
  file writes); and the **Databricks `/Workspace` placeholder-write** fix
  (`funhouse_agent/_fileio.py`: content-verified writes + rescue-to-`/tmp`).
- **Phase E — published-example validation (DONE):** 25 worked examples from
  GEC/Caltrans/FLAC implemented as **87 offline pytest checks**
  (`validation_examples/test_published_v0*.py` + `RESULTS.md`). **Zero
  analysis-result bugs** — every discrepancy resolved to units / method-variant
  / convention. One additive `fem2d` capability gained en route (`roller_base`
  BC + `initial_stress_relaxation` for excavation/cavity unloading; 369 fem2d
  tests, 0 regressions).
- **v5.2 coverage Batch 1 (DONE):** four additive, strictly default-preserving
  capability adds, each validated against its published example and flipping a
  `RESULTS.md` row to PASS — see `module_work/V5.2_COVERAGE.md`:
  - Q1 `settlement/hough.py` — Hough granular (C′-index) settlement.
  - Q2 `pile_group.meyerhof_group_settlement` — Meyerhof (1976) SPT group settlement.
  - Q3 `axial_pile` — per-layer `toe_friction_angle` + `head_depth` offset.
  - Q4 `retaining_walls.mse` — steel bar-mat/welded-grid Kr (2.5→1.2) + F* curves.
- **5.2.0 released 2026-07-05** (with geotech-references 1.3.0; 5.1.0 never
  shipped) after the rc5 71-Q eval review (archived
  `docs/geotech_eval_20260705.json`/`.md`; fixes: drilled_shaft per-layer
  breakdown, +6 dispatch aliases, P1 recovered-split, optional-dep preflight).
- **v5.3 train (released 2026-07-06 as 5.3.0):** Batch-2 coverage 5/5
  (drilled_shaft rational GEC-10 chains; MSE LRFD external-stability CDRs; soe
  basal-heave-sidewall-shear + FHWA apparent-pressure anchored + log-spiral
  Caquot-Kerisel Kp; full Reese-1974 sand p-y; fem2d monolithic Taylor-Hood
  u-p Biot consolidation); slope_stability round 2 (15 new Slide2/ACADS/Duncan
  validation problems V-026..V-040; SS-6 noncircular-search robustness fix +
  rejection diagnostics; rapid drawdown 2/3-stage; Newmark + Jibson; infinite
  slope; Ito-Matsui piles verified vs the ORIGINAL 1975 paper); pdf_import
  round 2 (scale calibration, label→region, cleanup, vision grid overlay,
  vision↔vector cross-check); + **12 adversarial-review fixes** (headline:
  log-spiral Kp δ=0 Rankine anchor — the clamp was ~44% unconservative).
  Plans: `module_work/V5.3_PLAN.md` (+ its review-outcome section).

---

## 3. What's on branch `v5.4` (BUILT + gated, awaiting owner release call)

All six owner directives (2026-07-06) are DONE on `v5.4` (13 commits over
master; final gate 8279 passed / 48 skipped). Plan of record + per-item log:
**`module_work/V5.4_PLAN.md`** (read it before continuing v5.4 work).

- **D1 PDF user manual** — `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf`
  (132 pp) + regenerable `docs/user_manual/build_manual.py`; problem catalog is
  auto-generated from MODULE_REGISTRY/METHOD_INFO so it can't drift from code.
- **D2 no-restart Databricks** — `funhouse_agent/runtime_check.py` hot-reloads
  a stale pre-imported `typing_extensions` before any langchain import;
  `dbutils.library.restartPython()` is now only the documented fallback;
  cluster-scoped install documented as the avoid-entirely alternative.
- **D3 layered disclaimers** — DISCLAIMER.md (ships in wheel), prominent README/
  PyPI section, ONE-TIME first-import stderr notice (marker
  `~/.geotech_staff_engineer/disclaimer_ack`; suppress `GEOTECH_NO_DISCLAIMER=1`;
  silent under pytest), `geotech-disclaimer` console script, and a standing
  basis-&-limitations block in both calc-package templates. Honest constraint:
  pip runs NO code on wheel install — these are the legitimate equivalents.
- **D4 visualization gallery** — `docs/gallery/index.html` + `build_gallery.py`:
  12 exhibits, every figure from a REAL validated run (ACADS search, drawdown
  3-stage, Newmark polarity, p-y families, GL99 SRM, V-023 consolidation,
  Duncan reliability, GEC-11 MSE CDRs, bearing/settlement, bearing graph,
  pdf_import demo).
- **D5 `drawing_ir/` module** — LLM-ready drawing digitization: unified IR
  (Line/Polyline/Arc/Circle/Text/Region w/ coords, layer/color, provenance
  dxf|pdf_vector|raster_trace, per-entity confidence), OpenCV raster leg
  (`[raster]` extra), and an agent query surface (digitize_drawing → handle;
  query_drawing: entities_in_bbox / lines_by_angle / text_near /
  candidate_ground_surface(proposal) / …; get_entities). Deterministic
  extractor owns coordinates; the LLM asks for slices. Deferred follow-up:
  geo_project ingestion wiring (flagged in drawing_ir/DESIGN.md).
- **D6 seismic reviewer** (first narrow reviewer) — shared checklist in
  `funhouse_agent/review_checklists.py`; surfaces: `.claude/agents/
  seismic-reviewer.md` (Claude Code) and `funhouse_agent.make_seismic_reviewer
  (engine)` / `make_seismic_reviewer_deep(model)` (Funhouse; scope = 10 seismic
  modules + 7 seismic references via allowed_agents). Template for the
  reviewer-family rollout (V5.4_PLAN F8).

**NEXT:**
1. **Owner Funhouse feedback on 5.3** (and optionally the v5.4 pieces). On the
   owner's OK: merge `v5.4`→master, bump 5.4.0, tag (auto-publishes).
2. **E1–E11 QC carryovers + F1–F8 creative builds** — all scoped in
   `module_work/V5.4_PLAN.md` (rapid-drawdown search wrapper, #96 Kc, pore-
   pressure grid, composite-EI, eval refresh w/ new-tool questions, more Slide2
   problems, Bray-Travasarou, reviewer family…).
3. 71-Q eval re-runs: `docs/geotech_eval_20260705.json`/`.md` hold the 5.1rc5
   baseline; re-run on 5.3+ with `[deep,full]` clears the 4 env-blocked fails.

---

## 4. How to run things (all offline unless noted)

**Environment:**
- Work in the git worktree `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.claude\worktrees\v5.1-todos`
  (branch `v5.2-coverage`), NOT the main checkout. Merge to master with
  `git merge --ff-only` from the main checkout, as every milestone has.
- **Venv python (system `python` has no pytest):**
  `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.venv\Scripts\python.exe`
- Git identity: `-c user.name=soconnell345 -c user.email=soconnell345@gmail.com`;
  end commit messages with the `Co-Authored-By` trailer.
- Tests: `pytest <module>/ -q`; validation: `pytest validation_examples/ -q`.

**Rebuild the test wheel** (from the main checkout, after merging to master):
`python -m build --wheel` → copy to `v5_test_wheel/`. **CRITICAL: bump the
version first** (`pyproject.toml`) — pip on Databricks skips a same-version
reinstall ("already installed with the same version… use `--force-reinstall`").
Also **verify the wheel contents** after building (a past rc shipped without a
just-added file): unzip and confirm the new files/tokens are present.

**Funhouse (Databricks) — notebook, needs API/`fh_prompter`:**
- Install: `%pip install "/tmp/...whl[deep]"`, then just `import funhouse_agent.deep`.
  `dbutils.library.restartPython()` is **no longer required** in the normal flow:
  `funhouse_agent/runtime_check.py` (run at the top of `funhouse_agent.deep`) reloads
  the freshly-installed `typing_extensions>=4.13` in place, curing the old
  `typing_extensions`/`extra_items` PEP 728 error without a restart. Restart is kept
  only as the fallback the auto-fix points to if an even older copy is winning at
  cluster scope; installing `typing_extensions>=4.13` as a **cluster-scoped library**
  avoids the situation entirely.
- Health check: `from funhouse_agent.deep.rc_wheel_check import run_rc_check; run_rc_check(fh_prompter)`.
- Eval suite (100 questions as of v5.4): `from funhouse_agent.deep.eval_harness import run_suite;
  run_suite(model, out="/tmp/eval")` (writes `.json` + a readable `.md`). Model =
  `PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")`. Real API
  calls; use `limit=` for a subset first. Correctness is PARTLY auto-scored
  (questions with `expected` keys) + partly eyeballed from the `.md`; process
  metrics (P1 hallucination rate, tool-error rate, rounds, latency, tokens) are
  always computed.
  - **Full coverage needs the optional-dependency extras:** install
    `%pip install "/tmp/...whl[deep,full]"` — with `[deep]` alone, ~12 questions
    (the gstools/pygef/ags4/pydiggs/ezdxf/SALib/pystrata/eqsig/liquepy/openseespy
    modules) fail honestly with "not installed" errors. `run_suite` runs an
    optional-dependency preflight and prints a "Missing optional packages" banner
    at the top of the `.md` when any are absent.
- Save outputs to `/tmp` or `/Volumes`, then `dbutils.fs.cp` out — NOT
  `/Workspace` (FUSE writes are non-durable / permission-blocked on the cluster).

---

## 5. Standing constraints & gotchas (carry forward)

- **No version bump→tag→publish without explicit owner OK.** A `v*` tag push
  auto-publishes to PyPI. rc strings are committed without tags on purpose.
- **`ANTHROPIC_API_KEY`** is read from the Windows *User* env at runtime — never
  pass it through chat/transcripts. Live tests are opt-in (`RUN_LIVE_TESTS=1`).
- **Owner is not a developer** — skip git/dev-procedure rationale; report
  actionable results. Prefers autonomous milestone-level operation, big batches.
- **Stagger subagents** — one at a time (usage-window limits); commit at
  milestones so a cutoff is cheap. Every module change must be **additive +
  default-preserving** (mirror the v5.2 Batch-1 pattern): new params/methods
  default to prior behavior; existing module tests stay green byte-for-byte.
- **Validate against published targets, don't tune to them.** A discrepancy is
  far more likely units / method-variant / convention than a module bug;
  investigate before "fixing." Record CONVENTION / N-A(scope) when the module is
  defensibly correct.
- **Edit/Write tools are pinned to the worktree** — to change a main-checkout
  file, author in the worktree + `cp`, or use a Python heredoc via Bash.
- **geotech-references is an editable install** resolving to the *main*
  checkout's submodule; after a submodule pointer change, `git submodule update`
  the main checkout.
- **`np.trapezoid`** (not `np.trapz`); SI units throughout.

---

## 6. File map (where the important things live)

- `HANDOFF.md` (this file) — current authoritative handoff.
- **`module_work/V5.4_PLAN.md`** — CURRENT plan of record (owner directives
  D1–D6 all done; QC carryovers E1–E11 + creative F1–F8 = the open backlog).
- `module_work/V5.3_PLAN.md` — v5.3 board incl. the adversarial-review outcome.
- `module_work/V5.2_COVERAGE.md` / `module_work/WEEKEND_QC_2026-06-13.md` —
  historical boards (complete).
- `validation_examples/INVENTORY.md` + `RESULTS.md` — published problems,
  verdicts, owner notes (coverage-gap backlog lives in the notes).
- `validation_examples/test_published_v0*.py` — the 136+ offline validation tests.
- `drawing_ir/` — NEW (v5.4): LLM-ready drawing IR + query surface; DESIGN.md
  carries the geo_project-wiring follow-up flag.
- `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf` + `docs/user_manual/` —
  the 132-pp manual + regenerable builder (rebuild each release).
- `docs/gallery/` — 12-exhibit module visualization gallery + build_gallery.py.
- `DISCLAIMER.md` + `funhouse_agent/_disclaimer` surfaces — professional-use
  terms (README/PyPI section, first-import notice, geotech-disclaimer script).
- `funhouse_agent/runtime_check.py` — Databricks no-restart typing_extensions fix.
- `funhouse_agent/review_checklists.py` + `funhouse_agent/reviewers.py` +
  `.claude/agents/seismic-reviewer.md` — the narrow-reviewer pattern (seismic
  first; family rollout = F8).
- `funhouse_agent/deep/rc_wheel_check.py` — one-cell Funhouse health check.
- `funhouse_agent/deep/eval_harness.py` — `run_suite` (100-question eval) + scorers.
- `funhouse_agent/geotech_test_suite.json` — the 97 eval questions (71 base + 26 v5.3/v5.4, E10).
- `funhouse_agent/_fileio.py` — verified-write + `/Workspace` rescue helpers.
- `docs/V5.1_SUMMARY.html` — everything-since-5.0 summary (owner-facing).
- `docs/funhouse_agent_guide.md` — install + Databricks gotchas + snippets +
  reviewer-agent usage.
- `CLAUDE.md` — project instructions (auto-loaded); its status block is kept
  current and points here.
- Reliability / geo_project / slope_stability / fem2d each have their own
  `DESIGN.md` + `VALIDATION.md` + `UPGRADE_PLAN.md`.
