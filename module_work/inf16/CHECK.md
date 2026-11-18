# INF-16: a kernel layer under the plotting box on Linux

Branch `claude/inf-16-kernel-layer`, from `master` at 58b7027. Cloud helper session of 2026-11-18. Nothing here turns plotting code on for any host.

## 1. What was confirmed on today's code before changing it

- `packages/aec-agent-core/aec_agent_core/plot_box/child.py`, `main()`: the layers go up in this order: static check (`check_code`), `os.chdir(root)`, OS limits (`_posix_limits`), the libraries (`_preload`), `RLIMIT_NPROC` 0 (`_posix_no_children`), the audit hook (`make_guard`, `sys.addaudithook`), the code. Everything except the OS limits is Python-level, as the BACKLOG item says.
- `child.py` at import time loads only the standard library (`layout_ink.py`, `figures.py` and `plan_furniture.py` import nothing else at their tops), so when `main()` starts no number library has been loaded and one thread exists. Measured: `threads_before` is 1 in every run of the box here.
- The child's environment is scrubbed (`plot_box._child_env`), so the child cannot read a setting of the host itself: the page's process reads it and passes it in the payload.
- `packages/aec-document-review/tests/test_review_switches_offline.py` (and the quiet application's copy) pins `review_flags.SETTINGS_ENVS` to `plot_box.SETTINGS_ENVS` plus two names. So the new setting is NOT added to either tuple, and `review_flags.py` is not edited.
- Already as the item wants: nothing. Wrong in the item: nothing. One thing found that the item does not say: `RLIMIT_NPROC` does not hold for a process run as root, so on a host where the app runs as root (this container does) the box's "no child processes" limit is the audit hook's alone. The seccomp filter of this change closes that where it is up.

## 2. What changed, and why

Not a repair: a new behaviour behind a setting, off by default.

| File | Function | What |
|---|---|---|
| `plot_box/kernel_layer.py` (new) | `put_up`, `box_paths`, `landlock_abi`, `file_layer_missing`, `mode`, `probe`, `_filter_program` | The layer, standard library only (`ctypes`). Its top comment is the documentation of the setting and of each part. |
| `plot_box/kernel_probe.py` (new) | `main` | The administrator's probe: `python -m aec_agent_core.plot_box.kernel_probe` (one JSON line); `--paths` lists what the file rules would allow. |
| `plot_box/child.py` | `_kernel_layer` (new), `main` | When the payload carries `kernel_layer`, loads `kernel_layer.py` from its file and puts the layer up right after `os.chdir(root)`: before the OS limits, before any library, long before the audit hook. Under `require`, reports an error and runs no code where the file rules did not go up. Without the key: not one new line runs. |
| `plot_box/__init__.py` | `kernel_layer_mode`, `_kernel_layer_refusal` (new), `run_plot_code`, `_run_in_box` | Reads the setting at each call; refuses before a process starts (`require` without Landlock, or an unknown value); passes the mode in the payload; copies the child's report to `box.kernel_layer` in the result and adds one note naming every part that is not up. |

**The setting.** `GEOTECH_PLOT_CODE_KERNEL_LAYER`, default unset = off. Read in `plot_box.kernel_layer_mode()`; documented in the top comments of `plot_box/__init__.py` and `plot_box/kernel_layer.py`.

| Value | What happens |
|---|---|
| unset, empty, `0`, `false`, `no`, `off` | The box exactly as on master. `kernel_layer.py` is not imported in the page's process and not opened in the child. |
| `1`, `true`, `yes`, `on` | The child puts up what the kernel offers. The result's `box.kernel_layer` says `up`, `not_up` (part: why), `landlock_version`, `threads_before`, `allowed` (number of paths). On Windows and macOS every part reads "not available on this system (win32)" and the box runs as before. |
| `require` | As `1`; where Landlock's file rules cannot go up the box refuses with a plain error. Where the kernel has no Landlock (or the system is not Linux) the refusal is made in the page's process and no process is started. Where the kernel has it but putting it up fails, the child refuses before any library is loaded. |
| anything else | Refused with a plain error naming the setting; no process is started. |

**The parts**, in order, all restrictions a process may put on itself:

1. `prctl(PR_SET_NO_NEW_PRIVS)`.
2. Landlock. Every file right the kernel's Landlock version knows is handled, so denied, except what section 5 lists. No path is given the execute right (loading a shared library needs only read; proven by the box drawing). TCP `bind` and `connect` are denied from version 4; signals to a process outside the box and abstract Unix sockets outside it from version 6.
3. A seccomp filter, x86_64 and aarch64 (little-endian). `EPERM` for: `socket` (every family), `fork`, `vfork`, `execve`, `execveat`, `clone` unless it makes a thread in this process and no namespace, `io_uring_setup`, `ptrace`, `process_vm_readv`, `process_vm_writev`, `unshare`, `setns`. `clone3` answers `ENOSYS`, so the C library falls back to `clone`, whose flags a filter can read (seen in the trace here: `clone3` ... `ENOSYS`, then `clone(... CLONE_THREAD ...)` succeeded for the layout check's thread). A call made with another machine's numbers (32-bit on x86_64, and the x32 numbers) is refused whole, so the list cannot be walked round by number.

**Was seccomp sound to build with the standard library alone?** Yes, and it is built. The filter is 24 small fixed rows on x86_64 (21 on aarch64), written out by `_filter_program` and handed to `prctl(PR_SET_SECCOMP)` through two `ctypes` structures. It is a deny-list, which is sound only if the architecture word is checked and foreign numbers are refused; both are done. The system call numbers were checked against the kernel headers in this container (`asm/unistd_64.h`, `asm-generic/unistd.h`). The aarch64 table could not be RUN here; `test_the_filter_refuses_sockets_and_processes_and_lets_threads` reads the program for both machines with a small interpreter. On any other machine the filter is not installed and the report says so.

**Order.** The layer goes up with one thread in the process (`threads_before: 1` in every report). The filter goes on before the libraries too.

**The page's process.** Only `landlock_abi()` can run there (under `require`): one system call that asks for a version number. `put_up` is called from `child.py` and from the probe's forked child, nowhere else.

## 3. A made-up before and after

A helper draws a profile from `data.csv` in a conversation folder.

Before (and now, setting unset):

```
"box": {"limits_applied": {"RLIMIT_AS": 2147483648, ...}, "run_seconds": 0.01, "start_seconds": 1.1, "layout_check_seconds": [0.05]}
```

Now, with `GEOTECH_PLOT_CODE_KERNEL_LAYER=1` on this container:

```
"box": {..., "kernel_layer": {"asked": "1", "system": "linux", "files": true,
  "up": ["no_new_privs", "files", "tcp", "scopes", "seccomp"], "not_up": {},
  "threads_before": 1, "landlock_version": 7, "allowed": 20}}
```

With `require` on a kernel without Landlock:

```
{"status": "error", "error": "The code was not run: this host requires the plotting box's kernel layer (GEOTECH_PLOT_CODE_KERNEL_LAYER=require) and it cannot go up here: Landlock is not built into this kernel (or the kernel is older than 5.13). This is a setting of the host, not of the code: nothing in the code changes it."}
```

## 4. The probe's output in this container

```
$ python -m aec_agent_core.plot_box.kernel_probe
{"available": true, "caller_after": {"NoNewPrivs": "0", "Seccomp": "0", "Seccomp_filters": "0"}, "caller_before": {"NoNewPrivs": "0", "Seccomp": "0", "Seccomp_filters": "0"}, "caller_changed": false, "kernel": "6.18.44-fc-v595", "landlock_files": true, "landlock_scopes": true, "landlock_tcp": true, "landlock_version": 7, "machine": "x86_64", "no_new_privs": true, "not_available": {}, "seccomp_filter": true, "system": "linux", "uid_is_root": true}
```

**Which parts went up here:** all five: no-new-privileges, Landlock files, Landlock TCP, Landlock scopes, the seccomp filter. Landlock version 7, x86_64, running as root.

## 5. The paths the file rules allow, and why

From `kernel_layer.box_paths` (run `kernel_probe --paths` on a host for its own list). Only paths that exist get a rule. Everything else is denied: reading, listing, writing, creating, deleting, renaming, executing.

| Path (in this container) | Right | Why it is needed |
|---|---|---|
| the conversation folder | read, write, new files and folders, cut short; no delete, no rename | the data the code reads, the figures it writes |
| the box's scratch folder (`/tmp/plotbox_...`) | the same, plus delete | the libraries' temporary files (matplotlib rebuilds its font list there when it has none); the page removes it after the run |
| `sysconfig` `stdlib` (`/usr/lib/python3.12`) | read | the interpreter's standard library |
| `sysconfig` `platstdlib` (`<venv>/lib/python3.12`) | read | the standard library's compiled modules |
| `sysconfig` `purelib`, `platlib` (`<venv>/lib/python3.12/site-packages`) | read | the installed libraries |
| the folder each of the 20 plotting libraries is installed in, where it is not one of the above | read | a library installed in a user site or a `PYTHONPATH` folder |
| `aec_agent_core/plot_box`, `aec_agent_core/charts` | read | the box's own code: the plan furniture is loaded after the layer, and a traceback reads source lines |
| the chart-lines file the page hands the box | read | the standard chart lines (only where an app registered them) |
| `/lib`, `/lib64`, `/usr/lib`, `/usr/lib64`, `/usr/local/lib`, `<base prefix>/lib` | read | shared libraries the interpreter and the compiled modules load |
| `/etc/ld.so.cache` | read | the loader's index of shared libraries |
| `/usr/share/fonts`, `/usr/local/share/fonts`, the X11 font folders, `~/.local/share/fonts`, `~/.fonts` | read | fonts |
| `/usr/share/zoneinfo` (and two other homes of it), `/etc/localtime` | read | time zones |
| matplotlib's settings folder and its cache folder (`MPLCONFIGDIR`, or `~/.config/matplotlib` and `~/.cache/matplotlib`) | read | `matplotlibrc` and the font list |
| the system's file-type tables (`/etc/mime.types` here) | read | read when a workbook is opened |
| `/dev/urandom` | read | random numbers |
| `/dev/null` | read, write | the null device |

A candidate for reading is left out when it is the file system's root, the home folder itself, or a folder that holds the conversation folder (`test_a_read_root_that_holds_the_conversation_folder_is_left_out`).

**Found by running** (the child under `strace` with the layer up, drawing with numpy, pandas with a time zone, SciPy, matplotlib to PNG, PDF and SVG, and Plotly): the kernel refused these and nothing needed them, so they are NOT allowed: `/proc/stat`, `/sys/devices/system/cpu/online` and `possible` (the C library counting processors; it falls back to the scheduler's answer), `/sys/kernel/mm/transparent_hugepage/enabled` (numpy), `/proc/sys/vm/overcommit_memory`, `/usr/lib/ssl/openssl.cnf` and `/etc/ssl/openssl.cnf`. All four files of that run were written (PNG, PDF, SVG, Plotly chart). With no font list on disk the box still drew (about 1.4 s a run here, layer on or off).

## 6. What the layer does NOT stop (for the owner's decision)

Said plainly. With every part up, code that had got past the Python-level layers could still:

1. **Read everything under the read paths**: every installed library's files, `/usr/lib`, fonts, matplotlib's settings. If a host keeps a secret in a file under one of them (a settings file inside an installed package, say), the layer does not hide it. Run `--paths` on the host and read the list.
2. **Do anything to files inside the conversation folder**: read the person's uploads, overwrite or empty any of them, write files of any kind and any number. Which kinds of file are written, the cap of 40 and "no overwriting what was there" are the audit hook's rules. That is one person's own folder, not a neighbour's.
3. **Learn that a file exists** elsewhere, with its size and dates. Landlock refuses opening and listing, not `stat`.
4. **Change the mode, owner or times of a file it owns elsewhere.** Landlock (through version 7) has no right for `chmod`, `chown`, `utime` or extended attributes. It cannot read or write the file; it can change who else may. On a host where every tester's files belong to one user, that is a real gap, held today only by the audit hook.
5. **Use root's other powers where the box runs as root** (it does in this container: `uid_is_root`). The layer refuses files, sockets, processes, tracing and namespaces; it does not go through the rest of what root may do (the clock, the host name, and so on). A box run as an ordinary user has none of those.
6. **Use up the machine within the box's limits**: memory and CPU up to the limits, threads (the filter lets threads through; a thread is not a process), disk up to 200 MB a file inside the conversation folder. The page's time limit still stops it.
7. **Whatever a kernel fault allows.** This is a second layer, not a virtual machine.
8. **Send its result back.** The page reads one line of JSON from the child, and the files in the conversation folder, as before.

And where only part of the layer is up:

- Landlock before version 4: no TCP rule (the filter still refuses every socket). Before version 6: a signal can be sent to another process of the same user, the page included (stopping it, not reading it). Before version 3: Landlock has no right for cutting a file short by its path, so that is not refused outside the folder for a file the box's user may write; said in the report as `truncate`.
- No seccomp filter (another machine type, or a host that refuses it): UDP and Unix sockets are not refused by the kernel, a pathname Unix socket outside the box can be connected to, and a new process is refused only by "nothing is executable" and the audit hook.
- No Landlock at all: `1` leaves files to the audit hook alone. That is what `require` is for.

## 7. What a host's administrator should know before running the probe

- It is one command, `python -m aec_agent_core.plot_box.kernel_probe`, and prints one line of JSON.
- It asks the kernel for one number (the Landlock version; `landlock_create_ruleset` with the version flag, which makes no rule set).
- It then forks ONE child that does nothing but try the restrictions on itself (no-new-privileges, an almost empty Landlock rule set, the seccomp filter) and exits at once. It starts no program, opens no network connection, and reads and writes no file except one empty temporary folder that is removed again.
- The process that ran the command is not restricted: `caller_before` and `caller_after` are its own kernel flags, and `caller_changed` is false. On a container whose runtime already filters system calls, both show `Seccomp: 2`; that is the runtime's filter, not the probe's.
- To a host's security monitoring, a Python process that calls `prctl`, `landlock_*` and installs a seccomp filter is a process sandboxing itself. That is what browsers and system services do, and the opposite of an escape, but an alert rule tuned for "unusual system calls from the web app" could fire. Tell whoever watches the host before the probe and before the setting is turned on. With the setting on, every box run makes the same few calls in the child.
- It needs no privilege. If the host forbids it, the JSON says which part and the error the kernel gave.

## 8. Tests

New file `packages/aec-agent-core/tests/test_inf16_kernel_layer.py`, 42 tests, made-up values only. On master the whole file fails at import (`aec_agent_core.plot_box.kernel_layer` does not exist), so each test fails on master and passes on the branch.

**Ran or skipped in this container: all 42 RAN, none skipped** (Landlock version 7 and seccomp are available here). On a runner without Landlock the six `needs_landlock` tests and, without x86_64 or aarch64 Linux, the filter test skip with the reason printed.

| Test | What it shows |
|---|---|
| `test_the_settings_values` (16 cases) | each value's meaning; an unknown value is `?` |
| `test_the_setting_is_not_one_of_the_boxs_numbers` | the setting is in neither `SETTINGS_ENVS` |
| `test_the_filter_refuses_sockets_and_processes_and_lets_threads` (x86_64, aarch64) | the filter read without a kernel: refused calls, threads allowed, a process and a namespace refused, `clone3` not implemented, foreign numbers refused |
| `test_the_paths_allow_writing_only_in_the_conversation_folder`, `test_a_read_root_that_holds_the_conversation_folder_is_left_out` | the path list |
| `test_landlock_alone_keeps_files_to_the_conversation_folder` | the layer ALONE, no Python guard: plain `open` outside fails, listing and writing outside fail, `/proc` fails, a TCP connect to a listener on this machine fails, nothing can be executed; reading, writing and making a folder inside work; the neighbour's file is unchanged |
| `test_the_filter_alone_refuses_sockets_and_new_processes` | the filter alone: TCP, UDP, `fork` and starting a program fail; a thread works |
| `test_the_whole_layer_alone` | both together |
| `test_the_probe_is_one_line_of_json_and_restricts_nothing`, `test_the_reason_landlock_is_missing_is_told_apart` | the probe; "not built in" and "not enabled at boot" |
| `test_on_another_system_nothing_is_tried` | Windows: no library call, every part "not available on this system" |
| `test_unset_the_box_is_as_it_was` (unset, `0`, `off`, empty) | the payload's keys and the result's `box` keys are master's |
| `test_unset_the_child_makes_none_of_the_new_system_calls` | the child run as the box runs it: the layer's file is never opened, and the process ends with `NoNewPrivs: 0`, `Seccomp: 0` |
| `test_asked_the_child_puts_the_layer_on_itself` | the same run with the key: the file is opened, `NoNewPrivs: 1` |
| `test_at_1_the_box_still_draws_with_matplotlib`, `..._with_plotly`, `test_require_draws_where_the_layer_is_there` | the box's normal entry with the layer up |
| `test_require_refuses_without_the_layer_and_starts_no_process` (2 cases), `test_a_value_the_box_does_not_know_is_refused_not_run_as_off` | plain error, no process started |
| `test_at_1_without_the_layer_the_box_runs_and_says_so` | the note for parts that are not up |
| `test_the_pages_process_is_not_restricted`, `test_unset_the_pages_process_does_not_load_the_layer` | after a box run with `1`, the calling process reads a file no rule allows, makes a socket, forks, and has `NoNewPrivs: 0`, `Seccomp: 0`; unset, the layer's module is not even imported |

**Runs** (venv (b) unless said; exit code read directly):

| Command | Result |
|---|---|
| `pytest packages/aec-agent-core/tests/test_inf16_kernel_layer.py` | 42 passed, 0 skipped |
| venv (a), from `packages/aec-agent-core`: `pytest tests/test_inf16_kernel_layer.py tests/test_core_imports.py` | 47 passed |
| `GEOTECH_PLOT_CODE_KERNEL_LAYER=1 pytest funhouse_agent/tests/test_plot_box.py funhouse_agent/tests/test_plot_box_drafts.py` | 89 passed (the existing plot box tests with the layer up) |
| the same with `=require` | 89 passed |
| the same with `=required` (a value the box refuses) | 49 failed, 40 passed: shown only to prove the setting reaches those tests' box runs |
| the same two files, unset, with `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_wheels.py` and the new file | 150 passed, 5 skipped (the opt-in wheel build tests) |
| `pytest funhouse_agent/deep/tests/test_figure_runs.py funhouse_agent/deep/tests/test_figure_turn.py` | 38 passed |
| venv (a), from `packages/aec-document-review`: `pytest tests/test_boundary.py tests/test_review_fingerprints.py tests/test_review_switches_offline.py` | 18 passed |
| guards: `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` | 13 passed |

**Unset is master, checked directly:** four box runs (a matplotlib figure with a Plotly chart, a refused import, a refused read, a script that saves nothing) and `plot_code_description()` were run from a worktree of master and from the branch; with the timings blanked the two outputs are identical, byte for byte.

No existing test was edited.

## 9. Fingerprints

Not moved. `webapp/tests/test_page_fingerprints.py` and Document Review's `tests/test_review_fingerprints.py` pass with no re-record. The tool's description is unchanged.

## 10. Calls made

1. **The setting's name and values** as the task suggested. Added: `true`/`yes`/`on` mean `1`, as the other switches read them, and **an unknown value is refused** rather than read as off. A mistyped `require` on a shared host must not run unprotected.
2. **`require` asks for the file rules only**, as the task says. The TCP rules, scopes and the filter are best effort under `require` too, and are named in the result when missing. If the owner wants `require` to mean "files and no network", that is one line in `_kernel_layer_refusal` and `child.main`.
3. **No path may be executed.** The task allowed execute "where loading a library needs it"; it does not (a shared library is mapped with the read right), so none is given.
4. **The read paths are narrower than the audit hook's.** The hook allows the whole interpreter prefix and any font file anywhere; the layer allows the standard library, the site folders, the shared-library folders and the font folders. Not `sys.prefix` itself, and not the folders on the app's own `sys.path` unless one of the plotting libraries is installed there.
5. **matplotlib's cache folder is read-only**, not writable, so one run cannot leave a font list for the next. With no list on disk matplotlib builds one in the scratch folder each run.
6. **No delete and no rename in the conversation folder** (the box does neither; the page retires drafts).
7. **The filter's list** goes a little past "socket creation and process start": `io_uring_setup` (it can make sockets without `socket`), `ptrace` and the two calls that read or write another process's memory, `unshare` and `setns`. Each is a way to the same two things.
8. **The scratch folder is writable.** The task says "write only inside the conversation folder"; the box has always given its libraries one temporary folder per run, removed afterwards, and matplotlib needs it when it has no font list. It holds nothing of anyone's.
9. **The probe reports `uid_is_root`** and the caller's kernel flags before and after, beyond what the task lists.
10. **The result carries the count of allowed paths, not the list** (20 long paths in every tool result would cost the model tokens); `kernel_probe --paths` prints the list.
11. **The layer goes up before the OS limits**, not between them and the libraries: it needs `ctypes` loaded, and loading it under the memory limit adds nothing.

## 11. Release note

None for a tester or a GSE user: with the setting unset nothing changes. For a host's operator: "`aec-agent-core` can put a kernel layer under the plotting box on Linux (`GEOTECH_PLOT_CODE_KERNEL_LAYER=1` or `require`; off by default): the box's child process gives up new privileges, is held by Landlock to reading its libraries and writing its conversation folder, and is refused sockets and new processes by a seccomp filter. `python -m aec_agent_core.plot_box.kernel_probe` says what a host's kernel offers."

## 12. For the lead

1. **Run the full suite on GitHub** (`ci/<name>`). Say whether the `needs_landlock` tests RAN there: the summary line shows skips, and `-rs` prints "this kernel offers no Landlock: ..." with the reason. GitHub's Ubuntu runners have a recent kernel; whether Landlock is enabled at boot there I could not check from here.
2. **`tinyapps/` host notes**: add the setting (not edited here, as told).
3. **What must be seen on the testers' host before the owner could consider turning plotting code on there.** None of it can be seen from this container.
   - The probe's line from inside the running app's container: `landlock_files: true` (without it `require` refuses everything and `1` protects no file), `landlock_version` 4 or more for the TCP rules and 6 or more for the signal scope, `seccomp_filter: true`, `caller_changed: false`. If Landlock is "not enabled at boot" on a managed host, nobody on our side can enable it, and this layer is not the answer there.
   - `uid_is_root`. If the app runs as root there, section 6 item 5 applies; the owner should know before deciding.
   - `kernel_probe --paths` run there, with the list read by a person: no read path may hold another tester's files, the app's settings, or a mounted document store. Section 6 item 1.
   - How testers' files are owned there. If one user owns every tester's files, section 6 item 4 (mode and owner changes) is open at the kernel level.
   - One real box run there with the setting at `require`, plotting code switched on for that one check: the run record's `box.kernel_layer` must show `files` and `seccomp` in `up`, a figure must come back, and `start_seconds` must be about what it is without the layer.
   - Word from whoever watches the host's security alerts that the probe and the layer's calls are expected (section 7).
   - Then `require`, not `1`, as the value on that host: `1` runs unprotected where the layer fails.
4. **Owner question (OWN-133 as it stands):** this builds the operating-system layer the question waited for. It does not answer the question. Section 6 is what the owner would be accepting.
5. **A model run** has not been made with the layer on; nothing a model sees changes unless a part is not up (then one note per result). Brief 9 could carry one arm with the setting on if the suite's host has Landlock.

## 13. Noted, not fixed

- `RLIMIT_NPROC` does not hold where the box runs as root (section 1). With the layer's filter up it no longer matters; without it, "no child processes" is the audit hook's alone on such a host.
- `child._read_roots` keeps its own copy of the library list that `kernel_layer.LIBRARIES` repeats; a library added to one should be added to the other.
- Under `require`, the child's own refusal (the kernel has Landlock, yet putting it up fails) has no test: I found no honest way to make the kernel refuse on demand.
- aarch64: the filter is checked by reading it, not by running it. The first box run on an aarch64 host should be looked at (`box.kernel_layer.up`).
