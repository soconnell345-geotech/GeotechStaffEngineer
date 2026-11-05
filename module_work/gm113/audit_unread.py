"""GM-113 reverse audit: METHOD_INFO names that no handler reads.

Pass 1 runs every method probe (funhouse_agent/tests/method_probes.json)
with its parameters wrapped in a dict that records each key looked up
(``[]``, ``get``, ``in``, ``pop``, ``setdefault``), so a name the handler
asks for is "read" even when the probe left it out. A handler that copies
the dict (``dict(p)``, ``{**p}``, ``f(**p)``) hides later reads, so pass 2
takes each name pass 1 did not see and looks for a read of it in the
handler and the adapter functions it calls (``["x"]``, ``.get("x"``,
``"x" in``, ``pop("x"``, ``helper(params, "x")``), or in the signature of a function the handler
passes ``**params`` (or ``**p``) to. Methods without a probe get pass 2 only; a calc
package's names are checked against its tool's METHOD_INFO.

Run from the repo root: ``python module_work/gm113/audit_unread.py``.
It prints what is left for a person to read; on 2026-11-05, after GM-113,
nothing (before: fem2d_local_fos gamma_w, max_iter, n_gp, t, tol).
"""
import copy
import inspect
import os
import re
import sys
import tempfile
import types

os.environ.setdefault("GEOTECH_NO_DISCLAIMER", "1")
sys.path.insert(0, os.path.join(os.getcwd(), "funhouse_agent", "tests"))
import method_probes as MP  # noqa: E402
from funhouse_agent import dispatch as D  # noqa: E402


class _Recorder(dict):
    log = None

    def __getitem__(self, k):
        self.log.add(k)
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        self.log.add(k)
        return dict.get(self, k, d)

    def __contains__(self, k):
        self.log.add(k)
        return dict.__contains__(self, k)

    def pop(self, k, *a):
        self.log.add(k)
        return dict.pop(self, k, *a)

    def setdefault(self, k, d=None):
        self.log.add(k)
        return dict.setdefault(self, k, d)


def _closure(mod, fn):
    funcs = {n: f for n, f in vars(mod).items()
             if isinstance(f, types.FunctionType)}
    seen, src, todo = set(), "", [fn]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        try:
            s = inspect.getsource(f)
        except (OSError, TypeError):
            continue
        src += s
        todo += [g for n, g in funcs.items()
                 if g not in seen and re.search(r"\b%s\b" % n, s)]
    return src


def _star_targets(mod, src):
    names = set()
    for m in re.finditer(r"(\w+)\(\s*\*\*(?:params|p)\b|_run\(\s*(\w+)", src):
        name = m.group(1) or m.group(2)
        f = getattr(mod, name, None)
        if f is None:             # imported inside the handler
            imp = re.search(r"from ([\w.]+) import[^\n]*\b%s\b" % name,
                            src + inspect.getsource(mod))
            if imp:
                import importlib
                f = getattr(importlib.import_module(imp.group(1)), name, None)
        if callable(f):
            try:
                names |= set(inspect.signature(f).parameters)
            except (TypeError, ValueError):
                pass
    return names


def _read_in(src, p):
    q = re.escape(p)
    # the last form: a helper handed the params and the names to try,
    # as liquefaction's _first(params, "FC", "fc", "fines")
    return re.search(r"""\[\s*['"]%s['"]\s*\]|\.get\(\s*['"]%s['"]|"""
                     r"""['"]%s['"]\s+(not\s+)?in\b|pop\(\s*['"]%s['"]|"""
                     r"""\w+\(\s*params\s*,[^)]*['"]%s['"]"""
                     % (q, q, q, q, q), src)


def main():
    here = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    seen = {}
    for key, probe in sorted(MP.PROBES.items()):
        if MP.skip_reason(key) or key in MP.SLOW:
            continue
        base = key.split("#")[0]
        agent, method = base.split("|")
        p = _Recorder(copy.deepcopy(probe["params"]))
        p.log = seen.setdefault(base, set())
        try:
            for b in probe.get("before", ()):
                D.call_agent(b[0], b[1], copy.deepcopy(b[2]))
            D._load_adapter(agent).METHOD_REGISTRY[method](p)
        except Exception as exc:                      # noqa: BLE001
            print("probe failed", key, repr(exc)[:120])
    os.chdir(here)
    left = []
    for agent in D.MODULE_REGISTRY:
        try:
            mod = D._load_adapter(agent)
        except Exception:                             # noqa: BLE001
            continue
        for method, info in mod.METHOD_INFO.items():
            if not isinstance(info, dict) or info.get("alias_of"):
                continue
            handler = mod.METHOD_REGISTRY.get(method)
            if handler is None or getattr(handler, "__wrapped__", None):
                continue                  # a reference function: signature
            params = info.get("parameters") or {}
            refused = info.get("refused_parameters") or {}
            if agent == "calc_package":
                continue                  # checked against its tool below
            src = _closure(mod, handler)
            star = _star_targets(mod, src)
            log = seen.get(f"{agent}|{method}", set())
            for p in params:
                if p in refused or p in log or p in star or _read_in(src, p):
                    continue
                if p in D._DISPATCH_PARAMS:
                    continue
                left.append((agent, method, p))
    cp = D._load_adapter("calc_package")
    for pkg, spec in cp._PACKAGE_TOOLS.items():
        info = cp.METHOD_INFO.get(pkg)
        try:
            ad = D._load_adapter(spec["adapter"])
        except Exception:                             # noqa: BLE001
            continue
        tool = set()
        for t in spec["tools"].values():
            ti = ad.METHOD_INFO[t]
            tool |= set(ti.get("parameters") or {}) | set(
                ti.get("parameter_aliases") or {})
        extra = (set((info or {}).get("parameters") or {}) - tool
                 - set(cp._COMMON_PARAMS) - set(spec.get("own", {}))
                 - set(spec.get("dropped", {})) - set(spec.get("older", {})))
        if "selector" in spec:
            extra -= {spec["selector"][0]}
        left += [("calc_package", pkg, p) for p in sorted(extra)]
    for row in left:
        print(*row)
    print(f"{len(left)} declared name(s) with no read found")


if __name__ == "__main__":
    main()
