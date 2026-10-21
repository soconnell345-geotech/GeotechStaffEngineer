"""GM-263: every did-you-mean hint target in an adapter names a key the tool
reads for at least one of the items the hint table serves. A hint to a key no
item takes is dead: the matcher filters it out, so the user sees no hint."""
import ast
import importlib
import pathlib

import pytest

_ADAPTERS = pathlib.Path(__file__).resolve().parents[1] / "adapters"
_CHECKS = {"check_nested_keys", "check_nested_list"}


def _scan(path):
    """Yield (table_name, valid_keys) for each nested-key check in a module
    whose hints= is a plain name and whose valid keys can be evaluated."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mod = importlib.import_module(f"funhouse_agent.adapters.{path.stem}")
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(
                node.func, "id", getattr(node.func, "attr", None))
                in _CHECKS and len(node.args) >= 2):
            continue
        hints = next((k.value for k in node.keywords if k.arg == "hints"),
                     None)
        if not isinstance(hints, ast.Name):
            continue
        table = getattr(mod, hints.id, None)
        if not isinstance(table, dict):
            continue
        try:
            valid = set(eval(ast.unparse(node.args[1]), vars(mod)))
        except Exception:
            # a loop variable or parameter: every key tuple of the module
            valid = _module_key_sets(mod)
        yield hints.id, table, valid


def _module_key_sets(mod):
    """Every string in a module-level tuple, list, or dict of tuples."""
    out = set()
    for name, val in vars(mod).items():
        if name.startswith("__"):
            continue
        for tup in (val.values() if isinstance(val, dict) else [val]):
            if isinstance(tup, (tuple, list, frozenset)):
                out |= {k for k in tup if isinstance(k, str)}
    return out


def _cases():
    served = {}
    for path in sorted(_ADAPTERS.glob("*.py")):
        try:
            for name, table, valid in _scan(path):
                entry = served.setdefault((path.stem, name), [table, set()])
                entry[1] |= valid
        except ImportError:
            continue
    return sorted((m, n, t, v) for (m, n), (t, v) in served.items())


@pytest.mark.parametrize("module,name,table,valid", _cases(),
                         ids=lambda x: x if isinstance(x, str) else "")
def test_every_hint_target_is_a_key_the_tool_takes(module, name, table,
                                                   valid):
    dead = {k: [t for t in targets if t not in valid]
            for k, targets in table.items()}
    dead = {k: v for k, v in dead.items() if v}
    assert not dead, f"{module}.{name}: dead hint targets {dead}"


def test_the_scan_finds_the_three_reported_tables():
    found = {(m, n) for m, n, _, _ in _cases()}
    assert ("pynite_adapter", "_KEY_HINTS") in found
    assert ("subsurface_adapter", "_SITE_KEY_HINTS") in found
    assert ("downdrag", "_LAYER_KEY_HINTS") in found


def _all_hint_tables():
    import funhouse_agent
    root = pathlib.Path(funhouse_agent.__file__).parent
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        name = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for attr, val in vars(mod).items():
            if attr.endswith("HINTS") and isinstance(val, dict):
                yield name, attr, val


def test_every_hint_key_is_in_the_normalised_form_the_lookup_uses():
    """The lookup normalises the user's key (lower case, letters and digits
    only), so a hint written 'node_i' never matches anything."""
    from funhouse_agent.arg_checks import _norm_key
    bad = {f"{m}.{a}": [k for k in t if k != _norm_key(k)]
           for m, a, t in _all_hint_tables()}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"hint keys the lookup can never match: {bad}"
