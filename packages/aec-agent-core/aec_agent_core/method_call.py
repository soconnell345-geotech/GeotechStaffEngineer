"""One method call: what every module method reached by name goes through.

The checks and notes around a method call that do not depend on WHICH
module is called: a top-level parameter the method does not take refused
with a did-you-mean, an attached file or a working-folder file found for a
method that reads one, every output path moved into the working folder and
said so, a call that did not fit the method's signature answered in plain
words, the declared defaults the call left in force listed as
``defaults_used``, and every working-folder path in the result given
relative to it.

Moved here from :mod:`funhouse_agent.dispatch` (INF-31, the app split's
step 6), so ``plot_data`` runs through the same code on a page without the
geotechnical dispatch (:mod:`funhouse_agent.charts.plot_tool`).
``dispatch.call_agent`` finds the module and the method and calls
:func:`run_method`; the names below are still importable from
``funhouse_agent.dispatch``, as the same objects.
"""

import contextlib
import inspect
import os
import re

def _method_params(mod, method: str) -> dict:
    """The documented parameters of ``method`` (``{}`` when undocumented)."""
    try:
        info = mod.METHOD_INFO.get(method) or {}
        params = info.get("parameters") or {}
        return params if isinstance(params, dict) else {}
    except Exception:                                  # noqa: BLE001
        return {}


def _resolve_attachment(parameters: dict, attachments: dict,
                        documented: dict = None) -> dict:
    """Turn an ``attachment_key`` into what the method reads.

    Bridges the upload system to the adapters. The key names an uploaded
    file: its bytes in ``attachments`` (decoded to ``content`` for a method
    that reads text, e.g. parse_diggs with DIGGS XML), or the file of that
    name in the working folder, where the app stages every upload (passed as
    ``file_path``). A method documenting ``file_path`` but no ``content``
    (parse_cpt, read_ags4...) gets the file. Until 2026-10-09 the deep
    agent's ``call_agent`` passed no attachments, so the key never resolved
    and the error then asked for an ``attachment_key`` (live smoke wave 1,
    A8).
    """
    key = parameters.get("attachment_key")
    if not key:
        return parameters
    from aec_agent_core._fileio import find_in_working_folder
    documented = documented or {}
    wants_path = "file_path" in documented and "content" not in documented
    params = dict(parameters)
    params.pop("attachment_key")
    key = str(key)
    on_disk = find_in_working_folder(key)
    if on_disk is None and os.path.isfile(key):
        on_disk = os.path.abspath(key)
    if attachments and key in attachments and not (wants_path and on_disk):
        raw = attachments[key]
        if isinstance(raw, (bytes, bytearray)):
            params["content"] = raw.decode("utf-8", errors="replace")
        else:
            params["content"] = str(raw)
        return params
    if on_disk:
        params["file_path"] = on_disk
        return params
    available = sorted(attachments.keys()) if attachments else []
    raise KeyError(
        f"attachment_key '{key}' is neither an attached file "
        f"(attached: {available or 'none'}) nor a file of that name in the "
        "working folder. Pass the uploaded file's name exactly as the "
        "attachment note gives it, or a 'file_path'.")


#: Parameters naming a file a module method READS (a bare name is looked up
#: in the working folder, as the file tools do).
_INPUT_PATH_PARAMS = ("file_path", "html_path")


_INPUT_PATH_LIST_PARAMS = ("file_paths",)


def _inputs_from_working_folder(parameters):
    """A bare file name given for a file a method reads is the file of that
    name in the working folder when there is no such file relative to the
    process (live smoke wave 1, A6: results and notes name files by their
    conversation-relative name, so the model passes those names back)."""
    if not isinstance(parameters, dict):
        return parameters
    from aec_agent_core._fileio import find_in_working_folder

    def _resolve(value):
        if not isinstance(value, str) or not value.strip():
            return value
        if os.path.isabs(os.path.expanduser(value)) or os.path.exists(value):
            return value
        return find_in_working_folder(value) or value

    out = parameters
    for key in _INPUT_PATH_PARAMS + _INPUT_PATH_LIST_PARAMS:
        if key not in parameters:
            continue
        value = parameters[key]
        new = ([_resolve(v) for v in value] if isinstance(value, list)
               else _resolve(value))
        if new != value:
            if out is parameters:
                out = dict(parameters)
            out[key] = new
    return out


#: What a TypeError says when the CALL did not fit the function's signature
#: (as opposed to a type problem inside the computation).
_SIGNATURE_ERROR = re.compile(
    r"(missing \d+ required (?:positional|keyword-only) argument"
    r"|unexpected keyword argument|got multiple values for argument"
    r"|takes \d+ positional arguments? but \d+ (?:was|were) given"
    r"|required (?:positional|keyword) argument)")


def _parameter_error(mod, agent_name: str, method: str, parameters: dict,
                     exc: BaseException) -> dict:
    """The answer to a call whose parameters did not fit the method: what
    was wrong in plain words, the parameters the method takes (required
    first) and where its documentation is -- never the raw TypeError, which
    named internals such as ``_build.<locals>.reference_get()`` (live smoke
    wave 1, A15b)."""
    documented = _method_params(mod, method)
    required = sorted(k for k, v in documented.items()
                      if isinstance(v, dict) and v.get("required"))
    optional = sorted(k for k in documented if k not in required)
    given = sorted(parameters or {})
    text = str(exc)
    m = re.search(r"unexpected keyword argument '([^']+)'", text)
    if m:
        what = f"'{m.group(1)}' is not a parameter of {method}"
    else:
        missing = re.findall(r"'([^']+)'", text.split("argument", 1)[-1]) \
            if "missing" in text else []
        if missing:
            what = (f"{method} needs "
                    + ", ".join(f"'{p}'" for p in missing))
        else:
            what = f"the parameters given do not fit {method}"
    # A did-you-mean for each given name the method does not take, as the
    # adapters' own check gives (GM-177): a reference function takes
    # 'phi_deg' where the analysis modules take 'phi'. Nothing is renamed.
    from aec_agent_core.arg_checks import suggest_params
    info = mod.METHOD_INFO.get(method) if hasattr(mod, "METHOD_INFO") else None
    aliases = (info.get("parameter_aliases") or {}) \
        if isinstance(info, dict) else {}
    hints = []
    for k in given:
        if documented and k not in documented and k not in aliases \
                and k not in _DISPATCH_PARAMS:
            near = suggest_params(k, documented)
            if near:
                hints.append(f"{k} -> {' or '.join(near)}")
    hint = f" Did you mean: {'; '.join(hints)}?" if hints else ""
    out = {"error": (f"{agent_name}.{method}: {what}.{hint} "
                     f"Parameters given: {given or 'none'}."),
           # the function's own words, minus internal qualifiers
           "detail": re.sub(r"[\w.]*<locals>\.", "", text),
           "required_parameters": required,
           "optional_parameters": optional,
           "directive": (f"Call call_agent('{agent_name}', '{method}', "
                         "{...}) with these parameter names, inside "
                         "'parameters'; describe_method gives each one's "
                         "meaning and units.")}
    if not documented:
        out.pop("required_parameters")
        out.pop("optional_parameters")
    return out


#: Parameters call_agent itself reads before the method sees them.
_DISPATCH_PARAMS = ("attachment_key",)


def _takes_catch_all(func) -> bool:
    """Whether ``func`` takes ``**kwargs`` (so its signature refuses no
    name)."""
    try:
        return any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in inspect.signature(func).parameters.values())
    except (TypeError, ValueError):
        return False


def _item_key_hint(mod, method: str, parameters: dict, valid,
                   aliases=()) -> str:
    """For each unknown top-level name that is a key of an item of one of
    the method's list parameters (``ru`` beside ``soil_layers``), a
    sentence naming that list (BACKLOG GM-113), from the adapter's
    ``ITEM_KEYS`` (``{method: {param: keys}}``); "" when none is.
    Nothing is moved into the list: a top-level value would stand for
    every item, which the caller has to say."""
    table = (getattr(mod, "ITEM_KEYS", None) or {}).get(method) or {}
    out = []
    for k in sorted(parameters):
        if k in valid or k in aliases:
            continue
        lists = [p for p, keys in table.items() if k in keys]
        if lists:
            out.append(f"'{k}' is a key of each item of "
                       f"{' or '.join(lists)}, not a parameter of its own: "
                       f"give it in each item "
                       f"({lists[0]}[i].{k}).")
    return " ".join(out)


def _check_parameter_names(mod, method: str, parameters: dict) -> None:
    """Refuse a top-level parameter ``method`` does not take (BACKLOG
    GM-110): the ValueError names the unknown key(s), a did-you-mean and
    the valid parameters, as ``reject_unknown_params`` does inside the
    adapters. Without it a misspelt name ('phi_deg' for 'phi') was read by
    nobody and the run used the default. Nothing is renamed.

    The method's METHOD_INFO is the list: its ``parameters``, plus the
    other spellings it reads (``parameter_aliases``, alias -> parameter)
    and, with ``parameters_of`` naming one of its parameters, the
    parameters of the method that one names (``render_figures`` takes a
    package's inputs). A name in ``refused_parameters`` (name -> reason, or
    ``{"reason", "accepts": [values]}``) is one the method deliberately
    does not take; it is refused with that reason instead of a
    did-you-mean, unless its value is one the entry accepts as "not used".
    Not checked here: a method without documented
    parameters, one marked ``open_parameters``, and a reference wrapper
    (``func(**params)``) without a ``**kwargs`` catch-all, whose own
    signature refuses an unknown name. A wrapper WITH a catch-all is
    checked here, against its named parameters plus what the catch-all
    reads (``_reference_common.extract_method_info``).
    """
    info = mod.METHOD_INFO.get(method)
    if not isinstance(info, dict) or info.get("open_parameters"):
        return
    declared = info.get("parameters")
    if not isinstance(declared, dict):
        return
    wrapped = getattr(mod.METHOD_REGISTRY.get(method), "__wrapped__", None)
    if wrapped is not None and not _takes_catch_all(wrapped):
        return
    valid = set(declared)
    aliases = dict(info.get("parameter_aliases") or {})
    refused = dict(info.get("refused_parameters") or {})
    of = info.get("parameters_of")
    if of:
        names = [of] + [a for a, c in aliases.items() if c == of]
        target = next((parameters[n] for n in names
                       if isinstance(parameters.get(n), str)), None)
        other = mod.METHOD_INFO.get(target) if target else None
        # No such method: the handler says what to name. An open one
        # checks its own.
        if (not isinstance(other, dict) or target == method
                or other.get("open_parameters")):
            return
        valid |= set(other.get("parameters") or {})
        aliases.update(other.get("parameter_aliases") or {})
        refused = {**(other.get("refused_parameters") or {}), **refused}
    from aec_agent_core.arg_checks import (
        accepted_as_unused, refused_reason, reject_unknown_params,
    )
    # A name the method deliberately does not take is refused with its
    # reason, never with a near-miss guess at a different quantity; a value
    # its entry accepts as "not used" (a slope of 0) passes.
    hits = sorted(k for k, v in parameters.items() if k in refused
                  and not accepted_as_unused(refused[k], v))
    rest = {k: v for k, v in parameters.items() if k not in refused}
    try:
        reject_unknown_params(rest, valid, method=method,
                              aliases=tuple(aliases) + _DISPATCH_PARAMS)
    except ValueError as exc:
        where = _item_key_hint(mod, method, rest, valid,
                               tuple(aliases) + _DISPATCH_PARAMS)
        if where:
            exc = ValueError(f"{exc} {where}")
        if not hits:
            raise exc from None
        also = f" Also: {exc}"
    else:
        also = ""
    if hits:
        # Names refused for the same reason share one sentence.
        why = {}
        for k in hits:
            why.setdefault(refused_reason(refused[k]), []).append(k)
        raise ValueError(
            f"{method} does not take {hits}: "
            + "; ".join(f"{', '.join(names)}: {reason}"
                        for reason, names in why.items())
            + "." + also)


#: A calc package's header and file format: not engineering choices, so
#: never listed in ``defaults_used``.
_PACKAGE_HEADER_PARAMS = frozenset({
    "project_name", "project_number", "engineer", "checker", "company",
    "date", "format"})


def _declared(mod, method: str):
    """(declared parameters, parameter_aliases) of ``method``, or
    ``({}, {})``."""
    info = mod.METHOD_INFO.get(method)
    declared = info.get("parameters") if isinstance(info, dict) else None
    if not isinstance(declared, dict):
        return {}, {}
    return declared, info.get("parameter_aliases") or {}


def _given_names(mod, method: str, given) -> set:
    """The parameter names a call gave, a name given under another spelling
    the method reads (``parameter_aliases``) under its own name too."""
    _, aliases = _declared(mod, method)
    return set(given) | {aliases[k] for k in given if k in aliases}


def _fold(value):
    """``value`` compared without case when it is a string."""
    return value.lower() if isinstance(value, str) else value


def default_in_force(spec: dict, values: dict):
    """The default one declared parameter takes in a call: its
    ``default``, or, where the default depends on another input, the one
    its ``default_depends_on`` gives for that input's value in force
    (``values``; BACKLOG GM-158: Terzaghi's N-gamma defaults to
    Kumbhojkar, every other factor method's to Vesic).
    ``default_depends_on`` is ``{"parameter": other input, "values":
    {its value: default}, "otherwise": default}``, strings compared
    without case. A parameter whose ``default_from`` names another input
    (GM-165: a foundation friction angle that defaults to the backfill's)
    takes that input's value in force."""
    source = spec.get("default_from")
    if source:
        return values.get(source)
    depends = spec.get("default_depends_on")
    if not depends:
        return spec.get("default")
    other = _fold(values.get(depends["parameter"]))
    for value, default in depends["values"].items():
        if _fold(value) == other:
            return default
    return depends["otherwise"]


def case_of(spec, declared=None):
    """The ``when_absent`` case ({case, basis}) a left-out input stands
    for (GM-165), else None. A parameter that takes another's value
    (``default_from``) takes that one's case when it is left out too,
    followed through ``declared`` (the method's parameters): a passive-side
    water table that defaults to the active side's, with neither given, is
    "no water table"."""
    seen = set()
    while isinstance(spec, dict):
        if spec.get("when_absent"):
            return spec["when_absent"]
        source = spec.get("default_from")
        if not source or declared is None or source in seen:
            return None
        seen.add(source)
        spec = declared.get(source)
    return None


def absent_case(spec, declared=None) -> str:
    """What a left-out input with no default stands for in the run, as
    ``defaults_used`` gives it (GM-165): "<case>: <basis>" from the
    parameter's ``when_absent`` (:func:`case_of`); "" when it declares
    none."""
    case = case_of(spec, declared)
    if not case:
        return ""
    return f"{case['case']}: {case['basis']}"


def _values_in_force(declared: dict, aliases: dict, parameters=None) -> dict:
    """Each declared parameter's value in force in a call: the call's
    (under the name the method reads), else its default, a default that
    depends on another input (:func:`default_in_force`) taken at that
    input's value in force."""
    values = {name: spec.get("default") for name, spec in declared.items()
              if isinstance(spec, dict)}
    named = set()
    for key, value in (parameters or {}).items():
        values[aliases.get(key, key)] = value
        named.add(aliases.get(key, key))
    for name, spec in declared.items():
        if isinstance(spec, dict) and spec.get("default_depends_on") \
                and name not in named:
            values[name] = default_in_force(spec, values)
    for name, spec in declared.items():
        if isinstance(spec, dict) and spec.get("default_from") \
                and name not in named:
            values[name] = default_in_force(spec, values)
    return values


def _left_out(mod, method: str, given, parameters=None) -> dict:
    """The declared defaults of ``method`` that a call left in force: name
    -> default, a default that depends on another input taken at the
    call's value of it (``parameters``), one taken from another input
    (``default_from``) at that input's value, and a left-out input with no
    default as the case it stands for (``when_absent``,
    :func:`absent_case`; GM-165). Left out: any other declared default of
    None (worked out by the method) or "" (an empty field), and a calc
    package's header and format."""
    declared, aliases = _declared(mod, method)
    named = _given_names(mod, method, given)
    if str(getattr(mod, "__name__", "")).endswith("calc_package"):
        named |= _PACKAGE_HEADER_PARAMS
    values = _values_in_force(declared, aliases, parameters)
    out = {}
    for name, spec in declared.items():
        if not isinstance(spec, dict) or name in named:
            continue
        value = values.get(name)
        if value is None:
            value = absent_case(spec, declared) or None
        if value is not None and value != "":
            out[name] = value
    return out


#: What a parameter's ``applies_when`` may ask of another parameter (GM-119).
_SET, _UNSET = "set", "unset"


def _is_set(value) -> bool:
    """A value in force that does something: not None, 0, False or empty."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and value == 0:
        return False
    if isinstance(value, (str, list, tuple, dict)) and not value:
        return False
    return True


def applies(condition, values: dict) -> bool:
    """Whether a parameter's default applies to a call (GM-119).

    ``condition`` is the parameter's ``applies_when``: absent (None) or
    ``{}``, always; False, never (the method takes the parameter and does
    not use it); a list of such dicts, any one of them; else a dict, every
    entry of which must hold, of another parameter's name to a list of
    values its value in force is one of (strings compared without case;
    None for "not given" where the parameter has no default), "set" (not
    None, 0, False or empty) or "unset". ``values`` maps each parameter to
    its value in force: the call's, else its declared default."""
    if condition is None:
        return True
    if condition is False:
        return False
    if isinstance(condition, list):
        return any(applies(c, values) for c in condition)
    for name, want in condition.items():
        value = values.get(name)
        if want == _SET:
            ok = _is_set(value)
        elif want == _UNSET:
            ok = not _is_set(value)
        else:
            ok = any(_fold(value) == _fold(w) for w in want)
        if not ok:
            return False
    return True


def _defaults_used(mod, method: str, given, parameters=None,
                   reported=(), took=None) -> dict:
    """The declared defaults a call left in force that apply to it, with
    the default each took (BACKLOG GM-115, GM-119): name -> default, from
    the method's METHOD_INFO ``parameters`` (GM-110), a default that
    depends on another input at the call's value of it (GM-158), and a
    left-out input with no default as the case it stands for or the other
    input's value it takes (GM-165, :func:`_left_out`). A
    parameter given under another spelling it reads
    (``parameter_aliases``) counts as given. A parameter whose
    ``applies_when`` (:func:`applies`) does not hold for this call is left
    out, unless the module said it used the default (``reported``, the
    names its ``choices`` called "default"), so ``defaults_used`` and
    ``choices`` never contradict each other. A default the module worked
    out from other inputs (``took``, name -> value; GM-232: a wall friction
    of 0.66 phi') is given as the value it took, not the declared one."""
    left = _left_out(mod, method, given, parameters)
    if not left:
        return {}
    declared, aliases = _declared(mod, method)
    values = _values_in_force(declared, aliases, parameters)
    used = {name: default for name, default in left.items()
            if name in reported
            or applies(declared[name].get("applies_when"), values)}
    for name, value in (took or {}).items():
        if name in used:
            used[name] = value
    return used


def _with_defaults_used(mod, method: str, given, result, parameters=None,
                        reported=(), took=None):
    """``result`` with ``defaults_used`` (:func:`_defaults_used`) added when
    the call left a declared default in force that applies to it: the agent
    then knows which values it never chose. A result that is not a dict, is
    an error, or already says what it used is returned unchanged."""
    if not isinstance(result, dict) or "error" in result \
            or "defaults_used" in result:
        return result
    try:
        used = _defaults_used(mod, method, given, parameters, reported,
                              took)
    except Exception:  # noqa: BLE001 - never the method's result
        return result
    if not used:
        return result
    return {**result, "defaults_used": used}


#: Parameters naming a file (``False``) or a folder of files (``True``) that a
#: module method WRITES. Every writer reached through ``call_agent`` takes one
#: of these names (``write_diggs``, the calc packages, ``html_to_pdf``,
#: ``render_figures``, the DXF export, ``snip_region``, the plots).
_OUTPUT_PARAMS = (("output_path", False), ("output_dir", True))


def _outputs_into_working_folder(parameters):
    """``(parameters, named)``: with a host working folder set, every output
    path the model gave is moved INTO that folder (file name kept,
    :func:`funhouse_agent._fileio.into_working_folder`) — the same folder
    ``write_docx``, ``save_file`` and ``annotate_document`` write to — so the
    file reaches the user whatever directory was named. ``named`` maps each
    output parameter given to what the model asked for. With no host folder
    nothing changes (a library caller keeps today's behaviour)."""
    from aec_agent_core._fileio import host_output_dir, into_working_folder
    folder = host_output_dir()
    if not folder or not isinstance(parameters, dict):
        return parameters, {}
    out, named = parameters, {}
    for key, is_dir in _OUTPUT_PARAMS:
        asked = parameters.get(key)
        if not isinstance(asked, str) or not asked.strip():
            continue
        named[key] = asked
        target = into_working_folder(asked, folder, is_dir=is_dir)
        if target != asked:
            if out is parameters:
                out = dict(parameters)
            out[key] = target
    return out, named


def _output_note(named: dict, parameters: dict) -> str:
    """What a writer's result says about where its file went."""
    from aec_agent_core._fileio import host_output_dir
    folder = host_output_dir() or ""
    moved = []
    for key, asked in named.items():
        went = str(parameters.get(key) or "")
        # Where the path would have gone as given (a bare name: into the
        # folder anyway; "/tmp/x" or an absolute path: somewhere else).
        meant = os.path.abspath(os.path.join(
            folder, os.path.expanduser(str(asked).strip())))
        if os.path.normcase(meant) != os.path.normcase(os.path.abspath(went)):
            moved.append(f"'{asked}' -> '{went}'")
    note = ("Written into this conversation's working folder, which is where "
            "the user receives files: it is attached to the conversation. "
            "Tell the user the file name; a /tmp path or a sandbox: link "
            "would not reach them.")
    if moved:
        note += (" The directory asked for was replaced by that folder (a "
                 "tool always writes there): " + "; ".join(moved) + ".")
    return note


class _NoInputs:
    """What a call records when no one asks how its inputs were given."""
    reported = ()
    took = None


@contextlib.contextmanager
def _no_inputs(given, defaults):
    yield _NoInputs()


def run_method(mod, agent_name: str, method: str, parameters: dict,
               attachments: dict = None, *, inputs=None, after=None):
    """Run ``mod.METHOD_REGISTRY[method](parameters)`` with the checks and
    notes of this module, and return its result (or ``{"error": ...}``).

    ``mod`` has ``METHOD_INFO`` and ``METHOD_REGISTRY``; ``agent_name``
    names it in an error. ``inputs(given, left_out)`` is a context manager
    that yields what the method reported of its inputs (``reported``, the
    names it called "default"; ``took``, name -> the value a left-out input
    took): the geotechnical modules' ``geotech_common.choices.call_inputs``;
    None records nothing. ``after(parameters, result)`` adjusts a result
    before its defaults are listed (dispatch checks a calc package's
    citations there).
    """
    from aec_agent_core._fileio import hide_working_folder
    from aec_agent_core.call_warnings import add_to_result, collecting
    # The Python warnings the method raises reach the agent with its result
    # (BACKLOG GM-316; aec_agent_core.call_warnings), also with an error.
    with collecting() as caught:
        result = _run_checked(mod, agent_name, method, parameters,
                              attachments, inputs, after)
    return hide_working_folder(add_to_result(result, caught.texts))


def _run_checked(mod, agent_name: str, method: str, parameters: dict,
                 attachments, inputs, after):
    """:func:`run_method`'s checks, the call and its notes; the result
    before working-folder paths are hidden."""
    given = set(parameters)
    try:
        _check_parameter_names(mod, method, parameters)
        if "attachment_key" in parameters:
            parameters = _resolve_attachment(
                parameters, attachments, _method_params(mod, method))
        parameters = _inputs_from_working_folder(parameters)
        parameters, named = _outputs_into_working_folder(parameters)
        # The names this call gave, so a module's choices say "given" for a
        # typed value equal to its default (GM-118).
        given |= set(parameters)
        call_inputs = inputs() if inputs is not None else _no_inputs
        try:
            with call_inputs(_given_names(mod, method, given),
                             _left_out(mod, method, given,
                                       parameters)) as recorded:
                result = mod.METHOD_REGISTRY[method](parameters)
        except TypeError as e:
            if not _SIGNATURE_ERROR.search(str(e)):
                raise
            return _parameter_error(mod, agent_name, method, parameters, e)
        if named and isinstance(result, dict) and "error" not in result:
            result = dict(result)
            result["output_note"] = _output_note(named, parameters)
        if after is not None:
            result = after(parameters, result)
        return _with_defaults_used(mod, method, given, result, parameters,
                                   recorded.reported, recorded.took)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
