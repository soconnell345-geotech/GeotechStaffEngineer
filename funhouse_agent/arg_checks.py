"""Argument checks shared by the agent's tools and the analysis adapters.

:func:`check_nested_keys` refuses a key a nested dict (a soil layer, a
markup's ``remove`` entry) carries that nothing reads, with a did-you-mean
from :func:`suggest_params`. Moved here from
:mod:`funhouse_agent.adapters` (INF-30 step 5), so a document tool can check
its arguments without loading the analysis adapters; the adapters import
these names from here.
"""


#: Unit suffixes an agent appends to a parameter name ('b_mm', 'fc_mpa'),
#: and those the adapters' own names carry ('phi_deg', 'su_ksf',
#: 'unit_weight_kN_per_m3', 'gamma_d_kNm3').
_UNIT_SUFFIXES = frozenset({
    "mm", "cm", "m", "km", "in", "ft", "kpa", "mpa", "gpa", "pa", "psi",
    "psf", "ksf", "tsf", "pci", "pcf", "kn", "knm", "knm3", "kip", "kips",
    "lb", "lbs", "kg", "deg", "degrees", "rad", "s", "sec", "day", "yr",
    "yrs", "years", "pct", "percent", "mph", "m2", "m3", "ft2", "ft3",
})

#: Of those, the ones an adapter's own name carries as a subscript, not a
#: unit ('E_s', 'R_s', 'f_s'): left on a valid name.
_SUBSCRIPT_SUFFIXES = frozenset({"s"})


def _without_unit(name: str, subscripts=frozenset()) -> str:
    """``name`` less its trailing unit, one unit or a compound one
    ('phi_deg' -> 'phi', 'unit_weight_kN_per_m3' -> 'unit_weight',
    'rate_mm_day' -> 'rate'); ``name`` itself when it ends in none. A suffix
    in ``subscripts`` is not taken for a unit."""
    parts = [p for p in str(name).lower().replace("-", "_").split("_") if p]
    units = _UNIT_SUFFIXES - frozenset(subscripts)
    if len(parts) < 2 or parts[-1] not in units:
        return str(name).lower()
    parts.pop()
    if len(parts) > 2 and parts[-1] == "per" and parts[-2] in units:
        del parts[-2:]
    elif len(parts) > 1 and parts[-1] in units:
        parts.pop()
    return "_".join(parts)

#: Domain words written more than one way; applied to a guessed name only to
#: rank suggestions, never to rename a parameter.
_WORD_SYNONYMS = (
    ("water_table", "gwt"), ("watertable", "gwt"), ("groundwater", "gwt"),
    ("ground_water", "gwt"), ("gwl", "gwt"),
)


def _param_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def suggest_params(unknown: str, valid, n: int = 2) -> list:
    """Valid parameter names closest to an unknown one ('x_label' -> 'xlabel',
    'fc_mpa' -> 'fc', 'water_table_depth' -> 'gwt_depth'), with or without
    the unit the valid name carries ('phi' -> 'phi_deg', 'B' -> 'b_m';
    GM-177). Suggestions only: nothing is renamed."""
    import difflib
    valid = list(valid)
    raw = str(unknown).lower()
    forms = {raw}
    for word, canon in _WORD_SYNONYMS:
        if word in raw:
            forms.add(raw.replace(word, canon))
    # 'E_s' (the soil's modulus) is not 'E' in seconds: offered 'E' it
    # would put the soil's modulus in axial_pile_capacity's pile modulus
    forms.add(_without_unit(raw, _SUBSCRIPT_SUFFIXES))
    keys = {_param_key(f) for f in forms}
    exact = [v for v in valid if _param_key(v) in keys]
    if exact:
        return exact[:n]
    # The same name with its unit left off, on either side: a reference
    # function takes 'phi_deg' where the analysis modules take 'phi'.
    bare = [v for v in valid
            if _param_key(_without_unit(v, _SUBSCRIPT_SUFFIXES)) in keys]
    # A one-letter tail taken for a unit may be a subscript: 'gamma' matched
    # 'gamma_m' with its 'm' read as metres, and the tool's other unit
    # weight is 'gamma_b'. Offer each name with the same stem and a
    # one-letter tail beside it (GM-197).
    stems = {_param_key(v.rsplit("_", 1)[0]) for v in bare
             if "_" in v and len(v.rsplit("_", 1)[1]) == 1}
    bare += [v for v in valid if v not in bare and "_" in v
             and len(v.rsplit("_", 1)[1]) == 1
             and _param_key(v.rsplit("_", 1)[0]) in stems]
    if bare:
        return bare[:n]
    # Spelling slips only ('unit_wieght'): a looser cutoff starts offering
    # a different quantity that shares a word ('base_inclination' would get
    # 'load_inclination'), which is worse than no suggestion.
    by_key = {_param_key(v): v for v in valid}
    near = []
    for k in keys:
        for hit in difflib.get_close_matches(k, list(by_key), n=n, cutoff=0.85):
            if by_key[hit] not in near:
                near.append(by_key[hit])
    return near[:n]


#: The same normalised form, for the nested-key check's ``hints`` lookup.
_norm_key = _param_key


def check_nested_keys(item, valid, *, method: str, label: str,
                      hints=None, notes=None) -> None:
    """Raise a clear ValueError when a nested dict (a soil layer, a nail, a
    pile, a load, ...) carries a key the module does not read.

    Without this check the key was read with ``.get`` and silently left out:
    a slope layer given ``c`` or ``cohesion`` for ``c_prime`` ran with
    c' = 0 (GM-52). The message names the item, the unknown key(s), the
    closest valid key(s) as a did-you-mean and the valid keys. Nothing is
    renamed.

    ``hints`` maps a normalised key ('cohesion') to the key(s) it may mean,
    for words the near-miss matcher cannot place; only those in ``valid``
    are offered, so one table can serve items of several kinds (when none
    is, the near-miss matcher is asked). An EMPTY tuple offers nothing: for
    a key whose spelling is close to a different quantity's ('N1_60' is not
    'N60'). ``notes`` maps a tuple of keys to a sentence added when every
    one of them is offered as a did-you-mean or was given and refused (to
    tell two meanings of one word apart, or say where a refused key IS
    read).
    """
    if not isinstance(item, dict):
        raise ValueError(f"{method}: {label} must be a dict of named values, "
                         f"got {type(item).__name__} {item!r}.")
    unknown = sorted((k for k in item if k not in valid), key=str)
    if not unknown:
        return
    parts, offered = [], set()
    for k in unknown:
        hinted = (hints or {}).get(_norm_key(k))
        if hinted is None:
            near = suggest_params(k, valid)
        else:
            near = [h for h in hinted if h in valid]
            if hinted and not near:
                near = suggest_params(k, valid)
        if near:
            parts.append(f"{k} -> {' or '.join(near)}")
            offered.update(near)
    hint = f" Did you mean: {'; '.join(parts)}?" if parts else ""
    involved = offered | set(unknown)
    said = []
    for keys, sentence in (notes or {}).items():
        if set(keys) <= involved and sentence not in said:
            said.append(sentence)
            hint += f" {sentence}"
    raise ValueError(
        f"{method}: {label} has unknown key(s) {unknown}.{hint} "
        f"Valid keys: {sorted(valid)}.")


def check_nested_list(items, valid, *, method: str, label: str,
                      hints=None, notes=None) -> None:
    """:func:`check_nested_keys` on every dict of a list parameter
    (``layers``, ``piles``, ``supports``, ...), each named by its index and,
    when it has one, its name or description: ``layers[2] ('Clay')``. A
    value that is not a list is left to the caller's own checks."""
    if not isinstance(items, (list, tuple)):
        return
    for i, item in enumerate(items):
        name = None
        if isinstance(item, dict):
            name = item.get("name") or item.get("description") \
                or item.get("label")
        tag = f" ('{str(name)[:40]}')" if name else ""
        check_nested_keys(item, valid, method=method,
                          label=f"{label}[{i}]{tag}", hints=hints,
                          notes=notes)
