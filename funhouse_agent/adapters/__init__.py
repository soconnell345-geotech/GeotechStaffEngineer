"""Adapter modules for funhouse_agent dispatch.

Each adapter bridges flat JSON parameters to structured analysis module APIs.
Modules are lazy-loaded: imports only happen when call_agent() is invoked.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Shared JSON-serialization helpers
# ---------------------------------------------------------------------------

def clean_value(v):
    """Convert numpy types and NaN to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def clean_result(result: dict) -> dict:
    """Recursively clean a result dict for JSON serialization."""
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [
                clean_result(item) if isinstance(item, dict)
                else clean_value(item)
                for item in v
            ]
        elif isinstance(v, dict):
            cleaned[k] = clean_result(v)
        else:
            cleaned[k] = clean_value(v)
    return cleaned


# ---------------------------------------------------------------------------
# Shared parameter-ergonomics helpers
# ---------------------------------------------------------------------------

def apply_aliases(params: dict, aliases: dict) -> dict:
    """Return a copy of ``params`` with alias keys renamed to canonical keys.

    ``aliases`` maps alias_name -> canonical_name. The canonical key wins if
    both are present (the alias is silently dropped in that case).
    """
    if not any(a in params for a in aliases):
        return params
    out = dict(params)
    for alias, canonical in aliases.items():
        if alias in out:
            val = out.pop(alias)
            out.setdefault(canonical, val)
    return out


def require_params(params: dict, required, *, method: str, valid=None):
    """Raise a clear ValueError if any of ``required`` keys is missing.

    Replaces raw ``KeyError: 'x'`` tracebacks with an actionable message that
    lists the missing and valid parameter names.
    """
    missing = [k for k in required if params.get(k) is None]
    if missing:
        msg = (f"{method}: missing required parameter(s) {missing}.")
        if valid:
            msg += f" Valid parameters: {sorted(valid)}."
        raise ValueError(msg)


def require_keys(item: dict, required, *, method: str, item_label: str = "layers[]"):
    """Raise a clear ValueError if a nested dict (e.g. a soil layer) lacks keys."""
    missing = [k for k in required if item.get(k) is None]
    if missing:
        raise ValueError(
            f"{method}: each {item_label} dict requires {list(required)}; "
            f"missing {missing} (got keys {sorted(item.keys())})."
        )


#: Unit suffixes an agent appends to a parameter name ('b_mm', 'fc_mpa').
_UNIT_SUFFIXES = frozenset({
    "mm", "cm", "m", "km", "in", "ft", "kpa", "mpa", "gpa", "pa", "psi",
    "psf", "ksf", "pci", "pcf", "kn", "knm", "kip", "kips", "deg", "degrees",
    "rad", "s", "sec", "yr", "yrs", "years", "pct", "percent",
})

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
    'fc_mpa' -> 'fc', 'water_table_depth' -> 'gwt_depth'). Suggestions only:
    nothing is renamed."""
    import difflib
    valid = list(valid)
    raw = str(unknown).lower()
    forms = {raw}
    for word, canon in _WORD_SYNONYMS:
        if word in raw:
            forms.add(raw.replace(word, canon))
    parts = [p for p in raw.replace("-", "_").split("_") if p]
    if len(parts) > 1 and parts[-1] in _UNIT_SUFFIXES:
        forms.add("_".join(parts[:-1]))
    keys = {_param_key(f) for f in forms}
    exact = [v for v in valid if _param_key(v) in keys]
    if exact:
        return exact[:n]
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


def field_keys(cls, *extra) -> tuple:
    """The keys a nested dict built into dataclass ``cls`` may carry: its
    init fields, which the adapter passes through one for one, plus
    ``extra`` (aliases the adapter also reads)."""
    import dataclasses
    return tuple(f.name for f in dataclasses.fields(cls) if f.init) + extra


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


def reject_unknown_params(params: dict, valid, *, method: str, aliases=()):
    """Raise a clear ValueError naming unknown keys and listing valid ones.

    Use for adapters that pass ``**params`` straight into a module function,
    where an invented parameter name would otherwise surface as a bare
    ``TypeError: unexpected keyword argument``. Each unknown name that is a
    near miss of a valid one gets a did-you-mean.
    """
    unknown = [k for k in params if k not in valid and k not in aliases]
    if unknown:
        hints = []
        for k in sorted(unknown):
            near = suggest_params(k, valid)
            if near:
                hints.append(f"{k} -> {' or '.join(near)}")
        hint = f" Did you mean: {'; '.join(hints)}?" if hints else ""
        raise ValueError(
            f"{method}: unknown parameter(s) {sorted(unknown)}.{hint} "
            f"Valid parameters: {sorted(valid)}. describe_method gives each "
            f"one's meaning and units."
        )


# ---------------------------------------------------------------------------
# Figure-output helpers (save a plot's HTML to a file instead of returning it)
# ---------------------------------------------------------------------------

def mark_required(method_info: dict, required: dict) -> dict:
    """Give every parameter of ``method_info`` an explicit ``required`` flag
    (True for the names in ``required[method]``, else False) and a
    ``description`` (copied from an older-style ``brief`` when it has none),
    so every adapter documents its parameters ONE way. Live smoke wave 1
    (G7): four adapters carried no ``required`` flags, and models spent
    calls discovering what the code demanded. Returns ``method_info``."""
    for method, info in method_info.items():
        must = set(required.get(method, ()))
        for name, spec in (info.get("parameters") or {}).items():
            if not isinstance(spec, dict):
                continue
            spec["required"] = name in must
            if "description" not in spec and "brief" in spec:
                spec["description"] = spec.pop("brief")
    return method_info


def figure_output_format(params: dict, default: str = "metadata") -> str:
    """Resolve a plot method's ``output_format``, honoring ``output_path``.

    A caller that passes ``output_path`` wants the figure written to a file, so
    the HTML must be produced regardless of ``output_format`` — this forces
    ``"html"`` in that case. Otherwise the explicit ``output_format`` (or
    ``default``) is used.
    """
    if params.get("output_path"):
        return "html"
    return params.get("output_format", default)


def write_plotly_sidecar(output_path, figure_json):
    """Write ``figure_json`` as ``<name>.plotly.json`` beside ``output_path``.

    One Plotly figure serialized with ``figure.to_json()``, saved next to the
    file it belongs to (the same stem, whatever that file's extension) so a
    front-end can render it natively with ``st.plotly_chart`` instead of an
    HTML iframe or a static image. Returns ``{"plotly_json_path": ...}`` on a
    verified write and ``None`` otherwise — the sidecar is always additive and
    best-effort, so a caller merges the dict and never fails on it.

    The naming rule and the :func:`funhouse_agent._fileio.save_verified` call
    live HERE only, so ``/Workspace`` durability and read-back verification
    come free for every producer of a sidecar.
    """
    if not output_path or not figure_json:
        return None
    import os as _os
    try:
        from funhouse_agent._fileio import save_verified
        sidecar = _os.path.splitext(str(output_path))[0] + ".plotly.json"
        saved = save_verified(sidecar, figure_json)
    except Exception:                                      # noqa: BLE001
        return None
    if saved.get("file_exists"):
        return {"plotly_json_path": saved.get("saved", sidecar)}
    return None


def save_html_output(result: dict, params: dict, *, html_key: str = "html",
                     figure_json: str = None) -> dict:
    """Save a plot result's HTML to ``params['output_path']`` when given.

    A self-contained Plotly figure serialized to HTML is ~MB; returning it
    inline burns model context and several turns of "now save it to a file".
    When ``output_path`` is set and ``result`` carries the figure HTML under
    ``html_key``, this writes the HTML to that path (verified) and REPLACES the
    blob with a compact confirmation (``output_path``/``file_exists``/size +
    ``renderer_note``). No-op when there is no ``output_path`` or no HTML.

    When ``figure_json`` (a Plotly ``figure.to_json()`` string) is supplied, a
    compact ``<name>.plotly.json`` sidecar is written next to the HTML so a
    front-end (e.g. the webapp) can render the figure natively with
    ``st.plotly_chart`` instead of an HTML iframe (see
    :func:`write_plotly_sidecar`). The sidecar is additive and best-effort — a
    sidecar failure never affects the HTML save. Its path is returned under
    ``plotly_json_path``.

    Reuses :func:`funhouse_agent._fileio.save_verified`, so a ``/Workspace``
    target goes through the durable Databricks workspace API and every write is
    read-back-verified with a ``/tmp`` rescue on failure.
    """
    output_path = params.get("output_path")
    if not output_path or html_key not in result:
        return result
    from funhouse_agent._fileio import resolve_output_path, save_verified

    # A bare/relative name defaults INTO the host working folder (e.g. the
    # webapp conversation's files/ dir); an absolute path is honored as given.
    output_path = resolve_output_path(str(output_path))
    html = result.pop(html_key)
    saved = save_verified(output_path, html)
    # Present the saved path as ``output_path`` to match the calc_package
    # response shape (output_path / file_exists / file_size_bytes).
    if "saved" in saved:
        saved["output_path"] = saved.pop("saved")
    result.update(saved)
    if saved.get("file_exists"):
        result["renderer_note"] = (
            "The self-contained figure HTML was written to the file above "
            "instead of returned inline (open it in a browser). Omit "
            "output_path to receive the HTML in the response."
        )
        side = write_plotly_sidecar(output_path, figure_json)
        if side:
            result.update(side)
    else:
        result["renderer_note"] = (
            "The figure HTML could NOT be stored at output_path — see 'error' "
            "and 'rescue_path': the rescue copy, by the name the error gives, "
            "is the file."
        )
    return result


# ---------------------------------------------------------------------------
# Module registry — defines all available modules and their adapter paths
# ---------------------------------------------------------------------------

MODULE_REGISTRY = {
    "bearing_capacity": {
        "adapter": "funhouse_agent.adapters.bearing_capacity",
        "brief": "Shallow foundation bearing capacity (Terzaghi/Vesic/Meyerhof/Hansen)",
    },
    "settlement": {
        "adapter": "funhouse_agent.adapters.settlement",
        "brief": "Foundation settlement (elastic, Schmertmann, consolidation)",
    },
    "slope_stability": {
        "adapter": "funhouse_agent.adapters.slope_stability",
        "brief": "Slope stability: LE methods + search, rapid drawdown, infinite slope, seismic ky/Newmark, nails/anchors/piles, FOSM/MC",
    },
    "seismic_geotech": {
        "adapter": "funhouse_agent.adapters.seismic_geotech",
        "brief": "Seismic evaluation (site class, M-O pressure); liquefaction triggering is in 'liquefaction'",
    },
    "liquefaction": {
        "adapter": "funhouse_agent.adapters.liquefaction_adapter",
        "brief": "Liquefaction triggering by input: CPT -> Boulanger & Idriss 2014 + LPI/LSN/LDI; SPT -> B&I 2014 or NCEER/Youd 2001 (method='nceer2001')",
    },
    "retaining_walls": {
        "adapter": "funhouse_agent.adapters.retaining_walls",
        "brief": "Retaining walls (cantilever + MSE, GEC-11)",
    },
    "axial_pile": {
        "adapter": "funhouse_agent.adapters.axial_pile",
        "brief": "Driven pile axial capacity (Nordlund/Tomlinson/Beta)",
    },
    "drilled_shaft": {
        "adapter": "funhouse_agent.adapters.drilled_shaft",
        "brief": "Drilled shaft capacity (GEC-10 alpha/beta/rock socket, LRFD)",
    },
    "sheet_pile": {
        "adapter": "funhouse_agent.adapters.sheet_pile",
        "brief": "Sheet pile walls (cantilever and anchored)",
    },
    "lateral_pile": {
        "adapter": "funhouse_agent.adapters.lateral_pile",
        "brief": "Lateral pile analysis (COM624P, 8 p-y models, FD solver); composite pile section EI",
    },
    "pile_group": {
        "adapter": "funhouse_agent.adapters.pile_group",
        "brief": "Pile group analysis (rigid cap, 6-DOF, Converse-Labarre)",
    },
    "ground_improvement": {
        "adapter": "funhouse_agent.adapters.ground_improvement",
        "brief": "Ground improvement (aggregate piers, wick drains, vibro, GEC-13)",
    },
    "wave_equation": {
        "adapter": "funhouse_agent.adapters.wave_equation",
        "brief": "Smith 1-D wave equation (bearing graph, drivability)",
    },
    "downdrag": {
        "adapter": "funhouse_agent.adapters.downdrag",
        "brief": "Pile downdrag (Fellenius NP + CGPR-56 methods, groups)",
    },
    "soe": {
        "adapter": "funhouse_agent.adapters.soe",
        "brief": "Support of excavation (braced/cantilever walls, stability, anchors)",
    },
    "pavement_design": {
        "adapter": "funhouse_agent.adapters.pavement_design_adapter",
        "brief": "Pavement design: AASHTO 1993 and UFC 3-250-01 (flexible SN/CBR, rigid slab D, ESALs)",
    },
    "dxf_export": {
        "adapter": "funhouse_agent.adapters.dxf_export",
        "brief": "Export cross-section geometry to DXF file format",
    },
    "calc_package": {
        "adapter": "funhouse_agent.adapters.calc_package",
        "brief": "Calc packages (HTML/LaTeX/PDF) for 15 modules with CSV calculation tables, their figures as PNGs, html_to_pdf for bespoke reports",
    },
    "profile_figure": {
        "adapter": "funhouse_agent.adapters.profile_figure_adapter",
        "brief": "PNG figures: subsurface profile schematic (layers, water, foundation); plot_data x/y or depth plots",
    },
    "worked_examples": {
        "adapter": "funhouse_agent.adapters.worked_examples_adapter",
        "brief": "Published worked examples (GEC/Caltrans/AASHTO/UFC/Slide2) with method calls and answers; consult before nontrivial designs",
    },
    "pystra": {
        "adapter": "funhouse_agent.adapters.pystra_adapter",
        "brief": "Structural reliability analysis (FORM/SORM/Monte Carlo)",
    },
    "reliability": {
        "adapter": "funhouse_agent.adapters.reliability_adapter",
        "brief": "GEOTECH reliability: FOSM (Duncan 2000), Rosenblueth PEM, Monte Carlo, FORM, published COVs, Vanmarcke averaging. SORM: pystra",
    },
    "salib": {
        "adapter": "funhouse_agent.adapters.salib_adapter",
        "brief": "Sensitivity analysis (Sobol, Morris screening)",
    },
    "dxf_import": {
        "adapter": "funhouse_agent.adapters.dxf_import_adapter",
        "brief": "DXF CAD import for slope stability + FEM (discover, parse, build geometry)",
    },
    "pdf_import": {
        "adapter": "funhouse_agent.adapters.pdf_import_adapter",
        "brief": "PDF cross-section import (PyMuPDF vector extraction, content discovery)",
    },
    "drawing_ir": {
        "adapter": "funhouse_agent.adapters.drawing_ir_adapter",
        "brief": "Drawing geometry from DXF/PDF/raster: slice queries, leader/dimension/title-block/bubble/cloud proposals, drawing-set search, region snip",
    },
    "opensees": {
        "adapter": "funhouse_agent.adapters.opensees_adapter",
        "brief": "OpenSees FE analyses (PM4Sand DSS, 1D site response)",
    },
    "pystrata": {
        "adapter": "funhouse_agent.adapters.pystrata_adapter",
        "brief": "1D site response (equivalent-linear and linear elastic, SHAKE-type)",
    },
    "liquepy": {
        "adapter": "funhouse_agent.adapters.liquepy_adapter",
        "brief": "liquepy B&I 2014 CPT/SPT triggering, LPI/LSN/LDI, CPT correlations (Vs/Dr/su/k); routine triggering: 'liquefaction'",
    },
    "seismic_signals": {
        "adapter": "funhouse_agent.adapters.seismic_signals_adapter",
        "brief": "Earthquake signal processing (response spectra, intensity measures, RotD, filtering)",
    },
    "fem2d": {
        "adapter": "funhouse_agent.adapters.fem2d_adapter",
        "brief": "2D plane-strain FEM (gravity, foundation, slope SRM, excavation, seepage, consolidation, staged)",
    },
    "gstools": {
        "adapter": "funhouse_agent.adapters.gstools_adapter",
        "brief": "Geostatistics: kriging (mean surface), variograms, random fields; conditional simulation for P(volume/area over a footprint)",
    },
    "section_props": {
        "adapter": "funhouse_agent.adapters.section_props_adapter",
        "brief": "Cross-section properties: A, I, Z, S, J, warping; steel shapes + polygons",
    },
    "concrete_props": {
        "adapter": "funhouse_agent.adapters.concrete_props_adapter",
        "brief": "RC section: cracked props, nominal Mn, N-M interaction",
    },
    "pynite": {
        "adapter": "funhouse_agent.adapters.pynite_adapter",
        "brief": "Elastic 2D/3D frames + continuous beams: reactions, M/V/deflection envelopes",
    },
    "subsurface": {
        "adapter": "funhouse_agent.adapters.subsurface_adapter",
        "brief": "Subsurface data: DIGGS read/validate/write (from logs you read), AGS4, GEF/BRO-XML CPT+borehole; Plotly depth/Atterberg/plan/section plots",
    },
    # --- geotech-references agents (reference modules + cross-reference DB) ---
    "reference_db": {
        "adapter": "funhouse_agent.adapters.reference_db_adapter",
        "brief": "Text search over the reference chapters (DM7, GEC, micropile); use first to scope, then reference_get",
    },
    "figure_db": {
        "adapter": "funhouse_agent.adapters.figure_db_adapter",
        "brief": "Find a reference chart (DM7/GEC figure catalogs) by meaning; read values off it with read_reference_figure",
    },
    "dm7": {
        "adapter": "funhouse_agent.adapters.dm7_adapter",
        "brief": "NAVFAC DM7 equations (340+): soil classification, stresses, settlement, seepage, foundations",
    },
    "em_2104": {
        "adapter": "funhouse_agent.adapters.em_2104_adapter",
        "brief": "EM 1110-2-2104 RC hydraulic structures (USACE): loads, service limits, flexure-axial (App B/D), shear",
    },
    "em_2107": {
        "adapter": "funhouse_agent.adapters.em_2107_adapter",
        "brief": "EM 1110-2-2107 hydraulic steel structures (USACE): LRFD loads, seismic amplification, Tainter gates (App F)",
    },
    "gec4": {
        "adapter": "funhouse_agent.adapters.gec4_adapter",
        "brief": "GEC-4 ground anchors (FHWA-IF-99-015): bond stress, load transfer, corrosion protection, anchored walls (ASD)",
    },
    "gec5": {
        "adapter": "funhouse_agent.adapters.gec5_adapter",
        "brief": "GEC-5 site characterization (FHWA NHI-16-072): investigation planning, classification, strength, stiffness, hazards",
    },
    "gec6": {
        "adapter": "funhouse_agent.adapters.gec6_adapter",
        "brief": "GEC-6 shallow foundations reference (FHWA-SA-02-054)",
    },
    "gec7": {
        "adapter": "funhouse_agent.adapters.gec7_adapter",
        "brief": "GEC-7 soil nail walls reference (FHWA-NHI-14-007)",
    },
    "gec8": {
        "adapter": "funhouse_agent.adapters.gec8_adapter",
        "brief": "GEC-8 CFA piles (FHWA-HIF-07-03): DD pile capacity (NeSmith), ASD, grout volume factor, p-multipliers, group efficiency",
    },
    "gec9": {
        "adapter": "funhouse_agent.adapters.gec9_adapter",
        "brief": "GEC-9 laterally loaded piles (FHWA-HIF-18-031): resistance factors, p-multipliers, p-y parameters (stiff clay, sand)",
    },
    "gec10": {
        "adapter": "funhouse_agent.adapters.gec10_adapter",
        "brief": "GEC-10 drilled shafts reference (FHWA-NHI-18-024)",
    },
    "gec11": {
        "adapter": "funhouse_agent.adapters.gec11_adapter",
        "brief": "GEC-11 MSE walls & reinforced soil slopes (FHWA-NHI-10-024)",
    },
    "gec12": {
        "adapter": "funhouse_agent.adapters.gec12_adapter",
        "brief": "GEC-12 driven piles reference (FHWA-NHI-16-009)",
    },
    "gec13": {
        "adapter": "funhouse_agent.adapters.gec13_adapter",
        "brief": "GEC-13 ground modification reference (FHWA-NHI-16-027)",
    },
    "gec14": {
        "adapter": "funhouse_agent.adapters.gec14_adapter",
        "brief": "GEC-14 QA for geotech reporting documents (FHWA-HIF-17-016): GRD types, ACMs/ADMs, QA process, checklists",
    },
    "micropile": {
        "adapter": "funhouse_agent.adapters.micropile_adapter",
        "brief": "Micropile design reference (FHWA-NHI-05-039): bond stresses",
    },
    "ufc_backfill": {
        "adapter": "funhouse_agent.adapters.ufc_backfill_adapter",
        "brief": "UFC 3-220-04N backfill design (compaction, filter criteria, drainage)",
    },
    "ufc_expansive": {
        "adapter": "funhouse_agent.adapters.ufc_expansive_adapter",
        "brief": "UFC 3-220-07 expansive soils (swell potential, heave, pier design)",
    },
    "ufc_pavement": {
        "adapter": "funhouse_agent.adapters.ufc_pavement_adapter",
        "brief": "UFC 3-250-01 roads/parking: CBR flexible (curve E-1), rigid (Eq 13-1), overlays, frost, drainage, joints, mixed traffic (US)",
    },
    "ufc_stabilization": {
        "adapter": "funhouse_agent.adapters.ufc_stabilization_adapter",
        "brief": "UFC 3-250-11 soil stabilization: additive selection, strength/durability criteria, equivalency factors",
    },
    "ufc_flexible_practice": {
        "adapter": "funhouse_agent.adapters.ufc_flexible_practice_adapter",
        "brief": "UFC 3-250-03 asphalt practice: HMA gradations, Marshall/Superpave, spray rates, seal coats",
    },
    "ufc_concrete_practice": {
        "adapter": "funhouse_agent.adapters.ufc_concrete_practice_adapter",
        "brief": "UFC 3-250-04 concrete practice: materials tables, dowel tolerances, joint spacing, RCC",
    },
    "ufc_structural": {
        "adapter": "funhouse_agent.adapters.ufc_structural_adapter",
        "brief": "UFC 3-301-01 DoD structural (IBC/ASCE 7-22 mods): risk categories, live loads, Table 3-1 seismic systems",
    },
    "ufc_collapse": {
        "adapter": "funhouse_agent.adapters.ufc_collapse_adapter",
        "brief": "UFC 4-023-03 progressive collapse: tie forces, Alternate Path LRFD, LIF/DIF, ELR, RC/steel m-factors",
    },
    "gsa_collapse": {
        "adapter": "funhouse_agent.adapters.gsa_collapse_adapter",
        "brief": "GSA Alternate Path 2016 (civilian sibling of UFC 4-023-03): FSL applicability, AP-only design, redundancy",
    },
    "wood_handbook": {
        "adapter": "funhouse_agent.adapters.wood_handbook_adapter",
        "brief": "USDA Wood Handbook: clear-wood properties, EMC/shrinkage, Ch 9 stress/deformation/stability, fastener yield model",
    },
    "fema_p2082": {
        "adapter": "funhouse_agent.adapters.fema_p2082_adapter",
        "brief": "FEMA P-2082 (2020 NEHRP): Vs30 site classes A-F incl. BC/CD/DE, SDS/SD1, two-period spectrum, SDC, Risk Category",
    },
    "california_trenching": {
        "adapter": "funhouse_agent.adapters.california_trenching_adapter",
        "brief": "Caltrans Trenching and Shoring Manual: Cal/OSHA soil types/slopes, log-spiral Kp, apparent pressures, bottom heave (US)",
    },
    "fhwa_pavements": {
        "adapter": "funhouse_agent.adapters.fhwa_pavements_adapter",
        "brief": "FHWA-NHI-05-037 Geotechnical Aspects of Pavements: Mr from CBR/R/DCP, drainage, frost, swell, stabilization (US)",
    },
    "eurocode_7_1": {
        "adapter": "funhouse_agent.adapters.eurocode_7_1_adapter",
        "brief": "EN 1997-1 (Eurocode 7-1): partial factors + DA1-DA3 (Annex A); earth pressure, bearing, PMT, settlement, rock, movement (Annexes C-H). SI",
    },
    "eurocode_7_2": {
        "adapter": "funhouse_agent.adapters.eurocode_7_2_adapter",
        "brief": "EN 1997-2 (Eurocode 7-2): CPT, Menard PMT, SPT, dynamic probing, vane, DMT, plate load correlations; cu/Eoed; lab minima. SI",
    },
    "aashto_1993": {
        "adapter": "funhouse_agent.adapters.aashto_1993_adapter",
        "brief": "AASHTO 1993 Pavement Design Guide: flexible SN, rigid slab D, or W18 for either; layer coeffs, roadbed Mr, ZR/So, mi/Cd, ESALs (US)",
    },
}
