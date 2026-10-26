"""Adapter modules for funhouse_agent dispatch.

Each adapter bridges flat JSON parameters to structured analysis module APIs.
Modules are lazy-loaded: imports only happen when call_agent() is invoked.
"""

import importlib.util
import math
import numpy as np

# The nested-key check and the name matcher it uses are in
# funhouse_agent.arg_checks (INF-30 step 5), which both pages use; the
# adapters import them from there.
from funhouse_agent.arg_checks import (  # noqa: F401
    _SUBSCRIPT_SUFFIXES,
    _UNIT_SUFFIXES,
    _WORD_SYNONYMS,
    _norm_key,
    _param_key,
    _without_unit,
    check_nested_keys,
    check_nested_list,
    suggest_params,
)


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


def field_keys(cls, *extra) -> tuple:
    """The keys a nested dict built into dataclass ``cls`` may carry: its
    init fields, which the adapter passes through one for one, plus
    ``extra`` (aliases the adapter also reads)."""
    import dataclasses
    return tuple(f.name for f in dataclasses.fields(cls) if f.init) + extra


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


def refused_reason(spec) -> str:
    """The reason of a ``refused_parameters`` entry: a string, or
    ``{"reason": ..., "accepts": [values]}``."""
    return spec if isinstance(spec, str) else str(spec.get("reason", ""))


def accepted_as_unused(spec, value) -> bool:
    """Whether a refused name's value is one its entry ``accepts`` because
    it means "not used" (a level backfill's slope of 0), so it passes
    silently instead of costing a retry."""
    if not isinstance(spec, dict) or isinstance(value, (list, dict)):
        return False
    return any(value == v and type(value) is not bool
               for v in spec.get("accepts", ()))


def without_unused(params: dict, refused: dict) -> dict:
    """``params`` without the refused names given a value their entry
    accepts as "not used" (GM-110); the rest is left to the checks."""
    drop = [k for k, v in params.items()
            if k in refused and accepted_as_unused(refused[k], v)]
    if not drop:
        return params
    return {k: v for k, v in params.items() if k not in drop}


def declare_aliases(method_info: dict, aliases: dict) -> dict:
    """Record, per method, the other spellings of its parameters the
    adapter reads (``{method: {alias: parameter}}``) as the method's
    ``parameter_aliases``. Dispatch refuses a top-level key that is neither
    a parameter nor one of these (BACKLOG GM-110), so an alias the adapter
    maps must be declared here. An alias of something the method does not
    take is left out (one alias table often serves several methods), so it
    is refused by name. Returns ``method_info``."""
    for method, table in aliases.items():
        info = method_info[method]
        params = info.get("parameters") or {}
        own = info.setdefault("parameter_aliases", {})
        own.update({a: c for a, c in table.items()
                    if a != c and a not in params and c in params})
    return method_info


def declare_applies_when(method_info: dict, table: dict) -> dict:
    """Record when each parameter's default applies to a call
    (``{method: {parameter: condition}}``, BACKLOG GM-119): the
    parameter's ``applies_when``, which ``dispatch.applies`` reads, so a
    result's ``defaults_used`` lists only the defaults the run used. A
    condition maps other parameters to a list of values, "set" or "unset"
    (a list of such maps: any one of them); False means the method never
    uses it. The parameter's spec is copied first (one spec dict often
    serves several methods). A name the method does not declare raises, so
    a slip in the table fails on import. Returns ``method_info``."""
    for method, conditions in table.items():
        params = method_info[method]["parameters"]
        for name, condition in conditions.items():
            if name not in params:
                raise KeyError(f"{method}: no parameter {name!r}")
            alternatives = (condition if isinstance(condition, list)
                            else [condition or {}])
            for other in (k for c in alternatives for k in c):
                if other not in params:
                    raise KeyError(f"{method}.{name}: applies_when names "
                                   f"no parameter {other!r}")
            params[name] = {**params[name], "applies_when": condition}
    return method_info


def declare_when_absent(method_info: dict, table: dict) -> dict:
    """Record what a call that leaves out an optional input with no default
    is analysed with (``{method: {parameter: case}}``, BACKLOG GM-165), so
    a result's ``defaults_used`` says so and a calc package prints it under
    Assumed Inputs. ``case`` is either ``(what the run takes, its basis)``
    -- the parameter's ``when_absent``, e.g. ("no water table", "...") --
    or the name of another parameter whose value in force the method takes
    in its place, the parameter's ``default_from`` (a foundation friction
    angle that defaults to the backfill's). The spec is copied first. A
    name the method does not declare, a parameter with a default, or an
    empty case raises, so a slip in the table fails on import. Returns
    ``method_info``."""
    for method, cases in table.items():
        params = method_info[method]["parameters"]
        for name, case in cases.items():
            if name not in params:
                raise KeyError(f"{method}: no parameter {name!r}")
            spec = params[name]
            if spec.get("default") is not None \
                    or spec.get("default_depends_on"):
                raise ValueError(f"{method}.{name}: has a default; a case "
                                 "is for an input with none")
            if isinstance(case, str):
                if case not in params or case == name:
                    raise KeyError(f"{method}.{name}: default_from names "
                                   f"no other parameter {case!r}")
                params[name] = {**spec, "default_from": case}
                continue
            what, basis = case
            if not (str(what).strip() and str(basis).strip()):
                raise ValueError(f"{method}.{name}: a case needs what the "
                                 "run takes and its basis")
            params[name] = {**spec, "when_absent": {"case": what,
                                                    "basis": basis}}
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
        "brief": "Shallow foundations (Terzaghi/Vesic/Meyerhof/Hansen)",
    },
    "settlement": {
        "adapter": "funhouse_agent.adapters.settlement",
        "brief": "Foundations (elastic, Schmertmann, consolidation)",
    },
    "slope_stability": {
        "adapter": "funhouse_agent.adapters.slope_stability",
        "brief": "LE methods + search, rapid drawdown, infinite slope, seismic ky/Newmark, nails/anchors/piles, FOSM/MC",
    },
    "seismic_geotech": {
        "adapter": "funhouse_agent.adapters.seismic_geotech",
        "brief": "Site class, M-O pressure; liquefaction triggering: 'liquefaction'",
    },
    "liquefaction": {
        "adapter": "funhouse_agent.adapters.liquefaction_adapter",
        "brief": "Triggering: CPT -> Boulanger & Idriss 2014 + LPI/LSN/LDI, FS and strain by depth; SPT -> B&I 2014 or NCEER/Youd 2001 (method='nceer2001'); feeds post-liquefaction downdrag",
    },
    "retaining_walls": {
        "adapter": "funhouse_agent.adapters.retaining_walls",
        "brief": "Cantilever + MSE walls (GEC-11); earth pressure coefficients: Rankine and Coulomb Ka/Kp, log-spiral Kp (wall and ground slope), at-rest K0 behind a slope",
    },
    "axial_pile": {
        "adapter": "funhouse_agent.adapters.axial_pile",
        "brief": "Driven pile capacity (Nordlund/Tomlinson/Beta)",
    },
    "drilled_shaft": {
        "adapter": "funhouse_agent.adapters.drilled_shaft",
        "brief": "Capacity (GEC-10 alpha/beta/rock socket, LRFD)",
    },
    "sheet_pile": {
        "adapter": "funhouse_agent.adapters.sheet_pile",
        "brief": "Cantilever and anchored walls",
    },
    "lateral_pile": {
        "adapter": "funhouse_agent.adapters.lateral_pile",
        "brief": "COM624P (8 p-y models, FD solver); composite pile section EI",
    },
    "pile_group": {
        "adapter": "funhouse_agent.adapters.pile_group",
        "brief": "Rigid cap, 6-DOF, Converse-Labarre",
    },
    "ground_improvement": {
        "adapter": "funhouse_agent.adapters.ground_improvement",
        "brief": "Aggregate piers, wick drains, vibro (GEC-13)",
    },
    "wave_equation": {
        "adapter": "funhouse_agent.adapters.wave_equation",
        "brief": "Smith 1-D (bearing graph, drivability)",
    },
    "downdrag": {
        "adapter": "funhouse_agent.adapters.downdrag",
        "brief": "Pile downdrag (Fellenius NP + CGPR-56 methods, groups), post-liquefaction downdrag (GEC-10)",
    },
    "soe": {
        "adapter": "funhouse_agent.adapters.soe",
        "brief": "Support of excavation (braced/cantilever walls, stability, anchors)",
    },
    "pavement_design": {
        "adapter": "funhouse_agent.adapters.pavement_design_adapter",
        "brief": "AASHTO 1993 and UFC 3-250-01 (flexible SN/CBR, rigid slab D, ESALs)",
    },
    "dxf_export": {
        "adapter": "funhouse_agent.adapters.dxf_export",
        "brief": "Cross-section geometry to a DXF file",
    },
    "calc_package": {
        "adapter": "funhouse_agent.adapters.calc_package",
        "brief": "HTML/LaTeX/PDF for 15 modules with CSV calculation tables, figures as PNGs; html_to_pdf for bespoke reports",
    },
    "profile_figure": {
        "adapter": "funhouse_agent.adapters.profile_figure_adapter",
        "brief": "PNGs: subsurface profile schematic (layers, water, foundation); plot_data x/y or depth plots",
    },
    "worked_examples": {
        "adapter": "funhouse_agent.adapters.worked_examples_adapter",
        "brief": "Published (GEC/Caltrans/AASHTO/UFC/Slide2) with method calls and answers; consult before nontrivial designs",
    },
    "pystra": {
        "adapter": "funhouse_agent.adapters.pystra_adapter",
        "brief": "Structural reliability (FORM/SORM/Monte Carlo)",
    },
    "reliability": {
        "adapter": "funhouse_agent.adapters.reliability_adapter",
        "brief": "GEOTECH: FOSM (Duncan 2000), Rosenblueth PEM, Monte Carlo, FORM, published COVs, Vanmarcke averaging; SORM: 'pystra'",
    },
    "salib": {
        "adapter": "funhouse_agent.adapters.salib_adapter",
        "brief": "Sensitivity (Sobol, Morris screening)",
    },
    "dxf_import": {
        "adapter": "funhouse_agent.adapters.dxf_import_adapter",
        "brief": "CAD import for slope stability + FEM (discover, parse, build geometry)",
    },
    "pdf_import": {
        "adapter": "funhouse_agent.adapters.pdf_import_adapter",
        "brief": "Cross-sections (vector extraction, content discovery)",
    },
    "drawing_ir": {
        "adapter": "funhouse_agent.adapters.drawing_ir_adapter",
        "brief": "DXF/PDF/raster geometry: slice queries, leader/dimension/title-block/bubble/cloud proposals, drawing-set search, region snip",
    },
    "opensees": {
        "adapter": "funhouse_agent.adapters.opensees_adapter",
        "brief": "FE analyses (PM4Sand DSS, 1D site response)",
    },
    "pystrata": {
        "adapter": "funhouse_agent.adapters.pystrata_adapter",
        "brief": "1D site response (equivalent-linear and linear elastic, SHAKE-type)",
    },
    "liquepy": {
        "adapter": "funhouse_agent.adapters.liquepy_adapter",
        "brief": "B&I 2014 CPT/SPT triggering, LPI/LSN/LDI, CPT correlations (Vs/Dr/su/k); routine triggering: 'liquefaction'",
    },
    "seismic_signals": {
        "adapter": "funhouse_agent.adapters.seismic_signals_adapter",
        "brief": "Earthquake records (response spectra, intensity measures, RotD, filtering)",
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
        "brief": "A, I, Z, S, J, warping; steel shapes + polygons",
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
        "brief": "DIGGS read/validate/write (from logs you read), AGS4, GEF/BRO-XML CPT+borehole; Plotly depth/Atterberg/plan/section plots",
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
        "brief": "EM 1110-2-2104 RC hydraulic structures (USACE): loads, service limits, flexure-axial, shear",
    },
    "em_2107": {
        "adapter": "funhouse_agent.adapters.em_2107_adapter",
        "brief": "EM 1110-2-2107 hydraulic steel structures (USACE): LRFD loads, seismic amplification, Tainter gates",
    },
    "gec4": {
        "adapter": "funhouse_agent.adapters.gec4_adapter",
        "brief": "Ground anchors (FHWA-IF-99-015): bond stress, load transfer, corrosion protection, anchored walls (ASD)",
    },
    "gec5": {
        "adapter": "funhouse_agent.adapters.gec5_adapter",
        "brief": "Site characterization (FHWA NHI-16-072): investigation planning, classification, strength, stiffness, hazards",
    },
    "gec6": {
        "adapter": "funhouse_agent.adapters.gec6_adapter",
        "brief": "Shallow foundations reference (FHWA-SA-02-054)",
    },
    "gec7": {
        "adapter": "funhouse_agent.adapters.gec7_adapter",
        "brief": "Soil nail walls reference (FHWA-NHI-14-007)",
    },
    "gec8": {
        "adapter": "funhouse_agent.adapters.gec8_adapter",
        "brief": "CFA piles (FHWA-HIF-07-03): DD pile capacity (NeSmith), ASD, grout volume factor, p-multipliers, group efficiency",
    },
    "gec9": {
        "adapter": "funhouse_agent.adapters.gec9_adapter",
        "brief": "Laterally loaded piles (FHWA-HIF-18-031): resistance factors, p-multipliers, p-y parameters (stiff clay, sand)",
    },
    "gec10": {
        "adapter": "funhouse_agent.adapters.gec10_adapter",
        "brief": "Drilled shafts reference (FHWA-NHI-18-024)",
    },
    "gec11": {
        "adapter": "funhouse_agent.adapters.gec11_adapter",
        "brief": "MSE walls & reinforced soil slopes (FHWA-NHI-10-024)",
    },
    "gec12": {
        "adapter": "funhouse_agent.adapters.gec12_adapter",
        "brief": "Driven piles reference (FHWA-NHI-16-009)",
    },
    "gec13": {
        "adapter": "funhouse_agent.adapters.gec13_adapter",
        "brief": "Ground modification reference (FHWA-NHI-16-027)",
    },
    "gec14": {
        "adapter": "funhouse_agent.adapters.gec14_adapter",
        "brief": "QA for geotech reporting documents (FHWA-HIF-17-016): GRD types, ACMs/ADMs, QA process, checklists",
    },
    "micropile": {
        "adapter": "funhouse_agent.adapters.micropile_adapter",
        "brief": "Design reference (FHWA-NHI-05-039): bond stresses",
    },
    "ufc_backfill": {
        "adapter": "funhouse_agent.adapters.ufc_backfill_adapter",
        "brief": "UFC 3-220-04N (compaction, filter criteria, drainage)",
    },
    "ufc_expansive": {
        "adapter": "funhouse_agent.adapters.ufc_expansive_adapter",
        "brief": "UFC 3-220-07 (swell potential, heave, pier design)",
    },
    "ufc_pavement": {
        "adapter": "funhouse_agent.adapters.ufc_pavement_adapter",
        "brief": "UFC 3-250-01 roads/parking: CBR flexible, rigid, overlays, frost, drainage, joints, mixed traffic (US)",
    },
    "ufc_stabilization": {
        "adapter": "funhouse_agent.adapters.ufc_stabilization_adapter",
        "brief": "UFC 3-250-11: additive selection, strength/durability criteria, equivalency factors",
    },
    "ufc_flexible_practice": {
        "adapter": "funhouse_agent.adapters.ufc_flexible_practice_adapter",
        "brief": "UFC 3-250-03 asphalt: HMA gradations, Marshall/Superpave, spray rates, seal coats",
    },
    "ufc_concrete_practice": {
        "adapter": "funhouse_agent.adapters.ufc_concrete_practice_adapter",
        "brief": "UFC 3-250-04: materials tables, dowel tolerances, joint spacing, RCC",
    },
    "ufc_structural": {
        "adapter": "funhouse_agent.adapters.ufc_structural_adapter",
        "brief": "UFC 3-301-01 DoD (IBC/ASCE 7-22 mods): risk categories, live loads, seismic systems",
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
        "brief": "USDA Wood Handbook: clear-wood properties, EMC/shrinkage, stress/deformation/stability, fastener yield model",
    },
    "fema_p2082": {
        "adapter": "funhouse_agent.adapters.fema_p2082_adapter",
        "brief": "2020 NEHRP: Vs30 site classes A-F incl. BC/CD/DE, SDS/SD1, two-period spectrum, SDC, Risk Category",
    },
    "california_trenching": {
        "adapter": "funhouse_agent.adapters.california_trenching_adapter",
        "brief": "Caltrans Trenching and Shoring Manual: Cal/OSHA soil types/slopes, log-spiral Kp, apparent pressures, bottom heave (US)",
    },
    "fhwa_pavements": {
        "adapter": "funhouse_agent.adapters.fhwa_pavements_adapter",
        "brief": "FHWA-NHI-05-037 Geotechnical Aspects of Pavements: Mr from CBR/R/DCP, drainage, frost, swell, stabilization (US)",
    },
    "fhwa_soils_foundations": {
        "adapter": "funhouse_agent.adapters.fhwa_soils_foundations_adapter",
        "brief": "FHWA NHI-06-088/089 Soils and Foundations manual text: soils, slopes (Taylor/Janbu charts), settlement, footings, piles, shafts, walls",
    },
    "eurocode_7_1": {
        "adapter": "funhouse_agent.adapters.eurocode_7_1_adapter",
        "brief": "EN 1997-1: partial factors + DA1-DA3; earth pressure, bearing, PMT, settlement, rock, movement. SI",
    },
    "eurocode_7_2": {
        "adapter": "funhouse_agent.adapters.eurocode_7_2_adapter",
        "brief": "EN 1997-2: CPT, Menard PMT, SPT, dynamic probing, vane, DMT, plate load correlations; cu/Eoed; lab minima. SI",
    },
    "aashto_1993": {
        "adapter": "funhouse_agent.adapters.aashto_1993_adapter",
        "brief": "AASHTO 1993 Pavement Design Guide: flexible SN, rigid slab D, or W18 for either; layer coeffs, roadbed Mr, ZR/So, mi/Cd, ESALs (US)",
    },
}

#: One rule for a package that can be missing where the app runs (BACKLOG
#: INF-53). A package that is a whole module is listed here: where it is
#: missing, the module is left out of the registry, so no agent is shown a
#: tool that cannot run. A package behind one method of a module is not: the
#: module stays, and that method errors at call time through its has_X().
#: module -> (package to install, its import name). The registry checks
#: that a whole-module package is present without importing it: an import
#: here would cost every process that loads the registry (Pynite's brings
#: matplotlib.pyplot), and an error inside it would take every tool down. A
#: package that is present but cannot import fails only its own calls,
#: which call_agent returns as error results.
#: PyNiteFEA declares numpy >= 2.4, so it does not install where numpy is
#: held at 1.26 (Foundry's transforms repos).
OPTIONAL_MODULES = {
    "pynite": ("PyNiteFEA", "Pynite"),
}

#: The optional modules left out of this registry: module -> missing package.
ABSENT_MODULES = {}
for _name, (_package, _import_name) in OPTIONAL_MODULES.items():
    if importlib.util.find_spec(_import_name) is None:
        del MODULE_REGISTRY[_name]
        ABSENT_MODULES[_name] = _package
