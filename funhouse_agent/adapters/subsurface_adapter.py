"""Subsurface characterization adapter — flat dict -> subsurface_characterization API -> dict.

Supports DIGGS XML ingestion via parse_diggs with a session cache so that
subsequent plot calls can reference the parsed site by key instead of
re-sending the entire data dict.
"""

import re

import numpy as np

from funhouse_agent.adapters import (
    clean_result, figure_output_format, require_params, save_html_output,
)


def _plotly_figure_json(result, params):
    """Compact Plotly figure JSON for the ``.plotly.json`` sidecar, or None.

    Only produced when a save is actually happening (``output_path`` set) — the
    sidecar lets a front-end render the figure natively (st.plotly_chart)
    instead of an HTML iframe. Best-effort: returns None if there is no figure
    or it cannot be serialized.
    """
    if not params.get("output_path"):
        return None
    fig = getattr(result, "figure", None)
    if fig is None:
        return None
    try:
        return fig.to_json()
    except Exception:
        return None


def _require_file_or_content(params: dict, *, method: str):
    """Raise a clear ValueError when neither file_path nor content is given."""
    if params.get("file_path") is None and params.get("content") is None:
        raise ValueError(
            f"{method}: provide 'file_path' (path to the file) or 'content' "
            "(file text), or an 'attachment_key' for an uploaded file."
        )

# Module-level cache: site_key -> SiteModel.  Populated by parse_diggs,
# consumed by any plot method that receives site_key instead of site_data.
_site_cache: dict = {}


def _resolve_site(params: dict):
    """Return a SiteModel from either site_key (cached) or site_data (dict)."""
    from subsurface_characterization import load_site_from_dict

    site_key = params.get("site_key")
    if site_key:
        if site_key not in _site_cache:
            available = sorted(_site_cache.keys()) or ["(none)"]
            raise KeyError(
                f"site_key '{site_key}' not found in cache. "
                f"Available keys: {available}. Call parse_diggs or load_site first."
            )
        return _site_cache[site_key]

    site_data = params.get("site_data")
    if site_data:
        return load_site_from_dict(site_data)

    raise ValueError("Provide either 'site_key' (from parse_diggs/load_site) or 'site_data' dict.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _run_parse_diggs(params: dict) -> dict:
    from subsurface_characterization import parse_diggs

    _require_file_or_content(params, method="parse_diggs")
    filepath = params.get("file_path")
    content = params.get("content")
    result = parse_diggs(filepath=filepath, content=content)

    # Cache the SiteModel for subsequent calls
    site_key = result.site.project_name or "site"
    # Avoid collisions by appending a counter
    base_key = site_key
    counter = 1
    while site_key in _site_cache:
        site_key = f"{base_key}_{counter}"
        counter += 1
    _site_cache[site_key] = result.site

    # Build per-investigation summary including available parameters
    inv_summaries = []
    for inv in result.site.investigations:
        param_names = sorted({m.parameter for m in inv.measurements})
        inv_summaries.append({
            "investigation_id": inv.investigation_id,
            "investigation_type": inv.investigation_type,
            # None where the file gives no position: unknown, never 0.
            "x": inv.x,
            "y": inv.y,
            "elevation_m": inv.elevation_m,
            "total_depth_m": inv.total_depth_m,
            "gwl_depth_m": inv.gwl_depth_m,
            "gwl_not_encountered": inv.gwl_not_encountered,
            "n_measurements": len(inv.measurements),
            "n_lithology": len(inv.lithology),
            "parameters": param_names,
        })

    return clean_result({
        "site_key": site_key,
        "project_name": result.site.project_name,
        "n_investigations": result.n_investigations,
        "n_measurements": result.n_measurements,
        "n_lithology_intervals": result.n_lithology_intervals,
        "warnings": result.warnings,
        "investigations": inv_summaries,
    })


def _run_load_site(params: dict) -> dict:
    from subsurface_characterization import load_site_from_dict
    require_params(params, ["site_data"], method="load_site")
    site = load_site_from_dict(params["site_data"])

    # Cache the SiteModel
    site_key = site.project_name or "site"
    base_key = site_key
    counter = 1
    while site_key in _site_cache:
        site_key = f"{base_key}_{counter}"
        counter += 1
    _site_cache[site_key] = site

    return {
        "site_key": site_key,
        "project_name": site.project_name,
        "n_investigations": len(site.investigations),
        "investigations": [
            {
                "investigation_id": inv.investigation_id,
                "investigation_type": inv.investigation_type,
                "x": inv.x,
                "y": inv.y,
                "elevation_m": inv.elevation_m,
                "total_depth_m": inv.total_depth_m,
                "n_measurements": len(inv.measurements),
                "n_lithology": len(inv.lithology),
            }
            for inv in site.investigations
        ],
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _run_plot_parameter_vs_depth(params: dict) -> dict:
    from subsurface_characterization import plot_parameter_vs_depth
    require_params(params, ["parameter"], method="plot_parameter_vs_depth")
    site = _resolve_site(params)
    result = plot_parameter_vs_depth(
        site=site,
        parameter=params["parameter"],
        color_by=params.get("color_by", "investigation"),
        use_elevation=params.get("use_elevation", False),
        show_trend=params.get("show_trend", False),
        show_bands=params.get("show_bands", False),
        band_sigma=params.get("band_sigma", 1.0),
        group_trends_by=params.get("group_trends_by", ""),
        title=params.get("title", ""),
    )
    output_format = figure_output_format(params)
    return save_html_output(
        clean_result(result.to_dict(output_format=output_format)), params,
        figure_json=_plotly_figure_json(result, params))


def _run_plot_atterberg_limits(params: dict) -> dict:
    from subsurface_characterization import plot_atterberg_limits
    site = _resolve_site(params)
    result = plot_atterberg_limits(
        site=site,
        use_elevation=params.get("use_elevation", False),
        title=params.get("title", ""),
    )
    output_format = figure_output_format(params)
    return save_html_output(
        clean_result(result.to_dict(output_format=output_format)), params,
        figure_json=_plotly_figure_json(result, params))


def _run_plot_multi_parameter(params: dict) -> dict:
    from subsurface_characterization import plot_multi_parameter
    require_params(params, ["parameters"], method="plot_multi_parameter")
    site = _resolve_site(params)
    result = plot_multi_parameter(
        site=site,
        parameters=params["parameters"],
        use_elevation=params.get("use_elevation", False),
        title=params.get("title", ""),
    )
    output_format = figure_output_format(params)
    return save_html_output(
        clean_result(result.to_dict(output_format=output_format)), params,
        figure_json=_plotly_figure_json(result, params))


def _run_plot_plan_view(params: dict) -> dict:
    from subsurface_characterization import plot_plan_view
    site = _resolve_site(params)
    result = plot_plan_view(
        site=site,
        color_by=params.get("color_by", "type"),
        label_field=params.get("label_field", "id"),
        parameter_for_color=params.get("parameter_for_color", ""),
        title=params.get("title", ""),
    )
    output_format = figure_output_format(params)
    return save_html_output(
        clean_result(result.to_dict(output_format=output_format)), params,
        figure_json=_plotly_figure_json(result, params))


def _run_plot_cross_section(params: dict) -> dict:
    from subsurface_characterization import plot_cross_section
    require_params(params, ["investigation_ids"], method="plot_cross_section")
    site = _resolve_site(params)
    result = plot_cross_section(
        site=site,
        investigation_ids=params["investigation_ids"],
        use_elevation=params.get("use_elevation", True),
        annotate_parameter=params.get("annotate_parameter", ""),
        column_width=params.get("column_width", 0.3),
        show_gwl=params.get("show_gwl", True),
        title=params.get("title", ""),
    )
    output_format = figure_output_format(params)
    return save_html_output(
        clean_result(result.to_dict(output_format=output_format)), params,
        figure_json=_plotly_figure_json(result, params))


# ---------------------------------------------------------------------------
# Format adapters — optional, dependency-backed file ingest/validate
# (folded in from the former pygef_agent / ags4_agent / pydiggs_agent modules)
# ---------------------------------------------------------------------------

def _run_parse_cpt(params: dict) -> dict:
    from subsurface_characterization.formats.gef import has_pygef, parse_cpt_file

    if not has_pygef():
        return {"error": "pygef is not installed. Install with: pip install pygef"}

    require_params(params, ["file_path"], method="parse_cpt")
    result = parse_cpt_file(
        file_path=params.get("file_path"),
        engine=params.get("engine", "auto"),
        index=params.get("index", 0),
    )
    return clean_result(result.to_dict())


def _run_parse_borehole(params: dict) -> dict:
    from subsurface_characterization.formats.gef import has_pygef, parse_bore_file

    if not has_pygef():
        return {"error": "pygef is not installed. Install with: pip install pygef"}

    require_params(params, ["file_path"], method="parse_bore")
    result = parse_bore_file(
        file_path=params.get("file_path"),
        engine=params.get("engine", "auto"),
        index=params.get("index", 0),
    )
    return clean_result(result.to_dict())


def _run_read_ags4(params: dict) -> dict:
    from subsurface_characterization.formats.ags4 import has_ags4, read_ags4

    if not has_ags4():
        return {"error": "python-ags4 is not installed. Install with: pip install python-ags4"}

    _require_file_or_content(params, method="read_ags4")
    result = read_ags4(
        filepath=params.get("file_path"),
        content=params.get("content"),
        encoding=params.get("encoding", "utf-8"),
        include_data=params.get("include_data", True),
        convert_numeric=params.get("convert_numeric", True),
    )
    return clean_result(result.to_dict())


def _run_validate_ags4(params: dict) -> dict:
    from subsurface_characterization.formats.ags4 import has_ags4, validate_ags4

    if not has_ags4():
        return {"error": "python-ags4 is not installed. Install with: pip install python-ags4"}

    _require_file_or_content(params, method="validate_ags4")
    result = validate_ags4(
        filepath=params.get("file_path"),
        content=params.get("content"),
        encoding=params.get("encoding", "utf-8"),
    )
    return clean_result(result.to_dict())


def _run_validate_diggs_schema(params: dict) -> dict:
    from subsurface_characterization.formats.diggs_validation import (
        has_schema_check, validate_diggs_schema,
    )

    version = params.get("schema_version", "2.6")
    if not has_schema_check(version):
        # DIGGS 2.6 is bundled and needs nothing extra; 2.5.a needs pydiggs.
        return {"error": f"The DIGGS {version} schema check is not installed "
                         "here (2.5.a needs pydiggs: pip install pydiggs). "
                         "DIGGS 2.6 is always available."}

    _require_file_or_content(params, method="validate_diggs_schema")
    result = validate_diggs_schema(
        filepath=params.get("file_path"),
        content=params.get("content"),
        schema_version=version,
    )
    return clean_result(result.to_dict())


def _run_validate_diggs_dictionary(params: dict) -> dict:
    from subsurface_characterization.formats.diggs_validation import (
        has_pydiggs, validate_diggs_dictionary,
    )

    if not has_pydiggs():
        return {"error": "pydiggs is not installed. Install with: pip install pydiggs"}

    _require_file_or_content(params, method="validate_diggs_dictionary")
    result = validate_diggs_dictionary(
        filepath=params.get("file_path"),
        content=params.get("content"),
    )
    return clean_result(result.to_dict())


# ---------------------------------------------------------------------------
# Writing DIGGS from extracted data
# ---------------------------------------------------------------------------
# Field session 2026-10-06: asked to build DIGGS from a report's boring logs,
# the agent had no way to WRITE DIGGS, so it hand-typed XML into save_file
# under the DIGGS 2.6 namespace and called it "a best-effort partial DIGGS
# XML". It fails the DIGGS schema at its first element. The package has had a
# real DIGGS 2.6 writer, with two gates, since the report-ingest train
# (report_ingest.diggs_writer); this method puts it behind the dispatch layer,
# so data the agent has read off logs and lab sheets becomes a DIGGS file
# that is checked against the schema and read back before anyone calls it
# DIGGS.

def _validation_problems(exc, where: str) -> list:
    """A pydantic ValidationError as short "path: message" lines."""
    out = []
    try:
        for err in exc.errors():
            path = err.get("loc", ())
            loc = "".join(f"[{p}]" if isinstance(p, int) else f".{p}"
                          for p in path)
            msg = err.get("msg")
            given = err.get("input")
            if (path and path[-1] == "n" and "spt" in path
                    and isinstance(given, str)):
                # 2026-10-14 on Funhouse: an agent put a printed ">50" in n,
                # was refused 17 times, and concluded refusal could not be
                # written at all. A printed refusal is now ACCEPTED as given
                # (_normalise_printed_n); what still reaches here is text
                # that is neither a whole number nor a refusal notation.
                msg = (f"{msg} (got {given!r}). n is the printed N as a "
                       "whole number. A printed refusal ('>50', '50+', "
                       "'50/5\"', 'R', 'Refusal') is accepted in n and "
                       "recorded as refusal: true with its text kept; "
                       f"{given!r} is not one -- leave n out and keep what "
                       "was printed in blows")
            out.append(f"{where}{loc}: {msg}")
    except Exception:                                  # noqa: BLE001
        out.append(f"{where}: {exc}")
    return out


#: How a log prints a refusal in its N column. One alternative per
#: notation, the whole text anchored and case-blind, so a word that merely
#: CONTAINS one of these ("WOH", "N/A", "twelve") is not taken for a refusal:
#: ">50", ">= 50", "≥50"; "50+"; "50/5\"", "50/10cm", "100/0.1m", "50/3";
#: "50 for 4in"; "R", "Ref", "Ref.", "Refusal", "Refused", "50R", "50 ref".
_UNIT = r'(?:cm|mm|m|ft|in\.?|inch(?:es)?|"|\'\')'
_REFUSAL_NOTATION = re.compile(
    r"^(?:"
    r"(?:>|>=|≥)\s*\d+"                                 # >50
    r"|\d+\s*\+"                                        # 50+
    rf"|\d+\s*/\s*\d*\.?\d+\s*{_UNIT}?"                 # 50/5"
    rf"|\d+\s*(?:for|in)\s*\d*\.?\d+\s*{_UNIT}"         # 50 for 4in
    r"|(?:\d+\s*)?(?:r|ref\.?|refusal|refused)"         # R, 50 ref
    r")$", re.IGNORECASE)


def _is_refusal_notation(text: str) -> bool:
    """Is this printed N a refusal notation (and not some other word)?"""
    return bool(_REFUSAL_NOTATION.match(str(text or "").strip()))


def _normalise_printed_n(raw, where: str, normalised: list):
    """A printed N that is not a whole number, recorded as the refusal it is.

    2026-10-14 (RI-27): an agent gave a printed ">50" as ``n``, the record's
    ``n`` is an integer, and one such row refused the WHOLE file -- 17 rows,
    a set of 21 borings never written. A refusal notation in ``n`` is now
    taken as given: ``n`` left empty, ``refusal: true``, the printed text in
    ``n_printed``, and each row named in ``normalised`` so nothing changes
    silently. Empty text in ``n`` is "not printed". Anything else (a
    negative N, "WOH", a fraction) is left for validation to refuse by name.
    Returns ``raw`` itself when nothing changed, else a changed copy.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("spt"), list):
        return raw
    name = str(raw.get("investigation_id") or where)
    rows = list(raw["spt"])
    changed = False
    for k, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("n"), str):
            continue
        text = row["n"].strip()
        depth = row.get("depth_top")
        at = (f" at {depth.get('value')} {depth.get('unit', '')}".rstrip()
              if isinstance(depth, dict) else "")
        if not text:
            row = {key: v for key, v in row.items() if key != "n"}
            normalised.append(f"{name} spt[{k}]{at}: n was empty text; "
                              f"read as no printed N")
        elif text.isdigit() or not _is_refusal_notation(text):
            continue            # a whole number, or for validation to refuse
        else:
            row = {key: v for key, v in row.items() if key != "n"}
            was = row.get("refusal")
            row["refusal"] = True
            if not str(row.get("n_printed") or "").strip():
                row["n_printed"] = text
            normalised.append(
                f"{name} spt[{k}]{at}: N printed as {text!r} is not a whole "
                f"number; written as a refusal (refusal: true"
                + ("" if was is True else ", which was not set")
                + ") with the printed text kept and n left empty")
        rows[k] = row
        changed = True
    if not changed:
        return raw
    out = dict(raw)
    out["spt"] = rows
    return out


def _as_list(value, name: str) -> list:
    import json
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"{name}: not valid JSON ({exc})") from None
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of objects")
    return value


# -- Records handed over as FILES (BACKLOG RI-31) ---------------------------
# Field session 2026-10-14, turn 3: helpers typed their boring-log records
# into their replies and the primary retyped 29,000 characters of them into
# write_diggs. A blow count was dropped, a printed 60/15" became 60/15in and
# one exploration got another's coordinates on the way. A helper now saves
# its records as a JSON file (save_file) and write_diggs reads that file, so
# the values reach the writer as the helper wrote them.

#: Each inline argument and the argument that gives the same objects as files.
_RECORD_FILE_ARGS = (("investigations", "investigations_file"),
                     ("lab_tests", "lab_tests_file"))

#: A records file larger than this is refused rather than read.
_RECORDS_FILE_MAX_BYTES = 20 * 1024 * 1024

#: A file saved as one fenced block (```json ... ```) is read inside the fence.
_FENCED = re.compile(r"^\s*```[A-Za-z]*[ \t]*\r?\n(.*?)\r?\n\s*```\s*$",
                     re.DOTALL)


def _file_names(value, arg: str) -> list:
    """The names a ``*_file`` argument gives: one name, a list of names, or
    such a list written as JSON text."""
    import json
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("["):
            return [text]
        try:
            value = json.loads(text)
        except ValueError:
            value = None
    if isinstance(value, (list, tuple)) and all(
            isinstance(v, str) and v.strip() for v in value):
        return [v.strip() for v in value]
    raise ValueError(f"{arg}: give a file name, or a list of file names")


def _inside_folder(path: str, folder: str) -> bool:
    """``path`` is in ``folder``, links and ``..`` resolved."""
    import os
    try:
        p = os.path.normcase(os.path.realpath(path))
        f = os.path.normcase(os.path.realpath(folder))
        return os.path.commonpath([p, f]) == f
    except (ValueError, OSError):             # another drive (Windows)
        return False


def _records_file_path(name: str):
    """``(path, shown)``: the file a records-file argument names, and the
    name a result gives it.

    With a working folder bound (the app), only a file IN that folder is
    read -- a bare name, a sub-path such as ``kept/x.json``, or an absolute
    path inside it -- never one outside it, as the file tools resolve a
    conversation's files. With none (a library caller), the path as given,
    else the name in the default output folder.
    """
    import os
    from funhouse_agent._fileio import find_in_working_folder, host_output_dir
    p = os.path.expanduser(str(name).strip())
    folder = host_output_dir()
    if folder:
        from funhouse_agent.vision_tools import files_here_text
        cand = os.path.abspath(p if os.path.isabs(p)
                               else os.path.join(folder, p))
        if not _inside_folder(cand, folder):
            raise FileNotFoundError(
                f"'{name}' is outside this conversation's files; records "
                f"files are read from its working folder only. Files here: "
                f"{files_here_text()}")
        if not os.path.isfile(cand):
            raise FileNotFoundError(
                f"no file '{name}' in this conversation's files (a file made "
                f"with write_file is in scratch space, which this cannot "
                f"read: save it with save_file). Files here: "
                f"{files_here_text()}")
        rel = os.path.relpath(cand, os.path.abspath(folder))
        return cand, rel.replace(os.sep, "/")
    if os.path.isfile(p):
        return os.path.abspath(p), str(name)
    found = find_in_working_folder(p)
    if found:
        return found, str(name)
    raise FileNotFoundError(f"no file '{name}'")


def _records_in_file(path: str, shown: str, group: str, arg: str,
                     also_other: bool):
    """``(objects, problems)``: what a records file holds for ``group``.

    The file holds a list of the objects ``group`` takes, one such object,
    or ``{group: [...]}``. Such a wrapper may also hold the OTHER group's
    list only when the same file is given for that group too
    (``also_other``), so nothing in a file is left out unread.
    """
    import json
    import os
    other = "lab_tests" if group == "investigations" else "investigations"
    size = os.path.getsize(path)
    if size > _RECORDS_FILE_MAX_BYTES:
        return [], [f"'{shown}': {size / 1e6:.1f} MB, too large to read"]
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8-sig", errors="replace")
    fenced = _FENCED.match(text)
    if fenced:
        text = fenced.group(1)
    try:
        data = json.loads(text)
    except ValueError as exc:
        return [], [f"'{shown}': not valid JSON ({exc}); {arg} takes a JSON "
                    f"file of the objects '{group}' takes"]
    problems: list = []
    if isinstance(data, dict) and (group in data or other in data):
        extra = sorted(str(k) for k in data if k not in (group, other))
        if extra:
            problems.append(
                f"'{shown}': unknown key(s) {extra} at the top of the file; a "
                f"records file holds a list of objects, or "
                f"{{\"{group}\": [...]}}")
        if group not in data:
            if not also_other:
                problems.append(f"'{shown}' holds '{other}', not '{group}': "
                                f"give it as {other}_file")
            return [], problems
        if data.get(other) not in (None, [], {}) and not also_other:
            problems.append(
                f"'{shown}' also holds '{other}', which {arg} does not read: "
                f"give the file as {other}_file too, so nothing in it is "
                f"left out")
        data = data[group]
    if data is None:
        data = []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        problems.append(f"'{shown}': holds {type(data).__name__}, not a list "
                        f"of '{group}' objects")
        return [], problems
    return data, problems


def records_from_files(params: dict) -> dict:
    """``{"investigations": [...], "lab_tests": [...]}``: the raw objects
    the ``*_file`` arguments of a ``write_diggs`` call name, best effort (a
    file that cannot be read is left out). For the coverage record, which
    marks the pages the written data cite."""
    out: dict = {}
    if not isinstance(params, dict):
        return out
    for group, arg in _RECORD_FILE_ARGS:
        try:
            names = _file_names(params.get(arg), arg)
        except ValueError:
            continue
        for name in names:
            try:
                path, shown = _records_file_path(name)
                objs, _ = _records_in_file(path, shown, group, arg, True)
            except Exception:                          # noqa: BLE001
                continue
            out.setdefault(group, []).extend(o for o in objs
                                             if isinstance(o, dict))
    return out


def _diggs_default_name(investigations, project=None) -> str:
    """A DIGGS file name made from what it holds: the investigation ids
    (``B-1.diggs.xml``, ``B-1_B-2.diggs.xml``, ``B-1_to_B-12.diggs.xml``),
    else the project name, else ``site.diggs.xml`` (live smoke wave 1, A15f:
    write_diggs refused a call that gave no output_path)."""
    import re

    def _clean(text) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "")).strip("._-")

    ids = [c for c in (_clean(getattr(i, "investigation_id", ""))
                       for i in investigations or ()) if c]
    if len(ids) == 1:
        stem = ids[0]
    elif 1 < len(ids) <= 3:
        stem = "_".join(ids)
    elif ids:
        stem = f"{ids[0]}_to_{ids[-1]}"
    else:
        stem = _clean(getattr(project, "name", "")) or "site"
    return f"{stem[:60]}.diggs.xml"


def _run_write_diggs(params: dict) -> dict:
    import json
    import os

    from funhouse_agent.adapters import reject_unknown_params
    from pydantic import ValidationError
    from report_ingest.diggs_writer import (
        diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
    )
    from report_ingest.model import Investigation, LabTest, Project

    _valid = ("investigations", "lab_tests", "investigations_file",
              "lab_tests_file", "project", "output_path", "document_id")
    reject_unknown_params(params, _valid, method="write_diggs")
    if all(params.get(k) is None for k in ("investigations",
                                           "investigations_file",
                                           "lab_tests", "lab_tests_file")):
        raise ValueError(
            "write_diggs: missing required parameter(s): give the records "
            "inline as 'investigations' (and 'lab_tests'), or as JSON files "
            "in the working folder with 'investigations_file' (and "
            f"'lab_tests_file'). Valid parameters: {sorted(_valid)}.")

    problems: list = []
    unread: list = []          # files that could not be read or used
    investigations = []
    normalised: list = []
    lab = []
    # Where each record came from: label -> {investigations, lab_tests}.
    sources: dict = {}
    origin: list = []          # (label, where, Investigation)

    def _take(group, raws, label, where_of):
        src = sources.setdefault(label, {"investigations": [],
                                         "lab_tests": 0})
        for i, raw in enumerate(raws):
            where = where_of(i)
            try:
                if group == "investigations":
                    raw = _normalise_printed_n(raw, where, normalised)
                    inv = Investigation.model_validate(raw)
                    investigations.append(inv)
                    origin.append((label, where, inv))
                    src["investigations"].append(inv.investigation_id)
                else:
                    lab.append(LabTest.model_validate(raw))
                    src["lab_tests"] += 1
            except ValidationError as exc:
                problems.extend(_validation_problems(exc, where))

    # The files each *_file argument names, resolved before any is read, so
    # a file given for both groups is read for both.
    files: dict = {}
    for group, arg in _RECORD_FILE_ARGS:
        files[group] = []
        for name in _file_names(params.get(arg), arg):
            try:
                path, shown = _records_file_path(name)
            except OSError as exc:
                unread.append(f"{arg}: {exc}")
                continue
            if any(path == p for p, _ in files[group]):
                unread.append(f"{arg}: '{shown}' is named twice; its records "
                              f"would be written twice")
                continue
            files[group].append((path, shown))
    for group, arg in _RECORD_FILE_ARGS:
        _take(group, _as_list(params.get(group), group), "inline",
              lambda i, g=group: f"{g}[{i}]")
        other = "lab_tests" if group == "investigations" else "investigations"
        for path, shown in files[group]:
            try:
                objs, bad = _records_in_file(
                    path, shown, group, arg,
                    any(path == p for p, _ in files[other]))
            except OSError as exc:
                objs, bad = [], [f"'{shown}' could not be read: {exc}"]
            unread.extend(bad)
            _take(group, objs, shown,
                  lambda i, s=shown, g=group: f"'{s}' {g}[{i}]")
    # One exploration in two inputs is refused, never merged or doubled.
    by_id: dict = {}
    for label, where, inv in origin:
        key = str(inv.investigation_id or "").strip()
        if key:
            by_id.setdefault(key, []).append((label, where, inv))
    for key, rows in by_id.items():
        if len({label for label, _, _ in rows}) < 2:
            continue
        first = rows[0][2].model_dump()
        same = all(r[2].model_dump() == first for r in rows[1:])
        unread.append(
            f"investigation_id {key!r} is given in more than one input: "
            + ", ".join(where for _, where, _ in rows)
            + (" (the same record each time)" if same
               else " (the records differ)"))
    project = None
    if params.get("project"):
        raw = params["project"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        try:
            project = Project.model_validate(raw)
        except ValidationError as exc:
            problems += _validation_problems(exc, "project")
    given_files = any(files.values())
    if problems or unread:
        problems = unread + problems
        refused = {"error": ("the data does not fit the DIGGS record format; "
                             if not unread else
                             "the records given could not all be used; ")
                            + "NO file was written",
                   "problems": problems[:25],
                   "n_problems": len(problems),
                   "hint": "fix the named fields (describe_method shows the "
                           "format) and call write_diggs again with ALL the "
                           "data -- nothing was kept from this call"}
        if given_files or unread:
            refused["hint"] = (
                "fix what is named -- in the named file (save it again with "
                "save_file) or inline -- and call write_diggs again with ALL "
                "the data, the same files included; nothing was kept from "
                "this call")
        if any("in more than one input" in u for u in unread):
            refused["hint"] += (
                ". An exploration in two inputs: give it once (two parts of "
                "one log: merge them into one record first); two "
                "explorations that print the same name: give both in one "
                "list or file, where each is written apart")
        if normalised:
            refused["normalised"] = normalised[:10]
        return refused
    if not investigations and not lab:
        return {"error": "nothing to write: no investigations and no lab "
                         "tests were given"}

    # Join the data up the way report ingest does before writing it: lab tests
    # linked to their boring and sample, a summary table compared with the
    # individual sheets, unit problems named (report_ingest.reconciler). It
    # never changes a value; every disagreement is reported with both values
    # and both pages. Owner, 2026-10-08: "tie the reconciler's cross checks to
    # write_diggs" -- in the 2026-10-06 field session a report's own summary
    # table disagreed with its lab sheet and nothing said so.
    from report_ingest.model import ReportRecord
    from report_ingest.reconciler import reconcile
    record = ReportRecord(
        project=project or Project(),
        investigations=[i.model_copy(deep=True) for i in investigations],
        lab_tests=[t.model_copy(deep=True) for t in lab])
    try:
        reconcile(record)
        cross = [q.model_dump(exclude_defaults=True) for q in record.qa]
        cross_error = None
    except Exception as exc:                           # noqa: BLE001
        cross, cross_error = [], f"{type(exc).__name__}: {exc}"
    # Every page a record cites, against the pages this conversation read or
    # looked at (BACKLOG EV-18: field session 2026-10-14, four page lists
    # were even splits of a section, not pages anything had read). One entry
    # per record citing a page nothing opened; never a refusal.
    try:
        from funhouse_agent.cited_pages import record_entries
        page_entries, pages_check = record_entries(
            {"investigations": investigations, "lab_tests": lab},
            document=params.get("document_id"))
    except Exception as exc:                           # noqa: BLE001
        page_entries, pages_check = [], {
            "checked": False,
            "why": f"the check could not run ({type(exc).__name__})"}
    cross = page_entries + cross

    notes: list = []
    xml = write_diggs(record, project,
                      document_id=str(params.get("document_id") or ""),
                      notes=notes)
    if params.get("output_path"):
        out = str(params["output_path"])
    else:
        from funhouse_agent._fileio import resolve_output_path
        out = resolve_output_path(_diggs_default_name(investigations, project))
    folder = os.path.dirname(os.path.abspath(out))
    os.makedirs(folder, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(xml)
    note = notes[0] if notes else None
    schema_ok, schema_errors = diggs_schema_gate(xml)
    checked = not any("not installed" in e for e in schema_errors)
    back_ok, diffs = diggs_roundtrip_gate(xml, investigations,
                                          project=project, lab_tests=lab,
                                          notes=note)
    if schema_ok and back_ok:
        verdict = ("valid DIGGS 2.6: it passed the schema and reads back "
                   "equal to the data given")
    elif not checked:
        verdict = ("written, but the schema could NOT be checked here "
                   "(no schema check is installed); "
                   + ("it reads back equal to the data given" if back_ok
                      else "and it does NOT read back equal to the data "
                           "given -- see read_back.diffs"))
    else:
        verdict = ("written, but it is NOT clean DIGGS: "
                   + ("" if schema_ok else "it fails the DIGGS 2.6 schema"
                      + ("" if back_ok else "; "))
                   + ("" if back_ok else "reading it back does not give the "
                                         "data given")
                   + " -- say so; do not call it valid")
    by_kind: dict = {}
    for q in cross:
        by_kind[q.get("kind", "note")] = by_kind.get(q.get("kind", "note"), 0) + 1
    if cross:
        verdict += (f". The cross-checks raised {len(cross)} point(s) "
                    f"({', '.join(f'{n} {k}' for k, n in sorted(by_kind.items()))})"
                    " -- list each for the user with its values and pages; "
                    "nothing was changed")
    written: dict = {}
    if note is not None:
        # What this layer normalised before the writer saw it goes first,
        # beside what the writer itself wrote differently.
        note.normalised[:0] = normalised
        if note.position_unknown:
            names = note.position_unknown
            verdict += (
                f". {len(names)} exploration(s) had no x, y and/or "
                f"elevation given ({', '.join(names[:8])}"
                + (", ..." if len(names) > 8 else "")
                + "): DIGGS requires a position, so each unknown coordinate "
                "is written as NaN and reads back as unknown, never 0 -- tell "
                "the user which positions are missing (written.assumed)")
        if note.normalised:
            verdict += (
                f". {len(note.normalised)} value(s) were recorded "
                f"differently from how they were given (e.g. a printed '>50' "
                f"as a refusal, 'not encountered' never as a water level) -- "
                f"tell the user (written.normalised)")
        for key, value in note.to_dict().items():
            if isinstance(value, list):
                written[key] = value[:10]
                if len(value) > 10:
                    written[f"n_{key}"] = len(value)
            else:
                written[key] = value
    extra: dict = {}
    if given_files:
        # Which input each investigation came from (RI-31).
        extra["sources"] = [
            {"from": label,
             "investigations": src["investigations"][:60],
             **({"n_investigations": len(src["investigations"])}
                if len(src["investigations"]) > 60 else {}),
             "lab_tests": src["lab_tests"]}
            for label, src in sources.items()
            if src["investigations"] or src["lab_tests"]
            or label != "inline"]
    return clean_result({
        "output_path": out,
        "file_exists": os.path.isfile(out),
        "file_size_bytes": os.path.getsize(out) if os.path.isfile(out) else 0,
        "verdict": verdict,
        **extra,
        "cross_checks": ({"n": len(cross), "by_kind": by_kind,
                          "entries": cross[:20]}
                         if cross_error is None
                         else {"error": f"the cross-checks could not run "
                                        f"({cross_error}); the file is "
                                        f"unaffected",
                               **({"n": len(cross), "by_kind": by_kind,
                                   "entries": cross[:20]} if cross else {})}),
        **({"cited_pages": pages_check}
           if pages_check.get("why") != "no record cites a page" else {}),
        "schema_check": {"checked": checked, "valid": bool(schema_ok),
                         "errors": list(schema_errors)[:10]},
        "read_back": {"equal": bool(back_ok), "diffs": list(diffs)[:15],
                      "n_diffs": len(diffs)},
        "written": written,
    })


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _run_compute_trend(params: dict) -> dict:
    from subsurface_characterization import compute_trend
    require_params(params, ["depths", "values"], method="compute_trend")
    depths = np.asarray(params["depths"], dtype=float)
    values = np.asarray(params["values"], dtype=float)
    result = compute_trend(
        depths=depths,
        values=values,
        trend_type=params.get("trend_type", "linear"),
        parameter=params.get("parameter", ""),
        group_label=params.get("group_label", ""),
    )
    return clean_result(result.to_dict())


def _run_spt_correction(params: dict) -> dict:
    from geotech_common.soil_properties import spt_n1_60
    _valid = ("N_field", "sigma_vo_eff", "energy_ratio_pct", "rod_length_m",
              "borehole_diameter_mm", "sampler_correction", "overburden_method")
    from funhouse_agent.adapters import reject_unknown_params
    reject_unknown_params(params, _valid, method="spt_correction")
    require_params(params, ["N_field", "sigma_vo_eff"],
                   method="spt_correction", valid=_valid)
    r = spt_n1_60(
        params["N_field"], params["sigma_vo_eff"],
        energy_ratio_pct=params.get("energy_ratio_pct", 60.0),
        rod_length_m=params.get("rod_length_m", 10.0),
        borehole_diameter_mm=params.get("borehole_diameter_mm", 100.0),
        sampler_correction=params.get("sampler_correction", 1.0),
        overburden_method=params.get("overburden_method", "liao_whitman"),
    )
    return {k: round(v, 4) for k, v in r.items()}


def _run_stress_dilatancy(params: dict) -> dict:
    from geotech_common.soil_properties import stress_dilatancy_bolton
    _valid = ("relative_density", "p_eff", "Q", "R", "stress_condition")
    from funhouse_agent.adapters import reject_unknown_params
    reject_unknown_params(params, _valid, method="stress_dilatancy")
    require_params(params, ["relative_density", "p_eff"],
                   method="stress_dilatancy", valid=_valid)
    r = stress_dilatancy_bolton(
        params["relative_density"], params["p_eff"],
        Q=params.get("Q", 10.0), R=params.get("R", 1.0),
        stress_condition=params.get("stress_condition", "triaxial"),
    )
    return {k: round(v, 4) for k, v in r.items()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "spt_correction": _run_spt_correction,
    "stress_dilatancy": _run_stress_dilatancy,
    "parse_diggs": _run_parse_diggs,
    "load_site": _run_load_site,
    "plot_parameter_vs_depth": _run_plot_parameter_vs_depth,
    "plot_atterberg_limits": _run_plot_atterberg_limits,
    "plot_multi_parameter": _run_plot_multi_parameter,
    "plot_plan_view": _run_plot_plan_view,
    "plot_cross_section": _run_plot_cross_section,
    "compute_trend": _run_compute_trend,
    # Format adapters (folded-in pygef / ags4 / pydiggs)
    "parse_cpt": _run_parse_cpt,
    "parse_bore": _run_parse_borehole,
    "read_ags4": _run_read_ags4,
    "validate_ags4": _run_validate_ags4,
    "validate_diggs_schema": _run_validate_diggs_schema,
    "validate_diggs_dictionary": _run_validate_diggs_dictionary,
    "write_diggs": _run_write_diggs,
}

_SITE_KEY_DOC = "Key returned by parse_diggs or load_site (preferred). Alternative to site_data."
_SITE_DATA_DOC = "Nested dict with project_name, investigations. Alternative to site_key."

METHOD_INFO = {
    "spt_correction": {
        "category": "Field Test Corrections",
        "brief": "Raw field SPT N -> N60 -> (N1)60 (energy + overburden corrections, Youd et al. 2001 / Liao-Whitman).",
        "parameters": {
            "N_field": {"type": "float", "required": True, "description": "Raw field SPT blow count."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Vertical effective stress at test depth (kPa)."},
            "energy_ratio_pct": {"type": "float", "required": False, "default": 60.0, "description": "Hammer energy ratio ER (%). Donut 45-60, safety 55-72, automatic 80-100."},
            "rod_length_m": {"type": "float", "required": False, "default": 10.0, "description": "Rod length (m); sets CR band (0.75 below 3 m up to 1.0 beyond 10 m)."},
            "borehole_diameter_mm": {"type": "float", "required": False, "default": 100.0, "description": "Borehole diameter (mm); CB = 1.0/1.05/1.15."},
            "sampler_correction": {"type": "float", "required": False, "default": 1.0, "description": "CS: 1.0 standard sampler, 1.1-1.3 without liners."},
            "overburden_method": {"type": "str", "required": False, "default": "liao_whitman", "allowed_values": ["liao_whitman", "iso"], "description": "C_N form: Liao & Whitman (1986, cap 1.7) or EN ISO 22476-3 (cap 2.0)."},
        },
        "returns": {"N60": "Energy-corrected blow count.", "N1_60": "Fully corrected (N1)60.", "CN": "Overburden factor."},
    },
    "stress_dilatancy": {
        "category": "Strength Correlations",
        "brief": "Bolton (1986) dilatancy: peak-friction excess and dilation angle from Dr and p'.",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "description": "Relative density Dr as a FRACTION (0-1)."},
            "p_eff": {"type": "float", "required": True, "description": "Mean effective stress p' (kPa)."},
            "Q": {"type": "float", "required": False, "default": 10.0, "description": "Bolton mineralogy constant (10 for silica sand)."},
            "R": {"type": "float", "required": False, "default": 1.0, "description": "Bolton fitting constant."},
            "stress_condition": {"type": "str", "required": False, "default": "triaxial", "allowed_values": ["triaxial", "plane_strain"], "description": "Loading mode."},
        },
        "returns": {"IR": "Relative dilatancy index.", "phi_excess_deg": "phi_peak - phi_crit (deg).", "dilation_angle_deg": "Dilation angle psi (deg)."},
    },
    "parse_diggs": {
        "category": "Subsurface Characterization",
        "brief": "Parse DIGGS 2.6/2.5.a XML into site model. Returns site_key for use in subsequent plot calls.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file (DBFS, workspace, or local). Provide file_path, content, or attachment_key."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
            "attachment_key": {"type": "str", "required": False, "description": "Key of a DIGGS file uploaded via the chat widget. The file bytes are decoded to XML automatically."},
        },
        "returns": {
            "site_key": "Key to reference this site in subsequent calls (plot_*, compute_trend).",
            "project_name": "Project name from DIGGS file.",
            "n_investigations": "Number of borings/CPTs/etc extracted.",
            "n_measurements": "Total number of test measurements (SPT, Atterberg, etc).",
            "n_lithology_intervals": "Total lithology intervals.",
            "warnings": "Any parsing warnings.",
            "investigations": "Per-investigation summary with available parameters. x, y or elevation_m is null when "
                              "the file gives none (unknown, not 0); gwl_not_encountered is true when the file says "
                              "groundwater was not encountered (gwl_depth_m is then null).",
        },
    },
    "load_site": {
        "category": "Subsurface Characterization",
        "brief": "Load site model from nested dict structure. Returns site_key for subsequent plot calls.",
        "parameters": {
            "site_data": {"type": "dict", "required": True, "description": "Nested dict with project_name, investigations [{investigation_id, x, y, elevation_m, measurements, lithology}]."},
        },
        "returns": {
            "site_key": "Key to reference this site in subsequent calls.",
            "project_name": "Site project name.",
            "n_investigations": "Number of investigation locations.",
            "investigations": "Summary of each investigation.",
        },
    },
    "plot_parameter_vs_depth": {
        "category": "Subsurface Characterization",
        "brief": "Scatter plot of a soil parameter vs depth (Plotly interactive).",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "parameter": {"type": "str", "required": True, "description": "Parameter name to plot (e.g., 'N_spt', 'cu_kPa', 'wn_pct')."},
            "color_by": {"type": "str", "required": False, "default": "investigation", "allowed_values": ["investigation", "uscs", "none"], "description": "Color coding."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation instead of depth."},
            "show_trend": {"type": "bool", "required": False, "default": False, "description": "Overlay linear trendline."},
            "show_bands": {"type": "bool", "required": False, "default": False, "description": "Overlay prediction bands."},
            "band_sigma": {"type": "float", "required": False, "default": 1.0, "description": "Number of sigma for prediction bands."},
            "group_trends_by": {"type": "str", "required": False, "default": "", "description": "Group trends by 'uscs' for separate trends per soil class."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
            "output_path": {"type": "str", "required": False, "description": "If given, SAVE the self-contained figure HTML under this file name (e.g. 'site_plot.html'; in the app it is written into this conversation's folder) and return a short {output_path, file_exists, size} confirmation instead of the large HTML blob. Forces html rendering; the write is verified. Omit to receive the HTML inline."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Number of data points plotted.",
            "n_investigations": "Number of investigations shown.",
        },
    },
    "plot_atterberg_limits": {
        "category": "Subsurface Characterization",
        "brief": "Plot Atterberg limits (LL-PL bracket) with natural moisture content overlay.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
            "output_path": {"type": "str", "required": False, "description": "If given, SAVE the self-contained figure HTML under this file name (e.g. 'site_plot.html'; in the app it is written into this conversation's folder) and return a short {output_path, file_exists, size} confirmation instead of the large HTML blob. Forces html rendering; the write is verified. Omit to receive the HTML inline."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Number of data points plotted.",
        },
    },
    "plot_multi_parameter": {
        "category": "Subsurface Characterization",
        "brief": "Side-by-side subplots of multiple parameters with shared depth axis.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "parameters": {"type": "array", "required": True, "description": "Parameter names for each panel (e.g., ['N_spt','cu_kPa','wn_pct'])."},
            "use_elevation": {"type": "bool", "required": False, "default": False, "description": "If True, Y-axis is elevation."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
            "output_path": {"type": "str", "required": False, "description": "If given, SAVE the self-contained figure HTML under this file name (e.g. 'site_plot.html'; in the app it is written into this conversation's folder) and return a short {output_path, file_exists, size} confirmation instead of the large HTML blob. Forces html rendering; the write is verified. Omit to receive the HTML inline."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_data_points": "Total data points across all panels.",
            "parameters": "Parameters plotted.",
        },
    },
    "plot_plan_view": {
        "category": "Subsurface Characterization",
        "brief": "Plan view map of investigation locations with optional parameter-based coloring.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "color_by": {"type": "str", "required": False, "default": "type", "allowed_values": ["type", "parameter"], "description": "Color mode: 'type' (investigation type) or 'parameter' (avg value)."},
            "label_field": {"type": "str", "required": False, "default": "id", "description": "Label: 'id', 'depth_to_rock', 'gwl', 'fill_thickness', or parameter name."},
            "parameter_for_color": {"type": "str", "required": False, "default": "", "description": "Parameter name when color_by='parameter'."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
            "output_path": {"type": "str", "required": False, "description": "If given, SAVE the self-contained figure HTML under this file name (e.g. 'site_plot.html'; in the app it is written into this conversation's folder) and return a short {output_path, file_exists, size} confirmation instead of the large HTML blob. Forces html rendering; the write is verified. Omit to receive the HTML inline."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_investigations": "Number of investigation locations shown.",
        },
    },
    "plot_cross_section": {
        "category": "Subsurface Characterization",
        "brief": "Cross-section profile view with USCS-colored lithology columns along a transect.",
        "parameters": {
            "site_key": {"type": "str", "required": False, "description": _SITE_KEY_DOC},
            "site_data": {"type": "dict", "required": False, "description": _SITE_DATA_DOC},
            "investigation_ids": {"type": "array", "required": True, "description": "Ordered list of investigation IDs defining the transect (e.g., ['B-1','B-2','B-3'])."},
            "use_elevation": {"type": "bool", "required": False, "default": True, "description": "If True, Y-axis is elevation (default). False for depth."},
            "annotate_parameter": {"type": "str", "required": False, "default": "", "description": "Parameter to annotate next to columns (e.g., 'N_spt')."},
            "column_width": {"type": "float", "required": False, "default": 0.3, "description": "Width of lithology columns (fraction of spacing)."},
            "show_gwl": {"type": "bool", "required": False, "default": True, "description": "Show groundwater level dashed line."},
            "title": {"type": "str", "required": False, "default": "", "description": "Custom plot title."},
            "output_format": {"type": "str", "required": False, "default": "metadata", "allowed_values": ["metadata", "html", "json"], "description": "Output format."},
            "output_path": {"type": "str", "required": False, "description": "If given, SAVE the self-contained figure HTML under this file name (e.g. 'site_plot.html'; in the app it is written into this conversation's folder) and return a short {output_path, file_exists, size} confirmation instead of the large HTML blob. Forces html rendering; the write is verified. Omit to receive the HTML inline."},
        },
        "returns": {
            "plot_type": "Type of plot.",
            "n_investigations": "Number of investigations in the cross-section.",
        },
    },
    "compute_trend": {
        "category": "Subsurface Characterization",
        "brief": "Fit linear or log-linear depth-value trend with statistics (R-squared, COV).",
        "parameters": {
            "depths": {"type": "array", "required": True, "description": "Depth values (m)."},
            "values": {"type": "array", "required": True, "description": "Measurement values."},
            "trend_type": {"type": "str", "required": False, "default": "linear", "allowed_values": ["linear", "log_linear"], "description": "Trend type."},
            "parameter": {"type": "str", "required": False, "default": "", "description": "Parameter name for labeling."},
            "group_label": {"type": "str", "required": False, "default": "", "description": "Group label (e.g., USCS class)."},
        },
        "returns": {
            "slope": "Regression slope.",
            "intercept": "Regression intercept.",
            "r_squared": "Coefficient of determination.",
            "std_residual": "Standard deviation of residuals.",
            "cov": "Coefficient of variation.",
        },
    },
    # -----------------------------------------------------------------------
    # Format adapters — file ingest / validation (optional, dependency-backed)
    # -----------------------------------------------------------------------
    "parse_cpt": {
        "category": "File Import",
        "brief": "Parse a CPT file (GEF or BRO-XML, via pygef) into depth/qc/fs/u2 arrays (kPa).",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to CPT file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "gef", "xml"], "description": "Parser engine."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_points": "Number of data points.",
            "alias": "Test ID or filename.",
            "final_depth_m": "Final penetration depth (m).",
            "gwl_m": "Groundwater level (m below surface) or null.",
            "depth_m": "Depth array (m).",
            "q_c_kPa": "Cone tip resistance array (kPa).",
            "f_s_kPa": "Sleeve friction array (kPa).",
            "u_2_kPa": "Pore pressure u2 array (kPa).",
            "Rf_pct": "Friction ratio array (%).",
        },
    },
    "parse_bore": {
        "category": "File Import",
        "brief": "Parse a borehole file (GEF or BRO-XML, via pygef) into layer descriptions.",
        "parameters": {
            "file_path": {"type": "str", "required": True, "description": "Path to borehole file (.gef or .xml)."},
            "engine": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "gef", "xml"], "description": "Parser engine."},
            "index": {"type": "int", "required": False, "default": 0, "description": "Record index for multi-record XML files."},
        },
        "returns": {
            "n_layers": "Number of soil layers.",
            "alias": "Borehole ID or filename.",
            "final_depth_m": "Total bore depth (m).",
            "gwl_m": "Groundwater level (m below surface) or null.",
            "layers": "List of layer dicts with top_m, bottom_m, soil_name, soil_code.",
        },
    },
    "read_ags4": {
        "category": "File Import",
        "brief": "Read and parse an AGS4 geotechnical data file (via python-ags4) into structured tables.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to AGS4 file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "Raw AGS4 content as string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8", "description": "File encoding."},
            "include_data": {"type": "bool", "required": False, "default": True, "description": "Include all table data in result."},
            "convert_numeric": {"type": "bool", "required": False, "default": True, "description": "Convert numeric columns from text."},
        },
        "returns": {
            "filepath": "Source file path or '<string>'.",
            "n_groups": "Number of AGS4 groups (tables) found.",
            "group_names": "Names of all groups.",
            "group_row_counts": "Row counts per group.",
            "tables": "Dict of group_name to list of row dicts (if include_data=True).",
        },
    },
    "validate_ags4": {
        "category": "File Validation",
        "brief": "Validate an AGS4 file against AGS4 rules (via python-ags4).",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to AGS4 file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "Raw AGS4 content as string."},
            "encoding": {"type": "str", "required": False, "default": "utf-8", "description": "File encoding."},
        },
        "returns": {
            "filepath": "Source file path.",
            "n_errors": "Number of errors found.",
            "n_warnings": "Number of warnings found.",
            "n_fyi": "Number of FYI messages.",
            "is_valid": "True if no errors (warnings/FYI acceptable).",
            "errors": "Error details grouped by rule number.",
        },
    },
    "validate_diggs_schema": {
        "category": "File Validation",
        "brief": "Validate DIGGS XML against the XSD schema (v2.6 bundled, always available; v2.5.a needs pydiggs). For DIGGS data EXTRACTION use parse_diggs instead.",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
            "schema_version": {"type": "str", "required": False, "default": "2.6", "allowed_values": ["2.6", "2.5.a"], "description": "DIGGS schema version."},
        },
        "returns": {
            "source": "Filename or 'content'.",
            "check_type": "Always 'schema'.",
            "schema_version": "Schema version validated against.",
            "is_valid": "Whether validation passed.",
            "n_errors": "Number of validation errors.",
            "errors": "List of error messages.",
        },
    },
    "write_diggs": {
        "category": "File Export",
        "brief": "WRITE a DIGGS 2.6 file from boring/pit/sounding logs and lab results you have read (e.g. off a report's logs), given inline or as JSON files a helper saved. Checks the file against the DIGGS schema and reads it back; never hand-write DIGGS XML.",
        "parameters": {
            "investigations": {"type": "array", "required": False, "description": (
                "One object per boring, pit or sounding, with the values read off its log. A number printed on the log "
                "goes in as printed. A depth read off the log's own drawn depth scale (log_grid's samples, layers and rows, "
                "or measure) is read from the log just as much as a printed one: write it, and note the reading in a prov "
                "{page, method: 'grid', note: e.g. 'sample depths off the depth scale +/-0.01 m'} on the value or once in "
                "the investigation's prov list. Never infer a value the log does not give, and "
                "never hold back a record whose values you have: leave out only what could not be read, and name each thing "
                "left out, and why, in your answer. "
                "Keys: investigation_id; kind (boring|test_pit|cpt|dcp|hand_auger|well|other); depth_unit ('m'|'ft'); "
                "x, y, coordinate_system; elevation {value, unit} (leave out any the source does not give: it is written as "
                "unknown, never 0); total_depth {value, unit}; date_started, date_finished; "
                "drilling {method, equipment, hammer_type, hammer_energy_ratio, sampler, contractor, logged_by}; "
                "layers [{top {value, unit}, bottom {value, unit}, description, uscs}]; "
                "samples [{sample_id (may be empty), top, bottom, kind (spt|ring|shelby|bulk|grab|core|cuttings|other), "
                "recovery {value, unit} (a printed length), recovery_percent, rqd_percent, "
                "water_content, fines_percent, liquid_limit, plastic_limit, uscs}]; "
                "spt [{depth_top, depth_bottom, blows [per increment; a string for refusal, e.g. '50/10cm'], "
                "increment {value, unit} (the length each drive is counted over, as the form prints it, e.g. 15 cm; "
                "6 in when left out), n (ONLY if printed; "
                "a printed refusal such as '>50', 'R' or '50/5\"' may be given as n and is recorded as refusal: true with its text kept), "
                "refusal, sample_id}]; "
                "water [{depth {value, unit}, when (while_drilling|at_completion|after_hours|not_encountered|unknown), hours}] "
                "(not_encountered: leave depth out -- it is never written as a water level); "
                "remarks; pages (0-based PDF pages the values were read from; none for a record no page gave, such as a "
                "placeholder). Every depth or elevation is {value, unit}. Unknown keys are refused by name.")},
            "lab_tests": {"type": "array", "required": False, "description": (
                "Lab results, one object per test on one specimen: kind (atterberg|gradation|compaction|cbr|moisture_content|density|"
                "chemical|specific_gravity|unconfined|unconfined_rock|direct_shear|triaxial|swell_consolidation|permeability|"
                "organic_content|summary_table|other), "
                "investigation_id, sample_id, depth_top {value, unit}, depth_bottom, standard, pages, and result {kind: the same kind, then: "
                "atterberg: ll, pl, pi, non_plastic; gradation: gravel_percent, sand_percent, silt_percent, clay_percent, fines_percent, "
                "percent_passing [{percent_passing, size {value, unit}, sieve}]; compaction: max_dry_density {value, unit}, optimum_wc; "
                "cbr: cbr_percent, swell_percent, soaked; moisture_content: wc; chemical: pH, sulfate, chloride, resistivity (numbers, "
                "{value, unit}, or the printed text e.g. '<10')}. A report's lab SUMMARY TABLE goes in too, as one object of kind "
                "summary_table with result {kind: summary_table, rows [{investigation_id, sample_id, depth_top, wc, ll, pl, pi, "
                "gravel_percent, sand_percent, fines_percent, max_dry_density, optimum_wc, pH, sulfate, chloride}]}: it is "
                "cross-checked against the individual sheets.")},
            "investigations_file": {"type": "str|array", "required": False, "description": (
                "The same objects as investigations, from a JSON file in this conversation's folder (a list of them, "
                "or {investigations: [...]}), or a list of such files read in order. Records a helper extracts: brief it "
                "to save them with save_file as such a file and to name the file in its reply, then pass the name here "
                "-- never retype them. Checked as inline records are, each problem naming its file. Combines with "
                "inline investigations; one investigation_id in two inputs is refused, naming both.")},
            "lab_tests_file": {"type": "str|array", "required": False, "description": (
                "The same for lab_tests: a JSON file (or files) of lab_tests objects, or {lab_tests: [...]}.")},
            "project": {"type": "dict", "required": False, "description": "{name, number, client, location, coordinate_system, elevation_datum} as printed."},
            "output_path": {"type": "str", "required": False, "description": "A file name for the .xml, e.g. 'site.diggs.xml'; omitted, it is named after the investigations (e.g. 'B-1_to_B-12.diggs.xml'). In the app it is written into this conversation's folder, whatever directory is given, and attached to the conversation."},
            "document_id": {"type": "str", "required": False, "description": "An identifier for the source document."},
        },
        "returns": {
            "output_path": "The file written (nothing is written when the data does not fit: 'problems' names each field).",
            "verdict": "Plain words: valid DIGGS, or what is wrong with it. Report THIS, not your own judgement.",
            "schema_check": "{checked, valid, errors} against the DIGGS 2.6 XSD (bundled; runs on every host).",
            "read_back": "{equal, diffs}: the file parsed back and compared value by value with the data given.",
            "cross_checks": "{n, by_kind, entries}: the data joined up as report ingest does -- each lab test linked to its "
                            "boring and sample, a summary table compared with the sheets, units without a conversion, drives "
                            "that do not add up to their sample's interval, a drive count unlike the log's other records, "
                            "any layer, sample, SPT or water depth below the total depth, and (page_not_read) any page a record "
                            "cites, in pages or a prov, that no tool or helper read or looked at in this conversation. Each entry "
                            "gives what disagrees, both values and the pages; nothing is changed or resolved. Tell the user "
                            "about every entry.",
            "cited_pages": "{checked, pages_cited, not_read}: the records' pages against the pages read in this "
                           "conversation, or why they could not be checked.",
            "written": "Counts written, and anything the writer had to leave out, assume (assumed; position_unknown) or record "
                       "differently from how it was given (normalised). Tell the user about each.",
            "sources": "When a file was given: each input (inline or the file) with the investigation_ids it gave.",
        },
    },
    "validate_diggs_dictionary": {
        "category": "File Validation",
        "brief": "Validate DIGGS propertyClass values against DIGGS dictionary (via pydiggs).",
        "parameters": {
            "file_path": {"type": "str", "required": False, "description": "Path to DIGGS XML file. Provide file_path or content, not both."},
            "content": {"type": "str", "required": False, "description": "DIGGS XML as string."},
        },
        "returns": {
            "source": "Filename or 'content'.",
            "check_type": "Always 'dictionary'.",
            "is_valid": "Whether all propertyClass values are valid.",
            "n_errors": "Number of undefined properties.",
            "errors": "List of undefined property messages.",
        },
    },
}
