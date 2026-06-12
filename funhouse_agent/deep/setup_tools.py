"""LangChain tools for the MODEL-SETUP sub-agent (staged, human-gated).

The tools operate on ONE shared :class:`ProjectStore` (closured in by
:func:`make_setup_tools`, the same factory pattern as
:mod:`funhouse_agent.deep.tools`). Every tool returns a JSON string.

The design inversion these tools enforce: the agent NEVER claims to have
read geometry correctly. It builds a Project (template / DXF / PDF / typed
points / quarantined vision draft), renders an ECHO-BACK cross-section
(numbers → image, which a human can verify at a glance), and requests
confirmation through ``request_confirmation`` — the ONLY way the staged
gates (geometry / materials / water_loads) get set. ``project_run`` refuses
until every gate is set, and any patch to a stage's data clears that
stage's gate again.

``request_confirmation`` human-gate mechanics (both offline-tested):

* **Durable path** — inside a LangGraph run WITH a checkpointer it calls
  ``langgraph.types.interrupt(payload)``; the host surfaces the payload
  (``__interrupt__``), collects the human's answer (chat or a rendered
  form) and resumes with ``Command(resume={"approved": bool,
  "edits": {path: value}})``.
* **Chat fallback** — with no checkpointer/run context, ``interrupt()``
  raises; the tool catches that (NEVER ``GraphInterrupt`` itself) and
  returns the payload as formatted chat text with numbered questions. The
  agent relays it to the user and calls ``request_confirmation`` again with
  ``user_response={"approved": ..., "edits": {...}}``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from geo_project.schema import CONFIRMATION_STAGES, Project
from geo_project.validate import has_errors, summarize, validate

from funhouse_agent.deep.tools import DEFAULT_MAX_RESULT_CHARS, _truncate

#: Stages request_confirmation accepts: the three gate stages plus the
#: non-gating 'analysis_plan' review (logged to the ledger only — the run
#: gate is the three flags, per the staged protocol).
CONFIRMABLE_STAGES = tuple(CONFIRMATION_STAGES) + ("analysis_plan",)


# ---------------------------------------------------------------------------
# ProjectStore
# ---------------------------------------------------------------------------

@dataclass
class ProjectStore:
    """Shared mutable state behind one setup-agent's tool set."""
    project: Optional[Project] = None
    render_dir: str = "model_setup_renders"
    last_image_path: Optional[str] = None
    last_vertex_table: str = ""
    render_count: int = 0
    history: List[str] = field(default_factory=list)

    def require_project(self) -> Project:
        if self.project is None:
            raise ValueError(
                "No project loaded — create one first with project_new.")
        return self.project


# ---------------------------------------------------------------------------
# Dot-path patching (targeted updates, never whole-document rewrites)
# ---------------------------------------------------------------------------

def _parse_path(path: str) -> List[Any]:
    """'stratigraphy[0].material.phi' → ['stratigraphy', 0, 'material', 'phi']."""
    tokens: List[Any] = []
    for part in path.split("."):
        while "[" in part:
            head, rest = part.split("[", 1)
            idx, part = rest.split("]", 1)
            if head:
                tokens.append(head)
            tokens.append(int(idx))
        if part:
            tokens.append(part)
    if not tokens:
        raise ValueError(f"Empty patch path: '{path}'")
    return tokens


def _stage_for_path(path: str) -> Optional[str]:
    """Which confirmation gate a patch path invalidates (None = none)."""
    p = path.strip()
    if p.startswith("geometry"):
        return "geometry"
    if p.startswith("stratigraphy"):
        return "materials" if ".material" in p else "geometry"
    if p.startswith(("water", "loads", "reinforcement")):
        return "water_loads"
    return None


def apply_patch(project: Project, path: str, value: Any = None,
                op: str = "set") -> Project:
    """Apply ONE targeted update and return the re-normalized Project.

    Operates on the JSON form (``to_dict`` → navigate → ``from_dict``) so
    every patched value passes through the same normalization/validation as
    a loaded document. ``confirmations`` paths are REFUSED — only the
    confirmation tool sets gates.

    Parameters
    ----------
    path : str
        Dot path with list indices, e.g. ``stratigraphy[0].material.phi``,
        ``water.gwt_points``, ``analyses[0].method``,
        ``geometry.layer_boundaries.clay_top``.
    value : Any
        JSON value for 'set'/'append'. Ignored for 'delete'.
    op : str
        'set' (default) | 'append' (target list) | 'delete' (list index or
        dict key).
    """
    if path.strip().startswith("confirmations"):
        raise ValueError(
            "confirmations are set ONLY via request_confirmation — "
            "patching them directly is refused.")
    if op not in ("set", "append", "delete"):
        raise ValueError(f"op must be set|append|delete, got '{op}'")

    doc = project.to_dict()
    tokens = _parse_path(path)
    parent = doc
    for tok in tokens[:-1]:
        if isinstance(tok, int):
            if not isinstance(parent, list) or not (-len(parent) <= tok < len(parent)):
                raise ValueError(f"Bad index [{tok}] in path '{path}'")
            parent = parent[tok]
        else:
            if not isinstance(parent, dict) or tok not in parent:
                raise ValueError(f"Unknown key '{tok}' in path '{path}'")
            parent = parent[tok]

    last = tokens[-1]
    if op == "set":
        if isinstance(last, int):
            if not isinstance(parent, list) or not (-len(parent) <= last < len(parent)):
                raise ValueError(f"Bad index [{last}] in path '{path}'")
            parent[last] = value
        else:
            if not isinstance(parent, dict):
                raise ValueError(f"Cannot set key '{last}' in path '{path}'")
            if last not in parent and not _is_open_dict(tokens[:-1]):
                raise ValueError(
                    f"Unknown field '{last}' in path '{path}' (schema "
                    f"fields only; check spelling)")
            parent[last] = value
    elif op == "append":
        if isinstance(last, int):
            raise ValueError("append targets a list FIELD, not an index")
        target = parent.get(last) if isinstance(parent, dict) else None
        if not isinstance(target, list):
            raise ValueError(f"'{path}' is not a list — cannot append")
        target.append(value)
    else:  # delete
        if isinstance(last, int):
            if not isinstance(parent, list) or not (-len(parent) <= last < len(parent)):
                raise ValueError(f"Bad index [{last}] in path '{path}'")
            parent.pop(last)
        else:
            if not isinstance(parent, dict) or last not in parent:
                raise ValueError(f"Unknown key '{last}' in path '{path}'")
            parent.pop(last)

    return Project.from_dict(doc)


#: Dict fields whose keys are user-defined (new keys allowed on set).
_OPEN_DICT_NAMES = {"layer_boundaries", "probabilistic"}


def _is_open_dict(parent_tokens: List[Any]) -> bool:
    """True when the parent container accepts arbitrary new keys.

    Open containers: ``geometry.layer_boundaries`` and
    ``material.probabilistic`` (plus the per-parameter dicts directly under
    ``probabilistic``).
    """
    names = [t for t in parent_tokens if isinstance(t, str)]
    if not names:
        return False
    if names[-1] in _OPEN_DICT_NAMES:
        return True
    return len(names) >= 2 and names[-2] == "probabilistic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json(obj: Any, cap: int) -> str:
    return _truncate(json.dumps(obj, default=str), cap)


def _validation_block(project: Project) -> Dict[str, Any]:
    issues = validate(project)
    return {
        "ok": not has_errors(issues),
        "n_errors": sum(1 for i in issues if i.level == "error"),
        "n_warnings": sum(1 for i in issues if i.level == "warning"),
        "issues": [i.to_dict() for i in issues],
    }


def _project_summary(project: Project) -> Dict[str, Any]:
    g = project.geometry
    return {
        "name": project.meta.name,
        "provenance": g.provenance,
        "n_surface_points": len(g.surface_points),
        "x_range": list(g.x_range),
        "layers": [
            {"name": L.name,
             "top": project.layer_top(i),
             "bottom": project.layer_bottom(i),
             "strength_model": L.material.strength_model}
            for i, L in enumerate(project.stratigraphy)
        ],
        "has_gwt": bool(project.water.gwt_points),
        "n_surcharges": len(project.loads.surcharges),
        "kh": project.loads.kh,
        "reinforcement": {
            "nails": len(project.reinforcement.nails),
            "anchors": len(project.reinforcement.anchors),
            "geosynthetics": len(project.reinforcement.geosynthetics),
        },
        "analyses": [
            {"name": a.name, "type": a.type} for a in project.analyses
        ],
        "confirmations": {
            s: getattr(project.confirmations, s)
            for s in CONFIRMATION_STAGES
        },
        "n_assumptions": len(project.assumptions),
    }


def _strip_private(obj: Any) -> Any:
    """Drop '_'-prefixed keys (raw result objects) for JSON-safe returns."""
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _interrupt_available() -> bool:
    """True when langgraph ``interrupt()`` can actually pause-and-resume.

    That requires BOTH a runnable graph context AND a checkpointer
    (``interrupt`` raises GraphInterrupt regardless, but without a
    checkpointer the run cannot be resumed — the turn would just die).
    Detection: the graph runtime injects the checkpointer under the
    ``__pregel_checkpointer`` configurable (``langgraph.constants.
    CONFIG_KEY_CHECKPOINTER``); absent/None means chat fallback.
    """
    try:
        from langgraph.config import get_config
        from langgraph.constants import CONFIG_KEY_CHECKPOINTER
        conf = get_config()
    except Exception:
        return False
    return bool((conf or {}).get("configurable", {})
                .get(CONFIG_KEY_CHECKPOINTER))


def _fallback_text(payload: Dict[str, Any]) -> str:
    """Render the confirmation payload as numbered chat questions."""
    lines = [
        f"=== CONFIRMATION REQUIRED: {payload['stage']} stage ===",
        payload.get("summary_markdown", ""),
    ]
    if payload.get("image_path"):
        lines.append(f"[Echo-back rendering: {payload['image_path']} — "
                     "open it and verify the section visually]")
    if payload.get("vertex_table"):
        lines.append(payload["vertex_table"])
    if payload.get("assumptions"):
        lines.append("Assumptions taken so far (confirm or correct):")
        for a in payload["assumptions"]:
            src = f" [{a.get('source')}]" if a.get("source") else ""
            lines.append(f"  - {a.get('field')}: {a.get('value')}{src}")
    qn = 0
    for item in payload.get("form_schema") or []:
        qn += 1
        unit = f" ({item['unit']})" if item.get("unit") else ""
        default = (f" [default: {item['default']}]"
                   if item.get("default") is not None else "")
        opts = (f" options: {item['options']}"
                if item.get("options") else "")
        lines.append(f"  {qn}. {item.get('label', item.get('name'))}"
                     f"{unit}{opts}{default}")
    qn += 1
    lines.append(f"  {qn}. Approve this stage as shown? (yes / no + "
                 "corrections)")
    return "\n".join(s for s in lines if s)


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

class _ProjectNewArgs(BaseModel):
    """Args for project_new (one source at a time)."""
    model_config = ConfigDict(extra="forbid")

    template: str = Field(
        default="",
        description=("Template name: simple_slope | benched_slope | "
                     "embankment_on_foundation | cut_with_berm. Use with "
                     "'params'."))
    params: Optional[Dict[str, Any]] = Field(
        default=None, description="Template keyword arguments (SI units).")
    dxf_path: str = Field(
        default="", description=("Path to a DXF file. REQUIRES "
                                 "'layer_mapping' from dxf_discover + the "
                                 "user."))
    layer_mapping: Optional[Dict[str, Any]] = Field(
        default=None,
        description=('USER-CONFIRMED DXF mapping: {"surface": layer, '
                     '"soil_boundaries": {dxf_layer: soil_name}, '
                     '"water_table": layer?, "nails": layer?}.'))
    units: str = Field(default="m", description="DXF drawing units.")
    pdf_path: str = Field(
        default="", description="Path to a PDF (vector line work).")
    role_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description=('PDF color→role mapping, e.g. {"#000000": "surface", '
                     '"#0000ff": "gwt", "#808080": "boundary_Clay"}.'))
    scale: float = Field(default=1.0, description="PDF units → meters.")
    surface_points: Optional[List[List[float]]] = Field(
        default=None, description="Typed-in surface [(x, z), ...].")
    layer_boundaries: Optional[Dict[str, List[List[float]]]] = Field(
        default=None, description="Typed-in boundary polylines by name.")
    layer_names: Optional[List[str]] = Field(
        default=None, description="Layer names top→bottom (points source).")
    gwt_points: Optional[List[List[float]]] = Field(
        default=None, description="GWT polyline (points source).")
    vision_draft: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("A vision-extracted draft {surface_points, "
                     "boundary_profiles?, gwt_points?}. QUARANTINED: "
                     "analysis stays blocked until the human confirms the "
                     "echo-back."))
    section_bottom: Optional[float] = Field(
        default=None, description="Flat bottom elevation of the section (m).")
    name: str = Field(default="", description="Project name.")


def make_setup_tools(store: Optional[ProjectStore] = None,
                     render_dir: Optional[str] = None,
                     max_result_chars: int = DEFAULT_MAX_RESULT_CHARS,
                     ) -> List[StructuredTool]:
    """Build the MODEL-SETUP tool set bound to one :class:`ProjectStore`.

    Parameters
    ----------
    store : ProjectStore, optional
        Shared state; a fresh one is created when omitted (reach it later
        via the ``store`` attribute stapled onto each returned tool's
        ``.metadata``).
    render_dir : str, optional
        Where echo-back PNGs are written (created on demand).
    max_result_chars : int
        Tool-result size cap (same convention as deep.tools).
    """
    store = store if store is not None else ProjectStore()
    if render_dir:
        store.render_dir = render_dir
    cap = max_result_chars

    # -- creation ----------------------------------------------------------

    def project_new(**kwargs) -> str:
        """Create the working Project from ONE source: a parametric template,
        a DXF (with a user-confirmed layer mapping), a vector PDF, typed-in
        points, or a QUARANTINED vision draft."""
        args = _ProjectNewArgs(**kwargs)
        sources = [bool(args.template), bool(args.dxf_path),
                   bool(args.pdf_path), args.surface_points is not None,
                   args.vision_draft is not None]
        if sum(sources) != 1:
            return _json({"error": (
                "Provide exactly ONE source: template | dxf_path | "
                "pdf_path | surface_points | vision_draft.")}, cap)
        try:
            if args.template:
                from geo_project.templates import TEMPLATES
                if args.template not in TEMPLATES:
                    return _json({
                        "error": f"Unknown template '{args.template}'.",
                        "available": {k: (v.__doc__ or "").split("\n")[0]
                                      for k, v in TEMPLATES.items()},
                    }, cap)
                params = dict(args.params or {})
                if args.name:
                    params.setdefault("name", args.name)
                project = TEMPLATES[args.template](**params)
            elif args.dxf_path:
                from geo_project.ingest import discover_dxf, from_dxf
                if not args.layer_mapping:
                    inventory = discover_dxf(args.dxf_path)
                    return _json({
                        "status": "layer_mapping_required",
                        "message": (
                            "DXF layers discovered. ASK THE USER which CAD "
                            "layer is the ground surface, which are soil "
                            "boundaries (and each soil's name), water "
                            "table, and nails — do NOT guess. Then call "
                            "project_new again with layer_mapping."),
                        "inventory": inventory,
                    }, cap)
                project = from_dxf(
                    args.dxf_path, layer_mapping=args.layer_mapping,
                    units=args.units, section_bottom=args.section_bottom,
                    name=args.name or "DXF import")
            elif args.pdf_path:
                from geo_project.ingest import from_pdf_vector
                project = from_pdf_vector(
                    args.pdf_path, role_mapping=args.role_mapping,
                    scale=args.scale, section_bottom=args.section_bottom,
                    name=args.name or "PDF vector import")
            elif args.vision_draft is not None:
                from geo_project.ingest import from_vision_draft
                project = from_vision_draft(
                    args.vision_draft,
                    name=args.name or "Vision draft (UNCONFIRMED)",
                    section_bottom=args.section_bottom)
            else:
                from geo_project.ingest import from_points
                project = from_points(
                    [tuple(p) for p in args.surface_points],
                    layer_boundaries={
                        k: [tuple(p) for p in v]
                        for k, v in (args.layer_boundaries or {}).items()
                    } or None,
                    layer_names=args.layer_names,
                    gwt_points=([tuple(p) for p in args.gwt_points]
                                if args.gwt_points else None),
                    section_bottom=args.section_bottom,
                    name=args.name or "User-defined section")
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"}, cap)

        store.project = project
        store.last_image_path = None
        store.last_vertex_table = ""
        store.history.append(f"project_new: {project.meta.name}")
        return _json({
            "status": "created",
            "summary": _project_summary(project),
            "validation": _validation_block(project),
            "next": ("Render the echo-back (project_render) and request "
                     "geometry confirmation BEFORE anything else."),
        }, cap)

    def dxf_discover(path: str) -> str:
        """Inventory a DXF's layers (entity counts/types, sample text, bbox)
        so the USER can say which layer is which. Always run this before
        mapping a DXF — never guess a layer mapping."""
        try:
            from geo_project.ingest import discover_dxf as _disc
            return _json(_disc(path), cap)
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"}, cap)

    # -- inspection ----------------------------------------------------------

    def project_show(section: str = "") -> str:
        """Show the working Project (full JSON document + validation).
        Optional 'section' limits output to one top-level key (e.g.
        'stratigraphy', 'geometry', 'analyses', 'assumptions')."""
        try:
            project = store.require_project()
        except ValueError as exc:
            return _json({"error": str(exc)}, cap)
        doc = project.to_dict()
        if section:
            if section not in doc:
                return _json({"error": f"Unknown section '{section}'.",
                              "sections": sorted(doc)}, cap)
            return _json({section: doc[section],
                          "validation": _validation_block(project)}, cap)
        return _json({"project": doc,
                      "validation": _validation_block(project)}, cap)

    def project_validate() -> str:
        """Run the deterministic validators; returns the issue list
        (level/code/message/fix_hint) and whether the project is runnable."""
        try:
            project = store.require_project()
        except ValueError as exc:
            return _json({"error": str(exc)}, cap)
        block = _validation_block(project)
        block["summary"] = summarize(validate(project))
        return _json(block, cap)

    # -- editing ----------------------------------------------------------

    def project_patch(path: str, value: Any = None, op: str = "set") -> str:
        """Apply ONE targeted update to the Project (never rewrite the whole
        document). path examples: 'stratigraphy[0].material.phi',
        'water.gwt_points', 'loads.surcharges' (op='append' with a surcharge
        dict), 'analyses' (op='append' with {'type': 'le'|'fem_srm', ...}),
        'geometry.layer_boundaries.clay_top'. Patching a stage's data CLEARS
        that stage's confirmation gate. 'confirmations' paths are refused."""
        try:
            project = store.require_project()
            new_project = apply_patch(project, path, value, op)
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"}, cap)

        stage = _stage_for_path(path)
        was_confirmed = bool(stage) and getattr(project.confirmations, stage)
        if was_confirmed:
            setattr(new_project.confirmations, stage, False)
        reset_stage = stage if was_confirmed else None
        store.project = new_project
        store.history.append(f"project_patch: {op} {path}")
        return _json({
            "status": "patched",
            "path": path,
            "op": op,
            "confirmation_reset": reset_stage,
            "confirmations": {s: getattr(new_project.confirmations, s)
                              for s in CONFIRMATION_STAGES},
            "validation": _validation_block(new_project),
        }, cap)

    # -- echo-back ----------------------------------------------------------

    def project_render(filename: str = "") -> str:
        """Render the ECHO-BACK cross-section PNG + numbered vertex table —
        the artifact the human verifies. ALWAYS render before requesting
        confirmation of a stage."""
        try:
            project = store.require_project()
            from geo_project.render import echo_back
            os.makedirs(store.render_dir, exist_ok=True)
            store.render_count += 1
            fname = filename or f"echo_back_{store.render_count:03d}.png"
            path = os.path.join(store.render_dir, fname)
            eb = echo_back(project, path)
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"}, cap)
        store.last_image_path = eb.image_path
        store.last_vertex_table = eb.vertex_table
        store.history.append(f"project_render: {eb.image_path}")
        return _json({
            "status": "rendered",
            "image_path": eb.image_path,
            "vertex_table": eb.vertex_table,
            "note": ("Show the image (and vertex table) to the user; they "
                     "confirm VISUALLY. Numbers→image is checkable; "
                     "image→numbers is never trusted."),
        }, cap)

    # -- knowledge ----------------------------------------------------------

    def cov_lookup(property: str, soil_type: str = "", test: str = "",
                   category: str = "") -> str:
        """Published COV guidance for a soil property (Duncan 2000 / ISSMGE
        TC304 / Phoon-Kulhawy), WITH sources — use these (and cite them in
        the assumption ledger) when proposing probabilistic inputs or
        sanity-checking parameter variability. COV values are in PERCENT."""
        try:
            from reliability.cov_database import cov_guidance
            rows = cov_guidance(property,
                                soil_type=soil_type or None,
                                test=test or None,
                                category=category or None)
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"}, cap)
        return _json({
            "property": property,
            "n_rows": len(rows),
            "rows": [r.to_dict() for r in rows],
            "note": ("COV values are PERCENT — divide by 100 for the "
                     "Project's probabilistic.cov fraction. Cite the row's "
                     "source in the assumption ledger."),
        }, cap)

    # -- the human gate ----------------------------------------------------------

    def request_confirmation(stage: str, summary_markdown: str,
                             form_schema: Optional[List[Dict[str, Any]]] = None,
                             user_response: Optional[Dict[str, Any]] = None,
                             ) -> str:
        """THE human gate. Present a stage (geometry | materials |
        water_loads | analysis_plan) for confirmation with a summary, the
        latest echo-back render, the assumption ledger, and an optional
        form_schema [{name(patch path), label, type: number|choice|bool|
        text, options?, default?, unit?}]. In a durable run this pauses via
        interrupt() and resumes with {approved, edits}; otherwise it returns
        numbered questions to relay in chat — then call this tool AGAIN with
        user_response={'approved': bool, 'edits': {path: value}}. Approval
        sets the stage's gate; edits are applied as patches FIRST (approve-
        with-edits = correct then confirm)."""
        try:
            project = store.require_project()
        except ValueError as exc:
            return _json({"error": str(exc)}, cap)
        if stage not in CONFIRMABLE_STAGES:
            return _json({"error": f"Unknown stage '{stage}'.",
                          "stages": list(CONFIRMABLE_STAGES)}, cap)

        payload = {
            "type": "model_setup_confirmation",
            "stage": stage,
            "summary_markdown": summary_markdown,
            "image_path": store.last_image_path,
            "vertex_table": store.last_vertex_table,
            "form_schema": list(form_schema or []),
            "assumptions": [
                {"field": a.field, "value": a.value, "source": a.source}
                for a in project.assumptions
            ],
            "resume_contract": {"approved": "bool",
                                "edits": "{patch_path: value}"},
        }

        if user_response is None:
            if _interrupt_available():
                # Durable path: pause the graph; the host resumes with
                # Command(resume={approved, edits}). GraphInterrupt MUST
                # propagate — it IS the pause mechanism.
                from langgraph.types import interrupt
                resume = interrupt(payload)
            else:
                # No checkpointer / not inside a graph: chat fallback.
                return _json({
                    "status": "needs_user",
                    "stage": stage,
                    "chat_text": _fallback_text(payload),
                    "instructions": (
                        "Relay chat_text (and the echo-back image path) to "
                        "the user verbatim, wait for their answer, then "
                        "call request_confirmation AGAIN with this stage "
                        "and user_response={'approved': bool, 'edits': "
                        "{patch_path: value}}. Do NOT proceed without it."),
                }, cap)
        else:
            resume = user_response

        if not isinstance(resume, dict):
            return _json({"error": (
                f"Resume value must be a dict {{approved, edits}}, got "
                f"{type(resume).__name__}.")}, cap)
        approved = bool(resume.get("approved"))
        edits = resume.get("edits") or {}

        applied, failed = [], []
        for epath, evalue in edits.items():
            try:
                store.project = apply_patch(store.project, epath, evalue)
                applied.append(epath)
            except Exception as exc:
                failed.append({"path": epath,
                               "error": f"{type(exc).__name__}: {exc}"})
        project = store.project

        if approved and not failed and stage in CONFIRMATION_STAGES:
            setattr(project.confirmations, stage, True)
        if approved and stage == "analysis_plan":
            project.add_assumption(
                "analyses", "analysis plan approved by user",
                source="request_confirmation(analysis_plan)")
        store.history.append(
            f"request_confirmation: {stage} -> "
            f"{'approved' if approved else 'NOT approved'}"
            f"{' (with edits)' if applied else ''}")

        return _json({
            "status": "confirmed" if approved and not failed else
                      ("edits_failed" if failed else "rejected"),
            "stage": stage,
            "approved": approved,
            "edits_applied": applied,
            "edits_failed": failed,
            "confirmations": {s: getattr(project.confirmations, s)
                              for s in CONFIRMATION_STAGES},
            "validation": _validation_block(project),
            "note": ("" if approved else
                     "Address the user's corrections (project_patch), "
                     "re-render, and request confirmation again."),
        }, cap)

    # -- the run gate ----------------------------------------------------------

    def project_run() -> str:
        """Execute the project's requested analyses. REFUSES unless every
        stage gate (geometry, materials, water_loads) is human-confirmed AND
        validation has no errors. Returns JSON-safe results (LE FOS +
        critical surface + optional FOSM/MC; FEM FOS)."""
        try:
            project = store.require_project()
        except ValueError as exc:
            return _json({"error": str(exc)}, cap)

        missing = project.confirmations.missing()
        if missing:
            return _json({
                "status": "refused",
                "error": ("UNCONFIRMED STAGES — project_run is gated on "
                          "human confirmation of every stage."),
                "missing_confirmations": missing,
                "directive": ("Walk the remaining stage(s): render the "
                              "echo-back, request_confirmation, and only "
                              "then run."),
            }, cap)
        issues = validate(project)
        if has_errors(issues):
            return _json({
                "status": "refused",
                "error": "Validation errors block the run.",
                "validation": _validation_block(project),
            }, cap)
        if not project.analyses:
            return _json({
                "status": "refused",
                "error": ("No analyses defined — append one first "
                          "(project_patch op='append' path='analyses')."),
            }, cap)
        try:
            from geo_project.builders import run_analyses
            results = run_analyses(project)
        except Exception as exc:
            return _json({"error": f"{type(exc).__name__}: {exc}"}, cap)
        store.history.append("project_run: executed")
        return _json({
            "status": "complete",
            "results": _strip_private(results),
        }, cap)

    tools = [
        StructuredTool.from_function(
            project_new, name="project_new", args_schema=_ProjectNewArgs,
            description=(
                "Create the working Project from ONE source: template "
                "(+params), dxf_path (+user-confirmed layer_mapping), "
                "pdf_path (+role_mapping), surface_points, or a "
                "QUARANTINED vision_draft. With dxf_path and no mapping it "
                "returns the layer inventory to ask the user about."),
        ),
        StructuredTool.from_function(
            dxf_discover, name="dxf_discover",
            description=(
                "Inventory a DXF's layers (counts, entity types, sample "
                "texts, bboxes) so the USER can map layers to roles. Run "
                "before any DXF import — never guess the mapping."),
        ),
        StructuredTool.from_function(
            project_show, name="project_show",
            description=("Show the Project JSON (optionally one section) "
                         "plus the current validation report."),
        ),
        StructuredTool.from_function(
            project_validate, name="project_validate",
            description=("Run deterministic checks; returns issues "
                         "(error/warning/info) with fix hints."),
        ),
        StructuredTool.from_function(
            project_patch, name="project_patch",
            description=(
                "Apply ONE targeted update via dot-path (set/append/"
                "delete). Patching a stage's data clears that stage's "
                "confirmation gate; 'confirmations' paths are refused."),
        ),
        StructuredTool.from_function(
            project_render, name="project_render",
            description=(
                "Render the ECHO-BACK cross-section PNG + numbered vertex "
                "table for human verification. Always render before "
                "requesting confirmation."),
        ),
        StructuredTool.from_function(
            cov_lookup, name="cov_lookup",
            description=(
                "Published COV guidance (Duncan 2000 / ISSMGE TC304) with "
                "sources for a property (phi, su, gamma, ...). Percent "
                "values; cite the source in the assumption ledger."),
        ),
        StructuredTool.from_function(
            request_confirmation, name="request_confirmation",
            description=(
                "THE human gate: present a stage (geometry | materials | "
                "water_loads | analysis_plan) with summary + echo-back + "
                "assumptions + optional form_schema. Durable runs pause "
                "via interrupt(); otherwise relay the returned numbered "
                "questions and call again with user_response. Only this "
                "tool sets confirmation gates."),
        ),
        StructuredTool.from_function(
            project_run, name="project_run",
            description=(
                "Run the requested analyses. REFUSES until geometry, "
                "materials AND water_loads are human-confirmed and "
                "validation is error-free."),
        ),
    ]
    for t in tools:
        t.metadata = dict(t.metadata or {})
        t.metadata["store"] = store
    return tools


__all__ = ["ProjectStore", "make_setup_tools", "apply_patch",
           "CONFIRMABLE_STAGES"]
