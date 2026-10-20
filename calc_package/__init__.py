"""
Calculation Package Generator.

Produces professional, Mathcad-style calculation packages as
self-contained HTML files or LaTeX/PDF documents. Each supported
analysis module provides a calc_steps.py that defines inputs,
equations, and figures.

Usage
-----
>>> from bearing_capacity import BearingCapacityAnalysis, Footing
>>> from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
>>> from calc_package import generate_calc_package
>>>
>>> footing = Footing(width=2.0, depth=1.5, shape="square")
>>> soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30, unit_weight=18))
>>> analysis = BearingCapacityAnalysis(footing=footing, soil=soil)
>>> result = analysis.compute()
>>>
>>> # HTML output (default)
>>> html = generate_calc_package(
...     module="bearing_capacity",
...     result=result,
...     analysis=analysis,
...     project_name="I-95 Bridge",
...     engineer="S. O'Connell",
... )
>>>
>>> # PDF output (requires pdflatex)
>>> generate_calc_package(
...     module="bearing_capacity",
...     result=result,
...     analysis=analysis,
...     project_name="I-95 Bridge",
...     engineer="S. O'Connell",
...     output_path="calc.pdf",
...     format="pdf",
... )
"""

from calc_package.data_model import (
    CalcPackageData,
    CalcSection,
    CalcStep,
    InputItem,
    CheckItem,
    FigureData,
    TableData,
)
from calc_package.renderer import render_html, save_html, figure_to_base64
from calc_package.latex_renderer import render_latex, save_latex, save_pdf, render_pdf
from calc_package.equation_converter import unicode_to_latex, escape_latex
from calc_package.tables import (
    csv_file_names, export_tables, input_table, module_tables, tables_section,
    choices_section, assumed_section, given_section,
)


# The Word renderer is in funhouse_agent.docx_renderer (INF-30 step 5), which
# both pages use; calc_package.docx_renderer is the same module, loaded when
# it is first imported by either name, so importing this package loads
# nothing of funhouse_agent (its first-import notice included). The old name
# goes in step 16.
import importlib as _importlib
import importlib.util as _importlib_util
import sys as _sys

_MOVED = {__name__ + ".docx_renderer": "funhouse_agent.docx_renderer"}


class _MovedModule:
    """Finder and loader for :data:`_MOVED`: an old name imports the moved
    module itself (one module object, so a patch by either name reaches the
    code), never a second copy of it."""

    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname not in _MOVED:
            return None
        return _importlib_util.spec_from_loader(fullname, _MovedModule())

    def create_module(self, spec):
        module = _importlib.import_module(_MOVED[spec.name])
        self._spec = module.__spec__
        return module

    def exec_module(self, module):
        module.__spec__ = self._spec    # the import system set the old name's


# First, so an old name never resolves to a file of the same name elsewhere.
if not any(getattr(f, "__qualname__", "") == "_MovedModule"
           and getattr(f, "__module__", "") == __name__
           for f in _sys.meta_path):
    _sys.meta_path.insert(0, _MovedModule)

# Registry of supported modules — populated lazily on first use
_MODULE_REGISTRY = {}

#: module name -> its calc_steps module. ONE list, so ``list_supported_modules``
#: cannot drift from what ``_ensure_registered`` accepts (pavement_design was
#: importable but unlisted until 2026-09-11).
_IMPORT_MAP = {
    "bearing_capacity": "bearing_capacity.calc_steps",
    "lateral_pile": "lateral_pile.calc_steps",
    "slope_stability": "slope_stability.calc_steps",
    "settlement": "settlement.calc_steps",
    "axial_pile": "axial_pile.calc_steps",
    "drilled_shaft": "drilled_shaft.calc_steps",
    "downdrag": "downdrag.calc_steps",
    "seismic_geotech": "seismic_geotech.calc_steps",
    "retaining_walls": "retaining_walls.calc_steps",
    "pavement_design": "pavement_design.calc_steps",
    "ground_improvement": "ground_improvement.calc_steps",
    "wave_equation": "wave_equation.calc_steps",
    "pile_group": "pile_group.calc_steps",
    "sheet_pile": "sheet_pile.calc_steps",
    "soe": "soe.calc_steps",
    "reliability": "reliability.calc_steps",
    "fem2d": "fem2d.calc_steps",
}


def _ensure_registered(module_name: str) -> dict:
    """Lazy-import a module's calc_steps and register it."""
    if module_name in _MODULE_REGISTRY:
        return _MODULE_REGISTRY[module_name]

    _import_map = _IMPORT_MAP

    if module_name not in _import_map:
        raise ValueError(
            f"Module '{module_name}' does not have calc package support. "
            f"Supported modules: {list_supported_modules()}"
        )

    import importlib
    calc_steps = importlib.import_module(_import_map[module_name])

    _MODULE_REGISTRY[module_name] = {
        "display_name": calc_steps.DISPLAY_NAME,
        "references": calc_steps.REFERENCES,
        "get_input_summary": calc_steps.get_input_summary,
        "get_calc_steps": calc_steps.get_calc_steps,
        "get_figures": calc_steps.get_figures,
        # Optional (GM-39): tables that need the inputs as well as the
        # result. Without it the result's own get_tables() is used.
        "get_tables": getattr(calc_steps, "get_tables", None),
    }
    return _MODULE_REGISTRY[module_name]


def list_supported_modules() -> list:
    """Return list of module names that support calc package generation."""
    return list(_IMPORT_MAP)


def package_tables(module: str, result, analysis=None) -> list:
    """Every calculation table of a calc package, in order: the input echo
    (CSV only; the package prints it as its first section), then the
    module's own tables (:func:`calc_package.tables.module_tables`).

    Each is a :class:`geotech_common.tables.CalcTable`."""
    reg = _ensure_registered(module)
    out = []
    echo = input_table(reg["get_input_summary"](result, analysis))
    if echo is not None:
        out.append(echo)
    out.extend(module_tables(reg, result, analysis))
    return out


def generate_calc_package(
    module: str,
    result,
    analysis=None,
    project_name: str = "Project",
    project_number: str = "",
    engineer: str = "",
    checker: str = "",
    company: str = "",
    date: str = "",
    output_path: str = None,
    format: str = "html",
    keep_tex: bool = False,
    compiler: str = "pdflatex",
    render_report: dict = None,
    tables=True,
    table_files: list = None,
    assumed_inputs: list = None,
    given_inputs: list = None,
) -> str:
    """Generate a Mathcad-style calculation package.

    Parameters
    ----------
    module : str
        Module name (e.g. ``"bearing_capacity"``).
    result : dataclass
        The module's Results object from running the analysis.
    analysis : object, optional
        The module's Analysis object (holds inputs). Required for most modules.
    project_name : str
        Project name for the header.
    project_number : str
        Project number for the header.
    engineer : str
        Engineer name for the header.
    checker : str
        Checker name for the header.
    company : str
        Company name for the header.
    date : str
        Date string. Auto-filled from today if empty.
    output_path : str, optional
        If provided, saves to this file path.
    format : str
        Output format: ``"html"`` (default), ``"latex"``, or ``"pdf"``.
    keep_tex : bool
        For ``format="pdf"``: keep the .tex source alongside the PDF.
    compiler : str
        For ``format="pdf"``: LaTeX compiler (``"pdflatex"`` or ``"xelatex"``).
    render_report : dict, optional
        If a dict is passed, it is populated in place. For ``format="pdf"``
        with the :func:`~calc_package.latex_renderer.render_pdf` report
        (``{"renderer": "pdflatex"|"pymupdf_story", "warnings": [...], ...}``)
        so the caller can learn which engine produced the PDF. For every
        format, when this call writes the table CSVs, ``"tables"``:
        ``[{"title", "name", "path", "rows"}]``.
    tables : bool or list of CalcTable
        ``True`` (default): the module's calculation tables (backlog GM-39;
        :func:`package_tables`) are printed in a "Calculation Tables"
        section and, when ``output_path`` is given, each is saved beside the
        package as a CSV (``<stem>_t1_<name>.csv`` ..., and the input echo
        as ``<stem>_inputs.csv``). A list: these tables instead. ``False``:
        no tables.
    table_files : list of str, optional
        With a ``tables`` list: the CSV file names the caller has already
        written, parallel to the list. The package names them under each
        table and writes no CSV itself.
    assumed_inputs : list of dict, optional
        The inputs that were not given, one ``{input, value, basis}`` each
        (GM-155): the analysis's defaults and the caller's own
        assumptions. Printed as an "Assumed Inputs" section before the
        Values Used, so an assumption never reads as given. ``None``: no
        such section.
    given_inputs : list of dict, optional
        The inputs the user gave, one ``{input, value, words}`` each
        (GM-155, GM-167; ``words``: the user's own, when known). Printed
        first, as "Given Inputs"; an empty list says no input was named as
        the user's. ``None``: no such section.

    Returns
    -------
    str
        For ``"html"``/``"latex"``: the rendered document string.
        For ``"pdf"``: the absolute path to the PDF file.
    """
    # Early validation
    if format == "pdf" and not output_path:
        raise ValueError("output_path is required for format='pdf'")

    reg = _ensure_registered(module)

    # Build sections
    sections = []

    # The inputs given, then those assumed (GM-155): kept apart, so an
    # assumption never reads as given.
    given_part = given_section(given_inputs)
    if given_part is not None:
        sections.append(given_part)
    assumed_part = assumed_section(assumed_inputs)
    if assumed_part is not None:
        sections.append(assumed_part)

    # Section 1: the module's echo of every value used -- given, assumed
    # and worked out from them (a wall's heel length), so not "inputs".
    input_items = reg["get_input_summary"](result, analysis)
    if input_items:
        sections.append(CalcSection(title="Values Used", items=input_items))

    # What the analysis chose (GM-115): beside the inputs, before the steps.
    choices_part = choices_section(result, analysis)
    if choices_part is not None:
        sections.append(choices_part)

    # Section 2+: Calculation Steps (may return multiple sections)
    calc_sections = reg["get_calc_steps"](result, analysis)
    if isinstance(calc_sections, list) and calc_sections:
        if isinstance(calc_sections[0], CalcSection):
            sections.extend(calc_sections)
        else:
            # Flat list of CalcStep — wrap in one section
            sections.append(CalcSection(title="Calculations", items=calc_sections))

    # Figures section
    figures = reg["get_figures"](result, analysis)
    if figures:
        sections.append(CalcSection(title="Figures", items=figures))

    # Calculation tables (GM-39): printed here, and one CSV per table saved
    # beside the package.
    if tables is True:
        echo = input_table(input_items)
        try:
            own = module_tables(reg, result, analysis)
        except Exception as exc:        # noqa: BLE001 - never cost the package
            import warnings
            own = []
            message = (f"the calculation tables of '{module}' could not be "
                       f"built ({type(exc).__name__}: {exc})")
            warnings.warn(message)
            if render_report is not None:
                render_report["tables_error"] = message
        calc_tables = ([echo] if echo is not None else []) + own
    else:
        calc_tables = list(tables or [])
    csv_names = list(table_files or [])
    if calc_tables and not csv_names and output_path:
        import os
        paths = export_tables(
            tables=calc_tables,
            out_dir=os.path.dirname(os.path.abspath(output_path)),
            stem=os.path.splitext(os.path.basename(output_path))[0])
        csv_names = [os.path.basename(p) for p in paths]
        if render_report is not None:
            render_report["tables"] = [
                {"title": t.title, "name": t.name, "path": p,
                 "rows": t.n_rows} for t, p in zip(calc_tables, paths)]
    tables_part = tables_section(calc_tables, csv_names)
    if tables_part is not None:
        sections.append(tables_part)

    # Assemble
    data = CalcPackageData(
        project_name=project_name,
        project_number=project_number,
        analysis_type=reg["display_name"],
        engineer=engineer,
        checker=checker,
        date=date,
        company=company,
        sections=sections,
        references=reg["references"],
    )

    # ── Dispatch by format ──
    if format == "pdf":
        if not output_path:
            raise ValueError("output_path is required for format='pdf'")
        # Use render_pdf (not save_pdf) so the chosen renderer + warnings can be
        # threaded back to the caller via ``render_report`` (which engine
        # produced the PDF helps diagnose environment differences). The return
        # value stays the PDF path, unchanged.
        report = render_pdf(data, output_path, keep_tex=keep_tex,
                            compiler=compiler)
        if render_report is not None:
            render_report.update(report)
        return report["path"]

    elif format == "latex":
        tex = render_latex(data)
        if output_path:
            save_latex(data, output_path)
        return tex

    else:  # html (default)
        if output_path:
            save_html(data, output_path)
        return render_html(data)
