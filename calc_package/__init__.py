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
from calc_package.latex_renderer import render_latex, save_latex, save_pdf
from calc_package.equation_converter import unicode_to_latex, escape_latex


# Registry of supported modules — populated lazily on first use
_MODULE_REGISTRY = {}


def _ensure_registered(module_name: str) -> dict:
    """Lazy-import a module's calc_steps and register it."""
    if module_name in _MODULE_REGISTRY:
        return _MODULE_REGISTRY[module_name]

    _import_map = {
        "bearing_capacity": "bearing_capacity.calc_steps",
        "lateral_pile": "lateral_pile.calc_steps",
        "slope_stability": "slope_stability.calc_steps",
        "settlement": "settlement.calc_steps",
        "axial_pile": "axial_pile.calc_steps",
        "drilled_shaft": "drilled_shaft.calc_steps",
        "downdrag": "downdrag.calc_steps",
        "seismic_geotech": "seismic_geotech.calc_steps",
        "retaining_walls": "retaining_walls.calc_steps",
        "ground_improvement": "ground_improvement.calc_steps",
        "wave_equation": "wave_equation.calc_steps",
        "pile_group": "pile_group.calc_steps",
        "sheet_pile": "sheet_pile.calc_steps",
    }

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
    }
    return _MODULE_REGISTRY[module_name]


def list_supported_modules() -> list:
    """Return list of module names that support calc package generation."""
    return [
        "bearing_capacity", "lateral_pile", "slope_stability",
        "settlement", "axial_pile", "drilled_shaft", "downdrag",
        "seismic_geotech", "retaining_walls", "ground_improvement",
        "wave_equation", "pile_group", "sheet_pile",
    ]


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

    # Section 1: Input Parameters
    input_items = reg["get_input_summary"](result, analysis)
    if input_items:
        sections.append(CalcSection(title="Input Parameters", items=input_items))

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
        return save_pdf(data, output_path, keep_tex=keep_tex, compiler=compiler)

    elif format == "latex":
        tex = render_latex(data)
        if output_path:
            save_latex(data, output_path)
        return tex

    else:  # html (default)
        if output_path:
            save_html(data, output_path)
        return render_html(data)
