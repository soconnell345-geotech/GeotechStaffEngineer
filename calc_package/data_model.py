"""
Data model for calculation packages.

Defines the data structures that per-module calc_steps.py files
produce and the renderer consumes. Pure data — no rendering logic.

All units SI: meters, kPa, kN, degrees.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class InputItem:
    """A single input parameter echoed in the calc package.

    Attributes
    ----------
    name : str
        Short variable name, e.g. "B".
    description : str
        Human-readable description, e.g. "Footing width".
    value : Any
        The value (number, string, etc.).
    unit : str
        Unit string, e.g. "m", "kPa", "deg".
    """
    name: str
    description: str
    value: Any
    unit: str = ""


@dataclass
class CalcStep:
    """A single calculation step shown Mathcad-style.

    Shows: title, general equation, equation with values substituted,
    and the computed result.

    Attributes
    ----------
    title : str
        Step title, e.g. "Bearing Capacity Factor Nq".
    equation : str
        General equation in text form (HTML entities OK).
    substitution : str
        Equation with numerical values substituted in.
    result_name : str
        Result variable name, e.g. "Nq".
    result_value : float
        Computed value.
    result_unit : str
        Unit of result, e.g. "kPa".
    reference : str
        Citation, e.g. "Vesic (1973), GEC-6 Eq. 6-2".
    notes : str
        Optional additional notes.
    """
    title: str
    equation: str
    substitution: str
    result_name: str
    result_value: float
    result_unit: str = ""
    reference: str = ""
    notes: str = ""


@dataclass
class CheckItem:
    """A pass/fail engineering check.

    Attributes
    ----------
    description : str
        What is being checked, e.g. "Bearing capacity adequacy".
    demand : float
        Applied load or demand value.
    demand_label : str
        Label for demand, e.g. "q_applied".
    capacity : float
        Resistance or capacity value.
    capacity_label : str
        Label for capacity, e.g. "q_allowable".
    unit : str
        Common unit for demand and capacity, e.g. "kPa".
    passes : bool
        True if demand <= capacity.
    """
    description: str
    demand: float
    demand_label: str
    capacity: float
    capacity_label: str
    unit: str = ""
    passes: bool = True


@dataclass
class FigureData:
    """A matplotlib figure rendered as a base64 image.

    Attributes
    ----------
    title : str
        Figure title.
    image_base64 : str
        Base64-encoded PNG image data.
    caption : str
        Figure caption shown below the image.
    width_percent : int
        Display width as percentage of page width.
    """
    title: str
    image_base64: str
    caption: str = ""
    width_percent: int = 80


@dataclass
class TableData:
    """A data table (e.g., per-layer breakdown, slice data).

    Attributes
    ----------
    title : str
        Table title.
    headers : list of str
        Column headers.
    rows : list of list
        Row data (each row is a list of values).
    notes : str
        Optional footnotes.
    """
    title: str
    headers: List[str] = field(default_factory=list)
    rows: List[list] = field(default_factory=list)
    notes: str = ""


@dataclass
class CalcSection:
    """A logical section of the calculation package.

    Attributes
    ----------
    title : str
        Section heading.
    items : list
        Mixed list of InputItem, CalcStep, CheckItem, FigureData,
        TableData, or str (plain text paragraphs).
    """
    title: str
    items: list = field(default_factory=list)


@dataclass
class CalcPackageData:
    """Complete data for rendering one calculation package.

    Attributes
    ----------
    project_name : str
        Project name, e.g. "I-95 Bridge Abutment".
    project_number : str
        Project number, e.g. "2024-001".
    analysis_type : str
        Display title, e.g. "Bearing Capacity Analysis".
    engineer : str
        Prepared by, e.g. "S. O'Connell".
    checker : str
        Checked by (optional).
    date : str
        Date string. Auto-filled from today if empty.
    company : str
        Company name (optional).
    sections : list of CalcSection
        Ordered sections of the calc package.
    references : list of str
        Reference citations.
    """
    project_name: str = "Project"
    project_number: str = ""
    analysis_type: str = "Geotechnical Analysis"
    engineer: str = ""
    checker: str = ""
    date: str = ""
    company: str = ""
    sections: List[CalcSection] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
