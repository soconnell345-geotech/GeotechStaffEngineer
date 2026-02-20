"""
Result dataclasses for AGS4 agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AGS4ReadResult:
    """Results from reading an AGS4 file.

    Attributes
    ----------
    filepath : str
        Source file path or '<string>' for StringIO input.
    n_groups : int
        Number of AGS4 groups (tables) found.
    group_names : list of str
        Names of all groups.
    group_row_counts : dict
        Mapping of group name to data row count.
    tables : dict or None
        Dict of group_name -> list of row dicts (for JSON serialization).
        Only populated if include_data=True.
    """
    filepath: str = ""
    n_groups: int = 0
    group_names: List[str] = field(default_factory=list)
    group_row_counts: Dict[str, int] = field(default_factory=dict)
    tables: Optional[Dict] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  AGS4 FILE READ RESULTS",
            "=" * 60,
            f"  Source:        {self.filepath}",
            f"  Groups found:  {self.n_groups}",
        ]
        for name in self.group_names:
            rows = self.group_row_counts.get(name, 0)
            lines.append(f"    {name:20s}  {rows} rows")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "filepath": self.filepath,
            "n_groups": self.n_groups,
            "group_names": list(self.group_names),
            "group_row_counts": dict(self.group_row_counts),
        }
        if self.tables is not None:
            d["tables"] = self.tables
        return d


@dataclass
class AGS4ValidationResult:
    """Results from validating an AGS4 file.

    Attributes
    ----------
    filepath : str
        Source file path.
    n_errors : int
        Number of errors found.
    n_warnings : int
        Number of warnings found.
    n_fyi : int
        Number of FYI messages.
    is_valid : bool
        True if no errors found (warnings/FYI are acceptable).
    errors : dict
        Error details grouped by rule number.
    """
    filepath: str = ""
    n_errors: int = 0
    n_warnings: int = 0
    n_fyi: int = 0
    is_valid: bool = True
    errors: Dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            "=" * 60,
            f"  AGS4 FILE VALIDATION — {status}",
            "=" * 60,
            f"  Source:     {self.filepath}",
            f"  Errors:    {self.n_errors}",
            f"  Warnings:  {self.n_warnings}",
            f"  FYI:       {self.n_fyi}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "filepath": self.filepath,
            "n_errors": self.n_errors,
            "n_warnings": self.n_warnings,
            "n_fyi": self.n_fyi,
            "is_valid": self.is_valid,
            "errors": self.errors,
        }
