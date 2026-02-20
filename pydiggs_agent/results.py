"""
Result dataclasses for pydiggs_agent module.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiggValidationResult:
    """
    Result of DIGGS XML validation.

    Attributes:
        source: Filename or "content" for string input
        check_type: Type of validation ("schema", "dictionary", "schematron")
        schema_version: DIGGS schema version ("2.6", "2.5.a")
        is_valid: Whether the validation passed
        n_errors: Number of validation errors found
        errors: List of error messages (empty if valid)
    """
    source: str
    check_type: str
    schema_version: Optional[str] = None
    is_valid: bool = True
    n_errors: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        Generate a text summary of the validation result.

        Returns:
            Multi-line string describing the validation outcome
        """
        lines = []
        lines.append("DIGGS Validation Result")
        lines.append("=" * 50)
        lines.append(f"Source: {self.source}")
        lines.append(f"Check Type: {self.check_type}")

        if self.schema_version:
            lines.append(f"Schema Version: {self.schema_version}")

        lines.append(f"Valid: {self.is_valid}")
        lines.append(f"Number of Errors: {self.n_errors}")

        if self.errors:
            lines.append("\nErrors:")
            for i, error in enumerate(self.errors, 1):
                # Truncate very long error messages
                error_str = str(error)
                if len(error_str) > 200:
                    error_str = error_str[:197] + "..."
                lines.append(f"  {i}. {error_str}")
        else:
            lines.append("\nNo errors found.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert result to dictionary for JSON serialization.

        Returns:
            Dictionary with all result fields
        """
        return {
            "source": self.source,
            "check_type": self.check_type,
            "schema_version": self.schema_version,
            "is_valid": self.is_valid,
            "n_errors": self.n_errors,
            "errors": self.errors,
        }
