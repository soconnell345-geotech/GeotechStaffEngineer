"""
LaTeX-to-PDF compiler wrapper.

Calls pdflatex (or xelatex) via subprocess to compile a .tex file
into a PDF.  Provides helpful error messages when the compiler is
not installed.
"""

import subprocess
import shutil
from pathlib import Path


def find_latex_compiler(compiler: str = "pdflatex") -> str:
    """Locate the LaTeX compiler executable on the system PATH.

    Parameters
    ----------
    compiler : str
        Compiler name: ``"pdflatex"`` or ``"xelatex"``.

    Returns
    -------
    str
        Absolute path to the compiler.

    Raises
    ------
    FileNotFoundError
        If the compiler is not found, with installation instructions.
    """
    path = shutil.which(compiler)
    if path is None:
        raise FileNotFoundError(
            f"'{compiler}' not found on PATH.\n"
            f"To generate PDFs you need a LaTeX distribution:\n"
            f"  - MiKTeX (recommended for Windows): https://miktex.org/download\n"
            f"  - TeX Live: https://tug.org/texlive/\n"
            f"After installing, ensure '{compiler}' is available on the system PATH."
        )
    return path


def compile_pdf(
    tex_path: str,
    output_dir: str = None,
    compiler: str = "pdflatex",
    runs: int = 2,
) -> str:
    """Compile a .tex file to PDF.

    Parameters
    ----------
    tex_path : str
        Path to the .tex source file.
    output_dir : str, optional
        Output directory.  Defaults to the .tex file's directory.
    compiler : str
        ``"pdflatex"`` or ``"xelatex"``.
    runs : int
        Number of compilation passes (2 for cross-references / TOC).

    Returns
    -------
    str
        Absolute path to the produced PDF file.

    Raises
    ------
    FileNotFoundError
        If the compiler is not installed.
    RuntimeError
        If compilation fails (includes a log excerpt).
    """
    compiler_path = find_latex_compiler(compiler)
    tex_path = Path(tex_path).resolve()

    if output_dir is None:
        output_dir = str(tex_path.parent)
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i in range(runs):
        result = subprocess.run(
            [
                compiler_path,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={output_dir}",
                str(tex_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Only raise on the last pass — earlier passes may have warnings
        if result.returncode != 0 and i == runs - 1:
            excerpt = _extract_error(result.stdout)
            raise RuntimeError(
                f"LaTeX compilation failed (exit code {result.returncode}).\n"
                f"Compiler: {compiler}\n"
                f"Source: {tex_path}\n"
                f"Error excerpt:\n{excerpt}"
            )

    pdf_path = Path(output_dir) / tex_path.with_suffix(".pdf").name
    if not pdf_path.exists():
        raise RuntimeError(
            f"LaTeX compilation completed but PDF not found at {pdf_path}.\n"
            f"Check the .log file for details."
        )
    return str(pdf_path.resolve())


def _extract_error(log_text: str, context_lines: int = 10) -> str:
    """Extract the first error message from a pdflatex log."""
    lines = log_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("!"):
            start = max(0, i - 2)
            end = min(len(lines), i + context_lines)
            return "\n".join(lines[start:end])
    # Fallback: last N lines
    return "\n".join(lines[-context_lines:])
