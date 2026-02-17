"""
Unicode-to-LaTeX equation converter.

Transforms the Unicode equation strings used by calc_steps.py modules
into valid LaTeX math markup.  Also provides escape_latex() for non-math
text (titles, descriptions, notes).

Design principle: rule-based conversion applied in a fixed order so that
every Unicode pattern cataloged across the 13 analysis modules renders
correctly in LaTeX math mode.
"""

import re

# ── Unicode → LaTeX character map ──────────────────────────────────────
# Applied first, before structural transforms.

_CHAR_MAP = {
    # Greek lowercase
    "\u03b1": r"\alpha ",
    "\u03b2": r"\beta ",
    "\u03b3": r"\gamma ",
    "\u03b4": r"\delta ",
    "\u03b5": r"\varepsilon ",
    "\u03b7": r"\eta ",
    "\u03b8": r"\theta ",
    "\u03bb": r"\lambda ",
    "\u03bc": r"\mu ",
    "\u03bd": r"\nu ",
    "\u03c0": r"\pi ",
    "\u03c1": r"\rho ",
    "\u03c3": r"\sigma ",
    "\u03c4": r"\tau ",
    "\u03c6": r"\varphi ",
    "\u03c9": r"\omega ",
    # Greek uppercase
    "\u0394": r"\Delta ",
    "\u03a3": r"\Sigma ",
    # Operators
    "\u00d7": r"\times ",
    "\u00b7": r"\cdot ",
    "\u00b1": r"\pm ",
    # Relations
    "\u2264": r"\leq ",
    "\u2265": r"\geq ",
    "\u2248": r"\approx ",
    "\u2260": r"\neq ",
    # Arrows
    "\u2192": r"\rightarrow ",
    # Superscripts
    "\u00b2": "^{2}",
    "\u00b3": "^{3}",
    "\u2074": "^{4}",
    "\u2070": "^{0}",
    "\u00b9": "^{1}",
    # Subscripts (Unicode subscript digits)
    "\u2080": "_{0}",
    "\u2081": "_{1}",
    "\u2082": "_{2}",
    # Degree
    "\u00b0": r"^{\circ}",
    # Prime
    "\u2032": "'",
    # Dashes (kept as-is in math text)
    "\u2014": r"\text{---}",
    "\u2013": r"\text{--}",
    # Square root placeholder — handled structurally below
    "\u221a": r"\SQRT",
    # Summation symbol
    "\u2211": r"\sum ",
}


def unicode_to_latex(text: str) -> str:
    """Convert a Unicode equation string to LaTeX math content.

    The result is intended to be placed inside a math environment
    (``$...$`` or ``\\[...\\]``) by the template — this function
    produces only the inner math markup.

    Parameters
    ----------
    text : str
        Equation or substitution string from a CalcStep.

    Returns
    -------
    str
        LaTeX math string.
    """
    if not text:
        return ""

    s = text

    # 1. Subscripts FIRST (before Unicode substitution, while _ positions
    #    are unambiguous). This prevents double-subscript errors like
    #    m_α_i → m_\alpha _{i} (two subscripts on m).
    #    Instead: m_α_i → m_{α_i} → m_{\alpha _{i}} (nested, valid).
    #    Multi-char subscripts: _word → _{word}
    s = re.sub(r"_([A-Za-z0-9\u0370-\u03FF][A-Za-z0-9\u0370-\u03FF_,]+)(?![}])",
               r"_{\1}", s)
    # Single-char subscripts: _X → _{X} (including Greek chars)
    s = re.sub(r"_([A-Za-z0-9\u0370-\u03FF])(?![A-Za-z0-9\u0370-\u03FF_{])",
               r"_{\1}", s)

    # 2. Combine consecutive Unicode subscript digits into a single group
    #    e.g. ₁₀ → _{10} instead of _{1}_{0} (double subscript error)
    _SUBSCRIPT_DIGITS = {
        "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3",
        "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7",
        "\u2088": "8", "\u2089": "9",
    }
    _sub_pattern = "[" + "".join(_SUBSCRIPT_DIGITS.keys()) + "]"

    def _replace_sub_group(m):
        return "_{" + "".join(_SUBSCRIPT_DIGITS[ch] for ch in m.group(0)) + "}"

    s = re.sub(_sub_pattern + "{2,}", _replace_sub_group, s)
    # Single subscript digits (still handled by _CHAR_MAP below)

    # 3. Direct Unicode character substitution
    for char, latex in _CHAR_MAP.items():
        s = s.replace(char, latex)

    # 4. Square root: \SQRT(...) → \sqrt{...}
    #    Also handle \SQRT followed by a single token (no parens)
    s = re.sub(r"\\SQRT\(([^)]*)\)", r"\\sqrt{\1}", s)
    # \SQRT followed by a single alphanumeric token like \SQRTKa
    s = re.sub(r"\\SQRT([A-Za-z_][A-Za-z0-9_]*)", r"\\sqrt{\1}", s)

    # 5. Math function names: add backslash before recognized names
    #    Must be careful not to double-backslash already-converted names
    for fn in ("arctan", "arcsin", "tan", "sin", "cos", "exp", "log", "ln",
               "min", "max"):
        # Only add backslash if not already preceded by one
        s = re.sub(r"(?<!\\)\b(" + fn + r")\b", r"\\\1", s)

    # 6. Multi-line: \n → \\ for LaTeX line breaks
    s = s.replace("\n", r" \\ ")

    return s.strip()


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in non-math text.

    Use for titles, descriptions, notes, references — anything
    that will NOT be inside a math environment.

    Parameters
    ----------
    text : str
        Plain text string.

    Returns
    -------
    str
        Text with LaTeX special characters escaped.
    """
    if not text:
        return ""

    s = str(text)

    # Order matters: backslash first, then others
    s = s.replace("\\", r"\textbackslash ")
    s = s.replace("&", r"\&")
    s = s.replace("%", r"\%")
    s = s.replace("$", r"\$")
    s = s.replace("#", r"\#")
    s = s.replace("_", r"\_")
    s = s.replace("{", r"\{")
    s = s.replace("}", r"\}")
    s = s.replace("~", r"\textasciitilde ")
    s = s.replace("^", r"\textasciicircum ")

    # Also convert Unicode Greek etc. for display in text mode
    _TEXT_CHARS = {
        "\u03c6": r"$\varphi$",
        "\u03b3": r"$\gamma$",
        "\u03b1": r"$\alpha$",
        "\u03b2": r"$\beta$",
        "\u03b5": r"$\varepsilon$",
        "\u03b7": r"$\eta$",
        "\u03b8": r"$\theta$",
        "\u03bb": r"$\lambda$",
        "\u03bc": r"$\mu$",
        "\u03bd": r"$\nu$",
        "\u03c3": r"$\sigma$",
        "\u03b4": r"$\delta$",
        "\u03c4": r"$\tau$",
        "\u03c9": r"$\omega$",
        "\u03c1": r"$\rho$",
        "\u0394": r"$\Delta$",
        "\u03a3": r"$\Sigma$",
        "\u03c0": r"$\pi$",
        "\u00b0": r"$^\circ$",
        "\u00b2": r"$^2$",
        "\u00b3": r"$^3$",
        "\u2074": r"$^4$",
        "\u2032": r"$'$",
        "\u00d7": r"$\times$",
        "\u2264": r"$\leq$",
        "\u2265": r"$\geq$",
        "\u221a": r"$\sqrt{}$",
    }
    for char, repl in _TEXT_CHARS.items():
        s = s.replace(char, repl)

    return s
