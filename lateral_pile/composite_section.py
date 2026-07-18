"""
Composite / transformed-section flexural and axial rigidity.

Computes the **uncracked (gross) transformed-section** flexural rigidity EI and
axial rigidity EA of a composite pile cross-section built from more than one
material — the stiffness a laterally loaded pile analysis actually needs when
the section is not a single homogeneous material.

Three common cases are covered:

  (a) ``filled_pipe``          — concrete/grout-filled steel pipe (micropile
                                 casing + grout, CFT pile).
  (b) ``cased_concrete``       — steel casing + concrete/grout core + an
                                 optional circular reinforcing-bar ring.
  (c) ``reinforced_concrete``  — circular OR rectangular reinforced concrete
                                 with a bar ring (circular) or bar layers
                                 (rectangular). No outer steel shell.

Method (transformed section, sum-of-rigidities form)
----------------------------------------------------
For a section made of non-overlapping material regions ``i`` with Young's
modulus ``E_i``, area ``A_i`` and self-centroidal inertia ``I0_i`` located at
signed distance ``y_i`` from the gross geometric centre::

    EA   = Σ E_i A_i
    y_na = Σ E_i A_i y_i / EA                 (transformed neutral axis)
    EI   = Σ E_i [ I0_i + A_i (y_i - y_na)^2 ]

Steel that is *embedded in* concrete (reinforcing bars sitting inside a grout
or concrete region that is itself counted over its full gross extent) is added
with the *net* modulus ``(E_bar - E_concrete)`` so the concrete it displaces is
not double-counted — the standard uncracked transformed-section treatment
(equivalent to the ``(n-1) A_s`` rule, ``n = E_bar / E_concrete``). A steel
pipe/casing and the concrete core it confines do NOT overlap, so those are
summed directly at their own moduli.

Concrete modulus
----------------
If ``E_concrete`` is not given it is taken from the compressive strength via the
ACI 318 normalweight correlation ``Ec = 4700 sqrt(f'c)`` (MPa, ``f'c`` in MPa;
converted to kPa internally) — the same correlation used elsewhere in this
module (``ReinforcedConcreteSection.Ec``, ``Pile.from_filled_pipe``).

Source basis (VERIFIED-IN-HAND, 2026-07-18 wiki-verification): ACI 318-08
§8.5.1 (owner-library PDF, p. 111) prints the normalweight simplification
``Ec = 57,000 sqrt(f'c)`` psi — exactly the SI ``Ec = 4700 sqrt(f'c)`` MPa
coded here (57,000 psi-form = 4733 -> 4700 in the 318M metric rendering).
The equivalent modern clause is ACI 318-19 §19.2.2.1(b) (that edition itself
is not in the library; the correlation is edition-stable). Ledger:
module_work/wiki_verification/small_checks_aci_astm_das.md.

Scope / limitations
-------------------
- **Uncracked / gross basis only.** The returned EI is the full transformed
  section with the concrete taken in both tension and compression. It is the
  low-moment (pre-cracking) stiffness and an *upper bound* on the working EI of
  a reinforced section. A cracked-section or full moment-curvature (``M-φ``,
  Branson / fibre) EI is genuinely softer and is **out of scope here**
  (owner decision pending). For a moment-dependent cracked EI of a *circular
  RC* section see ``ReinforcedConcreteSection`` / ``Pile.from_rc_section``.
- Bending is about a centroidal axis; for circular sections the reinforcing
  ring is treated as polar-symmetric (``I_ring = A_s R^2 / 2``, orientation-
  independent, textbook for a bar ring), so EI is the same for any lateral
  loading direction.

All units are SI: metres (m), kilopascals (kPa); EI in kN·m², EA in kN.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ACI 318 normalweight concrete modulus correlation, returned in kPa.
_ACI_EC_COEFF = 4700.0
_DEFAULT_E_STEEL = 200.0e6  # kPa (structural / reinforcing steel)


def aci_concrete_modulus(fc: float) -> float:
    """Concrete elastic modulus from f'c via ACI 318 (normalweight).

    Ec = 4700 * sqrt(f'c[MPa]) MPa.

    Parameters
    ----------
    fc : float
        Concrete compressive strength f'c (kPa).

    Returns
    -------
    float
        Ec (kPa).
    """
    if fc <= 0:
        raise ValueError(f"f'c must be positive, got {fc}")
    fc_MPa = fc / 1000.0
    return _ACI_EC_COEFF * math.sqrt(fc_MPa) * 1000.0  # kPa


@dataclass
class _Part:
    """One material region: modulus, area, self inertia, centroid offset."""
    name: str
    E: float          # kPa
    A: float          # m^2  (may be negative for a subtractive/void region)
    I0: float         # m^4  self-centroidal inertia
    y: float = 0.0    # m    centroid offset from the gross geometric centre


@dataclass
class CompositeSection:
    """Uncracked transformed-section properties of a composite pile section.

    Produced by :func:`composite_section_ei`. The physical rigidities ``EI`` and
    ``EA`` are modulus-invariant; ``area_transformed`` / ``inertia_transformed``
    are those rigidities expressed at the reference modulus ``E_ref`` (so
    ``E_ref * inertia_transformed == EI``), which makes the section a drop-in
    ``Pile`` (``E = E_ref``, ``moment_of_inertia = inertia_transformed``).

    Attributes
    ----------
    section_type : str
        'filled_pipe' | 'cased_concrete' | 'reinforced_concrete'.
    EI : float
        Composite flexural rigidity (kN·m²).
    EA : float
        Composite axial rigidity (kN).
    E_ref : float
        Reference modulus for the transformed properties (kPa). The stiffest
        material present (steel if any, else concrete).
    inertia_transformed : float
        Transformed moment of inertia I_t = EI / E_ref (m⁴) — the equivalent
        homogeneous inertia at E_ref.
    area_transformed : float
        Transformed area A_t = EA / E_ref (m²).
    inertia_gross : float
        Gross geometric moment of inertia of the outer section (m⁴), ignoring
        material differences (for reference / cracking-moment estimates).
    area_gross : float
        Gross geometric area of the outer section (m²).
    neutral_axis : float
        Transformed neutral-axis offset from the gross geometric centre (m);
        0.0 for symmetric sections.
    components : list of dict
        Per-material contributions: name, E, A, I (about the section NA),
        EI, EA.
    """
    section_type: str
    EI: float
    EA: float
    E_ref: float
    inertia_transformed: float
    area_transformed: float
    inertia_gross: float
    area_gross: float
    neutral_axis: float = 0.0
    components: List[dict] = field(default_factory=list)

    def equivalent_I(self, E: float) -> float:
        """Equivalent moment of inertia at a chosen modulus E (m⁴): EI / E."""
        if E <= 0:
            raise ValueError(f"E must be positive, got {E}")
        return self.EI / E

    def summary(self) -> str:
        """Return a text summary of the composite section."""
        lines = [
            "Composite Transformed-Section Properties",
            "=" * 44,
            f"Section type:         {self.section_type}",
            f"Composite EI:         {self.EI:.1f} kN-m^2",
            f"Composite EA:         {self.EA:.4g} kN",
            f"Reference modulus:    {self.E_ref:.4g} kPa",
            f"Transformed I:        {self.inertia_transformed:.6g} m^4"
            f"  (equivalent I at E_ref)",
            f"Transformed area:     {self.area_transformed:.6g} m^2",
            f"Gross geometric I:    {self.inertia_gross:.6g} m^4",
            f"Gross geometric area: {self.area_gross:.6g} m^2",
        ]
        if abs(self.neutral_axis) > 1e-9:
            lines.append(f"Neutral-axis offset:  {self.neutral_axis:.5f} m")
        lines.append("")
        lines.append("Component contributions (I about section NA):")
        lines.append(
            f"  {'material':<14}{'E [kPa]':>13}{'A [m^2]':>12}"
            f"{'EI [kN-m^2]':>14}{'% of EI':>9}")
        for c in self.components:
            pct = 100.0 * c["EI"] / self.EI if self.EI else 0.0
            lines.append(
                f"  {c['name']:<14}{c['E']:>13.4g}{c['A']:>12.5g}"
                f"{c['EI']:>14.1f}{pct:>8.1f}%")
        lines.append("")
        lines.append("Basis: UNCRACKED (gross) transformed section — upper "
                     "bound on working EI;")
        lines.append("cracked / moment-curvature EI is out of scope.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export as a plain dict (JSON-serializable)."""
        return {
            "section_type": self.section_type,
            "EI_kNm2": self.EI,
            "EA_kN": self.EA,
            "E_ref_kPa": self.E_ref,
            "inertia_transformed_m4": self.inertia_transformed,
            "area_transformed_m2": self.area_transformed,
            "inertia_gross_m4": self.inertia_gross,
            "area_gross_m2": self.area_gross,
            "neutral_axis_m": self.neutral_axis,
            "components": self.components,
            "basis": "uncracked_transformed_section",
        }


# ── geometry primitives ────────────────────────────────────────────────
def _circle(radius: float) -> Tuple[float, float]:
    """(area, centroidal inertia) of a solid circle."""
    A = math.pi * radius ** 2
    I = math.pi / 4.0 * radius ** 4
    return A, I


def _annulus(r_outer: float, r_inner: float) -> Tuple[float, float]:
    """(area, centroidal inertia) of a hollow circle (pipe wall)."""
    A = math.pi * (r_outer ** 2 - r_inner ** 2)
    I = math.pi / 4.0 * (r_outer ** 4 - r_inner ** 4)
    return A, I


def _rect(width: float, height: float) -> Tuple[float, float]:
    """(area, centroidal inertia) of a rectangle bending about the axis
    parallel to `width` (i.e. I = b h^3 / 12)."""
    A = width * height
    I = width * height ** 3 / 12.0
    return A, I


def _resolve_ec(E_concrete: Optional[float], fc: Optional[float]) -> float:
    if E_concrete is not None:
        if E_concrete <= 0:
            raise ValueError(f"E_concrete must be positive, got {E_concrete}")
        return float(E_concrete)
    if fc is not None:
        return aci_concrete_modulus(fc)
    raise ValueError("Provide either E_concrete (kPa) or fc (kPa) for the "
                     "concrete/grout modulus.")


def _assemble(parts: List[_Part], section_type: str,
              area_gross: float, inertia_gross: float,
              E_ref: float) -> CompositeSection:
    """Combine material parts into transformed-section rigidities.

    EA = Σ E A ; y_na = Σ E A y / EA ; EI = Σ E (I0 + A (y - y_na)^2).
    """
    EA = sum(p.E * p.A for p in parts)
    if EA <= 0:
        raise ValueError("Non-positive axial rigidity — check geometry/moduli.")
    y_na = sum(p.E * p.A * p.y for p in parts) / EA

    EI = 0.0
    components = []
    for p in parts:
        d = p.y - y_na
        EI_p = p.E * (p.I0 + p.A * d ** 2)
        EI += EI_p
        components.append({
            "name": p.name,
            "E": p.E,
            "A": p.A,
            "I": p.I0 + p.A * d ** 2,
            "EI": EI_p,
            "EA": p.E * p.A,
        })
    if EI <= 0:
        raise ValueError("Non-positive EI — check geometry/moduli.")

    return CompositeSection(
        section_type=section_type,
        EI=EI, EA=EA, E_ref=E_ref,
        inertia_transformed=EI / E_ref,
        area_transformed=EA / E_ref,
        inertia_gross=inertia_gross,
        area_gross=area_gross,
        neutral_axis=y_na,
        components=components,
    )


def _bar_ring_parts(n_bars: int, bar_diameter: float,
                    ring_diameter: float, E_bar: float,
                    E_concrete: float, host: str) -> List[_Part]:
    """Reinforcing-bar ring as a single embedded-steel part.

    A polar-symmetric ring of `n_bars` at pitch-circle radius R has a
    diameter-axis inertia I_ring = A_s R^2 / 2 (orientation independent). The
    bars are embedded in concrete counted over its full gross area, so they are
    added with the NET modulus (E_bar - E_concrete) — the (n-1) transformed
    rule that avoids double-counting the displaced concrete.
    """
    if n_bars <= 0 or bar_diameter <= 0:
        return []
    if ring_diameter is None or ring_diameter <= 0:
        raise ValueError("A bar ring needs a positive pitch-circle diameter "
                         "(bar_circle_diameter).")
    R = ring_diameter / 2.0
    A_bar = math.pi / 4.0 * bar_diameter ** 2
    A_s = n_bars * A_bar
    I0_bars = n_bars * (math.pi / 4.0) * (bar_diameter / 2.0) ** 4  # own inertia
    I_ring = A_s * R ** 2 / 2.0 + I0_bars
    return [_Part(name=f"rebar({host})", E=E_bar - E_concrete,
                  A=A_s, I0=I_ring, y=0.0)]


def composite_section_ei(section_type: str, *,
                         # concrete / grout modulus (either one)
                         E_concrete: Optional[float] = None,
                         fc: Optional[float] = None,
                         # steel pipe / casing (filled_pipe, cased_concrete)
                         outer_diameter: Optional[float] = None,
                         wall_thickness: Optional[float] = None,
                         E_steel: float = _DEFAULT_E_STEEL,
                         # reinforced_concrete gross section
                         diameter: Optional[float] = None,
                         width: Optional[float] = None,
                         height: Optional[float] = None,
                         # reinforcement (cased_concrete + reinforced_concrete)
                         n_bars: int = 0,
                         bar_diameter: float = 0.0,
                         bar_circle_diameter: Optional[float] = None,
                         bar_layers: Optional[List[Tuple[int, float]]] = None,
                         E_bar: float = _DEFAULT_E_STEEL) -> CompositeSection:
    """Uncracked transformed-section EI/EA of a composite pile cross-section.

    Parameters
    ----------
    section_type : {'filled_pipe', 'cased_concrete', 'reinforced_concrete'}
        - ``filled_pipe``: steel pipe of `outer_diameter`/`wall_thickness`
          filled with concrete/grout. No reinforcement.
        - ``cased_concrete``: same steel casing + grout core + an OPTIONAL
          circular bar ring (`n_bars`, `bar_diameter`, `bar_circle_diameter`).
        - ``reinforced_concrete``: circular (`diameter`) or rectangular
          (`width`, `height`) concrete with a circular bar ring
          (`bar_circle_diameter`) or, for rectangular, `bar_layers`
          [(n_bars_i, y_i_from_centre), ...]. No outer steel shell.
    E_concrete, fc : float, optional
        Concrete/grout modulus (kPa) directly, OR compressive strength f'c
        (kPa) from which Ec = 4700 sqrt(f'c[MPa]) (ACI 318). One is required.
    outer_diameter, wall_thickness : float
        Steel pipe/casing OD and wall thickness (m) — pipe/cased cases.
    E_steel : float
        Steel casing modulus (kPa). Default 200e6.
    diameter : float
        Concrete diameter (m) — circular reinforced_concrete.
    width, height : float
        Rectangular concrete breadth and depth (m); bending about the axis
        parallel to `width` (I_gross = width*height^3/12).
    n_bars, bar_diameter, bar_circle_diameter :
        Circular bar ring: bar count, individual bar diameter (m), and pitch-
        circle diameter (m). Use lateral_pile.rebar_diameter() for a standard
        bar size.
    bar_layers : list of (int, float)
        Rectangular reinforcement: (number of bars, signed distance from the
        centroid, m) per layer. Requires `bar_diameter`.
    E_bar : float
        Reinforcing-steel modulus (kPa). Default 200e6.

    Returns
    -------
    CompositeSection

    Examples
    --------
    Grout-filled micropile casing (FHWA NHI-05-039 SP-2 section):

    >>> sec = composite_section_ei('filled_pipe', outer_diameter=0.1969,
    ...                            wall_thickness=0.0151, fc=27600.0,
    ...                            E_steel=199947980.0)
    >>> round(sec.EI)                                    # doctest: +SKIP
    8107
    """
    if section_type in ("filled_pipe", "cased_concrete"):
        if outer_diameter is None or wall_thickness is None:
            raise ValueError(
                f"section_type '{section_type}' requires outer_diameter and "
                "wall_thickness (m).")
        if wall_thickness <= 0 or outer_diameter <= 0:
            raise ValueError("outer_diameter and wall_thickness must be "
                             "positive.")
        r_o = outer_diameter / 2.0
        r_i = r_o - wall_thickness
        if r_i <= 0:
            raise ValueError(
                f"wall_thickness ({wall_thickness}) must be less than the "
                f"radius ({r_o}).")
        Ec = _resolve_ec(E_concrete, fc)

        A_steel, I_steel = _annulus(r_o, r_i)
        A_core, I_core = _circle(r_i)          # grout fills the full bore
        parts = [
            _Part("steel_casing", E_steel, A_steel, I_steel, 0.0),
            _Part("grout_core", Ec, A_core, I_core, 0.0),
        ]
        if section_type == "cased_concrete" and n_bars > 0:
            parts += _bar_ring_parts(n_bars, bar_diameter,
                                     bar_circle_diameter, E_bar, Ec, "core")
        # gross = outer steel envelope
        area_gross, inertia_gross = _circle(r_o)
        return _assemble(parts, section_type, area_gross, inertia_gross,
                         E_ref=E_steel)

    if section_type == "reinforced_concrete":
        Ec = _resolve_ec(E_concrete, fc)
        if diameter is not None:
            # circular RC with a bar ring
            r = diameter / 2.0
            if r <= 0:
                raise ValueError("diameter must be positive.")
            A_c, I_c = _circle(r)
            parts = [_Part("concrete", Ec, A_c, I_c, 0.0)]
            if n_bars > 0:
                parts += _bar_ring_parts(n_bars, bar_diameter,
                                         bar_circle_diameter, E_bar, Ec,
                                         "section")
            area_gross, inertia_gross = A_c, I_c
        elif width is not None and height is not None:
            if width <= 0 or height <= 0:
                raise ValueError("width and height must be positive.")
            A_c, I_c = _rect(width, height)
            parts = [_Part("concrete", Ec, A_c, I_c, 0.0)]
            if bar_layers:
                if bar_diameter <= 0:
                    raise ValueError("bar_layers requires a positive "
                                     "bar_diameter.")
                A_bar = math.pi / 4.0 * bar_diameter ** 2
                for k, (n_k, y_k) in enumerate(bar_layers):
                    if n_k <= 0:
                        continue
                    A_layer = n_k * A_bar
                    I0_layer = n_k * (math.pi / 4.0) * (bar_diameter / 2.0) ** 4
                    parts.append(_Part(f"rebar_layer{k+1}", E_bar - Ec,
                                       A_layer, I0_layer, y_k))
            area_gross, inertia_gross = A_c, I_c
        else:
            raise ValueError(
                "reinforced_concrete requires either `diameter` (circular) or "
                "`width` and `height` (rectangular).")
        E_ref = E_bar if (n_bars > 0 or bar_layers) else Ec
        return _assemble(parts, section_type, area_gross, inertia_gross,
                         E_ref=E_ref)

    raise ValueError(
        f"Unknown section_type '{section_type}'. Expected one of "
        "'filled_pipe', 'cased_concrete', 'reinforced_concrete'.")
