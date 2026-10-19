"""
Results container for bearing capacity analysis.

Stores all computed factors and provides summary output and optional plotting.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

#: What the allowable loads on a footing are (GM-170): the structural load
#: on the net allowable pressure, and the gross load on the bearing surface,
#: which is not the structural load.
LOAD_BASIS = (
    "structural load Q_DL+LL = q_all,net x A' = (q_ult/FS - q) x A', "
    "A' = B'L' the effective area, the foundation and the soil over it "
    "taken at the soil's unit weight (UFC 3-220-20 Sec. 5-2.2 Eq. 5-3, "
    "p. 276; Sec. 5-2.4 Eq. 5-14, p. 279). gross load = q_all x A' = "
    "Q_DL+LL + W_F + W_S, the total load on the bearing surface with the "
    "foundation's and the overlying soil's weight in it (Eq. 5-1, p. 275), "
    "so it is not the structural load. GEC-6 Sec. 5.2.2 (p. 53) recommends "
    "the allowable bearing capacity on the gross pressure, the self-weight "
    "of the footing and the backfill over it counted in the applied load; "
    "a footing heavier than the soil it replaces leaves less for the "
    "column (GEC-6 Sec. 6.3.1, p. 113)")


@dataclass
class BearingCapacityResult:
    """Results from a bearing capacity analysis.

    Attributes
    ----------
    q_ultimate : float
        Ultimate bearing capacity qult (kPa).
    q_allowable : float
        Allowable bearing capacity qult/FS (kPa).
    q_net : float
        Net ultimate bearing capacity qult - q (kPa).
    factor_of_safety : float
        Factor of safety applied.

    Nc, Nq, Ngamma : float
        Bearing capacity factors.
    sc, sq, sgamma : float
        Shape factors.
    dc, dq, dgamma : float
        Depth factors.
    ic, iq, igamma : float
        Load inclination factors.
    bc, bq, bgamma : float
        Base inclination factors.
    gc, gq, ggamma : float
        Ground inclination factors.

    q_overburden : float
        Effective overburden pressure at footing base (kPa).
    gamma_eff : float
        Effective unit weight below footing (kN/m³).
    B_eff : float
        Effective footing width used in computation (m).
    L_eff : float
        Effective footing length used in computation (m).

    term_cohesion : float
        Cohesion term contribution to qult (kPa).
    term_overburden : float
        Overburden term contribution to qult (kPa).
    term_selfweight : float
        Self-weight term contribution to qult (kPa).

    ngamma_method : str
        Method used for Ngamma.
    factor_method : str
        Method used for correction factors.

    is_two_layer : bool
        True if this was a two-layer analysis.
    q_upper_layer : float, optional
        qult of upper layer (two-layer analysis only).
    q_lower_layer : float, optional
        qult of lower layer (two-layer analysis only).

    failure_mode : str
        "general", or "local" (Terzaghi's local-shear reduction).
    Ngamma_alternatives : dict
        Terzaghi only: Ngamma from the other cited source(s), by
        ``ngamma_method`` name, beside the one used.
    c_used, phi_used : float, optional
        Terzaghi only: the strengths the factors were computed with (the
        reduced c*, phi* for local shear).
    notes : list of str
        Method basis and applicability notes (Terzaghi results).
    area_m2 : float, optional
        The effective bearing area A' = B'L' (m^2), or for a strip the
        effective width B' (m^2 per metre run), that the allowable loads
        are worked on; set by ``BearingCapacityAnalysis.compute`` (GM-170).
    is_strip : bool
        A strip footing: the allowable loads are per metre run.
    """
    # Primary results
    q_ultimate: float = 0.0
    q_allowable: float = 0.0
    q_net: float = 0.0
    factor_of_safety: float = 3.0

    # Bearing capacity factors
    Nc: float = 0.0
    Nq: float = 0.0
    Ngamma: float = 0.0

    # Shape factors
    sc: float = 1.0
    sq: float = 1.0
    sgamma: float = 1.0

    # Depth factors
    dc: float = 1.0
    dq: float = 1.0
    dgamma: float = 1.0

    # Inclination factors
    ic: float = 1.0
    iq: float = 1.0
    igamma: float = 1.0

    # Base inclination factors
    bc: float = 1.0
    bq: float = 1.0
    bgamma: float = 1.0

    # Ground inclination factors
    gc: float = 1.0
    gq: float = 1.0
    ggamma: float = 1.0

    # Intermediate values
    q_overburden: float = 0.0
    gamma_eff: float = 0.0
    B_eff: float = 0.0
    L_eff: float = 0.0

    # Term breakdown
    term_cohesion: float = 0.0
    term_overburden: float = 0.0
    term_selfweight: float = 0.0

    # Method info
    ngamma_method: str = "vesic"
    factor_method: str = "vesic"

    # Two-layer info
    is_two_layer: bool = False
    q_upper_layer: Optional[float] = None
    q_lower_layer: Optional[float] = None

    # Terzaghi basis
    failure_mode: str = "general"
    Ngamma_alternatives: Dict[str, float] = field(default_factory=dict)
    c_used: Optional[float] = None
    phi_used: Optional[float] = None
    notes: List[str] = field(default_factory=list)

    # The area the allowable loads are worked on (GM-170)
    area_m2: Optional[float] = None
    is_strip: bool = False

    @property
    def q_allowable_net(self) -> float:
        """Net allowable bearing pressure q_all,net = q_ult/FS - q (kPa), q
        the overburden at the base the bearing equation used (UFC 3-220-20
        Sec. 5-2.4 Eq. 5-14; GM-170)."""
        return self.q_allowable - self.q_overburden

    def allowable_loads(self) -> Optional[Dict[str, Any]]:
        """The allowable loads on the footing (GM-170): the structural load
        on the net allowable pressure and the gross load on the bearing
        surface, with :data:`LOAD_BASIS`; per metre run for a strip. None
        when the area is not known."""
        if self.area_m2 is None:
            return None
        unit = "kN_per_m" if self.is_strip else "kN"
        return {
            f"structural_load_{unit}": round(
                self.q_allowable_net * self.area_m2, 1),
            f"gross_load_{unit}": round(self.q_allowable * self.area_m2, 1),
            ("width_m" if self.is_strip else "area_m2"): round(
                self.area_m2, 4),
            "basis": LOAD_BASIS,
        }

    def summary(self) -> str:
        """Return a formatted summary string.

        Returns
        -------
        str
            Human-readable summary of the bearing capacity results.
        """
        lines = [
            "=" * 60,
            "  BEARING CAPACITY ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"  Ultimate bearing capacity  qult = {self.q_ultimate:,.1f} kPa",
            f"  Factor of safety           FS   = {self.factor_of_safety:.1f}",
            f"  Allowable bearing capacity qall = {self.q_allowable:,.1f} kPa",
            f"  Net ultimate capacity      qnet = {self.q_net:,.1f} kPa",
            f"  Net allowable              qall,net = "
            f"{self.q_allowable_net:,.1f} kPa (UFC 3-220-20 Eq. 5-14)",
        ]
        loads = self.allowable_loads()
        if loads:
            per = " per m" if self.is_strip else ""
            unit = "kN_per_m" if self.is_strip else "kN"
            lines += [
                f"  Allowable structural load Q = "
                f"{loads['structural_load_' + unit]:,.0f} kN{per} "
                f"(qall,net x A'; Eq. 5-3)",
                f"  Gross load on the base   qall x A' = "
                f"{loads['gross_load_' + unit]:,.0f} kN{per} "
                f"(with the footing and the soil over it; Eq. 5-1)",
            ]
        lines += [
            "",
            f"  Effective footing: B' = {self.B_eff:.3f} m, L' = {self.L_eff:.3f} m",
            f"  Overburden pressure: q = {self.q_overburden:.1f} kPa",
            f"  Effective unit weight: gamma = {self.gamma_eff:.2f} kN/m³",
            "",
            "  Bearing Capacity Factors:",
            f"    Nc = {self.Nc:.2f}   Nq = {self.Nq:.2f}   Ng = {self.Ngamma:.2f}"
            f"  ({self.ngamma_method})",
            "",
        ]
        if self.factor_method == "terzaghi":
            lines.extend([
                f"  Terzaghi (1943), {self.failure_mode} shear:",
                f"    Shape:       sc={self.sc:.3f}  sq={self.sq:.3f}  sg={self.sgamma:.3f}",
                "    No depth, inclination, base or ground factors.",
                "",
            ])
        else:
            lines.extend([
                "  Correction Factors ({0}):".format(self.factor_method),
                f"    Shape:       sc={self.sc:.3f}  sq={self.sq:.3f}  sg={self.sgamma:.3f}",
                f"    Depth:       dc={self.dc:.3f}  dq={self.dq:.3f}  dg={self.dgamma:.3f}",
                f"    Inclination: ic={self.ic:.3f}  iq={self.iq:.3f}  ig={self.igamma:.3f}",
                f"    Base tilt:   bc={self.bc:.3f}  bq={self.bq:.3f}  bg={self.bgamma:.3f}",
                f"    Ground:      gc={self.gc:.3f}  gq={self.gq:.3f}  gg={self.ggamma:.3f}",
                "",
            ])
        lines.extend([
            "  Term Breakdown:",
            f"    Cohesion term:    {self.term_cohesion:,.1f} kPa"
            f"  ({self._pct(self.term_cohesion)}%)",
            f"    Overburden term:  {self.term_overburden:,.1f} kPa"
            f"  ({self._pct(self.term_overburden)}%)",
            f"    Self-weight term: {self.term_selfweight:,.1f} kPa"
            f"  ({self._pct(self.term_selfweight)}%)",
        ])

        if self.is_two_layer:
            lines.extend([
                "",
                "  Two-Layer Analysis:",
                f"    Upper layer qult: {self.q_upper_layer:,.1f} kPa",
                f"    Lower layer qult: {self.q_lower_layer:,.1f} kPa",
            ])

        if self.notes:
            lines.append("")
            lines.extend(f"  Note: {n}" for n in self.notes)

        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def _pct(self, term: float) -> str:
        if self.q_ultimate > 0:
            return f"{100 * term / self.q_ultimate:.0f}"
        return "0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a dictionary for LLM agent consumption.

        Returns
        -------
        dict
            All result fields as a flat dictionary.
        """
        d = {
            "q_ultimate_kPa": round(self.q_ultimate, 2),
            "q_allowable_kPa": round(self.q_allowable, 2),
            "q_net_kPa": round(self.q_net, 2),
            "q_allowable_net_kPa": round(self.q_allowable_net, 2),
            "factor_of_safety": self.factor_of_safety,
            "Nc": round(self.Nc, 3),
            "Nq": round(self.Nq, 3),
            "Ngamma": round(self.Ngamma, 3),
            "sc": round(self.sc, 4),
            "sq": round(self.sq, 4),
            "sgamma": round(self.sgamma, 4),
            "dc": round(self.dc, 4),
            "dq": round(self.dq, 4),
            "dgamma": round(self.dgamma, 4),
            "ic": round(self.ic, 4),
            "iq": round(self.iq, 4),
            "igamma": round(self.igamma, 4),
            "bc": round(self.bc, 4),
            "bq": round(self.bq, 4),
            "bgamma": round(self.bgamma, 4),
            "gc": round(self.gc, 4),
            "gq": round(self.gq, 4),
            "ggamma": round(self.ggamma, 4),
            "q_overburden_kPa": round(self.q_overburden, 2),
            "gamma_eff_kNm3": round(self.gamma_eff, 3),
            "B_eff_m": round(self.B_eff, 4),
            "L_eff_m": round(self.L_eff, 4),
            "term_cohesion_kPa": round(self.term_cohesion, 2),
            "term_overburden_kPa": round(self.term_overburden, 2),
            "term_selfweight_kPa": round(self.term_selfweight, 2),
            "ngamma_method": self.ngamma_method,
            "factor_method": self.factor_method,
            "is_two_layer": self.is_two_layer,
        }
        if self.factor_method == "terzaghi":
            d["method"] = "Terzaghi (1943)"
            d["failure_mode"] = self.failure_mode
            d["Ngamma_alternatives"] = {
                k: round(v, 3) for k, v in self.Ngamma_alternatives.items()}
            if self.failure_mode == "local":
                d["c_used_kPa"] = round(self.c_used, 3)
                d["phi_used_deg"] = round(self.phi_used, 3)
        if self.notes:
            d["notes"] = list(self.notes)
        loads = self.allowable_loads()
        if loads:
            d["allowable_load"] = loads
        return d

    def get_tables(self) -> list:
        """The bearing capacity factor and term breakdown: for the cohesion,
        overburden and self-weight terms, the factor N and the shape, depth,
        load-inclination, base-tilt and ground-slope factors, and the term's
        contribution to q_ult; and the capacity summary (CSV only; the calc
        package prints it). Values are this result's own, in its units.

        Returns
        -------
        list of geotech_common.tables.CalcTable
        """
        from geotech_common.tables import CalcTable

        rows = [
            ["Cohesion (c)", self.Nc, self.sc, self.dc, self.ic, self.bc,
             self.gc, self.term_cohesion],
            ["Overburden (q)", self.Nq, self.sq, self.dq, self.iq, self.bq,
             self.gq, self.term_overburden],
            ["Self-weight (γ)", self.Ngamma, self.sgamma, self.dgamma,
             self.igamma, self.bgamma, self.ggamma, self.term_selfweight],
        ]
        method = ("Terzaghi (1943)" if self.factor_method == "terzaghi"
                  else f"{self.factor_method} factors")
        note = (f"q_ult = c Nc sc dc ic bc gc + q Nq sq dq iq bq gq + "
                f"0.5 γ' B' Nγ sγ dγ iγ bγ "
                f"gγ ({method}, Nγ by {self.ngamma_method}); "
                f"q = {self.q_overburden:.2f} kPa, γ' = "
                f"{self.gamma_eff:.2f} kN/m³, B' = {self.B_eff:.3f} m")
        if self.is_two_layer:
            note += ("; two-layer case: the terms are those of the upper "
                     "layer, q_ult is the load-spread combination")
        tables = [CalcTable(
            title="Bearing capacity factors and terms",
            columns=["Term", ("Bearing capacity factor N", "", ".3f"),
                     ("Shape s", "", ".4f"), ("Depth d", "", ".4f"),
                     ("Load inclination i", "", ".4f"),
                     ("Base tilt b", "", ".4f"), ("Ground slope g", "", ".4f"),
                     ("Term value", "kPa", ".1f")],
            rows=rows, note=note, name="factors_and_terms",
        )]
        summary = [
            ["Ultimate bearing capacity", "q_ult", self.q_ultimate, "kPa"],
            ["Net ultimate bearing capacity", "q_net", self.q_net, "kPa"],
            ["Factor of safety", "FS", self.factor_of_safety, ""],
            ["Allowable bearing capacity", "q_all", self.q_allowable, "kPa"],
            ["Net allowable bearing pressure", "q_all,net",
             self.q_allowable_net, "kPa"],
            ["Effective overburden at the base", "q", self.q_overburden,
             "kPa"],
            ["Effective unit weight below the base", "γ'",
             self.gamma_eff, "kN/m³"],
            ["Effective width", "B'", self.B_eff, "m"],
            ["Effective length", "L'", self.L_eff, "m"],
        ]
        loads = self.allowable_loads()
        if loads:
            unit = "kN_per_m" if self.is_strip else "kN"
            shown = "kN/m" if self.is_strip else "kN"
            summary += [
                ["Allowable structural load", "Q_DL+LL",
                 loads["structural_load_" + unit], shown],
                ["Gross load on the base (with the footing and soil)",
                 "q_all A'", loads["gross_load_" + unit], shown],
            ]
        if self.is_two_layer:
            summary += [
                ["q_ult of the upper layer alone", "q_t",
                 self.q_upper_layer, "kPa"],
                ["q_ult of the lower layer alone", "q_b",
                 self.q_lower_layer, "kPa"],
            ]
        tables.append(CalcTable(
            title="Bearing capacity summary",
            columns=["Quantity", "Symbol", "Value", "Unit"], rows=summary,
            note=f"Failure mode: {self.failure_mode}",
            name="capacity_summary", printed=False,
        ))
        return tables
