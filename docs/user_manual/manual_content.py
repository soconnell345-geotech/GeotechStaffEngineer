"""Hand-written narrative + document structure for the user manual.

`build_manual.py` imports this module and calls :func:`render` with a context
object that proxies the Doc builder + shared HTML helpers. The programmatic
method/parameter catalog is injected by the builder (introspected from the live
dispatch registry); everything here is the practitioner-facing prose.
"""
from __future__ import annotations

APPENDIX_START_CH = 10  # 9 numbered chapters, then Appendix A, B, ...


# ===========================================================================
# Cover + disclaimer (rendered before the TOC)
# ===========================================================================

def cover_html(version: str, st: dict) -> str:
    return f"""
<section class="cover">
  <div class="mark">Geotechnical Staff Engineer</div>
  <h1>GeotechStaffEngineer</h1>
  <div class="sub">User Manual for the practicing geotechnical engineer</div>
  <hr class="rule">
  <div class="tagline">
    A toolkit that turns the staff engineer&rsquo;s repertoire into composable,
    machine-callable methods, wraps them in a probabilistic variability engine,
    and drives them with an engine-agnostic reasoning agent &mdash; because in
    geotechnical engineering the true answer is a distribution, not a single number.
  </div>
  <div class="meta">
    <div class="ver">Version {version}</div>
    <div>{st['n_analysis']} analysis modules &middot; {st['n_reference']} digitized references &middot;
         {st['n_methods']} agent-callable methods</div>
    <div>FOSM / PEM / Monte&nbsp;Carlo / FORM reliability &middot; validated against published benchmarks</div>
  </div>
</section>
"""


def disclaimer_html() -> str:
    return """
<section class="page-break" style="padding-top:6mm;">
  <h1 style="font-family:'Segoe UI',Arial,sans-serif;font-size:19pt;color:#0e2233;border-bottom:2.5px solid #9a4a2a;padding-bottom:8px;">
    Professional-use disclaimer</h1>
  <div class="callout limit" style="margin-top:14px;">
    <div class="co-h">Read this first</div>
    <p><strong>GeotechStaffEngineer is an analysis and research aid &mdash; a multiplier
    for a qualified engineer&rsquo;s judgment, not a replacement for it, and not a design
    deliverable.</strong></p>
  </div>
  <p>Geotechnical engineering is the practice of building on and in the ground, where the
  material is the earth itself: heterogeneous, layered, partly saturated, and sampled at
  only a handful of points across a site. Because the ground is variable and only partly
  known, a single number is never the answer. This toolkit exists to help an engineer run
  the industry-standard methods repeatedly across plausible subsurface and loading
  conditions and understand the <em>range and spread</em> of answers. It does not know your
  site, and it cannot exercise engineering judgment.</p>

  <h4>What this software is &mdash; and is not</h4>
  <ul>
    <li><strong>It is</strong> a library of deterministic methods, a probabilistic
        variability engine, digitized references, and an LLM agent that drives them.</li>
    <li><strong>It is not</strong> a stamped design, a professional opinion, or a
        substitute for site-specific investigation, characterization, and review.</li>
    <li>Using this software <strong>creates no engineer-of-record relationship</strong> and
        no professional engagement with the authors or contributors.</li>
  </ul>

  <h4>Your responsibilities as a user</h4>
  <ul>
    <li><strong>A qualified, licensed professional engineer familiar with the site must
        independently review every input, assumption, method selection, and result before
        it is relied upon.</strong> Outputs are candidate calculations to be checked, not
        conclusions to be adopted.</li>
    <li>Confirm that the <strong>method applies to your problem.</strong> Every method has
        applicability limits, sign conventions, and edge cases &mdash; documented in each
        module&rsquo;s <code>DESIGN.md</code> (and, where present, <code>VALIDATION.md</code>).</li>
    <li>Confirm the <strong>inputs and units.</strong> All quantities are <strong>SI</strong>
        (meters, kPa, kN, kN/m, degrees). Mixed or mis-scaled units are the user&rsquo;s
        responsibility to catch.</li>
    <li>Treat <strong>LLM-agent output with particular care.</strong> The agent can select
        the wrong method, mis-transcribe a value, or read a chart imperfectly. Its answers
        are a starting point for review, never a final basis for design.</li>
  </ul>

  <h4>Validation scope</h4>
  <p>Selected modules are validated against published worked examples and benchmarks
  (Fredlund &amp; Krahn 1977, ACADS, Duncan, Griffiths &amp; Lane, the Prandtl solution, and
  GEC / Caltrans / FLAC examples). The specific problems checked and their verdicts are
  recorded in <code>validation_examples/RESULTS.md</code> and the per-module
  <code>VALIDATION.md</code> files. <strong>Validation covers only those documented cases.</strong>
  Agreement on a benchmark does not certify correctness for any other geometry, parameter
  range, or loading condition.</p>

  <h4>No warranty</h4>
  <p class="small">This software is provided under the MIT License <strong>&ldquo;AS IS&rdquo;,
  WITHOUT WARRANTY OF ANY KIND</strong>, express or implied. In no event shall the authors
  or copyright holders be liable for any claim, damages, or other liability arising from its
  use. Responsibility for the safety, adequacy, and code-compliance of any design remains
  entirely with the licensed professional engineer who adopts it.</p>
</section>
"""


# ===========================================================================
# Catalog structure
# ===========================================================================

# Ordered domain groups for the Problem Catalog. Each: (title, blurb, [modules]).
MODULE_ORDER = [
    ("Shallow foundations",
     "Sizing and checking spread footings and mats: how much load the ground carries, "
     "and how far it settles.",
     ["bearing_capacity", "settlement"]),
    ("Deep foundations",
     "Driven piles, drilled shafts, and pile groups &mdash; axial capacity, lateral response, "
     "drivability, group action, and downdrag.",
     ["axial_pile", "lateral_pile", "pile_group", "drilled_shaft", "wave_equation", "downdrag"]),
    ("Earth retention &amp; excavation support",
     "Walls that hold back soil and the systems that keep an excavation open and stable.",
     ["sheet_pile", "soe", "retaining_walls", "ground_improvement"]),
    ("Slope stability &amp; continuum FEM",
     "Will the slope stand? Limit-equilibrium factors of safety and a full 2D finite-element "
     "stress/deformation and strength-reduction engine.",
     ["slope_stability", "fem2d"]),
    ("Seismic &amp; liquefaction",
     "Site response, seismic earth pressures, liquefaction triggering, and ground-motion "
     "processing.",
     ["seismic_geotech", "liquefaction", "liquepy", "opensees", "pystrata",
      "seismic_signals", "hvsrpy"]),
    ("Site characterization &amp; geostatistics",
     "Turning borings, cones, and lab data into a modelled ground &mdash; parse, visualize, "
     "correlate, and interpolate.",
     ["subsurface", "gstools", "swprocess"]),
    ("Reliability &amp; sensitivity",
     "The variability engine: propagate parameter uncertainty to a reliability index and a "
     "probability of failure, and find which input governs.",
     ["reliability", "salib", "pystra"]),
    ("Geometry import &amp; reporting",
     "Get a real cross-section in from CAD or a PDF drawing, and get a calc package out.",
     ["dxf_import", "pdf_import", "dxf_export", "calc_package"]),
]

MODULE_TITLES = {
    "bearing_capacity": "Bearing capacity",
    "settlement": "Settlement",
    "axial_pile": "Axial pile capacity",
    "lateral_pile": "Lateral pile analysis",
    "pile_group": "Pile groups",
    "drilled_shaft": "Drilled shafts",
    "wave_equation": "Wave-equation drivability",
    "downdrag": "Downdrag / neutral plane",
    "sheet_pile": "Sheet-pile walls",
    "soe": "Support of excavation",
    "retaining_walls": "Retaining walls (cantilever &amp; MSE)",
    "ground_improvement": "Ground improvement",
    "slope_stability": "Slope stability",
    "fem2d": "2D finite-element analysis (fem2d)",
    "seismic_geotech": "Seismic geotechnics",
    "liquefaction": "Liquefaction (unified tool)",
    "liquepy": "Liquefaction building blocks (liquepy)",
    "opensees": "Nonlinear site response &amp; PM4Sand (OpenSees)",
    "pystrata": "Equivalent-linear site response (pyStrata)",
    "seismic_signals": "Ground-motion signal processing",
    "hvsrpy": "HVSR site characterization",
    "subsurface": "Subsurface data I/O (DIGGS / GEF / AGS4)",
    "gstools": "Geostatistics (kriging &amp; random fields)",
    "swprocess": "MASW surface-wave dispersion",
    "reliability": "Reliability engine",
    "salib": "Global sensitivity analysis (SALib)",
    "pystra": "Structural reliability (pystra)",
    "dxf_import": "DXF import",
    "pdf_import": "PDF cross-section import",
    "dxf_export": "DXF export",
    "calc_package": "Calc-package reporting",
}

# The single method whose full parameter table is shown inline in the catalog
# (the complete per-method reference for every method lives in Appendix A).
HEADLINE_METHOD = {
    "bearing_capacity": "bearing_capacity_analysis",
    "settlement": "consolidation_settlement",
    "axial_pile": "axial_pile_capacity",
    "lateral_pile": "lateral_pile_analysis",
    "pile_group": "pile_group_6dof",
    "drilled_shaft": "drilled_shaft_capacity",
    "wave_equation": "bearing_graph",
    "downdrag": "downdrag_analysis",
    "sheet_pile": "cantilever_wall",
    "soe": "braced_excavation",
    "retaining_walls": "cantilever_wall",
    "ground_improvement": "aggregate_piers",
    "slope_stability": "search_critical_surface",
    "fem2d": "fem2d_slope_srm",
    "seismic_geotech": "seismic_earth_pressure",
    "liquefaction": "liquefaction_analysis",
    "reliability": "monte_carlo",
    "gstools": "kriging",
    "subsurface": None,
    "salib": "sobol_sample",
    "pystra": None,
    "dxf_import": None,
    "pdf_import": None,
    "dxf_export": None,
    "calc_package": None,
    "opensees": None,
    "pystrata": "eql_site_response",
    "seismic_signals": None,
    "hvsrpy": None,
    "liquepy": None,
    "swprocess": None,
}

# Worked example: agent -> (worked_examples.json key, natural-language ask, note)
WORKED = {
    "bearing_capacity": ("bearing_capacity.bearing_capacity_analysis",
        "A 2&nbsp;m &times; 10&nbsp;m strip footing bears 1.5&nbsp;m deep in a c&ndash;&phi; soil "
        "(&phi;=30&deg;, c=10&nbsp;kPa, &gamma;=18&nbsp;kN/m&sup3;), water 5&nbsp;m down. What is the "
        "bearing capacity and the allowable pressure at FOS&nbsp;=&nbsp;3?",
        "The result carries every N-, shape-, depth- and inclination factor separately, so a "
        "reviewer can trace the number back to the governing terms."),
    "settlement": ("settlement.consolidation_settlement",
        "A 4&nbsp;m normally-consolidated clay layer (Cc=0.25, e<sub>0</sub>=0.8) sees a 50&nbsp;kPa "
        "stress increase. How much primary consolidation settlement?",
        "Layer summation with per-layer stress on the e&ndash;log&nbsp;p curve; the same call also "
        "accepts a footing width so it computes the 2:1 / Boussinesq stress increase itself."),
    "axial_pile": ("axial_pile.axial_pile_capacity",
        "A closed-end 0.4&nbsp;m pipe pile driven 15&nbsp;m through two sand layers &mdash; what is "
        "the axial capacity, split into skin and toe?", ""),
    "drilled_shaft": ("drilled_shaft.drilled_shaft_capacity",
        "A 1.0&nbsp;m drilled shaft, 15&nbsp;m long, through stiff clay over sand. Capacity by the "
        "GEC-10 method?", ""),
    "lateral_pile": ("lateral_pile.lateral_pile_analysis",
        "A 0.61&nbsp;m free-head pile in sand takes a 200&nbsp;kN lateral load at the head. Give the "
        "deflection and maximum bending moment.",
        "The full deflection / moment / shear / soil-reaction profiles are returned as arrays "
        "(101 points here) for plotting; the summary scalars are shown above."),
    "pile_group": ("pile_group.pile_group_6dof",
        "A 3&times;3 pile group at 1.2&nbsp;m spacing under 6000&nbsp;kN vertical, 300&nbsp;kN "
        "lateral, and an 800&nbsp;kN&middot;m moment &mdash; how do the loads distribute to the piles?",
        "One right-hand sign convention is carried end-to-end; the leading-row piles pick up the "
        "moment as extra compression."),
    "downdrag": ("downdrag.downdrag_analysis",
        "A 0.4&nbsp;m pile, 18&nbsp;m long, through settling fill and soft clay over sand, with a "
        "400&nbsp;kN dead load. Where is the neutral plane and how large is the drag load?", ""),
    "wave_equation": ("wave_equation.bearing_graph",
        "For a 20&nbsp;m steel pile and a 45&nbsp;kN ram at 1.5&nbsp;m stroke, plot the bearing "
        "graph &mdash; blows per metre against static resistance.", ""),
    "sheet_pile": ("sheet_pile.cantilever_wall",
        "A cantilever sheet pile retains a 4&nbsp;m excavation in sand (&phi;=32&deg;) with a "
        "10&nbsp;kPa surcharge. What embedment is needed and what is the maximum moment?", ""),
    "soe": ("soe.braced_excavation",
        "An 8&nbsp;m braced cut in sand with two strut levels &mdash; give the apparent-pressure "
        "diagram, strut loads, and the required wall section.", ""),
    "retaining_walls": ("retaining_walls.cantilever_wall",
        "A 6&nbsp;m cantilever retaining wall on &phi;=32&deg; backfill with a 12&nbsp;kPa surcharge "
        "&mdash; check sliding, overturning, and bearing.",
        "Here the auto-sized base fails sliding (FOS 1.27 &lt; 1.5) &mdash; exactly the kind of "
        "check-and-revise loop the toolkit is built for."),
    "ground_improvement": ("ground_improvement.aggregate_piers",
        "0.76&nbsp;m aggregate piers on a 1.5&nbsp;m triangular grid under soft ground &mdash; what "
        "settlement reduction and bearing improvement do they buy?", ""),
    "slope_stability": ("slope_stability.search_critical_surface",
        "Find the critical circular slip surface and minimum factor of safety for a 10&nbsp;m "
        "c&ndash;&phi; slope (c&prime;=12&nbsp;kPa, &phi;&prime;=28&deg;) by Bishop&rsquo;s method.",
        "1,418 trial surfaces were evaluated in the entry&ndash;exit search; the minimum governs."),
    "fem2d": ("fem2d.fem2d_slope_srm",
        "Run a finite-element strength-reduction analysis on the same slope (c=10&nbsp;kPa, "
        "&phi;=20&deg;) &mdash; what SRF causes collapse?",
        "The SRF at which the Newton solve stops converging is the FEM factor of safety; T6 "
        "quadratic elements are used by default."),
    "seismic_geotech": ("seismic_geotech.seismic_earth_pressure",
        "For a 6&nbsp;m wall on &phi;=30&deg; backfill with a horizontal seismic coefficient "
        "k<sub>h</sub>=0.2, what are the Mononobe&ndash;Okabe active and passive coefficients and the "
        "seismic thrust?", ""),
    "liquefaction": ("liquefaction.liquefaction_analysis",
        "SPT profile (N<sub>160</sub> = 12/15/10/18) with water 2&nbsp;m down, a<sub>max</sub>=0.3g and "
        "M<sub>w</sub> 7.0 &mdash; is the ground liquefiable and what is the factor of safety by depth?",
        "The single <code>liquefaction</code> tool auto-routes: SPT input &rarr; Boulanger &amp; "
        "Idriss (2014) by default (or NCEER/Youd&nbsp;2001 for code work), CPT input &rarr; B&amp;I "
        "2014 with LPI/LSN/LDI."),
    "reliability": ("reliability.monte_carlo",
        "Resistance R (mean&nbsp;500, COV&nbsp;20%, lognormal) against load S (mean&nbsp;300, "
        "COV&nbsp;15%) &mdash; 50,000 Monte-Carlo trials: what is the reliability index and the "
        "probability of failure?",
        "This is the payoff of the whole architecture: any deterministic method plugs into the "
        "same <code>g()</code> contract and returns &beta; and P<sub>f</sub> instead of a lone number."),
}

# 21 reference modules: agent -> (title, standard, coverage)
REFERENCE_INFO = {
    "dm7": ("NAVFAC DM7", "NAVFAC DM7.01 / DM7.02",
            "The classic Navy design manual &mdash; soil classification, stress distribution, "
            "settlement, seepage, bearing, lateral earth pressure, and slope charts. 340+ equations "
            "plus figure catalogs."),
    "gec4": ("FHWA GEC-4", "FHWA-IF-99-015",
             "Ground anchors and anchored systems &mdash; apparent-pressure envelopes, tributary "
             "anchor loads, wall design."),
    "gec5": ("FHWA GEC-5", "FHWA-NHI-10-016",
             "Soil and rock properties and their measurement."),
    "gec6": ("FHWA GEC-6", "FHWA-SA-02-054",
             "Shallow foundations &mdash; Vesic/AASHTO bearing factors, groundwater corrections, "
             "the Hough settlement method."),
    "gec7": ("FHWA GEC-7", "FHWA-0-IF-03-017",
             "Soil-nail walls &mdash; nail-head strength, bond, facing, and global stability."),
    "gec8": ("FHWA GEC-8", "FHWA-NHI-13-026",
             "Design and construction of continuous flight auger piles."),
    "gec9": ("FHWA GEC-9", "FHWA-NHI-16-009",
             "Original design and construction of driven-pile foundations (companion to GEC-12)."),
    "gec10": ("FHWA GEC-10", "FHWA-NHI-18-024",
              "Drilled shafts &mdash; alpha / beta side resistance, base resistance, rock sockets, "
              "the rational OCR/Ko and Chen-2011 chains."),
    "gec11": ("FHWA GEC-11", "FHWA-NHI-10-025",
              "MSE walls and reinforced-soil slopes &mdash; external/internal stability, LRFD "
              "load-factor bookkeeping, seismic."),
    "gec12": ("FHWA GEC-12", "FHWA-NHI-16-064",
              "Driven-pile foundations &mdash; the Nordlund / Tomlinson worked examples the axial "
              "pile module is validated against."),
    "gec13": ("FHWA GEC-13", "FHWA ground-modification manual",
              "Ground modification &mdash; aggregate piers, stone columns (Priebe), prefabricated "
              "vertical drains, surcharge preloading."),
    "gec14": ("FHWA GEC-14", "FHWA-NHI ground-improvement",
              "Additional ground-improvement methods."),
    "micropile": ("FHWA Micropile Manual", "FHWA-NHI-05-039",
                  "Micropile design &mdash; bond stress, structural capacity, the LPILE benchmark "
                  "the lateral-pile module checks against."),
    "ufc_backfill": ("UFC 3-220-04N", "Unified Facilities Criteria",
                     "Backfill design &mdash; compaction, filter criteria, drainage."),
    "ufc_expansive": ("UFC 3-220-07", "Unified Facilities Criteria",
                      "Expansive soils &mdash; swell potential, heave, pier design."),
    "ufc_pavement": ("UFC 3-260-02", "Unified Facilities Criteria",
                     "Airfield pavement design &mdash; CBR, thickness, equivalent single-wheel load."),
    "fema_p2082": ("FEMA P-2082 / 2020 NEHRP", "NEHRP provisions",
                   "Seismic site classification &mdash; the BC/CD/DE site classes with the BC "
                   "baseline (the post-2020 multi-period approach)."),
    "california_trenching": ("Caltrans Trenching &amp; Shoring", "Caltrans T&amp;S Manual",
                             "Shoring and excavation &mdash; soldier piles, anchored walls, basal "
                             "heave, the worked examples validating soe / sheet_pile."),
    "fhwa_pavements": ("FHWA Pavements", "FHWA-NHI-05-037",
                       "Geotechnical aspects of pavements &mdash; resilient modulus, CBR, frost, "
                       "drainage."),
    "reference_db": ("Cross-reference text search", "&mdash;",
                     "A full-text (SQLite FTS5) search DB over the chapter text of every reference, "
                     "with synonym query expansion."),
    "figure_db": ("Figure catalog + vision read-off", "&mdash;",
                  "A searchable catalog of every figure across the references; pairs with the "
                  "<code>read_reference_figure</code> vision tool to render a chart and read a value "
                  "off it."),
}


# ===========================================================================
# Per-module narrative (problems / methods / limits). Populated from the
# module DESIGN.md files. Missing entries fall back to the catalog brief.
# ===========================================================================

MODULE_NARRATIVE: dict[str, dict] = {

    "bearing_capacity": {
        "problems": "Sizing a spread footing or mat and confirming the ground can carry it: the "
            "ultimate bearing capacity, the allowable pressure at a chosen factor of safety, and the "
            "sensitivity of both to width, embedment, groundwater, eccentricity, and a layered profile.",
        "methods": [
            "General bearing-capacity equation with N<sub>c</sub>, N<sub>q</sub>, N<sub>&gamma;</sub> "
            "by **Vesic**, **Meyerhof**, or **Hansen** (selectable); Vesic is the default and the AASHTO/"
            "GEC-6 factor set.",
            "Shape, depth, load-inclination, base-tilt, and ground-slope correction factors (Vesic form).",
            "Eccentric loading via the Meyerhof effective-area (B&prime;, L&prime;) reduction.",
            "Two-layer (strong-over-weak / weak-over-strong) bearing with load spread.",
            "Groundwater in the failure wedge through an effective-unit-weight model.",
            "References: Vesic (1973); Meyerhof (1963); Hansen (1970); FHWA GEC-6 (FHWA-SA-02-054); "
            "AASHTO LRFD.",
        ],
        "limits": [
            "The high-level <code>compute()</code> applies Vesic depth factors and averages the "
            "effective unit weight over the wedge; this runs higher than an example that sets the "
            "depth factor to 1.0 (cohesive overburden) or uses an AASHTO C<sub>w</sub> groundwater "
            "correction &mdash; both are defensible conventions (validation V-021).",
            "General shear failure is assumed; punching/local shear in very loose or soft soils is not "
            "modelled separately.",
        ],
    },

    "settlement": {
        "problems": "How far a foundation settles and how fast: immediate (elastic) movement, primary "
            "consolidation of clays, granular settlement of sands, and the settlement&ndash;time curve.",
        "methods": [
            "Elastic (immediate) settlement with shape/rigidity influence factors.",
            "**Schmertmann (1978)** strain-influence-factor method for granular soils.",
            "One-dimensional **C<sub>c</sub>/C<sub>r</sub> e&ndash;log&nbsp;p consolidation** with "
            "layer summation and preconsolidation (NC/OC).",
            "**Hough (1959)** granular C&prime;-index settlement (bearing-capacity index from corrected "
            "SPT N&prime;; GEC-6 Fig 5-19).",
            "Stress increase by the 2:1 method, Boussinesq, or Westergaard; secondary compression "
            "(C<sub>&alpha;</sub>); Terzaghi time-rate for the settlement&ndash;time curve.",
            "References: Schmertmann et al. (1978); Hough (1959); Terzaghi 1-D consolidation; FHWA GEC-6.",
        ],
        "limits": [
            "The Hough C&prime; index is the Hough bearing-capacity index, not C<sub>c</sub>/(1+e<sub>0</sub>) "
            "&mdash; use the consolidation path for cohesive soils and Hough/Schmertmann for granular.",
        ],
    },

    "axial_pile": {
        "problems": "Axial capacity of a driven pile in compression (and uplift): the skin friction and "
            "toe resistance developed as the pile passes through a layered profile, and how capacity "
            "grows with embedment depth.",
        "methods": [
            "**Nordlund (1963/1979)** method for shaft and toe in cohesionless soils (K&ndash;&delta; and "
            "correction-factor charts, bearing-capacity-factor toe with the Meyerhof q<sub>L</sub> limit).",
            "**Tomlinson / &alpha;-method** total-stress skin friction in cohesive soils; 9&middot;c<sub>u</sub> "
            "end bearing.",
            "**&beta;-method** effective-stress skin friction for all layers.",
            "Groundwater-split effective-stress integration; optional per-layer separate shaft/toe "
            "friction angle; below-grade pile head (<code>head_depth</code>); uplift capacity.",
            "Pile sections: closed/open pipe, square/circular concrete, and H-piles (with the box "
            "perimeter/toe-area convention).",
            "References: FHWA GEC-12 (FHWA-NHI-16-064); Nordlund; Tomlinson; validated against the GEC-12 "
            "Nordlund/&alpha; worked examples (validation V-001/V-002).",
        ],
        "limits": [
            "The Tomlinson &alpha;&ndash;c<sub>u</sub> curve runs conservative in the stiff-clay band "
            "(&alpha;&asymp;0.42 vs DrivenPiles &asymp;0.76 at s<sub>u</sub>&asymp;1.9&nbsp;ksf) &mdash; a "
            "defensible curve choice, documented, not tuned (V-002).",
            "Datum matters: reference the profile to the pile head (e.g. a below-grade footing) so no "
            "spurious skin friction is added above the head.",
        ],
    },

    "lateral_pile": {
        "problems": "Lateral response of a pile or drilled shaft under head shear, moment, and axial "
            "load: deflection, rotation, bending-moment, shear, and soil-reaction profiles with depth, "
            "for free or fixed heads and above-ground stickup.",
        "methods": [
            "**p&ndash;y curve** soil models (COM624P / LPILE family): Matlock soft clay, Reese stiff clay "
            "above/below water table, **Reese (1974) sand** (simplified and full four-segment "
            "construction), API sand, and others.",
            "Banded finite-difference beam-column solver with P-&Delta; (axial-load) effects.",
            "Free/fixed head conditions, above-ground stickup, and H-pile strong/weak-axis or "
            "concrete-filled-pipe sections.",
            "References: Reese, Cox &amp; Koop (1974); Matlock (1970); COM624P; validated against a "
            "COM624P oracle suite and the FHWA Micropile LPILE benchmark (V-017).",
        ],
        "limits": [
            "Section stiffness is a single linear EI &mdash; a composite/nonlinear-EI section (casing + "
            "grout + bar) is not yet modelled, which is the ~19% deflection gap in the micropile "
            "benchmark (V-017); the solver and p&ndash;y construction themselves are verified exact.",
        ],
    },

    "pile_group": {
        "problems": "How a rigid pile cap distributes structural loads to the individual piles, and how "
            "much a group settles: axial/lateral/torsional load sharing under a 6-DOF load set, group "
            "efficiency, and group settlement.",
        "methods": [
            "**Rigid-cap 6-DOF** stiffness analysis &mdash; per-pile axial and lateral stiffness, batter, "
            "and one right-hand sign convention carried end to end.",
            "**Converse&ndash;Labarre** group efficiency.",
            "**Meyerhof (1976)** SPT group settlement and an elastic equivalent-raft (2V:1H) settlement.",
            "References: Converse&ndash;Labarre; Meyerhof (1976); validated against GEC-12 Table D-23 (V-005).",
        ],
        "limits": [
            "The cap is treated as rigid; pile&ndash;soil&ndash;pile interaction (p-multipliers) is not "
            "part of the 6-DOF stiffness distribution.",
        ],
    },

    "drilled_shaft": {
        "problems": "Axial capacity of a drilled shaft (bored pile) in soil and rock: side resistance in "
            "clay and sand, base resistance, rock-socket capacity, belled bases, and the LRFD factored "
            "resistance.",
        "methods": [
            "**&alpha;-method** side resistance in clay (AASHTO 0.55 default, or the GEC-10 **Chen (2011)** "
            "rational &alpha; with the UU/UC&rarr;CIUC s<sub>u</sub> transform).",
            "**&beta;-method** side resistance in sand (O&rsquo;Neill&ndash;Reese depth default, or the "
            "GEC-10 rational OCR/K<sub>o</sub> chain).",
            "Base resistance in sand (0.60&middot;N<sub>60</sub> with the large-diameter reduction) and "
            "clay (N<sub>c</sub>&middot;s<sub>u</sub> cap); rock-socket side + base; permanent-casing and "
            "top/bottom exclusion zones.",
            "LRFD resistance factors and capacity-vs-depth curves.",
            "References: FHWA GEC-10 (FHWA-NHI-18-024); O&rsquo;Neill &amp; Reese; Chen &amp; Kulhawy; "
            "validated against the GEC-10 Appendix A example (V-006/V-007/V-008).",
        ],
        "limits": [
            "The large-diameter base reduction (GEC-10 &sect;13.3.4.3) is applied by default; an example "
            "reporting the unreduced nominal base resistance will differ by that factor (V-008b).",
        ],
    },

    "wave_equation": {
        "problems": "Pile drivability and the dynamic bearing check: the bearing graph (static resistance "
            "vs blow count), driving stresses, and refusal, from a hammer&ndash;cushion&ndash;pile&ndash;soil "
            "model &mdash; the analysis behind a driven-pile field acceptance criterion.",
        "methods": [
            "**Smith (1960)** one-dimensional wave-equation model: elasto-plastic soil springs (quake) "
            "with **smith** or **smith-viscous** damping, hammer/cushion/helmet stack, and a "
            "characteristic-line time integration.",
            "Single-blow analysis, bearing graph, and drivability vs depth; a hammer database plus "
            "custom ram/stroke/efficiency and cushion definitions.",
            "References: Smith (1960); GRLWEAP-style modelling; FHWA GEC-12.",
        ],
        "limits": [
            "Diesel hammers are an energy&rarr;velocity approximation, not a GRLWEAP combustion + "
            "ram-cycle model &mdash; treat a diesel bearing graph as a trend/limit check, not a tie-out "
            "(V-003).",
        ],
    },

    "downdrag": {
        "problems": "Negative skin friction (downdrag) on a pile through settling ground: the neutral "
            "plane, the maximum axial load in the pile, and the drag load added to the structural load.",
        "methods": [
            "**Fellenius neutral-plane** method &mdash; the load-vs-depth and resistance-vs-depth crossing "
            "with negative skin friction fully mobilized above the neutral plane and toe mobilization.",
            "Fill-induced downdrag, groundwater drawdown, and a maximum-load structural check.",
            "References: Fellenius; UFC 3-220-20; validated against GEC-12 (V-004, NP/Q<sub>max</sub>/DF "
            "within 1%).",
        ],
        "limits": [
            "For an H-pile the box toe area must be passed as the pile area; the module&rsquo;s intrinsic "
            "&beta;-shaft can differ from a Nordlund shaft, so feed the design distribution to isolate "
            "the neutral-plane equilibrium.",
        ],
    },

    "sheet_pile": {
        "problems": "Cantilever and single-anchor flexible retaining walls: the embedment for stability, "
            "the maximum bending moment, and the anchor force, from a layered earth-pressure diagram with "
            "groundwater.",
        "methods": [
            "**Rankine / Coulomb** active and passive coefficients (with wall friction), and a "
            "**Caquot&ndash;Kerisel log-spiral** passive option that avoids Coulomb&rsquo;s over-prediction "
            "at high &delta;.",
            "Free-earth-support solution for cantilever and anchored walls; layered active-pressure "
            "builder with hydrostatic water; embedment-increase factor and a FOS on passive resistance.",
            "References: Rankine; Coulomb; Caquot&ndash;Kerisel; NAVFAC DM7.2; validated against Caltrans "
            "T&amp;S Ex 7-2 / 8-1 (V-025/V-013).",
        ],
        "limits": [
            "<code>analyze_cantilever</code> is a continuous per-metre free-earth-support solver &mdash; "
            "soldier-pile effective-width/arching and the simplified toe-moment cubic are not modelled "
            "(V-012).",
        ],
    },

    "soe": {
        "problems": "Support of an open excavation: braced and cantilever walls, apparent-pressure "
            "diagrams, strut/anchor loads, wall-section selection, and the bottom-stability checks "
            "(basal heave, blowout, piping).",
        "methods": [
            "**FHWA apparent-pressure** envelopes (Terzaghi&ndash;Peck / Peck) for braced cuts and "
            "anchored walls, with tributary strut/anchor loads and hinge moments.",
            "Cantilever and single-anchor free-earth solutions; ground-anchor design; wall-section "
            "sizing.",
            "Basal heave &mdash; **Bjerrum&ndash;Eide** bearing ratio and a **Caltrans** force balance with "
            "sidewall shear; bottom blowout and piping (exit-gradient) checks.",
            "References: FHWA GEC-4 (FHWA-IF-99-015); Terzaghi &amp; Peck; Bjerrum &amp; Eide; Caltrans "
            "T&amp;S; validated V-014/V-016.",
        ],
        "limits": [
            "The single-anchor FHWA apparent-diagram embedment solver and a multi-anchor apparent-envelope "
            "helper are documented gaps; the primitives (K, p<sub>e</sub>, tributary loads, hinge "
            "moments) reproduce the worked examples (V-016).",
        ],
    },

    "retaining_walls": {
        "problems": "Gravity/cantilever concrete retaining walls and mechanically-stabilized-earth (MSE) "
            "walls: external stability (sliding, overturning, bearing, eccentricity) and MSE internal "
            "stability (reinforcement tension and pullout), by ASD and LRFD.",
        "methods": [
            "Cantilever-wall stability with **Rankine/Coulomb** thrust decomposition and **Meyerhof** "
            "bearing.",
            "**MSE coherent-gravity** method: external stability and internal K<sub>r</sub>/K<sub>a</sub> "
            "and F* pullout curves for ribbed strips and steel bar-mats/welded grids.",
            "**AASHTO/GEC-11 LRFD** external stability &mdash; Strength I max/min + Service I load-factor "
            "combinations, live-load exclusions, and per-mode capacity/demand ratios.",
            "References: FHWA GEC-11 (FHWA-NHI-10-025); AASHTO LRFD; validated against GEC-11 Ex E4/E7 "
            "(V-009/V-010/V-011).",
        ],
        "limits": [
            "MSE external stability is available both as an ASD factor-of-safety path (default, unchanged) "
            "and the packaged LRFD CDR path via <code>lrfd_external=True</code>.",
        ],
    },

    "ground_improvement": {
        "problems": "Improving weak ground before it is loaded: aggregate piers and stone columns, "
            "prefabricated vertical (wick) drains, surcharge preloading, and vibro-compaction &mdash; the "
            "settlement reduction, capacity increase, and time to consolidate.",
        "methods": [
            "Aggregate piers / stone columns: area-replacement ratio, equal-strain settlement-reduction "
            "factor, composite modulus, and the **Priebe (1995)** basic improvement factor (incl. n<sub>0</sub>).",
            "**Barron&ndash;Hansbo** radial consolidation for PVDs (with smear and well resistance) and "
            "combined vertical+radial time-rate; drain-spacing design.",
            "Surcharge preloading and vibro-compaction feasibility.",
            "References: FHWA GEC-13 (ground-modification manual); Priebe (1995); Barron (1948) / Hansbo "
            "(1981); validated V-018/V-019/V-020.",
        ],
        "limits": [
            "Aggregate-pier settlement uses the equal-strain settlement-reduction model, not the GEC-13 "
            "stiffness-modulus (k<sub>g</sub>) two-layer method; the Priebe factor depends on the "
            "area-ratio and column-friction conventions supplied (V-018/V-019).",
        ],
    },

    "slope_stability": {
        "problems": "Will the slope stand? The factor of safety of an earth or rock slope against sliding "
            "&mdash; for a specified surface or the critical one found by search &mdash; under drained or "
            "undrained conditions, groundwater, surcharge, seismicity, reinforcement, and rapid drawdown, "
            "and its probabilistic FOS.",
        "methods": [
            "Methods of slices: **Ordinary/Fellenius**, **Bishop simplified (1955)**, **Janbu** "
            "(corrected), **Spencer (1967)**, **Morgenstern&ndash;Price**, and the rigorous **GLE** "
            "(Fredlund&ndash;Krahn) with interslice-force functions.",
            "Critical-surface search: circular centre-grid, entry&ndash;exit arcs, random noncircular "
            "polylines with differential-evolution / PSO refinement, and weak-layer search.",
            "Reinforcement (soil nails, tieback anchors, geosynthetics, Ito&ndash;Matsui stabilizing "
            "piles), SHANSEP and Hoek&ndash;Brown layer strengths, ponded water, tension cracks, and "
            "pseudo-static k<sub>h</sub>.",
            "Seismic displacement (**Newmark** sliding block + **Jibson 2007** regression), **rapid "
            "drawdown** (USACE 2-stage / Duncan&ndash;Wright&ndash;Wong 3-stage), infinite-slope FOS, and "
            "**probabilistic FOS** (FOSM and Monte Carlo).",
            "References: Fredlund &amp; Krahn (1977); Bishop (1955); Spencer (1967); Janbu; Duncan, Wright "
            "&amp; Brandon (2014); Ito &amp; Matsui (1975); Jibson (2007); validated vs Fredlund&ndash;Krahn "
            "/ ACADS / Duncan / Slide2 (VALIDATION.md, V-015/V-026&hellip;V-040).",
        ],
        "limits": [
            "Pore pressure is a piezometric surface or per-layer r<sub>u</sub>, not an arbitrary "
            "interpolated flow-net grid (V-029).",
            "The tension crack is entry-side and keeps the cracked wedge as zero-strength driving soil "
            "with full hydrostatic thrust &mdash; a conservative convention vs mass-truncation codes "
            "(V-026).",
        ],
    },

    "fem2d": {
        "problems": "When limit equilibrium is not enough: a full 2D plane-strain finite-element engine "
            "for stress and deformation of soil masses, slope stability by strength reduction, bearing "
            "collapse, excavation unloading, seepage, and coupled consolidation &mdash; with structural "
            "beam members for walls.",
        "methods": [
            "Elements: **T6 quadratic triangle** (default, collapse-accurate), CST, Q4, and 2-node "
            "Euler&ndash;Bernoulli beams.",
            "Constitutive models: linear elastic, **Mohr&ndash;Coulomb** (3D-principal return mapping, "
            "non-associated flow), and **Hardening Soil** (Schanz, Vermeer &amp; Bonnier 1999).",
            "**Strength-reduction method** for slope FOS (Griffiths &amp; Lane 1999); staged construction / "
            "excavation; initial-stress relaxation for tunnels/cavities.",
            "Steady/transient **seepage** and **Biot consolidation** (staggered, and a monolithic "
            "Taylor&ndash;Hood u&ndash;p scheme); PLAXIS-style calc-package plots.",
            "References: Griffiths &amp; Lane (1999); de Souza Neto et al. (2008); Clausen et al. (2006); "
            "Schanz et al. (1999); validated vs Griffiths&ndash;Lane, the Prandtl solution (~2%), and "
            "Itasca FLAC (VALIDATION.md, V-023/V-024).",
        ],
        "limits": [
            "CST elements lock in plastic collapse (Prandtl N<sub>c</sub> overshoots) &mdash; use T6 for "
            "collapse/FOS work; CST is for elastic/seepage/Biot only.",
            "It is a research-grade solver, not a substitute for a commercial FE package under QA; a "
            "far-field boundary must be placed far enough to avoid truncation error.",
        ],
    },

    "seismic_geotech": {
        "problems": "Site seismic geotechnics: the ASCE-7 site class and design coefficients, "
            "Mononobe&ndash;Okabe seismic earth pressures on walls, SPT liquefaction triggering, and "
            "post-liquefaction residual strength.",
        "methods": [
            "**ASCE 7** site classification from V<sub>s30</sub>, N-bar, or s<sub>u</sub>-bar, with "
            "F<sub>pga</sub>/F<sub>a</sub>/F<sub>v</sub> and design spectral values.",
            "**Mononobe&ndash;Okabe** active/passive seismic earth-pressure coefficients "
            "(K<sub>AE</sub>/K<sub>PE</sub>), battered-wall correct, with the seismic thrust and point of "
            "application.",
            "SPT liquefaction triggering (**NCEER / Youd et al. 2001** simplified procedure) and residual "
            "strength.",
            "References: ASCE 7; Mononobe&ndash;Okabe; NCEER (1997) / Youd et al. (2001); validated vs "
            "GEC-11 Ex E7 (V-011, K<sub>AE</sub> within 0.06%).",
        ],
        "limits": [
            "Its SPT procedure is the NCEER/Youd-2001 method (for code-compliance work); the default "
            "unified liquefaction tool uses Boulanger &amp; Idriss (2014) &mdash; choose per the citation "
            "you need.",
        ],
    },

    "liquefaction": {
        "problems": "The single, discoverable liquefaction-triggering tool: is the ground liquefiable, and "
            "what is the factor of safety by depth &mdash; from SPT or CPT data, by the method your work "
            "requires.",
        "methods": [
            "Auto-routes by input type and method: **CPT** (q<sub>c</sub>/f<sub>s</sub>) &rarr; **Boulanger "
            "&amp; Idriss (2014)** via liquepy, with LPI / LSN / LDI indices.",
            "**SPT** (N<sub>160</sub>) &rarr; Boulanger &amp; Idriss (2014) by default, or **NCEER / Youd "
            "et al. (2001)** via <code>method='nceer2001'</code> for code-compliance work.",
            "References: Boulanger &amp; Idriss (2014); NCEER (1997) / Youd et al. (2001).",
        ],
        "limits": [
            "Routing lives at the agent layer (no cross-module import); the underlying per-module "
            "functions (<code>liquepy</code>, <code>seismic_geotech</code>) remain directly callable.",
        ],
    },

    "liquepy": {
        "problems": "Direct access to the Boulanger &amp; Idriss (2014) liquefaction building blocks and "
            "field correlations, wrapping the <code>liquepy</code> library &mdash; for when you want the "
            "individual CRR/CSR/MSF/K<sub>&sigma;</sub> terms rather than the unified tool.",
        "methods": [
            "B&amp;I-2014 **CPT** triggering (packaged <code>run_bi2014</code>) with LPI/LSN/LDI.",
            "B&amp;I-2014 **SPT** triggering composed from the module-level building blocks "
            "(CRR<sub>M7.5</sub>, r<sub>d</sub>, CSR, K<sub>&sigma;</sub>) plus the SPT fines correction "
            "and MSF.",
            "Field correlations (V<sub>s</sub>, D<sub>r</sub>, G<sub>0</sub> from N).",
            "References: Boulanger &amp; Idriss (2014); the liquepy library. Requires the "
            "<code>[liquepy]</code> extra.",
        ],
        "limits": [
            "liquepy ships a packaged CPT triggering object but no packaged SPT object &mdash; the SPT "
            "procedure is composed from its tested building blocks.",
        ],
    },

    "opensees": {
        "problems": "Nonlinear finite-element site response and element-test simulation, wrapping "
            "OpenSeesPy: cyclic direct-simple-shear behaviour of liquefiable sand and one-dimensional "
            "ground response.",
        "methods": [
            "**PM4Sand** cyclic DSS element tests (stress path, pore-pressure generation).",
            "One-dimensional nonlinear site response (soil column, base motion).",
            "References: PM4Sand (Boulanger &amp; Ziotopoulou); the OpenSeesPy library. Requires the "
            "<code>[opensees]</code> extra.",
        ],
        "limits": [
            "A wrapper around OpenSeesPy &mdash; model fidelity and convergence follow OpenSees; a heavy "
            "optional dependency.",
        ],
    },

    "pystrata": {
        "problems": "One-dimensional equivalent-linear site response (a SHAKE-type analysis): amplification "
            "of a base motion through a layered soil column to a surface spectrum and profiles of peak "
            "strain and acceleration.",
        "methods": [
            "Equivalent-linear and linear-elastic 1D response, wrapping **pyStrata**.",
            "Modulus-reduction/damping curves: **Darendeli (2001)**, **Menq (2003)**, or custom.",
            "References: the pyStrata library; Darendeli (2001); Menq (2003). Requires the "
            "<code>[pystrata]</code> extra.",
        ],
        "limits": [
            "Equivalent-linear (not fully nonlinear); large-strain problems may warrant the OpenSees "
            "nonlinear path.",
        ],
    },

    "seismic_signals": {
        "problems": "Ground-motion signal processing: response spectra, intensity measures, and rotated "
            "components from an acceleration record &mdash; the quantities behind a seismic demand.",
        "methods": [
            "Response spectra and Fourier processing (**eqsig**); rotated-component spectra RotD50/RotD100 "
            "(**pyrotd**); Arias intensity, significant duration, and other intensity measures.",
            "References: the eqsig and pyrotd libraries. Requires the <code>[seismic-signals]</code> extra.",
        ],
        "limits": [],
    },

    "hvsrpy": {
        "problems": "Site characterization from ambient-noise measurements: the horizontal-to-vertical "
            "spectral ratio (HVSR) and its peak frequency, an estimate of the fundamental site period.",
        "methods": [
            "HVSR processing of ambient-noise records with the modern statistical framework, wrapping "
            "**hvsrpy**.",
            "References: the hvsrpy library (Cheng, Vantassel et al.). Requires the <code>[hvsrpy]</code> "
            "extra.",
        ],
        "limits": [],
    },

    "subsurface": {
        "problems": "Getting field and lab data into a modelled ground: parse and validate DIGGS, GEF/"
            "BRO-XML CPT, and AGS4 files, visualize parameters against depth, and compute trend "
            "statistics &mdash; the ingest side of characterization.",
        "methods": [
            "**DIGGS** parser (20 test types) with Plotly parameter-vs-depth, Atterberg, and trend plots.",
            "Folded format adapters: **GEF/BRO-XML** CPT and borehole parsing (pygef), **AGS4** read/"
            "validate (python-ags4), and **DIGGS** schema/dictionary validation (pydiggs).",
            "References: the DIGGS, GEF, and AGS4 standards; pygef / python-ags4 / pydiggs. Requires the "
            "<code>[subsurface]</code> extra for the format adapters.",
        ],
        "limits": [
            "Attach a file via the agent&rsquo;s attachment mechanism (<code>attachment_key</code>) to "
            "parse a document&rsquo;s contents.",
        ],
    },

    "gstools": {
        "problems": "Geostatistics for spatially variable ground: fit a variogram to scattered data, krige "
            "a property between borings, and generate correlated random fields for spatial-variability "
            "studies.",
        "methods": [
            "Variogram estimation and model fitting; ordinary/simple **kriging**; conditional and "
            "unconditional **random-field** generation, wrapping **GSTools**.",
            "References: the GSTools library (Müller et al.). Requires the <code>[gstools]</code> extra.",
        ],
        "limits": [],
    },

    "swprocess": {
        "problems": "Surface-wave site characterization: process multichannel analysis of surface waves "
            "(MASW) into a dispersion curve &mdash; the field data behind a shear-wave-velocity profile.",
        "methods": [
            "MASW dispersion processing and curve extraction, wrapping **swprocess**.",
            "References: the swprocess library (Vantassel). Requires the <code>[swprocess]</code> extra.",
        ],
        "limits": [
            "Produces the dispersion curve; inversion to a V<sub>s</sub> profile is a separate step.",
        ],
    },

    "reliability": {
        "problems": "The variability engine: turn any deterministic method into a probabilistic one and "
            "report the reliability index &beta; and probability of failure P<sub>f</sub> &mdash; because "
            "with soil COVs of 10&ndash;40% the honest deliverable is a distribution, not a lone factor of "
            "safety.",
        "methods": [
            "**FOSM** (First-Order Second-Moment), **PEM** (Rosenblueth point-estimate), **Monte Carlo** "
            "(with Cholesky correlation and Latin Hypercube), and native **FORM** (Hasofer&ndash;Lind "
            "design point).",
            "A published **COV knowledge base** (Duncan 2000; TC304; Phoon &amp; Kulhawy) queried by "
            "<code>cov_guidance</code>; **Vanmarcke** spatial-averaging variance reduction; combined COV.",
            "Ready-made wrappers: <code>bearing_capacity_reliability</code>, "
            "<code>axial_pile_reliability</code>, <code>slope_reliability</code>.",
            "References: Duncan (2000); Phoon &amp; Kulhawy (1999); Baecher &amp; Christian; Vanmarcke "
            "(1977); validated vs Duncan (2000) reliability example.",
        ],
        "limits": [
            "The performance function is a scalar g(values); correlated depth-varying strength laws with a "
            "single gradient variable are a documented scope edge (V-030).",
        ],
    },

    "salib": {
        "problems": "Which uncertain input actually governs the answer? Global sensitivity analysis "
            "apportions the variance of a model output to its inputs, so investigation and design effort "
            "go where they matter.",
        "methods": [
            "**Sobol** variance-based indices (first-order and total) and **Morris** elementary-effects "
            "screening, wrapping **SALib**; sample generation and index computation.",
            "References: Sobol; Morris; the SALib library. Requires the <code>[salib]</code> extra.",
        ],
        "limits": [],
    },

    "pystra": {
        "problems": "Structural reliability for an explicit limit-state function: the reliability index and "
            "probability of failure by FORM, SORM, or Monte Carlo, with a full distribution library and "
            "correlation.",
        "methods": [
            "**FORM / SORM / Monte Carlo** on a user limit-state function, wrapping **pystra** (the Python "
            "port of FERUM).",
            "References: the pystra library. Requires the <code>[pystra]</code> extra.",
        ],
        "limits": [
            "Complements the geotechnical <code>reliability</code> engine; use pystra for a general "
            "structural limit state and <code>reliability</code> for the geotechnical wrappers + COV "
            "database.",
        ],
    },

    "dxf_import": {
        "problems": "Start an analysis from a real CAD cross-section: read a DXF, discover its layers, and "
            "build the slope-stability or FEM geometry the analysis modules consume.",
        "methods": [
            "Layer discovery; polyline parsing into ground surface, soil boundaries, water table, and "
            "reinforcement; conversion to <code>SlopeGeometry</code> and FEM inputs.",
            "References: the ezdxf library. Requires the <code>[dxf]</code> extra.",
        ],
        "limits": [
            "Workflow is discover&nbsp;&rarr;&nbsp;parse&nbsp;&rarr;&nbsp;build; the layer&rarr;role "
            "mapping is an explicit step so the engineer confirms what each layer represents.",
        ],
    },

    "pdf_import": {
        "problems": "Extract a cross-section from a PDF drawing when there is no CAD file: vector geometry "
            "from the PDF&rsquo;s line work, or a vision read of the rendered page, cross-checked against "
            "each other.",
        "methods": [
            "**Vector extraction** (PyMuPDF <code>get_drawings()</code>) for exact line work; **vision "
            "extraction** (an LLM reads the rendered page).",
            "Scale calibration from a known dimension, label&rarr;region association, geometry cleanup, a "
            "vision grid overlay, and a vision&harr;vector cross-check.",
            "References: the PyMuPDF library. Requires the <code>[pdf]</code> extra.",
        ],
        "limits": [
            "Vision extraction depends on the engine&rsquo;s image quality; always confirm the extracted "
            "geometry against the drawing before analysis.",
        ],
    },

    "dxf_export": {
        "problems": "Send a cross-section built or checked in the toolkit back to CAD: write the surface, "
            "soil boundaries, water table, reinforcement, and annotations to a DXF.",
        "methods": [
            "Geometry-to-DXF export of surface, boundaries, GWT, nails/anchors, and text annotations.",
            "References: the ezdxf library. Requires the <code>[dxf]</code> extra.",
        ],
        "limits": [],
    },

    "calc_package": {
        "problems": "Produce a reviewable calculation package from a module result: a Mathcad-style "
            "report with inputs, equations, figures, and results &mdash; the documentation a calc set "
            "needs.",
        "methods": [
            "Jinja2-templated **HTML** and **LaTeX** calc packages for about a dozen analysis modules, "
            "with matplotlib figures and (optionally) Plotly interactive viewers; PDF where "
            "<code>pdflatex</code> is available.",
            "References: uses the <code>[calc]</code> extra (Jinja2), plus <code>[plot]</code> and "
            "<code>[interactive]</code> for figures.",
        ],
        "limits": [
            "PDF output needs <code>pdflatex</code>; on Databricks (no LaTeX) emit self-contained HTML and "
            "print it to PDF from the browser.",
        ],
    },
}


# ===========================================================================
# Validation basis (Chapter 8). Summary + a representative row set.
# ===========================================================================

VALIDATION_KPIS = [
    ("42", "published problems checked"),
    ("75", "individual module-vs-source comparisons"),
    ("53", "PASS (within stated tolerance)"),
    ("0", "unexplained discrepancies"),
]

VALIDATION_INTRO = (
    "Beyond the per-module unit suites (which regress each method against textbook answers), "
    "the package carries a <em>published-example</em> validation layer: real worked examples "
    "from the design manuals and FE verification suites, with complete numeric inputs "
    "<em>and</em> published numeric answers, run through the analysis modules offline. The "
    "problems (V-001&hellip;V-040) and their sources are in <code>validation_examples/INVENTORY.md</code>; "
    "the run-by-run verdicts are in <code>validation_examples/RESULTS.md</code>. Flagship "
    "modules additionally carry a <code>VALIDATION.md</code> (slope_stability vs Fredlund &amp; "
    "Krahn 1977 / ACADS / Duncan; fem2d vs Griffiths &amp; Lane and the Prandtl solution; "
    "lateral_pile vs a COM624P oracle). Verdicts are graded honestly: <strong>PASS</strong> "
    "(reproduces the published answer within tolerance), <strong>CONVENTION</strong> (a "
    "defensibly different but documented convention &mdash; the module is not tuned to the one "
    "example), and <strong>N/A&nbsp;(scope)</strong> (the module implements a different method "
    "than the example &mdash; a documented coverage gap, not a failure). Across 75 comparisons "
    "there were <strong>zero unexplained discrepancies</strong>."
)

# (id, module, problem, source, verdict)
VALIDATION_ROWS = [
    ("V-001", "axial_pile", "Nordlund driven H-pile capacity vs depth", "GEC-12 (FHWA-NHI-16-064)", "PASS"),
    ("V-002", "axial_pile", "Tomlinson &alpha;-method H-pile in layered clay", "GEC-12", "PASS / CONVENTION"),
    ("V-004", "downdrag", "Fellenius neutral plane &amp; drag force", "GEC-12", "PASS"),
    ("V-005", "pile_group", "Meyerhof (1976) SPT group settlement", "GEC-12 Table D-23", "PASS"),
    ("V-006", "drilled_shaft", "Rational &beta; side resistance (sand, OCR/Ko chain)", "GEC-10 Appendix A", "PASS"),
    ("V-007", "drilled_shaft", "Rational &alpha; side resistance (clay, Chen-2011)", "GEC-10 Appendix A", "PASS"),
    ("V-008", "drilled_shaft", "Base resistance in sand (0.60&middot;N<sub>60</sub>)", "GEC-10 Appendix A", "PASS"),
    ("V-009", "retaining_walls", "MSE external stability &mdash; LRFD sliding/ecc/bearing CDRs", "GEC-11 Ex E4", "PASS"),
    ("V-010", "retaining_walls", "MSE internal stability &mdash; K<sub>r</sub>, T<sub>max</sub>, bar-mat pullout", "GEC-11 Ex E4", "PASS"),
    ("V-011", "seismic_geotech", "Mononobe&ndash;Okabe K<sub>AE</sub> + seismic sliding CDR", "GEC-11 Ex E7", "PASS"),
    ("V-012", "soe", "Cantilever soldier-pile wall coefficients", "Caltrans T&amp;S Ex 7-1B", "PASS / N/A"),
    ("V-013", "sheet_pile", "Single-anchor wall &mdash; log-spiral K<sub>p</sub>, FHWA apparent diagram", "Caltrans T&amp;S Ex 8-1", "PASS"),
    ("V-014", "soe", "Basal heave &mdash; Caltrans force balance w/ side shear", "Caltrans T&amp;S Ex 10-2", "PASS"),
    ("V-015", "slope_stability", "Fellenius &amp; Bishop FOS on a specified circle", "Caltrans T&amp;S Ex 10-3/4", "PASS"),
    ("V-016", "soe", "Two-tier anchored wall &mdash; apparent envelope, anchor loads", "GEC-4 Ex 1", "PASS / CONVENTION"),
    ("V-017", "lateral_pile", "Laterally loaded micropile (LPILE benchmark)", "FHWA Micropile Ex 2", "PASS / CONVENTION"),
    ("V-018", "ground_improvement", "Rammed aggregate pier settlement + baseline consolidation", "GEC-13 Ex 1", "PASS / N/A"),
    ("V-019", "ground_improvement", "Stone-column settlement (Priebe factor)", "GEC-13 Ex 2", "PASS / CONVENTION"),
    ("V-020", "ground_improvement", "PVD time to 90% consolidation (Barron&ndash;Hansbo)", "GEC-13 Ch 2", "PASS"),
    ("V-021", "bearing_capacity", "Spread-footing ultimate bearing (Vesic factors, GW)", "GEC-6 Ex B-1", "PASS / CONVENTION"),
    ("V-022", "settlement", "Hough granular settlement table", "GEC-6 Ex B-1", "PASS"),
    ("V-023", "fem2d", "1-D Terzaghi/Biot consolidation (monolithic u-p)", "Itasca FLAC verification", "PASS"),
    ("V-024", "fem2d", "Cylindrical hole in Mohr&ndash;Coulomb (Salencon)", "Itasca FLAC verification", "PASS"),
    ("V-025", "sheet_pile", "Layered active pressure with water", "Caltrans T&amp;S Ex 7-2", "PASS"),
    ("V-026", "slope_stability", "Tension crack + water (ACADS 1b / Slide2 #2)", "Giam &amp; Donald 1989", "CONVENTION"),
    ("V-030", "slope_stability", "Duncan (2000) LASH underwater slope + reliability", "Duncan 2000 / Slide2 #29", "PASS"),
    ("V-031", "slope_stability", "Two-layer slope with water, circular", "Pockoski &amp; Duncan / Slide2 #57", "PASS"),
    ("V-032", "slope_stability", "Homogeneous Mohr&ndash;Coulomb", "Baker 2003 / Slide2 #61", "PASS"),
    ("V-033", "slope_stability", "Pseudo-static critical seismic coefficient", "Loukidis 2003 / Slide2 #62", "PASS"),
    ("V-035", "slope_stability", "Infinite-slope FOS (cohesionless)", "Duncan&ndash;Wright / Slide2 #79", "PASS"),
    ("V-037", "slope_stability", "Rapid drawdown &mdash; USACE 2-stage", "EM 1110-2-1902 / Slide2 #95", "PASS"),
    ("V-039", "slope_stability", "Newmark seismic sliding block + Jibson (2007)", "Slide2 #104 / Tutorial 28", "PASS"),
    ("V-040", "slope_stability", "Stabilizing micro-piles (Ito &amp; Matsui 1975)", "Yamagami 2000 / Slide2 #54", "CONVENTION"),
]

# Version history (Appendix B)
VERSION_HISTORY = [
    ("5.3.0", "2026-07-06",
     "Batch-2 coverage (drilled-shaft rational GEC-10 chains; MSE LRFD external stability; "
     "soe basal-heave side shear + FHWA apparent-pressure anchored walls + log-spiral Caquot&ndash;"
     "Kerisel K<sub>p</sub>; full Reese-1974 sand p-y; fem2d monolithic Taylor&ndash;Hood u-p Biot "
     "consolidation). slope_stability round 2 (15 new Slide2/ACADS/Duncan problems V-026&hellip;V-040; "
     "rapid drawdown; Newmark + Jibson; infinite slope; Ito&ndash;Matsui stabilizing piles). "
     "pdf_import round 2 (scale calibration, label&rarr;region, vision&harr;vector cross-check)."),
    ("5.2.0", "2026-06",
     "Additive coverage batch 1: settlement Hough method, pile_group Meyerhof group settlement, "
     "axial_pile per-layer toe friction angle + head depth, retaining_walls MSE bar-mat Kr/F* curves."),
    ("5.1.0", "2026-06",
     "Adapter-ergonomics sweep (allowed_values rollout, param-name and smart-method resolution), "
     "reference consult sub-agent + figure vision read-off, calc-QC round 3, published-example "
     "validation layer (Phase E), lexical query-expansion retrieval."),
    ("5.0.0", "2026-06-09",
     "First deepagents-based agent (funhouse_agent.deep, opt-in via the [deep] extra); the stable "
     "v1 GeotechAgent remains the default import."),
]


# ===========================================================================
# render() — builds the whole document through the context proxy
# ===========================================================================

def render(ctx):
    _ch1_what(ctx)
    _ch2_install(ctx)
    _ch3_agent(ctx)
    _ch4_catalog(ctx)
    _ch5_references(ctx)
    _ch6_geometry(ctx)
    _ch7_reliability(ctx)
    _ch8_validation(ctx)
    _ch9_limitations(ctx)
    _appendix_reference(ctx)
    _appendix_history(ctx)


# ---- Chapter 1 -------------------------------------------------------------

def _ch1_what(ctx):
    ctx.chapter("What this package is")
    ctx.para(
        "Geotechnical engineering is the practice of building <strong>on and in the ground</strong> "
        "&mdash; foundations, retaining walls, slopes, excavations, embankments. Unlike a steel beam "
        "with a certified strength, the geotechnical engineer&rsquo;s material is the earth itself: "
        "heterogeneous, layered, partly saturated, and sampled at only a handful of points across an "
        "entire site. Because the ground is variable and only partly known, <strong>a single number "
        "is never the answer</strong>.", cls="lead")
    ctx.para(
        "Understanding a geotechnical problem means understanding how the answer moves as the inputs "
        "move. Analysis is repeated calculation across plausible subsurface and loading conditions "
        "&mdash; running a chained assortment of industry-standard formulas, empirical correlations, "
        "and numerical methods, comparing against the design requirements, and varying assumptions "
        "until the design is robust across the uncertainty. The engineer&rsquo;s job is to understand "
        "the <strong>range and spread</strong> of answers, not one point estimate. The true answer is "
        "a distribution.")

    ctx.section("What a geotechnical engineer actually does")
    ctx.raw(
        "<table><thead><tr><th>Step</th><th>Activity</th><th>Reality</th></tr></thead><tbody>"
        "<tr><td><strong>Characterize</strong></td><td>Drill borings, push CPT cones, run lab tests</td>"
        "<td>A sparse, noisy picture of strength, stiffness, and groundwater &mdash; never the whole truth</td></tr>"
        "<tr><td><strong>Idealize</strong></td><td>Collapse the data into a layered soil profile</td>"
        "<td>Design values of &phi;, c, &gamma;, water table &mdash; each an estimate with a spread</td></tr>"
        "<tr><td><strong>Analyze</strong></td><td>Run the method (bearing, settlement, pile, slope)</td>"
        "<td>A chained assortment of standard formulas and numerical methods</td></tr>"
        "<tr><td><strong>Check &amp; revise</strong></td><td>Compare to requirements; vary assumptions; re-run</td>"
        "<td>Loop until the design is robust across the uncertainty</td></tr>"
        "</tbody></table>")
    ctx.para(
        "Steps 3 and 4 are repeated calculations across plausible conditions. This toolkit packages "
        "those methods as clean, machine-callable Python and puts a reasoning agent on top so the "
        "engineer can explore more of the problem in the time they have.")

    ctx.section("The lineage: every tool amplified the engineer")
    ctx.para(
        "Slide rules and design charts &rarr; spreadsheets and FEM &rarr; scripting and Monte Carlo "
        "&rarr; <strong>LLM agents</strong>. Every tool in this lineage did the same thing: it let one "
        "engineer explore more of the problem. This project is the next step &mdash; <em>not a "
        "replacement for judgment, but a multiplier for it.</em> Take the methods a staff engineer "
        "uses, make every one a clean function that returns structured data, wrap the variability "
        "tooling around them, and drive them with an agent that reads the standards back to you.")

    ctx.section("The architecture in four layers")
    ctx.para("One stack, layered so each concern stays independent:")
    ctx.raw(
        "<table><thead><tr><th>Layer</th><th>What it is</th><th>Why it&rsquo;s separate</th></tr></thead><tbody>"
        "<tr><td><strong>Agent harness</strong></td><td>An engine-agnostic LLM driver "
        "(<code>funhouse_agent</code>) that discovers modules, calls methods, reads figures, and cites "
        "references</td><td>The reasoning that <em>combines</em> methods lives here, not inside the "
        "methods</td></tr>"
        "<tr><td><strong>Variability engine</strong></td><td><code>reliability</code> &mdash; FOSM, "
        "PEM, Monte Carlo, FORM &rarr; &beta; and P<sub>f</sub>, plus a published COV database and spatial "
        "averaging</td><td>Orthogonal: it wraps <em>any</em> deterministic method as a callable and "
        "characterizes how its output scatters</td></tr>"
        "<tr><td><strong>Deterministic analysis</strong></td><td>The analysis modules &mdash; "
        "dataclass in, dataclass out, one method per engineering problem</td><td>Independent, testable "
        "leaves; no analysis module imports another</td></tr>"
        "<tr><td><strong>Shared spine + references</strong></td><td><code>geotech_common</code>&rsquo;s "
        "<code>SoilProfile</code> that every module speaks, and the digitized reference library</td>"
        "<td>One way to describe the ground; one place to cite the standards</td></tr>"
        "</tbody></table>")
    ctx.callout("The design rule that makes outputs citable",
        "<p>Everything is SI (m, kPa, kN, kN/m, degrees). Every <code>analyze_*()</code> returns a "
        "dataclass with <code>.summary()</code> for human/agent reading and <code>.to_dict()</code> for "
        "JSON. Analysis modules never import each other &mdash; the routing that combines them lives in "
        "the agent layer. That discipline is what lets a result be traced, reviewed, and cited.</p>")


# ---- Chapter 2 -------------------------------------------------------------

def _ch2_install(ctx):
    ctx.chapter("Installation &amp; environments")
    ctx.section("Install from PyPI")
    ctx.raw("<pre class='call'><code># core: numpy + scipy + the digitized reference library\n"
            "pip install geotech-staff-engineer\n\n"
            "# everything (all optional analysis backends + the deep agent)\n"
            "pip install \"geotech-staff-engineer[deep,full]\"</code></pre>")
    ctx.para(
        "The core install already covers the native modules &mdash; bearing capacity, settlement, piles, "
        "walls, slope stability, the reliability engine, fem2d, and the reference library "
        "(<code>geotech-references</code> is a hard dependency, so DM7/GEC/UFC lookups work out of the "
        "box). The optional <em>extras</em> pull in the third-party libraries that the wrapper modules "
        "drive. Install only what you need, or take <code>[full]</code>.")

    ctx.section("The extras groups")
    rows = ["<table><thead><tr><th>Extra</th><th>Pulls in</th><th>Unlocks</th></tr></thead><tbody>"]
    unlocks = {
        "plot": "matplotlib figures in calc packages and module plots",
        "calc": "Jinja2-templated HTML/LaTeX calc packages",
        "interactive": "Plotly single-file interactive viewers",
        "pdf": "PDF cross-section import (vector + vision)",
        "dxf": "DXF CAD import and export",
        "groundhog": "the groundhog correlation library",
        "opensees": "OpenSeesPy nonlinear FE (PM4Sand, 1D site response)",
        "pystrata": "equivalent-linear 1D site response",
        "seismic-signals": "eqsig + pyrotd ground-motion processing",
        "liquepy": "Boulanger &amp; Idriss liquepy triggering",
        "hvsrpy": "HVSR ambient-noise site characterization",
        "gstools": "geostatistical kriging and random fields",
        "salib": "Sobol / Morris sensitivity analysis",
        "swprocess": "MASW surface-wave dispersion",
        "pystra": "structural FORM/SORM/Monte-Carlo reliability",
        "subsurface": "GEF/BRO-XML, AGS4, and DIGGS data I/O",
        "deep": "the deepagents-based v2 agent (<code>funhouse_agent.deep</code>)",
        "full": "all of the analysis backends above",
    }
    for name, libs in ctx.extras:
        rows.append(f"<tr><td><code>{name}</code></td><td class='small'>{libs}</td>"
                    f"<td>{unlocks.get(name,'&mdash;')}</td></tr>")
    rows.append("</tbody></table>")
    ctx.raw("".join(rows))
    ctx.callout("Honest failure when an extra is missing",
        "<p>Without <code>[full]</code>, the roughly one-in-six methods that need an optional backend "
        "fail cleanly with a &ldquo;not installed&rdquo; message rather than a confusing traceback. "
        "Install the matching extra to enable them.</p>")

    ctx.section("Local Python")
    ctx.para(
        "Requires Python 3.10+. The package is pure Python; the only heavy dependencies are numpy and "
        "scipy. For the LLM agent you need an engine (see the next chapter) and, for a Claude engine, "
        "an <code>ANTHROPIC_API_KEY</code> in the environment. Keys are read from the environment at "
        "run time &mdash; never pass them through a notebook cell or a chat transcript.")

    ctx.section("Databricks / Funhouse")
    ctx.para(
        "Install from a cluster-accessible location &mdash; <code>/tmp</code> or a Unity Catalog "
        "Volume, <em>not</em> <code>/Workspace</code> (the workspace FUSE mount mangles wheel filenames "
        "and does not durably store output files):")
    ctx.raw("<pre class='call'><code>%pip install \"geotech-staff-engineer[deep,full]\"\n"
            "# or a test wheel uploaded to /tmp or a UC Volume:\n"
            "%pip install \"/tmp/geotech_staff_engineer-<ver>-py3-none-any.whl[deep,full]\"</code></pre>")
    ctx.callout("The typing_extensions / restart-Python note (now handled automatically)",
        "<p>Databricks cluster runtimes pre-import an old <code>typing_extensions</code> (&lt;4.13) at "
        "kernel startup, which used to break the langgraph imports behind the <code>[deep]</code> agent "
        "with <code>TypeError: TypedDict() got an unexpected keyword argument 'extra_items'</code>, "
        "requiring <code>dbutils.library.restartPython()</code>. As of v5.4 the package repairs this in "
        "place: <code>funhouse_agent.runtime_check.ensure_typing_extensions()</code> runs automatically "
        "at the top of the first <code>funhouse_agent.deep</code> import, reloading the freshly-installed "
        "<code>typing_extensions&ge;4.13</code> so the restart is normally no longer needed. If an even "
        "older copy is pinned at cluster scope and the in-place reload can&rsquo;t win, it raises a clear "
        "message that falls back to the restart instruction.</p>")
    ctx.para(
        "Save calc packages, DXF, and plots to <code>/tmp</code> or a UC Volume and copy them out with "
        "<code>dbutils.fs.cp</code>; writes to <code>/Workspace</code> can silently leave an 11-byte "
        "placeholder. As of recent releases the file tools defend themselves &mdash; they verify the "
        "written bytes and return a <code>rescue_path</code> in <code>/tmp</code> if the target did not "
        "store the content.")

    ctx.section("Reference figures: GEOTECH_REFERENCES_DOCS")
    ctx.para(
        "The reference figure catalogs (captions, page numbers, cross-links) ship inside the wheel, so "
        "<code>figure_search</code> works from a clean install. The source PDF <em>pages</em> are not "
        "shipped (large + license). To enable the vision read-off &mdash; where the agent renders a "
        "catalogued chart and reads a value off it &mdash; place the reference PDFs in a folder and set "
        "<code>GEOTECH_REFERENCES_DOCS</code> to point at it. On Databricks, copy a <code>docs/</code> "
        "folder of PDFs in and set the variable.")


# ---- Chapter 3 -------------------------------------------------------------

def _ch3_agent(ctx):
    ctx.chapter("Using the agent")
    ctx.para(
        "The same validated methods are reachable at three altitudes: import a module and get a "
        "dataclass back; wrap that method to return the distribution; or describe the problem in "
        "natural language and let the agent run the methods and cite the standards. This chapter is "
        "about the third altitude &mdash; the agent.", cls="lead")

    ctx.section("The v1 agent: GeotechAgent")
    ctx.para(
        "<code>GeotechAgent</code> is engine-agnostic: it drives any backend that satisfies the small "
        "<code>GenAIEngine</code> protocol (a <code>chat()</code> method, optionally "
        "<code>analyze_image()</code> for vision). Ship it a Claude engine, a Funhouse "
        "<code>PrompterAPI</code>, or a native OpenAI-tool-calling engine.")
    ctx.raw("<pre class='call'><code>from funhouse_agent import GeotechAgent, ClaudeEngine, NativeToolEngine\n\n"
            "# Claude (reads ANTHROPIC_API_KEY from the environment)\n"
            "agent = GeotechAgent(genai_engine=ClaudeEngine())\n\n"
            "# Databricks / Funhouse with native OpenAI tool calling (recommended there)\n"
            "agent = GeotechAgent(genai_engine=NativeToolEngine(fh_prompter))\n\n"
            "result = agent.ask(\"2 m strip footing, 1.5 m deep, sand phi=30, c=10, water at 5 m. \"\n"
            "                   \"Bearing capacity and FOS - and how sensitive is it to phi? Cite the method.\")\n"
            "print(result.answer)        # the cited answer\n"
            "print(result.tool_calls)    # every method the agent ran\n"
            "print(result.rounds)        # ReAct iterations</code></pre>")

    ctx.section("How tool dispatch works")
    ctx.para(
        "The agent never has to know a method signature in advance. It discovers the toolbox with four "
        "dispatch tools, then calls the one it needs:")
    ctx.raw(
        "<table><thead><tr><th>Tool</th><th>What it returns</th></tr></thead><tbody>"
        "<tr><td><code>list_agents()</code></td><td>Every available module with a one-line brief</td></tr>"
        "<tr><td><code>list_methods(agent, category)</code></td><td>The methods on a module, grouped by category</td></tr>"
        "<tr><td><code>describe_method(agent, method)</code></td><td>Full parameter documentation &mdash; names, types, required flags, allowed values</td></tr>"
        "<tr><td><code>call_agent(agent, method, params)</code></td><td>Execute the calculation; returns the result dict or a helpful error</td></tr>"
        "</tbody></table>")
    ctx.para(
        "The dispatch layer is forgiving by design: a method name the model guesses is resolved through "
        "a curated alias map; a value that belongs to a selector parameter (say "
        "<code>&#39;vesic&#39;</code>) is redirected to the right call with a directive error; and a "
        "genuinely unknown name gets a &ldquo;did you mean&rdquo; suggestion. This cut the "
        "method-name guessing that otherwise dogs tool-using agents.")

    ctx.section("References through a scoped consultant")
    ctx.para(
        "The primary agent does not call the 21 reference modules directly. Reference access is routed "
        "through a single <code>consult_references</code> tool backed by a reference-scoped sub-agent "
        "&mdash; a second <code>GeotechAgent</code> restricted to the reference modules. This keeps the "
        "primary agent&rsquo;s tool surface small (which measurably cut module-name guessing) while "
        "still giving it the full reference library, figure search, and vision read-off on demand. The "
        "behavior is set by <code>reference_mode</code>: <code>&#39;anytime&#39;</code> (default) offers "
        "the consultant always; <code>&#39;after_calc&#39;</code> only after a calculation has run; "
        "<code>&#39;off&#39;</code> keeps the legacy direct-call behavior.")

    ctx.section("Vision: reading drawings and charts")
    ctx.para(
        "When the engine supports images, the agent gains vision tools: <code>analyze_image</code> and "
        "<code>analyze_pdf_page</code> for site plans and report pages, and "
        "<code>read_reference_figure</code>, which finds a design chart by meaning "
        "(<code>figure_search</code>) and then renders the actual page and reads a value off it &mdash; "
        "rather than answering a chart value from memory. Attach files before asking:")
    ctx.raw("<pre class='call'><code>with open(\"cross_section.pdf\", \"rb\") as f:\n"
            "    agent.add_attachment(\"section\", f.read())\n"
            "result = agent.ask(\"Extract the slope geometry from 'section' and find the FOS by Bishop.\")\n\n"
            "# direct geometry extraction (bypasses the ReAct loop)\n"
            "geo = agent.extract_geometry_from_pdf(pdf_bytes, page=0)\n"
            "geo.surface_points, geo.boundary_profiles</code></pre>")

    ctx.section("The v2 deep agent and notebook chat")
    ctx.para(
        "Installing the <code>[deep]</code> extra unlocks <code>funhouse_agent.deep.build_deep_agent</code>, "
        "a deepagents/LangGraph-based loop with the same toolbox and an optional staged, human-gated "
        "model-setup sub-agent (off by default). For interactive use in Jupyter or Databricks, "
        "<code>NotebookChat</code> (v1) and <code>DeepNotebookChat</code> (v2) give an ipywidgets chat "
        "with file upload, collapsible tool calls, and inline calc-package preview:")
    ctx.raw("<pre class='call'><code>from funhouse_agent.notebook import NotebookChat\n"
            "chat = NotebookChat(agent)\n"
            "chat.display()\n"
            "chat.attach(\"/Volumes/proj/site_plan.png\")   # or the upload widget</code></pre>")
    ctx.callout("Treat agent output as a first draft",
        "<p>The agent can select the wrong method, mis-transcribe a value, or read a chart imperfectly. "
        "Run it with <code>verbose=True</code> or an <code>on_tool_call</code> callback so you can see "
        "every method and parameter it used, and review the tool calls &mdash; not just the prose &mdash; "
        "before relying on anything.</p>")


# ---- Chapter 4: THE PROBLEM CATALOG ---------------------------------------

def _ch4_catalog(ctx):
    ctx.chapter("The problem catalog")
    ctx.para(
        "This is the heart of the manual: one entry per analysis module, covering the engineering "
        "problems it solves, the methods and theories it implements (with their references), a worked "
        "example run through the real code, and its applicability limits. The method and parameter "
        "tables in this chapter are <strong>generated programmatically</strong> from the shipped "
        "dispatch registry, so they match the installed package exactly. Every method&rsquo;s full "
        "parameter list is in Appendix&nbsp;A.", cls="lead")
    st = ctx.stats
    ctx.raw(
        f"<div class='kpi'>"
        f"<div class='box'><div class='n'>{st['n_analysis']}</div><div class='l'>analysis modules</div></div>"
        f"<div class='box'><div class='n'>{st['n_methods']}</div><div class='l'>agent-callable methods</div></div>"
        f"<div class='box'><div class='n'>{st['n_params']}</div><div class='l'>documented parameters</div></div>"
        f"<div class='box'><div class='n'>{st['n_reference']}</div><div class='l'>digitized references</div></div>"
        f"</div>")

    for group_title, blurb, modules in MODULE_ORDER:
        ctx.section(group_title)
        ctx.para(blurb)
        for mod in modules:
            _module_entry(ctx, mod)

    # groundhog note
    ctx.section("Also available: groundhog correlations")
    ctx.para(
        "Beyond the dispatch registry, the package ships <code>groundhog_agent</code> (about 90 methods) "
        "wrapping the <code>groundhog</code> site-investigation and soil-mechanics library &mdash; SPT/CPT "
        "correlations, lab-test interpretation, and unit conversions. It is available with the "
        "<code>[groundhog]</code> extra.")


def _module_entry(ctx, mod):
    methods = ctx.module_methods(mod)
    if not methods:
        return
    title = MODULE_TITLES.get(mod, mod)
    is_ref = False
    ctx.subsection(f"<code>{mod}</code> &mdash; {title}")
    narr = MODULE_NARRATIVE.get(mod, {})

    # Problems it solves
    if narr.get("problems"):
        ctx.raw('<div class="field-label">Problems it solves</div>')
        ctx.para(narr["problems"], cls="mod-purpose")
    else:
        # fallback to the first method's brief / module brief
        ctx.raw('<div class="field-label">Problems it solves</div>')
        ctx.para(ctx.cat["modules"][mod].get("brief", "") or next(iter(methods.values()))["brief"],
                 cls="mod-purpose")

    # Methods & theory
    if narr.get("methods"):
        ctx.raw('<div class="field-label">Methods &amp; theories implemented</div>')
        ctx.bullets(narr["methods"])

    # programmatic method table
    ctx.raw('<div class="field-label">Agent-callable methods (from the shipped registry)</div>')
    ctx.methods_table(mod)

    # headline parameter table
    hm = HEADLINE_METHOD.get(mod)
    if hm and hm in methods:
        ctx.raw(f'<div class="field-label">Parameters of <code>{hm}</code></div>')
        ctx.params_table(mod, hm)

    # worked example
    if mod in WORKED:
        key, ask, note = WORKED[mod]
        ctx.worked(key, ask, note)

    # limits
    if narr.get("limits"):
        body = ctx.inline("")  # noop
        items = "".join(f"<li>{ctx.inline(x)}</li>" for x in narr["limits"])
        ctx.callout("Applicability limits &amp; conventions", f"<ul>{items}</ul>", kind="limit")

    ctx.raw('<hr class="soft">')


# ---- Chapter 5: references -------------------------------------------------

def _ch5_references(ctx):
    ctx.chapter("The reference library")
    ctx.para(
        "The agent doesn&rsquo;t just compute &mdash; it reads the standards back to you. "
        "<code>geotech-references</code> is a companion package of digitized design manuals: their "
        "tables and equations as callable lookups, their chapter text in a full-text search index, and "
        "their figures in a searchable catalog paired with a vision read-off. There are "
        f"{ctx.stats['n_reference']} reference modules.", cls="lead")

    ctx.section("The references")
    rows = ["<table><thead><tr><th>Module</th><th>Standard</th><th>What it covers</th></tr></thead><tbody>"]
    for mod in sorted(REFERENCE_INFO):
        title, std, cov = REFERENCE_INFO[mod]
        rows.append(f"<tr><td><code>{mod}</code><br><span class='small'>{title}</span></td>"
                    f"<td class='small'>{std}</td><td>{cov}</td></tr>")
    rows.append("</tbody></table>")
    ctx.raw("".join(rows))

    ctx.section("Find with text, read with pixels")
    ctx.para(
        "Retrieval is lexical: <code>reference_search</code> runs a full-text (SQLite FTS5) query over "
        "the chapter text and equation captions, expanded with a curated synonym map so a query for "
        "&ldquo;passive resistance&rdquo; also finds &ldquo;K<sub>p</sub>&rdquo;. That roughly quadrupled "
        "recall@5 in the retrieval eval with no disturbance to the top hit. Figures are handled the same "
        "way for search, then differently for reading: <code>figure_search</code> finds a chart by "
        "meaning, and <code>read_reference_figure</code> renders the actual page and hands it to a vision "
        "model to read a value off the axes &mdash; because a design chart&rsquo;s value lives in its "
        "pixels, not its caption.")
    ctx.callout("Why not image embeddings?",
        "<p>CLIP-style image embeddings were evaluated and rejected &mdash; they are a poor fit for "
        "axis-labelled line charts. The architecture is deliberately &ldquo;find lexically, read "
        "visually.&rdquo; This keeps retrieval cheap and explainable while still getting an accurate "
        "value off the chart.</p>")
    ctx.para(
        "In agent workflows all of this is reached through the <code>consult_references</code> "
        "sub-agent described in Chapter&nbsp;3, so the primary agent stays focused on the analysis while "
        "a scoped consultant handles citations and figure read-offs.")


# ---- Chapter 6: geometry ---------------------------------------------------

def _ch6_geometry(ctx):
    ctx.chapter("Geometry import &amp; model setup")
    ctx.para(
        "Real analyses start from a real cross-section &mdash; a CAD file or a PDF drawing, not a set of "
        "hand-typed coordinates. Three modules get geometry in and a fourth gets a calc package out; a "
        "fifth orchestrates a staged, human-gated model build.", cls="lead")

    ctx.section("DXF import")
    ctx.para(
        "<code>dxf_import</code> reads a DXF, discovers its layers, and converts polylines into the "
        "<code>SlopeGeometry</code> and FEM inputs the analysis modules consume. The workflow is "
        "discover&nbsp;&rarr;&nbsp;parse&nbsp;&rarr;&nbsp;build: list the layers, map them to surface / "
        "soil boundaries / water table / reinforcement, then build the model. It needs the "
        "<code>[dxf]</code> extra (ezdxf).")

    ctx.section("PDF cross-section import")
    ctx.para(
        "<code>pdf_import</code> extracts geometry from a PDF drawing two ways: <strong>vector "
        "extraction</strong> (PyMuPDF <code>get_drawings()</code> for exact line work) and "
        "<strong>vision extraction</strong> (an LLM reads the rendered page). Round 2 added scale "
        "calibration from a known dimension, label&ndash;to&ndash;region association, geometry cleanup, "
        "a vision grid overlay, and a vision&harr;vector cross-check so the two methods can confirm each "
        "other. Needs the <code>[pdf]</code> extra.")

    ctx.section("DXF export")
    ctx.para(
        "<code>dxf_export</code> writes a cross-section back out to DXF &mdash; surface, soil boundaries, "
        "water table, nails/anchors, and annotations &mdash; so a model built or checked in the toolkit "
        "can return to CAD.")

    ctx.section("Staged, human-gated model setup (geo_project)")
    ctx.para(
        "<code>geo_project</code> is a canonical project document for building 2D LE/FEM models in "
        "stages with confirmation gates. The agent never trusts its own read of a drawing: it ingests "
        "geometry with provenance quarantine, then renders an <strong>echo-back</strong> cross-section "
        "so a human visually confirms the numbers before they enter the model. It is wired to the v2 "
        "deep agent as an opt-in model-setup sub-agent.")

    ctx.section("Calc-package reporting")
    ctx.para(
        "<code>calc_package</code> turns a module result into a Mathcad-style calculation package &mdash; "
        "HTML or LaTeX, with figures and equation traces &mdash; for about a dozen of the analysis "
        "modules. On a system with <code>pdflatex</code> it can emit PDF; on Databricks (no LaTeX), emit "
        "the self-contained HTML and print it to PDF from the browser. It needs the <code>[calc]</code> "
        "extra (Jinja2), plus <code>[plot]</code> for figures and <code>[interactive]</code> for the "
        "Plotly viewers.")


# ---- Chapter 7: reliability ------------------------------------------------

def _ch7_reliability(ctx):
    ctx.chapter("Reliability &amp; probabilistic analysis")
    ctx.para(
        "This is the payoff of the whole architecture. Soil properties carry coefficients of variation "
        "of 10&ndash;40% &mdash; far larger than structural materials &mdash; so the honest deliverable is "
        "not a lone factor of safety but a <strong>reliability index &beta;</strong> and a "
        "<strong>probability of failure P<sub>f</sub></strong>. The <code>reliability</code> module turns any "
        "deterministic method into a probabilistic one.", cls="lead")

    ctx.section("Declare the uncertainty, pick an engine")
    ctx.para(
        "Declare the uncertain inputs as random variables, hand the engine a performance function "
        "<code>g(values)</code> that returns a factor of safety or a margin, and pick an engine. All "
        "four share the same <code>g()</code> contract:")
    ctx.raw(
        "<table><thead><tr><th>Engine</th><th>Cost</th><th>What it does</th></tr></thead><tbody>"
        "<tr><td><strong>FOSM</strong></td><td>cheap</td><td>First-Order Second-Moment &mdash; propagates "
        "means and variances for an instant &beta;, no sampling</td></tr>"
        "<tr><td><strong>PEM</strong></td><td>robust</td><td>Rosenblueth point-estimate &mdash; evaluates "
        "g() at &plusmn; points; handles non-linearity without a derivative</td></tr>"
        "<tr><td><strong>Monte Carlo</strong></td><td>exact-ish</td><td>Samples the joint distribution N "
        "times and counts failures; correlations via Cholesky, optional Latin Hypercube</td></tr>"
        "<tr><td><strong>FORM</strong></td><td>efficient</td><td>Native first-order reliability &mdash; "
        "finds the design point; &beta; at a fraction of Monte Carlo&rsquo;s calls</td></tr>"
        "</tbody></table>")

    ctx.worked("reliability.monte_carlo",
               WORKED["reliability"][1], WORKED["reliability"][2])

    ctx.section("Don&rsquo;t assume a COV &mdash; look it up")
    ctx.para(
        "<code>cov_guidance</code> is a queryable knowledge base of published coefficients of variation "
        "(Duncan 2000, TC304, Phoon &amp; Kulhawy) keyed by soil property, so the input uncertainty is "
        "defensible rather than assumed. <code>variance_reduction</code> applies Vanmarcke spatial "
        "averaging &mdash; a footing averages over its footprint, a long slope samples many weak and "
        "strong zones, so the variance that actually matters is reduced over the volume the structure "
        "loads. <code>combined_cov</code> composes several sources of uncertainty.")

    ctx.section("Which input governs? &mdash; global sensitivity")
    ctx.para(
        "<code>salib</code> wraps SALib for variance-based global sensitivity: Sobol indices and Morris "
        "screening answer <em>which</em> input drives the spread of the answer, so the site investigation "
        "and the design effort go where they matter. <code>pystra</code> adds a dedicated structural "
        "reliability engine (FORM / SORM / Monte Carlo) for limit-state functions.")

    ctx.section("Probabilistic paths that ship out of the box")
    ctx.para(
        "The reliability wrappers &mdash; <code>bearing_capacity_reliability</code>, "
        "<code>axial_pile_reliability</code>, and <code>slope_reliability</code> &mdash; and the "
        "slope-stability module&rsquo;s built-in FOSM and Monte-Carlo FOS mean the probabilistic path is "
        "available directly, without hand-wiring the performance function for the most common problems.")


# ---- Chapter 8: validation -------------------------------------------------

def _ch8_validation(ctx):
    ctx.chapter("Validation basis")
    st_boxes = "".join(
        f"<div class='box'><div class='n'>{n}</div><div class='l'>{l}</div></div>"
        for n, l in VALIDATION_KPIS)
    ctx.raw(f"<div class='kpi'>{st_boxes}</div>")
    ctx.para(VALIDATION_INTRO)

    ctx.section("Published-example validation (V-001&hellip;V-040)")
    rows = ["<table><thead><tr><th>ID</th><th>Module</th><th>Problem</th><th>Source</th><th>Verdict</th></tr></thead><tbody>"]
    for vid, mod, prob, src, verd in VALIDATION_ROWS:
        rows.append(f"<tr><td>{vid}</td><td><code>{mod}</code></td><td>{prob}</td>"
                    f"<td class='small'>{src}</td><td>{verd}</td></tr>")
    rows.append("</tbody></table>")
    ctx.raw("".join(rows))
    ctx.para(
        "The table above is a representative selection; the complete run-by-run record &mdash; every "
        "delta, tolerance, and the reasoning behind each convention/scope verdict &mdash; is in "
        "<code>validation_examples/RESULTS.md</code>.", cls="small")

    ctx.section("How to read the verdicts")
    ctx.callout("PASS", "<p>The module reproduces the published answer within the stated tolerance. "
        "Example: the Mononobe&ndash;Okabe seismic coefficient K<sub>AE</sub> reproduces the GEC-11 value "
        "to within 0.06%; the Fellenius/Bishop factors of safety match the Caltrans slice table to the "
        "printed digit.</p>")
    ctx.callout("CONVENTION",
        "<p>The module and the example use a defensibly different convention, and the delta is documented "
        "&mdash; the module is <em>not</em> tuned to the one example. Example: for the GEC-6 spread "
        "footing, the packaged high-level call applies Vesic depth factors while the example deliberately "
        "sets them to 1.0 for cohesive overburden; assembled the example&rsquo;s way, the module recovers "
        "the published closed form exactly.</p>", kind="warn")
    ctx.callout("N/A (scope)",
        "<p>The module implements a different method than the example &mdash; a documented coverage gap, "
        "not a failure. Example: a slope problem whose pore pressure is supplied as an interpolated "
        "flow-net grid, which the module (piezometric surface or per-layer r<sub>u</sub>) does not accept.</p>",
        kind="limit")
    ctx.para(
        "The philosophy is visible in these notes: where the module and a source differ, the manual says "
        "why, in the source&rsquo;s own terms, rather than quietly fitting the code to one benchmark. "
        "Several checks even surfaced errors in the <em>published</em> examples (for instance an "
        "under-iterated Bishop value that the module iterates to convergence).")


# ---- Chapter 9: limitations ------------------------------------------------

def _ch9_limitations(ctx):
    ctx.chapter("Limitations &amp; responsible use")
    ctx.para(
        "Everything in the disclaimer at the front of this manual applies. This chapter makes the "
        "boundaries concrete, because knowing where a tool stops is part of using it well.", cls="lead")

    ctx.section("It is a multiplier for judgment, not a replacement")
    ctx.para(
        "The toolkit runs the methods; it does not know your site and cannot exercise engineering "
        "judgment. A qualified, licensed professional engineer familiar with the site must independently "
        "review every input, assumption, method selection, and result before it is relied upon. Using "
        "this software creates no engineer-of-record relationship.")

    ctx.section("Validation covers only the documented cases")
    ctx.para(
        "Agreement on a benchmark (Chapter&nbsp;8) does not certify correctness for any other geometry, "
        "parameter range, or loading condition. The absence of a validation entry means a result has not "
        "been independently checked here. Read the module&rsquo;s <code>DESIGN.md</code> and "
        "<code>VALIDATION.md</code> for its exact scope, sign conventions, and edge cases before "
        "trusting it on a new problem.")

    ctx.section("Known method boundaries")
    ctx.para("A few honest scope notes drawn straight from the validation record:")
    ctx.bullets([
        "Diesel-hammer drivability in <code>wave_equation</code> is an energy&ndash;velocity "
        "approximation, not a GRLWEAP combustion + ram-cycle model; treat diesel bearing graphs as a "
        "trend, not a tie-out.",
        "<code>slope_stability</code> accepts a piezometric surface or per-layer r<sub>u</sub>, not an "
        "arbitrary interpolated pore-pressure grid; its tension crack is entry-side and keeps the cracked "
        "wedge as driving soil (a conservative convention).",
        "<code>ground_improvement</code> uses an equal-strain settlement-reduction model for aggregate "
        "piers, not the GEC-13 stiffness-modulus two-layer method; the Priebe factor depends on the "
        "area-ratio and column-friction conventions you feed it.",
        "The high-level <code>bearing_capacity</code> call applies Vesic depth factors and an "
        "effective-unit-weight groundwater model &mdash; defensible, but different from an AASHTO "
        "C<sub>w</sub> groundwater correction; know which convention your check needs.",
        "Soldier-pile arching and the FHWA apparent-diagram single-anchor embedment solver are documented "
        "gaps in <code>soe</code>/<code>sheet_pile</code>.",
    ])

    ctx.section("Treat agent output with particular care")
    ctx.para(
        "The LLM agent can select the wrong method, mis-transcribe a value, or read a chart imperfectly. "
        "Its answers are a starting point for review, never a final basis for design. Inspect the tool "
        "calls (<code>result.tool_calls</code>, or <code>verbose=True</code>) &mdash; the methods and "
        "parameters actually used &mdash; not just the prose. Confirm the inputs and units: everything is "
        "SI, and mis-scaled units are the user&rsquo;s responsibility to catch.")

    ctx.callout("The bottom line",
        "<p>Responsibility for the safety, adequacy, and code-compliance of any design remains entirely "
        "with the licensed professional engineer who adopts it. This software is provided under the MIT "
        "License &ldquo;AS IS&rdquo;, without warranty of any kind.</p>", kind="limit")


# ---- Appendix A: full method reference ------------------------------------

def _appendix_reference(ctx):
    ctx.chapter("Full method &amp; parameter reference", appendix=True)
    ctx.para(
        "Every agent-callable method on every module, with its full parameter list, generated directly "
        "from the shipped dispatch registry. This is the authoritative, complete catalog &mdash; the "
        "narrative chapters show one headline method per module; here is all of them.", cls="lead")

    cat = ctx.cat
    # analysis modules first (in catalog group order), then references
    seen = set()
    ordered = []
    for _, _, mods in MODULE_ORDER:
        for m in mods:
            if m in cat["modules"]:
                ordered.append(m); seen.add(m)
    # any analysis modules not in a group
    for m in sorted(cat["modules"]):
        if m in seen:
            continue
        if m in REFERENCE_INFO:
            continue
        ordered.append(m); seen.add(m)

    ctx.section("Analysis modules")
    for m in ordered:
        _appendix_module(ctx, m)

    ctx.section("Reference modules")
    ctx.para(
        "The reference modules expose lookups for tables, equations, and figures plus text/figure search. "
        "Method counts are shown; the individual lookup signatures are discoverable at run time with "
        "<code>describe_method</code>.")
    rows = ["<table><thead><tr><th>Module</th><th>Standard</th><th>Methods</th></tr></thead><tbody>"]
    for m in sorted(REFERENCE_INFO):
        n = len(cat["modules"].get(m, {}).get("methods", {}))
        title = REFERENCE_INFO[m][0]
        rows.append(f"<tr><td><code>{m}</code></td><td class='small'>{title}</td><td>{n}</td></tr>")
    rows.append("</tbody></table>")
    ctx.raw("".join(rows))


def _appendix_module(ctx, m):
    methods = ctx.module_methods(m)
    if not methods:
        return
    title = MODULE_TITLES.get(m, m)
    ctx.subsection(f"<code>{m}</code> &mdash; {title}", monospace=True)
    for meth in methods:
        ctx.raw(f'<h4><code>{meth}</code> &mdash; {ctx.inline(methods[meth].get("brief",""))}</h4>')
        ctx.params_table(m, meth)


# ---- Appendix B: version history ------------------------------------------

def _appendix_history(ctx):
    ctx.chapter("Version history", appendix=True)
    ctx.para(
        "A summary of the 5.x line. The full running history and the intentional behavior changes "
        "between minor versions are tracked in <code>CLAUDE.md</code> and "
        "<code>docs/V5.1_SUMMARY.html</code>.", cls="lead")
    rows = ["<table><thead><tr><th>Version</th><th>Date</th><th>Highlights</th></tr></thead><tbody>"]
    for ver, dt, note in VERSION_HISTORY:
        rows.append(f"<tr><td><strong>{ver}</strong></td><td class='small'>{dt}</td><td>{note}</td></tr>")
    rows.append("</tbody></table>")
    ctx.raw("".join(rows))
