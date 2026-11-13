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
      "seismic_signals"]),
    ("Site characterization &amp; geostatistics",
     "Turning borings, cones, and lab data into a modelled ground &mdash; parse, visualize, "
     "correlate, and interpolate.",
     ["subsurface", "gstools"]),
    ("Reliability &amp; sensitivity",
     "The variability engine: propagate parameter uncertainty to a reliability index and a "
     "probability of failure, and find which input governs.",
     ["reliability", "salib", "pystra"]),
    ("Structural analysis",
     "Cross-section properties, reinforced-concrete section capacity, and elastic "
     "frame / continuous-beam analysis.",
     ["section_props", "concrete_props", "pynite"]),
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
    "subsurface": "Subsurface data I/O (DIGGS / GEF / AGS4)",
    "gstools": "Geostatistics (kriging, random fields &amp; conditional simulation)",
    "reliability": "Reliability engine",
    "salib": "Global sensitivity analysis (SALib)",
    "pystra": "Structural reliability (pystra)",
    "section_props": "Cross-section properties",
    "concrete_props": "RC section analysis",
    "pynite": "Frame &amp; beam analysis (PyNite)",
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
    "liquepy": None,
    "section_props": None,
    "concrete_props": None,
    "pynite": None,
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
        "For a 20&nbsp;m steel pile driven by a Vulcan 014 hammer, plot the bearing "
        "graph &mdash; blows per metre against static resistance.",
        "With no cushion given, the Vulcan 014 takes its hammer cushion and helmet as GEC-12 "
        "Vol 2 Table 12-2 prints them (p. 183); any other hammer needs the cushion, its COR "
        "and the helmet weight given, or <code>driving_system</code>."),
    "sheet_pile": ("sheet_pile.cantilever_wall",
        "A cantilever sheet pile retains a 4&nbsp;m excavation in sand (&phi;=32&deg;) with a "
        "10&nbsp;kPa surcharge. What embedment is needed and what is the maximum moment?", ""),
    "soe": ("soe.braced_excavation",
        "An 8&nbsp;m braced cut in sand with two strut levels &mdash; give the apparent-pressure "
        "diagram, strut loads, and the required wall section.", ""),
    "retaining_walls": ("retaining_walls.cantilever_wall",
        "A 6&nbsp;m cantilever retaining wall on &phi;=32&deg; backfill with a 12&nbsp;kPa surcharge "
        "&mdash; check sliding, overturning, and bearing.",
        "Here the auto-sized base fails sliding (FOS 1.14 &lt; 1.5) &mdash; exactly the kind of "
        "check-and-revise loop the toolkit is built for."),
    "ground_improvement": ("ground_improvement.aggregate_piers",
        "0.76&nbsp;m aggregate piers on a 1.5&nbsp;m triangular grid under soft ground &mdash; what "
        "settlement reduction and bearing improvement do they buy?", ""),
    "slope_stability": ("slope_stability.search_critical_surface",
        "Find the critical circular slip surface and minimum factor of safety for a 10&nbsp;m "
        "c&ndash;&phi; slope (c&prime;=12&nbsp;kPa, &phi;&prime;=28&deg;) by Bishop&rsquo;s method.",
        "2,300 trial surfaces were evaluated: the entry&ndash;exit arcs, then small circles through "
        "the toe and circles a little above it. A toe circle governs (FOS 1.583); the "
        "entry&ndash;exit arcs&rsquo; minimum is 1.587."),
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

# The reference modules: agent -> (title, designation and date as printed on
# the cover, coverage). Designations read off each PDF's cover in
# geotech-references/docs (DOC-23, module_work/doc23/CHECK.md).
REFERENCE_INFO = {
    "dm7": ("DM 7.1 / DM 7.2",
            "UFS 3-220-10, Soil Mechanics (DM 7.1), 16 January 2026; UFC 3-220-20, Foundations "
            "and Earth Structures (DM 7.2), 16 January 2025",
            "The successors to NAVFAC DM 7.01 and DM 7.02 &mdash; soil classification, field and "
            "laboratory testing, stress distribution, consolidation and settlement, seepage, slope "
            "stability, correlations, earth pressures, shallow and deep foundations and retaining "
            "structures. About 390 equation, table and figure lookups, plus chapter text and figure "
            "catalogs."),
    "gec4": ("FHWA GEC-4", "FHWA-IF-99-015, June 1999",
             "Ground anchors and anchored systems &mdash; chapter text and figures, with lookups for "
             "anchor bond stress and load transfer (Tables 6 to 8) and corrosion protection (Table 20)."),
    "gec5": ("FHWA GEC-5", "FHWA NHI-16-072, April 2017",
             "Geotechnical site characterization &mdash; chapter text and figure catalog."),
    "gec6": ("FHWA GEC-6", "FHWA-SA-02-054, September 2002",
             "Shallow foundations &mdash; bearing-capacity factors, rock properties, presumptive "
             "bearing, the Hough bearing-capacity index (Fig. 5-19), chapter text and figures."),
    "gec7": ("FHWA GEC-7", "FHWA-NHI-14-007, February 2015",
             "Soil nail walls &mdash; bond strength, resistance factors, seismic coefficients, "
             "chapter text and figures."),
    "gec8": ("FHWA GEC-8", "FHWA-HIF-07-03, April 2007",
             "Design and construction of continuous flight auger (CFA) piles."),
    "gec9": ("FHWA GEC-9", "FHWA-HIF-18-031, April 2018",
             "Design, analysis and testing of laterally loaded deep foundations &mdash; p-y "
             "parameters (Appendix A), p-multipliers (Table 7-1), resistance factors (Table 4-1)."),
    "gec10": ("FHWA GEC-10", "FHWA-NHI-18-024, September 2018",
              "Drilled shafts &mdash; alpha / beta side resistance, base resistance, rock sockets, "
              "the rational OCR/Ko and Chen-2011 chains."),
    "gec11": ("FHWA GEC-11", "FHWA-NHI-10-024 (Vol. I) and FHWA-NHI-10-025 (Vol. II), November 2009",
              "MSE walls and reinforced soil slopes &mdash; reinforcement, pullout, durability and "
              "LRFD load and resistance factor tables, the K<sub>r</sub>/K<sub>a</sub> curve "
              "(Fig. 4-10), chapter text and figures."),
    "gec12": ("FHWA GEC-12", "FHWA-NHI-16-009 (Vol. I) and FHWA-NHI-16-010 (Vol. II; its cover prints 16-009), July 2016",
              "Driven-pile foundations &mdash; the Nordlund, alpha and beta charts and tables of "
              "chapter 7, resistance factors, setup, chapter text and figures. The axial-pile module "
              "is validated against the design examples of Vol. III (FHWA-NHI-16-064)."),
    "gec13": ("FHWA GEC-13", "FHWA-NHI-16-027 (Vol. I) and FHWA-NHI-16-028 (Vol. II), April 2017",
              "Ground modification methods &mdash; PVDs and preloading, lightweight fills, deep "
              "compaction, aggregate columns, column-supported embankments, deep mixing, grouting, "
              "pavement support stabilization and reinforced soil structures."),
    "gec14": ("FHWA GEC-14", "FHWA-HIF-17-016, August 2016",
              "Assuring quality in geotechnical reporting documents &mdash; chapter text and figure "
              "catalog."),
    "micropile": ("FHWA Micropile Manual", "FHWA NHI-05-039, December 2005",
                  "Micropile design and construction &mdash; bond stress, structural capacity, the "
                  "LPILE benchmark the lateral-pile module checks against."),
    "ufc_backfill": ("UFC 3-220-04FA", "UFC 3-220-04FA, Backfill for Subsurface Structures, 16 January 2004",
                     "Backfill for subsurface structures &mdash; density requirements, Table 5-1 compaction "
                     "criteria, backfill materials, lift thickness, compaction equipment and Fig. 2-5's "
                     "compaction-induced lateral pressures."),
    "ufc_expansive": ("UFC 3-220-07", "UFC 3-220-07, Foundations in Expansive Soils, 16 January 2004",
                      "Expansive soils &mdash; swell potential, the active zone, heave prediction (Table "
                      "4-2's swell relations, Van Der Merwe's Eq. 4-1, Eqs. 5-2 and 5-8), Table 6-1's "
                      "foundation systems and the grade-beam void."),
    "ufc_pavement": ("UFC 3-250-01",
                     "UFC 3-250-01, Pavement Design for Roads and Parking Areas, 14 November 2016",
                     "Road and parking-area pavement design (not airfields) &mdash; CBR flexible design "
                     "(Fig. E-1) and the vehicle curves, rigid design (Fig. F-1), mixed traffic, "
                     "stabilized layers, overlays, frost and drainage."),
    "fema_p2082": ("FEMA P-2082-1 / 2020 NEHRP",
                   "FEMA P-2082-1, NEHRP Recommended Seismic Provisions, 2020 Edition, Vol. I, "
                   "September 2020",
                   "Seismic site classification &mdash; the BC/CD/DE site classes with the BC "
                   "baseline (the post-2020 multi-period approach), design spectra and seismic design "
                   "category."),
    "california_trenching": ("Caltrans Trenching &amp; Shoring",
                             "Caltrans Trenching and Shoring Manual, June 2011, Revision 2, July 2025",
                             "Shoring and excavation &mdash; earth pressures, apparent pressure "
                             "envelopes, surcharges, basal heave, the worked examples validating soe / "
                             "sheet_pile."),
    "fhwa_pavements": ("FHWA Pavements", "FHWA NHI-05-037, May 2006",
                       "Geotechnical aspects of pavements &mdash; resilient modulus, CBR, frost, "
                       "drainage."),
    "fhwa_soils_foundations": ("FHWA Soils and Foundations",
                               "FHWA NHI-06-088 (Vol. I) and FHWA NHI-06-089 (Vol. II), December 2006",
                               "Soils and foundations reference manual &mdash; chapter text read off "
                               "the print, and figure catalog."),
    "eurocode_7_1": ("Eurocode 7 Part 1", "EN 1997-1:2004",
                     "Geotechnical design, general rules &mdash; Annex A partial factors (design "
                     "approaches 1-3) and the methods of Annexes C to H."),
    "eurocode_7_2": ("Eurocode 7 Part 2", "EN 1997-2:2007",
                     "Ground investigation and testing &mdash; the CPT, PMT, SPT, DP, FVT, DMT and PLT "
                     "correlations of the informative annexes."),
    "aashto_1993": ("AASHTO 1993", "AASHTO Guide for Design of Pavement Structures, 1993",
                    "Flexible structural number and rigid slab design equations, layer coefficients, "
                    "reliability, Appendix D load-equivalency factors, the composite k worksheet and "
                    "Appendix G swelling and frost-heave loss (US units)."),
    "em_1905": ("USACE EM 1110-1-1905", "EM 1110-1-1905, Geotechnical Design of Shallow Foundations "
                "on Soils, 31 July 2025",
                "Shallow foundations on soils (it supersedes EM 1110-1-1904 and the 1992 EM "
                "1110-1-1905) &mdash; minimum factors of safety (Table 2-1), bearing capacity "
                "factors and the shape, depth, inclination and undrained factors (Tables 5-2 to "
                "5-4), in-situ bearing factors (Table 5-5), the ultimate limit state equations "
                "(Eqs. 5-1 to 5-36: effective dimensions, drained and undrained bearing capacity, "
                "punching, sliding, uplift), stress distribution (Eqs. 6-1 to 6-13), settlement "
                "(Eqs. 7-1 to 7-47: consolidation, constrained modulus, Menard, Schmertmann, "
                "Burland and Burbidge, time rate, secondary compression) and the table printed on "
                "Figure 7-16; chapter text by the printed paragraph numbers (5-5a(2), B-1b) and a "
                "figure catalog (US units)."),
    "em_2104": ("USACE EM 1110-2-2104", "EM 1110-2-2104, 1 November 2023",
                "Strength design for reinforced concrete hydraulic structures &mdash; load factors and "
                "combinations, flexure and axial load, shear, serviceability, detailing (US units)."),
    "em_2107": ("USACE EM 1110-2-2107", "EM 1110-2-2107, 1 August 2022",
                "Design of hydraulic steel structures &mdash; LRFD loads and combinations, seismic "
                "amplification, fatigue and fracture, connections, Tainter-gate loads (US units)."),
    "ufc_structural": ("UFC 3-301-01", "UFC 3-301-01, Structural Engineering, 11 April 2023, Change 4, 3 June 2025",
                       "The DoD modifications to the IBC and ASCE 7 &mdash; risk category, the "
                       "seismic-force-resisting-system table, seismic load combinations, evaluation "
                       "and retrofit, nonstructural seismic criteria."),
    "ufc_collapse": ("UFC 4-023-03",
                     "UFC 4-023-03, Design of Buildings to Resist Progressive Collapse, 14 July 2009, "
                     "Change 4, 10 June 2024",
                     "Progressive collapse &mdash; tie forces, alternate path, enhanced local "
                     "resistance, by material."),
    "gsa_collapse": ("GSA Alternate Path Guidelines",
                     "GSA Alternate Path Analysis &amp; Design Guidelines for Progressive Collapse "
                     "Resistance, 24 October 2013, Revision 1, 28 January 2016",
                     "Progressive collapse for federal civilian buildings &mdash; alternate path "
                     "analysis, redundancy, by material."),
    "ufc_flexible_practice": ("UFC 3-250-03",
                              "UFC 3-250-03, O&amp;M Manual: Standard Practice for Flexible Pavements, 30 May 2018",
                              "Hot-mix asphalt construction practice &mdash; mix design, spray "
                              "applications, seal coats."),
    "ufc_concrete_practice": ("UFC 3-250-04",
                              "UFC 3-250-04, Standard Practice for Concrete Pavement Construction, "
                              "29 January 2024",
                              "Concrete pavement construction practice &mdash; materials, mixture "
                              "design, placement, joints, dowels."),
    "ufc_stabilization": ("UFC 3-250-11",
                          "UFC 3-250-11, Soil Stabilization and Modification for Pavements, 30 November 2020",
                          "Soil stabilization &mdash; additive selection, strength and durability "
                          "criteria, cement, lime and bituminous stabilization."),
    "wood_handbook": ("Wood Handbook", "USDA Forest Products Laboratory, FPL-GTR-282 (2021)",
                      "Wood as an engineering material &mdash; moisture relations, mechanical "
                      "properties, fastenings and structural analysis equations (chapters 4, 5, 8 "
                      "and 9)."),
    "reference_db": ("Cross-reference text search", "&mdash;",
                     "A full-text (SQLite FTS5) search DB over the chapter text of the references "
                     "that have it indexed (all but UFC 3-220-04FA and UFC 3-220-07, which are "
                     "lookups and figures only), with synonym query expansion."),
    "figure_db": ("Figure catalog + vision read-off", "&mdash;",
                  "A searchable catalog of the figures the references' catalogs list; pairs with "
                  "the <code>read_reference_figure</code> vision tool to render a chart and read a "
                  "value off it where the reference PDFs reach the host (a docs folder or the "
                  "SharePoint fetcher); without them a figure is found and cited, not read."),
    "table_db": ("Table catalog + printed-page read", "&mdash;",
                 "A searchable catalog of the printed tables in the references that have a "
                 "table catalog. Each row says where the table's values are: in a stored "
                 "section of text, in a lookup function, or only on the printed page. Pairs "
                 "with the <code>read_reference_table</code> vision tool to read a table "
                 "that is only on its page where the reference PDFs reach the host; without "
                 "them the table is found and its printed page cited, not read."),
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
            "General bearing-capacity equation with one of two correction-factor sets: **Vesic** "
            "(default; GEC-6 Eq. 5-7, shape factors Table 5-2) or **Meyerhof** (1963; UFC 3-220-20 "
            "Table 5-6). N<sub>&gamma;</sub> is the set&rsquo;s own, 2(N<sub>q</sub>+1)tan&nbsp;&phi; (GEC-6 "
            "Eq. 5-8) or (N<sub>q</sub>&minus;1)tan(1.4&phi;), or Brinch Hansen&rsquo;s "
            "1.5(N<sub>q</sub>&minus;1)tan&nbsp;&phi; (UFC Table 5-2) when named.",
            "**Terzaghi** bearing capacity (general or local shear) with the UFC 3-220-20 Table 5-5 shape "
            "factors and no depth or inclination factors (inclined loads, slopes and tilted bases are "
            "refused, UFC Table 5-3); "
            "N<sub>&gamma;</sub> after Kumbhojkar (1993) as tabulated in Das by default, or Coduto et al. "
            "(2016), the basis of UFC Table 5-2, each named in the result.",
            "Correction factors multiplied (UFC 3-220-20 Eqs. 5-18 to 5-20; GEC-6 Eq. 5-14): shape "
            "(GEC-6 Table 5-2); depth by Brinch Hansen&rsquo;s k (UFC Table 5-8, the default), GEC-6&rsquo;s "
            "d<sub>q</sub> (Table 5-4) or none (<code>depth_factor_method</code>); load inclination (Vesic, "
            "EN 1997-1 Annex D.4); base tilt (GEC-6 Table 5-5); ground slope (Vesic 1975). At &phi; = 0 "
            "the Vesic set adds d<sub>c</sub> = 0.4k, as UFC Eq. 5-21 does. The Meyerhof set refuses a "
            "base tilt or ground slope (UFC Table 5-3; p. 292).",
            "Eccentric loading via the Meyerhof effective-area (B&prime;, L&prime;) reduction.",
            "Two-layer bearing: a stronger layer over a weaker one by 2V:1H load spread to the weak "
            "layer (NAVFAC DM-7.01 ch. 4; Bowles Sec. 4-7); a weaker layer over a stronger one as a "
            "bounded transition between the two single-layer capacities, an estimate, not a "
            "closed-form solution.",
            "Groundwater in the failure wedge through an effective-unit-weight model.",
            "References: FHWA GEC-6 (FHWA-SA-02-054) ch. 5; UFC 3-220-20 ch. 5 (Tables 5-2 to 5-8); "
            "EN 1997-1 Annex D; Terzaghi (1943); Meyerhof (1963); Brinch Hansen (1970); Vesic (1975).",
        ],
        "limits": [
            "By default <code>compute()</code> applies Brinch Hansen&rsquo;s depth factors and averages the "
            "effective unit weight over a depth B below the base. This runs higher than an example that "
            "takes d<sub>q</sub> = 1.0 (GEC-6 Table 5-4 note) and GEC-6&rsquo;s C<sub>wq</sub>, "
            "C<sub>w&gamma;</sub> groundwater factors (Eq. 5-14): +17.6&nbsp;% on GEC-6 Example B-1 (V-021). "
            "<code>depth_factor_method</code> \"gec6\" or \"none\" gives the example&rsquo;s depth factor; "
            "C<sub>w</sub> is not offered.",
            "The Vesic and Meyerhof sets assume general shear. Local shear is offered with Terzaghi&rsquo;s "
            "factors only (c* = 2/3&nbsp;c, tan&nbsp;&phi;* = 2/3&nbsp;tan&nbsp;&phi;; UFC 3-220-20 Table 5-4). "
            "Punching shear (Meyerhof &amp; Hanna 1978) is not modelled.",
        ],
    },

    "settlement": {
        "problems": "How far a foundation settles and how fast: immediate (elastic) movement, primary "
            "consolidation of clays, granular settlement of sands, and the settlement&ndash;time curve.",
        "methods": [
            "Elastic (immediate) settlement with shape/rigidity influence factors.",
            "**Schmertmann (1978)** strain-influence-factor method for granular soils (NHI-06-089 Eqs. 8-16 "
            "to 8-18): the diagram interpolated in L/B between the axisymmetric and plane-strain cases, the "
            "peak at the effective stress there (<code>gamma_soil</code> required).",
            "One-dimensional **C<sub>c</sub>/C<sub>r</sub> e&ndash;log&nbsp;p consolidation** with "
            "layer summation and preconsolidation (NC/OC).",
            "**Hough (1959)** granular C&prime;-index settlement (bearing-capacity index from corrected "
            "SPT N&prime;; GEC-6 Fig 5-19).",
            "Stress increase by the 2:1 method, Boussinesq, or Westergaard (read off UFS 3-220-10 Fig. 4-10); secondary compression "
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
            "**Nordlund** method in cohesionless soils (GEC-12 Sec. 7.2.1.3.1, Eqs. 7-8, 7-9): &delta;/&phi; "
            "from Fig. 7-9, K<sub>&delta;</sub> from Tables 7-6 and 7-7, C<sub>F</sub> from Fig. 7-14, "
            "&alpha;<sub>t</sub> and N&prime;<sub>q</sub> from Fig. 7-16; &sigma;&prime;<sub>p</sub> at the toe "
            "limited to 3&nbsp;ksf (Step 8) and the unit toe resistance to q<sub>L</sub> from Fig. 7-15 "
            "(after Meyerhof 1976).",
            "**&alpha;-method** in cohesive soils (GEC-12 Sec. 7.2.1.3.2, Eq. 7-10): the adhesion "
            "C<sub>a</sub> read off Fig. 7-17 (Tomlinson) by s<sub>u</sub>, pile type and D/b; Fig. 7-18&rsquo;s "
            "stratigraphy cases, or the smaller of the two, on request (p. 246). Past the chart&rsquo;s "
            "curves, API &alpha; (Eqs. 7-13, 7-14) in a clay the engineer calls highly overconsolidated, "
            "else C<sub>a</sub> held at the curve&rsquo;s end (p. 247). A layer&rsquo;s own &alpha; or "
            "C<sub>a</sub>, with its basis, replaces the chart. Toe 9&middot;s<sub>u</sub> (Eq. 7-11).",
            "**&beta;-method** (Fellenius 1991; GEC-12 Sec. 7.2.1.3.4, Eqs. 7-19, 7-20) in every layer "
            "on request: &beta; and N<sub>t</sub> from Table 7-9 by soil type, where a layer gives none "
            "(Steps 2b, 5b).",
            "Factored resistance (<code>Q_factored</code>) by GEC-12 Eq. 7-3 with Table 7-1&rsquo;s "
            "factors by method (Nordlund 0.45, &alpha;-method 0.35; none for the &beta;-method, p. 258), "
            "and factored uplift by Eq. 7-38 (0.35 / 0.25 / 0.20). The allowable capacity uses FS 2.5 "
            "(NHI-06-089 Table 9-5, indicator piles with wave-equation analysis).",
            "Groundwater-split effective-stress integration; optional per-layer separate shaft/toe "
            "friction angle; below-grade pile head (<code>head_depth</code>); uplift capacity.",
            "Pile sections: closed and open pipe, square and circular concrete, timber, H-piles (box "
            "perimeter and toe area, Vol. I pp. 247, 408) and a custom section. An open pipe is the lesser "
            "of the plugged and unplugged cases (Eqs. 7-88, 7-89), or Tomlinson&rsquo;s Eq. 7-90 under "
            "24&nbsp;in in mainly cohesive soil (Sec. 7.10.7).",
            "References: FHWA GEC-12 Vol. I (FHWA-NHI-16-009) ch. 7; validated against GEC-12 Vol. III "
            "(FHWA-NHI-16-064) Tables D-6, D-100 and D-51 (V-001, V-002, V-059).",
        ],
        "limits": [
            "With the choices GEC-12&rsquo;s South Abutment example made (Vol. III Table D-100: "
            "<code>adhesion_curve=\"smooth_steel\"</code>, <code>adhesion_beyond_chart=\"chart_end\"</code>) "
            "the shaft is within 1&nbsp;% of the print (V-002). The defaults take Fig. 7-17&rsquo;s "
            "corrugated curve for an H-pile (Vol. I p. 247) and read 13&ndash;14&nbsp;% higher.",
            "Where a clay&rsquo;s s<sub>u</sub> is past Fig. 7-17&rsquo;s curves, GEC-12 takes API "
            "&alpha; for highly overconsolidated clay but prints no s<sub>u</sub>/&sigma;&prime;<sub>vo</sub> "
            "that marks one (p. 247; Step 1, p. 250). The run is refused until "
            "<code>highly_overconsolidated_su_ratio</code> or the layer&rsquo;s "
            "<code>highly_overconsolidated</code> is given.",
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
            "above/below water table, **Reese (1974) sand** (GEC 9's four-segment construction by "
            "default, a simplified one on request), API sand, and others.",
            "Banded finite-difference beam-column solver with P-&Delta; (axial-load) effects.",
            "Free/fixed head conditions, above-ground stickup, and H-pile strong/weak-axis or "
            "concrete-filled-pipe sections.",
            "References: FHWA GEC 9 (FHWA-HIF-18-031) App. A; Reese, Cox &amp; Koop (1974); Matlock (1970); "
            "COM624P; validated against GEC 9 Example B.1, a COM624P oracle suite and the FHWA Micropile "
            "LPILE benchmark (V-017).",
        ],
        "limits": [
            "Section stiffness is a linear EI (an uncracked composite section by transformed "
            "section) or Branson's cracked EI for reinforced concrete; no moment&ndash;curvature "
            "analysis. In the micropile benchmark (V-017) the uncracked composite EI gives 3.32 mm "
            "against LPILE's 3.3 mm.",
        ],
    },

    "pile_group": {
        "problems": "How a rigid pile cap distributes structural loads to the individual piles, and how "
            "much a group settles: axial/lateral/torsional load sharing under a 6-DOF load set, group "
            "efficiency, and group settlement.",
        "methods": [
            "**Rigid-cap 6-DOF** stiffness analysis &mdash; per-pile axial and lateral stiffness, batter, "
            "and one right-hand sign convention carried end to end.",
            "**Group efficiency factor** &eta;<sub>g</sub> read from the printed rows by soil, the cap's "
            "contact with the ground and the spacing (GEC-12 Sec. 7.2.2; UFC 3-220-20 Table 6-26 for "
            "drilled shafts, CFA, displacement piles and micropiles); **block failure** by GEC-12 "
            "Eqs. 7-36 and 7-37; lateral **p-multipliers** by row (AASHTO as GEC-12 prints it, or "
            "Mokwa, UFC 3-220-20 Table 6-38).",
            "**Group uplift** (GEC-12 Sec. 7.2.3.2; UFC 3-220-20 Eqs. 6-40 to 6-43): the lesser of the "
            "piles&rsquo; factored tension resistances and the block&rsquo;s (the <code>group_uplift</code> tool; the "
            "rigid-cap tools flag every pile in tension and say to run it, and the pile-group calc package "
            "checks the piles in tension when it has the block&rsquo;s inputs).",
            "**Meyerhof (1976)** SPT group settlement as GEC-12 Eqs. 7-50 to 7-52 print it, and an "
            "elastic equivalent-footing settlement (GEC-12 Sec. 7.3.5.3, Eq. 7-55, 2V:1H spread) summed to "
            "where the stress falls to 10&nbsp;% of the footing&rsquo;s bearing stress (UFC 3-220-20 p. 449; "
            "GEC-12 step 1c&rsquo;s 10&nbsp;% of the existing effective stress as "
            "<code>influence_depth_m</code>).",
            "References: FHWA GEC-12 Vol. I ch. 7; UFC 3-220-20 ch. 6; validated against GEC-12 "
            "Vol. III Tables D-13, D-14 and D-23 and UFC 3-220-20 Appendix B-5.4 (V-005, V-060).",
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
            "**&alpha;-method** side resistance in clay (GEC-10 (2018) Eq. 10-18 rational &alpha; by "
            "default, on s<sub>u</sub>(CIUC): c<sub>u</sub> is taken as s<sub>u</sub>(CIUC) unless "
            "<code>su_test_type</code> says it came from a UU or UC test, when Eqs. 10-16/10-17 transform it; the "
            "O&rsquo;Neill &amp; Reese (1999) AASHTO &alpha; by name).",
            "**&beta;-method** side resistance in sand (GEC-10 (2018) Eq. 10-13 by default: K<sub>o</sub> "
            "from the OCR, &sigma;&prime;<sub>p</sub> from each layer&rsquo;s N<sub>60</sub>; or the "
            "O&rsquo;Neill&ndash;Reese depth &beta; by name).",
            "Base resistance in sand (GEC-10 Eq. 10-14 on the mean N<sub>60</sub> over two diameters below "
            "the base) and clay (N*<sub>c</sub>&middot;s<sub>u</sub>, Eq. 10-19, s<sub>u</sub> averaged over "
            "two diameters); no large-base reduction unless asked for, as the 2018 edition prints none.",
            "Rock sockets: side resistance by Eq. 10-21 with C = 1.0 by default (<code>rock_side_C</code>; "
            "GEC-10 gives 0.65, Eq. 10-22&rsquo;s value at &alpha;<sub>E</sub> = 1, and 1.9 for roughened "
            "sockets, a C above 1.0 to be verified by load tests or local experience; Eq. 10-22 for caving sockets), q<sub>u</sub> held no higher than the concrete&rsquo;s "
            "f&prime;<sub>c</sub> when it is given; base 2.5&nbsp;q<sub>u</sub> (Eq. 10-23) only where the "
            "engineer states the conditions of p. 10-22 or the rock below the base is at RQD 100 (a rock layer given no RQD is taken at 100, so a profile "
            "without RQD gets 2.5&nbsp;q<sub>u</sub>, and the result&rsquo;s choices say so), else the "
            "Hoek&ndash;Brown base or a refusal. Permanent casing, fractured rock at the top of the socket, "
            "and a top exclusion zone in clay (1.5&nbsp;m by default; none in sand or rock unless given).",
            "LRFD resistance factors and capacity-vs-depth curves.",
            "References: FHWA GEC-10 (FHWA-NHI-18-024); O&rsquo;Neill &amp; Reese; Chen &amp; Kulhawy; "
            "validated against the GEC-10 Appendix A example (V-006/V-007/V-008).",
        ],
        "limits": [
            "The O&rsquo;Neill &amp; Reese large-base reduction is off by default "
            "(<code>large_base_reduction</code>), because GEC-10 (2018) prints none; an example worked to "
            "the 2010 edition or to NHI-06-089 with the reduction will differ by that factor.",
            "Rock q<sub>u</sub> is not checked against the concrete&rsquo;s f&prime;<sub>c</sub> unless "
            "<code>concrete_fc</code> is given, and the result says so.",
        ],
    },

    "wave_equation": {
        "problems": "Pile drivability and the dynamic bearing check: the bearing graph (static resistance "
            "vs blow count), driving stresses, and refusal, from a hammer&ndash;cushion&ndash;pile&ndash;soil "
            "model &mdash; the analysis behind a driven-pile field acceptance criterion.",
        "methods": [
            "**Smith (1960)** one-dimensional wave-equation model: the pile as lumped masses and springs "
            "about 1&nbsp;m long (GEC-12 Vol. 2 p. 155), elasto-plastic soil springs (quake) with "
            "**smith** or **smith-viscous** damping (UFC 3-220-20 p. 488), one cushion spring with the "
            "helmet&rsquo;s mass lumped with the first pile segment, and explicit central-difference time "
            "integration.",
            "Single-blow analysis, bearing graph, and drivability vs depth: one pile in the leads, "
            "embedded to each depth (<code>embedded_length</code>; GEC-12 Vol. 2 pp. 224, 227). The "
            "permanent set is the maximum toe displacement less the quakes (p. 158). Refusal is at "
            "120&nbsp;blows/ft by default (p. 161). Fourteen built-in hammers take the ram weights, "
            "strokes and energies of Vol. 2 Appendix C; custom ram, stroke and efficiency are accepted. "
            "With no cushion given, only the Vulcan 014 takes Table 12-2&rsquo;s cushion and helmet "
            "(p. 183); any other hammer needs the cushion, its COR and the helmet weight.",
            "References: Smith (1960); GRLWEAP-style modelling; FHWA GEC-12.",
        ],
        "limits": [
            "Every hammer is a rigid ram striking at v = &radic;(2gh&eta;) (GEC-12 Vol. 2 Eq. 15-1). A "
            "diesel&rsquo;s combustion and changing stroke are not modelled, and the result warns that its "
            "blow counts are likely low (p. 206). Treat a diesel bearing graph as a trend, not a tie-out "
            "(V-003: 48 against the published 120&nbsp;blows/ft; GEC-12 Example 6, ICE 42-S: 50 and 103 "
            "against 99 and 209).",
        ],
    },

    "downdrag": {
        "problems": "Negative skin friction (downdrag) on a pile through settling ground: the neutral "
            "plane, the maximum axial load in the pile, and the drag load added to the structural load.",
        "methods": [
            "**Neutral plane method** &mdash; every method the references give, side by side with its "
            "neutral plane, drag force and settlement: the intersection of the load and resistance curves "
            "at 100 / 50 / 0&nbsp;% toe mobilization (UFC 3-220-20 &sect;6-7.4, &sect;6-5.8.4.2; GEC-12 "
            "&sect;7.3.6.1), the toe resistance compatible with the toe penetration (UFC &sect;6-5.8.4.3), "
            "the neutral plane at the bearing layer or at the pile toe, and Endo et al. (1969). The headline "
            "is UFC 3-220-20&rsquo;s &sect;6-7.4 procedure; skin friction on the final effective stress.",
            "Fill-induced downdrag, groundwater drawdown, and a maximum-load structural check.",
            "The toe&rsquo;s N<sub>t</sub> from GEC-12 Table 7-9 (UFC Table 6-17, Fellenius) by soil type, or "
            "Cheng (2004) by name; 9&middot;s<sub>u</sub> in clay. The pile&rsquo;s weight is left out of the "
            "load curve, as UFC Fig. 6-26 and GEC-12 &sect;7.3.6.1 do (<code>include_pile_weight</code>). "
            "The toe load is counted below the toe to 10&nbsp;% of the equivalent footing&rsquo;s bearing "
            "stress (UFC &sect;6-5.8.4.3), or GEC-12&rsquo;s 10&nbsp;% of the effective stress on request.",
            "References: UFC 3-220-20; FHWA GEC-12; Fellenius; CGPR #56; validated against GEC-12 (V-004, "
            "NP/Q<sub>max</sub>/DF within 0.2%) and UFC App. B-5.5.",
        ],
        "limits": [
            "For an H-pile pass the box toe area as <code>pile_area</code>; it is also the area used for "
            "AE and the pile weight. The module&rsquo;s own shaft is the &alpha;-method in clay (UFC 3-220-20 "
            "Table 6-16, Fig. 6-14) and &beta; from GEC-12 Table 7-9 in sand; it can differ from a Nordlund "
            "shaft, so give each layer&rsquo;s &beta; or &alpha; to reproduce a design shaft (as V-004 does).",
        ],
    },

    "sheet_pile": {
        "problems": "Cantilever and single-anchor flexible retaining walls: the embedment for stability, "
            "the maximum bending moment, and the anchor force, from a layered earth-pressure diagram with "
            "groundwater.",
        "methods": [
            "**Rankine / Coulomb** active and passive coefficients (with wall friction), and a "
            "**log-spiral** passive option (Coulomb active with it), which avoids Coulomb&rsquo;s "
            "over-prediction of K<sub>p</sub> at high &delta;: UFC 3-220-20 Eq.&nbsp;4-12 by default, or "
            "Figs.&nbsp;4-12/4-13 or the Caquot&ndash;Kerisel (1948) chart (Caltrans T&amp;S "
            "Fig.&nbsp;4-20); wall friction from UFC Table&nbsp;4-1 by the soil against the wall.",
            "The cantilever by the Caltrans T&amp;S Simplified Method (Sec.&nbsp;7-5.02): moments about "
            "the pivot O give D<sub>0</sub>, with the passive pressure divided by FOS_passive (UFC "
            "3-220-20 Fig.&nbsp;4-41 step 2), then D = 1.2&nbsp;D<sub>0</sub> (Caltrans 7-5.02 step 3; "
            "UFC Fig.&nbsp;4-41 step 6); the maximum moment at zero shear on that factored diagram, to O. "
            "The anchored wall by free earth support about the anchor (UFC Fig.&nbsp;4-37). Layered "
            "pressures with water on each side: water standing in the excavation above the dredge line "
            "counts, and the net water pressure is the retained side&rsquo;s less the excavated "
            "side&rsquo;s (UFC Sec.&nbsp;4-4.1, Figs.&nbsp;4-5/4-6).",
            "References: Caltrans T&amp;S Manual; UFC 3-220-20 (DM&nbsp;7.2); USS Steel Sheet Piling "
            "Design Manual; validated against Caltrans T&amp;S Ex 7-2&rsquo;s layered pressures (V-025), "
            "Ex 8-1&rsquo;s K<sub>a</sub> and log-spiral K<sub>p</sub> (V-013; that example&rsquo;s "
            "embedment uses the FHWA apparent diagram, which <code>soe</code> reproduces), UFC "
            "Fig.&nbsp;4-43 with water on both sides and, for the anchored wall&rsquo;s maximum moment, the USS "
            "Steel Sheet Piling Design Manual&rsquo;s Design Example No. 2 (within 1&nbsp;%).",
        ],
        "limits": [
            "<code>analyze_cantilever</code> is the Caltrans Simplified Method for a continuous wall, per "
            "metre run: soldier-pile effective widths and arching (Caltrans T&amp;S Ch.&nbsp;7) are not "
            "modelled (V-012). It is not UFC 3-220-20 Fig.&nbsp;4-41&rsquo;s full method, which gives a "
            "longer D: on UFC Fig.&nbsp;4-43, 15.6&nbsp;ft against the chart&rsquo;s 17&nbsp;ft. The "
            "maximum moment is on the factored diagram (UFC Fig.&nbsp;4-41 step 5); "
            "<code>soe</code>&rsquo;s cantilever takes it at FS 1.0 (Caltrans Ex 7-3), so the two tools "
            "differ.",
        ],
    },

    "soe": {
        "problems": "Support of an open excavation: braced and cantilever walls, apparent-pressure "
            "diagrams, strut/anchor loads, wall-section selection, and the bottom-stability checks "
            "(basal heave, blowout, piping).",
        "methods": [
            "**FHWA apparent-pressure** envelopes (Terzaghi&ndash;Peck / Peck) for braced cuts and "
            "anchored walls; braced-wall support loads by the Caltrans hinge method (or GEC-4's, or "
            "UFC 3-220-20's simple spans) and the GEC-4 tributary area method for anchored walls.",
            "The cantilever by the Caltrans T&amp;S Simplified Method (Sec.&nbsp;7-5.02; D = "
            "1.2&nbsp;D<sub>0</sub>; moment and shear at FS 1.0, Ex 7-3); the embedment below the lowest "
            "support of a braced or anchored wall, one support or several, by free earth support about it "
            "(Sec.&nbsp;8-4.02); ground-anchor design by GEC-4 Sec.&nbsp;5.3; wall-section sizing at "
            "0.60&nbsp;F<sub>y</sub> by default (the Caltrans examples).",
            "Basal heave &mdash; **Terzaghi** (the default; GEC-4 Eq.&nbsp;35 with side shear, "
            "Eq.&nbsp;37 for a wide cut, N<sub>c</sub> off GEC-4 Fig.&nbsp;51(c); UFC 3-220-20 "
            "Fig.&nbsp;2-7(b) given beside it), **Bjerrum&ndash;Eide** (N<sub>c</sub> off Caltrans "
            "Fig.&nbsp;10-15 with Eq.&nbsp;10-3-4) and the **Caltrans** force balance with sidewall shear "
            "(Sec.&nbsp;10-3.01); bottom blowout (soil-plug weight against the excess head) and piping "
            "(critical gradient against the average exit gradient), in the module&rsquo;s own forms, "
            "since none of these references prints one.",
            "References: FHWA GEC-4 (FHWA-IF-99-015); Terzaghi &amp; Peck; Bjerrum &amp; Eide; Caltrans "
            "T&amp;S; validated V-014/V-016.",
        ],
        "limits": [
            "The apparent-pressure envelopes, anchor loads and moments reproduce the worked examples "
            "(V-016, Caltrans Ex 8-2); the embedment below the lowest support is free earth support "
            "about it (Caltrans T&amp;S 8-4).",
            "Where no moment balance exists within six times the excavation depth, the free-earth "
            "solution returns no embedment (and nothing computed from it) with a plain note, never a "
            "depth without a solution.",
        ],
    },

    "retaining_walls": {
        "problems": "Cantilever (semi-gravity) concrete retaining walls and mechanically-stabilized-earth (MSE) "
            "walls: external stability (sliding, overturning, bearing, eccentricity) and MSE internal "
            "stability (reinforcement tension and pullout), by ASD and LRFD.",
        "methods": [
            "Cantilever-wall stability: the active thrust on the vertical plane through the back of the "
            "heel, by **Rankine** or **Coulomb** with &delta; = &beta; on that plane (FHWA NHI-06-089 "
            "p.&nbsp;10-20, Ex 10-3), no tension in a cohesive backfill (UFC 3-220-20 Fig.&nbsp;4-5 Note "
            "A); sliding by GEC-6 Eqs.&nbsp;5-38 to 5-40 (FS 2.0 when passive is counted, UFC "
            "Sec.&nbsp;4-5.1.3); e within B/6 on soil and B/4 on rock; bearing as the uniform "
            "q<sub>eq</sub> = V/(B&nbsp;&minus;&nbsp;2e) against a given allowable pressure for B&prime; "
            "(GEC-6 Eq.&nbsp;6-6, Sec.&nbsp;6.4.2), or the trapezoid&rsquo;s q<sub>max</sub> on rock; an "
            "optional water table behind the wall with uplift on the base (UFC Sec.&nbsp;4-4.1; EM "
            "1110-2-2104 para.&nbsp;E-12).",
            "MSE walls by GEC-11&rsquo;s **Simplified Method**: external stability (sliding, also on the "
            "lowest geosynthetic sheet; eccentricity; bearing on L&nbsp;&minus;&nbsp;2e) and internal "
            "K<sub>r</sub>/K<sub>a</sub> (Fig.&nbsp;4-10) and F* pullout for steel strips, bar mats and "
            "welded grids and for geogrids and geotextiles (F* = 2/3&nbsp;tan&nbsp;&phi;, Eq.&nbsp;3-8); "
            "the bilinear surface for inextensible and the linear one for extensible reinforcement "
            "(Fig.&nbsp;4-9); level, sloping or broken backslopes and a face battered 10&deg; or more "
            "(Vol&nbsp;I p.&nbsp;4-15, Eq.&nbsp;4-26).",
            "**AASHTO/GEC-11 LRFD** external stability &mdash; Strength I max/min + Service I load-factor "
            "combinations, live-load exclusions, and per-mode capacity/demand ratios.",
            "References: FHWA GEC-11 (FHWA-NHI-10-024 Vol&nbsp;I, NHI-10-025 Vol&nbsp;II); AASHTO LRFD; "
            "FHWA GEC-6; FHWA NHI-06-089; UFC 3-220-20; validated against GEC-11 Ex E4/E7 "
            "(V-009/V-010/V-011) and, for the cantilever, UFC 3-220-20 App.&nbsp;B-1 and NHI-06-089 Ex "
            "10-3 (V-061).",
        ],
        "limits": [
            "MSE external stability is an ASD factor-of-safety check (<code>mse_wall</code>) or an LRFD "
            "capacity/demand check (<code>mse_lrfd_external_stability</code>; in the library "
            "<code>analyze_mse_wall(lrfd_external=True)</code> runs both). The LRFD check and a broken "
            "backslope take a vertical face only. The MSE checks have no water table (the cantilever wall "
            "has one).",
        ],
    },

    "ground_improvement": {
        "problems": "Improving weak ground before it is loaded: aggregate piers and stone columns, "
            "prefabricated vertical (wick) drains, surcharge preloading, and vibro-compaction &mdash; the "
            "settlement reduction, capacity increase, and time to consolidate.",
        "methods": [
            "Aggregate piers / stone columns: area-replacement ratio, equal-strain settlement-reduction "
            "factor, composite modulus, and the settlement ratio off GEC-13 Fig. 5-21 (weighted "
            "modulus, Balaam, and the **Priebe curve as GEC-13 prints it**, &phi; = 37.5&deg;; no Priebe "
            "formula).",
            "Radial consolidation for PVDs by GEC-13 Eq. 2-1 with F(n) = ln(n) &minus; 0.75 (Eq. 2-2), a 2.5 in drain "
            "and smear ignored by default (GEC-13 p. 2-35; smear optional by UFC 3-220-10 Eq. 5-25; "
            "well resistance not modelled, GEC-13 p. 2-36), combined with vertical drainage by "
            "Carrillo (UFC 3-220-10 Eq. 5-27); drain-spacing design.",
            "Surcharge preloading; vibro-compaction feasibility by GEC-13's fines limits (p. 4-65) and "
            "SPT N (Table 4-8), with the probe spacing for a target relative density off GEC-13 Fig. 4-34.",
            "References: FHWA GEC-13 (ground-modification manual); UFC 3-220-10 (2026); "
            "validated V-018/V-019/V-020.",
        ],
        "limits": [
            "Aggregate-pier settlement uses the equal-strain settlement-reduction model, not the GEC-13 "
            "stiffness-modulus (k<sub>g</sub>) two-layer method (V-018); the Priebe curve is GEC-13's, "
            "drawn for &phi; = 37.5&deg; only and read at A/A<sub>c</sub> 1.6 to 8, and depends on the "
            "area-ratio convention supplied (V-019).",
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
            "Critical-surface search: circular centre-grid with the best centres refined, entry&ndash;exit "
            "arcs, random noncircular polylines with differential-evolution / PSO refinement, and "
            "weak-layer search; then small circles through each face&rsquo;s toe and, wherever "
            "c&prime;&nbsp;=&nbsp;0 soil forms a face, shallow planar surfaces parallel to it. Every trial "
            "surface carries the reinforcement it crosses; the result names the governing surface, each "
            "family&rsquo;s minimum and the reinforcement crossing it, and warns when the governing circle "
            "ends at a profile end (the profile is then carried on level and the lower FOS reported).",
            "Advanced per-layer strength models (the layer <code>strength_model</code> selector): "
            "Mohr&ndash;Coulomb (default), **SHANSEP** undrained-strength-ratio "
            "(s<sub>u</sub> = S&middot;OCR<sup>m</sup>&middot;&sigma;&prime;<sub>v</sub>, with a "
            "s<sub>u,min</sub> floor), **Generalized Hoek&ndash;Brown** for rock "
            "(&sigma;<sub>ci</sub>, GSI, m<sub>i</sub>, D), and **anisotropic** undrained strength "
            "(s<sub>u</sub> active / DSS / passive by the slice base&rsquo;s inclination, Casagrande&ndash;Carrillo).",
            "Reinforcement crossing the slip surface: **soil nails** (pullout + bar-tension capacity, "
            "FHWA GEC-7), **tieback anchors** (allowable tension when the bond zone crosses the "
            "surface), horizontal **geosynthetic** layers, and **Ito &amp; Matsui (1975)** stabilizing "
            "piles, applied in every method as Wright &amp; Duncan (1991) apply them: active (the default; "
            "the support reduces the driving force) or passive (the support is added to the resisting "
            "force and so divided by F); plus ponded water, tension cracks, and pseudo-static "
            "k<sub>h</sub>.",
            "Seismic displacement (the yield coefficient k<sub>y</sub>, **Newmark** sliding block on a record, "
            "the **Jibson 2007** and **Bray &amp; Travasarou 2007** regressions), **rapid drawdown** (USACE "
            "2-stage, Lowe&ndash;Karafiath 2-stage, Duncan&ndash;Wright&ndash;Wong 3-stage; on one surface or by "
            "search), infinite-slope FOS, and "
            "**probabilistic FOS** (FOSM and Monte Carlo).",
            "References: Fredlund &amp; Krahn (1977); Bishop (1955); Spencer (1967); Janbu; Duncan, Wright "
            "&amp; Brandon (2014); Ito &amp; Matsui (1975); Jibson (2007); validated vs Fredlund&ndash;Krahn "
            "/ ACADS / Duncan / Slide2 (VALIDATION.md, V-015/V-026&hellip;V-040).",
        ],
        "limits": [
            "Pore pressure is a piezometric surface, per-layer r<sub>u</sub>, or scattered (x, z, u) "
            "points (<code>pore_pressure_points</code>) interpolated linearly on their Delaunay "
            "triangulation, with suction set to zero. The ACADS 5 / Slide2 #10 flow net could not be "
            "pinned: Fig. 10.2&rsquo;s grid values did not survive the text extraction, and they are not invented (V-029).",
            "A tension crack sits at the entry side (default) or the exit side "
            "(<code>tension_crack_side</code>). By default the cracked wedge is kept as zero-strength "
            "driving soil with the full hydrostatic thrust, which is conservative. "
            "<code>tension_crack_model='truncation'</code> removes the wedge from the sliding mass, as "
            "Slide2 does, and reproduces ACADS 1(b) within 0.5&nbsp;% (V-026).",
            "When a rigorous method finds no solution on a surface that supports cross, the result gives "
            "no FOS for that method and says so, rather than an unreinforced value.",
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
            "**Strength-reduction method** for slope FOS (Griffiths &amp; Lane 1999), within 2&nbsp;% "
            "across meshes once four or more elements span the slope; staged construction, with movements reported per stage and counted "
            "from the end of the in-situ stage; initial-stress relaxation for tunnels/cavities.",
            "**Footing capacity** (<code>fem2d_footing_capacity</code>): a strip footing loaded to "
            "collapse, the soil&rsquo;s weight applied first and a surcharge beside the footing held "
            "while the footing pressure rises. A flexible footing (the default) reads Prandtl&rsquo;s "
            "(2&nbsp;+&nbsp;&pi;)c within about 2&nbsp;%; on a frictional soil its q<sub>ult</sub> is a "
            "lower bound that more iterations raise, and with no surcharge on a cohesionless soil it is "
            "set by the mesh (both warned). <code>footing='rigid'</code> is offered but reads high on the "
            "clay check and gives no mesh-converged value on a frictional soil; "
            "<code>mesh='graded'</code> is for &phi;&nbsp;=&nbsp;0 clay only.",
            "**Staged excavation** (<code>analyze_excavation</code>): half of a symmetric excavation, "
            "in-situ stresses by K<sub>0</sub>, then one stage per lift that removes the soil and "
            "releases the forces it carried; the wall is wished in place and each strut installed "
            "stress-free when the dig reaches it. Below the water table the excavation is dewatered by "
            "default, with steady seepage under the wall (<code>excavation_water</code>: "
            "<code>'dewatered'</code>, or <code>'full'</code> / <code>'level'</code>); the result gives "
            "each stage, the wall&rsquo;s largest moment and shear with their depths, the net water "
            "pressure and the upward gradient in front of the wall. Benchmarked against the USS Steel "
            "Sheet Piling Design Manual&rsquo;s cantilever in granular soil (M<sub>max</sub> within "
            "3&nbsp;%).",
            "Steady **seepage** (Laplace equation, CST elements) and **Biot consolidation** (staggered, and a monolithic "
            "Taylor&ndash;Hood u&ndash;p scheme); PLAXIS-style calc-package plots.",
            "References: Griffiths &amp; Lane (1999); de Souza Neto et al. (2008); Clausen et al. (2006); "
            "Schanz et al. (1999); validated vs Griffiths&ndash;Lane, the Prandtl solution (~2%), and "
            "Itasca FLAC (VALIDATION.md, V-023/V-024).",
        ],
        "limits": [
            "CST elements lock in plastic collapse (Prandtl N<sub>c</sub> overshoots) &mdash; use T6 for "
            "collapse/FOS work; CST is for elastic/seepage/Biot only.",
            "It is a research-grade solver, not a substitute for a commercial FE package under QA; a "
            "far-field boundary must be placed far enough to avoid truncation error. The strength "
            "reduction warns when the mesh has too few elements across the slope, and when "
            "&phi;&nbsp;=&nbsp;0 soil lies below the toe with level ground shorter than 1.5 times its "
            "depth (a deep circle cannot form). By default over such soil the model carries the level "
            "ground to twice that depth each side, so Taylor&rsquo;s deep circle can form (a slightly "
            "higher FOS than the narrower model and about twice the run time); "
            "<code>x_extend</code> sets the margins, one number or a (left, right) pair. An undrained "
            "layer with Poisson&rsquo;s ratio below 0.45 is warned: use 0.49.",
        ],
    },

    "seismic_geotech": {
        "problems": "Site seismic geotechnics: the AASHTO seismic site class and site factors as FHWA "
            "GEC-12 reprints them, Mononobe&ndash;Okabe seismic earth pressures on walls, SPT "
            "liquefaction triggering, post-liquefaction residual strength, and small-strain shear "
            "modulus and modulus reduction.",
        "methods": [
            "**AASHTO site class** (GEC-12 Vol I Table 7-26) from V<sub>s30</sub>, N-bar (field N, "
            "at most 100 blows/ft) or s<sub>u</sub>-bar over the upper 30&nbsp;m; site factors "
            "F<sub>a</sub>/F<sub>v</sub> (Tables 7-28, 7-29) and, when PGA is given, F<sub>pga</sub> "
            "(Table 7-27) and A<sub>s</sub> = F<sub>pga</sub>&middot;PGA (Eq. 7-78); S<sub>DS</sub>, "
            "S<sub>D1</sub> (Eq. 7-79). Site Class F and Class E by soft-clay thickness are not checked.",
            "**Mononobe&ndash;Okabe** K<sub>AE</sub> (UFC 3-220-20 Eq. 4-15) with battered walls and "
            "sloping backfill, the seismic thrust and its increment &Delta;P<sub>AE</sub> at 0.6&nbsp;H; "
            "K<sub>PE</sub> (FHWA GEC-4 Eq. 44) on its own wall friction, 0.4&phi; unless given "
            "(UFC Fig. 4-8&rsquo;s limit for a planar passive wedge).",
            "SPT liquefaction triggering by the **NCEER / Youd et al. (2001)** simplified procedure, "
            "CSR and r<sub>d</sub> as FHWA GEC-13 Eq. 5-9 prints them; residual strength by Seed "
            "&amp; Harder (1990) (default) or Idriss &amp; Boulanger (2007) read off GEC-12 Fig. 7-78, "
            "with void redistribution &ldquo;significant&rdquo; or &ldquo;negligible&rdquo;.",
            "G<sub>max</sub> from V<sub>s</sub>, CPT (Rix &amp; Stokoe; Mayne &amp; Rix), Hardin (1978) "
            "or Andersen (2015), and Ishibashi &amp; Zhang (1993) modulus reduction and damping "
            "(FHWA GEC-5 Ch. 8 for the first three).",
            "References: FHWA GEC-12 Vol I (FHWA-NHI-16-009) Sec. 7.4.2.1, after AASHTO (2014); UFC "
            "3-220-20 Sec. 4-4.4.1; FHWA GEC-4 Eq. 44; FHWA GEC-13 Eq. 5-9; NCEER (1997) / Youd et al. "
            "(2001); Idriss &amp; Boulanger (2007) as GEC-12 Fig. 7-78 prints it; FHWA GEC-5 Ch. 8. "
            "Validated against GEC-11 Example E7 (V-011, K<sub>AE</sub> within 0.06%; V-062, site "
            "factor and k<sub>max</sub>).",
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
            "CPT field correlations: V<sub>s</sub> (McGann 2015, Robertson 2009 or Andrus 2007), "
            "D<sub>r</sub>, s<sub>u</sub>/&sigma;&prime;<sub>v</sub>, permeability and I<sub>c</sub> "
            "profiles.",
            "References: Boulanger &amp; Idriss (2014); the liquepy library (installed with the app).",
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
            "One-dimensional nonlinear site response (soil column, base motion), run in a separate "
            "process with a time limit (<code>time_limit_s</code>, default 600&nbsp;s).",
            "References: PM4Sand (Boulanger &amp; Ziotopoulou); the OpenSeesPy library. Site response is "
            "checked against the OpenSees wiki&rsquo;s layered soil column (total stress, McGann &amp; "
            "Arduino) and the OpenSeesPy effective-stress example, Kramer (1996) &sect;7.2&rsquo;s uniform "
            "damped layer on elastic rock, and pyStrata. OpenSeesPy is installed with the app.",
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
            "References: the pyStrata library; Darendeli (2001); Menq (2003). pyStrata is installed "
            "with the app (0.5 on Python 3.12, 0.6 on Python 3.13).",
        ],
        "limits": [
            "Equivalent-linear (not fully nonlinear); large-strain problems may warrant the OpenSees "
            "nonlinear path.",
        ],
    },

    "section_props": {
        "problems": "Structural cross-section properties for rectangles, circles, CHS, RHS and I-sections given by their dimensions, and any polygon: area, moments "
            "of inertia, elastic and plastic moduli, radii of gyration, torsion and warping constants "
            "&mdash; the inputs every member check starts from. Dimensions in mm.",
        "methods": [
            "Exact closed-form integration over the section outline (Green&rsquo;s theorem): parametric "
            "rectangle / circle / CHS / RHS / I-section, plus arbitrary polygon outlines.",
            "Torsion and warping constants from the published closed forms &mdash; the exact St.&nbsp;Venant "
            "series for a rectangle, Bredt for closed boxes, El&nbsp;Darwish &amp; Johnston / AISC for "
            "I-sections; arbitrary outlines fall back to a finite-difference Prandtl solve.",
        ],
        "limits": [
            "Geometric properties only &mdash; no member design checks; nominal member capacity comes "
            "from the design reference layer as it grows.",
        ],
    },

    "concrete_props": {
        "problems": "Reinforced-concrete rectangular sections: gross and cracked stiffness, cracking "
            "moment, nominal moment capacity (sagging/hogging), and the N-M interaction diagram. "
            "mm and MPa in, kN&middot;m out.",
        "methods": [
            "Strain-compatibility section analysis against the ACI&nbsp;318-19 material model "
            "(Ec = 4700&radic;f&rsquo;c, rectangular stress block with ACI &beta;1, fr = 0.62&radic;f&rsquo;c).",
            "Capacities are NOMINAL &mdash; apply code &phi; factors separately.",
        ],
        "limits": [
            "Rectangular sections with edge bar layers, one layer per face.",
        ],
    },

    "pynite": {
        "problems": "Elastic frame and continuous-beam analysis: reactions, member moment/shear/axial "
            "envelopes, and deflections for 2D/3D skeletal structures. SI units (m, kN, kPa).",
        "methods": [
            "Matrix displacement analysis (**PyNiteFEA**, MIT); plane frames are auto-stabilized "
            "out-of-plane; the continuous-beam method reports textbook sagging-positive conventions.",
        ],
        "limits": [
            "Linear-static only; the toolkit has no nonlinear or dynamic frame analysis. Where numpy is "
            "held at 1.26 (Foundry) PyNiteFEA cannot install and the tool is left out.",
        ],
    },

    "seismic_signals": {
        "problems": "Ground-motion signal processing: response spectra, intensity measures, and rotated "
            "components from an acceleration record &mdash; the quantities behind a seismic demand.",
        "methods": [
            "Response spectra by Nigam and Jennings (1969), in sub-steps at short periods (the OpenSees site "
            "response's code); bandpass filtering and baseline correction, PGV and PGD (**eqsig**); rotated-component spectra RotD0/RotD50/RotD100 "
            "(Boore 2010) from the same time-domain oscillator responses; Arias intensity, significant "
            "duration, and other intensity measures.",
            "References: Nigam and Jennings (1969); Boore (2010); the eqsig library (installed with "
            "the app).",
        ],
        "limits": [],
    },


    "subsurface": {
        "problems": "Getting field and lab data into a modelled ground: parse and validate DIGGS, GEF/"
            "BRO-XML CPT, and AGS4 files, visualize parameters against depth, and compute trend "
            "statistics &mdash; the ingest side of characterization.",
        "methods": [
            "**DIGGS** parser (18 test types) with Plotly parameter-vs-depth, Atterberg, plan-view and "
            "trend plots; the plan view labels its axes from the stated coordinate system.",
            "**DIGGS 2.6 writer** (<code>write_diggs</code>) for logs and lab results read off a report: "
            "records given inline or as saved JSON files, checked against the schema and read back, with "
            "cross-checks (lab links, drives against sample intervals, readings below the hole&rsquo;s "
            "depth) that never change a value; groundwater not encountered is never written as a level, "
            "an unknown position is never written as 0, and a printed refusal is kept as one.",
            "Folded format adapters: **GEF/BRO-XML** CPT and borehole parsing (pygef), **AGS4** read/"
            "validate (python-ags4; an optional extra, <code>[ags4]</code>, since 5.34.0), and **DIGGS** "
            "schema/dictionary validation (pydiggs).",
            "References: the DIGGS, GEF, and AGS4 standards; pygef / python-ags4 / pydiggs. GEF/BRO-XML "
            "parsing is installed with the app; AGS4 read/validate needs the <code>[ags4]</code> extra "
            "and DIGGS schema/dictionary validation the <code>[pydiggs]</code> extra.",
        ],
        "limits": [
            "Attach a file via the agent&rsquo;s attachment mechanism (<code>attachment_key</code>) to "
            "parse a document&rsquo;s contents.",
        ],
    },

    "gstools": {
        "problems": "Geostatistics for spatially variable ground: fit a variogram to scattered data, krige "
            "a property between borings, generate correlated random fields for spatial-variability "
            "studies, and find the probability that a TOTAL over a footprint exceeds a value &mdash; "
            "the volume of rock above an excavation level, the area where soft soil runs below a level "
            "&mdash; by conditional simulation.",
        "methods": [
            "Variogram estimation and model fitting; ordinary/simple/universal **kriging** (the "
            "smoothed best estimate between points, with the kriged surface&rsquo;s totals over plan "
            "polygons and Gaussian pointwise expectations; a large grid goes to a CSV); unconditional "
            "**random-field** generation, wrapping **GSTools**.",
            "**Conditional simulation**: many realizations that honour the observations, by "
            "conditioning by kriging (Journel &amp; Huijbregts 1978; Chil&egrave;s &amp; Delfiner 2012 "
            "&sect;7.3.1), each integrated over plan polygons and levels on the same realizations: "
            "mean, percentiles, the chance of any encounter, an exceedance table with standard errors, "
            "a convergence table, CSV files, an exceedance curve and a plan map of P(z &gt; level). "
            "GSTools&rsquo; CondSRF is offered to reproduce a CondSRF notebook; it spreads totals too "
            "widely, so it is not the default.",
            "References: the GSTools library (M&uuml;ller et al.). Installed with the app (no extra needed).",
        ],
        "limits": [
            "The spread is what the chosen model implies: a Gaussian, stationary field with the "
            "covariance model, variance and length scale given (or computed from the data and "
            "echoed). With few borings these cannot be checked against the data and usually move "
            "the answer more than the method does &mdash; run cases.",
            "Plan view (2D) only; inequality data (a boring that ended above rock) by rejection "
            "sampling.",
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
            "Ready-made library functions (not agent tools): <code>bearing_capacity_reliability</code>, "
            "<code>axial_pile_reliability</code>, <code>slope_reliability</code>.",
            "References: Duncan (2000); Phoon &amp; Kulhawy (1999); Baecher &amp; Christian; Vanmarcke "
            "(1977); validated vs Duncan (2000) reliability example.",
        ],
        "limits": [
            "The performance function is a scalar g(values) of the named variables; a depth-varying "
            "undrained strength is varied as one correlated (a, b) pair through slope_stability&rsquo;s "
            "probabilistic FOS (<code>linear_su</code>, V-030), not through this engine.",
        ],
    },

    "salib": {
        "problems": "Which uncertain input actually governs the answer? Global sensitivity analysis "
            "apportions the variance of a model output to its inputs, so investigation and design effort "
            "go where they matter.",
        "methods": [
            "**Sobol** variance-based indices (first-order and total) and **Morris** elementary-effects "
            "screening, wrapping **SALib**; sample generation and index computation.",
            "References: Sobol; Morris; the SALib library (installed with the app).",
        ],
        "limits": [],
    },

    "pystra": {
        "problems": "Structural reliability for an explicit limit-state function: the reliability index and "
            "probability of failure by FORM, SORM, or Monte Carlo, with normal, lognormal, Gumbel, "
            "uniform, Weibull, gamma, beta and constant variables and a correlation matrix.",
        "methods": [
            "**FORM / SORM / Monte Carlo** on a user limit-state function, wrapping **pystra** (built on PyRe "
            "and FERUM).",
            "References: the pystra library (installed with the app).",
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
            "soil nails; conversion to <code>SlopeGeometry</code> and FEM inputs.",
            "References: the ezdxf library (installed with the app).",
        ],
        "limits": [
            "Workflow is discover&nbsp;&rarr;&nbsp;parse&nbsp;&rarr;&nbsp;build; the layer&rarr;role "
            "mapping is an explicit step so the engineer confirms what each layer represents.",
        ],
    },

    "pdf_import": {
        "problems": "Extract a cross-section from a PDF drawing when there is no CAD file: vector geometry "
            "from the PDF&rsquo;s line work, which can be checked against a vision read made separately.",
        "methods": [
            "**Vector extraction** (PyMuPDF <code>get_drawings()</code>, grouped by stroke colour) for "
            "exact line work, from the separate <code>planlens</code> package (<code>planlens.pdf</code>).",
            "Scale from two points and a known distance, or proposed from scale notes on the page; a "
            "proposed colour&rarr;role mapping from the labels; geometry cleanup; a feature-by-feature "
            "cross-check of the vector result against a vision extraction "
            "(<code>planlens.pdf.extract_geometry_vision</code>, outside this tool); conversion to "
            "<code>SlopeGeometry</code> and fem2d inputs.",
            "References: the PyMuPDF library; planlens. Both are installed with the app.",
        ],
        "limits": [
            "Vision extraction depends on the engine&rsquo;s image quality; always confirm the extracted "
            "geometry against the drawing before analysis.",
        ],
    },

    "dxf_export": {
        "problems": "Send a cross-section built or checked in the toolkit back to CAD: write the surface, "
            "soil boundaries, water table, soil nails, and annotations to a DXF. "
            "Draw a sheet: a typed description of a drawing (outlines, bars, dimensions, callouts, notes, "
            "a table) written by fixed code as a DXF with real dimension, hatch and leader objects, and "
            "the same drawing as a PDF sheet and a PNG. The first drawing is the reinforced cantilever "
            "retaining wall section.",
        "methods": [
            "Geometry-to-DXF export of surface, boundaries, GWT, soil nails, and text annotations.",
            "Drawing sheets (<code>draw_sheet</code>, <code>draw_wall_section</code>): model space in "
            "millimetres at true size, the sheet drawn round it at a standard scale the code chooses; "
            "the PDF page and the PNG are drawn from the DXF itself. A dimension prints its own "
            "measurement.",
            "The wall section turns a hooked stem bar toward the opposite face of the stem, the correct "
            "orientation of USACE EM 1110-2-2104 (2023) Figure 2-1; the hook&rsquo;s bend and tail are "
            "schematic.",
            "References: the ezdxf library (installed with the app).",
        ],
        "limits": [],
    },

    "calc_package": {
        "problems": "Produce a reviewable calculation package from a module result: a Mathcad-style "
            "report with inputs, equations, figures, and results &mdash; the documentation a calc set "
            "needs.",
        "methods": [
            "Jinja2-templated **HTML** and **LaTeX** calc packages for 17 analysis modules "
            "(<code>calc_package.list_supported_modules()</code>), with matplotlib figures (the Plotly "
            "interactive viewers in <code>calc_package.interactive</code> are library functions with no agent tool); PDF by <code>pdflatex</code> where it is installed and otherwise by "
            "PyMuPDF&rsquo;s HTML engine (simpler layout). <code>html_to_pdf</code> turns a report you "
            "compose in HTML into a PDF the same way.",
            "**Calculation tables** in the style of Ensoft&rsquo;s output: a printed &ldquo;Calculation "
            "Tables&rdquo; section and one CSV per table, units in the headers, for lateral and axial "
            "piles, pile groups, drilled shafts, settlement, bearing capacity, sheet piles, slope "
            "stability, downdrag, fem2d, liquefaction, the wave-equation bearing graph, ground "
            "improvement, support of excavation and reliability.",
            "References: Jinja2, matplotlib and Plotly, all installed with the app.",
        ],
        "limits": [
            "Without <code>pdflatex</code> (Databricks, Foundry) the PDF comes from PyMuPDF&rsquo;s HTML "
            "engine, a simpler layout with a CSS subset; the result names the engine used.",
        ],
    },
}


# ===========================================================================
# Validation basis (Chapter 8). Summary + a representative row set.
# ===========================================================================

VALIDATION_KPIS = [
    ("63", "published problems checked"),
    ("107", "individual module-vs-source comparisons"),
    ("79", "PASS (within stated tolerance)"),
    ("0", "unexplained discrepancies"),
]

VALIDATION_INTRO = (
    "Beyond the per-module unit suites (which regress each method against textbook answers), "
    "the package carries a <em>published-example</em> validation layer: real worked examples "
    "from the design manuals and FE verification suites, with complete numeric inputs "
    "<em>and</em> published numeric answers, run through the analysis modules offline. The "
    "problems (V-001 onward) and their sources are in <code>validation_examples/INVENTORY.md</code>; "
    "the run-by-run verdicts are in <code>validation_examples/RESULTS.md</code>. Flagship "
    "modules additionally carry a <code>VALIDATION.md</code> (slope_stability vs Fredlund &amp; "
    "Krahn 1977 / ACADS / Duncan; fem2d vs Griffiths &amp; Lane and the Prandtl solution; "
    "lateral_pile vs a COM624P oracle). Verdicts are graded honestly: <strong>PASS</strong> "
    "(reproduces the published answer within tolerance), <strong>CONVENTION</strong> (a "
    "defensibly different but documented convention &mdash; the module is not tuned to the one "
    "example), and <strong>N/A&nbsp;(scope)</strong> (the module implements a different method "
    "than the example &mdash; a documented coverage gap, not a failure). Across 107 comparisons "
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
    ("V-016", "soe", "Two-tier anchored wall &mdash; apparent envelope, anchor loads", "GEC-4 Ex 1", "PASS"),
    ("V-017", "lateral_pile", "Laterally loaded micropile (LPILE benchmark)", "FHWA Micropile Ex 2", "PASS"),
    ("V-018", "ground_improvement", "Rammed aggregate pier settlement + baseline consolidation", "GEC-13 Ex 1", "PASS / N/A"),
    ("V-019", "ground_improvement", "Stone-column settlement (Priebe curve)", "GEC-13 Ex 2", "PASS"),
    ("V-020", "ground_improvement", "PVD time to 90% consolidation (GEC-13 Eq. 2-1)", "GEC-13 Ch 2", "PASS"),
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
    ("5.4.0 &ndash; 5.35.0", "2026-07 &ndash; 2026-11",
     "Not yet summarised in this appendix. Each release&rsquo;s notes, including the changes that "
     "move a calculation&rsquo;s answer, are in the repository&rsquo;s <code>HANDOFF.md</code> and "
     "<code>CLAUDE.md</code>; the method and parameter tables in this manual are generated from "
     "the release it is built from."),
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
        "references; the chat web app (<code>webapp</code>, two pages: GeotechStaffEngineer and "
        "Document Review); and below both the shared core <code>aec-agent-core</code> (model "
        "connections, vision calls, guards and records, Word and Excel), a separate package this one "
        "pins exactly</td><td>The reasoning that <em>combines</em> methods lives here, not inside the "
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
    ctx.raw("<pre class='call'><code># the whole stack: analysis modules, reference library, deep agent, webapp\n"
            "pip install geotech-staff-engineer\n\n"
            "# Document Review alone, with nothing geotechnical (a separate package)\n"
            "pip install aec-document-review</code></pre>")
    ctx.para(
        "The plain install already covers the native modules &mdash; bearing capacity, settlement, piles, "
        "walls, slope stability, the reliability engine, fem2d, and the reference library "
        "(<code>geotech-references</code> is a hard dependency, so DM7/GEC/UFC lookups work out of the "
        "box). It also brings two other packages of the same project: the shared core "
        "<code>aec-agent-core</code> (pinned exactly) and <code>planlens</code>. Since 5.8.1 the plain "
        "install also brings the third-party libraries the wrapper modules drive; the extra names below "
        "(<code>[deep]</code>, <code>[full]</code> and the rest) are kept as empty aliases so old install "
        "lines still work. Three extras still add a package: <code>[ags4]</code> and "
        "<code>[subsurface]</code> (python-ags4) and <code>[pydiggs]</code>.")

    ctx.section("The extras groups")
    rows = ["<table><thead><tr><th>Extra</th><th>Pulls in</th><th>Unlocks</th></tr></thead><tbody>"]
    unlocks = {
        "plot": "matplotlib figures in calc packages and module plots",
        "calc": "Jinja2-templated HTML/LaTeX calc packages",
        "interactive": "Plotly single-file interactive viewers",
        "pdf": "PDF cross-section import (vector)",
        "dxf": "DXF CAD import and export",
        "opensees": "OpenSeesPy nonlinear FE (PM4Sand, 1D site response)",
        "pystrata": "equivalent-linear 1D site response",
        "seismic-signals": "eqsig ground-motion processing (RotD computed in the time domain by the app's own code)",
        "liquepy": "Boulanger &amp; Idriss liquepy triggering",
        "gstools": "geostatistical kriging, random fields and conditional simulation",
        "salib": "Sobol / Morris sensitivity analysis",
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
        "<p>A method whose backend is not installed (python-ags4 or pydiggs without their extra) fails "
        "cleanly with a &ldquo;not installed&rdquo; message rather than a confusing traceback; a library a "
        "host cannot install (PyNiteFEA on numpy 1.26) leaves its tool out.</p>")

    ctx.section("Local Python")
    ctx.para(
        "Requires Python 3.11 or newer (the shared core and deepagents need it). The package is pure "
        "Python; the only heavy dependencies are numpy and "
        "scipy. For the LLM agent you need an engine (see the next chapter) and, for a Claude engine, "
        "an <code>ANTHROPIC_API_KEY</code> in the environment. Keys are read from the environment at "
        "run time &mdash; never pass them through a notebook cell or a chat transcript.")

    ctx.section("The chat web app")
    ctx.para(
        "The install includes the Streamlit chat app: two pages over one shell, Document Review (the "
        "app root) and GeotechStaffEngineer (<code>/geotech</code>). Its files are inside the installed "
        "package, so point Streamlit at them:")
    ctx.raw("<pre class='call'><code>streamlit run \"$(python -c 'import webapp, os; "
            "print(os.path.dirname(webapp.__file__))')/pages_entry.py\"</code></pre>")
    ctx.para(
        "Put <code>app.py</code> in place of <code>pages_entry.py</code> for the geotech page alone; in "
        "Windows PowerShell write <code>\\pages_entry.py</code> after the closing parenthesis. "
        "Databricks, Foundry and the app&rsquo;s settings are in the repository&rsquo;s "
        "<code>webapp/README.md</code>.")

    ctx.section("Databricks / Funhouse")
    ctx.para(
        "Install from a cluster-accessible location &mdash; <code>/tmp</code> or a Unity Catalog "
        "Volume, <em>not</em> <code>/Workspace</code> (the workspace FUSE mount mangles wheel filenames "
        "and does not durably store output files):")
    ctx.raw("<pre class='call'><code>%pip install \"geotech-staff-engineer\"\n"
            "# or a test wheel uploaded to /tmp or a UC Volume; when the test set carries\n"
            "# the core's wheel too, put it on the same line:\n"
            "%pip install \"/tmp/geotech_staff_engineer-&lt;ver&gt;-py3-none-any.whl\" "
            "\"/tmp/aec_agent_core-&lt;core ver&gt;-py3-none-any.whl\"</code></pre>")
    ctx.callout("No compiled meshing dependency",
        "<p>The cross-section and RC-section engines used to wrap "
        "<code>sectionproperties</code> and <code>concreteproperties</code>, which both "
        "require <code>cytriangle</code> &mdash; a compiled wheel corporate package proxies "
        "quarantine. A Nexus malware-defense 403 on it failed the entire 5.12.0 install on "
        "the cluster, so in 5.13.0 both were re-implemented natively on numpy/scipy: exact "
        "closed-form integration for the geometry, published closed forms for torsion, and "
        "ACI strain compatibility for the RC section. The retired libraries&rsquo; outputs were "
        "pinned as regression oracles first; every RC scalar agrees to better than 0.04%.</p>")
    ctx.callout("The typing_extensions / restart-Python note (now handled automatically)",
        "<p>Databricks cluster runtimes pre-import an old <code>typing_extensions</code> (&lt;4.13) at "
        "kernel startup, which used to break the langgraph imports behind the <code>[deep]</code> agent "
        "with <code>TypeError: TypedDict() got an unexpected keyword argument 'extra_items'</code>, "
        "requiring <code>dbutils.library.restartPython()</code>. As of v5.4 the package repairs this in "
        "place: <code>ensure_typing_extensions()</code> (in the shared core&rsquo;s "
        "<code>aec_agent_core.runtime_check</code>; the old name <code>funhouse_agent.runtime_check</code> "
        "is an alias) runs automatically "
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
        "The primary agent does not call the reference modules directly. Reference access is routed "
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
        "The plain install includes <code>funhouse_agent.deep.build_deep_agent</code>, "
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

    # native correlations note
    ctx.section("Also available: native soil correlations")
    ctx.para(
        "SPT field corrections (N &rarr; N60 &rarr; (N1)60 per Youd et al. 2001 / "
        "Liao &amp; Whitman), Bolton stress-dilatancy, the small-strain Gmax family "
        "(Vs, CPT, Hardin-Black, Andersen), Ishibashi &amp; Zhang modulus-reduction/damping "
        "curves, and Brinkgreve HS-parameter estimation are implemented natively "
        "(<code>geotech_common.soil_properties</code>, "
        "<code>seismic_geotech.dynamic_properties</code>, "
        "<code>fem2d.hs_correlations</code>) and exposed through the subsurface, "
        "seismic, and fem2d agents. The former third-party groundhog wrapper was "
        "removed in favor of these.")


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
        "soil boundaries / water table / soil nails, then build the model. ezdxf is installed "
        "with the app.")

    ctx.section("PDF cross-section import")
    ctx.para(
        "<code>pdf_import</code> extracts geometry from a PDF drawing by <strong>vector "
        "extraction</strong> (PyMuPDF <code>get_drawings()</code> for exact line work, through the "
        "<code>planlens</code> package). It calibrates the scale from two points and a known distance or "
        "proposes it from the scale notes, proposes a colour&ndash;to&ndash;role mapping from the labels, "
        "cleans up the geometry, and cross-checks the vector result against a vision extraction made "
        "outside the tool, so the two methods can confirm each other.")

    ctx.section("DXF export")
    ctx.para(
        "<code>dxf_export</code> writes a cross-section back out to DXF &mdash; surface, soil boundaries, "
        "water table, soil nails, and annotations &mdash; so a model built or checked in the toolkit "
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
        "HTML or LaTeX, with figures and equation traces &mdash; for 17 of the analysis modules. It "
        "emits PDF by <code>pdflatex</code> where that is installed and otherwise by PyMuPDF&rsquo;s HTML "
        "engine (Databricks, Foundry), a simpler layout.")


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

    ctx.section("Published-example validation")
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
        "footing, the packaged high-level call applies Brinch Hansen&rsquo;s depth factors while the "
        "example takes d<sub>q</sub> = 1.0 and GEC-6&rsquo;s C<sub>w</sub> groundwater factors; assembled the "
        "example&rsquo;s way, the module recovers the published closed form exactly.</p>", kind="warn")
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
        "<code>wave_equation</code> models every hammer as a rigid ram at its rated energy; a diesel&rsquo;s "
        "combustion and changing stroke are not modelled and the result warns that its blow counts are "
        "likely low (GEC-12 Vol. 2 p. 206); treat diesel bearing graphs as a trend, not a tie-out.",
        "<code>slope_stability</code>&rsquo;s tension crack keeps the cracked wedge as zero-strength "
        "driving soil by default (a conservative convention); <code>tension_crack_model='truncation'</code> "
        "removes it from the sliding mass as Slide2 does.",
        "<code>ground_improvement</code> uses an equal-strain settlement-reduction model for aggregate "
        "piers, not the GEC-13 stiffness-modulus two-layer method; the Priebe curve is GEC-13's "
        "printed curve (&phi; = 37.5&deg;) and depends on the area-ratio convention you feed it.",
        "The high-level <code>bearing_capacity</code> call applies Brinch Hansen&rsquo;s depth factors "
        "by default (<code>depth_factor_method</code>) and an effective-unit-weight groundwater model "
        "&mdash; different from GEC-6&rsquo;s C<sub>w</sub> groundwater factors, which are not offered; "
        "know which convention your check needs.",
        "Soldier-pile effective widths and arching are not modelled in <code>soe</code> or "
        "<code>sheet_pile</code> (continuous walls per metre run); <code>sheet_pile</code>&rsquo;s anchored "
        "embedment is classical free earth support, while the FHWA apparent-diagram embedment is in "
        "<code>soe</code>.",
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
        "between minor versions are tracked in <code>CLAUDE.md</code> and <code>HANDOFF.md</code>; "
        "<code>docs/V5.1_SUMMARY.html</code> covers 5.0 to 5.1 only.", cls="lead")
    rows = ["<table><thead><tr><th>Version</th><th>Date</th><th>Highlights</th></tr></thead><tbody>"]
    for ver, dt, note in VERSION_HISTORY:
        rows.append(f"<tr><td><strong>{ver}</strong></td><td class='small'>{dt}</td><td>{note}</td></tr>")
    rows.append("</tbody></table>")
    ctx.raw("".join(rows))
