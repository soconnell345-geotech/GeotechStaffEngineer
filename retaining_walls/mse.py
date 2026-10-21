"""
MSE (Mechanically Stabilized Earth) wall analysis.

Implements external stability (sliding, overturning, bearing) and
internal stability (tensile rupture, pullout) per FHWA GEC-11.

All units are SI: kN, kPa, kN/m³, degrees, meters.

References:
    FHWA GEC-11 (Berg, Christopher & Samtani 2009), a TWO-VOLUME publication:
      - Vol I  = FHWA-NHI-10-024 — design method, Chapters 4-5 (in the in-house
                 geotech-references `gec_11` module, ch 1-11).
      - Vol II = FHWA-NHI-10-025 — appendices incl. Appendix E worked example
                 E4 (the external-stability validation anchor). Vol II is NOT in
                 the in-house reference module.
    Both citations below are correct — they name the two volumes, not a typo.
    AASHTO LRFD Section 11.10

    Sloping backfill and broken backslope (GM-112): GEC-11 Vol I §4.4.4
    (Eqs. 4-2, 4-3, 4-7; Figs. 4-3, 4-4), §4.4.6 (Eqs. 4-10 to 4-21; Fig.
    4-8) and §4.4.7 (Eq. 4-29; Figs. 4-9a, 4-11, 4-15), reproduced against
    Vol II Examples E3 (a slope that continues) and E1 (a broken backslope);
    see DESIGN.md.

    A live load over the reinforced zone (GM-164): ``geom.surcharge`` is the
    live load (traffic). On the ASD path (``check_external_stability``) it is
    not a resisting force in sliding or overturning by default, and it does
    load the base for bearing, as GEC-11 Vol I pp. 4-19, 4-21, 4-22 and 4-26
    (Fig. 4-2) have it and as the LRFD path already did. Its thrust drives
    every check. ``live_load_resists`` and ``surcharge_in_bearing`` choose
    the other readings, and the result's ``choices`` say which was used and
    why (:data:`LIVE_LOAD_BASIS`, :data:`BEARING_BASIS`). The bearing
    eccentricity is therefore on other loads than the overturning check's
    (GEC-11 p. 4-26: "eB = eccentricity for bearing calculation (not equal
    to eccentricity check e)").

    Source basis (Kr/K0 ratio and F* pullout curves): DIGITIZED from GEC-11
    Figs 4-10/4-11/E4-5 (also AASHTO Fig. 11.10.6.2.1-3). Authoring-time in-hand
    status not recorded, but the curves are anchored by the GEC-11 Example E4
    reproduction (CDRs and F* within ~3%; see tests). The built-in
    reinforcements are NOT a manufacturer's catalog, each is a GEC-11 Vol
    II worked example's as printed: the W11 bar mat is Example E4's 4W11
    mat, its Tal from E4's corroded wire (GM-262); the strip is Example
    E3's 50 x 4 mm ribbed strip, Tal 0.55 Fy Ec at E3's corroded thickness
    (GM-267); the geogrids are Example E1's GG-I to GG-III, Tal = Tult / RF
    from Table E1-7.3 (GM-271) — see DESIGN.md.
"""

import math
from typing import Dict, Any, List, Optional, Sequence

from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.earth_pressure import (
    rankine_Ka, coulomb_Ka, horizontal_force_active, active_force_components,
)
from retaining_walls.reinforcement import (
    BAR_MAT_ALLOWABLE_FY, STRIP_ALLOWABLE_FY, Reinforcement,
)
from retaining_walls.results import MSEWallResult


# 20 ft = 6.096 m: the GEC-11 / AASHTO depth below which Kr/Ka and F* become
# constant for inextensible (metallic) reinforcement.
_Z20_FT_M = 20.0 * 0.3048  # 6.096 m

# GEC-11 Vol I (FHWA-NHI-10-024) p. 3-20, Eq. 3-4: for ribbed steel strips
# F* = 1.2 + log Cu at the top of the structure, 2.0 maximum, Cu the
# backfill's uniformity coefficient D60/D10; "If the specific Cu for the
# wall backfill is unknown at design time, a Cu = 4 should be assumed
# (i.e., F* = 1.8 at the top of the wall)" (GM-246).
CU_UNKNOWN = 4.0
F_STAR_STRIP_MAX = 2.0

#: GEC-11 Vol I Table 3-6 note (i) (p. 3-17), read on the rendered page:
#: it covers every empirical F* in the table (strips, grids, sheets), so a
#: given Cu is checked against it whatever the reinforcement (GM-246).
F_STAR_NOTE_I = (
    "GEC-11 Vol I Table 3-6 note (i) (p. 3-17): the empirical F* values "
    "are acceptable 'in the absence of product and backfill specific test "
    "data, provided granular reinforced fill as specified in Table 3-1 for "
    "MSE walls is used and Cu >= 4. For fill outside these limits, tests "
    "must be run.'")
_NOTE_I_WITHIN = ("at least 4, within the fill for which GEC-11 Table 3-6 "
                  "note (i) accepts the empirical F*")
_NOTE_I_OUTSIDE = ("below 4, outside the fill for which GEC-11 Table 3-6 "
                   "note (i) accepts the empirical F*: 'For fill outside "
                   "these limits, tests must be run'")
_NOTE_I_TESTS = ("pullout test data for this reinforcement in the wall's "
                 "backfill in place of the empirical F*, which note (i) "
                 "requires outside these limits (GEC-11 Vol I p. 3-20: the "
                 "procedures are in Appendix B)")

F_STAR_STRIP_BASIS = (
    "GEC-11 Vol I (FHWA-NHI-10-024) p. 3-20, for steel ribbed "
    "reinforcement: Eq. 3-4, F* = 1.2 + log Cu at the top of the "
    "structure, 2.0 maximum; Eq. 3-5, F* = tan phi at a depth of 20 ft "
    "(6 m) and below; Cu is the uniformity coefficient of the backfill "
    "(D60/D10). 'If the specific Cu for the wall backfill is unknown at "
    "design time, a Cu = 4 should be assumed (i.e., F* = 1.8 at the top of "
    "the wall) for reinforced fills meeting the requirements of Section "
    "3.1 of this chapter.' " + F_STAR_NOTE_I + " F* is taken linearly "
    "between, as Vol II Examples E3 and E5 interpolate it, over the same "
    "depths as Kr/Ka (p. 4-49)")

#: GEC-11 Vol I (FHWA-NHI-10-024), read on the rendered pages (GM-256).
COVERAGE_BASIS = (
    "GEC-11 Vol I Eq. 3-1 (p. 3-12): Rc = b/Sh, b 'the gross width of the "
    "strip, sheet, or grid' (for grids from the centre to centre of the "
    "outside longitudinal bars) and Sh the 'center-to-center horizontal "
    "spacing between strips, sheets, or grids'; p. 3-13: 'Rc = 1 in the "
    "case of continuous reinforcement, i.e., each reinforcement layer "
    "covers the entire horizontal surface of the reinforced soil zone'. "
    "Eq. 3-2 (p. 3-16) gives the pullout resistance per unit width of "
    "reinforcement, and Eq. 4-35 (p. 4-49) checks it per unit width of "
    "wall as F* alpha sigma_v Le C Rc; the tensile resistance per unit "
    "width of wall is the reinforcement's Tal times Rc (p. 4-47)")

#: Typical spacings GEC-11 Vol II prints, named in a refusal and never
#: used for one.
_TYPICAL_COVERAGE = (
    "The spacings in GEC-11's worked examples, for comparison only (the "
    "tool assumes none of them): Vol II Example E3 (pp. E3-2, E3-19) "
    "uses 50 mm (1.969 in.) ribbed strips at Sh 0.76 m and 0.51 m (2.50 and "
    "1.67 ft: 2 or 3 strips per 1.5 m panel), Rc 0.066 and 0.098, with 'a "
    "minimum 2 strips at a horizontal spacing not exceeding 2.5 ft'; "
    "Example E5 (p. E5-30) Sh 0.40 to 0.52 m (1.3 to 1.7 ft); Example E4 "
    "(p. E4-18) bar mats of 4 longitudinal W11 wires at 0.15 m (0.5 ft), b "
    "0.46 m between the outside wires, one mat per 1.5 m (5 ft) panel, Rc "
    "0.30")

NEEDS_COVERAGE = (
    "A {kind} needs its coverage ratio Rc = b/Sh: a steel strip or grid "
    "covers only part of each level, and GEC-11 Vol I takes its pullout "
    "and tensile resistance per unit width of wall times Rc (Eq. 3-1, p. "
    "3-12; Eq. 4-35, p. 4-49; p. 4-47). Give coverage_ratio, or "
    "reinforcement_horizontal_spacing (Sh, m, centre to centre) with "
    "reinforcement_width (b, m{width}). " + _TYPICAL_COVERAGE)

#: GEC-11 Vol I p. 3-12 describes a layer of geosynthetic strips 'by the
#: width of the strips and the center-to-center horizontal distance between
#: them': a width with no spacing leaves Rc unknown (GM-256 review).
GEOSYNTHETIC_WIDTH_NEEDS_SH = (
    "the geosynthetic's width b {b:g} m (reinforcement_width) is used only "
    "for Rc = b/Sh (GEC-11 Vol I Eq. 3-1, p. 3-12) and needs its horizontal "
    "spacing Sh (reinforcement_horizontal_spacing, m, centre to centre); a "
    "continuous layer, Rc = 1 (p. 3-13), takes neither")


def coverage_ratio(reinforcement: Reinforcement) -> float:
    """The coverage ratio Rc = b/Sh of ``reinforcement`` (GEC-11 Vol I Eq.
    3-1, p. 3-12; GM-256): its ``coverage_ratio`` when given; else its
    width b over its horizontal spacing Sh; else 1 for a geosynthetic, a
    continuous layer ("Rc = 1 in the case of continuous reinforcement", p.
    3-13); a width with no spacing does not change that (the choice says so,
    and the tool refuses it: :data:`GEOSYNTHETIC_WIDTH_NEEDS_SH`). A steel
    strip or grid given neither is refused (ValueError,
    :data:`NEEDS_COVERAGE`): GEC-11 prints no spacing to assume, and taking
    Rc = 1 overstated a strip wall's pullout and rupture many times (before
    GM-256). One value for the wall: a spacing or ratio given per level is
    taken by :func:`coverage_ratios` (GM-261)."""
    r = reinforcement
    _not_both(r)
    if r.per_level:
        raise ValueError(
            f"{r.name}'s coverage is given per reinforcement level: take "
            "it with coverage_ratios(reinforcement, depths) (GM-261)")
    return _coverage_at(r, r.coverage_ratio, r.horizontal_spacing)


def coverage_ratios(reinforcement: Reinforcement,
                    depths: Sequence[float]) -> List[float]:
    """The coverage ratio Rc = b/Sh at each reinforcement level, the
    levels at ``depths`` (m) from the top down (GM-261): a spacing or
    coverage ratio given per level takes one value a level, as GEC-11 Vol
    II Example E3 changes Sh with depth (Table E3-7.3, p. E3-19); a single
    value holds at every level. A list whose length is not the number of
    levels is refused, naming the levels; otherwise as
    :func:`coverage_ratio`. A reinforcement given per level (a list, one a
    level; :func:`level_reinforcements`, GM-266) takes each level's own
    width and Sh or Rc."""
    r = reinforcement
    if not isinstance(r, Reinforcement):
        out = []
        for i, ri in enumerate(level_reinforcements(r, depths), start=1):
            try:
                _not_both(ri)
                out.append(_coverage_at(ri, ri.coverage_ratio,
                                        ri.horizontal_spacing, level=i))
            except ValueError as e:
                raise ValueError(f"level {i} from the top ({ri.name}): "
                                 f"{e}") from None
        return out
    _not_both(r)
    n = len(depths)
    for name, given, values in (
            ("reinforcement_horizontal_spacing", "Sh", r.horizontal_spacing),
            ("coverage_ratio", "Rc", r.coverage_ratio)):
        if isinstance(values, tuple) and len(values) != n:
            raise ValueError(
                f"{name} gives {len(values)} values of {given}, one per "
                "reinforcement level from the top down, and this wall has "
                f"{n} levels, at depths "
                + ", ".join(f"{z:.2f}" for z in depths) + " m (the "
                "reinforcement spacing Sv sets them)")

    def at(values, i):
        return values[i] if isinstance(values, tuple) else values
    return [_coverage_at(r, at(r.coverage_ratio, i),
                         at(r.horizontal_spacing, i),
                         level=i + 1 if r.per_level else None)
            for i in range(n)]


#: GEC-11 Vol I, read on the rendered page (GM-266). The refusal is this
#: tool's choice, not a rule GEC-11 states: GEC-11 draws the surface (and
#: Kr/Ka, F*) for one kind or the other and is silent on a wall with both.
ONE_KIND_PER_WALL = (
    "this tool takes a reinforcement per level only when every level is "
    "inextensible (steel) or every level extensible (geosynthetic), since "
    "the internal checks need one critical failure surface: GEC-11 Vol I "
    "p. 4-33 takes it as 'approximately bilinear in the case of "
    "inextensible reinforcements' and 'approximately linear in the case of "
    "extensible reinforcements' (Fig. 4-9, p. 4-34), through the toe of "
    "the wall in both cases, and gives no surface for a wall that mixes "
    "them; got {got}. Run the wall with one kind, or check a mixed wall "
    "outside this tool")


def level_reinforcements(reinforcement, depths: Sequence[float]
                         ) -> List[Reinforcement]:
    """The reinforcement at each level, the levels at ``depths`` (m) from
    the top down (GM-266): one :class:`Reinforcement` at every level, or a
    list of them, one a level, as GEC-11 Vol II Example E4 takes 4W11 mats
    at Levels 1, 3 and 4, a 3W11 mat at Level 2 and 4W15 mats from Level 5
    down (Table E4-7.4, p. E4-18). A list is refused when its length is not
    the number of levels (naming the levels), when it mixes steel with
    geosynthetic reinforcement (:data:`ONE_KIND_PER_WALL`), and when an
    entry gives its spacing or coverage ratio per level: each entry is one
    level's, with its own Sh or Rc."""
    if isinstance(reinforcement, Reinforcement):
        return [reinforcement] * len(depths)
    rs = (list(reinforcement)
          if isinstance(reinforcement, (list, tuple)) else [])
    if not rs or not all(isinstance(r, Reinforcement) for r in rs):
        raise ValueError(
            "reinforcement is one Reinforcement for every level, or a list "
            "of them, one per reinforcement level from the top down; got "
            f"{reinforcement!r}")
    n = len(depths)
    if len(rs) != n:
        raise ValueError(
            f"reinforcement_per_level gives {len(rs)} reinforcements, one "
            "per reinforcement level from the top down, and this wall has "
            f"{n} levels, at depths " + ", ".join(f"{z:.2f}" for z in depths)
            + " m (the reinforcement spacing Sv sets them)")
    if len({r.is_metallic for r in rs}) > 1:
        kinds = ["steel" if r.is_metallic else "geosynthetic" for r in rs]
        raise ValueError(ONE_KIND_PER_WALL.format(got=" and ".join(
            f"{kind} at {_levels_of(spans)}"
            for kind, spans in _grouped(kinds))))
    for i, r in enumerate(rs, start=1):
        if r.per_level:
            raise ValueError(
                f"level {i}'s reinforcement ({r.name}) gives its horizontal "
                "spacing or coverage ratio per level; in a reinforcement "
                "given per level each entry is one level's, with one Sh or "
                "Rc")
    return rs


def _grouped(values, key=None):
    """Each distinct value (by ``key``) with the spans of levels holding
    it, in the order the values first appear from the top: [(value,
    [(first, last), ...]), ...], levels counted from 1 (GM-266)."""
    key = key or (lambda v: v)
    out = []
    for first, last, v in _level_runs([key(v) for v in values]):
        for entry in out:
            if entry[0] == v:
                entry[1].append((first, last))
                break
        else:
            out.append((v, [(first, last)]))
    # the values themselves, not their keys
    firsts = {}
    for i, v in enumerate(values, start=1):
        firsts.setdefault(key(v), v)
    return [(firsts[k], spans) for k, spans in out]


def _levels_of(spans):
    """'level 2', 'levels 5-7' or 'levels 1, 3-4'."""
    if len(spans) == 1:
        return _levels(*spans[0])
    return "levels " + ", ".join(str(a) if a == z else f"{a}-{z}"
                                 for a, z in spans)


def _per_level_text(values, unit=""):
    """A per-level input in a message: the one value, or the list."""
    if isinstance(values, tuple):
        return ("[" + ", ".join(f"{v:g}" for v in values) + "]"
                + (f" {unit}" if unit else "") + " per level")
    return f"{values:g}" + (f" {unit}" if unit else "")


def _not_both(r: Reinforcement) -> None:
    if r.coverage_ratio is not None and r.horizontal_spacing is not None:
        raise ValueError(
            "give the coverage ratio (coverage_ratio) or the horizontal "
            "spacing (reinforcement_horizontal_spacing) with the width, not "
            f"both: Rc {_per_level_text(r.coverage_ratio)} against b/Sh "
            f"from Sh {_per_level_text(r.horizontal_spacing, 'm')}")


def _coverage_at(r: Reinforcement, Rc, Sh, level=None) -> float:
    """Rc at one level from its given Rc or Sh (GM-256, GM-261)."""
    b = r.width
    if Rc is not None:
        return Rc
    kind = ("steel grid" if r.is_grid else "steel strip"
            if r.is_metallic else "geosynthetic")
    if Sh is not None:
        if b is None:
            raise ValueError(
                f"reinforcement_horizontal_spacing Sh "
                f"{_per_level_text(r.horizontal_spacing, 'm')} needs the "
                f"{kind}'s width b (reinforcement_width, m"
                + (", centre to centre of the outside longitudinal bars"
                   if r.is_grid else "") + "): Rc = b/Sh (GEC-11 Vol I "
                "Eq. 3-1, p. 3-12). " + _TYPICAL_COVERAGE)
        if b > Sh:
            raise ValueError(
                f"the {kind}'s width b {b:g} m is more than its horizontal "
                f"spacing Sh {Sh:g} m"
                + (f" at level {level} from the top" if level else "")
                + ": Rc = b/Sh is at most 1 (GEC-11 Vol I Eq. 3-1; Rc = 1 "
                "is a continuous layer, p. 3-13)")
        return b / Sh
    if not r.is_metallic:
        return 1.0
    width = ""
    if b is not None:
        width = (f"; this {'grid' if r.is_grid else 'strip'} is {b:g} m "
                 "wide, so Sh alone will do")
    elif r.is_grid:
        width = ", centre to centre of the outside longitudinal bars"
    raise ValueError(NEEDS_COVERAGE.format(kind=kind, width=width))


def coverage_choice(reinforcement: Reinforcement,
                    internal: Optional[List[Dict[str, Any]]] = None) -> dict:
    """The coverage ratio Rc the internal checks took, as a ``choices``
    entry (``geotech_common.choices``; GM-256), with its b and Sh."""
    from geotech_common.choices import choice, how_given
    r = reinforcement
    basis = COVERAGE_BASIS
    if not isinstance(r, Reinforcement):
        return _level_list_coverage_choice(r, internal, basis)
    if r.per_level:
        return _per_level_coverage_choice(r, internal, basis)
    Rc = coverage_ratio(r)
    alts = []
    if r.coverage_ratio is not None:
        chosen = f"Rc {Rc:.3g}, given"
        how = how_given(Rc, None, name="coverage_ratio")
    elif r.horizontal_spacing is not None:
        chosen = (f"Rc {Rc:.3g} = b {r.width:g} m / Sh "
                  f"{r.horizontal_spacing:g} m")
        how = how_given(r.horizontal_spacing, None,
                        name="reinforcement_horizontal_spacing")
    else:
        chosen = "Rc 1, a continuous geosynthetic layer"
        if r.width is not None:
            chosen += (f" (its width b {r.width:g} m is not used: Rc = b/Sh "
                       "needs a horizontal spacing)")
        how = how_given(None, None, name="coverage_ratio")
        alts.append(
            "geosynthetic strips at less than full coverage: Rc = b/Sh "
            "(reinforcement_width with reinforcement_horizontal_spacing, or "
            "coverage_ratio); the pullout and tensile resistance per metre "
            "of wall then fall in proportion")
    if internal:
        chosen += (f"; tensile resistance Tal x Rc "
                   f"{r.Tallowable * Rc:.4g} kN/m of wall (Tal "
                   f"{r.Tallowable:.4g} kN/m of reinforcement width)")
    return choice(
        "Coverage ratio Rc of the reinforcement (pullout and tensile "
        "resistance per metre of wall)", chosen, how, basis, alts)


#: GEC-11 Vol II, read on the rendered page (GM-261).
PER_LEVEL_BASIS = (
    "; a spacing per level as GEC-11 Vol II Example E3 takes it: 'The last "
    "column of Table E3-7.3 provides horizontal spacing of the reinforcing "
    "strips which is obtained by dividing the panel width, wp, by the "
    "governing number of strips, Ng' (p. E3-19), 2 strips a 5 ft panel at "
    "Levels 1 to 8 and 3 at Levels 9 to 12")


def _level_runs(values):
    """Consecutive levels with the same value: (first, last, value), the
    levels counted from 1 at the top."""
    runs = []
    for i, v in enumerate(values, start=1):
        if runs and runs[-1][2] == v:
            runs[-1][1] = i
        else:
            runs.append([i, i, v])
    return runs


def _levels(first, last):
    return f"level {first}" if first == last else f"levels {first}-{last}"


def _per_level_coverage_choice(r, internal, basis):
    """The coverage choice for a spacing or ratio given per level
    (GM-261)."""
    from geotech_common.choices import choice, how_given
    Rcs = [lv["Rc"] for lv in internal] if internal else None
    if isinstance(r.coverage_ratio, tuple):
        chosen = "Rc given per level, from the top: " + "; ".join(
            f"{v:.3g} at {_levels(a, z)}"
            for a, z, v in _level_runs(r.coverage_ratio))
        how = how_given(r.coverage_ratio, None, name="coverage_ratio")
    else:
        chosen = (f"Rc = b {r.width:g} m / Sh per level, from the top: "
                  + "; ".join(f"Sh {v:g} m at {_levels(a, z)} (Rc "
                              f"{r.width / v:.3g})"
                              for a, z, v in _level_runs(
                                  r.horizontal_spacing)))
        how = how_given(r.horizontal_spacing, None,
                        name="reinforcement_horizontal_spacing")
    if Rcs:
        lo, hi = (f"{r.Tallowable * v:.4g}" for v in (min(Rcs), max(Rcs)))
        chosen += (f"; tensile resistance Tal x Rc "
                   + (lo if lo == hi else f"{lo} to {hi}")
                   + f" kN/m of wall (Tal {r.Tallowable:.4g} kN/m of "
                   "reinforcement width)")
    return choice(
        "Coverage ratio Rc of the reinforcement (pullout and tensile "
        "resistance per metre of wall)", chosen, how,
        basis + PER_LEVEL_BASIS, [])


def _level_list_coverage_choice(rs, internal, basis):
    """The coverage choice for a reinforcement given per level (GM-266):
    each level's Rc from its own width and Sh, or its Rc given."""
    from geotech_common.choices import choice, how_given
    rs = list(rs)

    def rc_text(r):
        if r.coverage_ratio is not None:
            return f"Rc {r.coverage_ratio:.3g}, given"
        if r.horizontal_spacing is not None:
            return (f"Rc {r.width / r.horizontal_spacing:.3g} = b "
                    f"{r.width:g} m / Sh {r.horizontal_spacing:g} m")
        return "Rc 1, a continuous geosynthetic layer"
    chosen = "Rc per level, from the top: " + "; ".join(
        f"{text} at {_levels_of(spans)}"
        for text, spans in _grouped([rc_text(r) for r in rs]))
    alts = []
    if any(r.coverage_ratio is not None for r in rs):
        how = how_given(tuple(r.coverage_ratio for r in rs), None,
                        name="coverage_ratio")
    elif any(r.horizontal_spacing is not None for r in rs):
        how = how_given(tuple(r.horizontal_spacing for r in rs), None,
                        name="reinforcement_horizontal_spacing")
    else:
        how = how_given(None, None, name="coverage_ratio")
        alts.append(
            "geosynthetic strips at less than full coverage: Rc = b/Sh "
            "(reinforcement_horizontal_spacing with each level's width, or "
            "coverage_ratio); the pullout and tensile resistance per metre "
            "of wall then fall in proportion")
    if len({r.horizontal_spacing for r in rs}) > 1:
        basis += PER_LEVEL_BASIS
    if internal:
        tal_rc = [lv["Tal_Rc_kN_per_m"] for lv in internal]
        lo, hi = f"{min(tal_rc):.4g}", f"{max(tal_rc):.4g}"
        chosen += ("; tensile resistance Tal x Rc "
                   + (lo if lo == hi else f"{lo} to {hi}")
                   + " kN/m of wall, each level's Tal times its Rc")
    return choice(
        "Coverage ratio Rc of the reinforcement (pullout and tensile "
        "resistance per metre of wall)", chosen, how, basis, alts)


#: GEC-11 Vol I, read on the rendered pages (GM-262 review).
_FLEXIBLE_FACING = (
    "a grid connected to a flexible facing (a welded wire facing) or "
    "continuous with it takes the strips' resistance factor, 0.75 (GEC-11 "
    "Vol I p. 3-23; Table 4-7 note C, p. 4-48: 'For grid reinforcements "
    "connected to a flexible facing mat or which are continuous with the "
    "facing mat, use the resistance factor for strip reinforcements'), and "
    "so the strips' 0.55 Fy (p. 4-77): {tal}, every FOS_rupture{where} x "
    "{ratio:.3f} (lowest {low:.3g}); give it as a custom reinforcement "
    "({how})")

_ONE_REINFORCEMENT = (
    "a stronger reinforcement, or a closer spacing, at the levels where "
    "FOS_rupture is below {req:.1f}: a stronger one is "
    "reinforcement_per_level, each level its own reinforcement, as GEC-11 "
    "Vol II Example E4 takes 4W15 mats from Level 5 down (Table E4-7.4, p. "
    "E4-18), and a closer spacing is reinforcement_horizontal_spacing per "
    "level, as Example E3 takes 3 strips a panel below Level 8 (Table "
    "E3-7.3, p. E3-19)")

#: A built-in is a GEC-11 Vol II worked example's reinforcement (GM-271
#: review): the one specified for the wall is a custom reinforcement.
_SPECIFIED_PRODUCT = (
    "the reinforcement specified for the wall, where there is one: {name} "
    "is the worked example's, with its Tal as printed, not a product. Give "
    "the specified one as a custom reinforcement, reinforcement_type "
    "'{type}' with reinforcement_Tallowable{width}: {tal}")
_SPECIFIED_STEEL_TAL = (
    "Fy Ac / b with Ac after the corrosion loss over the wall's design life "
    "(GEC-11 Vol I Eq. 3-11a, p. 3-22), as the allowable 0.55 Fy for strips "
    "or 0.48 Fy for bar mats (p. 4-77)")
_SPECIFIED_GEOSYNTHETIC_TAL = (
    "Tult / RF, the product's Tult (its minimum average roll value from "
    "wide-width tests, ASTM D6637 for geogrids, D4595 for geotextiles) over "
    "RF = RFID x RFCR x RFD for the wall's backfill and design life (GEC-11 "
    "Vol I Eq. 3-12, p. 3-28)")

_ONE_GEOSYNTHETIC = (
    "a stronger grade at the levels where FOS_rupture is below {req:.1f}: "
    "reinforcement_per_level, each level its own reinforcement, as GEC-11 "
    "Vol II Example E1 takes its GG-I geogrid at Layers 1 to 4 and GG-II "
    "below (Table E1-7.4, p. E1-16); or a closer vertical spacing "
    "(reinforcement_spacing)")

#: GEC-11 Vol I, read on the rendered pages (GM-262 review). The factored
#: tensile resistance is phi Tal Rc (Eq. 4-33, p. 4-48) against Tmax
#: factored by gamma_P-EV 1.35 (p. 4-40); Tal is "the nominal long-term
#: reinforcement design strength" for steel and geosynthetics alike (p.
#: 4-47), Fy Ac / b (Eq. 3-11a) or Tult / RF (Eq. 3-12). This module
#: checks the unfactored Tmax, so it requires Tal Rc / Tmax >= 1.35 / phi:
#: 1.0 on a steel reinforcement's Tal given as GEC-11's allowable 0.55 Fy
#: or 0.48 Fy (p. 4-77), phi / 1.35 already in it, and 1.35 / 0.90 = 1.5
#: on a geosynthetic's Tult / RF (phi 0.90, Table 4-7), as on pullout (phi
#: 0.90, FOS_pullout 1.5).
GEOSYNTHETIC_RUPTURE_FOS = 1.5

GEOSYNTHETIC_RUPTURE_BASIS = (
    "GEC-11 Vol I: Tal is 'the nominal long-term reinforcement design "
    "strength' (p. 4-47), for a geosynthetic Tal = Tult / RF, RF = RFID x "
    "RFCR x RFD (Eq. 3-12, p. 3-28); the factored tensile resistance phi "
    "Tal Rc (Eq. 4-33, p. 4-48), phi 0.90 for geosynthetic reinforcement, "
    "static (Table 4-7, p. 4-48), against Tmax factored by gamma_P-EV "
    "1.35 (p. 4-40), as Vol II Example E1 takes its geogrids (Tr = 0.90 "
    "Tal, Table E1-7.3, p. E1-15). Against the unfactored Tmax that is "
    "Tal Rc / Tmax >= 1.35 / 0.90 = 1.5, the factor of safety this tool "
    "requires of a geosynthetic, as of pullout (phi 0.90, 1.5); a steel "
    "reinforcement's Tal is GEC-11's allowable, 0.55 Fy or 0.48 Fy (p. "
    "4-77), phi / 1.35 already in it, at 1.0")


def rupture_FOS_required(reinforcement: Reinforcement) -> float:
    """The factor of safety the tensile rupture check requires of Tal x Rc
    over the unfactored Tmax (GM-262 review): 1.35 / phi with GEC-11's
    phi (Table 4-7) and gamma_P-EV 1.35 (p. 4-40). 1.0 for steel, whose
    Tal is given as the 0.55 Fy or 0.48 Fy allowable (p. 4-77); 1.5 for a
    geosynthetic, whose Tal is Tult / RF (Eq. 3-12)
    (:data:`GEOSYNTHETIC_RUPTURE_BASIS`). A reinforcement given per level
    is one kind at every level (:func:`level_reinforcements`)."""
    if not isinstance(reinforcement, Reinforcement):
        reinforcement = list(reinforcement)[0]
    return 1.0 if reinforcement.is_metallic else GEOSYNTHETIC_RUPTURE_FOS


def tensile_strength_choice(reinforcement: Reinforcement,
                            internal: List[Dict[str, Any]]) -> list:
    """The reinforcement's long-term tensile strength Tal in the rupture
    check, as a result's ``choices`` (GM-262 review): a built-in's, with
    where it comes from (``Reinforcement.basis``), the reinforcement
    specified for the wall as the other way (GM-271 review), and for the
    built-in grid the flexible-facing reading; a geosynthetic's, with the factor of
    safety 1.5 the check requires of it (:func:`rupture_FOS_required`);
    for any reinforcement whose FOS_rupture is below the required value at
    some level, that the same Tal and width are taken at every level.
    Nothing for a given steel Tal that passes."""
    from geotech_common.choices import choice, how_given, MODULE
    if not isinstance(reinforcement, Reinforcement):
        return _level_list_tensile_choice(list(reinforcement), internal)
    r = reinforcement
    req = rupture_FOS_required(r)
    low = [lv for lv in internal
           if lv["FOS_rupture"] < lv.get("FOS_rupture_required", req)]
    if not (r.basis or low or not r.is_metallic):
        return []
    unit = ("mat" if r.is_grid else "strip" if r.is_metallic
            else "reinforcement") + " width"
    chosen = f"{r.name}: Tal {r.Tallowable:.4g} kN/m of {unit}"
    if r.basis and r.width is not None:
        chosen += f" ({r.Tallowable * r.width:.3g} kN a {r.width:g} m " \
                  f"{'mat' if r.is_grid else 'strip'})"
    chosen += ", the same at every level"
    if not r.is_metallic:
        chosen += (f"; FOS_rupture required {req:.1f} (1.35 / phi 0.90) on "
                   "this Tal, the long-term strength Tult / RF")
    alts = []
    if r.basis and r.is_grid:
        ratio = STRIP_ALLOWABLE_FY / BAR_MAT_ALLOWABLE_FY
        alts.append(_FLEXIBLE_FACING.format(
            tal=f"Tal {r.Tallowable * ratio:.4g} kN/m of mat width",
            where="", ratio=ratio, how="reinforcement_Tallowable",
            low=ratio * min(lv["FOS_rupture"] for lv in internal)))
    if low:
        z = [lv["depth_m"] for lv in low]
        chosen += (f"; FOS_rupture is below {req:.1f} at "
                   + (f"z = {z[0]:g} m" if len(z) == 1
                      else f"{len(z)} levels, z = {z[0]:g} to {z[-1]:g} m")
                   + f" (lowest {min(lv['FOS_rupture'] for lv in low):.3g})")
        alts.append((_ONE_REINFORCEMENT if r.is_metallic
                     else _ONE_GEOSYNTHETIC).format(req=req))
    if r.basis:
        alts.append(_SPECIFIED_PRODUCT.format(
            name=r.name, type=r.type,
            width=" and reinforcement_width" if r.is_metallic else "",
            tal=(_SPECIFIED_STEEL_TAL if r.is_metallic
                 else _SPECIFIED_GEOSYNTHETIC_TAL)))
    how = MODULE if r.basis else how_given(
        r.Tallowable, None, name="reinforcement_Tallowable")
    if r.is_metallic:
        basis = r.basis or _STEEL_TAL_GIVEN
    else:
        basis = ((r.basis + ". ") if r.basis else "") \
            + GEOSYNTHETIC_RUPTURE_BASIS
    return [choice(
        "Long-term tensile strength Tal of the reinforcement (tensile "
        "rupture check)", chosen, how, basis, alts)]


_STRONGER_AT_LEVELS = (
    "a stronger reinforcement, or a closer spacing, at the levels where "
    "FOS_rupture is below {req:.1f}: another entry of "
    "reinforcement_per_level at those levels, or a closer "
    "reinforcement_horizontal_spacing there"
    "{vertical}")

_SPECIFIED_PRODUCTS = (
    "the reinforcement specified for the wall, where there is one: "
    "{names} {verb} the worked example's, with {its} Tal as printed, not a "
    "product. Give the specified one as a custom entry of "
    "reinforcement_per_level, type {type} with Tallowable{width}: {tal}")

_STEEL_TAL_GIVEN = (
    "GEC-11 Vol I Eq. 3-11a (p. 3-22), Fy Ac / b, given as the "
    "allowable 0.55 Fy for strips or 0.48 Fy for bar mats (p. "
    "4-77); the rupture check compares Tal x Rc with the "
    "unfactored Tmax at a factor of safety of 1.0 (p. 4-47)")


def _level_list_tensile_choice(rs, internal) -> list:
    """The tensile strength choice for a reinforcement given per level
    (GM-266): each level's Tal, by runs of the same reinforcement, where
    each built-in's comes from, and the levels below the required
    FOS_rupture."""
    from geotech_common.choices import choice, how_given, MODULE
    req = rupture_FOS_required(rs[0])
    low = [lv for lv in internal
           if lv["FOS_rupture"] < lv.get("FOS_rupture_required", req)]
    metallic = rs[0].is_metallic
    built_in = [r for r in rs if r.basis]

    def unit(r):
        return ("mat" if r.is_grid else "strip" if r.is_metallic
                else "reinforcement") + " width"
    groups = _grouped(rs, key=lambda r: (r.name, r.Tallowable, r.width))
    chosen = "Tal per level, from the top: " + "; ".join(
        f"{r.name}: Tal {r.Tallowable:.4g} kN/m of {unit(r)} at "
        f"{_levels_of(spans)}" for r, spans in groups)
    if not metallic:
        chosen += (f"; FOS_rupture required {req:.1f} (1.35 / phi 0.90) on "
                   "each Tal, the long-term strength Tult / RF")
    alts = []
    grids = [i for i, r in enumerate(rs) if r.basis and r.is_grid]
    if grids:
        ratio = STRIP_ALLOWABLE_FY / BAR_MAT_ALLOWABLE_FY
        mats = _grouped([rs[i] for i in grids],
                        key=lambda r: (r.name, r.Tallowable))
        spans = [s for s in _level_runs([i in grids
                                         for i in range(len(rs))])
                 if s[2]]
        alts.append(_FLEXIBLE_FACING.format(
            tal="Tal " + ", ".join(
                f"{r.Tallowable * ratio:.4g} kN/m of mat width for "
                f"{r.name}" for r, _ in mats),
            where=("" if len(grids) == len(rs) else
                   " at " + _levels_of([(a, z) for a, z, _ in spans])),
            ratio=ratio, how="an entry of reinforcement_per_level with "
            "Tallowable",
            low=ratio * min(internal[i]["FOS_rupture"] for i in grids)))
    if low:
        z = [lv["depth_m"] for lv in low]
        chosen += (f"; FOS_rupture is below {req:.1f} at "
                   + (f"z = {z[0]:g} m" if len(z) == 1
                      else f"{len(z)} levels, z = {z[0]:g} to {z[-1]:g} m")
                   + f" (lowest {min(lv['FOS_rupture'] for lv in low):.3g})")
        alts.append(_STRONGER_AT_LEVELS.format(
            req=req, vertical="" if metallic else
            ", or a closer vertical spacing (reinforcement_spacing)"))
    if built_in:
        names = [r.name for r, _ in _grouped(built_in, key=lambda r: r.name)]
        alts.append(_SPECIFIED_PRODUCTS.format(
            names=", ".join(names), verb="is" if len(names) == 1 else "are",
            its="its" if len(names) == 1 else "their",
            type=(f"'{rs[0].type}'" if len({r.type for r in rs}) == 1
                  else "its type"),
            width=" and width" if metallic else "",
            tal=(_SPECIFIED_STEEL_TAL if metallic
                 else _SPECIFIED_GEOSYNTHETIC_TAL)))
    how = MODULE if len(built_in) == len(rs) else how_given(
        tuple(r.Tallowable for r in rs), None, name="reinforcement_per_level")
    bases = [r.basis for r, _ in _grouped(built_in, key=lambda r: r.basis)]
    if len(built_in) < len(rs) and metallic:
        bases.append(_STEEL_TAL_GIVEN)
    if not metallic:
        bases.append(GEOSYNTHETIC_RUPTURE_BASIS)
    return [choice(
        "Long-term tensile strength Tal of the reinforcement (tensile "
        "rupture check), per level", chosen, how, ". ".join(bases), alts)]


#: GEC-11 Vol I, read on the rendered pages (GM-256).
F_STAR_TESTED_BASIS = (
    "GEC-11 Vol I Table 3-6 note (i) (p. 3-17): the empirical F* values "
    "are for use 'in the absence of product and backfill specific test "
    "data'; p. 3-20: 'Some additional economy can be obtained from "
    "site/product specific testing', the procedures in Appendix B. The "
    "tested F* replaces Eqs. 3-4 to 3-8 at every level")
ALPHA_BASIS = (
    "GEC-11 Vol I p. 3-16: the scale effect correction factor alpha 'can "
    "be obtained from pullout tests on reinforcements with different "
    "lengths as presented in Appendix B'; Table 3-6 (p. 3-17) prints the "
    "default values, and its note (ii): 'Pullout testing to determine "
    "alpha is recommended if alpha shown in table is less than 1.0'")

#: GEC-11 Vol I Table 3-6 (p. 3-17), "alpha Default Value", by the
#: reinforcement's type, with the table's row. A "geosynthetic" of no
#: stated kind has no row: its alpha is given (GM-256).
ALPHA_TABLE_3_6 = {
    "metallic_strip": (1.0, "inextensible strips"),
    "metallic_grid": (1.0, "inextensible grids (bar mats and welded wire)"),
    "geogrid": (0.8, "extensible grids (geogrids)"),
    "geotextile": (0.6, "extensible sheets (geotextiles)"),
}
ALPHA_TABLE_BASIS = (
    "GEC-11 Vol I Table 3-6 (p. 3-17), alpha default value: 1.0 for "
    "inextensible strips and grids, 0.8 for extensible grids, 0.6 for "
    "extensible sheets; p. 3-16: 'In the absence of test data, alpha = 0.8 "
    "for geogrids and alpha = 0.6 for geotextiles (extensible sheets) is "
    "recommended'; note (ii): 'Pullout testing to determine alpha is "
    "recommended if alpha shown in table is less than 1.0. These values of "
    "alpha represent highly extensible geosynthetics.'")
_ALPHA_GRID_NOTE = (
    "; Table 3-6 note *, on the steel grids' 1.0: 'For longitudinal "
    "bars/wires spacing greater than 6 inches, alpha may be less than 1.0 "
    "and pullout tests are required'")
NEEDS_ALPHA = (
    "A geosynthetic of no stated kind needs its scale effect correction "
    "factor alpha, in the pullout resistance F* alpha sigma_v Le C Rc "
    "(GEC-11 Vol I Eq. 3-2, p. 3-16): 'In the absence of test data, alpha = "
    "0.8 for geogrids and alpha = 0.6 for geotextiles (extensible sheets) "
    "is recommended' (p. 3-16; Table 3-6, p. 3-17). Give reinforcement_type "
    "'geogrid' (alpha 0.8) or 'geotextile' (alpha 0.6), or alpha_tested "
    "from pullout tests (Appendix B)")


def table_alpha(reinforcement: Reinforcement) -> Optional[float]:
    """GEC-11 Vol I Table 3-6's alpha for the reinforcement's type (p.
    3-17): 1.0 for steel strips and grids, 0.8 for a geogrid, 0.6 for a
    geotextile; None for a geosynthetic of no stated kind."""
    row = ALPHA_TABLE_3_6.get(reinforcement.type)
    return None if row is None else row[0]


def pullout_alpha(reinforcement: Reinforcement,
                  alpha_tested: float = None) -> float:
    """The scale effect correction factor alpha the pullout check takes:
    the tested one when given, else Table 3-6's for the type. A
    geosynthetic of no stated kind given no tested alpha is refused, as
    Table 3-6 prints 0.8 for a geogrid and 0.6 for a geotextile (GM-256;
    every geosynthetic took 1.0 before)."""
    if alpha_tested is not None:
        return alpha_tested
    alpha = table_alpha(reinforcement)
    if alpha is None:
        raise ValueError(NEEDS_ALPHA)
    return alpha


def _level_alphas(reinforcement, rs, alpha_tested) -> List[float]:
    """Each level's alpha (:func:`pullout_alpha`); with a reinforcement
    per level a refusal names the level (GM-266)."""
    if isinstance(reinforcement, Reinforcement):
        return [pullout_alpha(reinforcement, alpha_tested)] * len(rs)
    out = []
    for i, r in enumerate(rs, start=1):
        try:
            out.append(pullout_alpha(r, alpha_tested))
        except ValueError as e:
            raise ValueError(
                f"level {i} from the top ({r.name}): {e}") from None
    return out


def _check_tested(F_star_tested, alpha_tested) -> None:
    """Refuse a tested F* or alpha that is not one (GM-256)."""
    if alpha_tested is not None and not 0 < alpha_tested <= 1:
        raise ValueError(
            "alpha_tested is the scale effect correction factor from pullout "
            "tests, more than 0 and at most 1 (GEC-11 Vol I p. 3-16: "
            "'generally 1.0 for metallic reinforcements and 0.6 to 1.0 for "
            f"geosynthetic reinforcements'); got {alpha_tested}")
    if F_star_tested is None:
        return
    if isinstance(F_star_tested, (int, float)):
        if not F_star_tested > 0:
            raise ValueError(f"F_star_tested must be positive, got "
                             f"{F_star_tested}")
        return
    try:
        pts = [(float(z), float(f)) for z, f in F_star_tested]
    except (TypeError, ValueError):
        raise ValueError(
            "F_star_tested is the pullout resistance factor from pullout "
            "tests: one number for every level, or [[depth_m, F*], ...] "
            "with depth below the top of the wall, taken linearly between "
            "and at the end values beyond") from None
    if not pts or any(z < 0 or not f > 0 for z, f in pts) or any(
            b[0] <= a[0] for a, b in zip(pts, pts[1:])):
        raise ValueError(
            "F_star_tested as [[depth_m, F*], ...] needs depths of 0 or "
            "more in increasing order and each F* more than 0; got "
            f"{F_star_tested}")


def F_star_from_tests(F_star_tested, z: float) -> float:
    """The tested F* at depth ``z`` (m below the top of the wall): the one
    value, or linear between the [depth, F*] points and the end value
    beyond them."""
    if isinstance(F_star_tested, (int, float)):
        return float(F_star_tested)
    pts = [(float(a), float(f)) for a, f in F_star_tested]
    if z <= pts[0][0]:
        return pts[0][1]
    for (z0, f0), (z1, f1) in zip(pts, pts[1:]):
        if z <= z1:
            return f0 + (f1 - f0) * (z - z0) / (z1 - z0)
    return pts[-1][1]


def _tested_text(F_star_tested) -> str:
    if isinstance(F_star_tested, (int, float)):
        return f"F* {F_star_tested:g} at every level"
    return "F* " + ", ".join(f"{f:g} at {z:g} m" for z, f in F_star_tested) \
        + " (linear between)"


def Kr_Ka_ratio(z: float, reinforcement_type: str = "metallic") -> float:
    """Ratio Kr/Ka as a function of depth per GEC-11 / AASHTO.

    Two inextensible (metallic) lateral-stress-ratio curves, plus geosynthetic:

    - ribbed metallic STRIP ("metallic" / "metallic_strip", DEFAULT):
      1.7 at z=0 -> 1.2 at z>=20 ft (6.096 m), constant below (GEC-11 Fig.
      4-10, p. 4-37; the text, p. 4-35, says "20 ft (6 m)").
    - steel bar-mat / welded-grid ("bar_mat" / "welded_grid" /
      "metallic_grid"): 2.5 at z=0 -> 1.2 at z>=20 ft (6.096 m), constant below
      (GEC-11 Fig. E4-5 / AASHTO Fig. 11.10.6.2.1-3).
    - geosynthetic / extensible ("geosynthetic", "geogrid", "geotextile"):
      constant 1.0.

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    reinforcement_type : str, optional
        "metallic"/"metallic_strip" (ribbed strip, DEFAULT),
        "bar_mat"/"welded_grid"/"metallic_grid" (steel grid),
        or "geosynthetic"/"geogrid"/"geotextile".

    Returns
    -------
    float
        Kr/Ka ratio.

    References
    ----------
    FHWA GEC-11, Figures 4-10 and E4-5; AASHTO LRFD Fig. 11.10.6.2.1-3
    """
    if reinforcement_type in ("geosynthetic", "geogrid", "geotextile"):
        return 1.0

    if reinforcement_type in ("bar_mat", "welded_grid", "metallic_grid"):
        # Steel bar mat / welded grid: 2.5 at z=0 -> 1.2 at z=20 ft (6.096 m)
        if z <= 0:
            return 2.5
        elif z >= _Z20_FT_M:
            return 1.2
        else:
            return 2.5 - (2.5 - 1.2) / _Z20_FT_M * z

    # Ribbed metallic strip (default): linear 1.7 at z=0 to 1.2 at z=20 ft
    # (6.096 m), as GEC-11 Fig. 4-10 draws it for strips and bar mats alike
    if z <= 0:
        return 1.7
    elif z >= _Z20_FT_M:
        return 1.2
    else:
        return 1.7 - (0.5 / _Z20_FT_M) * z


def F_star_strip_top(Cu: float = None) -> float:
    """F* of ribbed steel strips at the top of the wall, GEC-11 Vol I Eq.
    3-4 (p. 3-20): 1.2 + log10(Cu), 2.0 at most. ``Cu`` None takes the
    Cu = 4 that page gives for a backfill whose Cu is not known (F* 1.8).
    """
    if Cu is None:
        Cu = CU_UNKNOWN
    if not Cu >= 1.0:
        raise ValueError(
            f"Cu_backfill is the backfill's uniformity coefficient D60/D10, "
            f"1 or more; got {Cu:g}")
    return min(1.2 + math.log10(Cu), F_STAR_STRIP_MAX)


def F_star_metallic(z: float, phi_backfill: float = 34.0,
                    reinforcement_type: str = "metallic",
                    t_over_St: float = None, Cu: float = None) -> float:
    """Pullout friction-bearing factor F* for inextensible reinforcement.

    Two curves:

    - ribbed metallic STRIP ("metallic" / "metallic_strip", DEFAULT):
      F* = 1.2 + log10(Cu), 2.0 at most, at z=0 -> tan(phi) at z>=20 ft
      (6.096 m), linear between (GEC-11 Vol I Eqs. 3-4/3-5, p. 3-20; Vol II
      Examples E3 and E5 interpolate linearly). ``Cu`` left out takes 4,
      F* 1.8 at the top, as p. 3-20 says to when the backfill's Cu is
      unknown (GM-246; 2.0 whatever the backfill before).
    - steel bar-mat / welded-grid ("bar_mat" / "welded_grid" /
      "metallic_grid"): F* = 20·(t/St) at z=0 -> 10·(t/St) at z>=20 ft
      (6.096 m), where t = transverse-bar diameter and St = transverse-bar
      (grid) spacing. ``t_over_St`` is required for the grid curve.

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    phi_backfill : float, optional
        Backfill friction angle (degrees). Default 34. (Strip curve only.)
    reinforcement_type : str, optional
        "metallic"/"metallic_strip" (ribbed strip, DEFAULT) or
        "bar_mat"/"welded_grid"/"metallic_grid" (steel grid).
    t_over_St : float, optional
        Transverse-bar diameter / spacing ratio (unit-free) for the steel-grid
        curve. Required when ``reinforcement_type`` is a grid type.
    Cu : float, optional
        Uniformity coefficient D60/D10 of the backfill (strip curve only).
        Default None: 4 (:data:`CU_UNKNOWN`).

    Returns
    -------
    float
        Pullout resistance factor F*.

    References
    ----------
    FHWA GEC-11 Vol I Eqs. 3-4 and 3-5 (ribbed steel strips) and Figure
    E4-5 (bar mats)
    """
    if reinforcement_type in ("bar_mat", "welded_grid", "metallic_grid"):
        if t_over_St is None:
            raise ValueError(
                "F_star_metallic: a steel-grid (bar_mat) F* curve requires "
                "t_over_St = transverse-bar diameter / spacing."
            )
        F_surface = 20.0 * t_over_St
        F_deep = 10.0 * t_over_St
        if z <= 0:
            return F_surface
        elif z >= _Z20_FT_M:
            return F_deep
        else:
            return F_deep + (_Z20_FT_M - z) * (F_surface - F_deep) / _Z20_FT_M

    F_star_surface = F_star_strip_top(Cu)
    # At depth >= 20 ft (6.096 m), F* = tan(phi) for ribbed metallic strips
    # (GEC-11 Eq. 3-5)
    F_star_deep = math.tan(math.radians(phi_backfill))

    if z <= 0:
        return F_star_surface
    elif z >= _Z20_FT_M:
        return F_star_deep
    else:
        return F_star_surface - (F_star_surface - F_star_deep) / _Z20_FT_M * z


def Tmax_at_level(z: float, gamma_backfill: float, Ka: float,
                  Kr_Ka: float, Sv: float,
                  q_surcharge: float = 0.0) -> float:
    """Maximum tensile force in reinforcement at depth z.

    sigma_h = Kr * sigma_v = (Kr/Ka) * Ka * (gamma*z + q)
    Tmax = sigma_h * Sv

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    Ka : float
        Active earth pressure coefficient of backfill.
    Kr_Ka : float
        Kr/Ka ratio at this depth.
    Sv : float
        Vertical spacing of reinforcement (m).
    q_surcharge : float, optional
        Surcharge (kPa). Default 0.

    Returns
    -------
    float
        Maximum tensile force Tmax (kN/m width).
    """
    sigma_v = gamma_backfill * z + q_surcharge
    sigma_h = Kr_Ka * Ka * sigma_v
    return sigma_h * Sv


def pullout_resistance(z: float, gamma_backfill: float,
                       Le: float, F_star: float,
                       alpha_pullout: float = 1.0,
                       C: float = 2.0,
                       q_surcharge: float = 0.0,
                       Rc: float = 1.0) -> float:
    """Pullout resistance of reinforcement at depth z.

    Pr = F* * alpha * sigma_v' * Le * C * Rc

    Parameters
    ----------
    z : float
        Depth below top of wall (m).
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    Le : float
        Effective length beyond failure surface (m).
    F_star : float
        Pullout resistance factor.
    alpha_pullout : float, optional
        Scale effect correction factor. Default 1.0.
    C : float, optional
        Overall reinforcement surface area geometry factor.
        C=2 for strips/grids (top and bottom surface). Default 2.
    q_surcharge : float, optional
        Distributed dead-load surcharge (kPa). Default 0. Traffic loads are
        left out of pullout (GEC-11 p. 4-49).
    Rc : float, optional
        Coverage ratio Rc = b/Sh (GEC-11 Vol I Eq. 3-1); 1 for a continuous
        layer. Default 1.0: Pr per metre of reinforcement width (Eq. 3-2).
        With the wall's Rc, Pr is per metre of wall (Eq. 4-35), which is
        what :func:`check_internal_stability` takes (:func:`coverage_ratio`).

    Returns
    -------
    float
        Pullout resistance Pr (kN/m width).
    """
    sigma_v = gamma_backfill * z + q_surcharge
    return F_star * alpha_pullout * sigma_v * Le * C * Rc


# ---------------------------------------------------------------------------
# Sloping backfill (GEC-11 Vol I, FHWA-NHI-10-024, §4.4.4, §4.4.6, §4.4.7)
# ---------------------------------------------------------------------------

def _rise_integral(x: float, tan_b: float, S: Optional[float]) -> float:
    """Area under the slope surface above the top of the wall from the face
    to x (m2/m): the rise is x tan(beta), not more than S."""
    if S is None or x * tan_b <= S:
        return 0.5 * tan_b * x * x
    x_c = S / tan_b
    return 0.5 * S * x_c + S * (x - x_c)


def _rise_first_moment(x: float, tan_b: float, S: Optional[float]) -> float:
    """First moment about the face of the area under the slope surface from
    the face to x (m3/m)."""
    if S is None or x * tan_b <= S:
        return tan_b * x ** 3 / 3.0
    x_c = S / tan_b
    return S * x_c ** 2 / 3.0 + 0.5 * S * (x * x - x_c * x_c)


def backslope_geometry(geom: MSEWallGeometry) -> Dict[str, Any]:
    """Geometry of the backfill slope above an MSE wall, per GEC-11 Vol I.

    The slope rises at ``geom.backfill_slope`` (beta) from the top of the
    face. It either continues (GEC-11 Fig. 4-3) or is a broken backslope
    whose crest is ``geom.broken_backslope_height`` (S) above the top of the
    wall with level ground behind it (Fig. 4-4). With a level backfill every
    quantity reduces to the level case (h = H, V2 = 0, level_length = L,
    S_eq = 0, H1 = H).

    Returns
    -------
    dict
        beta_deg : the slope beta (deg).
        I_deg : the angle used for Kab and as the interface angle delta.
            For a slope that continues delta = beta (p. 4-15); for a broken
            backslope beta = delta = I, the angle from the top of the wall
            to the ground surface 2H behind the face (p. 4-15, Fig. 4-4).
        h : height of the pressure plane at the back of the reinforced zone,
            H + L tan(beta) (Eq. 4-7), and not more than H + S for a broken
            backslope (p. 4-20).
        S : the broken backslope height, or None.
        crest_x : distance of the slope crest from the face (m), or None.
        V2_area, V2_arm : area (m2/m) of the slope fill over the reinforced
            zone and the distance of its centroid from the face (the
            triangle of Figs. 4-3/4-4, arm 2L/3; a crest inside L adds the
            level part behind it).
        level_length : length of level ground over the reinforced zone (L
            for a level backfill; L - crest_x behind a crest inside L).
        S_eq : the equivalent uniform height of the sloping surcharge for
            internal stability, (1/2)(0.7H) tan(beta), not more than S
            (Fig. 4-11, p. 4-36).
        H1 : height of the zone of maximum stress for inextensible
            reinforcement, H + tan(beta) 0.3H / (1 - 0.3 tan(beta)) (Fig.
            4-9a); the rise not more than S.
    """
    H = geom.wall_height
    L = geom.reinforcement_length
    beta = geom.backfill_slope
    S = geom.broken_backslope_height
    if beta <= 0:
        return {"beta_deg": 0.0, "I_deg": 0.0, "h": H, "S": None,
                "crest_x": None, "V2_area": 0.0, "V2_arm": 2.0 * L / 3.0,
                "level_length": L, "S_eq": 0.0, "H1": H}
    t = math.tan(math.radians(beta))

    def rise(x):
        return x * t if S is None else min(x * t, S)

    I_deg = (beta if S is None
             else math.degrees(math.atan(rise(2.0 * H) / (2.0 * H))))
    area = _rise_integral(L, t, S)
    crest_x = None if S is None else S / t
    level_length = 0.0 if crest_x is None else max(L - crest_x, 0.0)
    if level_length < 1e-9 * L:          # a crest at the back of the zone
        level_length = 0.0
    S_eq = 0.5 * 0.7 * H * t
    dH = t * 0.3 * H / (1.0 - 0.3 * t)
    if S is not None:
        S_eq = min(S_eq, S)
        dH = min(dH, S)
    return {
        "beta_deg": beta, "I_deg": I_deg, "h": H + rise(L), "S": S,
        "crest_x": crest_x, "V2_area": area,
        "V2_arm": _rise_first_moment(L, t, S) / area,
        "level_length": level_length, "S_eq": S_eq, "H1": H + dH,
    }


def _mean_rise(a: float, b: float, geom: MSEWallGeometry) -> float:
    """Mean rise of the slope surface above the top of the wall between a
    and b from the face (m)."""
    if geom.backfill_slope <= 0 or b <= a:
        return 0.0
    t = math.tan(math.radians(geom.backfill_slope))
    S = geom.broken_backslope_height
    return (_rise_integral(b, t, S) - _rise_integral(a, t, S)) / (b - a)


def _check_backslope(geom: MSEWallGeometry, phi_retained: Optional[float],
                     q: float) -> None:
    """Refuse, with the reason, what the sloping-backfill check does not
    cover: a slope steeper than the retained fill's friction angle, and a
    surcharge on a slope that continues."""
    beta = geom.backfill_slope
    if phi_retained is not None and beta > phi_retained:
        raise ValueError(
            f"backfill_slope {beta:g} deg is steeper than the friction angle "
            f"of the fill behind the reinforced zone, {phi_retained:g} deg: "
            "sin(phi - beta) < 0 under the root in GEC-11 Eq. 4-3, so the "
            "Coulomb Kab has no real value, and a fill slope steeper than "
            "its friction angle does not stand.")
    if q > 0 and geom.broken_backslope_height is None:
        raise ValueError(
            "a surcharge on a backfill slope that continues is not covered: "
            "GEC-11 gives the sloping-backfill case with no surcharge "
            "(Section 4.4.6.b, Eq. 4-16). For traffic on level ground behind "
            "the crest of a broken backslope, give broken_backslope_height, "
            "the height of the crest above the top of the wall (GEC-11 "
            "Example E1).")


def backslope_loads(geom: MSEWallGeometry, gamma_backfill: float,
                    phi_retained: float, gamma_retained: float,
                    q: float = 0.0) -> Dict[str, Any]:
    """Unfactored external loads on an MSE wall with a sloping backfill.

    Per GEC-11 Vol I (FHWA-NHI-10-024) §4.4.4 and Figs. 4-3, 4-4 and 4-8,
    as worked in Vol II Examples E3 (a slope that continues) and E1 (a broken
    backslope with traffic behind the crest). Per unit length of wall, with
    the moment arms about the toe of the reinforced zone (Point A).

    * Kab: Coulomb, Eqs. 4-2/4-3, vertical wall (theta = 90 deg), with the
      slope and the interface angle both I (I = beta for a slope that
      continues).
    * F_T = 1/2 Kab gamma_b h^2 (Eq. 4-7), on the vertical plane at the back
      of the reinforced zone, inclined at I: F_TH = F_T cos I at h/3 (Eq.
      4-10), F_TV = F_T sin I at L.
    * V1 = gamma_r H L at L/2; V2 = gamma_r times the slope fill over the
      reinforced zone, at its centroid (2L/3 for the triangle,
      V2 = gamma_r L (h - H) / 2).
    * q (broken backslope only): F2 = Kab q h at h/2, inclined at I
      (Example E1); Vs = q over any level ground on the reinforced zone.

    Raises ValueError for a slope steeper than ``phi_retained`` (at beta =
    phi Eq. 4-3 gives Gamma = 1) and for a surcharge on a slope that
    continues.
    """
    _check_backslope(geom, phi_retained, q)
    H = geom.wall_height
    L = geom.reinforcement_length
    bs = backslope_geometry(geom)
    I_deg, h = bs["I_deg"], bs["h"]
    Kab = coulomb_Ka(phi_retained, I_deg, 90.0, I_deg)
    c, s = math.cos(math.radians(I_deg)), math.sin(math.radians(I_deg))
    FT = 0.5 * Kab * gamma_retained * h ** 2
    F2 = Kab * q * h
    V1 = gamma_backfill * H * L
    V2 = gamma_backfill * bs["V2_area"]
    Vs = q * bs["level_length"]
    return {
        "geometry": bs, "Kab": Kab, "I_deg": I_deg, "h": h,
        "V1": V1, "MV1": V1 * L / 2.0,
        "V2": V2, "MV2": V2 * bs["V2_arm"],
        "Vs": Vs, "MVs": Vs * (L - bs["level_length"] / 2.0),
        "FT": FT, "FTH": FT * c, "FTV": FT * s,
        "MFTH": FT * c * h / 3.0, "MFTV": FT * s * L,
        "F2": F2, "F2H": F2 * c, "F2V": F2 * s,
        "MF2H": F2 * c * h / 2.0, "MF2V": F2 * s * L,
    }


_TRAFFIC_VERTICAL_BASIS = {
    False: ("The traffic thrust's vertical component F2 sin(I) is not "
            "counted in the sliding or eccentricity resistance: GEC-11 Vol I "
            "p. 4-22 includes external loads that increase sliding "
            "resistance 'only if those loads are permanent'. Bearing counts "
            "it (p. 4-26: the bearing check applies the live load above "
            "both the reinforced zone and the retained backfill). GEC-11 "
            "Example E1 (Vol II pp. E1-7 to E1-9) counts it in all three: "
            "traffic_vertical_resists=true."),
    True: ("The traffic thrust's vertical component F2 sin(I) is counted in "
           "the sliding and eccentricity resistance (LL factor), as GEC-11 "
           "Example E1 does (Vol II pp. E1-7 to E1-9). The general text "
           "counts loads that increase sliding resistance 'only if those "
           "loads are permanent' (Vol I p. 4-22): "
           "traffic_vertical_resists=false."),
}


#: Whether the live-load surcharge over the reinforced zone resists sliding
#: and overturning on the ASD path (GM-164), by ``live_load_resists``: the
#: basis of each reading.
LIVE_LOAD_BASIS = {
    False: ("GEC-11 Vol I (FHWA-NHI-10-024) p. 4-19: 'The live load "
            "surcharge is not considered as a stabilizing force when "
            "checking sliding, i.e., the sliding stability check only "
            "applies the live load above the retained backfill, as shown in "
            "Figure 4-2'; p. 4-21 (Eq. 4-12): 'the live load is excluded "
            "since it increases sliding stability'; p. 4-22: 'External loads "
            "that increase sliding resistance should only be included if "
            "those loads are permanent' and 'The eccentricity limit check "
            "only applies the live load above the retained backfill' "
            "(limiting eccentricity is 'formerly known as overturning', "
            "p. 4-18). The ASD procedure is not updated; its users 'may want "
            "to refer to current LRFD procedures for any updates' (Vol II "
            "App. F.1, p. F-2)"),
    True: ("counted: the free body with the live load over the reinforced "
           "zone in the resisting weight and moment, W = gamma H L + q L "
           "(the module's ASD reading before GM-164); GEC-11 Fig. 4-2 "
           "applies it there for bearing and overall stability only"),
}

#: Whether the live-load surcharge over the reinforced zone loads the base
#: in the bearing pressure on the ASD path (GM-164), by
#: ``surcharge_in_bearing``.
BEARING_BASIS = {
    True: ("GEC-11 Vol I p. 4-26: 'The bearing check applies live load above "
           "both the reinforced zone and the retained backfill, as shown in "
           "Figure 4-2'; Eqs. 4-19 and 4-20 (p. 4-27) put q L in the "
           "vertical load and in eB, the 'eccentricity for bearing "
           "calculation (not equal to eccentricity check e)' (p. 4-26)"),
    False: ("left out of the bearing pressure: the live load taken as not "
            "reaching over the reinforced zone, as in the sliding and "
            "overturning checks; GEC-11 p. 4-26 applies it for bearing"),
}


def _backslope_summary(ld: Dict[str, Any],
                       traffic_vertical_resists: bool = False
                       ) -> Dict[str, Any]:
    """The sloping-backfill basis a result reports (SI, rounded)."""
    bs = ld["geometry"]
    out = {
        "backfill_slope_deg": bs["beta_deg"],
        "broken_backslope_height_m": bs["S"],
        "crest_from_face_m": (None if bs["crest_x"] is None
                              else round(bs["crest_x"], 3)),
        "interface_angle_deg": round(ld["I_deg"], 3),
        "Kab": round(ld["Kab"], 4),
        "h_m": round(ld["h"], 3),
        "FT_kN_per_m": round(ld["FT"], 3),
        "FT_horizontal_kN_per_m": round(ld["FTH"], 3),
        "FT_vertical_kN_per_m": round(ld["FTV"], 3),
        "V2_kN_per_m": round(ld["V2"], 3),
        "V2_arm_from_face_m": round(bs["V2_arm"], 3),
        "basis": (
            "GEC-11 (FHWA-NHI-10-024) Section 4.4.4: Coulomb Kab (Eqs. 4-2, "
            "4-3) with delta = beta (= I, Fig. 4-4, for a broken backslope); "
            "F_T = 1/2 Kab gamma_b h^2 (Eq. 4-7) on h = H + L tan(beta) "
            "(not more than H + S), inclined at delta; the slope fill over "
            "the reinforced zone, V2, added to the weight (Figs. 4-3, 4-8)"),
    }
    if ld["F2"] > 0:
        out["F2_kN_per_m"] = round(ld["F2"], 3)
        out["Vs_kN_per_m"] = round(ld["Vs"], 3)
        out["traffic_vertical_resists"] = bool(traffic_vertical_resists)
        out["traffic_basis"] = _TRAFFIC_VERTICAL_BASIS[
            bool(traffic_vertical_resists)]
    return out


def uses_strip_F_star(reinforcement: Reinforcement) -> bool:
    """Whether the pullout check takes the ribbed steel strip F* curve,
    which the backfill's Cu sets: a metallic strip, or a steel grid given
    no transverse-bar geometry (t/St), which falls back to it."""
    return reinforcement.is_metallic and not (
        reinforcement.is_grid and reinforcement.t_over_St is not None)


def pullout_factor_choices(geom: MSEWallGeometry, gamma_backfill: float,
                           phi_backfill: float, reinforcement: Reinforcement,
                           Cu_backfill: float = None,
                           internal: Optional[List[Dict[str, Any]]] = None,
                           F_star_tested=None, alpha_tested: float = None,
                           ) -> list:
    """The ribbed steel strip F* at the top of the wall (GM-246), as a
    result's ``choices`` (``geotech_common.choices``): the Cu it took,
    given or the Cu = 4 GEC-11 Vol I p. 3-20 takes when the backfill's is
    unknown, with FOS_pullout at the top level and the lowest at the other
    reading (Cu = 4, or a Cu of 6.3 or more, where F* reaches its 2.0
    maximum). For a reinforcement whose F* takes no Cu: nothing when Cu
    is left out, and a given Cu checked against GEC-11 Table 3-6 note
    (i), which accepts every empirical F* only for Cu >= 4. A tested F*
    (``F_star_tested``) replaces all of that with one entry, and a tested
    alpha adds one (GM-256), each with the FOS_pullout at the printed
    empirical value; a Cu given beside a tested F* is in no check, and the
    entry says so and what it bears on: the empirical F* beside it (a
    strip's) and note (i)'s check on that F*. A reinforcement given per
    level (GM-266) gives alpha and the empirical F* by the levels each
    type is at, and the strip F*'s Cu where a level takes it."""
    from geotech_common.choices import choice, how_given
    args = (geom, gamma_backfill, phi_backfill, reinforcement)
    rs = level_reinforcements(reinforcement, geom.reinforcement_depths)

    def by_levels(key, text):
        """``text(r)`` for the one kind, or each kind with its levels."""
        kinds = _grouped(rs, key=key)
        if len(kinds) == 1:
            return text(kinds[0][0])
        return "; ".join(f"{text(r)} at {_levels_of(spans)}"
                         for r, spans in kinds)
    if internal is None:
        internal = check_internal_stability(
            *args, Cu_backfill=Cu_backfill, F_star_tested=F_star_tested,
            alpha_tested=alpha_tested)

    def lowest(levels):
        return min(lv["FOS_pullout"] for lv in levels)

    def against(other):
        if not (internal and other):
            return ""
        return (f"; FOS_pullout {other[0]['FOS_pullout']:.3g} at the top "
                f"level, {other[0]['depth_m']:g} m (here "
                f"{internal[0]['FOS_pullout']:.3g}), lowest "
                f"{lowest(other):.3g} (here {lowest(internal):.3g})")

    out = []
    alpha_what = ("Scale effect correction factor alpha of the pullout "
                  "resistance (alpha_tested)")
    def row_of(r):
        return ALPHA_TABLE_3_6.get(r.type, (None, ""))[1]
    alpha0s = [table_alpha(r) for r in rs]
    one_type = len({r.type for r in rs}) == 1
    any_grid = any(r.is_grid for r in rs)
    below_1 = any(a is not None and a < 1.0 for a in alpha0s)
    alpha0 = alpha0s[0]
    note = _ALPHA_GRID_NOTE if any_grid else ""
    if alpha_tested is not None:
        if one_type:
            printed = ([(alpha0, row_of(rs[0]))] if alpha0 is not None
                       else [(0.8, "geogrids"), (0.6, "geotextiles")])
            alts = [f"alpha {a:g}, Table 3-6's default value for {r}"
                    + against(check_internal_stability(
                        *args, Cu_backfill=Cu_backfill,
                        F_star_tested=F_star_tested, alpha_tested=a))
                    for a, r in printed]
        else:
            alts = ["Table 3-6's default values by type: " + by_levels(
                lambda r: r.type, lambda r: (
                    f"alpha {table_alpha(r):g} for {row_of(r)}"
                    if table_alpha(r) is not None else
                    "none printed for a geosynthetic of no stated kind"))
                + (against(check_internal_stability(
                    *args, Cu_backfill=Cu_backfill,
                    F_star_tested=F_star_tested))
                   if None not in alpha0s else "")]
        out.append(choice(
            alpha_what, f"alpha {alpha_tested:g}, from pullout tests"
            + ("" if isinstance(reinforcement, Reinforcement)
               else ", at every level"),
            how_given(alpha_tested, None, name="alpha_tested"),
            ALPHA_BASIS + note, alts))
    elif below_1 or any_grid:
        # the table's value carries a condition: note (ii) below 1.0, the
        # note * on a steel grid's 1.0
        took = (f"alpha {alpha0:g}, GEC-11 Vol I Table 3-6's default value "
                f"for {row_of(rs[0])}, p. 3-17" if one_type else
                "GEC-11 Vol I Table 3-6's default values (p. 3-17) by type: "
                + by_levels(lambda r: r.type, lambda r: (
                    f"alpha {table_alpha(r):g} for {row_of(r)}")))
        why = []
        if below_1:
            why.append("which note (ii) recommends")
        if any_grid:
            why.append("which note * requires for longitudinal bars more "
                       "than 6 in. (150 mm) apart")
        alt = "alpha from pullout tests (alpha_tested), " + (
            why[0] if len(why) == 1 else
            why[0] + " below 1.0, and " + why[1] + " in a steel grid")
        if below_1:
            alt += ("; at alpha 1.0, the most a test can give"
                    + against(check_internal_stability(
                        *args, Cu_backfill=Cu_backfill,
                        F_star_tested=F_star_tested, alpha_tested=1.0)))
        # GM-266 review: no ``took``. alpha_tested is an alpha FROM TESTS,
        # so the table's value given back under that name would claim a
        # test; defaults_used gives the declared case (Table 3-6's alpha by
        # type, with its basis), each level its alpha, and this entry both
        out.append(choice(
            alpha_what, took,
            how_given(None, None, name="alpha_tested"),
            ALPHA_TABLE_BASIS + note, [alt]))
    any_strip = any(uses_strip_F_star(r) for r in rs)
    if F_star_tested is not None:
        def empirical_of(r):
            if uses_strip_F_star(r) and Cu_backfill is None:
                return ("Eqs. 3-4 and 3-5 at the Cu = 4 p. 3-20 takes when "
                        "the backfill's is unknown (F* 1.8 at the top)")
            if uses_strip_F_star(r):
                return (f"Eqs. 3-4 and 3-5 at Cu {Cu_backfill:g} (F* "
                        f"{F_star_strip_top(Cu_backfill):.2f} at the top)")
            if r.is_metallic:
                return ("20 t/St at the top to 10 t/St at 20 ft (Eqs. 3-6, "
                        "3-7)")
            return "(2/3) tan phi_backfill (Eq. 3-8)"
        empirical = by_levels(empirical_of, empirical_of)
        chosen = _tested_text(F_star_tested) + ", from pullout tests"
        if Cu_backfill is not None:
            # GM-256 review: a Cu given beside a tested F* is in no check;
            # say what it does bear on
            if any_strip:
                role = ("the tested F* replaces the empirical F* of Eqs. "
                        "3-4 and 3-5, which is what takes Cu. It sets only "
                        "that empirical F*, given beside this, and is GEC-11 "
                        "Table 3-6 note (i)'s check on it")
            else:
                role = ("the tested F* replaces the empirical F*, and this "
                        "reinforcement's empirical F* takes no Cu. It serves "
                        "only as GEC-11 Table 3-6 note (i)'s check on that "
                        "empirical F*, given beside this")
            chosen += (f". Cu_backfill {Cu_backfill:g} is not used in the "
                       f"checks: {role}: "
                       + (_NOTE_I_WITHIN if Cu_backfill >= CU_UNKNOWN
                          else _NOTE_I_OUTSIDE))
        out.append(choice(
            "Pullout resistance factor F* (F_star_tested)", chosen,
            how_given(F_star_tested, None, name="F_star_tested"),
            F_STAR_TESTED_BASIS,
            ["the empirical F*, " + empirical + against(
                check_internal_stability(*args, Cu_backfill=Cu_backfill,
                                         alpha_tested=alpha_tested))]))
        return out
    if not any_strip:
        if Cu_backfill is None:
            return out
        own = ("20 t/St at the top to 10 t/St at 20 ft (GEC-11 Fig. E4-5)"
               if rs[0].is_metallic else "0.67 tan phi_backfill")
        return out + [choice(
            "Backfill uniformity coefficient (Cu_backfill) against the "
            "empirical pullout factor F*",
            f"Cu {Cu_backfill:g}: "
            + (_NOTE_I_WITHIN if Cu_backfill >= CU_UNKNOWN
               else _NOTE_I_OUTSIDE)
            + f"; this reinforcement's F*, {own}, takes no Cu",
            how_given(Cu_backfill, None, name="Cu_backfill"), F_STAR_NOTE_I,
            [_NOTE_I_TESTS])]
    Cu = CU_UNKNOWN if Cu_backfill is None else Cu_backfill
    top = F_star_strip_top(Cu)
    capped = 1.2 + math.log10(Cu) > F_STAR_STRIP_MAX
    chosen = (f"F* {top:.2f} at the top of the wall (Cu {Cu:g}"
              + (": 1.2 + log Cu is above the 2.0 maximum" if capped else "")
              + f"), linear to tan phi' = "
              f"{math.tan(math.radians(phi_backfill)):.3f} at 20 ft "
              f"(6.1 m) and below")
    if Cu < CU_UNKNOWN:
        chosen += "; Cu is " + _NOTE_I_OUTSIDE
    if not all(uses_strip_F_star(r) for r in rs):
        strip = [s for s in _level_runs([uses_strip_F_star(r) for r in rs])
                 if s[2]]
        chosen += (", at " + _levels_of([(a, z) for a, z, _ in strip])
                   + ", whose F* is the ribbed strip curve")

    if Cu == CU_UNKNOWN:
        alt_cu = 10.0 ** (F_STAR_STRIP_MAX - 1.2)    # 6.31: F* 2.0
        alt = ("a Cu of 6.3 or more, measured on the wall's backfill: F* "
               "2.0 at the top, the maximum (GEC-11 Vol II Examples E3 and "
               "E5 take it for a backfill with Cu 7.0)")
    else:
        alt_cu = CU_UNKNOWN
        alt = ("Cu 4, which GEC-11 Vol I p. 3-20 says to assume when the "
               "backfill's Cu is unknown: F* 1.8 at the top")
    alt += against(check_internal_stability(
        *args, Cu_backfill=alt_cu, alpha_tested=alpha_tested))
    basis = F_STAR_STRIP_BASIS
    if any(r.is_grid and uses_strip_F_star(r) for r in rs):
        basis += (". This steel grid was given no transverse-bar geometry "
                  "(t/St), so its pullout takes the ribbed strip curve")
    return out + [choice(
        "Pullout resistance factor F* of ribbed steel strips, from the "
        "backfill's uniformity coefficient (Cu_backfill)", chosen,
        how_given(Cu_backfill, None, name="Cu_backfill"), basis, [alt])]


def check_internal_stability(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    reinforcement: Reinforcement,
    FOS_pullout: float = 1.5,
    FOS_rupture: Optional[float] = None,
    Cu_backfill: float = None,
    F_star_tested=None,
    alpha_tested: float = None,
) -> List[Dict[str, Any]]:
    """Check internal stability at each reinforcement level.

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry with reinforcement layout.
    gamma_backfill : float
        Backfill unit weight (kN/m³).
    phi_backfill : float
        Backfill friction angle (degrees).
    reinforcement : Reinforcement or list of Reinforcement
        Reinforcement properties: one for every level, or one per level
        from the top down (:func:`level_reinforcements`, GM-266), each
        level then taking its own Tal, width and Sh or Rc, Kr/Ka, F* and
        alpha.
    FOS_pullout : float, optional
        Required FOS for pullout. Default 1.5.
    FOS_rupture : float, optional
        Required FOS for rupture, Tal x Rc over the unfactored Tmax.
        Default None: :func:`rupture_FOS_required`, 1.0 for steel (Tal
        the 0.55 Fy or 0.48 Fy allowable) and 1.5 for a geosynthetic (Tal
        = Tult / RF), GEC-11's 1.35 / phi (GM-262 review). Each level
        gives it as ``FOS_rupture_required``.
    Cu_backfill : float, optional
        Uniformity coefficient D60/D10 of the reinforced backfill, for the
        ribbed steel strip F* (:func:`F_star_strip_top`). Default None: 4,
        as GEC-11 Vol I p. 3-20 says to assume when it is unknown. A
        reinforcement whose F* takes none (:func:`uses_strip_F_star`)
        keeps its F*; the Cu is then only checked against GEC-11 Table 3-6
        note (i), which accepts every empirical F* only for Cu >= 4
        (:func:`pullout_factor_choices`).
    F_star_tested : float or list of [depth_m, F*], optional
        The pullout resistance factor F* from pullout tests on the product
        in the wall's backfill (GEC-11 Vol I Table 3-6 note (i), Appendix
        B): one value at every level, or values at depths below the top of
        the wall, linear between and the end value beyond. Replaces the
        empirical F* (Eqs. 3-4 to 3-8). Default None (the empirical F*).
    alpha_tested : float, optional
        The scale effect correction factor alpha from pullout tests (GEC-11
        Vol I p. 3-16), 0 < alpha <= 1. Default None: Table 3-6's for the
        reinforcement's type (p. 3-17), 1.0 for steel strips and grids, 0.8
        for a geogrid, 0.6 for a geotextile; a "geosynthetic" of no stated
        kind is refused (:func:`pullout_alpha`, GM-256).

    Returns
    -------
    list of dict
        Per-level results. With a sloping backfill each level also gives
        ``Zp_m``, the depth below the slope surface used for pullout.
        ``Pr_kN_per_m`` and ``Tal_Rc_kN_per_m`` are per metre of wall, the
        reinforcement's pullout and tensile resistance times its coverage
        ratio ``Rc`` (GEC-11 Vol I Eq. 4-35, p. 4-49; p. 4-47; GM-256),
        the level's own when the spacing or ratio is given per level
        (:func:`coverage_ratios`, GM-261); ``Tallowable_kN_per_m`` is Tal
        per metre of reinforcement width. With a reinforcement per level
        each level also gives ``reinforcement``, its name (GM-266).

    Notes
    -----
    A sloping backfill (``geom.backfill_slope`` > 0) follows GEC-11 Vol I:
    Ka stays the level-backfill Rankine value (Eq. 4-25, beta = 0); the slope
    adds sigma_2 = S_eq gamma_r to the vertical stress for Tmax, S_eq =
    (1/2)(0.7H) tan(beta) (Fig. 4-11, Eq. 4-29); the zone of maximum stress
    for inextensible reinforcement is 0.3 H1 from the face over the upper
    H1/2 (Fig. 4-9a); pullout uses sigma_v = gamma_r Zp, Zp the depth below
    the slope surface averaged over the resistant zone (Fig. 4-15; Example
    E3, Fig. E3-5). A surcharge counts only where level ground lies over the
    reinforced zone (behind the crest of a broken backslope).

    The surcharge (``geom.surcharge``, the live load) is in Tmax and not in
    the pullout resistance: pullout sigma_v is taken "neglecting traffic
    loads" (GEC-11 p. 4-49; p. 4-51; AASHTO C11.10.6.2.1).
    """
    Ka = rankine_Ka(phi_backfill)
    H = geom.wall_height
    L = geom.reinforcement_length
    Sv = geom.reinforcement_spacing
    sloping = geom.backfill_slope > 0
    # one reinforcement a level, all steel or all geosynthetic (GM-266)
    rs = level_reinforcements(reinforcement, geom.reinforcement_depths)
    per_level = not isinstance(reinforcement, Reinforcement)
    if FOS_rupture is None:
        FOS_rupture = rupture_FOS_required(rs[0])   # GM-262 review
    # GEC-11 Eq. 3-1 (GM-256), one value a level (GM-261)
    Rcs = coverage_ratios(reinforcement, geom.reinforcement_depths)
    _check_tested(F_star_tested, alpha_tested)
    alphas = _level_alphas(reinforcement, rs, alpha_tested)
    if sloping:
        _check_backslope(geom, None, geom.surcharge)
        bs = backslope_geometry(geom)
        sigma_2 = gamma_backfill * bs["S_eq"]             # Fig. 4-11
        q_int = geom.surcharge if bs["level_length"] > 0 else 0.0
        H1 = bs["H1"]                                     # Fig. 4-9a
    # Select the depth-variation curve family by reinforcement type:
    #  - steel bar mat / welded grid -> bar-mat curves (Kr/Ka 2.5->1.2,
    #    F* 20(t/St)->10(t/St))
    #  - ribbed metallic strip       -> strip curves (Kr/Ka 1.7->1.2,
    #    F* 1.2 + log Cu (2.0 at most) -> tan-phi)  [the default]
    #  - geosynthetic                -> Kr/Ka 1.0, F* 0.67*tan-phi
    # each level by its own reinforcement (GM-266)
    if Cu_backfill is not None:
        # a grid with t/St or a geosynthetic takes no Cu in its F*; GEC-11
        # Table 3-6 note (i) still asks Cu >= 4 of every empirical F*
        # (pullout_factor_choices), so the value is checked, not refused
        F_star_strip_top(Cu_backfill)

    results = []
    for z, Rc, r, alpha in zip(geom.reinforcement_depths, Rcs, rs, alphas):
        if r.is_grid:
            r_type = "metallic_grid"
        elif r.is_metallic:
            r_type = "metallic"
        else:
            r_type = "geosynthetic"
        t_over_St = r.t_over_St if r.is_grid else None
        Tal_Rc = r.Tallowable * Rc   # per metre of wall, p. 4-47
        # Kr/Ka ratio
        kr_ka = Kr_Ka_ratio(z, r_type)

        # Tmax at this level (a sloping backfill adds sigma_2, Eq. 4-29)
        if sloping:
            T = Tmax_at_level(z, gamma_backfill, Ka, kr_ka, Sv,
                              q_int + sigma_2)
        else:
            T = Tmax_at_level(z, gamma_backfill, Ka, kr_ka, Sv, geom.surcharge)

        # Effective length Le = total length - active zone length La.
        # GEC-11 / AASHTO Fig. 11.10.6.3.1-1: the failure surface depends
        # on reinforcement extensibility (RW-2):
        #  - Inextensible (metallic): bilinear coherent-gravity surface —
        #    vertical at 0.3H from the face over the upper half of the
        #    wall, then tapering linearly to the toe:
        #       La = 0.3*H        for z <= H/2
        #       La = 0.6*(H - z)  for z >  H/2
        #    Under a sloping backfill H becomes H1 (GEC-11 Fig. 4-9a): the
        #    upper H1/2 at 0.3*H1, the lower H1/2 tapering to the toe.
        #  - Extensible (geosynthetic): Rankine wedge at 45+phi/2:
        #       La = (H - z)*tan(45 - phi/2)
        if r.is_metallic and sloping:
            La = 0.3 * H1 if (H - z) >= H1 / 2.0 else 0.6 * (H - z)
        elif r.is_metallic:
            La = 0.3 * H if z <= H / 2.0 else 0.6 * (H - z)
        else:
            La = (H - z) * math.tan(math.radians(45 - phi_backfill / 2))
        Le = max(L - La, 1.0)  # minimum 1m effective length

        # Pullout resistance
        if F_star_tested is not None:
            # from pullout tests (GEC-11 Table 3-6 note (i))
            F_star = F_star_from_tests(F_star_tested, z)
        elif r.is_grid and t_over_St is not None:
            # Steel bar-mat / welded-grid F* = 20(t/St) -> 10(t/St)
            F_star = F_star_metallic(z, phi_backfill, "metallic_grid", t_over_St)
        elif r.is_metallic:
            # Ribbed strip F* = 1.2 + log Cu -> tan(phi) (also the fallback
            # for a grid whose transverse geometry t/St was not supplied)
            F_star = F_star_metallic(z, phi_backfill, Cu=Cu_backfill)
        else:
            F_star = 0.67 * math.tan(math.radians(phi_backfill))

        # Pullout sigma_v is "the unfactored vertical stress at the
        # reinforcement level in the resistant zone, including distributed
        # dead load surcharges, neglecting traffic loads" (GEC-11 p. 4-49).
        # geom.surcharge is the live load (the LRFD check factors it with
        # gamma_LL), so it is in Tmax above and not in pullout here.
        if sloping:
            # sigma_v = gamma_r Zp, Zp below the slope surface averaged over
            # the resistant zone (GEC-11 Fig. 4-15, Fig. E3-5)
            Zp = z + _mean_rise(max(L - Le, 0.0), L, geom)
            Pr = pullout_resistance(
                Zp, gamma_backfill, Le, F_star, alpha_pullout=alpha, Rc=Rc,
            )
        else:
            Pr = pullout_resistance(
                z, gamma_backfill, Le, F_star, alpha_pullout=alpha, Rc=Rc,
            )

        # FOS checks, both per metre of wall: the tensile resistance is Tal
        # times the coverage ratio (GEC-11 p. 4-47, Eqs. 4-33 and 4-34)
        FOS_po = Pr / T if T > 0 else 99.9
        FOS_ru = Tal_Rc / T if T > 0 else 99.9

        passes = (FOS_po >= FOS_pullout) and (FOS_ru >= FOS_rupture)

        level = {"depth_m": round(z, 2)}
        if per_level:
            level["reinforcement"] = r.name
        level.update({
            "Tmax_kN_per_m": round(T, 2),
            "Pr_kN_per_m": round(Pr, 2),
            "La_m": round(La, 2),
            "Le_m": round(Le, 2),
            "Kr_Ka": round(kr_ka, 3),
            "F_star": round(F_star, 3),
            "alpha": alpha,
            "Rc": round(Rc, 4),
            "FOS_pullout": round(FOS_po, 3),
            "FOS_rupture": round(FOS_ru, 3),
            "FOS_rupture_required": FOS_rupture,
            "Tallowable_kN_per_m": round(r.Tallowable, 2),
            "Tal_Rc_kN_per_m": round(Tal_Rc, 2),
            "passes": passes,
        })
        if sloping:
            level["Zp_m"] = round(Zp, 2)
        results.append(level)

    return results


def check_external_stability(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    gamma_foundation: float,
    phi_foundation: float,
    c_foundation: float = 0.0,
    q_allowable: float = None,
    FOS_sliding_req: float = 1.5,
    FOS_overturning_req: float = 2.0,
    phi_retained: float = None,
    gamma_retained: float = None,
    traffic_vertical_resists: bool = False,
    live_load_resists: bool = False,
    surcharge_in_bearing: bool = True,
) -> Dict[str, Any]:
    """Check external stability of MSE wall as rigid block (ASD).

    FOS_sliding = (V tan(delta_b) + ca L) / P_a, FOS_overturning =
    M_stabilizing / M_overturning, both about the toe of the reinforced
    zone, and the Meyerhof bearing pressure sigma_v = V_b / (L - 2 e_B).
    V and M_stabilizing count the live-load surcharge over the reinforced
    zone only with ``live_load_resists`` (default: not, GM-164); V_b and
    e_B count it as ``surcharge_in_bearing`` says (default: they do). Its
    thrust drives every check.

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry.
    gamma_backfill : float
        Reinforced fill unit weight (kN/m³).
    phi_backfill : float
        Reinforced fill friction angle (degrees).
    gamma_foundation : float
        Foundation soil unit weight (kN/m³). Not used by these checks:
        bearing compares the bearing pressure with ``q_allowable`` and no
        passive resistance is counted. The LRFD path works a bearing
        resistance out from it (``check_external_stability_lrfd``).
    phi_foundation : float
        Foundation friction angle (degrees).
    c_foundation : float, optional
        Foundation cohesion (kPa). Default 0.
    q_allowable : float, optional
        Allowable bearing (kPa). Default None: the bearing check is NOT RUN
        (FOS_bearing and passes_bearing are None, bearing_checked False);
        sigma_v is still reported.
    FOS_sliding_req : float, optional
        Required sliding FOS. Default 1.5.
    FOS_overturning_req : float, optional
        Required overturning FOS. Default 2.0.
    phi_retained : float, optional
        Retained soil friction angle (degrees). If None, uses phi_backfill.
    gamma_retained : float, optional
        Retained soil unit weight (kN/m³). If None, uses gamma_backfill.
    traffic_vertical_resists : bool, optional
        Broken backslope with traffic behind the crest only: count the
        traffic thrust's vertical component F2 sin(I) as sliding and
        overturning resistance, as GEC-11 Example E1 does. Default False:
        GEC-11 Vol I p. 4-22 counts loads that increase sliding resistance
        "only if those loads are permanent". It is counted in bearing
        either way (p. 4-26).
    live_load_resists : bool, optional
        Count the live-load surcharge over the reinforced zone (q L; behind
        the crest of a broken backslope, q over the level ground on the
        zone) as resisting weight in sliding and overturning. Default False
        (:data:`LIVE_LOAD_BASIS`, GM-164); True is the module's reading
        before GM-164.
    surcharge_in_bearing : bool, optional
        Count that surcharge in the bearing pressure and its eccentricity
        e_B. Default True (:data:`BEARING_BASIS`).

    Returns
    -------
    dict
        FOS_sliding / FOS_overturning / FOS_bearing, passes_sliding /
        passes_overturning / passes_bearing, bearing_checked, sigma_v_kPa,
        L_eff_m, eccentricity_m (e_B, the eccentricity for bearing, on the
        bearing loads), and ``passes``: True only when every check was run
        and passed, False when one failed, None when none failed but the
        bearing check was not run (GM-100). With a sloping backfill also
        ``backslope``: the GEC-11 basis (Kab, h, F_T and its components, V2).
        The ASD path has no separate eccentricity (middle-third) check; the
        overturning check stands for it.

    Notes
    -----
    A sloping backfill (``geom.backfill_slope`` > 0, GEC-11 Vol I §4.4.4)
    takes the Coulomb Kab with delta = beta and the thrust F_T on the plane
    of height h = H + L tan(beta) at the back of the reinforced zone,
    inclined at beta (Eqs. 4-2, 4-3, 4-7; ``backslope_loads``). Its
    horizontal component drives sliding and overturning (Eq. 4-10); its
    vertical component, at the back of the zone, and the slope fill over
    the zone, V2, resist them and load the base (Eqs. 4-13, 4-16, 4-21),
    unfactored here. A negative eccentricity leaves the full width L
    (GEC-11 p. 4-28).
    """
    return _external_dict(
        _external_terms(geom, gamma_backfill, phi_backfill, phi_foundation,
                        c_foundation, phi_retained, gamma_retained,
                        traffic_vertical_resists, live_load_resists,
                        surcharge_in_bearing),
        q_allowable, FOS_sliding_req, FOS_overturning_req)


def _external_terms(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    phi_foundation: float,
    c_foundation: float = 0.0,
    phi_retained: float = None,
    gamma_retained: float = None,
    traffic_vertical_resists: bool = False,
    live_load_resists: bool = False,
    surcharge_in_bearing: bool = True,
) -> Dict[str, Any]:
    """The ASD external-stability terms, unrounded (GM-164), as
    :func:`check_external_stability` sums them.

    ``vertical_forces``: each vertical force on the base, in the order the
    checks sum them, as ``{"component", "kind", "W", "x", "M", "resists",
    "bearing"}`` (kN/m; the arm x from the toe of the reinforced zone, m;
    ``kind`` "weight", "surcharge" or "thrust"; ``resists``: counted in
    sliding and overturning; ``bearing``: counted in the bearing pressure).
    ``horizontal_forces``: ``{"component", "P", "z", "M"}`` (z above the
    base). Then ``Ka`` (the retained fill's; Kab with a sloping backfill),
    the ``phi_retained``, ``gamma_retained``, ``phi_backfill`` and
    ``phi_foundation`` used; ``V`` and ``M_stabilizing`` (sliding and
    overturning), ``V_left_out``, ``V_bearing`` and ``M_bearing``, ``Pa``
    and ``M_overturning``;
    ``delta_b``, ``ca``, ``resisting``; and the bearing resultant
    ``x_resultant``, ``eccentricity`` (e_B), ``L_eff``, ``sigma_v``.
    With a sloping backfill ``backslope_loads`` (:func:`backslope_loads`),
    else None."""
    H = geom.wall_height
    L = geom.reinforcement_length

    # Use retained fill properties for external active pressure if provided
    phi_ext = phi_retained if phi_retained is not None else phi_backfill
    gamma_ext = gamma_retained if gamma_retained is not None else gamma_backfill

    def weight(component, W, x, kind="weight", resists=True, bearing=True):
        return {"component": component, "kind": kind, "W": W, "x": x,
                "M": W * x, "resists": resists, "bearing": bearing}

    def thrust(component, P, z):
        return {"component": component, "P": P, "z": z, "M": P * z}

    ld = None
    if geom.backfill_slope > 0:
        ld = backslope_loads(geom, gamma_backfill, phi_ext, gamma_ext,
                             geom.surcharge)
        Ka = ld["Kab"]
        bs = ld["geometry"]
        # The weights on the base, the live load over any level ground on
        # the zone (Vs) as chosen (GM-164), and the thrust's vertical
        # component at the back. The traffic thrust's vertical component
        # resists only by choice (Example E1; not by p. 4-22); bearing
        # takes it.
        vertical = [weight("Reinforced fill (V1)", ld["V1"], L / 2.0)]
        if ld["V2"] > 0:
            vertical.append(weight("Slope fill over the reinforced zone (V2)",
                                   ld["V2"], bs["V2_arm"]))
        if ld["Vs"] > 0:
            vertical.append(weight(
                "Live-load surcharge over the level ground on the reinforced "
                "zone", ld["Vs"], L - bs["level_length"] / 2.0,
                "surcharge", live_load_resists, surcharge_in_bearing))
        vertical.append(weight("Vertical component of F_T (F_T sin I)",
                               ld["FTV"], L, "thrust"))
        if ld["F2V"] > 0:
            vertical.append(weight(
                "Vertical component of the traffic thrust (F2 sin I)",
                ld["F2V"], L, "thrust", traffic_vertical_resists, True))
        horizontal = [thrust("Retained fill thrust (F_T cos I)", ld["FTH"],
                             ld["h"] / 3.0)]
        if ld["F2H"] > 0:
            horizontal.append(thrust("Traffic thrust (F2 cos I)", ld["F2H"],
                                     ld["h"] / 2.0))
    else:
        Ka = rankine_Ka(phi_ext)
        # Weight of the reinforced block (reinforced fill properties) and
        # the live load over it, counted as chosen (GM-164)
        vertical = [weight("Reinforced fill (V1)", gamma_backfill * H * L,
                           L / 2.0)]
        if geom.surcharge > 0:
            vertical.append(weight(
                "Live-load surcharge over the reinforced zone (q L)",
                geom.surcharge * L, L / 2.0, "surcharge", live_load_resists,
                surcharge_in_bearing))
        # Active thrust of the retained fill behind the reinforced zone,
        # and of the live load over it (GEC-11 Fig. 4-2)
        parts = active_force_components(gamma_ext, H, Ka, q=geom.surcharge)
        horizontal = [thrust("Retained fill thrust (F1)",
                             parts["Pa_earth"], H / 3.0)]
        if parts["Pa_surcharge"] > 0:
            horizontal.append(thrust("Live-load surcharge thrust (F2)",
                                     parts["Pa_surcharge"], H / 2.0))

    V = sum(f["W"] for f in vertical if f["resists"])
    M_stab = sum(f["M"] for f in vertical if f["resists"])
    V_bearing = sum(f["W"] for f in vertical if f["bearing"])
    M_bearing = sum(f["M"] for f in vertical if f["bearing"])
    Pa = sum(f["P"] for f in horizontal)                  # horizontal driving
    M_over = sum(f["M"] for f in horizontal)

    # Sliding: soil-on-soil interface, use min(phi_backfill, phi_foundation)
    # per GEC-11 Section 4.3 (no 2/3 reduction for soil-on-soil)
    delta_b = min(phi_backfill, phi_foundation)
    ca = 2.0 / 3.0 * c_foundation
    R_sliding = V * math.tan(math.radians(delta_b)) + ca * L

    # Bearing — Meyerhof uniform pressure over the effective width
    # (AASHTO 11.10.5.4 / GEC-11 Eq. 4-13): sigma_v = V_b / (L - 2e_B).
    # This replaces the trapezoidal q_toe = (W/L)(1 + 6e/L) convention,
    # which is not the AASHTO/GEC-11 MSE convention (RW-3). A negative e
    # (resultant behind the centre) leaves L - 2e = L (GEC-11 p. 4-28).
    x_R = (M_bearing - M_over) / V_bearing if V_bearing > 0 else L / 2.0
    e = L / 2.0 - x_R
    L_eff = L - 2.0 * max(e, 0.0)
    if L_eff > 0:
        sigma_v = V_bearing / L_eff
    else:
        sigma_v = 1.0e6  # resultant off the base — bearing effectively fails

    return {
        "vertical_forces": vertical, "horizontal_forces": horizontal,
        "Ka": Ka, "phi_retained": phi_ext, "gamma_retained": gamma_ext,
        "V": V, "M_stabilizing": M_stab,
        "V_left_out": sum(f["W"] for f in vertical if not f["resists"]),
        "V_bearing": V_bearing, "M_bearing": M_bearing,
        "Pa": Pa, "M_overturning": M_over,
        "phi_backfill": phi_backfill, "phi_foundation": phi_foundation,
        "delta_b": delta_b, "ca": ca, "L": L, "resisting": R_sliding,
        "x_resultant": x_R, "eccentricity": e, "L_eff": L_eff,
        "sigma_v": sigma_v,
        "traffic_vertical_resists": bool(traffic_vertical_resists),
        "live_load_resists": bool(live_load_resists),
        "surcharge_in_bearing": bool(surcharge_in_bearing),
        "backslope_loads": ld,
    }


def live_load_choices(geom: MSEWallGeometry,
                      live_load_resists: bool = False,
                      surcharge_in_bearing: bool = True) -> list:
    """How the ASD path (:func:`check_external_stability`) counted a
    live-load surcharge over the reinforced zone (GM-164), as a result's
    ``choices`` (``geotech_common.choices``): one entry for sliding and
    overturning and one for bearing; none when no live load stands over
    the zone (none given, or all of it behind the crest of a broken
    backslope)."""
    from geotech_common.choices import choice, how_given
    if geom.surcharge <= 0:
        return []
    if (geom.backfill_slope > 0
            and backslope_geometry(geom)["level_length"] <= 0):
        return []
    counted = "counted as a resisting force"
    left_out = "not a resisting force"
    flag = live_load_resists
    out = [choice(
        "live-load surcharge over the reinforced zone, in sliding and "
        "overturning (ASD)",
        f"{counted if flag else left_out} (live_load_resists={flag})",
        how_given(flag, False, name="live_load_resists"),
        LIVE_LOAD_BASIS[flag],
        [f"{left_out if flag else counted} (live_load_resists={not flag}): "
         f"{LIVE_LOAD_BASIS[not flag]}"])]
    flag = surcharge_in_bearing
    out.append(choice(
        "live-load surcharge over the reinforced zone, in the bearing "
        "pressure (ASD)",
        f"{'counted' if flag else 'left out'} (surcharge_in_bearing={flag})",
        how_given(flag, True, name="surcharge_in_bearing"),
        BEARING_BASIS[flag],
        [f"{'left out' if flag else 'counted'} "
         f"(surcharge_in_bearing={not flag}): {BEARING_BASIS[not flag]}"]))
    return out


def _external_dict(t: Dict[str, Any], q_allowable: float = None,
                   FOS_sliding_req: float = 1.5,
                   FOS_overturning_req: float = 2.0) -> Dict[str, Any]:
    """The :func:`check_external_stability` result from its terms
    (:func:`_external_terms`)."""
    Pa, M_over = t["Pa"], t["M_overturning"]
    FOS_sliding = t["resisting"] / Pa if Pa > 0 else 99.9

    # Overturning about toe
    FOS_overturning = (t["M_stabilizing"] / M_over if M_over > 0 else 99.9)

    sigma_v, L_eff, e = t["sigma_v"], t["L_eff"], t["eccentricity"]
    backslope = (None if t["backslope_loads"] is None
                 else _backslope_summary(t["backslope_loads"],
                                         t["traffic_vertical_resists"]))

    # Bearing check (GM-100): without q_allowable it is NOT RUN, never a
    # placeholder FOS reported as a pass.
    if q_allowable is None:
        FOS_bearing = None
        passes_bearing = None
    else:
        FOS_bearing = q_allowable / sigma_v
        passes_bearing = FOS_bearing >= 1.0

    passes_sliding = FOS_sliding >= FOS_sliding_req
    passes_overturning = FOS_overturning >= FOS_overturning_req
    if not (passes_sliding and passes_overturning
            and passes_bearing is not False):
        passes = False
    elif passes_bearing is None:
        passes = None       # nothing failed, but bearing was not checked
    else:
        passes = True

    out = {
        "FOS_sliding": round(FOS_sliding, 3),
        "FOS_overturning": round(FOS_overturning, 3),
        "FOS_bearing": None if FOS_bearing is None else round(FOS_bearing, 3),
        "passes_sliding": passes_sliding,
        "passes_overturning": passes_overturning,
        "passes_bearing": passes_bearing,
        "bearing_checked": q_allowable is not None,
        # Meyerhof uniform bearing pressure over effective width L - 2e
        # (key name kept for backward compatibility)
        "q_toe_kPa": round(sigma_v, 1),
        "sigma_v_kPa": round(sigma_v, 1),
        "L_eff_m": round(max(L_eff, 0.0), 3),
        "eccentricity_m": round(e, 3),
        "passes": passes,
    }
    if backslope is not None:
        out["backslope"] = backslope
    return out


#: GEC-11 Vol I p. 4-28: the nominal bearing resistance of Eq. 4-22.
NOMINAL_BEARING_BASIS = (
    "GEC-11 Vol I Eq. 4-22 (AASHTO Eq. 10.6.3.1.2a-1): qn = c_f Nc + "
    "0.5 L' gamma_f Ngamma over L' = L - 2 eB of each combination (L' = L "
    "when eB < 0), Nc and Ngamma from Table 4-6 (AASHTO Table "
    "10.6.3.1.2a-1: Prandtl's Nc, Vesic's Ngamma); a level grade in front "
    "of the wall and no groundwater influence, the wall's embedment "
    "neglected (p. 4-28); qR = phi_b qn (Eq. 4-23)")


def bearing_factors(phi_deg: float) -> Dict[str, float]:
    """Nc, Nq and Ngamma of GEC-11 Table 4-6 (AASHTO Table
    10.6.3.1.2a-1), from the equations the table notes: Nq (Reissner) =
    e^(pi tan phi) tan^2(45 + phi/2), Nc (Prandtl) = (Nq - 1) cot phi
    (5.14 at phi = 0), Ngamma (Vesic) = 2 (Nq + 1) tan phi. They agree with
    the table within 0.1 (phi = 30: 30.1, 18.4, 22.4)."""
    if phi_deg < 0 or phi_deg >= 90:
        raise ValueError(f"phi_foundation must be in [0, 90), got {phi_deg}")
    t = math.tan(math.radians(phi_deg))
    Nq = math.exp(math.pi * t) * math.tan(math.radians(45 + phi_deg / 2)) ** 2
    Nc = (Nq - 1.0) / t if phi_deg > 0 else 2.0 + math.pi
    return {"Nc": Nc, "Nq": Nq, "Ngamma": 2.0 * (Nq + 1.0) * t}


def nominal_bearing_resistance(L_eff: float, gamma_foundation: float,
                               phi_foundation: float,
                               c_foundation: float = 0.0) -> float:
    """The nominal bearing resistance qn (kPa) under an MSE wall, GEC-11 Vol
    I Eq. 4-22: qn = c_f Nc + 0.5 L' gamma_f Ngamma, for a level grade in
    front of the wall and no groundwater influence, the embedment neglected
    (p. 4-28). ``L_eff`` is L' = L - 2 eB (m), ``gamma_foundation`` the
    foundation soil's unit weight (kN/m3). Reproduces Vol II Example E2:
    L' = 12.46 ft, 125 pcf, phi = 30 deg -> 17.44 ksf."""
    f = bearing_factors(phi_foundation)
    return c_foundation * f["Nc"] + 0.5 * L_eff * gamma_foundation * f["Ngamma"]


#: A bearing resistance worked out from the foundation soil takes that
#: soil's own friction angle, never the reinforced fill's (GM-179 review).
NEEDS_PHI_FOUNDATION = (
    "gamma_foundation works the bearing resistance out by GEC-11 Eq. 4-22 "
    "with Nc and Ngamma for the foundation soil's friction angle: give "
    "phi_foundation (and c_foundation if it has cohesion); the reinforced "
    "fill's phi_backfill is not taken for it")


def bearing_resistance_choices(phi_bearing: float) -> list:
    """How the LRFD path worked the strength bearing resistance out from
    the foundation soil (GM-179), as a result's ``choices``
    (``geotech_common.choices``)."""
    from geotech_common.choices import choice, how_given
    return [choice(
        "factored bearing resistance qR (Strength I)",
        f"phi_b qn, phi_b = {phi_bearing:g} (phi_bearing), qn by GEC-11 "
        f"Eq. 4-22",
        how_given(phi_bearing, 0.65, name="phi_bearing"),
        "GEC-11 Vol I Table 4-5 (AASHTO Table 11.5.6-1): bearing "
        "resistance factor 0.65; " + NOMINAL_BEARING_BASIS,
        ["a factored resistance given as bearing_resistance_strength in "
         "place of gamma_foundation: with a slope in front of the wall or "
         "a high water table, qn by AASHTO 10.6.3.1.2 (GEC-11 Vol I "
         "p. 4-28)"])]


def check_external_stability_lrfd(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    phi_foundation: float,
    *,
    phi_retained: float = None,
    gamma_retained: float = None,
    live_load: float = None,
    bearing_resistance_strength: float = None,
    bearing_resistance_service: float = None,
    c_foundation: float = 0.0,
    gamma_EV_max: float = 1.35,
    gamma_EV_min: float = 1.00,
    gamma_EH_max: float = 1.50,
    gamma_EH_min: float = 0.90,
    gamma_LL: float = 1.75,
    phi_sliding: float = 1.0,
    ecc_limit_ratio: float = 0.25,
    traffic_vertical_resists: bool = False,
    gamma_foundation: float = None,
    phi_bearing: float = 0.65,
) -> Dict[str, Any]:
    """AASHTO/GEC-11 LRFD external stability for an MSE wall (rigid block).

    Computes the full capacity:demand ratio (CDR) set — sliding, limiting
    eccentricity (overturning), and bearing — under the AASHTO Strength I
    (max and min load-factor) and Service I load combinations, with the GEC-11
    load-side bookkeeping built in:

    * The **live-load surcharge is EXCLUDED from the resisting side** for
      sliding and eccentricity (a vertical LL over the mass is destabilizing
      only through its horizontal thrust, so counting its weight would be
      unconservative) and **INCLUDED for bearing** (it raises the bearing
      stress).
    * Each mode is checked for the governing load combination. The **critical**
      ("min-vertical") combination pairs the minimum vertical load factor
      (least resistance / largest eccentricity, ``gamma_EV_min`` on all vertical
      gravity loads) with the maximum horizontal/driving factors
      (``gamma_EH_max``, ``gamma_LL``).

    This is the LRFD counterpart to the ASD ``check_external_stability``; it does
    NOT change that function. It reproduces the GEC-11 Example E4 (FHWA-NHI-10-025,
    Appendix E) external-stability results (sliding CDR 1.85/2.08/1.37,
    eccentricity eL 2.87/3.87 ft vs L/4, bearing sigma_v 6.70 ksf / CDR 1.57,
    Service sigma_v 4.66 ksf).

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry. ``geom.surcharge`` is the live-load surcharge unless
        ``live_load`` overrides it.
    gamma_backfill, phi_backfill : float
        Reinforced fill unit weight (kN/m3) and friction angle (deg).
    phi_foundation : float
        Foundation friction angle (deg); the sliding interface uses
        min(phi_backfill, phi_foundation).
    phi_retained, gamma_retained : float, optional
        Retained (behind-the-mass) fill properties for the external active
        thrust. Default to the reinforced-fill values.
    live_load : float, optional
        Live-load surcharge q (kPa). Default: ``geom.surcharge``.
    bearing_resistance_strength : float, optional
        Factored bearing resistance qR at the strength limit (kPa) = phi_b*qn.
        If given, the strength/critical bearing CDR = qR / sigma_v is reported.
        Give it or ``gamma_foundation``, not both.
    gamma_foundation : float, optional
        Foundation soil unit weight (kN/m3) (GM-179). With it the strength
        bearing resistance is worked out for each combination instead of
        given: qn by GEC-11 Eq. 4-22 over that combination's L' with
        ``phi_foundation`` and ``c_foundation``
        (:func:`nominal_bearing_resistance`), qR = ``phi_bearing`` qn
        (Eq. 4-23), as Vol II Example E2 does. Service I keeps its given
        resistance, a settlement criterion, not qn.
    phi_bearing : float, optional
        Bearing resistance factor phi_b for a qn worked out from
        ``gamma_foundation``. Default 0.65 (GEC-11 Table 4-5).
    bearing_resistance_service : float, optional
        Service bearing resistance (kPa). If given, the Service I bearing CDR is
        reported.
    c_foundation : float, optional
        Foundation cohesion (kPa) for sliding adhesion (2/3 c). Default 0.
    gamma_EV_max, gamma_EV_min : float, optional
        Vertical earth (EV) load factors. Defaults 1.35 / 1.00 (AASHTO Str I).
    gamma_EH_max, gamma_EH_min : float, optional
        Horizontal earth (EH) load factors. Defaults 1.50 / 0.90.
    gamma_LL : float, optional
        Live-load (LL) factor. Default 1.75.
    phi_sliding : float, optional
        Sliding resistance factor. Default 1.0 (AASHTO 11.10.5.3).
    ecc_limit_ratio : float, optional
        Limiting-eccentricity ratio e_max/L. Default 0.25 (L/4, MSE on soil;
        use 1/6 for foundations on rock).
    traffic_vertical_resists : bool, optional
        Broken backslope with traffic behind the crest only: count the
        traffic thrust's vertical component F2 sin(I) as sliding and
        eccentricity resistance (LL factor), as GEC-11 Example E1 does
        (Vol II pp. E1-7 to E1-9). Default False: GEC-11 Vol I p. 4-22
        counts external loads that increase sliding resistance "only if
        those loads are permanent". Bearing counts it either way (p. 4-26).

    Returns
    -------
    dict
        Nested {sliding, eccentricity, bearing} CDR/eL/sigma_v results (SI:
        kN/m, kN-m/m, kPa, m), plus the unfactored load components and the load
        factors used. All ratios are capacity:demand (>=1.0 passes). The
        bearing checks run only against a resistance given: without
        ``bearing_resistance_strength`` the Strength I bearing check is not
        run, without ``bearing_resistance_service`` the Service I one is not
        (GM-100). ``checks_failed`` and ``checks_not_run`` name them;
        ``bearing.passes`` is None when neither ran; the overall ``passes`` is
        True only when every check ran and passed, False when one failed,
        None when none failed but one was not run. With a sloping backfill
        also ``backslope`` (the GEC-11 basis: Kab, h, F_T, V2).

    Notes
    -----
    A sloping backfill (``geom.backfill_slope`` > 0) follows GEC-11 Vol I
    §4.4.6 as worked in Vol II Examples E3 and E1 (``backslope_loads``): the
    thrust F_T (Eq. 4-7) is an EH load, its horizontal component driving
    (Eq. 4-11) and its vertical component resisting at the back of the zone
    with the same EH factor (Eq. 4-13, Eq. 4-16), "to maintain consistency
    between the total inclined lateral force and its components" (Example
    E3, Tables E3-6.1 to E3-6.3); the slope fill V2 is an EV load. Traffic
    behind the crest of a broken backslope gives F2 = Kab q h, inclined at
    I, with the LL factor (Example E1); whether its vertical component also
    resists sliding and eccentricity is ``traffic_vertical_resists``. A
    negative bearing eccentricity leaves L' = L (p. 4-28).
    """
    H = geom.wall_height
    L = geom.reinforcement_length
    phi_ext = phi_retained if phi_retained is not None else phi_backfill
    gamma_ext = gamma_retained if gamma_retained is not None else gamma_backfill
    q = geom.surcharge if live_load is None else live_load
    if gamma_foundation is not None:
        if bearing_resistance_strength is not None:
            raise ValueError(
                "Give bearing_resistance_strength (a factored resistance qR) "
                "or gamma_foundation (to work qR out by GEC-11 Eq. 4-22), "
                "not both.")
        if not gamma_foundation > 0:
            raise ValueError(
                f"gamma_foundation must be > 0, got {gamma_foundation}")
        if not 0 < phi_bearing <= 1:
            raise ValueError(f"phi_bearing must be in (0, 1], got "
                             f"{phi_bearing}")
    soil = (None if gamma_foundation is None else
            (gamma_foundation, phi_foundation, c_foundation, phi_bearing, L))
    if geom.backfill_slope > 0:
        return _external_lrfd_sloping(
            geom, gamma_backfill, phi_backfill, phi_foundation, phi_ext,
            gamma_ext, q, bearing_resistance_strength,
            bearing_resistance_service, c_foundation, gamma_EV_max,
            gamma_EV_min, gamma_EH_max, gamma_EH_min, gamma_LL, phi_sliding,
            ecc_limit_ratio, traffic_vertical_resists, soil)
    Ka = rankine_Ka(phi_ext)

    # Unfactored load components about the toe (Point A). Vertical loads V*
    # resist overturning (arm L/2); horizontal loads F* drive it.
    V1 = gamma_backfill * H * L          # EV: reinforced mass weight
    Vs = q * L                           # LL: vertical surcharge over the mass
    F1 = 0.5 * Ka * gamma_ext * H ** 2   # EH: active earth thrust (arm H/3)
    F2 = Ka * q * H                      # LL: surcharge thrust (arm H/2)
    MV1 = V1 * (L / 2.0)
    MVs = Vs * (L / 2.0)
    MF1 = F1 * (H / 3.0)
    MF2 = F2 * (H / 2.0)

    delta_b = min(phi_backfill, phi_foundation)
    tan_d = math.tan(math.radians(delta_b))
    ca = (2.0 / 3.0) * c_foundation

    # --- Sliding (LL excluded from resistance) -----------------------------
    def _sliding_cdr(ev, eh):
        demand = eh * F1 + gamma_LL * F2
        resist = phi_sliding * (ev * V1 * tan_d + ca * L)
        return (resist / demand) if demand > 0 else 99.9

    cdr_slide_max = _sliding_cdr(gamma_EV_max, gamma_EH_max)   # all-max
    cdr_slide_min = _sliding_cdr(gamma_EV_min, gamma_EH_min)   # all-min
    cdr_slide_crit = _sliding_cdr(gamma_EV_min, gamma_EH_max)  # min V, max H

    # --- Limiting eccentricity (LL excluded from resistance) ---------------
    MOA = gamma_EH_max * MF1 + gamma_LL * MF2   # max overturning about toe

    def _eL(ev):
        Vr = ev * V1
        a = (ev * MV1 - MOA) / Vr
        return L / 2.0 - a

    eL_max = _eL(gamma_EV_max)     # Strength I max (EV=1.35)
    eL_crit = _eL(gamma_EV_min)    # min-vertical (EV=1.00): largest eccentricity
    e_limit = ecc_limit_ratio * L

    # --- Bearing (LL included; min-vertical combo uses gamma_EV_min on all
    #     vertical gravity loads) ---------------------------------------------
    def _bearing(ev_soil, ev_ll):
        SV = ev_soil * V1 + ev_ll * Vs
        MR = ev_soil * MV1 + ev_ll * MVs
        a = (MR - MOA) / SV
        eL = L / 2.0 - a
        Bp = L - 2.0 * eL
        sigma = (SV / Bp) if Bp > 0 else 1.0e6
        return sigma, eL, Bp

    sig_str, eL_bear, Bp_str = _bearing(gamma_EV_max, gamma_LL)   # Str I max
    sig_crit, _, Bp_crit = _bearing(gamma_EV_min, gamma_EV_min)   # min-vertical

    # Service I: all factors 1.0, LL included, overturning without EH/LL factors.
    SV_s = V1 + Vs
    MR_s = MV1 + MVs
    MO_s = MF1 + MF2
    a_s = (MR_s - MO_s) / SV_s
    eL_s = L / 2.0 - a_s
    Bp_s = L - 2.0 * eL_s
    sig_svc = (SV_s / Bp_s) if Bp_s > 0 else 1.0e6

    return _lrfd_result(
        cdr_slide_max, cdr_slide_min, cdr_slide_crit, eL_max, eL_crit,
        e_limit, sig_str, eL_bear, Bp_str, sig_crit, sig_svc,
        bearing_resistance_strength, bearing_resistance_service,
        {"V1": round(V1, 3), "Vs": round(Vs, 3),
         "F1": round(F1, 3), "F2": round(F2, 3),
         "Ka_retained": round(Ka, 4)},
        {"EV_max": gamma_EV_max, "EV_min": gamma_EV_min,
         "EH_max": gamma_EH_max, "EH_min": gamma_EH_min,
         "LL": gamma_LL, "phi_sliding": phi_sliding,
         "ecc_limit_ratio": ecc_limit_ratio},
        soil, Bp_crit)


def _external_lrfd_sloping(
    geom, gamma_backfill, phi_backfill, phi_foundation, phi_ext, gamma_ext,
    q, bearing_resistance_strength, bearing_resistance_service, c_foundation,
    gamma_EV_max, gamma_EV_min, gamma_EH_max, gamma_EH_min, gamma_LL,
    phi_sliding, ecc_limit_ratio, traffic_vertical_resists=False, soil=None,
) -> Dict[str, Any]:
    """``check_external_stability_lrfd`` with a sloping backfill (GEC-11 Vol
    I §4.4.6; Vol II Examples E3 and E1). Same combinations and result as the
    level case: Strength I max (EV max, EH max), Strength I min (EV min, EH
    min), the critical min-vertical one (EV min, EH max) and Service I."""
    L = geom.reinforcement_length
    ld = backslope_loads(geom, gamma_backfill, phi_ext, gamma_ext, q)
    V12 = ld["V1"] + ld["V2"]                       # EV
    M12 = ld["MV1"] + ld["MV2"]
    tan_d = math.tan(math.radians(min(phi_backfill, phi_foundation)))
    ca = (2.0 / 3.0) * c_foundation
    # The traffic thrust's vertical component F2V resists sliding and
    # eccentricity only by choice: Example E1 counts it; the general text
    # counts resisting loads "only if those loads are permanent" (p. 4-22).
    # Bearing takes it either way (p. 4-26).
    ll_v = gamma_LL if traffic_vertical_resists else 0.0

    # Sliding (Eqs. 4-11, 4-13): F_TH drives and F_TV resists with the same
    # EH factor; the traffic thrust F2H drives with the LL factor (Example
    # E1); live load over the reinforced zone (Vs) excluded.
    def _sliding_cdr(ev, eh):
        demand = eh * ld["FTH"] + gamma_LL * ld["F2H"]
        resist = phi_sliding * (
            (ev * V12 + eh * ld["FTV"] + ll_v * ld["F2V"]) * tan_d
            + ca * L)
        return (resist / demand) if demand > 0 else 99.9

    # Resultant on the base about the toe (Eq. 4-16, Eqs. 4-19 to 4-21);
    # ll / ll_vert = the factors on F2H / F2V, vs = the factor on the live
    # load over the zone (0 = excluded).
    def _base(ev, eh, ll, ll_vert, vs):
        SV = (ev * V12 + eh * ld["FTV"] + ll_vert * ld["F2V"]
              + vs * ld["Vs"])
        MR = (ev * M12 + eh * ld["MFTV"] + ll_vert * ld["MF2V"]
              + vs * ld["MVs"])
        MO = eh * ld["MFTH"] + ll * ld["MF2H"]
        eL = L / 2.0 - (MR - MO) / SV
        Bp = L - 2.0 * max(eL, 0.0)             # L' = L when e < 0 (p. 4-28)
        sigma = (SV / Bp) if Bp > 0 else 1.0e6
        return sigma, eL, Bp

    cdr_slide_max = _sliding_cdr(gamma_EV_max, gamma_EH_max)
    cdr_slide_min = _sliding_cdr(gamma_EV_min, gamma_EH_min)
    cdr_slide_crit = _sliding_cdr(gamma_EV_min, gamma_EH_max)
    eL_max = _base(gamma_EV_max, gamma_EH_max, gamma_LL, ll_v, 0.0)[1]
    eL_crit = _base(gamma_EV_min, gamma_EH_max, gamma_LL, ll_v, 0.0)[1]
    sig_str, eL_bear, Bp_str = _base(gamma_EV_max, gamma_EH_max, gamma_LL,
                                     gamma_LL, gamma_LL)
    sig_crit, _, Bp_crit = _base(gamma_EV_min, gamma_EH_max, gamma_LL,
                                 gamma_LL, gamma_EV_min)
    sig_svc = _base(1.0, 1.0, 1.0, 1.0, 1.0)[0]

    unfactored = {
        "V1": round(ld["V1"], 3), "V2": round(ld["V2"], 3),
        "Vs": round(ld["Vs"], 3),
        "FT": round(ld["FT"], 3), "FT_H": round(ld["FTH"], 3),
        "FT_V": round(ld["FTV"], 3),
        "F2": round(ld["F2"], 3), "F2_H": round(ld["F2H"], 3),
        "F2_V": round(ld["F2V"], 3),
        "Ka_retained": round(ld["Kab"], 4),
    }
    out = _lrfd_result(
        cdr_slide_max, cdr_slide_min, cdr_slide_crit, eL_max, eL_crit,
        ecc_limit_ratio * L, sig_str, eL_bear, Bp_str, sig_crit, sig_svc,
        bearing_resistance_strength, bearing_resistance_service, unfactored,
        {"EV_max": gamma_EV_max, "EV_min": gamma_EV_min,
         "EH_max": gamma_EH_max, "EH_min": gamma_EH_min,
         "LL": gamma_LL, "phi_sliding": phi_sliding,
         "ecc_limit_ratio": ecc_limit_ratio},
        soil, Bp_crit)
    out["backslope"] = _backslope_summary(ld, traffic_vertical_resists)
    return out


def _lrfd_result(
    cdr_slide_max, cdr_slide_min, cdr_slide_crit, eL_max, eL_crit, e_limit,
    sig_str, eL_bear, Bp_str, sig_crit, sig_svc,
    bearing_resistance_strength, bearing_resistance_service,
    unfactored, load_factors, soil=None, Bp_crit=None,
) -> Dict[str, Any]:
    """The LRFD external-stability result from its CDRs, eccentricities and
    bearing stresses (shared by the level and sloping-backfill cases).
    ``soil`` = (gamma_f, phi_f, c_f, phi_b, L) works the strength bearing
    resistance out by GEC-11 Eq. 4-22 over each combination's L'
    (``Bp_str``, ``Bp_crit``; L' = L when eB < 0, p. 4-28) instead of
    taking it given (GM-179)."""
    cdr_slide_gov = min(cdr_slide_max, cdr_slide_min, cdr_slide_crit)
    eL_gov = max(eL_max, eL_crit)

    def _cdr(resistance, demand):
        if resistance is None or demand <= 0:
            return None
        return resistance / demand

    qR_str = qR_crit = bearing_resistance_strength
    worked_out = None
    if soil is not None:
        gamma_f, phi_f, c_f, phi_b, L = soil
        Lq_str = min(max(Bp_str, 0.0), L)
        Lq_crit = min(max(Bp_crit, 0.0), L)
        qn_str = nominal_bearing_resistance(Lq_str, gamma_f, phi_f, c_f)
        qn_crit = nominal_bearing_resistance(Lq_crit, gamma_f, phi_f, c_f)
        qR_str, qR_crit = phi_b * qn_str, phi_b * qn_crit
        f = bearing_factors(phi_f)
        worked_out = {
            "qn_strength_max_kPa": round(qn_str, 2),
            "qR_strength_max_kPa": round(qR_str, 2),
            "B_eff_critical_m": round(Lq_crit, 4),
            "qn_critical_kPa": round(qn_crit, 2),
            "qR_critical_kPa": round(qR_crit, 2),
            "Nc": round(f["Nc"], 2), "Ngamma": round(f["Ngamma"], 2),
            "phi_bearing": phi_b,
            "resistance_basis": NOMINAL_BEARING_BASIS,
        }

    cdr_bear_str = _cdr(qR_str, sig_str)
    cdr_bear_crit = _cdr(qR_crit, sig_crit)
    cdr_bear_svc = _cdr(bearing_resistance_service, sig_svc)
    bear_cdrs = [c for c in (cdr_bear_str, cdr_bear_crit) if c is not None]
    cdr_bear_gov = min(bear_cdrs) if bear_cdrs else None

    # Each check is run, or not run for want of its resistance (GM-100); a
    # check not run is never counted as a pass. Service I bearing counts
    # when its resistance is given (every load combination must pass).
    checks = {
        "sliding": cdr_slide_gov >= 1.0,
        "eccentricity": eL_gov <= e_limit,
        "bearing (Strength I)": (None if cdr_bear_gov is None
                                 else cdr_bear_gov >= 1.0),
        "bearing (Service I)": (None if cdr_bear_svc is None
                                else cdr_bear_svc >= 1.0),
    }
    checks_failed = [k for k, v in checks.items() if v is False]
    checks_not_run = [k for k, v in checks.items() if v is None]
    if checks_failed:
        passes = False
    elif checks_not_run:
        passes = None
    else:
        passes = True
    bearing_ran = [v for k, v in checks.items()
                   if k.startswith("bearing") and v is not None]
    bearing_passes = all(bearing_ran) if bearing_ran else None

    bearing = {
        "sigma_v_strength_max_kPa": round(sig_str, 2),
        "eL_strength_max_m": round(eL_bear, 4),
        "B_eff_strength_max_m": round(Bp_str, 4),
        "sigma_v_critical_kPa": round(sig_crit, 2),
        "sigma_v_service_kPa": round(sig_svc, 2),
        "CDR_strength": round(cdr_bear_str, 3) if cdr_bear_str else None,
        "CDR_critical": round(cdr_bear_crit, 3) if cdr_bear_crit else None,
        "CDR_service": round(cdr_bear_svc, 3) if cdr_bear_svc else None,
        "CDR_governing": round(cdr_bear_gov, 3) if cdr_bear_gov else None,
        "checked": bool(bearing_ran),
        "passes": bearing_passes,
    }
    if worked_out:
        bearing.update(worked_out)

    out = {
        "sliding": {
            "CDR_strength_max": round(cdr_slide_max, 3),
            "CDR_strength_min": round(cdr_slide_min, 3),
            "CDR_critical": round(cdr_slide_crit, 3),
            "CDR_governing": round(cdr_slide_gov, 3),
            "passes": cdr_slide_gov >= 1.0,
        },
        "eccentricity": {
            "eL_strength_max_m": round(eL_max, 4),
            "eL_critical_m": round(eL_crit, 4),
            "eL_governing_m": round(eL_gov, 4),
            "e_limit_m": round(e_limit, 4),
            "passes": eL_gov <= e_limit,
        },
        "bearing": bearing,
        "passes": passes,
        "checks_failed": checks_failed,
        "checks_not_run": checks_not_run,
        "unfactored_kN_m": unfactored,
        "load_factors": load_factors,
    }
    if worked_out:
        out["choices"] = bearing_resistance_choices(soil[3])
    return out


def analyze_mse_wall(
    geom: MSEWallGeometry,
    gamma_backfill: float,
    phi_backfill: float,
    reinforcement: Reinforcement,
    gamma_foundation: float = None,
    phi_foundation: float = None,
    c_foundation: float = 0.0,
    q_allowable: float = None,
    phi_retained: float = None,
    gamma_retained: float = None,
    lrfd_external: bool = False,
    bearing_resistance_strength: float = None,
    bearing_resistance_service: float = None,
    live_load: float = None,
    traffic_vertical_resists: bool = False,
    live_load_resists: bool = False,
    surcharge_in_bearing: bool = True,
    Cu_backfill: float = None,
    F_star_tested=None,
    alpha_tested: float = None,
) -> MSEWallResult:
    """Run complete MSE wall analysis.

    Parameters
    ----------
    geom : MSEWallGeometry
        Wall geometry.
    gamma_backfill : float
        Reinforced fill unit weight (kN/m³).
    phi_backfill : float
        Reinforced fill friction angle (degrees).
    reinforcement : Reinforcement or list of Reinforcement
        Reinforcement properties: one for every level, or one per level
        from the top down (GM-266; :func:`level_reinforcements`).
    gamma_foundation : float, optional
        Foundation soil unit weight (kN/m3). Used only on the LRFD path
        (``lrfd_external=True``) with no ``bearing_resistance_strength``:
        the strength bearing resistance is then worked out by GEC-11 Eq.
        4-22 (see ``check_external_stability_lrfd``); a given
        ``bearing_resistance_strength`` is used as given. The ASD bearing
        check compares the bearing pressure with ``q_allowable`` and counts
        no passive resistance, so it does not use it (GM-179).
    phi_foundation : float, optional
        Foundation friction angle. If None, uses phi_backfill.
    c_foundation : float, optional
        Foundation cohesion (kPa). Default 0.
    q_allowable : float, optional
        Allowable bearing pressure (kPa). Default None: the bearing check is
        NOT RUN and the result says so (``checks_not_run``), never a pass.
    phi_retained : float, optional
        Retained soil friction angle (degrees). If None, uses phi_backfill.
    gamma_retained : float, optional
        Retained soil unit weight (kN/m³). If None, uses gamma_backfill.
    lrfd_external : bool, optional
        If True, ALSO run the AASHTO/GEC-11 LRFD external-stability path
        (``check_external_stability_lrfd``) and attach its CDR set to
        ``result.external_lrfd``. Default False (ASD FOS path only — unchanged).
    bearing_resistance_strength, bearing_resistance_service : float, optional
        Factored strength / service bearing resistances (kPa) for the LRFD
        bearing CDRs (only used when lrfd_external=True).
    live_load : float, optional
        Live-load surcharge (kPa) for the LRFD path. Default: geom.surcharge.
    traffic_vertical_resists : bool, optional
        Broken backslope with traffic behind the crest only: count the
        traffic thrust's vertical component as sliding and eccentricity /
        overturning resistance, as GEC-11 Example E1 does. Default False
        (GEC-11 Vol I p. 4-22); see ``check_external_stability_lrfd``.
    live_load_resists : bool, optional
        ASD path: count the live-load surcharge over the reinforced zone as
        resisting weight in sliding and overturning. Default False
        (:data:`LIVE_LOAD_BASIS`, GM-164). The LRFD path leaves it out
        either way.
    surcharge_in_bearing : bool, optional
        ASD path: count that surcharge in the bearing pressure. Default
        True (:data:`BEARING_BASIS`).
    Cu_backfill : float, optional
        Uniformity coefficient D60/D10 of the reinforced backfill, for the
        ribbed steel strip F* (GEC-11 Vol I Eq. 3-4). Default None: 4, as
        GEC-11 p. 3-20 takes when it is unknown (F* 1.8 at the top). See
        :func:`check_internal_stability`.
    F_star_tested, alpha_tested : optional
        F* and alpha from pullout tests (GEC-11 Vol I Table 3-6 note (i),
        p. 3-16); see :func:`check_internal_stability`. Default None.

    Returns
    -------
    MSEWallResult
        With a sloping backfill (``geom.backfill_slope`` > 0, optionally a
        broken backslope) ``result.backslope`` carries the GEC-11 basis;
        see ``check_external_stability`` and ``check_internal_stability``.
        ``external_terms`` holds the ASD terms as the checks used them
        (:func:`_external_terms`) and ``choices`` how a live load over the
        reinforced zone was counted (:func:`live_load_choices`), the
        coverage ratio Rc (:func:`coverage_choice`), a built-in's Tal and
        a FOS_rupture below 1.0 (:func:`tensile_strength_choice`), and for
        ribbed steel strips the F* at the top of the wall, or a tested F*
        or alpha
        (:func:`pullout_factor_choices`). A steel strip or grid with no
        coverage ratio and no horizontal spacing is refused
        (:func:`coverage_ratio`, GM-256), and so is a geosynthetic of no
        stated kind given no tested alpha (:func:`pullout_alpha`).
    """
    # refuse before any work (GM-256; per level, GM-261 and GM-266)
    rs = level_reinforcements(reinforcement, geom.reinforcement_depths)
    coverage_ratios(reinforcement, geom.reinforcement_depths)
    _check_tested(F_star_tested, alpha_tested)
    _level_alphas(reinforcement, rs, alpha_tested)
    if (lrfd_external and gamma_foundation is not None
            and bearing_resistance_strength is None and phi_foundation is None):
        raise ValueError(NEEDS_PHI_FOUNDATION)
    if phi_foundation is None:
        phi_foundation = phi_backfill

    terms = _external_terms(
        geom, gamma_backfill, phi_backfill, phi_foundation, c_foundation,
        phi_retained=phi_retained, gamma_retained=gamma_retained,
        traffic_vertical_resists=traffic_vertical_resists,
        live_load_resists=live_load_resists,
        surcharge_in_bearing=surcharge_in_bearing,
    )
    external = _external_dict(terms, q_allowable)

    internal = check_internal_stability(
        geom, gamma_backfill, phi_backfill, reinforcement,
        Cu_backfill=Cu_backfill, F_star_tested=F_star_tested,
        alpha_tested=alpha_tested,
    )

    all_pass_internal = all(r["passes"] for r in internal)

    # The sloping-backfill basis, external (Kab, h, F_T, V2) and internal
    # (S_eq, sigma_2, H1), when the backfill slopes (GEC-11 §4.4.4, §4.4.7)
    backslope = external.get("backslope")
    if backslope is not None:
        bs = backslope_geometry(geom)
        backslope = dict(backslope,
                         S_eq_m=round(bs["S_eq"], 3),
                         sigma_2_kPa=round(gamma_backfill * bs["S_eq"], 2),
                         H1_m=round(bs["H1"], 3))

    external_lrfd = None
    if lrfd_external:
        external_lrfd = check_external_stability_lrfd(
            geom, gamma_backfill, phi_backfill, phi_foundation,
            phi_retained=phi_retained, gamma_retained=gamma_retained,
            live_load=live_load, c_foundation=c_foundation,
            bearing_resistance_strength=bearing_resistance_strength,
            bearing_resistance_service=bearing_resistance_service,
            traffic_vertical_resists=traffic_vertical_resists,
            gamma_foundation=(gamma_foundation
                              if bearing_resistance_strength is None
                              else None),
        )

    return MSEWallResult(
        FOS_sliding=external["FOS_sliding"],
        FOS_overturning=external["FOS_overturning"],
        FOS_bearing=external["FOS_bearing"],
        passes_external=external["passes"],
        passes_sliding=external["passes_sliding"],
        passes_overturning=external["passes_overturning"],
        passes_bearing=external["passes_bearing"],
        q_allowable=q_allowable,
        sigma_v=external["sigma_v_kPa"],
        internal_results=internal,
        all_pass_internal=all_pass_internal,
        wall_height=geom.wall_height,
        reinforcement_length=geom.reinforcement_length,
        external_lrfd=external_lrfd,
        backslope=backslope,
        external_terms=terms,
        choices=live_load_choices(geom, live_load_resists,
                                  surcharge_in_bearing)
        + [coverage_choice(reinforcement, internal)]
        + tensile_strength_choice(reinforcement, internal)
        + pullout_factor_choices(geom, gamma_backfill, phi_backfill,
                                 reinforcement, Cu_backfill, internal,
                                 F_star_tested, alpha_tested),
    )
