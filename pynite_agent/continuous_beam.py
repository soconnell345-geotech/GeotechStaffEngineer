"""Continuous-beam convenience wrapper on the frame engine.

Builds the node/member model for a multi-span beam (pin at the first
support, vertical rollers elsewhere by default; any support may be
'fixed'), applies UDLs and point loads, and reports per-support and
per-span envelopes.

Sign conventions at THIS interface (converted from PyNite local signs):
sagging moments positive, hogging negative; downward loads entered as
positive w/P magnitudes.
"""

import math

from pynite_agent.frame import build_model, extract_results
from pynite_agent.results import ContinuousBeamResult

_COMBO = "Combo 1"


def analyze_continuous_beam(
    span_lengths_m,
    E_kPa,
    I_m4,
    A_m2=0.01,
    udl_kN_m=0.0,
    span_udls_kN_m=None,
    point_loads=None,
    support_types=None,
) -> ContinuousBeamResult:
    """Analyze a continuous beam on simple (or fixed) supports.

    Parameters
    ----------
    span_lengths_m : list of float
        Span lengths, left to right (m).
    E_kPa : float
        Elastic modulus (kPa; e.g. steel 200e6, concrete ~25e6).
    I_m4 : float
        Major-axis second moment of area (m^4).
    A_m2 : float
        Cross-sectional area (m^2). Default 0.01 (axially rigid enough).
    udl_kN_m : float
        Uniform load on ALL spans, downward positive (kN/m).
    span_udls_kN_m : list of float, optional
        Per-span UDLs (downward positive); overrides ``udl_kN_m`` per span.
    point_loads : list of dict, optional
        ``{"span": 1-based index, "x": m from span's left support,
        "P": kN downward positive}``.
    support_types : list of str, optional
        One per support (n_spans + 1): 'pinned', 'roller_y', or 'fixed'.
        Default: pinned at the first support, rollers elsewhere.

    Returns
    -------
    ContinuousBeamResult
    """
    spans = [float(s) for s in span_lengths_m]
    if not spans or any(not math.isfinite(s) or s <= 0 for s in spans):
        raise ValueError("span_lengths_m must be positive numbers")
    n_sup = len(spans) + 1
    if support_types is None:
        support_types = ["pinned"] + ["roller_y"] * (len(spans))
    if len(support_types) != n_sup:
        raise ValueError(
            f"support_types needs {n_sup} entries, got {len(support_types)}")
    if span_udls_kN_m is not None and len(span_udls_kN_m) != len(spans):
        raise ValueError("span_udls_kN_m must have one entry per span")

    # nodes at supports
    xs = [0.0]
    for s in spans:
        xs.append(xs[-1] + s)
    nodes = [{"name": f"S{i}", "x": x, "y": 0.0} for i, x in enumerate(xs)]
    members = [{"name": f"SPAN{i + 1}", "i": f"S{i}", "j": f"S{i + 1}",
                "E": float(E_kPa), "A": float(A_m2), "Iz": float(I_m4)}
               for i in range(len(spans))]
    supports = [{"node": f"S{i}", "type": support_types[i]}
                for i in range(n_sup)]

    dist_loads = []
    for i in range(len(spans)):
        w = (span_udls_kN_m[i] if span_udls_kN_m is not None
             else udl_kN_m)
        w = float(w)
        if w:
            dist_loads.append({"member": f"SPAN{i + 1}", "direction": "FY",
                               "w": -abs(w) if w > 0 else -w})
    pt_loads = []
    for ld in (point_loads or []):
        span = int(ld["span"])
        if not 1 <= span <= len(spans):
            raise ValueError(f"point load span {span} out of range")
        x = float(ld["x"])
        if not 0.0 <= x <= spans[span - 1]:
            raise ValueError(
                f"point load x={x} outside span {span} (L={spans[span - 1]})")
        P = float(ld["P"])
        pt_loads.append({"member": f"SPAN{span}", "direction": "FY",
                         "value": -abs(P) if P > 0 else -P, "x": x})

    model, supported = build_model(
        nodes, members, supports,
        member_dist_loads=dist_loads, member_point_loads=pt_loads)
    model.analyze()
    frame = extract_results(model, supported)

    reactions = [frame.reactions[f"S{i}"]["FY_kN"] for i in range(n_sup)]

    # support (hogging) moments: internal member-end moment at each support.
    # PyNite local sign: sagging negative for -Y loads -> flip so sagging
    # is POSITIVE at this interface.
    support_moments = []
    for i in range(n_sup):
        if i == 0:
            m_end = model.members["SPAN1"].moment_array("Mz", n_points=2)[1][0]
        else:
            m_end = model.members[f"SPAN{i}"].moment_array(
                "Mz", n_points=2)[1][-1]
        support_moments.append(-float(m_end))

    span_sagging = []
    span_defl = []
    for i in range(len(spans)):
        mem = model.members[f"SPAN{i + 1}"]
        span_sagging.append(-float(mem.min_moment("Mz")))
        d = max(abs(float(mem.max_deflection("dy"))),
                abs(float(mem.min_deflection("dy"))))
        span_defl.append(d * 1000.0)

    max_sag = max(span_sagging) if span_sagging else 0.0
    max_hog = min(support_moments) if support_moments else 0.0

    return ContinuousBeamResult(
        n_spans=len(spans),
        span_lengths_m=spans,
        support_reactions_kN=reactions,
        support_moments_kNm=support_moments,
        span_max_sagging_kNm=span_sagging,
        span_max_deflection_mm=span_defl,
        max_sagging_kNm=max_sag,
        max_hogging_kNm=max_hog,
        max_deflection_mm=max(span_defl) if span_defl else 0.0,
        frame=frame,
    )
