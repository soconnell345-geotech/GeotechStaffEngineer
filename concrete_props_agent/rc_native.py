"""Reinforced-concrete section mechanics: transformed, cracked and ultimate.

Plain strain-compatibility analysis of a rectangular section with layers of
reinforcement, written against the published ACI 318-19 material model rather
than any third-party section library:

* **Transformed gross section** -- steel replaced by ``(n-1)*As`` of concrete
  (the bar displaces concrete, so only the excess is added), n = Es/Ec.
* **Cracked section** -- concrete in tension ignored; the neutral axis is the
  depth at which the transformed first moments balance; bars inside the
  compression zone are transformed at ``(n-1)`` and those in tension at ``n``.
* **Cracking moment** -- ``M_cr = fr * I_transformed / y_tension``.
* **Ultimate capacity** -- ACI equivalent rectangular stress block
  (``alpha*f'c`` over ``beta1*c``) with elastic-perfectly-plastic steel and
  ``eps_cu`` at the compression fibre; the neutral axis depth follows from
  axial equilibrium.

Sign and reference conventions match what the module has always reported:
depths ``y`` are measured from the bottom fibre, compression is positive, and
moments are taken about the **gross concrete centroid** (h/2), so a pure
tension force on eccentric steel correctly carries a moment.

Bar self-inertia (pi*d^4/64 per bar) is neglected in the transformed
properties, as in the standard hand calculation -- it is ~0.01% of a typical
beam's Ixx.
"""

import math

__all__ = [
    "Layer",
    "transformed_gross",
    "cracked_properties",
    "cracking_moment",
    "ultimate_capacity",
    "interaction_diagram",
    "squash_load",
]


class Layer:
    """One layer of reinforcement: total area at a height above the soffit."""

    __slots__ = ("area", "y")

    def __init__(self, area, y):
        self.area = float(area)
        self.y = float(y)

    def __repr__(self):                                  # pragma: no cover
        return f"Layer(area={self.area:.1f}, y={self.y:.1f})"


# ------------------------------------------------------- elastic properties

def transformed_gross(b, h, layers, n_modular):
    """Transformed uncracked section: ``(area, y_centroid, ixx_centroidal)``."""
    area = b * h
    moment = area * (h / 2.0)
    for lay in layers:
        extra = (n_modular - 1.0) * lay.area
        area += extra
        moment += extra * lay.y
    y_c = moment / area
    ixx = b * h ** 3 / 12.0 + b * h * (h / 2.0 - y_c) ** 2
    for lay in layers:
        ixx += (n_modular - 1.0) * lay.area * (lay.y - y_c) ** 2
    return area, y_c, ixx


def cracked_properties(b, h, layers, n_modular, sagging=True):
    """Cracked transformed section.

    Returns ``(ixx_cracked, y_na)`` with the neutral-axis height measured from
    the bottom fibre. Concrete carries no tension; the compression zone is the
    part of the section on the compression side of the neutral axis.
    """
    def net_first_moment(y_na):
        """First moment of the transformed cracked section about the NA."""
        if sagging:
            depth = h - y_na                     # compression block above NA
        else:
            depth = y_na                         # compression block below NA
        depth = max(depth, 0.0)
        total = b * depth * (depth / 2.0)        # concrete, always compressive
        for lay in layers:
            arm = (lay.y - y_na) if sagging else (y_na - lay.y)
            factor = (n_modular - 1.0) if arm > 0 else n_modular
            # arm > 0 means the bar sits on the compression side
            total += factor * lay.area * arm
        return total

    lo, hi = 1e-9, h - 1e-9
    f_lo, f_hi = net_first_moment(lo), net_first_moment(hi)
    if f_lo * f_hi > 0:
        raise ValueError("cracked neutral axis not bracketed - check geometry")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if net_first_moment(lo) * net_first_moment(mid) <= 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-10 * h:
            break
    y_na = 0.5 * (lo + hi)

    depth = (h - y_na) if sagging else y_na
    ixx = b * depth ** 3 / 3.0
    for lay in layers:
        arm = (lay.y - y_na) if sagging else (y_na - lay.y)
        factor = (n_modular - 1.0) if arm > 0 else n_modular
        ixx += factor * lay.area * arm ** 2
    return ixx, y_na


def cracking_moment(fr, ixx_transformed, y_centroid, h, sagging=True):
    """``M_cr = fr * I / y_t`` on the transformed uncracked section (N*mm)."""
    y_t = y_centroid if sagging else (h - y_centroid)
    if y_t <= 0:
        raise ValueError("degenerate tension-fibre distance")
    return fr * ixx_transformed / y_t


# ------------------------------------------------------ ultimate resistance

def _section_forces(c, b, h, layers, fc, fy, es, alpha, beta1, eps_cu,
                    sagging=True):
    """Axial force and moment for a neutral-axis depth ``c``.

    ``c`` is measured from the compression fibre. Returns ``(N, M)`` in
    N and N*mm, compression positive, moment about the gross centroid and
    positive in the sense that puts the named compression face in compression.
    """
    y_c = h / 2.0
    a = min(beta1 * c, h)
    force = alpha * fc * b * a
    # centroid of the stress block, as a height above the soffit
    y_block = (h - a / 2.0) if sagging else (a / 2.0)
    n_total = force
    m_total = force * (y_block - y_c)

    for lay in layers:
        depth = (h - lay.y) if sagging else lay.y      # from compression fibre
        strain = eps_cu * (c - depth) / c              # + = compression
        stress = max(-fy, min(fy, es * strain))
        if 0.0 < depth <= a:
            # the bar sits inside the stress block: it displaces concrete that
            # has already been counted, so remove that part
            stress -= alpha * fc
        f = lay.area * stress
        n_total += f
        m_total += f * (lay.y - y_c)

    return n_total, m_total if sagging else -m_total


def _solve_neutral_axis(target_n, b, h, layers, fc, fy, es, alpha, beta1,
                        eps_cu, sagging=True):
    """Neutral-axis depth giving the requested axial force (bisection)."""
    def residual(c):
        return _section_forces(c, b, h, layers, fc, fy, es, alpha, beta1,
                               eps_cu, sagging)[0] - target_n

    lo, hi = 1e-6, 1e-6
    # grow the upper bound until the section is stiff enough in compression
    hi = 10.0 * h
    if residual(lo) > 0:
        raise ValueError("section cannot reach the requested axial force")
    if residual(hi) < 0:
        raise ValueError("section cannot reach the requested axial force")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10 * h:
            break
    return 0.5 * (lo + hi)


def ultimate_capacity(b, h, layers, fc, fy, es=200e3, alpha=0.85, beta1=0.85,
                      eps_cu=0.003, sagging=True, axial_n=0.0):
    """Nominal moment capacity (N*mm) at a given axial force (N).

    No strength-reduction factors are applied.
    """
    if not layers:
        raise ValueError("at least one reinforcement layer is required")
    c = _solve_neutral_axis(axial_n, b, h, layers, fc, fy, es, alpha, beta1,
                            eps_cu, sagging)
    _, m = _section_forces(c, b, h, layers, fc, fy, es, alpha, beta1, eps_cu,
                           sagging)
    return abs(m), c


def squash_load(b, h, layers, fc, fy, alpha=0.85):
    """Concentric ultimate compression ``(N, M)`` about the gross centroid."""
    y_c = h / 2.0
    steel_area = sum(lay.area for lay in layers)
    # concrete over the gross section less the area the bars displace, plus
    # every bar at yield
    n = alpha * fc * (b * h - steel_area) + sum(lay.area * fy for lay in layers)
    # the full rectangle's resultant acts at the centroid and carries no
    # moment; what remains is each bar's yield force minus the concrete it
    # displaced, at that bar's eccentricity
    m = sum(lay.area * (fy - alpha * fc) * (lay.y - y_c) for lay in layers)
    return n, m


def pure_tension(layers, fy, h):
    """Pure tension ``(N, M)`` about the gross centroid, compression positive."""
    y_c = h / 2.0
    n = 0.0
    m = 0.0
    for lay in layers:
        f = -lay.area * fy
        n += f
        m += f * (lay.y - y_c)
    return n, m


def interaction_diagram(b, h, layers, fc, fy, es=200e3, alpha=0.85,
                        beta1=0.85, eps_cu=0.003, n_points=24):
    """N-M interaction points, from pure compression down to pure tension.

    Returns a list of ``(N, M)`` in N and N*mm with compression positive and
    moments about the gross centroid, ordered by decreasing axial force. The
    pure-compression, pure-bending and pure-tension control points are always
    included.
    """
    n_points = max(4, int(n_points))
    n_squash, m_squash = squash_load(b, h, layers, fc, fy, alpha)
    n_tens, m_tens = pure_tension(layers, fy, h)
    pts = [(n_squash, m_squash), (n_tens, m_tens)]

    # Sample uniformly in axial force rather than in neutral-axis depth: the
    # diagram is read at a given N, and a depth sweep crowds its points into
    # the bending end of the curve while leaving the compression branch bare.
    span = n_squash - n_tens
    for i in range(1, n_points + 1):
        target = n_squash - span * i / (n_points + 1.0)
        try:
            c = _solve_neutral_axis(target, b, h, layers, fc, fy, es, alpha,
                                    beta1, eps_cu, True)
        except ValueError:
            continue                     # axial force the section cannot reach
        pts.append(_section_forces(c, b, h, layers, fc, fy, es, alpha, beta1,
                                   eps_cu, True))

    # the balanced point (extreme tension steel just at yield) sets the peak
    # moment, and pure bending is the value everyone quotes -- pin both
    d_max = max((lay.y for lay in layers), default=None)
    if d_max is not None:
        d_eff = h - min(lay.y for lay in layers)
        c_bal = eps_cu / (eps_cu + fy / es) * d_eff
        pts.append(_section_forces(c_bal, b, h, layers, fc, fy, es, alpha,
                                   beta1, eps_cu, True))
    m_pure, _ = ultimate_capacity(b, h, layers, fc, fy, es, alpha, beta1,
                                  eps_cu, True, 0.0)
    pts.append((0.0, m_pure))

    pts.sort(key=lambda p: -p[0])
    # drop duplicates that the sweep and the control points can both produce
    out = []
    for n, m in pts:
        if out and abs(n - out[-1][0]) < 1e-6 * max(1.0, abs(n)) \
                and abs(m - out[-1][1]) < 1e-6 * max(1.0, abs(m)):
            continue
        out.append((n, m))
    return out


def modular_ratio(es, ec):
    """Es/Ec, guarded against a zero concrete modulus."""
    if ec <= 0:
        raise ValueError("concrete modulus must be positive")
    return es / ec


def bar_area(dia_mm):
    """Area of one round bar."""
    return math.pi * float(dia_mm) ** 2 / 4.0
