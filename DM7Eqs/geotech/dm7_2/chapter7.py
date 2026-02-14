"""
UFC 3-220-20, Chapter 7: Probability and Reliability in Geotechnical Engineering

Equations 7-1 through 7-16 covering basic statistics, probability distributions,
reliability index, first-order second-moment (FOSM) method, point estimate
method, Monte Carlo simulation concepts, hazard analysis, and load and
resistance factor design (LRFD) concepts.

Reference:
    UFC 3-220-20, Foundations and Earth Structures,
    16 January 2025, Chapter 7, pp. 546-595.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Table 7-1 -- Common Statistics (supporting functions for the equations)
# ---------------------------------------------------------------------------


def sample_mean(x: Sequence[float]) -> float:
    """Arithmetic average of *n* items in a sample (Table 7-1).

    .. math::
        \\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i

    Parameters
    ----------
    x : Sequence[float]
        Sample observations.  Must contain at least one value.

    Returns
    -------
    float
        Sample mean.

    Raises
    ------
    ValueError
        If *x* is empty.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Table 7-1, p. 548.
    """
    n = len(x)
    if n == 0:
        raise ValueError("x must contain at least one value.")
    return sum(x) / n


def sample_variance(x: Sequence[float]) -> float:
    """Unbiased sample variance (Table 7-1).

    Uses (n - 1) in the denominator to produce an unbiased estimate.

    .. math::
        Var_X = s_X^2 = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})^2

    Parameters
    ----------
    x : Sequence[float]
        Sample observations.  Must contain at least two values.

    Returns
    -------
    float
        Unbiased sample variance.

    Raises
    ------
    ValueError
        If *x* contains fewer than two values.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Table 7-1, p. 548.
    """
    n = len(x)
    if n < 2:
        raise ValueError("x must contain at least two values.")
    x_bar = sample_mean(x)
    return sum((xi - x_bar) ** 2 for xi in x) / (n - 1)


def sample_standard_deviation(x: Sequence[float]) -> float:
    """Sample standard deviation (Table 7-1).

    Square root of the unbiased sample variance.

    .. math::
        s_X = \\sqrt{Var_X}

    Parameters
    ----------
    x : Sequence[float]
        Sample observations.  Must contain at least two values.

    Returns
    -------
    float
        Sample standard deviation.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Table 7-1, p. 548.
    """
    return math.sqrt(sample_variance(x))


def coefficient_of_variation(x: Sequence[float]) -> float:
    """Coefficient of variation (COV) from sample data (Table 7-1).

    Standard deviation divided by the mean.  A relative measure of
    dispersion commonly used in geotechnical engineering.

    .. math::
        COV_X = \\frac{s_X}{\\bar{x}}

    Parameters
    ----------
    x : Sequence[float]
        Sample observations.  Must contain at least two values.

    Returns
    -------
    float
        Coefficient of variation (dimensionless).

    Raises
    ------
    ValueError
        If the sample mean is zero (COV undefined).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Table 7-1, p. 548.
    """
    x_bar = sample_mean(x)
    if x_bar == 0.0:
        raise ValueError("Sample mean is zero; COV is undefined.")
    return sample_standard_deviation(x) / x_bar


def coefficient_of_variation_from_params(std_dev: float, mean: float) -> float:
    """Coefficient of variation from known mean and standard deviation
    (Table 7-1).

    .. math::
        COV_X = \\frac{\\sigma_X}{\\mu_X}

    Parameters
    ----------
    std_dev : float
        Standard deviation of the parameter.  Must be non-negative.
    mean : float
        Mean value of the parameter.  Must be non-zero.

    Returns
    -------
    float
        Coefficient of variation (dimensionless).

    Raises
    ------
    ValueError
        If *std_dev* is negative or *mean* is zero.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Table 7-1, p. 548.
    """
    if std_dev < 0.0:
        raise ValueError("std_dev must be non-negative.")
    if mean == 0.0:
        raise ValueError("mean must be non-zero; COV is undefined.")
    return std_dev / mean


# ---------------------------------------------------------------------------
# Equation 7-1 -- Cumulative Mass Function (CMF) for discrete RV
# ---------------------------------------------------------------------------


def cumulative_mass_function(pmf_values: Sequence[float],
                             index: int) -> float:
    """Cumulative mass function for a discrete random variable
    (Equation 7-1).

    The CMF at index *i* is the probability that the random variable X
    takes on a value less than or equal to x_i.  It is computed as the
    sum of the probability mass function (PMF) values from the first
    value through index *i*.

    .. math::
        F_X(x_i) = P(X \\leq x_i) = \\sum_{j=1}^{i} P_X(x_j)

    Parameters
    ----------
    pmf_values : Sequence[float]
        Probability mass function values for each discrete outcome,
        ordered from smallest to largest x.  Each value must be in
        [0, 1] and the total must sum to 1.0 (within tolerance).
    index : int
        Zero-based index *i* of the outcome for which the CMF is
        desired.  Must be in the range [0, len(pmf_values) - 1].

    Returns
    -------
    float
        Cumulative probability F_X(x_i).

    Raises
    ------
    ValueError
        If *pmf_values* is empty, any value is outside [0, 1], the
        total does not sum to approximately 1.0, or *index* is out
        of range.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-1, p. 553.
    """
    n = len(pmf_values)
    if n == 0:
        raise ValueError("pmf_values must not be empty.")
    for p in pmf_values:
        if p < 0.0 or p > 1.0:
            raise ValueError(
                "Each PMF value must be in the range [0, 1]."
            )
    if abs(sum(pmf_values) - 1.0) > 1e-6:
        raise ValueError(
            "PMF values must sum to 1.0 (within tolerance of 1e-6)."
        )
    if index < 0 or index >= n:
        raise ValueError(
            f"index must be in the range [0, {n - 1}]."
        )
    return sum(pmf_values[: index + 1])


# ---------------------------------------------------------------------------
# Equation 7-2 -- Cumulative Density Function (CDF) for continuous RV
# ---------------------------------------------------------------------------


def cumulative_density_function(
    pdf_func: Callable[[float], float],
    x0: float,
    lower_bound: float = -10.0,
    n_steps: int = 1000,
) -> float:
    """Cumulative density function for a continuous random variable
    (Equation 7-2).

    Numerically integrates the probability density function (PDF) from
    the lower bound to *x0* using the trapezoidal rule.

    .. math::
        F_X(x_0) = P(X \\leq x_0) = \\int_{-\\infty}^{x_0} f_X(x)\\, dx

    Parameters
    ----------
    pdf_func : Callable[[float], float]
        Probability density function f_X(x).  Must return non-negative
        values and integrate to 1 over the full domain.
    x0 : float
        Upper limit of integration; the value at which the CDF is
        evaluated.
    lower_bound : float, optional
        Practical lower bound for integration (approximation to
        negative infinity).  Default is -10.0.
    n_steps : int, optional
        Number of integration steps for the trapezoidal rule.
        Default is 1000.

    Returns
    -------
    float
        Cumulative probability F_X(x0).

    Raises
    ------
    ValueError
        If *n_steps* is less than 1 or *x0* is less than
        *lower_bound*.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-2, p. 553.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if x0 < lower_bound:
        return 0.0
    h = (x0 - lower_bound) / n_steps
    total = 0.5 * (pdf_func(lower_bound) + pdf_func(x0))
    for i in range(1, n_steps):
        total += pdf_func(lower_bound + i * h)
    return total * h


# ---------------------------------------------------------------------------
# Equation 7-3 -- PDF as derivative of CDF
# ---------------------------------------------------------------------------


def pdf_from_cdf(
    cdf_func: Callable[[float], float],
    x: float,
    dx: float = 1e-6,
) -> float:
    """Probability density function as the derivative of the CDF
    (Equation 7-3).

    Estimates the PDF at a point *x* by computing the numerical
    derivative of the CDF using a central difference approximation.

    .. math::
        f_X(x) = \\frac{d}{dx} F_X(x)

    Parameters
    ----------
    cdf_func : Callable[[float], float]
        Cumulative distribution function F_X(x).
    x : float
        Value at which to evaluate the PDF.
    dx : float, optional
        Step size for the central difference.  Default is 1e-6.

    Returns
    -------
    float
        Estimated probability density f_X(x).

    Raises
    ------
    ValueError
        If *dx* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-3, p. 555.
    """
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    return (cdf_func(x + dx) - cdf_func(x - dx)) / (2.0 * dx)


# ---------------------------------------------------------------------------
# Equation 7-4 -- Probability over an interval for continuous RV
# ---------------------------------------------------------------------------


def probability_over_interval(
    pdf_func: Callable[[float], float],
    x1: float,
    x2: float,
    n_steps: int = 1000,
) -> float:
    """Probability over an interval for a continuous random variable
    (Equation 7-4).

    Numerically integrates the PDF from *x1* to *x2* using the
    trapezoidal rule.

    .. math::
        P(x_1 \\leq X \\leq x_2) = \\int_{x_1}^{x_2} f_X(x)\\, dx

    Parameters
    ----------
    pdf_func : Callable[[float], float]
        Probability density function f_X(x).
    x1 : float
        Lower bound of the interval.
    x2 : float
        Upper bound of the interval.  Must be >= *x1*.
    n_steps : int, optional
        Number of integration steps.  Default is 1000.

    Returns
    -------
    float
        Probability that X falls within [x1, x2].

    Raises
    ------
    ValueError
        If *x2* < *x1* or *n_steps* < 1.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-4, p. 555.
    """
    if x2 < x1:
        raise ValueError("x2 must be greater than or equal to x1.")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1.")
    if x1 == x2:
        return 0.0
    h = (x2 - x1) / n_steps
    total = 0.5 * (pdf_func(x1) + pdf_func(x2))
    for i in range(1, n_steps):
        total += pdf_func(x1 + i * h)
    return total * h


# ---------------------------------------------------------------------------
# Equation 7-5 -- Combined COV from multiple uncertainty sources
# ---------------------------------------------------------------------------


def combined_cov(cov_w: float, cov_e: float, cov_t: float) -> float:
    """Combined coefficient of variation from multiple uncertainty
    sources (Equation 7-5).

    Estimates the total COV of a geotechnical parameter by combining
    inherent variability, measurement error, and transformation
    uncertainty using a first-order approximation.

    .. math::
        COV_{\\theta} = \\sqrt{COV_w^2 + COV_e^2 + COV_t^2}

    Parameters
    ----------
    cov_w : float
        Coefficient of variation of inherent (natural) variability
        (dimensionless).  Must be non-negative.
    cov_e : float
        Coefficient of variation of measurement error
        (dimensionless).  Must be non-negative.
    cov_t : float
        Coefficient of variation of transformation uncertainty
        (dimensionless).  Must be non-negative.

    Returns
    -------
    float
        Combined coefficient of variation of the geotechnical
        parameter (dimensionless).

    Raises
    ------
    ValueError
        If any COV component is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-5, p. 562.
    """
    if cov_w < 0.0:
        raise ValueError("cov_w must be non-negative.")
    if cov_e < 0.0:
        raise ValueError("cov_e must be non-negative.")
    if cov_t < 0.0:
        raise ValueError("cov_t must be non-negative.")
    return math.sqrt(cov_w ** 2 + cov_e ** 2 + cov_t ** 2)


# ---------------------------------------------------------------------------
# Equation 7-6 -- Standard deviation from HCV and LCV
# ---------------------------------------------------------------------------


def std_dev_from_range(hcv: float, lcv: float, n: float = 6.0) -> float:
    """Estimate standard deviation from the highest and lowest
    conceivable values (Equation 7-6).

    A judgment-based method to estimate the standard deviation of a
    geotechnical parameter when limited data are available.  The HCV
    and LCV are assumed to be separated by *n* standard deviations.

    .. math::
        \\sigma = \\frac{HCV - LCV}{n}

    Parameters
    ----------
    hcv : float
        Highest conceivable value of the parameter (any consistent
        unit).
    lcv : float
        Lowest conceivable value of the parameter (same unit as
        *hcv*).  Must be less than *hcv*.
    n : float, optional
        Number of standard deviations separating HCV and LCV.
        Default is 6.0 as recommended by the UFC.  Values of 3 or 4
        may be used for a conservatively high estimate of sigma.

    Returns
    -------
    float
        Estimated standard deviation (same unit as *hcv* and *lcv*).

    Raises
    ------
    ValueError
        If *hcv* <= *lcv* or *n* <= 0.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-6, p. 568.
    """
    if hcv <= lcv:
        raise ValueError("hcv must be greater than lcv.")
    if n <= 0.0:
        raise ValueError("n must be positive.")
    return (hcv - lcv) / n


# ---------------------------------------------------------------------------
# Equation 7-7 -- Reliability Index
# ---------------------------------------------------------------------------


def reliability_index(mu_g: float, sigma_g: float) -> float:
    """Reliability index from the mean and standard deviation of the
    limit state function (Equation 7-7).

    The reliability index (beta) is the number of standard deviations
    that separate the mean design condition from a state of failure
    (g(X) = 0).

    .. math::
        \\beta = \\frac{\\mu_{g(X)}}{\\sigma_{g(X)}}

    Parameters
    ----------
    mu_g : float
        Mean value of the limit state function g(X).
    sigma_g : float
        Standard deviation of the limit state function g(X).
        Must be positive.

    Returns
    -------
    float
        Reliability index beta (dimensionless).

    Raises
    ------
    ValueError
        If *sigma_g* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-7, p. 571.
    """
    if sigma_g <= 0.0:
        raise ValueError("sigma_g must be positive.")
    return mu_g / sigma_g


# ---------------------------------------------------------------------------
# Equation 7-8 -- FOSM: Mean of the limit state function
# ---------------------------------------------------------------------------


def fosm_mean(
    g_func: Callable[..., float],
    means: Sequence[float],
) -> float:
    """Mean of the limit state function using the FOSM method
    (Equation 7-8).

    The mean of g(X) is simply the function evaluated at the mean
    values of all random variables.

    .. math::
        \\mu_{g(X)} = g(\\mu_{X_1}, \\mu_{X_2}, \\ldots, \\mu_{X_n})

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).  Must accept *n*
        positional float arguments and return a float.
    means : Sequence[float]
        Mean values of each random variable
        [mu_X1, mu_X2, ..., mu_Xn].

    Returns
    -------
    float
        Mean value of the limit state function.

    Raises
    ------
    ValueError
        If *means* is empty.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-8, p. 571.
    """
    if len(means) == 0:
        raise ValueError("means must contain at least one value.")
    return g_func(*means)


# ---------------------------------------------------------------------------
# Equation 7-9 -- FOSM: Variance (analytical partial derivatives)
# ---------------------------------------------------------------------------


def fosm_variance_analytical(
    partial_derivatives: Sequence[float],
    std_devs: Sequence[float],
) -> float:
    """Variance of the limit state function using the FOSM method
    with analytically determined partial derivatives (Equation 7-9).

    The variance is approximated by keeping only first-order terms
    in the Taylor series expansion about the mean.

    .. math::
        \\sigma_{g(X)}^2 = \\sum_{i=1}^{n}
            \\left( \\frac{\\partial g}{\\partial X_i} \\right)^2
            \\sigma_{X_i}^2

    where all partial derivatives are evaluated at the mean values
    of the random variables.

    Parameters
    ----------
    partial_derivatives : Sequence[float]
        Values of the partial derivative of g with respect to each
        random variable, evaluated at the mean values.  Length *n*.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Length *n*.
        Each value must be non-negative.

    Returns
    -------
    float
        Estimated variance of the limit state function.

    Raises
    ------
    ValueError
        If lengths do not match, sequences are empty, or any
        standard deviation is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-9, p. 572.
    """
    n = len(partial_derivatives)
    if n == 0:
        raise ValueError("partial_derivatives must not be empty.")
    if len(std_devs) != n:
        raise ValueError(
            "partial_derivatives and std_devs must have the same length."
        )
    for s in std_devs:
        if s < 0.0:
            raise ValueError("Each standard deviation must be non-negative.")
    return sum(
        (partial_derivatives[i] ** 2) * (std_devs[i] ** 2)
        for i in range(n)
    )


# ---------------------------------------------------------------------------
# Equation 7-10 -- Central difference for approximate derivative
# ---------------------------------------------------------------------------


def fosm_central_difference(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
    var_index: int,
) -> float:
    """Central difference approximation for one random variable
    (Equation 7-10).

    Evaluates g(X) at values of x_i that are one standard deviation
    above and below the mean, with all other variables at their mean
    values.  The result is the central difference for variable *i*.

    .. math::
        \\Delta g_i = g(\\mu_{X_1}, \\ldots, \\mu_{X_i} + \\sigma_{X_i},
            \\ldots, \\mu_{X_n})
            - g(\\mu_{X_1}, \\ldots, \\mu_{X_i} - \\sigma_{X_i},
            \\ldots, \\mu_{X_n})

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).
    means : Sequence[float]
        Mean values of each random variable.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Each must be
        non-negative.
    var_index : int
        Zero-based index of the random variable for which the
        central difference is calculated.

    Returns
    -------
    float
        Central difference delta_g_i for the specified variable.

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, index is out of
        range, or any standard deviation is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-10, p. 573.
    """
    n = len(means)
    if n == 0:
        raise ValueError("means must not be empty.")
    if len(std_devs) != n:
        raise ValueError("means and std_devs must have the same length.")
    if var_index < 0 or var_index >= n:
        raise ValueError(f"var_index must be in range [0, {n - 1}].")
    for s in std_devs:
        if s < 0.0:
            raise ValueError("Each standard deviation must be non-negative.")

    args_plus = list(means)
    args_plus[var_index] = means[var_index] + std_devs[var_index]

    args_minus = list(means)
    args_minus[var_index] = means[var_index] - std_devs[var_index]

    return g_func(*args_plus) - g_func(*args_minus)


# ---------------------------------------------------------------------------
# Equation 7-11 -- FOSM variance using central differences
# ---------------------------------------------------------------------------


def fosm_variance_numerical(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
) -> float:
    """Variance of the limit state function using central differences
    (Equation 7-11).

    Uses the central difference method (Equation 7-10) to approximate
    the partial derivatives, then computes the variance as the sum of
    the squared central differences divided by four.

    .. math::
        \\sigma_{g(X)}^2 = \\sum_{i=1}^{n}
            \\left( \\frac{\\Delta g_i}{2} \\right)^2

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).
    means : Sequence[float]
        Mean values of each random variable.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Each must be
        non-negative.

    Returns
    -------
    float
        Estimated variance of the limit state function.

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, or any standard
        deviation is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-11, p. 573.
    """
    n = len(means)
    if n == 0:
        raise ValueError("means must not be empty.")
    if len(std_devs) != n:
        raise ValueError("means and std_devs must have the same length.")
    variance = 0.0
    for i in range(n):
        dg = fosm_central_difference(g_func, means, std_devs, i)
        variance += (dg / 2.0) ** 2
    return variance


# ---------------------------------------------------------------------------
# Equation 7-12 -- Lognormal reliability index
# ---------------------------------------------------------------------------


def reliability_index_lognormal(mu_g: float, sigma_g: float) -> float:
    """Lognormal reliability index (Equation 7-12).

    When the limit state function is defined using the safety factor
    format, a lognormal distribution of g(X) is often appropriate.
    The reliability index is computed in log-space.

    .. math::
        \\beta_{LN} = \\frac{\\ln \\mu_{g(X)}
            - 0.5 \\ln(1 + COV_g^2)}
            {\\sqrt{\\ln(1 + COV_g^2)}}

    where COV_g = sigma_{g(X)} / mu_{g(X)}.

    Parameters
    ----------
    mu_g : float
        Mean value of the limit state function g(X).  Must be
        positive for a lognormal distribution.
    sigma_g : float
        Standard deviation of the limit state function g(X).
        Must be positive.

    Returns
    -------
    float
        Lognormal reliability index (dimensionless).

    Raises
    ------
    ValueError
        If *mu_g* is not positive or *sigma_g* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-12, p. 573.
    """
    if mu_g <= 0.0:
        raise ValueError("mu_g must be positive for lognormal distribution.")
    if sigma_g <= 0.0:
        raise ValueError("sigma_g must be positive.")
    cov_g = sigma_g / mu_g
    ln_term = math.log(1.0 + cov_g ** 2)
    return (math.log(mu_g) - 0.5 * ln_term) / math.sqrt(ln_term)


# ---------------------------------------------------------------------------
# Equation 7-13 -- Point Estimate Method: Mean of g(X)
# ---------------------------------------------------------------------------


def point_estimate_mean(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
) -> float:
    """Mean of the limit state function using the point estimate
    method for uncorrelated symmetric random variables
    (Equation 7-13).

    Evaluates g(X) for all 2^n combinations of each random variable
    at one standard deviation above or below the mean, then averages
    the results (equal weights of 2^{-n} for uncorrelated RV).

    .. math::
        \\mu_{g(X)} \\approx \\sum_{i=1}^{2^n} P_i \\cdot g(X)_i

    where P_i = 2^{-n} for uncorrelated random variables.

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).
    means : Sequence[float]
        Mean values of each random variable.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Each must be
        non-negative.

    Returns
    -------
    float
        Estimated mean of the limit state function.

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, or any standard
        deviation is negative.

    Notes
    -----
    The number of function evaluations is 2^n, which grows
    exponentially with the number of random variables.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-13, p. 574.
    """
    n = len(means)
    if n == 0:
        raise ValueError("means must not be empty.")
    if len(std_devs) != n:
        raise ValueError("means and std_devs must have the same length.")
    for s in std_devs:
        if s < 0.0:
            raise ValueError("Each standard deviation must be non-negative.")

    num_cases = 2 ** n
    weight = 1.0 / num_cases
    total = 0.0

    for case in range(num_cases):
        args = []
        for j in range(n):
            if (case >> j) & 1:
                args.append(means[j] + std_devs[j])
            else:
                args.append(means[j] - std_devs[j])
        total += weight * g_func(*args)

    return total


# ---------------------------------------------------------------------------
# Equation 7-14 -- Point Estimate Method: Variance of g(X)
# ---------------------------------------------------------------------------


def point_estimate_variance(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
) -> float:
    """Variance of the limit state function using the point estimate
    method for uncorrelated symmetric random variables
    (Equation 7-14).

    .. math::
        \\sigma_{g(X)}^2 \\approx \\sum_{i=1}^{2^n} P_i \\cdot g(X)_i^2
            - \\left( \\sum_{i=1}^{2^n} P_i \\cdot g(X)_i \\right)^2

    where P_i = 2^{-n} for uncorrelated random variables.

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).
    means : Sequence[float]
        Mean values of each random variable.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Each must be
        non-negative.

    Returns
    -------
    float
        Estimated variance of the limit state function.

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, or any standard
        deviation is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-14, p. 574.
    """
    n = len(means)
    if n == 0:
        raise ValueError("means must not be empty.")
    if len(std_devs) != n:
        raise ValueError("means and std_devs must have the same length.")
    for s in std_devs:
        if s < 0.0:
            raise ValueError("Each standard deviation must be non-negative.")

    num_cases = 2 ** n
    weight = 1.0 / num_cases
    sum_g = 0.0
    sum_g2 = 0.0

    for case in range(num_cases):
        args = []
        for j in range(n):
            if (case >> j) & 1:
                args.append(means[j] + std_devs[j])
            else:
                args.append(means[j] - std_devs[j])
        g_val = g_func(*args)
        sum_g += weight * g_val
        sum_g2 += weight * g_val ** 2

    return sum_g2 - sum_g ** 2


# ---------------------------------------------------------------------------
# Equation 7-15 -- Rate of exceedance from return period
# ---------------------------------------------------------------------------


def rate_of_exceedance(return_period: float) -> float:
    """Rate of exceedance from the mean recurrence interval
    (Equation 7-15).

    Calculates the annual rate of exceedance (lambda) for a hazard
    event given its return period.

    .. math::
        \\lambda = \\frac{1}{R}

    Parameters
    ----------
    return_period : float
        Mean recurrence interval *R* (years).  Must be positive.

    Returns
    -------
    float
        Annual rate of exceedance (1/years).

    Raises
    ------
    ValueError
        If *return_period* is not positive.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-15, p. 586.
    """
    if return_period <= 0.0:
        raise ValueError("return_period must be positive.")
    return 1.0 / return_period


# ---------------------------------------------------------------------------
# Equation 7-16 -- Rate of exceedance from probability and exposure
# ---------------------------------------------------------------------------


def rate_of_exceedance_from_probability(
    prob_exceedance: float,
    exposure_period: float,
) -> float:
    """Rate of exceedance from probability of exceedance and exposure
    period (Equation 7-16).

    Determines the annual rate of exceedance that corresponds to a
    threshold probability of exceedance over a given exposure period.

    .. math::
        \\lambda(y^*) = \\frac{-\\ln(1 - P(Y > y^*))}{T_R}

    Parameters
    ----------
    prob_exceedance : float
        Threshold probability of exceedance P(Y > y*) during the
        exposure period.  Must be in the range (0, 1).
    exposure_period : float
        Exposure period T_R (years).  Must be positive.

    Returns
    -------
    float
        Annual rate of exceedance lambda(y*) (1/years).

    Raises
    ------
    ValueError
        If *prob_exceedance* is not in (0, 1) or *exposure_period*
        is not positive.

    Examples
    --------
    10% probability of exceedance in 50 years (common seismic
    design criterion):

    >>> round(rate_of_exceedance_from_probability(0.10, 50.0), 6)
    0.002107

    This corresponds to a return period of about 475 years.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equation 7-16, p. 587.
    """
    if prob_exceedance <= 0.0 or prob_exceedance >= 1.0:
        raise ValueError("prob_exceedance must be in the range (0, 1).")
    if exposure_period <= 0.0:
        raise ValueError("exposure_period must be positive.")
    return -math.log(1.0 - prob_exceedance) / exposure_period


# ---------------------------------------------------------------------------
# Convenience / composite functions built from the equations above
# ---------------------------------------------------------------------------


def probability_of_failure_normal(beta: float) -> float:
    """Probability of unsatisfactory performance assuming a normal
    distribution of g(X).

    Uses the standard normal CDF (Phi) to convert the reliability
    index to a probability of failure.

    .. math::
        P_u = \\Phi(-\\beta)

    This function implements the standard normal CDF using the
    complementary error function (erfc) available in the math module.

    Parameters
    ----------
    beta : float
        Reliability index (dimensionless).

    Returns
    -------
    float
        Probability of unsatisfactory performance (failure).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Section 7-4.2, p. 571.
    """
    return 0.5 * math.erfc(beta / math.sqrt(2.0))


def first_order_second_moment(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Complete FOSM reliability analysis using the approximate
    derivative (central difference) method (Equations 7-8, 7-10,
    7-11, 7-7).

    Computes the mean and variance of the limit state function using
    the FOSM approximate derivative approach, then calculates the
    reliability index and the probability of unsatisfactory
    performance (assuming a normal distribution of g(X)).

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).  Must accept *n*
        positional float arguments and return a float.
    means : Sequence[float]
        Mean values of each random variable
        [mu_X1, mu_X2, ..., mu_Xn].
    std_devs : Sequence[float]
        Standard deviations of each random variable
        [sigma_X1, sigma_X2, ..., sigma_Xn].  Each must be
        non-negative.

    Returns
    -------
    tuple of (float, float, float, float)
        - mu_g : Mean of the limit state function.
        - sigma_g : Standard deviation of the limit state function.
        - beta : Reliability index.
        - p_u : Probability of unsatisfactory performance
          (assuming normal distribution of g(X)).

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, or any standard
        deviation is negative.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equations 7-8, 7-10,
    7-11, 7-7, pp. 571-573.
    """
    mu_g = fosm_mean(g_func, means)
    var_g = fosm_variance_numerical(g_func, means, std_devs)
    sigma_g = math.sqrt(var_g)
    beta = reliability_index(mu_g, sigma_g)
    p_u = probability_of_failure_normal(beta)
    return mu_g, sigma_g, beta, p_u


def point_estimate_method(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Complete point estimate reliability analysis for uncorrelated
    symmetric random variables (Equations 7-13, 7-14, 7-7).

    Computes the mean and variance of the limit state function using
    the Rosenblueth (1975) point estimate method, then calculates
    the reliability index and the probability of unsatisfactory
    performance (assuming a normal distribution of g(X)).

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).
    means : Sequence[float]
        Mean values of each random variable.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Each must be
        non-negative.

    Returns
    -------
    tuple of (float, float, float, float)
        - mu_g : Mean of the limit state function.
        - sigma_g : Standard deviation of the limit state function.
        - beta : Reliability index.
        - p_u : Probability of unsatisfactory performance
          (assuming normal distribution of g(X)).

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, or any standard
        deviation is negative.

    Notes
    -----
    The number of function evaluations is 2^n.  This method is best
    suited for problems with a small number of random variables.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Equations 7-13, 7-14, 7-7,
    pp. 574.
    """
    mu_g = point_estimate_mean(g_func, means, std_devs)
    var_g = point_estimate_variance(g_func, means, std_devs)
    sigma_g = math.sqrt(var_g)
    beta = reliability_index(mu_g, sigma_g)
    p_u = probability_of_failure_normal(beta)
    return mu_g, sigma_g, beta, p_u


def monte_carlo_simulation(
    g_func: Callable[..., float],
    means: Sequence[float],
    std_devs: Sequence[float],
    n_trials: int = 10000,
    seed: Optional[int] = None,
    distribution: str = "normal",
) -> Tuple[float, float, float, float]:
    """Monte Carlo simulation for reliability analysis
    (Section 7-4.2.4).

    Generates random samples from the specified distributions for
    each random variable, evaluates the limit state function for
    each trial, and directly estimates the probability of
    unsatisfactory performance as the fraction of trials with
    g(X) < 0.

    Parameters
    ----------
    g_func : Callable[..., float]
        Limit state function g(X1, X2, ..., Xn).
    means : Sequence[float]
        Mean values of each random variable.
    std_devs : Sequence[float]
        Standard deviations of each random variable.  Each must be
        non-negative.
    n_trials : int, optional
        Number of Monte Carlo trials.  Default is 10000.
    seed : int or None, optional
        Seed for the random number generator.  Default is None
        (non-reproducible).
    distribution : str, optional
        Type of distribution for each random variable.  Either
        ``"normal"`` or ``"lognormal"``.  Default is ``"normal"``.

    Returns
    -------
    tuple of (float, float, float, float)
        - mu_g : Mean of g(X) from the simulation.
        - sigma_g : Standard deviation of g(X) from the simulation.
        - beta : Reliability index (mu_g / sigma_g).
        - p_u : Probability of unsatisfactory performance (fraction
          of trials with g(X) < 0).

    Raises
    ------
    ValueError
        If input lengths do not match, are empty, any standard
        deviation is negative, n_trials < 1, or distribution type
        is unsupported.

    Notes
    -----
    Uses Python's built-in ``random`` module for normal sampling.
    For lognormal distribution, the parameters of the underlying
    normal distribution are derived from the specified mean and
    standard deviation.  The random number generator quality is
    limited to Python's Mersenne Twister implementation.

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 7, Section 7-4.2.4,
    pp. 578-579.
    """
    import random as _random

    n = len(means)
    if n == 0:
        raise ValueError("means must not be empty.")
    if len(std_devs) != n:
        raise ValueError("means and std_devs must have the same length.")
    for s in std_devs:
        if s < 0.0:
            raise ValueError("Each standard deviation must be non-negative.")
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1.")
    if distribution not in ("normal", "lognormal"):
        raise ValueError(
            "distribution must be 'normal' or 'lognormal'."
        )

    rng = _random.Random(seed)
    results: List[float] = []
    n_failures = 0

    # Pre-compute lognormal parameters if needed
    ln_mu: List[float] = []
    ln_sigma: List[float] = []
    if distribution == "lognormal":
        for i in range(n):
            if means[i] <= 0.0:
                raise ValueError(
                    "All means must be positive for lognormal distribution."
                )
            cov_sq = (std_devs[i] / means[i]) ** 2
            sigma_ln = math.sqrt(math.log(1.0 + cov_sq))
            mu_ln = math.log(means[i]) - 0.5 * sigma_ln ** 2
            ln_mu.append(mu_ln)
            ln_sigma.append(sigma_ln)

    for _ in range(n_trials):
        args = []
        for i in range(n):
            if distribution == "normal":
                args.append(rng.gauss(means[i], std_devs[i]))
            else:
                args.append(
                    math.exp(rng.gauss(ln_mu[i], ln_sigma[i]))
                )
        g_val = g_func(*args)
        results.append(g_val)
        if g_val < 0.0:
            n_failures += 1

    mu_g = sum(results) / n_trials
    var_g = sum((r - mu_g) ** 2 for r in results) / (n_trials - 1)
    sigma_g = math.sqrt(var_g)
    beta_val = mu_g / sigma_g if sigma_g > 0.0 else float("inf")
    p_u = n_failures / n_trials

    return mu_g, sigma_g, beta_val, p_u
