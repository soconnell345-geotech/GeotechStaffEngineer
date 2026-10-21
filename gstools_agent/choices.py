"""What a geostatistics result chose (BACKLOG GM-265): the ``choices`` list
every other module's result carries (``geotech_common.choices``).

The covariance model, whether the variogram was fitted, the nugget, the
kriging type, the simulation method, the number of realizations and the
seed are each stated as given or default, with the basis and the other
options the tool offers. ``how_given`` says "given" only for an input the
call named, even when it equals the default.
"""

from geotech_common.choices import MODULE, choice, how_given

MODELS = ["Gaussian", "Exponential", "Matern", "Spherical", "Linear",
          "Stable", "Rational", "Cubic", "HyperSpherical"]
KRIGING_TYPES = ["ordinary", "simple", "universal"]

_KRIGING_BASIS = {
    "ordinary": "the mean of the field is unknown but constant (Journel & "
                "Huijbregts 1978)",
    "simple": "the mean is taken as known: the sample mean of these "
              "observations",
    "universal": "a linear trend in x and y is removed (linear drift); the "
                 "variance is then that of the residual",
}


def model_choice(model_type, default="Gaussian", fitted=False):
    """The covariance model. The module does not test it against the data."""
    basis = ("the model shape is not chosen from the data; Gaussian is the "
             "smoothest, Exponential the roughest of the common models")
    if fitted:
        basis += ("; the variance, correlation length and nugget are "
                  "fitted for this model")
    return choice("covariance model", model_type,
                  how_given(model_type, default, name="model_type"),
                  basis, MODELS)


def kriging_type_choice(kriging_type):
    return choice("kriging type", kriging_type,
                  how_given(kriging_type, "ordinary", name="kriging_type"),
                  _KRIGING_BASIS.get(kriging_type, ""), KRIGING_TYPES)


def fit_choice(fitted, fit_variogram, default, nugget_fitted, nugget):
    """Whether the variogram was fitted to the data."""
    if fitted and nugget_fitted:
        chosen = ("variance (sill), correlation length and nugget fitted to "
                  "the empirical variogram")
    elif fitted:
        chosen = ("variance (sill) and correlation length fitted to the "
                  f"empirical variogram; nugget held at {nugget:.4g}")
    else:
        chosen = ("not fitted: the variance, correlation length and nugget "
                  "are used as set")
    if fitted:
        basis = ("least-squares fit of the model to the empirical "
                 "variogram, lag classes from the data's own pair distances")
    else:
        basis = ("a fit needs about 30 pairs per lag class (Journel & "
                 "Huijbregts 1978); with a few borings it is unstable, so "
                 "the sill and range are better set from judgement or "
                 "published values")
    return choice("variogram", chosen,
                  how_given(fit_variogram, default, name="fit_variogram"),
                  basis, ["fitted to the data (fit_variogram true)",
                          "set by the user (fit_variogram false)"])


def nugget_choice(value, fitted, how, basis):
    """The nugget: fitted, or set (held) at a value."""
    chosen = f"fitted ({value:.4g})" if fitted else f"{value:.4g}"
    return choice("nugget", chosen, how, basis,
                  ["fitted with the sill and range",
                   "held at a number, for example the measurement error"])


def scale_choices(variance, len_scale, variance_v, len_v, extent_note,
                  n, fitted):
    """Sill and correlation length when not fitted: given, or from the
    observations."""
    out = []
    start = " (starting value for the fit)" if fitted else ""
    for what, given, used, basis in (
            ("variance (sill)", variance, variance_v,
             f"the sample variance of the {n} observations{start}"),
            ("correlation length", len_scale, len_v,
             f"one third of the data extent{extent_note}{start}")):
        if given is None:
            out.append(choice(what, f"{used:.4g}", MODULE, basis))
        else:
            out.append(choice(
                what, f"{used:.4g}",
                how_given(given, None, name=("variance" if "sill" in what
                                             else "len_scale")),
                "the value in the call" + start))
    return out


def realizations_choice(n_realizations, kept):
    return choice("number of realizations", f"{kept}",
                  how_given(n_realizations, 1000, name="n_realizations"),
                  "the convergence table shows whether more are needed",
                  ["up to 20,000"])


def seed_choice(seed):
    return choice("random seed", f"{seed} (realization i uses seed + i)",
                  how_given(seed, 42, name="seed"),
                  "a fixed seed makes the run repeatable; another seed "
                  "gives another set of realizations")


def method_choice(method, text):
    return choice(
        "simulation method", method,
        how_given(method, "kriging_residual", name="simulation_method"),
        text, ["kriging_residual", "condsrf"])
