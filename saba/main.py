import numpy as np
from collections import OrderedDict
import itertools
from sherpa.fit import Fit
from sherpa.models import UserModel, Parameter, SimulFitModel
from sherpa.sim import MCMC

from astropy.utils import format_doc
from astropy.utils.exceptions import AstropyUserWarning
from astropy.modeling.fitting import Fitter

from . import Dataset, OptMethod, Stat, EstMethod

__all__ = ("SherpaFitter", "SherpaMCMC")


def _remove_model_from_param_name(name):
    return name.split(".", 1)[-1]


class SherpaMCMC(object):
    """
    An interface which makes use of sherpa's MCMC(pyBLoCXS) functionality.

    Parameters
    ----------
    fitter: a `SherpaFitter` instance.
            used to caluate the fit statstics, must have been fit as
            the covariance matrix is used.
    sampler: string
            the name of a valid sherpa sampler.
    walker: string
            the name of a valid sherpa walker.
    """

    def __init__(self, fitter, sampler="mh", walker="mh"):
        self._mcmc = MCMC()

        if hasattr(fitter.fit_info, "statval"):
            self._fitter = fitter._fitter

            if not hasattr(fitter.error_info, "extra_output"):
                fitter.est_errors()
            self._cmatrix = fitter.error_info.extra_output
            pars = fitter._fitmodel.sherpa_model.pars
            self.parameter_map = OrderedDict(map(lambda x: (x.name, x), pars))
        else:
            raise AstropyUserWarning(
                "Must have valid fit! " "Covariance matrix is not present"
            )

    def __call__(self, niter=200000):
        """
        based on the `sherpa.sim.get_draws`

        Parameters
        ----------
        niter: int
            the number of samples you wish to draw.

        Returns
        -------
        stat_values: array(float)
            the fit statistic of the draw
        accepted: array(bool)
            if the fit was accepted
        parameters: dict
            the parameter values for each draw
        """

        draws = self._mcmc.get_draws(self._fitter, self._cmatrix, niter=niter)
        self._stat_vals, self._accepted, self._parameter_vals = draws
        self.acception_rate = self._accepted.sum() * 100.0 / self._accepted.size
        self.parameters = OrderedDict()

        parameter_key_list = list(self.parameter_map.keys())
        for n, parameter_set in enumerate(self._parameter_vals):
            pname = parameter_key_list[n]
            self.parameters[pname] = self._parameter_vals[n, :]
        return draws

    def set_sampler_options(self, opt, value):
        """
        Set an option for the current MCMC sampler.

        Parameters
        ----------
        opt : str
           The option to change. Use `get_sampler` to view the
           available options for the current sampler.
        value :
           The value for the option.

        Notes
        -----
        The options depend on the sampler. The options include:

        defaultprior
           Set to ``False`` when the default prior (flat, between the
           parameter's soft limits) should not be used. Use
           `set_prior` to set the form of the prior for each
           parameter.

        inv:
           A bool, or array of bools, to indicate which parameter is
           on the inverse scale.

        log:
           A bool, or array of bools, to indicate which parameter is
           on the logarithm (natural log) scale.

        original:
           A bool, or array of bools, to indicate which parameter is
           on the original scale.

        p_M:
           The proportion of jumps generatd by the Metropolis
           jumping rule.

        priorshape:
           An array of bools indicating which parameters have a
           user-defined prior functions set with `set_prior`.

        scale:
           Multiply the output of `covar` by this factor and
           use the result as the scale of the t-distribution.

        Examples
        --------
        >> mcmc = SherpaMCMC(sfit)
        >> mcmc.set_sampler_opt('scale', 3)
        """
        self._mcmc.set_sampler_opt(opt, value)

    def get_sampler(self):
        return self._mcmc.get_sampler()

    def set_prior(self, parameter, prior):
        """
        Set the prior function to use with a parameter.
        The default prior used by the `SherpaMCMC` function call for each parameter
        is flat, varying between the minimum and maximum
        values of the parameter (as given by the ``min`` and
        ``max`` attributes of the parameter object).

        Parameters
        ----------
        par : sherpa.models.parameter.Parameter instance
           A parameter of a model instance.

        prior : function or `sherpa.models.model.Model` instance
           The function to use for a prior. It must accept a
           single argument and return a value of the same size
           as the input.

        Examples
        --------
        Create a function (``lognorm``) and use it as the prior the
        ``nH`` parameter

        >> def lognorm(x):
                sigma = 0.5
                x0 = 20
                dx = np.log10(x) - x0
                norm = sigma / np.sqrt(2 * np.pi)
                return norm * np.exp(-0.5*dx*dx/(sigma*sigma))

        >> mcmc.set_prior('nH', lognorm)

        """

        if parameter in self.parameter_map:
            self._mcmc.set_prior(self.parameter_map[parameter], prior)
        else:
            raise AstropyUserWarning(
                "Parmater {name} not found in parameter" "map".format(name=parameter)
            )

    @property
    def accepted(self):
        """
        The stored list of bools if each draw was accepted or not.
        """
        return self._accepted

    @property
    def stat_values(self):
        """
        The stored values for the fit statistic of each run.
        """
        return self._stat_values


class SherpaFitter(Fitter):
    __doc__ = """
    Sherpa Fitter for astropy models.

    Parameters
    ----------
    optimizer : string
        the name of a sherpa optimizer.
        posible options include:
        {opt}
    statistic : string
        the name of a sherpa statistic.
        posible options include:
        {stat}
    estmethod : string
        the name of a sherpa estmethod.
        possible options include:
        {est}

    """.format(
        opt=", ".join(OptMethod.get_available()),
        stat=", ".join(Stat.get_available()),
        est=", ".join(EstMethod.get_available()),
    )  # is this evil?

    supported_constraints = ["bounds", "fixed", "tied"]

    def __init__(self, optimizer="levmar", statistic="leastsq", estmethod="covariance"):
        try:
            optimizer = optimizer.value
        except AttributeError:
            optimizer = OptMethod(optimizer).value

        try:
            statistic = statistic.value
        except AttributeError:
            statistic = Stat(statistic).value

        super(SherpaFitter, self).__init__(optimizer=optimizer, statistic=statistic)

        try:
            self._est_method = estmethod.value()
        except AttributeError:
            self._est_method = EstMethod(estmethod).value()

        self.fit_info = {}
        self._fitter = None  # a handle for sherpa fit function
        self._fitmodel = None  # a handle for sherpa fit model
        self._data = None  # a handle for sherpa dataset
        self.error_info = {}

        setattr(
            self.__class__,
            "opt_config",
            property(lambda s: s._opt_config, doc=self._opt_method.__doc__),
        )

        # sherpa doesn't currently have a docstring for est_method but maybe the future
        setattr(
            self.__class__,
            "est_config",
            property(lambda s: s._est_config, doc=self._est_method.__doc__),
        )

    def __call__(
        self,
        models,
        x,
        y,
        z=None,
        xbinsize=None,
        ybinsize=None,
        err=None,
        bkg=None,
        bkg_scale=1,
        **kwargs
    ):
        """
        Fit the astropy model with a the sherpa fit routines.


        Parameters
        ----------
        models : `astropy.modeling.FittableModel` or list of `astropy.modeling.FittableModel`
            model to fit to x, y, z
        x : array or list of arrays
            input coordinates (independent for 1D & 2D fits)
        y : array or list of arrays
            input coordinates (dependent for 1D fits or independent for 2D fits)
        z : array or list of arrays (optional)
            input coordinates (dependent for 2D fits)
        xbinsize : array or list of arrays (optional)
            an array of xbinsizes in x  - this will be x -/+ (binsize  / 2.0)
        ybinsize : array or list of arrays (optional)
            an array of xbinsizes in y  - this will be y -/+ (ybinsize / 2.0)
        err : array or list of arrays (optional)
            an array of errors in dependent variable
        bkg : array or list of arrays (optional)
            this will act as background data
        bkg_sale : float or list of floats (optional)
            the scaling factor for the dataset if a single value
            is supplied it will be copied for each dataset
        **kwargs :
            keyword arguments will be passed on to sherpa fit routine

        Returns
        -------
        model_copy : `astropy.modeling.FittableModel` or a list of models.
            a copy of the input model with parameters set by the fitter
        """

        tie_list = []
        try:
            n_inputs = models[0].n_inputs
        except TypeError:
            n_inputs = models.n_inputs

        self._data = Dataset(n_inputs, x, y, z, xbinsize, ybinsize, err, bkg, bkg_scale)

        if self._data.ndata > 1:

            if len(models) == 1:
                self._fitmodel = ConvertedModel(
                    [models.copy() for _ in range(self._data.ndata)], tie_list
                )
                # Copy the model so each data set has the same model!
            elif len(models) == self._data.ndata:
                self._fitmodel = ConvertedModel(models, tie_list)
            else:
                raise Exception(
                    "Don't know how to handle multiple models "
                    "unless there is one foreach dataset"
                )
        else:
            if len(models) > 1:
                self._data.make_simfit(len(models))
                self._fitmodel = ConvertedModel(models, tie_list)
            else:
                self._fitmodel = ConvertedModel(models)

        self._fitter = Fit(
            self._data.data,
            self._fitmodel.sherpa_model,
            self._stat_method,
            self._opt_method,
            self._est_method,
            **kwargs
        )
        self.fit_info = self._fitter.fit()

        return self._fitmodel.get_astropy_model()

    def est_errors(
        self, sigma=None, maxiters=None, numcores=1, methoddict=None, parlist=None
    ):
        """
        Use sherpa error estimators based on the last fit.

        Parameters
        ----------
        sigma: float
            this will be set as the confidance interval for which the errors are found too.
        maxiters: int
            the maximum number of iterations the error estimator will run before giving up.
        methoddict: dict
            !not quite sure couldn't figure this one out yet!
        parlist: list
            a list of parameters to find the confidance interval of if none are provided all free
            parameters will be estimated.
        """
        if self._fitter is None:
            ValueError("Must complete a valid fit before errors can be calculated")
        if sigma is not None:
            self._fitter.estmethod.config["sigma"] = sigma
        if maxiters is not None:
            self._fitter.estmethod.config["maxiters"] = maxiters
        if "numcores" in self._fitter.estmethod.config:
            if not numcores == self._fitter.estmethod.config["numcores"]:
                self._fitter.estmethod.config["numcores"] = numcores

        self.error_info = self._fitter.est_errors(
            methoddict=methoddict, parlist=parlist
        )
        pnames = list(map(_remove_model_from_param_name, self.error_info.parnames))
        return (
            pnames,
            self.error_info.parvals,
            self.error_info.parmins,
            self.error_info.parmaxes,
        )

    @property
    def _est_config(self):
        """This returns the est.config property"""
        return self._est_method.config

    @property
    def _opt_config(self):
        """This returns the opt.config property"""
        return self._opt_method.config

    # Here is the MCMC wrapper!
    @format_doc("{__doc__}\n{mcmc_doc}", mcmc_doc=SherpaMCMC.__doc__)
    def get_sampler(self, *args, **kwargs):
        """
        This returns and instance of `SherpaMCMC` with it's self as the fitter
        """
        return SherpaMCMC(self, *args, **kwargs)


class ConvertedModel(object):
    """
    This  wraps the model convertion to sherpa models and from astropy models and back!

    Parameters
    ----------
    models: `astropy.modeling.FittableModel` (or list of)
    tie_list: list (optional)
        a list of parameter pairs which will be tied accross models
        e.g. [(modelB.y, modelA.x)] will mean that y in modelB will be tied to x of modelA
    """

    def __init__(self, models, tie_list=None):
        self.model_dict = OrderedDict()
        try:
            models.parameters  # does it quack
            self.sherpa_model = self._astropy_to_sherpa_model(models)
            self.model_dict[models] = self.sherpa_model
        except AttributeError:
            for mod in models:
                self.model_dict[mod] = self._astropy_to_sherpa_model(mod)

                if tie_list is not None:
                    for par1, par2 in tie_list:
                        getattr(self.model_dict[par1._model], par1.name).link = getattr(
                            self.model_dict[par2._model], par2.name
                        )

            self.sherpa_model = SimulFitModel(
                "wrapped_fit_model", self.model_dict.values()
            )

    @staticmethod
    def _prepare_param(model_name, parameter):
        attrs = {
            "name": parameter.name,
            "val": parameter.value,
            "min": parameter.min,
            "max": parameter.max,
            "hard_min": parameter.min,
            "hard_max": parameter.max,
            # TODO add this back in "units": None,
            "frozen": parameter.fixed,
            "alwaysfrozen": False,
        }
        return Parameter(
            modelname="wrap_" + model_name,
            **{k: v for k, v in attrs.items() if v is not None}
        )

    @classmethod
    def _astropy_to_sherpa_model(cls, astropy_model):
        """
        Converts the model using sherpa's usermodel suppling the parameter detail to sherpa
        then using a decorator to allow the call method to act like the calc method
        """

        def _calc2call(func):
            """This decorator makes call and calc work together."""

            def _converter(model_params, *inputs):
                return func.evaluate(*itertools.chain(inputs, model_params))

            return _converter

        if len(astropy_model.ineqcons) > 0 or len(astropy_model.eqcons) > 0:
            AstropyUserWarning(
                "In/eqcons are not supported by sherpa these will be ignored!"
            )

        pars = []
        linkedpars = []
        for param_name in astropy_model.param_names:
            astropy_param = getattr(astropy_model, param_name)

            pars.append(
                cls._prepare_param(astropy_model.name or "astropy_model", astropy_param)
            )

            if astropy_param.tied:
                linkedpars.append(param_name)

        sherpa_model = UserModel(astropy_model.name, pars)
        sherpa_model.calc = _calc2call(astropy_model)

        for param_name in linkedpars:
            astropy_param = getattr(astropy_model, param_name)
            sherpa_param = getattr(sherpa_model, param_name)
            # Call the astropy link function on the sherpa model evaluate the link
            sherpa_param.link = astropy_param.tied(sherpa_model)

        return sherpa_model

    def get_astropy_model(self):
        """Returns an astropy model based on the sherpa model"""
        return_models = []

        for astropy_model, sherpa_model in self.model_dict.items():
            return_models.append(astropy_model.copy())
            for pname, pval in map(
                lambda p: (_remove_model_from_param_name(p.name), p.val),
                sherpa_model.pars,
            ):
                getattr(return_models[-1], pname).value = pval

        if len(return_models) > 1:
            return return_models
        else:
            return return_models[0]
