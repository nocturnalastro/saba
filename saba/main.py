import numpy as np
from collections import OrderedDict
from sherpa.fit import Fit
from sherpa.data import Data1D, Data1DInt, Data2D, Data2DInt, DataSimulFit
from sherpa.data import Data
from sherpa.models import UserModel, Parameter, SimulFitModel
from sherpa.stats import Chi2, Chi2ConstVar, Chi2DataVar, Chi2Gehrels
from sherpa.stats import Chi2ModVar, Chi2XspecVar, LeastSq
from sherpa.stats import CStat, WStat, Cash
from sherpa.optmethods import GridSearch, LevMar, MonCar, NelderMead
from sherpa.estmethods import Confidence, Covariance, Projection
from sherpa.sim import MCMC

from astropy.utils import format_doc
from astropy.utils.exceptions import AstropyUserWarning

# from astropy.tests.helper import catch_warnings


# with catch_warnings(AstropyUserWarning) as warns:
#    """this is to stop the import warning
#    occuring when we import into saba"""
from astropy.modeling.fitting import Fitter

#
# for w in warns:
#    if "SherpaFitter" not in w.message.message:
#        warnings.warn(w)

# from astropy.modeling

__all__ = ("SherpaFitter", "SherpaMCMC")


class SherpaWrapper(object):
    value = None

    def __init__(self, value=None):
        if value is not None:
            self.set(value)

    def set(self, value):
        try:
            self.value = self._sherpa_values[value.lower()]
        except KeyError:
            UserWarning("Value not found")  # todo handle


class Stat(SherpaWrapper):
    """
    A wrapper for the fit statistics of sherpa

    Parameters
    ----------

        value: String
            the name of a sherpa statistics.
    """

    _sherpa_values = {
        "cash": Cash,
        "wstat": WStat,
        "cstat": CStat,
        "chi2": Chi2,
        "chi2constvar": Chi2ConstVar,
        "chi2datavar": Chi2DataVar,
        "chi2gehrels": Chi2Gehrels,
        "chi2modvar": Chi2ModVar,
        "chi2xspecvar": Chi2XspecVar,
        "leastsq": LeastSq,
    }


class OptMethod(SherpaWrapper):
    """
    A wrapper for the optimization methods of sherpa

    Parameters
    ----------

        value: String
            the name of a sherpa optimization method.
    """

    _sherpa_values = {
        "simplex": GridSearch,
        "levmar": LevMar,
        "moncar": MonCar,
        "neldermead": NelderMead,
    }


class EstMethod(SherpaWrapper):
    """
    A wrapper for the error estimation methods of sherpa

    Parameters
    ----------
        value: String
            the name of a sherpa statistics.
    """

    _sherpa_values = {
        "confidence": Confidence,
        "covariance": Covariance,
        "projection": Projection,
    }


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

    def get_sampler(self,):
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
        posible options include
        {opt}
    statistic : string
        the name of a sherpa statistic.
        posible options include
        {stat}
    estmethod : string
        the name of a sherpa estmethod.
        possible options include
        {est}

    """.format(
        opt=", ".join(OptMethod._sherpa_values.keys()),
        stat=", ".join(Stat._sherpa_values.keys()),
        est=", ".join(EstMethod._sherpa_values.keys()),
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
        pnames = [
            p.split(".", 1)[-1] for p in self.error_info.parnames
        ]  # this is to remove the model name
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


class Dataset(SherpaWrapper):

    """
    Parameters
    ----------
    n_dim: int
        Used to verify required number of dimensions.
    x : array (or list of arrays)
        input coordinate (Independent for 1D & 2D fits)
    y : array (or list of arrays)
        input coordinates(Dependant for 1D fits or Independent for 2D fits)
    z : array (or list of arrays) (optional)
        input coordinates (Dependant for 2D fits)
    xbinsize : array (or list of arrays) (optional)
        an array of errors in x
    ybinsize : array (or list of arrays) (optional)
        an array of errors in y
    err : array (or list of arrays) (optional)
        an array of errors in z
    bkg : array or list of arrays (optional)
            this will act as background data
    bkg_sale : float or list of floats (optional)
            the scaling factor for the dataset if a single value
            is supplied it will be copied for each dataset
    Returns
    -------
    _data: a sherpa dataset
    """

    def __init__(
        self,
        n_dim,
        x,
        y,
        z=None,
        xbinsize=None,
        ybinsize=None,
        err=None,
        bkg=None,
        bkg_scale=1,
    ):

        x = np.array(x)
        y = np.array(y)
        if x.ndim == 2 or (x.dtype == np.object or y.dtype == np.object):
            data = []
            if z is None:
                z = len(x) * [None]

            if xbinsize is None:
                xbinsize = len(x) * [None]

            if ybinsize is None:
                ybinsize = len(y) * [None]

            if err is None:
                err = len(z) * [None]

            if bkg is None:
                bkg = len(x) * [None]
            try:
                iter(bkg_scale)
            except TypeError:
                bkg_scale = len(x) * [bkg_scale]

            for nn, (xx, yy, zz, xxe, yye, zze, bkg, bkg_scale) in enumerate(
                zip(x, y, z, xbinsize, ybinsize, err, bkg, bkg_scale)
            ):
                data.append(
                    self._make_dataset(
                        n_dim,
                        x=xx,
                        y=yy,
                        z=zz,
                        xbinsize=xxe,
                        ybinsize=yye,
                        err=zze,
                        bkg=bkg,
                        bkg_scale=bkg_scale,
                        n=nn,
                    )
                )
            self.data = DataSimulFit("wrapped_data", data)
            self.ndata = nn + 1
        else:
            self.data = self._make_dataset(
                n_dim,
                x=x,
                y=y,
                z=z,
                xbinsize=xbinsize,
                ybinsize=ybinsize,
                err=err,
                bkg=bkg,
                bkg_scale=bkg_scale,
            )
            self.ndata = 1

    @staticmethod
    def _make_dataset(
        n_dim,
        x,
        y,
        z=None,
        xbinsize=None,
        ybinsize=None,
        err=None,
        bkg=None,
        bkg_scale=1,
        n=0,
    ):
        """
        Parameters
        ----------
        n_dim: int
            Used to veirfy required number of dimentions.
        x : array
            input coordinates
        y : array
            input coordinates
        z : array (optional)
            input coordinatesbkg
        xbinsize : array (optional)
            an array of errors in x
        ybinsize : array (optional)
            an array of errors in y
        err : array (optional)
            an array of errors in z
        n  : int
            used in error reporting

        Returns
        -------
        _data: a sherpa dataset
        """

        if (z is None and n_dim > 1) or (z is not None and n_dim == 1):
            raise ValueError("Model and data dimentions don't match in dataset %i" % n)

        if z is None:
            assert x.shape == y.shape, "shape of x and y don't match in dataset %i" % n
        else:
            z = np.asarray(z)
            assert x.shape == y.shape == z.shape, (
                "shapes x,y and z don't match in dataset %i" % n
            )

        if xbinsize is not None:
            xbinsize = np.array(xbinsize)
            assert x.shape == xbinsize.shape, (
                "x's and xbinsize's shapes do not match in dataset %i" % n
            )

        if z is not None and err is not None:
            err = np.array(err)
            assert z.shape == err.shape, (
                "z's and err's shapes do not match in dataset %i" % n
            )

            if ybinsize is not None:
                ybinsize = np.array(ybinsize)
                assert y.shape == ybinsize.shape, (
                    "y's and ybinsize's shapes do not match in dataset %i" % n
                )

        else:
            if err is not None:
                err = np.array(err)
                assert y.shape == err.shape, (
                    "y's and err's shapes do not match in dataset %i" % n
                )

        if xbinsize is not None:
            bs = xbinsize / 2.0

        if z is None:
            if xbinsize is None:
                if err is None:
                    if bkg is None:
                        data = Data1D("wrapped_data", x=x, y=y)
                    else:
                        data = Data1DBkg(
                            "wrapped_data", x=x, y=y, bkg=bkg, bkg_scale=bkg_scale
                        )
                else:
                    if bkg is None:
                        data = Data1D("wrapped_data", x=x, y=y, staterror=err)
                    else:
                        data = Data1DBkg(
                            "wrapped_data",
                            x=x,
                            y=y,
                            staterror=err,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
            else:
                if err is None:
                    if bkg is None:

                        data = Data1DInt("wrapped_data", xlo=x - bs, xhi=x + bs, y=y)
                    else:
                        data = Data1DIntBkg(
                            "wrapped_data",
                            xlo=x - bs,
                            xhi=x + bs,
                            y=y,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
                else:
                    if bkg is None:
                        data = Data1DInt(
                            "wrapped_data", xlo=x - bs, xhi=x + bs, y=y, staterror=err
                        )
                    else:
                        data = Data1DIntBkg(
                            "wrapped_data",
                            xlo=x - bs,
                            xhi=x + bs,
                            y=y,
                            staterror=err,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
        else:
            if xbinsize is None and ybinsize is None:
                if err is None:
                    if bkg is None:
                        data = Data2D("wrapped_data", x0=x, x1=y, y=z)
                    else:
                        data = Data2DBkg(
                            "wrapped_data",
                            x0=x,
                            x1=y,
                            y=z,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
                else:
                    if bkg is None:
                        data = Data2D("wrapped_data", x0=x, x1=y, y=z, staterror=err)
                    else:
                        data = Data2DBkg(
                            "wrapped_data",
                            x0=x,
                            x1=y,
                            y=z,
                            staterror=err,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
            elif xbinsize is not None and ybinsize is not None:
                ys = ybinsize / 2.0
                if err is None:
                    if bkg is None:
                        data = Data2DInt(
                            "wrapped_data",
                            x0lo=x - bs,
                            x0hi=x + bs,
                            x1lo=y - ys,
                            x1hi=y + ys,
                            y=z,
                        )
                    else:
                        data = Data2DIntBkg(
                            "wrapped_data",
                            x0lo=x - bs,
                            x0hi=x + bs,
                            x1lo=y - ys,
                            x1hi=y + ys,
                            y=z,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
                else:
                    if bkg is None:
                        data = Data2DInt(
                            "wrapped_data",
                            x0lo=x - bs,
                            x0hi=x + bs,
                            x1lo=y - ys,
                            x1hi=y + ys,
                            y=z,
                            staterror=err,
                        )
                    else:
                        data = Data2DIntBkg(
                            "wrapped_data",
                            x0lo=x - bs,
                            x0hi=x + bs,
                            x1lo=y - ys,
                            x1hi=y + ys,
                            y=z,
                            staterror=err,
                            bkg=bkg,
                            bkg_scale=bkg_scale,
                        )
            else:
                raise ValueError("Set xbinsize and ybinsize, or set neither!")

        return data

    def make_simfit(self, numdata):
        """
        This makes a single datasets into a simdatafit at allow fitting of multiple models by copying the single dataset!

        Parameters
        ----------
        numdata: int
            the number of times you want to copy the dataset i.e if you want 2 datasets total you put 1!
        """

        self.data = DataSimulFit("wrapped_data", [self.data for _ in range(numdata)])
        self.ndata = numdata + 1


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
    def _astropy_to_sherpa_model(model):
        """
        Converts the model using sherpa's usermodel suppling the parameter detail to sherpa
        then using a decorator to allow the call method to act like the calc method
        """

        def _calc2call(func):
            """This decorator makes call and calc work together."""

            def _converter(inp, *x):
                if func.n_inputs == 1:
                    retvals = func.evaluate(x[0], *inp)
                else:
                    retvals = func.evaluate(x[0], x[1], *inp)
                return retvals

            return _converter

        if len(model.ineqcons) > 0 or len(model.eqcons) > 0:
            AstropyUserWarning(
                "In/eqcons are not supported by sherpa these will be ignored!"
            )

        pars = []
        linkedpars = []
        for pname in model.param_names:
            param = getattr(model, pname)
            vals = [
                param.name,
                param.value,
                param.min,
                param.max,
                param.min,
                param.max,
                None,
                param.fixed,
                False,
            ]
            attrnames = [
                "name",
                "val",
                "min",
                "max",
                "hard_min",
                "hard_max",
                "units",
                "frozen",
                "alwaysfrozen",
            ]
            if model.name is None:
                model._name = ""

            pars.append(
                Parameter(
                    modelname="wrap_" + model.name,
                    **dict(
                        [
                            (atr, val)
                            for atr, val in zip(attrnames, vals)
                            if val is not None
                        ]
                    )
                )
            )
            if param.tied is not False:
                linkedpars.append(pname)

        smodel = UserModel(model.name, pars)
        smodel.calc = _calc2call(model)

        for pname in linkedpars:
            param = getattr(model, pname)
            sparam = getattr(smodel, pname)
            sparam.link = param.tied(smodel)

        return smodel

    def get_astropy_model(self):
        """Returns an astropy model based on the sherpa model"""
        return_models = []

        for apymod, shmod in self.model_dict.items():
            return_models.append(apymod.copy())
            for pname, pval in map(lambda p: (p.name, p.val), shmod.pars):
                getattr(return_models[-1], pname.split(".")[-1]).value = pval

        if len(return_models) > 1:
            return return_models
        else:
            return return_models[0]


class Data1DIntBkg(Data1DInt):
    """
    Data1DInt which tricks sherpa into using the background object without using DataPHA

    Parameters
    ----------
    name: string
        dataset name
    xlo: array
       the array which represents the lower x value for the x bins
    xhi: array
       the array which represents the upper x value for the x bins
    y: array
       the array which represents y data
    bkg: array
       the array which represents bkgdata
    staterror: array (optional)
        the array which represents the errors on z
    bkg_scale: float
        the scaling factor for background data
    src_scale: float
        the scaling factor for source data
    """

    _response_ids = [0]
    _background_ids = [0]

    @property
    def response_ids(self):
        return self._response_ids

    @property
    def background_ids(self):
        return self._background_ids

    @property
    def backscal(self):
        return self._bkg_scale

    def get_background(self, index):
        return self._backgrounds[index]

    def __init__(
        self, name, xlo, xhi, y, bkg, staterror=None, bkg_scale=1, src_scale=1
    ):
        self._bkg = np.asanyarray(bkg)
        self._bkg_scale = src_scale
        self.exposure = 1

        self.subtracted = False

        self._backgrounds = [BkgDataset(bkg, bkg_scale)]
        Data.__init__(self, (xlo, xhi), y, staterror)


class Data1DBkg(Data1D):
    """
    Data1D which tricks sherpa into using the background object without using DataPHA

    Parameters
    ----------
    name: string
        dataset name
    x: array
       the array which represents the x values
    y: array
       the array which represents y data
    bkg: array
       the array which represents background data
    staterror: array (optional)
        the array which represents the errors on z
    bkg_scale: float
        the scaling factor for background data
    src_scale: float
        the scaling factor for source data
    """

    _response_ids = [0]
    _background_ids = [0]

    @property
    def response_ids(self):
        return self._response_ids

    @property
    def background_ids(self):
        return self._background_ids

    @property
    def backscal(self):
        return self._bkg_scale

    def get_background(self, index):
        return self._backgrounds[index]

    def __init__(self, name, x, y, bkg, staterror=None, bkg_scale=1, src_scale=1):
        self._bkg = np.asanyarray(bkg)
        self._bkg_scale = src_scale
        self.exposure = 1
        self.subtracted = False

        self._backgrounds = [BkgDataset(bkg, bkg_scale)]
        Data.__init__(self, name, (x,), y, staterror)


class Data2DIntBkg(Data2DInt):
    """
    Data2DInt which tricks sherpa into using the background object without using DataPHA

    Parameters
    ----------
    name: string
        dataset name
    xlo: array
       the array which represents the lower x value for the x bins
    xhi: array
       the array which represents the upper x value for the x bins
    ylo: array
       the array which represents the lower y value for the y bins
    yhi: array
       the array which represents the upper y value for the y bins
    z: array
       the array which represents z data
    bkg: array
       the array which represents bkgdata
    staterror: array (optional)
        the array which represents the errors on z
    bkg_scale: float
        the scaling factor for background data
    src_scale: float
        the scaling factor for source data
    """

    _response_ids = [0]
    _background_ids = [0]

    @property
    def response_ids(self):
        return self._response_ids

    @property
    def background_ids(self):
        return self._background_ids

    @property
    def backscal(self):
        return self._bkg_scale

    def get_background(self, index):
        return self._backgrounds[index]

    def __init__(
        self, name, xlo, xhi, ylo, yhi, z, bkg, staterror=None, bkg_scale=1, src_scale=1
    ):
        self._bkg = np.asanyarray(bkg)
        self._bkg_scale = src_scale
        self.exposure = 1

        self.subtracted = False

        self._backgrounds = [BkgDataset(bkg, bkg_scale)]
        Data.__init__(self, (xlo, xhi, ylo, yhi), z, staterror)


class Data2DBkg(Data2D):
    """
    Data2D which tricks sherpa into using the background object without
    using DataPHA

    Parameters
    ----------
    name: string
        dataset name
    x: array
       the array which represents x data
    y: array
       the array which represents y data
    z: array
       the array which represents z data
    bkg: array
       the array which represents bkgdata
    staterror: array (optional)
        the array which represents the errors on z
    bkg_scale: float
        the scaling factor for background data
    src_scale: float
        the scaling factor for source data
    """

    _response_ids = [0]
    _background_ids = [0]

    @property
    def response_ids(self):
        return self._response_ids

    @property
    def background_ids(self):
        return self._background_ids

    @property
    def backscal(self):
        return self._bkg_scale

    def get_background(self, index):
        return self._backgrounds[index]

    def __init__(self, name, x, y, z, bkg, staterror=None, bkg_scale=1, src_scale=1):
        self._bkg = np.asanyarray(bkg)
        self._bkg_scale = src_scale
        self.exposure = 1
        self.subtracted = False

        self._backgrounds = [BkgDataset(bkg, bkg_scale)]
        Data.__init__(self, name, (x, y), z, staterror)


class BkgDataset(object):
    """
    The background object which is used to caclulate fit
    stat's which require it.

    Parameters
    ----------
    bkg: array
        the background data
    bkg_scale: float
        the ratio of src/bkg
    """

    def __init__(self, bkg, bkg_scale):
        self._bkg = bkg
        self._bkg_scale = bkg_scale
        self.exposure = 1

    def get_dep(self, flag):
        return self._bkg

    @property
    def backscal(self):
        return self._bkg_scale
