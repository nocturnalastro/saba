# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Module to test fitting routines
"""

import numpy as np

from numpy.testing import assert_allclose

try:
    from sherpa.data import DataSimulFit

    HAS_SHERPA = True
except ImportError:
    HAS_SHERPA = False

import warnings

from saba import SherpaFitter, Dataset, ConvertedModel
from astropy.modeling.models import Gaussian1D, Gaussian2D, Polynomial1D

_RANDOM_SEED = 0x1337
np.random.seed(_RANDOM_SEED)


class TestSherpaFitter(object):
    def setup_class(self):
        # make data and models to use later!
        err = 0.1

        self.x1 = np.arange(1, 10, 0.1)
        self.dx1 = np.ones(self.x1.shape) * (0.1 / 2.0)
        self.x2 = np.arange(1, 10, 0.05)
        self.dx2 = np.ones(self.x2.shape) * (0.05 / 2.0)

        self.model1d = Gaussian1D(mean=5, amplitude=10, stddev=0.8)
        self.model1d_2 = Gaussian1D(mean=4, amplitude=5, stddev=0.2)

        self.tmodel1d = self.model1d.copy()
        self.tmodel1d_2 = self.model1d_2.copy()

        self.y1 = self.model1d(self.x1)
        self.y1 += err * np.random.uniform(-1.0, 1.0, size=self.y1.size)
        self.dy1 = err * np.random.uniform(0.5, 1.0, size=self.y1.size)

        self.y2 = self.model1d_2(self.x2)
        self.y2 += err * np.random.uniform(-1.0, 1.0, size=self.y2.size)
        self.dy2 = err * np.random.uniform(0.5, 1.0, size=self.y2.size)

        self.model1d.mean = 4
        self.model1d.amplitude = 6
        self.model1d.stddev = 0.5

        self.model1d_2.mean = 5
        self.model1d_2.amplitude = 10
        self.model1d_2.stddev = 0.3

        self.xx2, self.xx1 = np.mgrid[2:12:1, 2:12:1]
        self.shape = self.xx2.shape
        self.xx1 = self.xx1.flatten()
        self.xx2 = self.xx2.flatten()

        self.model2d = Gaussian2D(
            amplitude=10, x_mean=5, y_mean=6, x_stddev=0.8, y_stddev=1.1
        )
        self.model2d.theta.fixed = True

        self.yy = self.model2d(self.xx1, self.xx2)
        self.dxx1 = err * np.random.uniform(0.5, 1.0, size=self.xx1.size)
        self.dxx2 = err * np.random.uniform(0.5, 1.0, size=self.xx2.size)
        self.dyy = err * np.random.uniform(0.5, 1.0, size=self.yy.size)

        self.tmodel2d = self.model2d.copy()
        self.model2d.amplitude = 5
        self.model2d.x_mean = 6
        self.model2d.y_mean = 5
        self.model2d.x_stddev = 0.2
        self.model2d.y_stddev = 0.7

        # to stop stddev going negitive and getting div by zero error
        self.model1d.stddev.min = 1e-99
        self.model1d_2.stddev.min = 1e-99
        self.model2d.x_stddev.min = 1e-99
        self.model2d.y_stddev.min = 1e-99

        # Lets define some tophats
        self.rsp1 = np.zeros_like(self.x1)
        self.rsp1[(self.x1 > 4) & (self.x1 < 6)] = 1
        self.rsp2 = np.zeros_like(self.x2)
        self.rsp2[(self.x2 > 4) & (self.x2 < 6)] = 1

        self.rsp2d = np.zeros_like(self.xx1)
        self.rsp2d[
            (self.xx1 > 4) & (self.xx1 < 6) & (self.xx2 > 4) & (self.xx2 < 6)
        ] = 1
        self.rsp2d.flatten()
        self.fitter = SherpaFitter(statistic="Chi2")

    def test_make_datasets_single(self):
        """
        Makes 1d datasets with and without errors to check it's working correctly.
        """
        data_noerr = Dataset(1, x=self.x1, y=self.y1).data
        assert_allclose(data_noerr.get_x(), self.x1)
        assert_allclose(data_noerr.get_y(), self.y1)

        data_binsize = Dataset(1, x=self.x1, y=self.y1, xbinsize=self.dx1).data
        assert_allclose(data_binsize.get_x(), self.x1)
        assert_allclose(data_binsize.get_y(), self.y1)
        assert_allclose(data_binsize.get_xerr(), self.dx1)

        data_yerr = Dataset(1, x=self.x1, y=self.y1, err=self.dy1).data
        assert_allclose(data_yerr.get_x(), self.x1)
        assert_allclose(data_yerr.get_y(), self.y1)
        assert_allclose(data_yerr.get_yerr(), self.dy1)

        data_botherr = Dataset(
            1, x=self.x1, y=self.y1, err=self.dy1, xbinsize=self.dx1
        ).data
        assert_allclose(data_botherr.get_x(), self.x1)
        assert_allclose(data_botherr.get_y(), self.y1)
        assert_allclose(data_botherr.get_yerr(), self.dy1)
        assert_allclose(data_botherr.get_xerr(), self.dx1)

    def test_make_datasets_2d(self):
        """
        Makes 2d datasets with and without errors to check it's working correctly.
        """
        data_noerr = Dataset(2, x=self.xx1, y=self.xx2, z=self.yy).data
        assert_allclose(data_noerr.get_x0(), self.xx1)
        assert_allclose(data_noerr.get_x1(), self.xx2)
        assert_allclose(data_noerr.get_y(), self.yy)

        data_xyerr = Dataset(
            2, x=self.xx1, y=self.xx2, xbinsize=self.dxx1, ybinsize=self.dxx2, z=self.yy
        ).data
        assert_allclose(
            data_xyerr.get_indep(),
            np.vstack(
                [
                    self.xx1 - self.dxx1 / 2,
                    self.xx2 - self.dxx2 / 2,
                    self.xx1 + self.dxx1 / 2,
                    self.xx2 + self.dxx2 / 2,
                ]
            ),
        )
        assert_allclose(data_xyerr.get_y(), self.yy)

        data_xyzerr = Dataset(
            2,
            x=self.xx1,
            y=self.xx2,
            xbinsize=self.dxx1,
            ybinsize=self.dxx2,
            z=self.yy,
            err=self.dyy,
        ).data
        assert_allclose(
            data_xyerr.get_indep(),
            np.vstack(
                [
                    self.xx1 - self.dxx1 / 2,
                    self.xx2 - self.dxx2 / 2,
                    self.xx1 + self.dxx1 / 2,
                    self.xx2 + self.dxx2 / 2,
                ]
            ),
        )
        assert_allclose(data_xyzerr.get_y(), self.yy)
        assert_allclose(data_xyzerr.get_yerr(), self.dyy)

    def test_make_datasets_multi(self):
        """
        Makes DataSimulFit datasets with and without errors.
        """

        data_noerr = Dataset(1, x=[self.x1, self.x2], y=[self.y1, self.y2]).data
        assert len(data_noerr.datasets) == 2
        assert isinstance(data_noerr, DataSimulFit)
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_x(), data_noerr.datasets),
            [self.x1, self.x2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_y(), data_noerr.datasets),
            [self.y1, self.y2],
        )

        data_binsize = Dataset(
            1, x=[self.x1, self.x2], y=[self.y1, self.y2], xbinsize=[self.dx1, self.dx2]
        ).data
        assert len(data_binsize.datasets) == 2
        assert isinstance(data_binsize, DataSimulFit)
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_x(), data_binsize.datasets),
            [self.x1, self.x2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_y(), data_binsize.datasets),
            [self.y1, self.y2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_xerr(), data_binsize.datasets),
            [self.dx1, self.dx2],
        )

        data_yerr = Dataset(
            1, x=[self.x1, self.x2], y=[self.y1, self.y2], err=[self.dy1, self.dy2]
        ).data
        assert len(data_yerr.datasets) == 2
        assert isinstance(data_yerr, DataSimulFit)
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_x(), data_yerr.datasets),
            [self.x1, self.x2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_y(), data_yerr.datasets),
            [self.y1, self.y2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_yerr(), data_yerr.datasets),
            [self.dy1, self.dy2],
        )

        data_xyerr = Dataset(
            1,
            x=[self.x1, self.x2],
            y=[self.y1, self.y2],
            err=[self.dy1, self.dy2],
            xbinsize=[self.dx1, self.dx2],
        ).data
        assert len(data_xyerr.datasets) == 2
        assert isinstance(data_xyerr, DataSimulFit)
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_x(), data_xyerr.datasets),
            [self.x1, self.x2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_y(), data_xyerr.datasets),
            [self.y1, self.y2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_yerr(), data_xyerr.datasets),
            [self.dy1, self.dy2],
        )
        map(
            lambda x, y: assert_allclose(x, y),
            map(lambda x: x.get_xerr(), data_xyerr.datasets),
            [self.dx1, self.dx2],
        )

    def test_convert_model_1d(self):
        """
        Check that the parameter constraints (Tied,Fixed and bounds)
        are correctly converted to the sherpa models.
        """

        def tiefunc(mo):
            return mo.stddev

        amodel = self.model1d.copy()
        amodel.amplitude.fixed = True
        amodel.stddev.max = 10.0
        amodel.mean.tied = tiefunc
        smodel = ConvertedModel(amodel).sherpa_model

        assert smodel.amplitude.frozen
        assert smodel.stddev.max == 10.0
        assert smodel.mean.link == smodel.stddev
        amodel.mean.value = amodel.stddev.value
        assert_allclose(smodel(self.x1), amodel(self.x1))

    def test_convert_model_2d(self):
        """
        Check that the parameter constraints (Tied,Fixed and bounds)
        are correctly converted to the sherpa models.
        """

        def tiefunc(mo):
            return mo.x_stddev

        amodel = self.model2d.copy()
        amodel.amplitude.fixed = True
        amodel.x_stddev.max = 10.0
        amodel.y_stddev.tied = tiefunc
        smodel = ConvertedModel(amodel).sherpa_model

        assert smodel.amplitude.frozen
        assert smodel.x_stddev.max == 10.0
        assert smodel.y_stddev.link == smodel.x_stddev

        amodel.y_stddev.value = amodel.x_stddev.value
        assert_allclose(smodel(self.xx1, self.xx2), amodel(self.xx1, self.xx2))

    def test_single_dataset_single_model(self):
        """Test a single model with a single dataset."""
        fmod = self.fitter(self.model1d.copy(), self.x1, self.y1, err=self.dy1)
        for pp in fmod.param_names:
            assert_allclose(getattr(fmod, pp), getattr(self.tmodel1d, pp), rtol=0.05)

    def test_single_dataset_two_models(self):
        """Test a two models with a single dataset."""
        fmod = self.fitter(
            [self.model1d.copy(), self.model1d.copy()], self.x1, self.y1, err=self.dy1
        )
        for ff in fmod:
            for nn, pp in enumerate(ff.param_names):
                assert_allclose(getattr(ff, pp), getattr(self.tmodel1d, pp), rtol=0.05)

    def test_two_dataset_single_model(self):
        """Tests two datasets with a single model."""
        fmod = self.fitter(
            self.model1d.copy(),
            [self.x1, self.x1],
            [self.y1, self.y1],
            err=[self.dy1, self.dy1],
        )
        for ff in fmod:
            for nn, pp in enumerate(ff.param_names):
                assert_allclose(getattr(ff, pp), getattr(self.tmodel1d, pp), rtol=0.05)

    def test_two_dataset_two_models(self):
        """Tests two models with two datasets simultaneously."""
        fmod = self.fitter(
            [self.model1d.copy(), self.model1d_2.copy()],
            [self.x1, self.x2],
            [self.y1, self.y2],
            err=[self.dy1, self.dy2],
        )
        for ff, mm in zip(fmod, [self.tmodel1d, self.tmodel1d_2]):
            for nn, pp in enumerate(ff.param_names):
                assert_allclose(getattr(ff, pp), getattr(mm, pp), rtol=0.05)

    def test_2d_fit(self):
        """
        Fits a 2d dataset.
        """
        fmod = self.fitter(
            self.model2d.copy(), self.xx1, self.xx2, self.yy, err=self.dyy
        )
        for pp in fmod.param_names:
            if getattr(self.tmodel2d, pp).fixed is False:
                assert_allclose(
                    getattr(fmod, pp), getattr(self.tmodel2d, pp), rtol=0.05
                )

    def test_condition_fit(self):
        """
        Some deliberatly bad fits to check that contraints are working!
        """

        def tiefunc(mo):
            return mo.stddev

        amodel = self.model1d.copy()
        amodel.amplitude.fixed = True
        amodel.stddev.max = 0.7
        amodel.stddev.value = 0.6
        amodel.stddev.min = 0.5
        amodel.mean.tied = tiefunc
        with warnings.catch_warnings():
            # Sherpa uses something which throws this warning
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            fmod = self.fitter(amodel, self.x1, self.y1, err=self.dy1)

        assert fmod.amplitude.value == self.model1d.amplitude.value
        assert round(fmod.stddev.value, 1) == 0.7 or round(fmod.stddev.value, 1) == 0.5
        assert fmod.mean.value == fmod.stddev.value

    def test_bkg_doesnt_explode(self):
        """
        Check this goes through the motions
        """

        m = Polynomial1D(2)

        x = np.arange(0, 10, 0.1)
        y = 2 + 0.5 * x + 3 * x ** 2
        bkg = x

        sfit = SherpaFitter(statistic="cash", estmethod="covariance")
        sfit(m, x, y, bkg=bkg)
        # TODO: Make this better!

    def test_entry_points(self):
        # a little to test that entry points can be loaded!
        from pkg_resources import iter_entry_points

        for entry_point in iter_entry_points(group="astropy.modeling", name=None):
            if entry_point.module_name == "saba":
                entry_point.load()


class TestMCMC(object):
    def setup_class(self):
        self.model = Polynomial1D(2)
        self.x = np.arange(0, 10, 0.1)

        self.params = (2, 0.5, 3)
        # a simple polynomial
        self.y = self.params[0]
        self.y += self.params[1] * self.x
        self.y += self.params[2] * self.x ** 2

        self.y += np.random.uniform(-0.1, 0.1, self.x.size)

        sfit = SherpaFitter(statistic="cash", estmethod="covariance")
        sfit(self.model, self.x, self.y)
        self.sampler = sfit.get_sampler()

    def test_run_samples(self, n=1000):
        """
        check to see sampler runs and
        returns the correct number of samples
        """
        stat_vals, param_vals, accepted = self.sampler(niter=n)
        assert len(stat_vals) == len(param_vals) == len(accepted[0]) == (n + 1)

    def test_check_fit(self, nbins=100):
        """
        Check that the parameters values found are reasonable
        """
        parameters = self.sampler.parameters
        for nn, pname in enumerate(("c0", "c1", "c2")):
            y, xx = np.histogram(parameters[pname][self.sampler.accepted], nbins)
            cdf = [y[0]]
            for yy in y[1:]:
                cdf.append(cdf[-1] + yy)
            cdf = np.array(cdf)
            cdf = cdf / float(cdf[-1])

            med_ind = np.argmin(abs(cdf - 0.5))
            x_med = (xx[med_ind] + xx[med_ind + 1]) / 2.0

            assert_allclose(self.params[nn], x_med, atol=0.1)
