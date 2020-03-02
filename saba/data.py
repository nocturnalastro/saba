import numpy as np
from sherpa.data import Data, Data1D, Data1DInt, Data2D, Data2DInt, DataSimulFit

from .sherpa_wrapper import SherpaWrapper


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
