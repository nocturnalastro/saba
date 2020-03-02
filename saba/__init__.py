# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Bridge between Sherpa and Astropy modeling.
"""

# Affiliated packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *

# ----------------------------------------------------------------------------

import sys

__minimum_python_version__ = "3.5"


class UnsupportedPythonError(Exception):
    pass


if sys.version_info < tuple(
    (int(val) for val in __minimum_python_version__.split("."))
):
    raise UnsupportedPythonError(
        "packagename does not support Python < {}".format(__minimum_python_version__)
    )


# For egg_info test builds to pass, put package imports here.
if not _ASTROPY_SETUP_:
    from .main import (
        SherpaFitter,
        SherpaMCMC,
        Stat,
        OptMethod,
        EstMethod,
        Dataset,
        ConvertedModel,
    )
