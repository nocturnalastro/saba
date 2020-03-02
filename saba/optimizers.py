from sherpa.optmethods import GridSearch, LevMar, MonCar, NelderMead
from .sherpa_wrapper import SherpaWrapper


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
