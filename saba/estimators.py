from sherpa.estmethods import Confidence, Covariance, Projection
from .sherpa_wrapper import SherpaWrapper


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
