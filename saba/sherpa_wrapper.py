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
