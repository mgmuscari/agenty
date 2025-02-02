class AgentyException(Exception):
    pass


class AgentyValueError(AgentyException, ValueError):
    pass
