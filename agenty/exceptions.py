class AgentyException(Exception):
    pass


class InvalidResponse(AgentyException):
    pass


class AgentyValueError(AgentyException, ValueError):
    pass


class UnsupportedModel(AgentyValueError):
    pass
