class AgentyException(Exception):
    pass


class InvalidResponse(AgentyException):
    pass


class AgentyValueError(AgentyException, ValueError):
    pass


class AgentyTypeError(AgentyException, TypeError):
    pass


class UnsupportedModel(AgentyValueError):
    pass
