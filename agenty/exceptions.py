class AgentyException(Exception):
    """Base exception class for all exceptions raised by Agenty.

    All custom exceptions in the Agenty library inherit from this class,
    allowing users to catch any Agenty-specific exception using this base class.
    """

    pass


# Value and Type Exceptions


class AgentyValueError(AgentyException, ValueError):
    """Exception raised when an invalid value is provided to an Agenty operation.

    This exception combines AgentyException with ValueError to indicate that
    a value provided to an Agenty operation was invalid while maintaining
    the Agenty exception hierarchy.
    """

    pass


class AgentyTypeError(AgentyException, TypeError):
    """Exception raised when an incorrect type is used in an Agenty operation.

    This exception combines AgentyException with TypeError to indicate that
    a value of an incorrect type was provided while maintaining the Agenty
    exception hierarchy.
    """

    pass


class UnsupportedModel(AgentyValueError):
    """Exception raised when attempting to use an unsupported model.

    This exception indicates that the specified language model or configuration
    is not supported by the Agenty library or the current operation.
    """

    pass


# Response Exceptions


class InvalidResponse(AgentyException):
    """Exception raised when receiving an invalid or unexpected response.

    This exception indicates that the response received (typically from a model
    or external service) does not meet the expected format or requirements.
    """

    pass


# Pipeline Exceptions


class PipelineException(AgentyException):
    """Base exception class for pipeline-related errors.

    This exception serves as the parent class for all exceptions that occur
    during pipeline operations, allowing users to catch any pipeline-specific
    errors.
    """

    pass


class InvalidStepResponse(PipelineException, InvalidResponse):
    """Exception raised when a pipeline step provides and invalid response.

    This exception indicates that a specific step in the pipeline encountered
    an error during execution, providing more specific error handling for
    pipeline operations.
    """

    pass
