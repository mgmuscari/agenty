from typing import Union, TypedDict, Sequence, Any, Optional
from typing_extensions import TypeVar
from typing_inspect import get_args

from pydantic import BaseModel
from rich.json import JSON

__all__ = [
    "BaseIO",
    "AgentIO",
    "AgentIOBase",
    "AgentInputT",
    "AgentOutputT",
]


class BaseIO(BaseModel):
    """Base class for all agent input/output models.

    This class extends Pydantic's BaseModel to represent basic IO between agents. All structured
    input/output models should inherit from this class.
    """

    def __str__(self) -> str:
        """Convert the model to a JSON string.

        Returns:
            str: JSON string representation of the model
        """
        return self.model_dump_json()

    def __rich__(self) -> JSON:
        """Create a rich console representation of the model.

        Returns:
            JSON: Rich-formatted JSON representation
        """
        json_str = self.model_dump_json()
        return JSON(json_str)


AgentIOBase = Union[
    bool,
    int,
    float,
    str,
    BaseIO,
]
"""Union type for basic agent I/O types.

This type represents the allowed primitive types and BaseIO models that can be
used for agent inputs and outputs.
"""

AgentIO = Union[AgentIOBase, Sequence[AgentIOBase]]
"""All supported data types for agent communication.

Extends the core types (AgentIOBase) to also support sequences/lists of those types.
"""

AgentInputT = TypeVar(
    "AgentInputT",
    bound=AgentIO,
    default=str,
)
"""Type variable for agent input types.

This type variable is used for generic agent implementations to specify their input schema
"""

AgentOutputT = TypeVar(
    "AgentOutputT",
    bound=AgentIO,
    default=str,
)
"""Type variable for agent output types.

This type variable is used for generic agent implementations to specify their output schema
"""

PipelineOutputT = TypeVar(
    "PipelineOutputT",
    bound=AgentIO,
    default=str,
)


class NOT_GIVEN:
    """Sentinel class used to distinguish between unset and None values."""

    ...


NOT_GIVEN_ = NOT_GIVEN()


def is_sequence_type(output_type: Any) -> bool:
    """Check if the given output type is a sequence (list or tuple)."""
    from typing_inspect import get_origin

    return output_type in (list, tuple) or get_origin(output_type) in (list, tuple)


def get_sequence_item_type(output_type: Any) -> Optional[Any]:
    """Get the item type of a sequence type."""
    if not is_sequence_type(output_type):
        return None
    try:
        return get_args(output_type)[0]
    except IndexError:
        return None


def matches_schema(data: Any, schema: type) -> bool:
    """Check if the given value matches the expected schema type. Supports sequence types.

    Args:
        data: The data to validate
        schema: The expected schema type

    Raises:
        AgentyTypeError: If data type doesn't match the schema
    """
    check_type: Any = type(data)
    check_schema: Any = schema

    if is_sequence_type(check_type) and is_sequence_type(schema):
        check_schema = get_sequence_item_type(schema)
        try:
            check_type = type(data[0])
        except IndexError:
            # list is empty so technically type is correct
            check_type = check_schema

    return check_type is check_schema
