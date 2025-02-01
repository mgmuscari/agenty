from typing import Union, Sequence, Type, TypedDict
from typing_extensions import TypeVar

from pydantic import BaseModel
from rich.json import JSON

__all__ = [
    "BaseIO",
    "AgentIO",
    "AgentIOBase",
    "AgentIOType",
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
    TypedDict,
    BaseIO,
]
"""Union type for basic agent I/O types.

This type represents the allowed primitive types and BaseIO models that can be
used for agent inputs and outputs.
"""

AgentIO = Union[AgentIOBase, Sequence[AgentIOBase]]
"""Union type for all valid agent I/O types.

This type extends AgentIOBase to include sequences of basic types, allowing for
both single values and collections in agent I/O.
"""

AgentIOType = Union[
    Type[bool],
    Type[int],
    Type[float],
    Type[str],
    Type[BaseIO],
    Type[Sequence[AgentIOBase]],
]
"""Union type for agent I/O type annotations.

This type represents the allowed types that can be used to annotate agent
inputs and outputs at a type level.
"""

AgentInputT = TypeVar(
    "AgentInputT",
    bound=AgentIO,
    default=str,
)
"""Type variable for agent input types.

This type variable is used for generic agent implementations to specify their
input type, bounded by AgentIO with a default of str.
"""

AgentOutputT = TypeVar(
    "AgentOutputT",
    bound=AgentIO,
    default=str,
)
"""Type variable for agent output types.

This type variable is used for generic agent implementations to specify their
output type, bounded by AgentIO with a default of str.
"""
