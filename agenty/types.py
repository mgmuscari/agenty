from typing import Union, List, Tuple, TypedDict
from typing_extensions import TypeVar

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
    TypedDict,
    BaseIO,
]
"""Union type for basic agent I/O types.

This type represents the allowed primitive types and BaseIO models that can be
used for agent inputs and outputs.
"""

AgentIO = Union[AgentIOBase, List[AgentIOBase], Tuple[AgentIOBase]]
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

AgentTeamT = TypeVar(
    "AgentTeamT",
    bound=AgentIO,
    default=str,
)
"""This type variable is used for team-based agent communication.
"""
