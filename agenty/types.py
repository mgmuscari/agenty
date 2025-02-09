from typing import Any, Literal, Sequence, Tuple, Type, Union, get_origin, get_args
from typing_extensions import TypeVar

from pydantic import BaseModel

from rich.json import JSON


class NotGiven:
    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class BaseIO(BaseModel):
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

AgentIO = Union[AgentIOBase, Sequence[AgentIOBase]]

AgentInputT = TypeVar(
    "AgentInputT",
    bound=AgentIO,
    default=str,
    contravariant=True,
)

AgentOutputT = TypeVar(
    "AgentOutputT",
    bound=AgentIO,
    default=str,
)

PipelineOutputT = TypeVar(
    "PipelineOutputT",
    bound=AgentIO,
    default=str,
)


def normalize(type_: Any) -> Tuple[Type[Any], Tuple[Any, ...]]:
    """Normalize a type annotation to a standard form."""
    # If there's no origin, return the type itself
    origin = get_origin(type_) or type_
    args = get_args(type_)
    return (origin, args)


def matches(type1: Any, type2: Any) -> bool:
    """Check if two types match."""
    return normalize(type1) == normalize(type2)
