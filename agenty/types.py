from typing import Union, Sequence, Type
from typing_extensions import TypeVar

from pydantic import BaseModel
from rich.json import JSON


class BaseIO(BaseModel):

    def __str__(self):
        return self.model_dump_json()

    def __rich__(self):
        json_str = self.model_dump_json()
        return JSON(json_str)

    def __hash__(self):
        return hash(tuple(sorted(self.model_fields.keys())))


AgentIOBase = Union[
    bool,
    int,
    float,
    str,
    BaseIO,
]
AgentIO = Union[AgentIOBase, Sequence[AgentIOBase]]

AgentIOType = Union[
    Type[bool],
    Type[int],
    Type[float],
    Type[str],
    Type[BaseIO],
    Type[Sequence[AgentIOBase]],
]

AgentInputT = TypeVar(
    "AgentInputT",
    bound=AgentIO,
    default=str,
    # covariant=True,
)

AgentOutputT = TypeVar(
    "AgentOutputT",
    bound=AgentIO,
    default=str,
)
