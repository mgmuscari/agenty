from typing import Generic, Type, Protocol, Optional, Dict, Any
from typing_extensions import TypeVar
from agenty.components.memory import AgentMemory
from agenty.types import AgentIO, AgentInputT, AgentOutputT, PipelineOutputT


__all__ = ["AgentProtocol", "AgentIOProtocol"]


class AgentIOProtocol(Generic[AgentInputT, AgentOutputT], Protocol):
    @property
    def input_schema(self) -> Type[AgentIO]: ...
    @property
    def output_schema(self) -> Type[AgentIO]: ...

    async def run(
        self,
        input_data: AgentInputT,
        name: Optional[str] = None,
    ) -> AgentOutputT: ...

    def __or__(
        self,
        other: "AgentIOProtocol[AgentOutputT, PipelineOutputT]",
    ) -> "AgentIOProtocol[AgentInputT, PipelineOutputT]": ...


class AgentProtocol(AgentIOProtocol[AgentInputT, AgentOutputT]):
    @property
    def name(self) -> str: ...

    @property
    def memory(self) -> AgentMemory: ...

    @memory.setter
    def memory(self, value: AgentMemory) -> None: ...
