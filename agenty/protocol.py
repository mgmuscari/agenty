from typing import Generic, Type, Protocol
from typing_extensions import TypeVar
from agenty.types import AgentIO, AgentInputT, AgentOutputT, PipelineOutputT


__all__ = ["AgentProtocol"]


class AgentProtocol(Generic[AgentInputT, AgentOutputT], Protocol):
    @property
    def input_schema(self) -> Type[AgentIO]: ...
    @property
    def output_schema(self) -> Type[AgentIO]: ...

    async def run(
        self,
        input_data: AgentInputT,
    ) -> AgentOutputT: ...

    def __or__(
        self,
        other: "AgentProtocol[AgentOutputT, PipelineOutputT]",
    ) -> "AgentProtocol[AgentInputT, PipelineOutputT]": ...
