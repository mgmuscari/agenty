from typing import Generic, Type, Protocol, Optional
from agenty.agent.chat_history import ChatHistory
from agenty.types import AgentIO, AgentInputT, AgentOutputT, PipelineOutputT


__all__ = ["AgentProtocol", "AgentIOProtocol"]


class AgentIOProtocol(Generic[AgentInputT, AgentOutputT], Protocol):
    @property
    def input_schema(self) -> Type[AgentIO]: ...
    @property
    def output_schema(self) -> Type[AgentIO]: ...

    async def run(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT: ...

    def reset(self) -> None: ...

    def __or__(
        self,
        other: "AgentIOProtocol[AgentOutputT, PipelineOutputT]",
    ) -> "AgentIOProtocol[AgentInputT, PipelineOutputT]": ...


class AgentProtocol(AgentIOProtocol[AgentInputT, AgentOutputT], Protocol):
    @property
    def input_schema(self) -> Type[AgentIO]: ...
    @property
    def output_schema(self) -> Type[AgentIO]: ...

    @property
    def name(self) -> str: ...

    @property
    def chat_history(self) -> ChatHistory: ...

    @chat_history.setter
    def chat_history(self, value: ChatHistory) -> None: ...
