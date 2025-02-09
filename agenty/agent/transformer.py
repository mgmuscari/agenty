from .base import Agent
from typing import Optional

from agenty.types import (
    AgentInputT,
    AgentOutputT,
)


class Transformer(Agent[AgentInputT, AgentOutputT]):
    async def transform(
        self,
        input_data: Optional[AgentInputT],
    ) -> AgentOutputT:
        raise NotImplementedError("Subclasses must implement this method")

    async def run(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT:
        return await self.transform(input_data)
