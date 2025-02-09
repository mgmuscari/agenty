"""Protocol definitions for agent interfaces in the agenty framework.

This module defines the core protocols that all agents must implement. It provides
two main protocols:

- AgentIOProtocol: Defines the basic input/output interface for agents
- AgentProtocol: Extends AgentIOProtocol with additional agent-specific functionality

These protocols ensure consistent behavior across different agent implementations
and enable composition of agents in pipelines.
"""

from typing import Generic, Type, Protocol, Optional
from agenty.agent.chat_history import ChatHistory
from agenty.types import AgentIO, AgentInputT, AgentOutputT, PipelineOutputT


__all__ = ["AgentProtocol", "AgentIOProtocol"]


class AgentIOProtocol(Generic[AgentInputT, AgentOutputT], Protocol):
    """Protocol defining the basic input/output interface for agents.

    This protocol establishes the core contract that agents must fulfill to process
    input data and produce output. It supports both synchronous and asynchronous
    execution, and enables pipeline composition through the | operator.

    Type Parameters:
        AgentInputT: The type of input data the agent accepts
        AgentOutputT: The type of output data the agent produces
    """

    @property
    def input_schema(self) -> Type[AgentIO]:
        """Get the input schema type for this agent.

        Returns:
            Type[AgentIO]: The Pydantic model class defining the expected input structure
        """
        ...

    @property
    def output_schema(self) -> Type[AgentIO]:
        """Get the output schema type for this agent.

        Returns:
            Type[AgentIO]: The Pydantic model class defining the produced output structure
        """
        ...

    async def run(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT:
        """Execute the agent asynchronously with the given input data.

        Args:
            input_data: The input data to process, must conform to input_schema
            name: Optional name for this execution context

        Returns:
            The processed output data conforming to output_schema
        """
        ...

    def run_sync(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT:
        """Execute the agent synchronously with the given input data.

        Args:
            input_data: The input data to process, must conform to input_schema
            name: Optional name for this execution context

        Returns:
            The processed output data conforming to output_schema
        """
        ...

    def reset(self) -> None:
        """Reset the agent's internal state.

        This method should clear any cached data or state that could affect
        future executions.
        """
        ...

    def __or__(
        self,
        other: "AgentIOProtocol[AgentOutputT, PipelineOutputT]",
    ) -> "AgentIOProtocol[AgentInputT, PipelineOutputT]":
        """Compose this agent with another agent to form a pipeline.

        This operator enables functional composition of agents, where the output
        of this agent becomes the input to the other agent.

        Args:
            other: Another agent whose input type matches this agent's output type

        Returns:
            A new agent representing the composed pipeline
        """
        ...


class AgentProtocol(AgentIOProtocol[AgentInputT, AgentOutputT], Protocol):
    """Protocol extending AgentIOProtocol with agent-specific functionality.

    This protocol adds properties for agent identification and chat history
    management on top of the basic input/output operations defined in AgentIOProtocol.

    Type Parameters:
        AgentInputT: The type of input data the agent accepts
        AgentOutputT: The type of output data the agent produces
    """

    @property
    def name(self) -> str:
        """Get the name of this agent instance.

        Returns:
            str: The unique identifier for this agent
        """
        ...

    @property
    def chat_history(self) -> ChatHistory:
        """Get the chat history for this agent.

        Returns:
            ChatHistory: The object tracking all interactions with this agent
        """
        ...

    @chat_history.setter
    def chat_history(self, value: ChatHistory) -> None:
        """Set the chat history for this agent.

        Args:
            value: The new chat history object to use
        """
        ...
