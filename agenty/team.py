from typing import Any, Dict, Generic, List, Optional, Type, cast, Sequence

from agenty.agent import Agent
from agenty.components.memory import AgentMemory, ChatMessage
from agenty.exceptions import AgentyValueError
from agenty.types import AgentIO, AgentInputT, AgentOutputT, NOT_GIVEN, NOT_GIVEN_
from agenty.protocol import AgentProtocol


class Team(Generic[AgentInputT, AgentOutputT]):
    """A team of agents that can communicate with each other.

    The Team class enables creation of agent groups that share context and can interact
    with each other through a shared memory system. Each agent in the team can access
    the team's memory and communicate with other agents.

    Type Parameters:
        AgentInputT: The type of input the team accepts
        AgentOutputT: The type of output the team produces

    Attributes:
        input_schema: The expected schema for team input data
        output_schema: The expected schema for team output data
        agents: Dictionary of named agents in the team
        memory: Shared memory system for inter-agent communication
    """

    input_schema: Type[AgentIO] = str
    output_schema: Type[AgentIO] = str

    def __init__(
        self,
        agents: Sequence[AgentProtocol[Any, Any]],
        input_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
        output_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
    ) -> None:
        """Initialize a new Team instance.

        Args:
            agents: Dictionary of named agents that form the team
            input_schema: Optional input schema override
            output_schema: Optional output schema override
        """
        if not isinstance(input_schema, NOT_GIVEN):
            self.input_schema = input_schema
        if not isinstance(output_schema, NOT_GIVEN):
            self.output_schema = output_schema

        if not agents:
            raise AgentyValueError("Team must have at least one agent")

        self.agents: Dict[str, AgentProtocol[Any, Any]] = {}
        self.shared_memory = AgentMemory()

        self._last_speaker: Optional[str] = None

        for agent in agents:
            if agent.name in self.agents:
                raise AgentyValueError(f"Agent '{agent.name}' already exists in team")
            self.agents[agent.name] = agent
            agent.memory = self.shared_memory

    async def next_speaker(self) -> str:
        if self._last_speaker is None:
            # Start with the first agent in the team
            self._last_speaker = list(self.agents.keys())[0]
        return self._last_speaker

    async def run(
        self,
        input_data: AgentInputT,
    ) -> AgentOutputT:
        next_input = input_data

        i = 0
        while i < 5:
            speaker_name = await self.next_speaker()
            if speaker_name not in self.agents:
                raise AgentyValueError(f"Agent '{speaker_name}' not found in team")
            speaker = self.agents[speaker_name]
            output = await speaker.run(next_input)
            next_input = output
            i += 1

        # # Add input to memory if it's a new conversation
        # if self._last_speaker != agent_name:
        #     self.memory.add("user", input_data)

        # Run the agent with string conversion
        # self._last_speaker = agent_name

        return cast(AgentOutputT, output)

    async def broadcast(
        self,
        message: Any,
        from_agent: str,
        to_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Broadcast a message from one agent to others in the team.

        Args:
            message: The message to broadcast
            from_agent: Name of the sending agent
            to_agents: Optional list of recipient agent names. If None, broadcasts to all agents except sender.

        Returns:
            Dictionary mapping agent names to their responses

        Raises:
            AgentyValueError: If any specified agent is not found in the team
        """
        if from_agent not in self.agents:
            raise AgentyValueError(f"Agent '{from_agent}' not found in team")

        # Determine recipient agents
        if to_agents is None:
            to_agents = [name for name in self.agents.keys() if name != from_agent]
        else:
            for agent in to_agents:
                if agent not in self.agents:
                    raise AgentyValueError(f"Agent '{agent}' not found in team")

        # Collect responses from recipients
        responses = {}
        for agent_name in to_agents:
            output = await self.agents[agent_name].run(str(message))
            responses[agent_name] = output

        return responses

    def get_conversation_history(self) -> List[ChatMessage]:
        """Get the full conversation history from the team's shared memory.

        Returns:
            List of ChatMessage objects representing the conversation history
        """
        return list(self.memory)
