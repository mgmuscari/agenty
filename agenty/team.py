from typing import Any, Dict, Generic, List, Optional, Type, cast, Sequence, Tuple

from agenty.agent import Agent
from agenty.exceptions import AgentyValueError, MaxTurnsExceeded
from agenty.types import (
    AgentIO,
    AgentInputT,
    AgentOutputT,
    PipelineOutputT,
    BaseIO,
    NOT_GIVEN,
    NOT_GIVEN_,
    normalize_type,
)
from agenty.pipeline import Pipeline
from agenty.protocol import AgentProtocol, AgentIOProtocol
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty


from abc import ABC


class StopCondition(BaseIO, ABC):
    def __call__(
        self,
        team: AgentIOProtocol[Any, Any],
        speaker: AgentProtocol[Any, Any],
        output: AgentIO,
    ) -> Optional[str]:
        raise NotImplementedError

    def __or__(self, other: "StopCondition") -> "OrStop":
        return OrStop(conditions=[self, other])

    def __and__(self, other: "StopCondition") -> "AndStop":
        return AndStop(conditions=[self, other])


class AndStop(StopCondition):
    conditions: List[StopCondition]

    def __call__(
        self,
        team: AgentIOProtocol[Any, Any],
        speaker: AgentProtocol[Any, Any],
        output: AgentIO,
    ) -> Optional[str]:
        stop_messages: List[str] = []
        for condition in self.conditions:
            result = condition(team, speaker, output)
            if result is None:
                return None
            stop_messages.append(result)
        return " & ".join(stop_messages)


class OrStop(StopCondition):
    conditions: List[StopCondition]

    def __call__(
        self,
        team: AgentIOProtocol[Any, Any],
        speaker: AgentProtocol[Any, Any],
        output: AgentIO,
    ) -> Optional[str]:
        for condition in self.conditions:
            result = condition(team, speaker, output)
            if result is not None:
                return result
        return None


class TextMentionStop(StopCondition):
    # stops on the first output that is a string that contains the text
    text: str = "<TERMINATE>"

    def __call__(
        self,
        team: AgentIOProtocol[Any, Any],
        speaker: AgentProtocol[Any, Any],
        output: AgentIO,
    ) -> Optional[str]:
        if not isinstance(output, str):
            return None
        return "TextMention" if self.text in output else None


class OutputSchemaStop(StopCondition):
    # stops on the first output that matches the response type

    def __call__(
        self,
        team: AgentIOProtocol[Any, Any],
        speaker: AgentProtocol[Any, Any],
        output: AgentIO,
    ) -> Optional[str]:
        output_schema = normalize_type(speaker.output_schema)
        team_output_schema = normalize_type(team.output_schema)
        if output_schema == team_output_schema:
            return "ResponseType"
        return None


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
    max_turns: int = 10
    # the maximum number of turns a team can run with before stopping.
    # This is a safety feature that raises an exception if the team runs for too long.
    # This is different from the stop_condition which is a user-defined condition that
    # can stop the team at any time and does not raise an exception.

    def __init__(
        self,
        agents: Sequence[AgentProtocol[Any, Any]],
        input_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
        output_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
        stop_condition: StopCondition = TextMentionStop(),
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
        self.subscriptions: Dict[Tuple, List[AgentProtocol]] = {}
        self.manager: TeamManager = RoundRobin()
        self.stop_condition = stop_condition

        for agent in agents:
            if agent.name in self.agents:
                raise AgentyValueError(f"Agent '{agent.name}' already exists in team")
            self.agents[agent.name] = agent
            input_tuple = normalize_type(agent.input_schema)
            output_tuple = normalize_type(agent.output_schema)
            self.subscriptions[input_tuple] = self.subscriptions.get(input_tuple, [])
            self.subscriptions[output_tuple] = self.subscriptions.get(output_tuple, [])
            self.subscriptions[input_tuple].append(agent)

        self._console = Console()

    def _handle_subscriptions(
        self,
        speaker_name: Optional[str],
        output: AgentIO,
        output_schema: Type[AgentIO],
    ) -> None:
        # publish output to all subscribers
        subbed_agents = self.subscriptions.get(normalize_type(output_schema), [])
        count = 0
        for agent in subbed_agents:
            if agent.name != speaker_name:
                agent.memory.add(
                    "user",
                    output,
                    name=speaker_name,
                    inject_name=True,
                )
                count += 1
            # self._console.print(
            #     Panel(
            #         agent.memory,
            #         title=f"AgentMemory: {agent.name}",
            #     )
            # )
        if count == 0 and normalize_type(self.output_schema) != normalize_type(
            output_schema
        ):
            # we can ignore having no relevant subscribers if the output schema of the team matches the output schema
            raise AgentyValueError(f"No agents subscribed to output: {output_schema}")

    async def run(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT:
        output: Any = None
        if input_data is None:
            raise AgentyValueError("Team input data cannot be None")

        self._handle_subscriptions(None, input_data, self.input_schema)

        stop_message = ""
        for i in range(self.max_turns):
            speaker_name = await self.manager.run(TeamInput(team=self))
            if speaker_name == "":
                break
            if speaker_name not in self.agents:
                raise AgentyValueError(f"Agent '{speaker_name}' not found in team")
            speaker = self.agents[speaker_name]
            output = await speaker.run(None)

            # self._console.print(
            #     Panel(
            #         Pretty(output),
            #         title=f"Agent: {speaker_name}",
            #     )
            # )
            stop_message = self.stop_condition(self, speaker, output)
            if stop_message is not None:
                break
            self._handle_subscriptions(speaker_name, output, speaker.output_schema)
        else:
            raise MaxTurnsExceeded
        # handle stop_message (?)
        return cast(AgentOutputT, output)

    def __or__(
        self, other: AgentIOProtocol[AgentOutputT, PipelineOutputT]
    ) -> AgentIOProtocol[AgentInputT, PipelineOutputT]:
        return Pipeline[AgentInputT, PipelineOutputT](
            agents=[self, other],
            input_schema=self.input_schema,
            output_schema=other.output_schema,
        )


class TeamInput(BaseIO):
    model_config = {
        "arbitrary_types_allowed": True,
    }
    team: Team[Any, Any]


class TeamManager(Agent[TeamInput, str]):
    input_schema = TeamInput
    output_schema = str

    async def run(
        self,
        input_data: Optional[TeamInput],
        name: Optional[str] = None,
    ) -> str:
        raise NotImplementedError


class RoundRobin(TeamManager):
    def __init__(self) -> None:
        self.prev_speaker: Optional[str] = None

    async def run(
        self,
        input_data: Optional[TeamInput],
        name: Optional[str] = None,
    ) -> str:
        if input_data is None:
            raise AgentyValueError("TeamManager input data cannot be None")
        team = input_data.team
        if self.prev_speaker is None:
            self.prev_speaker = list(team.agents.keys())[0]
            return self.prev_speaker
        idx = list(team.agents.keys()).index(self.prev_speaker)
        next_idx = (idx + 1) % len(team.agents)
        self.prev_speaker = list(team.agents.keys())[next_idx]
        return self.prev_speaker


async def _main_test() -> None:
    import os
    from pydantic_ai.models.openai import OpenAIModel

    class Pizza(BaseIO):
        toppings: List[str]
        crust: str

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
    OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")
    model = OpenAIModel(
        "llama3x-8b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    agent1 = Agent(
        model=model,
        name="agent1",
        system_prompt="Create a pizza.",
        output_schema=Pizza,
    )
    agent2 = Agent(
        model=model,
        name="agent2",
        system_prompt="Rate the pizza.",
        input_schema=Pizza,
    )
    agent3 = Agent(
        model=model,
        name="agent3",
        system_prompt="You're always angry.",
    )
    team = Team(agents=[agent1, agent2, agent3])
    await team.run("I love sauaage and pineapple.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main_test())
