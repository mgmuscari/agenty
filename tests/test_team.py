import pytest
from unittest.mock import AsyncMock
from pydantic_ai.models.test import TestModel

from agenty import Agent
from agenty.exceptions import AgentyValueError
from agenty.team import Team
from agenty.types import BaseIO


class TestInput(BaseIO):
    message: str


class TestOutput(BaseIO):
    response: str


@pytest.mark.asyncio
async def test_team_initialization():
    agent1 = Agent(
        model=TestModel(),
        input_schema=TestInput,
        output_schema=TestOutput,
        name="agent1",
    )
    agent2 = Agent(
        model=TestModel(),
        input_schema=TestInput,
        output_schema=TestOutput,
        name="agent2",
    )

    team = Team(
        agents={"agent1": agent1, "agent2": agent2},
        input_schema=TestInput,
        output_schema=TestOutput,
    )

    assert len(team.agents) == 2
    assert team.get_agent("agent1") == agent1
    assert team.get_agent("agent2") == agent2
    assert team.input_schema == TestInput
    assert team.output_schema == TestOutput


@pytest.mark.asyncio
async def test_team_run_agent():
    agent1 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"response": "Hello from agent1"},
        ),
        input_schema=TestInput,
        output_schema=TestOutput,
    )

    team = Team(agents={"agent1": agent1})
    result = await team.run(
        TestInput(message="Hello"),
        agent_name="agent1",
    )

    assert isinstance(result, TestOutput)
    assert result.response == "Hello from agent1"

    # Test running non-existent agent
    with pytest.raises(AgentyValueError):
        await team.run(
            TestInput(message="Hello"),
            agent_name="non_existent",
        )


@pytest.mark.asyncio
async def test_team_broadcast():
    agent1 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"response": "Response from agent1"},
        ),
        input_schema=TestInput,
        output_schema=TestOutput,
        name="agent1",
    )
    agent2 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"response": "Response from agent2"},
        ),
        input_schema=TestInput,
        output_schema=TestOutput,
        name="agent2",
    )

    team = Team(agents={"agent1": agent1, "agent2": agent2})

    # Test broadcast to all agents
    responses = await team.broadcast(
        TestInput(message="Broadcast message"),
        from_agent="agent1",
    )

    assert len(responses) == 1  # Should only include agent2's response
    assert isinstance(responses["agent2"], TestOutput)
    assert responses["agent2"].response == "Response from agent2"

    # Test broadcast to specific agents
    responses = await team.broadcast(
        TestInput(message="Targeted message"),
        from_agent="agent1",
        to_agents=["agent2"],
    )

    assert len(responses) == 1
    assert responses["agent2"].response == "Response from agent2"

    # Test broadcast from non-existent agent
    with pytest.raises(AgentyValueError):
        await team.broadcast(
            TestInput(message="Invalid broadcast"),
            from_agent="non_existent",
        )


@pytest.mark.asyncio
async def test_team_shared_memory():
    agent1 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"response": "First response"},
        ),
        input_schema=TestInput,
        output_schema=TestOutput,
    )
    agent2 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"response": "Second response"},
        ),
        input_schema=TestInput,
        output_schema=TestOutput,
    )

    team = Team(agents={"agent1": agent1, "agent2": agent2})

    # Run agents sequentially
    await team.run(TestInput(message="First message"), "agent1")
    await team.run(TestInput(message="Second message"), "agent2")

    # Check conversation history
    history = team.get_conversation_history()
    assert len(history) == 4  # 2 user messages + 2 assistant responses

    # Verify memory is shared between agents
    assert agent1.memory == team.memory
    assert agent2.memory == team.memory


@pytest.mark.asyncio
async def test_team_agent_names():
    # Test automatic name assignment
    unnamed_agent = Agent(
        model=TestModel(),
        input_schema=TestInput,
        output_schema=TestOutput,
    )

    team = Team(agents={"custom_name": unnamed_agent})
    assert unnamed_agent.name == "custom_name"

    # Test preserving existing names
    named_agent = Agent(
        model=TestModel(),
        input_schema=TestInput,
        output_schema=TestOutput,
        name="original_name",
    )

    team = Team(agents={"new_name": named_agent})
    assert named_agent.name == "original_name"
