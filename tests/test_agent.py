from unittest.mock import AsyncMock, Mock

import pytest
import pydantic_ai as pai
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import RunResult
from pydantic_ai.models import ModelSettings

from agenty import Agent
from agenty.exceptions import AgentyValueError, UnsupportedModel
from agenty.types import BaseIO


@pytest.mark.asyncio
async def test_agent_out_str():
    agent = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_text="success",
        ),
        input_schema=str,
        output_schema=str,
    )
    resp = await agent.run("test")
    assert resp == "success"


@pytest.mark.asyncio
async def test_agent_out_list():
    agent = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=("1", "2", 3),  # note these are not all int
        ),
        input_schema=str,
        output_schema=list[int],
    )
    resp = await agent.run("test")
    assert resp == [1, 2, 3]


@pytest.mark.asyncio
async def test_agent_out_baseio():
    class TestIO(BaseIO):
        a: int
        b: str
        c: bool

    agent = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"a": 2, "b": "agenty", "c": "False"},
        ),
        input_schema=str,
        output_schema=TestIO,
    )
    resp = await agent.run("test")
    assert resp != {"a": 2, "b": "agenty", "c": False}
    assert resp == TestIO(a=2, b="agenty", c=False)

    with pytest.raises(pai.exceptions.UnexpectedModelBehavior):
        agent = Agent(
            model=TestModel(
                call_tools=[],
                custom_result_args=({"a": 1, "b": "test", "c": "True", "d": "extra"},),
            ),
            input_schema=str,
            output_schema=TestIO,
        )
        resp = await agent.run("test")


@pytest.mark.asyncio
async def test_agent_out_none():
    agent = Agent(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
    )
    mock_result = Mock(spec=RunResult)
    mock_result.data = None
    # Test that the agent raises the correct error when the model returns None data (perhaps unable to parse a response)

    agent.pai_agent.run = AsyncMock(return_value=mock_result)
    with pytest.raises(AgentyValueError):
        await agent.run("test")


@pytest.mark.asyncio
async def test_agent_model_settings():
    agent = Agent(
        model=TestModel(),
        model_settings=ModelSettings(
            temperature=0.79,
            top_p=0.99,
        ),
        input_schema=str,
        output_schema=str,
    )

    assert agent.model_settings is not None
    assert isinstance(agent.model_settings, dict)
    assert agent.model_settings.get("temperature") == 0.79
    assert agent.model_settings.get("top_p") == 0.99


@pytest.mark.asyncio
async def test_agent_template_context():
    class TestAgent(Agent):
        TEST_ID: str

    agent = TestAgent(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
        system_prompt="You are a helpful assistant with ID {{ TEST_ID }}",
    )

    agent.TEST_ID = "test-123"
    rendered = agent.render_system_prompt()
    assert rendered == "You are a helpful assistant with ID test-123"
    assert agent.template_context() == {"TEST_ID": "test-123"}


@pytest.mark.asyncio
async def test_agent_pipeline():
    agent1 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_text="intermediate",
        ),
        input_schema=str,
        output_schema=str,
    )

    agent2 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_text="final",
        ),
        input_schema=str,
        output_schema=str,
    )

    pipeline = agent1 | agent2
    result = await pipeline.run("test input")
    assert result == "final"


@pytest.mark.asyncio
async def test_agent_configuration():
    agent = Agent(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
        retries=3,
        result_retries=2,
        end_strategy="early",
        name="TestAgent",
    )

    assert agent.retries == 3
    assert agent.result_retries == 2
    assert agent.end_strategy == "early"
    assert agent.name == "TestAgent"


@pytest.mark.asyncio
async def test_agent_model_none():
    agent = Agent(
        model=None,
        input_schema=str,
        output_schema=str,
    )

    with pytest.raises(UnsupportedModel):
        await agent.run("test")


@pytest.mark.asyncio
async def test_agent_schema_getters():
    class CustomInput(BaseIO):
        field: str

    class CustomOutput(BaseIO):
        result: int

    agent = Agent(
        model=TestModel(),
        input_schema=CustomInput,
        output_schema=CustomOutput,
    )

    input_schema = agent.get_input_schema()
    output_schema = agent.get_output_schema()

    assert input_schema == CustomInput
    assert output_schema == CustomOutput
