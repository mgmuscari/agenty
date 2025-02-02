from typing import TypedDict
from unittest.mock import AsyncMock, Mock

import pytest
import pydantic_ai as pai
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import RunResult

from agenty import Agent
from agenty.exceptions import AgentyValueError
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
async def test_agent_out_dict():
    class TestDict(TypedDict):
        a: int
        b: str
        c: bool

    agent = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args={"a": 1, "b": "test", "c": "True"},
        ),
        input_schema=str,
        output_schema=TestDict,
    )
    resp = await agent.run("test")
    assert resp == {"a": 1, "b": "test", "c": True}

    with pytest.raises(pai.exceptions.UnexpectedModelBehavior):
        agent = Agent(
            model=TestModel(
                call_tools=[],
                custom_result_args=({"a": 1, "b": "test", "c": "True", "d": "extra"},),
            ),
            input_schema=str,
            output_schema=TestDict,
        )
        resp = await agent.run("test")


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
