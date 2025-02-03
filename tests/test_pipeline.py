from typing import TypedDict
from unittest.mock import AsyncMock, Mock

import pytest
import pydantic_ai as pai
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import RunResult

from agenty import Agent
from agenty.exceptions import AgentyTypeError
from agenty.types import BaseIO


@pytest.mark.asyncio
async def test_pipeline_list_users():
    class User(BaseIO):
        first_name: str
        last_name: str

    from typing import List

    agent1 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=[
                {"first_name": "John", "last_name": "Doe"},
                {"first_name": "Agenty", "last_name": "Rocks"},
            ],
        ),
        input_schema=str,
        output_schema=List[User],
    )
    agent2 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=[
                "John Doe",
                "Agenty Rocks",
            ],
        ),
        input_schema=List[User],
        output_schema=List[str],
    )
    resp = await (agent1 | agent2).run("test")
    assert resp == ["John Doe", "Agenty Rocks"]


@pytest.mark.asyncio
async def test_pipeline_list_users_error():
    class User(BaseIO):
        first_name: str
        last_name: str

    from typing import List

    agent1 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=[
                {"first_name": "John", "last_name": "Doe"},
                {"first_name": "Agenty", "last_name": "Rocks"},
            ],
        ),
        input_schema=str,
        output_schema=List[User],
    )
    agent2 = Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=[
                "John Doe",
                "Agenty Rocks",
            ],
        ),
        input_schema=str,
        output_schema=List[str],
    )
    with pytest.raises(AgentyTypeError):
        await (agent1 | agent2).run("test")


@pytest.mark.asyncio
async def test_multi_pipeline():
    from typing import List

    agent1 = Agent[str, int](
        model=TestModel(
            call_tools=[],
            custom_result_args=1,
        ),
        input_schema=str,
        output_schema=int,
    )
    agent2 = Agent[int, float](
        model=TestModel(
            call_tools=[],
            custom_result_args=1.0,
        ),
        input_schema=int,
        output_schema=float,
    )
    agent3 = Agent[float, List[float]](
        model=TestModel(
            call_tools=[],
            custom_result_args=[
                1.0,
            ],
        ),
        input_schema=float,
        output_schema=List[float],
    )
    resp = await (agent1 | agent2 | agent3).run("test")
    assert resp == [1.0]
