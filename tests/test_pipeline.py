from typing import List
import pytest
from unittest.mock import AsyncMock

from pydantic_ai.models.test import TestModel

from agenty import Agent
from agenty.exceptions import AgentyTypeError
from agenty.types import BaseIO
from agenty.pipeline import Pipeline


class User(BaseIO):
    """Test user model for pipeline testing."""

    first_name: str
    last_name: str


@pytest.fixture
def test_users() -> List[dict]:
    """Fixture providing test user data."""
    return [
        {"first_name": "John", "last_name": "Doe"},
        {"first_name": "Agenty", "last_name": "Rocks"},
    ]


@pytest.fixture
def user_names() -> List[str]:
    """Fixture providing expected user name strings."""
    return ["John Doe", "Agenty Rocks"]


@pytest.fixture
def user_extractor(test_users) -> Agent[str, List[User]]:
    """Fixture providing an agent that extracts user objects."""
    return Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=test_users,
        ),
        input_schema=str,
        output_schema=List[User],
    )


@pytest.fixture
def name_formatter(user_names) -> Agent[List[User], List[str]]:
    """Fixture providing an agent that formats user objects to strings."""
    return Agent(
        model=TestModel(
            call_tools=[],
            custom_result_args=user_names,
        ),
        input_schema=List[User],
        output_schema=List[str],
    )


@pytest.mark.asyncio
class TestPipeline:
    """Test suite for Pipeline functionality."""

    async def test_pipeline_basic_chaining(
        self, user_extractor: Agent, name_formatter: Agent, user_names: List[str]
    ):
        """Test basic pipeline chaining with compatible agents."""
        pipeline = user_extractor | name_formatter
        result = await pipeline.run("test input")
        assert result == user_names
        assert isinstance(pipeline, Pipeline)
        assert pipeline.input_schema is str
        assert pipeline.output_schema == List[str]

    async def test_pipeline_type_validation_error(
        self, user_extractor: Agent, test_users: List[dict]
    ):
        """Test pipeline raises error when agent schemas are incompatible."""
        incompatible_agent = Agent(
            model=TestModel(
                call_tools=[],
                custom_result_args=test_users,
            ),
            input_schema=str,  # Should be List[User]
            output_schema=List[str],
        )

        pipeline = user_extractor | incompatible_agent
        with pytest.raises(AgentyTypeError) as exc_info:
            await pipeline.run("test input")
        assert "Input data type" in str(exc_info.value)

    @pytest.mark.parametrize(
        "input_data,expected",
        [
            ("", []),  # Empty input
            ("test", [1.0]),  # Normal input
        ],
    )
    async def test_multi_agent_pipeline(self, input_data: str, expected: List[float]):
        """Test pipeline with multiple agents and different data types."""
        agent1 = Agent[str, int](
            model=TestModel(call_tools=[], custom_result_args=1),
            input_schema=str,
            output_schema=int,
        )
        agent2 = Agent[int, float](
            model=TestModel(call_tools=[], custom_result_args=1.0),
            input_schema=int,
            output_schema=float,
        )
        agent3 = Agent[float, List[float]](
            model=TestModel(call_tools=[], custom_result_args=expected),
            input_schema=float,
            output_schema=List[float],
        )

        pipeline = agent1 | agent2 | agent3
        result = await pipeline.run(input_data)
        assert result == expected

    async def test_pipeline_empty_sequence_handling(self):
        """Test pipeline handles empty sequence inputs/outputs correctly."""
        agent1 = Agent[str, List[int]](
            model=TestModel(call_tools=[], custom_result_args=[]),
            input_schema=str,
            output_schema=List[int],
        )
        agent2 = Agent[List[int], List[str]](
            model=TestModel(call_tools=[], custom_result_args=[]),
            input_schema=List[int],
            output_schema=List[str],
        )

        pipeline = agent1 | agent2
        result = await pipeline.run("test")
        assert result == []

    async def test_pipeline_error_propagation(self):
        """Test that pipeline properly propagates errors from agents."""

        agent1 = Agent[str, str](
            model=TestModel(call_tools=[], custom_result_text="pipeline test"),
            input_schema=str,
            output_schema=str,
        )
        # Confirm step 1 works
        output = await agent1.run("test")
        assert output == "pipeline test"

        error_msg = "Test error"

        async def raise_error(*args, **kwargs):
            raise ValueError(error_msg)

        test_model = TestModel(call_tools=[])
        test_agent_model = await test_model.agent_model(
            function_tools=[],
            allow_text_result=True,
            result_tools=[],
        )
        test_agent_model.request = raise_error
        test_model.agent_model = AsyncMock(
            spec=test_model.agent_model, return_value=test_agent_model
        )

        agent2 = Agent(
            model=test_model,
            input_schema=str,
            output_schema=str,
        )
        # Confirm step 2 fails
        with pytest.raises(ValueError) as exc_info:
            await agent2.run("test")
        assert error_msg in str(exc_info.value)

        # Confirm pipeline fails with same error
        with pytest.raises(ValueError) as exc_info:
            await (agent1 | agent2).run("test")
        assert error_msg in str(exc_info.value)
