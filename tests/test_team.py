import pytest
from typing import List

from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

from agenty.team import Team, TextMentionStop, OutputSchemaStop
from agenty.agent import Agent
from agenty.types import BaseIO
from agenty.exceptions import AgentyValueError


# Fixtures
@pytest.fixture
def success_model() -> FunctionModel:
    async def success_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Output from success_agent")])

    return FunctionModel(success_function)


@pytest.fixture
def another_success_model() -> FunctionModel:
    async def another_success_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Output from another_success_agent")])

    return FunctionModel(another_success_function)


@pytest.fixture
def success_agent(success_model) -> Agent[str, str]:
    return Agent(
        model=success_model,
        name="success_agent",
    )


@pytest.fixture
def another_success_agent(another_success_model) -> Agent[str, str]:
    return Agent(
        model=another_success_model,
        name="another_success_agent",
    )


@pytest.fixture
def intermediate_model() -> FunctionModel:
    async def intermediate_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Output from intermediate_agent")])

    return FunctionModel(intermediate_function)


@pytest.fixture
def intermediate_agent(intermediate_model) -> Agent[str, str]:
    return Agent(model=intermediate_model)


@pytest.fixture
def none_model() -> FunctionModel:
    async def none_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[])

    return FunctionModel(none_function)


@pytest.fixture
def none_agent(none_model) -> Agent[str, str]:
    return Agent(model=none_model)


# Test classes and fixtures
class Pizza(BaseIO):
    toppings: List[str]
    crust: str


class Rating(BaseIO):
    score: int
    comment: str


# Test cases
@pytest.mark.asyncio
class TestTeam:
    """Test Team initialization and basic functionality"""

    async def test_initialization(self, success_agent, another_success_agent):
        """Test basic team initialization"""
        team = Team(agents=[success_agent, another_success_agent])
        assert len(team.agents) == 2

        """Test initialization with custom IO schemas"""
        team_with_schema = Team(
            agents=[success_agent, another_success_agent],
            input_schema=Pizza,
            output_schema=Rating,
        )
        assert team_with_schema.input_schema == Pizza
        assert team_with_schema.output_schema == Rating

    async def test_initialization_errors(self, success_model):
        """Test initialization error cases"""
        with pytest.raises(AgentyValueError, match="Team must have at least one agent"):
            Team(agents=[])

        agent1 = Agent(model=success_model, name="agent1")
        agent2 = Agent(model=success_model, name="agent1")
        with pytest.raises(
            AgentyValueError, match="Agent 'agent1' already exists in team"
        ):
            Team(agents=[agent1, agent2])

    async def test_subscriptions(self, success_model):
        """Test agent subscriptions and type handling"""
        pizza_maker = Agent(
            model=success_model,
            output_schema=Pizza,
            name="pizza_maker",
        )
        pizza_rater = Agent(
            model=success_model,
            input_schema=Pizza,
            output_schema=Rating,
            name="pizza_rater",
        )
        team = Team(agents=[pizza_maker, pizza_rater])

        """Verify Pizza type subscription"""
        pizza_subs = team.subscriptions.get((Pizza, ()))
        assert pizza_subs and len(pizza_subs) == 1
        assert pizza_rater in pizza_subs

    async def test_structured_io(self):
        """Test team with structured IO types"""

        class Order(BaseIO):
            items: List[str]
            priority: int

        class Response(BaseIO):
            status: str
            eta: int

        async def output_order_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={
                            "items": ["pizza", "burger"],
                            "priority": 1,
                        },
                        tool_call_id=None,
                    )
                ],
            )

        async def output_response_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={
                            "status": "Order received",
                            "eta": 10,
                        },
                        tool_call_id=None,
                    )
                ],
            )

        order_taker = Agent(
            model=FunctionModel(output_order_model),
            input_schema=str,
            output_schema=Order,
            name="order_taker",
        )
        processor = Agent[Order, Response](
            model=FunctionModel(output_response_model),
            input_schema=Order,
            output_schema=Response,
            name="processor",
        )

        team = Team[str, Response](
            agents=[order_taker, processor],
            input_schema=str,
            output_schema=Response,
            stop_condition=OutputSchemaStop(),
        )

        """Verify type subscriptions"""
        assert (str, ()) in team.subscriptions
        assert (Order, ()) in team.subscriptions

        """Test run with structured IO"""
        await team.run("I want to order a pizza")

    async def test_error_cases(self, success_agent, another_success_agent):
        """Test error handling cases"""
        team = Team(agents=[success_agent, another_success_agent])

        """Test None input handling"""
        with pytest.raises(AgentyValueError, match="Team input data cannot be None"):
            await team.run(None)

    async def test_list_type_subscriptions(self, success_model):
        async def list_output_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="final_result",
                        args={
                            "response": [1, 2, 3],
                        },
                        tool_call_id=None,
                    )
                ],
            )

        """Test list type handling in subscriptions"""
        list_output_agent = Agent[List[str], List[int]](
            model=FunctionModel(list_output_model),
            input_schema=List[str],
            output_schema=List[int],
            name="list_output_agent",
        )
        list_input_agent = Agent(
            model=success_model,
            input_schema=List[int],
            output_schema=str,
            name="list_input_agent",
        )

        """Create team with list input schema"""
        team = Team[List[str], str](
            agents=[list_output_agent, list_input_agent],
            input_schema=List[str],
            output_schema=str,
            stop_condition=TextMentionStop(text="Output from"),
        )

        """Verify list type subscriptions"""
        assert (list, (str,)) in team.subscriptions

        """Test run with list input"""
        result = await team.run(["test1", "test2"])

        assert "Output from success_agent" in result
