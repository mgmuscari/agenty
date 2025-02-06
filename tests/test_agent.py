from typing import List

import pytest
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.models import ModelSettings

from agenty import Agent
from agenty.exceptions import AgentyValueError
from agenty.types import BaseIO

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def success_model() -> FunctionModel:
    async def success_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart("success"),
            ],
            model_name="test",
        )

    return FunctionModel(success_function)


@pytest.fixture
def list_model() -> FunctionModel:
    async def list_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"response": [1, 2, 3]},
                    tool_call_id=None,
                )
            ],
            model_name="test",
        )

    return FunctionModel(list_function)


@pytest.fixture
def baseio_model() -> FunctionModel:
    async def baseio_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"a": 2, "b": "agenty", "c": False},
                    tool_call_id=None,
                )
            ],
            model_name="test",
        )

    return FunctionModel(baseio_function)


@pytest.fixture
def invalid_baseio_model() -> FunctionModel:
    async def invalid_baseio_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"a": 1, "b": "test", "d": True, "e": "extra"},
                    tool_call_id=None,
                )
            ],
            model_name="test",
        )

    return FunctionModel(invalid_baseio_function)


@pytest.fixture
def none_model() -> FunctionModel:
    async def none_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[], model_name="test")

    return FunctionModel(none_function)


@pytest.fixture
def intermediate_model() -> FunctionModel:
    async def intermediate_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"response": "intermediate"},
                    tool_call_id=None,
                )
            ],
            model_name="test",
        )

    return FunctionModel(intermediate_function)


@pytest.fixture
def final_model() -> FunctionModel:
    async def final_function(
        messages: list[ModelMessage], info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="final_result",
                    args={"response": "final"},
                    tool_call_id=None,
                )
            ],
            model_name="test",
        )

    return FunctionModel(final_function)


@pytest.mark.asyncio
async def test_agent_out_str(success_model):
    class StringAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StringAgent(model=success_model)
    resp = await agent.run("test")
    assert resp == "success"


@pytest.mark.asyncio
async def test_agent_out_list(list_model):
    class ListAgent(Agent[str, List[int]]):
        input_schema = str
        output_schema = List[int]

    agent = ListAgent(model=list_model)
    resp = await agent.run("test")
    assert resp == [1, 2, 3]


@pytest.mark.asyncio
async def test_agent_out_baseio(baseio_model, invalid_baseio_model):
    class TestIO(BaseIO):
        a: int
        b: str
        c: bool

    class BaseIOAgent(Agent[str, TestIO]):
        input_schema = str
        output_schema = TestIO

    agent = BaseIOAgent(model=baseio_model)
    resp = await agent.run("test")
    assert resp != {"a": 2, "b": "agenty", "c": False}
    assert resp == TestIO(a=2, b="agenty", c=False)

    with pytest.raises(AgentyValueError):
        agent = BaseIOAgent(model=invalid_baseio_model)
        resp = await agent.run("test")


@pytest.mark.asyncio
async def test_agent_out_none(none_model):
    class StringAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StringAgent(model=none_model)
    with pytest.raises(AgentyValueError):
        await agent.run("test")


@pytest.mark.asyncio
async def test_agent_model_settings(success_model):
    class StringAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StringAgent(
        model=success_model,
        model_settings=ModelSettings(
            temperature=0.79,
            top_p=0.99,
        ),
    )

    assert agent.model_settings is not None
    assert isinstance(agent.model_settings, dict)
    assert agent.model_settings.get("temperature") == 0.79
    assert agent.model_settings.get("top_p") == 0.99


@pytest.mark.asyncio
async def test_agent_template_context(success_model):
    class TestAgent(Agent[str, str]):
        TEST_ID: str
        input_schema = str
        output_schema = str

    agent = TestAgent(
        model=success_model,
        system_prompt="You are a helpful assistant with ID {{ TEST_ID }}",
    )

    agent.TEST_ID = "test-123"
    rendered = agent.render_system_prompt()
    assert rendered == "You are a helpful assistant with ID test-123"
    assert agent.template_context() == {"TEST_ID": "test-123"}


@pytest.mark.asyncio
async def test_agent_configuration(success_model):
    class StringAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StringAgent(
        model=success_model,
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
    class StringAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StringAgent(model=None)
    with pytest.raises(AgentyValueError):
        await agent.run("test")
