from typing import Optional

import pytest

from agenty import Agent, ChatHistory
from agenty.decorators import tool
from agenty.exceptions import AgentyAttributeError, AgentyTypeError, InvalidResponse
from agenty.models import FunctionModel, ModelSettings, allow_model_requests
from agenty.types import BaseIO

allow_model_requests(False)


class FakeInput(BaseIO):
    query: str
    context: Optional[str] = None


class FakeOutput(BaseIO):
    response: str
    confidence: float


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization with different parameters.

    Verifies that both default and custom initialization parameters are correctly set,
    including model, name, system prompt, retries, end strategy, and chat history.
    """

    class CustomAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    # Test default initialization
    agent = CustomAgent()
    assert agent.model is None, "Default model should be None"
    assert agent.model_name == "", "Default model name should be empty"
    assert agent.name == "", "Default name should be empty"
    assert agent.system_prompt == "", "Default system prompt should be empty"
    assert agent.retries == 1, "Default retries should be 1"
    assert agent.end_strategy == "early", "Default end strategy should be 'early'"
    assert isinstance(agent.chat_history, ChatHistory), (
        "Default chat history should be ChatHistory instance"
    )

    # Test custom initialization
    custom_chat_history = ChatHistory()
    agent = CustomAgent(
        name="test-agent",
        system_prompt="test system prompt",
        retries=3,
        end_strategy="exhaustive",
        chat_history=custom_chat_history,
    )
    assert agent.model is None, "Custom model should be None"
    assert agent.model_name == "", "Custom model name should be empty"
    assert agent.name == "test-agent", "Custom name should match"
    assert agent.system_prompt == "test system prompt", (
        "Custom system prompt should match"
    )
    assert agent.retries == 3, "Custom retries should match"
    assert agent.end_strategy == "exhaustive", "Custom end strategy should match"
    assert agent.chat_history is custom_chat_history, "Custom chat history should match"

    with pytest.raises(AgentyAttributeError):
        agent.pai_agent


def test_template_context():
    """Test template context generation.

    Verifies that only uppercase class variables are included in the template context,
    while private and normal variables are excluded.
    """

    class TemplateAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        TEST_VAR = "test value"
        _private_var = "private"
        normal_var = "normal"

    agent = TemplateAgent()
    context = agent.template_context()

    assert "TEST_VAR" in context, "TEST_VAR should be in template context"
    assert context["TEST_VAR"] == "test value", "TEST_VAR value should match"
    assert "_private_var" not in context, "Private variables should not be in context"
    assert "normal_var" not in context, "Normal variables should not be in context"


@pytest.mark.asyncio
async def test_agent_no_model():
    """Test that running an agent without a model raises an error."""

    class StrStrAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StrStrAgent()
    with pytest.raises(AgentyAttributeError):
        await agent.run("test")


@pytest.mark.asyncio
async def test_agent_str_str(str_test_model: FunctionModel):
    """Test basic string input/output agent functionality."""

    class StrStrAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

    agent = StrStrAgent()
    resp = await agent.run("agenty")
    assert resp == "success: agenty", "Response should match expected format"


@pytest.mark.asyncio
async def test_agent_schema(schema_test_model: FunctionModel):
    """Test agent with custom input/output schemas.

    Verifies that agents properly handle custom schema types for both input and output.
    """

    class SchemaAgent(Agent[FakeInput, FakeOutput]):
        input_schema = FakeInput
        output_schema = FakeOutput
        model = schema_test_model

    agent = SchemaAgent()
    test_input = FakeInput(query="agenty", context="0.5")
    resp = await agent.run(test_input)
    assert isinstance(resp, FakeOutput), "Response should be FakeOutput instance"
    assert resp.response == "success: agenty", "Response content should match"
    assert resp.confidence == 0.5, "Confidence should match input context"


@pytest.mark.asyncio
async def test_agent_schema_error(error_test_model: FunctionModel):
    """Test error handling with custom input/output schemas.

    Verifies proper error handling for invalid inputs and responses.
    """

    class SchemaAgent(Agent[FakeInput, FakeOutput]):
        input_schema = FakeInput
        output_schema = FakeOutput
        model = error_test_model

    agent = SchemaAgent()
    with pytest.raises(AgentyTypeError):
        await agent.run("bad input")  # type: ignore

    test_input = FakeInput(query="agenty", context="0.5")
    with pytest.raises(InvalidResponse):
        await agent.run(test_input)


def test_model_name(str_test_model: FunctionModel):
    """Test model name property for agents with and without models."""

    class ModelAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

    class NoModelAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    no_model = NoModelAgent()
    assert no_model.model_name == "", "Agent without model should have empty model name"

    model = ModelAgent()
    assert model.model_name.startswith("function:"), (
        "Model name should start with 'function:'"
    )


@pytest.mark.asyncio
async def test_agent_errors():
    """Test agent error handling and noop model behavior."""

    class ErrorAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = ErrorAgent()

    # check that the underlying pai_agent is noop
    assert "_noop" in agent._pai_agent.model.name()  # type: ignore
    # check that public property pai_agent raises an error if no model is set
    with pytest.raises(AgentyAttributeError):
        agent.pai_agent


@pytest.mark.asyncio
async def test_chat_history(str_test_model: FunctionModel):
    """Test chat history functionality and message tracking.

    Verifies that messages are properly added to chat history and that
    the history maintains the correct order and content.
    """

    class ChatAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

    agent = ChatAgent()

    first_message = "agenty: first message"
    await agent.run(first_message)
    assert len(agent.chat_history) == 2, (
        "History should have 2 messages after first run"
    )
    assert agent.chat_history[0].content == first_message, (
        "First message content should match"
    )
    assert agent.chat_history[1].content == f"success: {first_message}", (
        "Response content should match"
    )

    second_message = "agenty: second message"
    await agent.run(second_message)
    assert len(agent.chat_history) == 4, (
        "History should have 4 messages after second run"
    )
    assert agent.chat_history[2].content == second_message, (
        "Second message content should match"
    )
    assert "success:" in str(agent.chat_history[3].content), (
        "Response should contain success"
    )

    # check that None input doesn't add to history (only response should be added)
    await agent.run(None)
    assert len(agent.chat_history) == 5, (
        "History should have 5 messages after None input"
    )
    assert "failure:" in str(agent.chat_history[4].content), (
        "Response should contain failure"
    )


def test_system_prompt_rendering():
    """Test system prompt rendering with template context variables."""

    class PromptAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        MODEL_NAME = "test-model"
        system_prompt = "Using model: {{MODEL_NAME}}"

    agent = PromptAgent()
    rendered = agent.system_prompt_rendered
    assert rendered == f"Using model: {agent.MODEL_NAME}", (
        "Rendered prompt should include MODEL_NAME"
    )


@pytest.mark.asyncio
async def test_result_retries_and_model_settings(str_test_model: FunctionModel):
    """Test result_retries parameter and model settings configuration."""

    class RetryAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model
        result_retries = 2
        model_settings = ModelSettings(temperature=0.7)

    agent = RetryAgent()
    assert agent.result_retries == 2, "Default result_retries should match class value"
    assert agent.model_settings is not None, "Model settings should not be None"
    assert (
        "temperature" in agent.model_settings
        and agent.model_settings["temperature"] == 0.7
    ), "Temperature should match class value"

    agent = RetryAgent(
        result_retries=3,
        model_settings=ModelSettings(temperature=0.5),
    )
    assert agent.result_retries == 3, "Custom result_retries should match"
    assert agent.model_settings is not None, "Model settings should not be None"
    assert (
        "temperature" in agent.model_settings
        and agent.model_settings["temperature"] == 0.5
    ), "Temperature should match custom value"


@pytest.mark.asyncio
async def test_chat_history_with_names(str_test_model: FunctionModel):
    """Test chat history with named and unnamed messages."""

    class ChatNameAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

    agent = ChatNameAgent()

    # Test with named message
    message = "agenty: named message"
    await agent.run(message, name="test_user")
    assert len(agent.chat_history) == 2, (
        "History should have 2 messages after first run"
    )
    assert agent.chat_history[0].name == "test_user", "Message should have correct name"

    # Test without name
    message2 = "agenty: unnamed message"
    await agent.run(message2)
    assert len(agent.chat_history) == 4, (
        "History should have 4 messages after second run"
    )
    assert agent.chat_history[2].name is None, "Message should have no name"


@pytest.mark.asyncio
async def test_end_strategy():
    """Test different end strategies (early vs exhaustive)."""

    class EndStrategyAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    # Test early strategy
    early_agent = EndStrategyAgent(end_strategy="early")
    assert early_agent.end_strategy == "early", "End strategy should be 'early'"

    # Test exhaustive strategy
    exhaustive_agent = EndStrategyAgent(end_strategy="exhaustive")
    assert exhaustive_agent.end_strategy == "exhaustive", (
        "End strategy should be 'exhaustive'"
    )


def test_template_context_inheritance():
    """Test template context inheritance and variable overriding between parent and child agents."""

    class ParentAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        PARENT_VAR = "parent value"
        SHARED_VAR = "parent shared"

    class ChildAgent(ParentAgent):
        CHILD_VAR = "child value"
        SHARED_VAR = "child shared"  # Override parent

    parent = ParentAgent()
    child = ChildAgent()

    parent_context = parent.template_context()
    child_context = child.template_context()

    assert "PARENT_VAR" in parent_context, "Parent var should be in parent context"
    assert parent_context["PARENT_VAR"] == "parent value", (
        "Parent var value should match"
    )

    assert "CHILD_VAR" in child_context, "Child var should be in child context"
    assert child_context["CHILD_VAR"] == "child value", "Child var value should match"

    assert child_context["SHARED_VAR"] == "child shared", (
        "Child should override shared var"
    )


def test_tool_registration(str_test_model: FunctionModel):
    """Test tool registration and method decoration."""

    class ToolAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

        @tool
        def tool1(self) -> str:
            return "tool1"

        @tool
        def tool2(self) -> str:
            return "tool2"

    agent = ToolAgent()
    tools = [t for t in dir(agent) if hasattr(getattr(agent, t), "_is_tool")]
    assert len(tools) == 2, "Should have exactly 2 tools"
    assert "tool1" in tools, "tool1 should be registered"
    assert "tool2" in tools, "tool2 should be registered"
