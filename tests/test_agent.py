import pytest
from typing import Optional
from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ModelResponsePart,
    ToolCallPart,
)
from pydantic_ai.models import ModelSettings
from agenty.agent import Agent
from agenty.agent.chat_history import ChatHistory
from agenty.models import allow_model_requests, AgentInfo, FunctionModel
import agenty.exceptions as exc
from agenty.types import BaseIO
from agenty.decorators import tool

allow_model_requests(False)

TEST_STR_QUERY = "test str"
TEST_SCHEMA_QUERY = "test schema"
TEST_ERROR_SCHEMA_QUERY = "test error schema"


class FakeInput(BaseIO):
    query: str
    context: Optional[str] = None


class FakeOutput(BaseIO):
    response: str
    confidence: float


@pytest.fixture
def agenty_test_model() -> FunctionModel:
    async def test_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        response_parts: list[ModelResponsePart] = []

        def process_request(message: ModelRequest):
            for part in message.parts:
                match part.part_kind:
                    case "user-prompt":
                        if TEST_STR_QUERY in part.content:
                            response_parts.append(TextPart(f"success: {part.content}"))
                        elif TEST_SCHEMA_QUERY in part.content:
                            input = TypeAdapter(FakeInput).validate_json(part.content)
                            output = FakeOutput(
                                response=f"success: {input.query}",
                                confidence=float(input.context)
                                if input.context
                                else -1.0,
                            )
                            response_parts.append(
                                ToolCallPart(
                                    tool_name="final_result",
                                    args=output.model_dump(),
                                    tool_call_id=None,
                                )
                            )
                        elif TEST_ERROR_SCHEMA_QUERY in part.content:
                            input = TypeAdapter(FakeInput).validate_json(part.content)
                            response_parts.append(
                                ToolCallPart(
                                    tool_name="final_result",
                                    args={"error": "bad output"},
                                    tool_call_id=None,
                                )
                            )
                        else:
                            response_parts.append(TextPart(f"failure: {part.content}"))
                    case "system-prompt":
                        pass
                    case "tool-return":
                        pass
                    case "retry-prompt":
                        pass

        for message in messages:
            if message.kind == "request":
                process_request(message)

        return ModelResponse(
            parts=response_parts,
            model_name="agenty-test",
        )

    return FunctionModel(test_func)


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test agent initialization with different parameters"""

    class CustomAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    # Test default initialization
    agent = CustomAgent()
    assert agent.model is None
    assert agent.model_name == ""
    assert agent.name == ""
    assert agent.system_prompt == ""
    assert agent.retries == 1
    assert agent.end_strategy == "early"
    assert isinstance(agent.chat_history, ChatHistory)

    # Test custom initialization
    custom_chat_history = ChatHistory()
    agent = CustomAgent(
        name="test-agent",
        system_prompt="test system prompt",
        retries=3,
        end_strategy="exhaustive",
        chat_history=custom_chat_history,
    )
    assert agent.model is None
    assert agent.model_name == ""
    assert agent.name == "test-agent"
    assert agent.system_prompt == "test system prompt"
    assert agent.retries == 3
    assert agent.end_strategy == "exhaustive"
    assert agent.chat_history is custom_chat_history

    with pytest.raises(exc.AgentyAttributeError):
        agent.pai_agent


def test_template_context():
    """Test template context generation"""

    class TemplateAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        TEST_VAR = "test value"
        _private_var = "private"
        normal_var = "normal"

    agent = TemplateAgent()
    context = agent.template_context()

    assert "TEST_VAR" in context
    assert context["TEST_VAR"] == "test value"
    assert "_private_var" not in context
    assert "normal_var" not in context


@pytest.mark.asyncio
async def test_agent_no_model():
    class StrStrAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = StrStrAgent()
    with pytest.raises(exc.AgentyAttributeError):
        await agent.run(TEST_STR_QUERY)


@pytest.mark.asyncio
async def test_agent_str_str(agenty_test_model: FunctionModel):
    class StrStrAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = agenty_test_model

    agent = StrStrAgent()
    resp = await agent.run(TEST_STR_QUERY)
    assert resp == f"success: {TEST_STR_QUERY}"


@pytest.mark.asyncio
async def test_agent_schema(agenty_test_model: FunctionModel):
    """Test agent with custom input/output schemas"""

    class SchemaAgent(Agent[FakeInput, FakeOutput]):
        input_schema = FakeInput
        output_schema = FakeOutput
        model = agenty_test_model

    agent = SchemaAgent()
    test_input = FakeInput(query=TEST_SCHEMA_QUERY, context="0.5")
    resp = await agent.run(test_input)
    assert isinstance(resp, FakeOutput)
    assert resp.response == f"success: {TEST_SCHEMA_QUERY}"
    assert resp.confidence == 0.5


@pytest.mark.asyncio
async def test_agent_schema_error(agenty_test_model: FunctionModel):
    """Test agent with custom input/output schemas"""

    class SchemaAgent(Agent[FakeInput, FakeOutput]):
        input_schema = FakeInput
        output_schema = FakeOutput
        model = agenty_test_model

    agent = SchemaAgent()
    with pytest.raises(exc.AgentyTypeError):
        await agent.run("bad input")  # type: ignore

    test_input = FakeInput(query=TEST_ERROR_SCHEMA_QUERY, context="0.5")
    with pytest.raises(exc.InvalidResponse):
        await agent.run(test_input)


def test_model_name(agenty_test_model: FunctionModel):
    """Test model name property"""

    class ModelAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = agenty_test_model

    class NoModelAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    no_model = NoModelAgent()
    assert no_model.model_name == ""

    model = ModelAgent()
    assert model.model_name.startswith("function:")


@pytest.mark.asyncio
async def test_agent_errors():
    """Test agent error handling"""

    class ErrorAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    agent = ErrorAgent()

    # check that the underlying pai_agent is noop
    assert "_noop" in agent._pai_agent.model.name()  # type: ignore
    # check that public property pai_agent raises an error if no model is set
    with pytest.raises(exc.AgentyAttributeError):
        agent.pai_agent


@pytest.mark.asyncio
async def test_chat_history(agenty_test_model: FunctionModel):
    """Test chat history functionality"""

    class ChatAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = agenty_test_model

    agent = ChatAgent()

    first_message = f"{TEST_STR_QUERY}: first message"
    await agent.run(first_message)
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].content == first_message
    assert agent.chat_history[1].content == f"success: {first_message}"

    second_message = f"{TEST_STR_QUERY}: second message"
    await agent.run(second_message)
    assert len(agent.chat_history) == 4
    assert agent.chat_history[2].content == second_message

    # The output message includes all of the previous user_queries because
    assert "success:" in str(agent.chat_history[3].content)

    # check that None input doesn't add to history (only response should be added)
    await agent.run(None)
    assert len(agent.chat_history) == 5
    print(agent.chat_history[4].content)
    assert "failure:" in str(agent.chat_history[4].content)


def test_system_prompt_rendering():
    """Test system prompt rendering with template context"""

    class PromptAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        MODEL_NAME = "test-model"
        system_prompt = "Using model: {{MODEL_NAME}}"

    agent = PromptAgent()
    rendered = agent.system_prompt_rendered
    assert rendered == f"Using model: {agent.MODEL_NAME}"


@pytest.mark.asyncio
async def test_result_retries_and_model_settings(agenty_test_model: FunctionModel):
    """Test result_retries parameter and model settings"""

    class RetryAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = agenty_test_model
        result_retries = 2
        model_settings = ModelSettings(temperature=0.7)

    agent = RetryAgent()
    assert agent.result_retries == 2
    assert agent.model_settings is not None
    assert (
        "temperature" in agent.model_settings
        and agent.model_settings["temperature"] == 0.7
    )

    agent = RetryAgent(
        result_retries=3,
        model_settings=ModelSettings(temperature=0.5),
    )
    assert agent.result_retries == 3
    assert agent.model_settings is not None
    assert (
        "temperature" in agent.model_settings
        and agent.model_settings["temperature"] == 0.5
    )


@pytest.mark.asyncio
async def test_chat_history_with_names(agenty_test_model: FunctionModel):
    """Test chat history with named messages"""

    class ChatNameAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = agenty_test_model

    agent = ChatNameAgent()

    # Test with named message
    message = f"{TEST_STR_QUERY}: named message"
    await agent.run(message, name="test_user")
    assert len(agent.chat_history) == 2
    assert agent.chat_history[0].name == "test_user"

    # Test without name
    message2 = f"{TEST_STR_QUERY}: unnamed message"
    await agent.run(message2)
    assert len(agent.chat_history) == 4
    assert agent.chat_history[2].name is None


@pytest.mark.asyncio
async def test_end_strategy():
    """Test different end strategy"""

    class EndStrategyAgent(Agent[str, str]):
        input_schema = str
        output_schema = str

    # Test early strategy
    early_agent = EndStrategyAgent(end_strategy="early")
    assert early_agent.end_strategy == "early"

    # Test exhaustive strategy
    exhaustive_agent = EndStrategyAgent(end_strategy="exhaustive")
    assert exhaustive_agent.end_strategy == "exhaustive"


def test_template_context_inheritance():
    """Test template context inheritance and overriding"""

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

    assert "PARENT_VAR" in parent_context
    assert parent_context["PARENT_VAR"] == "parent value"

    assert "CHILD_VAR" in child_context
    assert child_context["CHILD_VAR"] == "child value"

    assert child_context["SHARED_VAR"] == "child shared"


def test_tool_registration(agenty_test_model: FunctionModel):
    """Test tool registration and decoration"""

    class ToolAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = agenty_test_model

        @tool
        def tool1(self) -> str:
            return "tool1"

        @tool
        def tool2(self) -> str:
            return "tool2"

    agent = ToolAgent()
    tools = [t for t in dir(agent) if hasattr(getattr(agent, t), "_is_tool")]
    assert len(tools) == 2
    assert "tool1" in tools
    assert "tool2" in tools
