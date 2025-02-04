import pytest

from agenty import Agent
from agenty.decorators import hook
from agenty.exceptions import AgentyValueError
from pydantic_ai.models.test import TestModel


@pytest.mark.asyncio
async def test_hooks_in_agent():
    """Test that hooks are properly collected in Agent class"""

    class AgentWithHooks(Agent[str, str]):
        input_schema = str
        output_schema = str

        @hook.input
        def add_prefix(self, input: str) -> str:
            return f"prefix_{input}"

        @hook.output
        def add_suffix(self, output: str) -> str:
            return f"{output}_suffix"

    agent = AgentWithHooks(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
    )
    assert hasattr(agent, "_input_hooks")
    assert hasattr(agent, "_output_hooks")
    assert len(agent._input_hooks) == 1
    assert len(agent._output_hooks) == 1


@pytest.mark.asyncio
async def test_hooks_transform_input_output():
    """Test that hooks properly transform input and output"""

    class AgentWithHooks(Agent[str, str]):
        input_schema = str
        output_schema = str

        @hook.input
        def add_prefix(self, input: str) -> str:
            return f"prefix_{input}"

        @hook.output
        def add_suffix(self, output: str) -> str:
            return f"{output}_suffix"

    agent = AgentWithHooks(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
    )
    result = agent._input_hooks[0](agent, "test")
    assert result == "prefix_test"

    result = agent._output_hooks[0](agent, "test")
    assert result == "test_suffix"


@pytest.mark.asyncio
async def test_multiple_hooks_order():
    """Test that multiple hooks are called in the correct order"""

    class AgentWithMultipleHooks(Agent[str, str]):
        input_schema = str
        output_schema = str
        prefix = "custom"

        @hook.input
        def add_prefix(self, input: str) -> str:
            return f"{self.prefix}_{input}"

        @hook.input
        def add_second_prefix(self, input: str) -> str:
            return f"second_{input}"

        @hook.output
        def add_suffix(self, output: str) -> str:
            return f"{output}_first"

        @hook.output
        def add_second_suffix(self, output: str) -> str:
            return f"{output}_second"

    agent = AgentWithMultipleHooks(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
    )

    # Input hooks are called in reverse order of definition
    result = agent._input_hooks[0](agent, "test")  # second_prefix
    assert result == "second_test"
    result = agent._input_hooks[1](agent, "test")  # add_prefix
    assert result == "custom_test"

    # Output hooks are called in reverse order of definition
    result = agent._output_hooks[0](agent, "test")  # add_second_suffix
    assert result == "test_second"
    result = agent._output_hooks[1](agent, "test")  # add_suffix
    assert result == "test_first"


@pytest.mark.asyncio
async def test_hooks_can_access_instance_attributes():
    """Test that hooks can access instance attributes"""

    class AgentWithAttributes(Agent[str, str]):
        input_schema = str
        output_schema = str
        prefix = "custom"

        @hook.input
        def add_prefix(self, input: str) -> str:
            return f"{self.prefix}_{input}"

    agent = AgentWithAttributes(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
    )
    agent.prefix = "modified"
    result = agent._input_hooks[0](agent, "test")
    assert result == "modified_test"


@pytest.mark.asyncio
async def test_invalid_hook_type():
    """Test that hooks must preserve input/output types"""

    class AgentWithInvalidHook(Agent[str, str]):
        input_schema = str
        output_schema = str

        @hook.input
        def invalid_hook(self, input: str) -> int:
            return 42

    agent = AgentWithInvalidHook(
        model=TestModel(),
        input_schema=str,
        output_schema=str,
    )

    with pytest.raises(AgentyValueError) as exc:
        await agent.run("test")
    assert "returned invalid type" in str(exc.value)


@pytest.mark.asyncio
async def test_hooks_end_to_end():
    """Test hooks in a complete agent run flow"""

    class AgentWithHooks(Agent[str, str]):
        input_schema = str
        output_schema = str

        @hook.input
        def add_prefix(self, input: str) -> str:
            return f"prefix_{input}"

        @hook.output
        def add_suffix(self, output: str) -> str:
            return f"{output}_suffix"

    agent = AgentWithHooks(
        model=TestModel(
            call_tools=[],
            custom_result_text="test output",
        ),
        input_schema=str,
        output_schema=str,
    )
    result = await agent.run("test")
    assert result == "test output_suffix"
