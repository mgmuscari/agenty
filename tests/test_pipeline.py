from typing import Optional

import pytest

from agenty import Agent, Pipeline
from agenty.exceptions import AgentyTypeError, AgentyValueError
from agenty.models import FunctionModel
from tests.conftest import FakeInput, FakeOutput


class StringAgent(Agent[str, str]):
    """Basic string-to-string agent for testing."""

    input_schema = str
    output_schema = str


def test_pipeline_initialization(str_test_model: FunctionModel):
    """Test pipeline initialization with different parameters.

    Verifies that pipelines can be initialized:
    - Empty (no agents)
    - With agents
    - With custom input/output schemas
    """
    # Test empty pipeline
    pipeline = Pipeline()
    assert len(pipeline.agents) == 0, "Empty pipeline should have no agents"
    assert pipeline.current_step is None, "Empty pipeline should have no current step"
    assert len(pipeline.output_history) == 0, (
        "Empty pipeline should have no output history"
    )

    # Test pipeline with agents
    agent = StringAgent()
    pipeline = Pipeline(agents=[agent])
    assert len(pipeline.agents) == 1, "Pipeline should have one agent"

    # Test pipeline with custom schemas
    pipeline = Pipeline(input_schema=FakeInput, output_schema=FakeOutput)
    assert pipeline.input_schema == FakeInput, "Input schema should match"
    assert pipeline.output_schema == FakeOutput, "Output schema should match"


@pytest.mark.asyncio
async def test_empty_pipeline():
    """Test that running an empty pipeline raises an error."""
    pipeline = Pipeline()
    with pytest.raises(AgentyValueError):
        await pipeline.run("test")


@pytest.mark.asyncio
async def test_simple_pipeline(str_test_model: FunctionModel):
    """Test pipeline with a single string agent.

    Verifies:
    - Basic pipeline execution
    - Output history tracking
    - Pipeline state reset between runs
    """
    agent = StringAgent(model=str_test_model)
    pipeline = Pipeline(agents=[agent])

    # Test failure case
    result = await pipeline.run("test")
    assert result == "failure: test", "Failed input should include failure message"
    assert len(pipeline.output_history) == 1, "Output history should have one entry"
    assert pipeline.current_step == 0, "Current step should be 0"

    # Test success case (agents are reset between pipeline runs)
    result = await pipeline.run("agenty")
    assert result == "success: agenty", (
        "Successful input should include success message"
    )
    assert "success" in result, "Result should contain success indicator"
    assert len(pipeline.output_history) == 1, "Output history should have one entry"
    assert pipeline.current_step == 0, "Current step should be 0"


@pytest.mark.asyncio
async def test_schema_pipeline(schema_test_model: FunctionModel):
    """Test pipeline with schema input/output.

    Verifies that pipelines properly handle:
    - Custom input/output schemas
    - Schema validation
    - Output history tracking
    """

    class SchemaAgent(Agent[FakeInput, FakeOutput]):
        input_schema = FakeInput
        output_schema = FakeOutput
        model = schema_test_model

    agent = SchemaAgent()
    pipeline = Pipeline[FakeInput, FakeOutput](
        agents=[agent],
        input_schema=FakeInput,
        output_schema=FakeOutput,
    )

    input_data = FakeInput(query="agenty", context="0.5")
    result = await pipeline.run(input_data)

    assert isinstance(result, FakeOutput), "Result should be FakeOutput instance"
    assert f"success: {input_data.query}" in result.response, (
        "Response should contain input query"
    )
    assert result.confidence == 0.5, "Confidence should match input context"
    assert len(pipeline.output_history) == 1, "Output history should have one entry"
    assert pipeline.current_step == 0, "Current step should be 0"


@pytest.mark.asyncio
async def test_pipeline_chaining(str_test_model: FunctionModel):
    """Test pipeline chaining using the | operator.

    Verifies that agents can be chained together using the pipeline
    operator to create a multi-step pipeline.
    """

    class FirstAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

    class SecondAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = str_test_model

    first_agent = FirstAgent()
    second_agent = SecondAgent()

    pipeline = first_agent | second_agent

    input_data = "agenty PIPELINE"
    result = await pipeline.run(input_data)
    assert "success" in result, "Chained pipeline result should contain success"


@pytest.mark.asyncio
async def test_pipeline_type_validation(schema_test_model: FunctionModel):
    """Test pipeline input/output type validation.

    Verifies that pipelines properly validate input types and
    raise appropriate errors for invalid inputs.
    """

    class SchemaAgent(Agent[FakeInput, FakeOutput]):
        input_schema = FakeInput
        output_schema = FakeOutput
        model = schema_test_model

    pipeline = Pipeline(
        agents=[SchemaAgent()],
        input_schema=FakeInput,
        output_schema=FakeOutput,
    )

    # Test invalid input type
    with pytest.raises(AgentyTypeError):
        await pipeline.run("invalid input")  # type: ignore


@pytest.mark.asyncio
async def test_pipeline_with_none_input(error_test_model: FunctionModel):
    """Test pipeline handling of None input.

    Verifies that pipelines properly handle None inputs by raising
    appropriate type errors.
    """
    agent = StringAgent(model=error_test_model)
    pipeline = Pipeline(agents=[agent])

    with pytest.raises(AgentyTypeError):
        await pipeline.run(None)


@pytest.mark.asyncio
async def test_pipeline_name_propagation(str_test_model: FunctionModel):
    """Test that pipeline name is propagated to agents.

    Verifies that names passed to pipeline.run() are correctly
    propagated to the underlying agents.
    """

    class NameCheckAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        received_name: Optional[str] = None
        model = str_test_model

        async def run(
            self, input_data: Optional[str], name: Optional[str] = None
        ) -> str:
            if input_data is None:
                raise ValueError("Input data cannot be None")
            self.received_name = name
            return input_data

    agent = NameCheckAgent()
    pipeline = Pipeline(agents=[agent])

    test_name = "test-run-1"
    await pipeline.run("test", name=test_name)
    assert agent.received_name == test_name, "Agent should receive pipeline run name"
