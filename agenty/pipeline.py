from typing import Any, Generic, List, Optional, Type, cast

from pydantic import TypeAdapter, ValidationError

from agenty.exceptions import AgentyTypeError, AgentyValueError
from agenty.protocol import AgentIOProtocol
from agenty.types import (
    AgentIO,
    AgentInputT,
    AgentOutputT,
    NOT_GIVEN,
    NotGiven,
    PipelineOutputT,
)


class Pipeline(Generic[AgentInputT, AgentOutputT]):
    """A pipeline for chaining multiple agents together for sequential processing.

    The Pipeline class enables the creation of agent chains where data flows from one
    agent to the next. Each agent in the pipeline must have compatible input/output
    schemas, where each agent's output schema matches the next agent's input schema.

    Type Parameters:
        AgentInputT: The type of input the pipeline accepts (e.g., str, dict, List[User])
        AgentOutputT: The type of output the pipeline produces (e.g., List[str], Report)

    Attributes:
        input_schema: The expected schema for pipeline input data
        output_schema: The expected schema for pipeline output data
        agents: List of agents in the pipeline, executed in order
        output_history: List of outputs from each agent's execution
        current_step: The index of the currently executing agent (1-based)

    Example:
        >>> from typing import List
        >>> from agenty.types import User
        >>> extractor = UserExtractor()  # output_schema = List[User]
        >>> title_agent = TitleAgent()   # input_schema = List[User]
        >>> pipeline = extractor | title_agent
        >>> result = await pipeline.run("Some text input")
    """

    input_schema: Type[AgentIO] = str
    output_schema: Type[AgentIO] = str

    def __init__(
        self,
        agents: List[AgentIOProtocol[Any, Any]] = list(),
        input_schema: Type[AgentIO] | NotGiven = NOT_GIVEN,
        output_schema: Type[AgentIO] | NotGiven = NOT_GIVEN,
    ) -> None:
        super().__init__()

        if not isinstance(input_schema, NotGiven):
            self.input_schema = input_schema
        if not isinstance(output_schema, NotGiven):
            self.output_schema = output_schema
        self.agents = agents
        self.output_history: List[AgentOutputT] = []
        self.current_step: Optional[int] = None

    async def run(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT:
        """Run the pipeline by executing each agent in sequence.

        The pipeline processes input data through a chain of agents, where each agent's
        output becomes the input for the next agent. Type validation is performed at
        each step to ensure data compatibility.

        Args:
            input_data: The input data to process through the pipeline. Must match
                the pipeline's input_schema type.
            name: Optional name to identify this pipeline run, passed to each agent.

        Returns:
            The final output after processing through all agents in the pipeline.
            Will match the pipeline's output_schema type.

        Raises:
            AgentyTypeError: If agent's output type doesn't match the next agent's input schema.
            AgentyValueError: If the pipeline contains no agents.

        Example:
            >>> pipeline = extractor | processor | formatter
            >>> result = await pipeline.run(
            ...     input_data="Raw text to process",
            ...     name="document-123"
            ... )
        """
        if not self.agents:
            raise AgentyValueError("Pipeline must contain at least one agent")

        def validate_data(data: Any, schema: Type[AgentIO], context: str) -> Any:
            """Validate data against a schema."""
            try:
                return TypeAdapter(schema).validate_python(data)
            except ValidationError:
                raise AgentyTypeError(
                    f"{context} data type {type(data)} does not match schema {schema}"
                )

        current_input = validate_data(input_data, self.input_schema, "Pipeline input")
        output = None

        self.reset()
        for i, agent in enumerate(self.agents, 0):
            self.current_step = i
            typed_input = validate_data(
                current_input, agent.input_schema, f"Agent {i} input"
            )
            output = await agent.run(typed_input, name=name)
            current_input = output
            self.output_history.append(output)

        final_output = validate_data(output, self.output_schema, "Pipeline output")
        return cast(AgentOutputT, final_output)

    def reset(self) -> None:
        """Reset the pipeline to its initial state."""
        self.current_step: Optional[int] = None
        self.output_history = []
        for agent in self.agents:
            agent.reset()

    def __or__(
        self, other: AgentIOProtocol[AgentOutputT, PipelineOutputT]
    ) -> AgentIOProtocol[AgentInputT, PipelineOutputT]:
        """Chain this pipeline with another agent using the | operator.

        This enables fluent pipeline construction using the | operator to chain
        multiple agents together. The output type of this pipeline must match
        the input type of the other agent.

        Args:
            other: Another agent to append to this pipeline. Its input schema must
                match this pipeline's output schema.

        Returns:
            A new Pipeline instance containing both this pipeline's agents and
            the new agent, preserving type information for the full chain.

        Example:
            >>> # Chain multiple agents to create a processing pipeline
            >>> pipeline = (
            ...     extractor |      # Extract entities
            ...     classifier |     # Classify entities
            ...     formatter        # Format results
            ... )
        """
        # Create a new pipeline with combined agents
        new_pipeline = Pipeline[AgentInputT, PipelineOutputT](
            agents=self.agents + [other],
            input_schema=self.input_schema,
            output_schema=other.output_schema,
        )
        return new_pipeline
