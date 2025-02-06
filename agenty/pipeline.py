from typing import Any, Generic, List, Type, Optional

from pydantic import TypeAdapter, ValidationError

from agenty.types import (
    AgentIO,
    AgentInputT,
    AgentOutputT,
    PipelineOutputT,
    NOT_GIVEN,
    NOT_GIVEN_,
)
from agenty.exceptions import AgentyTypeError, AgentyValueError
from agenty.protocol import AgentIOProtocol


class Pipeline(Generic[AgentInputT, AgentOutputT]):
    """A pipeline for chaining multiple agents together for sequential processing.

    The Pipeline class enables the creation of agent chains where data flows from one
    agent to the next. Each agent in the pipeline must have compatible input/output
    schemas, where each agent's output schema matches the next agent's input schema.

    Type Parameters:
        AgentInputT: The type of input the pipeline accepts
        AgentOutputT: The type of output the pipeline produces

    Attributes:
        input_schema: The expected schema for pipeline input data
        output_schema: The expected schema for pipeline output data
        agents: List of agents in the pipeline, executed in order

    Example:
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
        input_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
        output_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
    ) -> None:
        super().__init__()

        if not isinstance(input_schema, NOT_GIVEN):
            self.input_schema = input_schema
        if not isinstance(output_schema, NOT_GIVEN):
            self.output_schema = output_schema
        self.agents = agents
        self.output_history = []
        self.current_step: Optional[int] = None

    async def run(
        self,
        input_data: Optional[AgentInputT],
        name: Optional[str] = None,
    ) -> AgentOutputT:
        """Run the pipeline by executing each agent in sequence.

        Args:
            input_data: The input data to process through the pipeline. Must match
                the pipeline's input_schema type.

        Returns:
            The final output after processing through all agents in the pipeline.
            Will match the pipeline's output_schema type.

        Raises:
            AgentyTypeError: If input data type doesn't match an agent's input schema,
                or if an agent's output type doesn't match the pipeline's output schema.
            AgentyValueError: If the pipeline contains no agents.
        """
        current_input: Any = input_data
        output: Any = None
        self.current_step = 0

        agent = None

        for agent in self.agents:
            adapter = TypeAdapter(agent.input_schema)
            try:
                typed_input = adapter.validate_python(current_input)
            except ValidationError:
                raise AgentyTypeError(
                    f"Input data type {type(current_input)} does not match schema {agent.input_schema}"
                )
            output = await agent.run(
                typed_input,
                name=name,
            )
            current_input = output
            self.output_history.append(output)
            self.current_step += 1

        if agent is None:
            raise AgentyValueError("Pipeline must contain at least one agent")

        adapter = TypeAdapter(agent.output_schema)
        try:
            typed_input = adapter.validate_python(output)
        except ValidationError:
            raise AgentyTypeError(
                f"Input data type {type(current_input)} does not match schema {agent.output_schema}"
            )

        return output

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
        """
        return Pipeline[AgentInputT, PipelineOutputT](
            agents=[self, other],
            input_schema=self.input_schema,
            output_schema=other.output_schema,
        )
