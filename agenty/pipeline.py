from typing import Any, Generic, List, Type, Optional

from agenty.types import (
    AgentIO,
    AgentInputT,
    AgentOutputT,
    PipelineOutputT,
    NOT_GIVEN,
    NOT_GIVEN_,
    is_sequence_type,
    get_sequence_item_type,
)
from agenty.exceptions import AgentyTypeError
from agenty.protocol import AgentProtocol


def _validate_schema(data: Any, schema: Type[AgentIO], context: str) -> None:
    """Validate that data matches the expected schema type.

    Args:
        data: The data to validate
        schema: The expected schema type
        context: Context string for error message (e.g. 'input' or 'output')

    Raises:
        AgentyTypeError: If data type doesn't match the schema
    """
    check_type: Any = type(data)
    check_schema: Any = schema

    if is_sequence_type(check_type) and is_sequence_type(schema):
        check_schema = get_sequence_item_type(schema)
        try:
            check_type = type(data[0])
        except IndexError:
            # list is empty so technically type is correct
            check_type = check_schema

    if check_type is not check_schema:
        raise AgentyTypeError(
            f"{context.capitalize()} data type {type(data)} does not match schema {schema}"
        )


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
        agents: List[AgentProtocol[Any, Any]] = list(),
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
        input_data: AgentInputT,
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
        """
        current_input: Any = input_data
        res: Any = None
        self.current_step = 0

        for agent in self.agents:
            _validate_schema(
                data=current_input,
                schema=agent.input_schema,
                context="input",
            )
            res = await agent.run(current_input)
            current_input = res
            self.output_history.append(res)
            self.current_step += 1

        _validate_schema(
            data=res,
            schema=self.output_schema,
            context="output",
        )
        return res

    def __or__(
        self, other: AgentProtocol[AgentOutputT, PipelineOutputT]
    ) -> AgentProtocol[AgentInputT, PipelineOutputT]:
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
