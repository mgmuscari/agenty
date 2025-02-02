from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    Union,
    cast,
)
import logging

import pydantic_ai as pai
from pydantic_ai.agent import EndStrategy
from pydantic_ai.models import KnownModelName, Model, ModelSettings

from agenty.components.memory import AgentMemory, ChatMessage
from agenty.components.usage import AgentUsage, AgentUsageLimits
from agenty.exceptions import AgentyValueError
from agenty.template import apply_template
from agenty.types import AgentIO, AgentInputT, AgentOutputT

__all__ = ["Agent"]

logger = logging.getLogger(__name__)


class AgentMeta(type):
    """Metaclass for Agent that handles tool registration and agent configuration.

    This metaclass automatically processes tool decorators and configures the underlying
    pydantic-ai agent during class creation.

    Args:
        name (str): The name of the class being created
        bases (tuple[type, ...]): Base classes
        namespace (dict[str, Any]): Class namespace dictionary

    Returns:
        Type: The configured agent class
    """

    def __new__(
        mcls: Type["AgentMeta"],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        tools: List[Callable] = []
        for key, value in namespace.items():
            if hasattr(value, "_is_tool"):
                tools.append(value)
        cls = super().__new__(mcls, name, bases, namespace)
        try:
            pai_agent = pai.Agent(
                namespace.get("model"),
                deps_type=cls,
                name=name,
                result_type=getattr(cls, "output_schema", str),
                system_prompt=getattr(cls, "system_prompt", ""),
                model_settings=getattr(cls, "model_settings", None),
                retries=getattr(cls, "retries", 1),
                result_retries=getattr(cls, "result_retries", None),
                end_strategy=getattr(cls, "end_strategy", "early"),
            )
            # Set the pai agent as a private class attribute
            setattr(cls, "_pai_agent", pai_agent)

            # Add tools to the pai agent
            # TODO: Add support for tool decorator with parameters
            tool_decorator = pai_agent.tool(
                retries=None,
                docstring_format="auto",
                require_parameter_descriptions=False,
            )
            for tool in tools:
                tool_decorator(tool)
                logger.debug(
                    {
                        "tool": tool.__name__,
                        "agent": cls.__name__,
                        "msg": "added tool to agent",
                    }
                )
        except Exception:
            pass
        return cls


class NOT_GIVEN:
    """Sentinel class used to distinguish between unset and None values."""

    ...


NOT_GIVEN_ = NOT_GIVEN()


class Agent(Generic[AgentInputT, AgentOutputT], metaclass=AgentMeta):
    """Base class for creating AI agents with specific input and output types.

    This class provides the foundation for creating AI agents with type-safe inputs
    and outputs, memory management, usage tracking, and tool integration.

    Attributes:
        model (Union[KnownModelName, Model]): The AI model to use
        system_prompt (str): System prompt for the agent
        model_settings (Optional[ModelSettings]): Model-specific settings
        input_schema (Type[AgentIO]): Input validation schema
        output_schema (Type[AgentIO]): Output validation schema
        retries (int): Number of retries for failed runs
        result_retries (Optional[int]): Number of retries for result parsing
        end_strategy (EndStrategy): Strategy for ending conversations
    """

    model: Union[KnownModelName, Model] = "gpt-4o"
    system_prompt: str = ""
    model_settings: Optional[ModelSettings] = None
    input_schema: Type[AgentIO] = str
    output_schema: Type[AgentIO] = str
    retries: int = 1
    result_retries: Optional[int] = None
    end_strategy: EndStrategy = "early"

    _pai_agent: pai.Agent["Agent[Any, Any]", AgentIO]

    def __init__(
        self,
        model: Union[KnownModelName, Model] | NOT_GIVEN = NOT_GIVEN_,
        model_settings: Optional[ModelSettings] | NOT_GIVEN = NOT_GIVEN_,
        input_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
        output_schema: Type[AgentIO] | NOT_GIVEN = NOT_GIVEN_,
        retries: int | NOT_GIVEN = NOT_GIVEN_,
        result_retries: Optional[int] | NOT_GIVEN = NOT_GIVEN_,
        end_strategy: EndStrategy | NOT_GIVEN = NOT_GIVEN_,
        system_prompt: str = "",
        name: Optional[str] = None,
        memory: Optional[AgentMemory] = None,
        usage: Optional[AgentUsage] = None,
        usage_limits: Optional[AgentUsageLimits] = None,
    ) -> None:
        """Initialize a new Agent instance.

        Args:
            model: The AI model to use
            model_settings: Model-specific settings
            input_schema: Input validation schema
            output_schema: Output validation schema
            retries: Number of retries for failed runs
            result_retries: Number of retries for result parsing
            system_prompt: System prompt for the agent
            end_strategy: Strategy for ending conversations
            memory: Memory component for storing conversation history
            usage: Usage tracking component
            usage_limits: Usage limits configuration
        """
        if system_prompt:
            self.system_prompt = system_prompt
        self.memory = memory or AgentMemory()
        self.usage = usage or AgentUsage()
        self.usage_limits = usage_limits or AgentUsageLimits()
        self.name = name or self.__class__.__name__

        # Update instance attributes if provided
        _regenerate_pai_agent = False
        if not isinstance(model, NOT_GIVEN):
            self.model = model
            _regenerate_pai_agent = True
        if not isinstance(model_settings, NOT_GIVEN):
            self.model_settings = model_settings
            _regenerate_pai_agent = True
        if not isinstance(input_schema, NOT_GIVEN):
            self.input_schema = input_schema
            _regenerate_pai_agent = True
        if not isinstance(output_schema, NOT_GIVEN):
            self.output_schema = output_schema
            _regenerate_pai_agent = True
        if not isinstance(retries, NOT_GIVEN):
            self.retries = retries
            _regenerate_pai_agent = True
        if not isinstance(result_retries, NOT_GIVEN):
            self.result_retries = result_retries
            _regenerate_pai_agent = True
        if not isinstance(end_strategy, NOT_GIVEN):
            self.end_strategy = end_strategy
            _regenerate_pai_agent = True

        # If any attributes were updated, recreate the pydantic-ai agent
        if _regenerate_pai_agent:
            logger.debug("Creating object-specific pydantic-ai agent")
            self._pai_agent = pai.Agent(
                self.model,
                result_type=self.output_schema,
                deps_type=self.__class__,
                model_settings=self.model_settings,
                retries=self.retries,
                result_retries=self.result_retries,
                end_strategy=self.end_strategy,
            )

    async def run(
        self,
        input_data: AgentInputT,
    ) -> AgentOutputT:
        """Run the agent with the provided input.

        Args:
            input_data: The input data for the agent to process

        Returns:
            The processed output data
        """
        self.memory.add("user", input_data)

        system_prompt = ChatMessage(
            role="system", content=self.system_prompt
        ).to_pydantic_ai(ctx=self.template_context())

        result = await self.pai_agent.run(
            str(input_data),
            message_history=[system_prompt]
            + self.memory.to_pydantic_ai(ctx=self.template_context()),
            deps=self,
            usage_limits=self.usage_limits[self.model_name],
            usage=self.usage[self.model_name],
            # Note: Omitting result_type allows None schema which enables raw text responses
        )
        if result.data is None:
            raise AgentyValueError("No data returned from agent")

        self.memory.add("assistant", result.data)
        return cast(AgentOutputT, result.data)

    async def run_stream(
        self,
        input_data: AgentInputT,
    ) -> AsyncIterator[AgentOutputT]:
        self.memory.add("user", input_data)

        system_prompt = ChatMessage(
            role="system", content=self.system_prompt
        ).to_pydantic_ai(ctx=self.template_context())

        async with self.pai_agent.run_stream(
            str(input_data),
            message_history=[system_prompt]
            + self.memory.to_pydantic_ai(ctx=self.template_context()),
            deps=self,
            usage_limits=self.usage_limits[self.model_name],
            usage=self.usage[self.model_name],
        ) as result:
            async for message in result.stream():
                # TODO: Add to agent's memory at appropriate point in stream
                yield cast(AgentOutputT, message)

    def render_system_prompt(self) -> str:
        """Render the system prompt with the current template context.

        Returns:
            str: The rendered system prompt
        """
        return apply_template(self.system_prompt, self.template_context())

    def template_context(self) -> Dict[str, Any]:
        """Get a dictionary of instance variables for use in templates. By default, this includes all variables that start with an uppercase letter.

        Returns:
            Dict[str, Any]: Dictionary of variables
        """
        return {key: getattr(self, key) for key in dir(self) if key[0].isupper()}

    @property
    def model_name(self) -> str:
        """Get the name of the current model.

        Returns:
            str: The model name

        Raises:
            ValueError: If no model is set
        """
        if self.pai_agent.model is None:
            raise AgentyValueError("Model is not set")

        if isinstance(self.pai_agent.model, str):
            model_name = self.pai_agent.model
        else:
            model_name = getattr(
                self.pai_agent.model,
                "model_name",
                self.pai_agent.model.name(),
            )

        return model_name

    @property
    def pai_agent(self) -> pai.Agent["Agent[Any, Any]", AgentIO]:
        """Get the underlying pydantic-ai agent instance.

        Returns:
            pai.Agent: The pydantic-ai agent instance
        """
        return self._pai_agent
