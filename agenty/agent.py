from typing import Any, Dict, List, Callable, Union, Optional, cast, Generic, Type
from functools import wraps
import logging

import pydantic_ai as pai
from pydantic_ai.agent import EndStrategy
from pydantic_ai.models import KnownModelName, Model, ModelSettings
from pydantic_ai.tools import RunContext

from agenty.components.memory import AgentMemory, ChatMessage
from agenty.components.usage import AgentUsage, AgentUsageLimits
from agenty.template import apply_template
from agenty.types import AgentInputT, AgentOutputT, AgentIO

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
                result_type=namespace.get("output_schema", str),
                system_prompt=namespace.get("system_prompt", ""),
                model_settings=namespace.get("model_settings"),
                retries=namespace.get("retries", 1),
                result_retries=namespace.get("result_retries"),
                end_strategy=namespace.get("end_strategy", "early"),
            )
            setattr(cls, "_pai_agent", pai_agent)
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
                        "msg": f"Added tool to agent",
                    }
                )
        except Exception:
            pass
        return cls


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
    model_settings: Optional[ModelSettings]
    input_schema: Type[AgentIO]
    output_schema: Type[AgentIO]
    retries: int
    result_retries: Optional[int]
    end_strategy: EndStrategy

    _pai_agent: pai.Agent["Agent[Any, Any]", AgentIO]

    def __init__(
        self,
        model: Union[KnownModelName, Model] = "gpt-4o",
        system_prompt: str = "",
        model_settings: Optional[ModelSettings] = None,
        input_schema: Type[AgentIO] = str,
        output_schema: Type[AgentIO] = str,
        retries: int = 1,
        result_retries: Optional[int] = None,
        end_strategy: EndStrategy = "early",
        memory: Optional[AgentMemory] = None,
        usage: Optional[AgentUsage] = None,
        usage_limits: Optional[AgentUsageLimits] = None,
    ) -> None:
        """Initialize a new Agent instance.

        Args:
            model: The AI model to use
            system_prompt: System prompt for the agent
            model_settings: Model-specific settings
            input_schema: Input validation schema
            output_schema: Output validation schema
            retries: Number of retries for failed runs
            result_retries: Number of retries for result parsing
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

        # If any of the following attributes are set, recreate the pydantic-ai agent for the entire class
        if any(
            [
                model != "gpt-4o",
                model_settings is not None,
                input_schema is not str,
                output_schema is not str,
                retries != 1,
                result_retries is not None,
                end_strategy != "early",
            ]
        ):
            logger.debug("Recreating pydantic-ai agent")
            self.__class__._pai_agent = pai.Agent(
                model,
                result_type=output_schema or str,
                deps_type=self.__class__,
                model_settings=model_settings,
                retries=retries,
                result_retries=result_retries,
                end_strategy=end_strategy,
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
        from devtools import debug

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
        )

        self.memory.add("assistant", result.data)
        return cast(AgentOutputT, result.data)

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
            raise ValueError("Model is not set")

        if isinstance(self.pai_agent.model, str):
            model_name = self.pai_agent.model
        else:
            model_name = self.pai_agent.model.name()

        return model_name

    @property
    def pai_agent(self) -> pai.Agent["Agent[Any, Any]", AgentIO]:
        """Get the underlying pydantic-ai agent instance.

        Returns:
            pai.Agent: The pydantic-ai agent instance
        """
        return self._pai_agent
