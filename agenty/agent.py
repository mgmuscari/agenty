from typing import Any, List, Callable, Union, Optional, cast, Generic
from functools import wraps
import logging
from typing import Type

import pydantic_ai as pai
from pydantic_ai.agent import EndStrategy
from pydantic_ai.models import KnownModelName, Model, ModelSettings
from pydantic_ai.tools import RunContext

from agenty.memory import AgentMemory, ChatMessage
from agenty.types import AgentInputT, AgentOutputT, AgentIO
from agenty.usage import AgentUsage, AgentUsageLimits

logger = logging.getLogger(__name__)


class AgentMeta(type):

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
    memory: AgentMemory
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

    @property
    def pai_agent(self) -> pai.Agent["Agent[Any, Any]", AgentIO]:
        return self._pai_agent

    async def run(
        self,
        input_data: AgentInputT,
    ) -> AgentOutputT:
        self.memory.add("user", input_data)
        system_prompt = ChatMessage(
            role="system", content=self.system_prompt
        ).to_pydantic_ai()
        result = await self.pai_agent.run(
            str(input_data),
            message_history=[system_prompt] + self.memory.to_pydantic_ai(),
            deps=self,
            usage_limits=self.usage_limits[self.model_name],
            usage=self.usage[self.model_name],
        )
        for msg in result.new_messages():
            logger.debug(
                {
                    "agent": type(self).__name__,
                    "msg": msg,
                }
            )
        self.memory.add("assistant", result.data)
        return cast(AgentOutputT, result.data)

    @property
    def model_name(self) -> str:
        if self.pai_agent.model is None:
            raise ValueError("Model is not set")

        if isinstance(self.pai_agent.model, str):
            model_name = self.pai_agent.model
        else:
            model_name = self.pai_agent.model.name()

        return model_name
