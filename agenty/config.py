from dataclasses import dataclass
from typing import Optional, Union, Type

from pydantic_ai.models import KnownModelName, Model, ModelSettings
from pydantic_ai.agent import EndStrategy

from agenty.types import AgentIO
from agenty.components.usage import AgentUsage, AgentUsageLimits


@dataclass
class AgentConfig:
    model: Union[KnownModelName, Model] = "gpt-4o"
    system_prompt: str = ""
    model_settings: Optional[ModelSettings] = None
    usage: Optional[AgentUsage] = None
    usage_limits: Optional[AgentUsageLimits] = None
    input_schema: Type[AgentIO] = str
    output_schema: Type[AgentIO] = str
    retries: int = 1
    result_retries: Optional[int] = None
    end_strategy: EndStrategy = "early"
