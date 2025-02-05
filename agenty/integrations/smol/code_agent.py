import asyncio
import json
import os
from typing import Dict, List, Any, cast, Optional, Union
from pydantic.type_adapter import TypeAdapter
from pydantic import ValidationError
from agenty import Agent
from agenty.exceptions import UnsupportedModel, InvalidResponse, AgentyTypeError
from agenty.types import (
    AgentInputT,
    AgentOutputT,
    BaseIO,
)

try:
    from smolagents.agents import (
        CODE_SYSTEM_PROMPT,
        CodeAgent as smolCodeAgent,
        Tool as smolTool,
    )
    from smolagents.models import (
        Model as smolModel,
        OpenAIServerModel,
        LiteLLMModel,
    )
    from smolagents.agent_types import AgentText
except ImportError as _import_error:
    raise ImportError(
        "Please install `smolagents` to use this integration: "
        "you can use the `smol` optional group — `pip install 'agenty[smol]'`"
    ) from _import_error


class SmolCodeAgent(Agent[AgentInputT, AgentOutputT]):
    """Agent that uses smolagents CodeAgent for code-related tasks.

    This agent wraps smolagents.CodeAgent to provide code generation, analysis,
    and execution capabilities while maintaining compatibility with the agenty
    framework.

    Attributes:
        smol_grammar: Optional grammar rules for code generation
        smol_additional_authorized_imports: Additional Python imports to allow
        smol_planning_interval: Interval for planning steps
        smol_use_e2b_executor: Whether to use e2b code executor
        smol_max_print_outputs_length: Maximum length of print outputs
        smol_tools: List of smolagents tools to use
        smol_verbosity: Verbosity level for smolagents (0-2)

        system_prompt: Default system prompt
        input_schema: Recommended to use str for compatibility with smolagents
        output_schema: Recommended to use str for compatibility with smolagents
    """

    smol_grammar: Dict[str, str] | None = None
    smol_additional_authorized_imports: List[str] | None = None
    smol_planning_interval: int | None = None
    smol_use_e2b_executor: bool = False
    smol_max_print_outputs_length: int | None = None
    smol_tools: List[smolTool] = []
    smol_verbosity_level: int = 0

    system_prompt = CODE_SYSTEM_PROMPT
    input_schema = str
    output_schema = str

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the SmolCodeAgent.

        Args:
            smol_tools: Additional smolagents tools to use
            *args: Arguments passed to parent Agent class
            **kwargs: Keyword arguments passed to parent Agent class
        """
        # Extract smol-specific kwargs
        smol_kwargs = {}
        for key in list(kwargs):
            if key.startswith("smol_"):
                smol_kwargs[key[5:]] = kwargs.pop(key)

        super().__init__(*args, **kwargs)

        # Initialize agent configuration
        self.smol_agent = None
        self._smol_kwargs = {
            **smol_kwargs,
            "tools": self.smol_tools + smol_kwargs.get("tools", []),
        }

    async def get_smol_agent(self, **kwargs: Any) -> smolCodeAgent:
        """Create a smolagents CodeAgent instance.

        Args:
            **kwargs: Configuration options for the CodeAgent. These are passed directly
                to the smolagents CodeAgent constructor and may include options like
                grammar, additional_authorized_imports, planning_interval, etc.

        Returns:
            Configured smolagents CodeAgent instance
        """
        return smolCodeAgent(
            model=self.get_smol_model(),
            system_prompt=self.system_prompt,
            **kwargs,
        )

    def get_smol_model(self) -> smolModel:
        """Convert pydantic-ai model to smolagents model.

        Returns:
            Configured smolagents model

        Raises:
            UnsupportedModel: If model type is not supported
        """
        pai_model = self.pai_agent.model
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.groq import GroqModel
        from pydantic_ai.models.mistral import MistralModel
        from pydantic_ai.models.cohere import CohereModel

        if isinstance(pai_model, OpenAIModel):
            return OpenAIServerModel(
                model_id=self.model_name,
                api_key=pai_model.client.api_key,
                api_base=str(pai_model.client.base_url),
                organization=pai_model.client.organization,
                project=pai_model.client.project,
            )
        elif isinstance(pai_model, AnthropicModel):
            return LiteLLMModel(
                model_id=f"anthropic/{self.model_name}",
                api_key=pai_model.client.api_key,
            )
        elif isinstance(pai_model, GroqModel):
            return LiteLLMModel(
                model_id=f"groq/{self.model_name}",
                api_key=pai_model.client.api_key,
            )
        elif isinstance(pai_model, CohereModel):
            cohere_key = os.environ.get("COHERE_API_KEY", "")
            if not cohere_key:
                raise UnsupportedModel(
                    f"Unable to automatically fetch API key for {type(pai_model)}. Set COHERE_API_KEY via environment variable instead."
                )
            return LiteLLMModel(
                model_id=f"{self.model_name}",
                api_key=cohere_key,
            )
        elif isinstance(pai_model, MistralModel):
            mistral_key = pai_model.client.sdk_configuration.security
            from mistralai.models import Security

            if not isinstance(mistral_key, Security):
                raise UnsupportedModel(
                    f"Unable to automatically fetch API key for {type(pai_model)}. Set MISTRAL_API_KEY via environment variable instead."
                )
            return LiteLLMModel(
                model_id=f"mistral/{self.model_name}",
                api_key=mistral_key.api_key,
            )
        else:
            raise UnsupportedModel(f"{type(pai_model)} not supported")

    async def run(
        self,
        input_data: AgentInputT,
        name: Optional[str] = None,
    ) -> AgentOutputT:
        """Run the agent with the provided input.

        Args:
            input_data: The input prompt for the agent

        Returns:
            The agent's response as a string

        Raises:
            ValidationError: If response validation fails
            InvalidResponse: If response conversion fails
        """
        if self.smol_agent is None:
            self.smol_agent = await self.get_smol_agent(**self._smol_kwargs)
        input_str = str(input_data)
        self.memory.add("user", input_str)
        resp = await asyncio.to_thread(self.smol_agent.run, input_str)

        if isinstance(resp, AgentText):
            resp = resp.to_string()

        try:
            adapter = TypeAdapter(self.output_schema)

            typed_resp = adapter.validate_strings(str(resp))
        except ValidationError:
            # Special case for converting float to int. Unsure if desirable.
            if isinstance(resp, float) and self.output_schema is int:
                return cast(AgentOutputT, int(resp))
            raise AgentyTypeError(
                f"Unable to convert response to expected type. Response Type: {type(resp)} | Schema: {self.output_schema}"
            )

        return cast(AgentOutputT, typed_resp)
