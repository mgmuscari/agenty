from pydantic_ai import models
from pydantic_ai.models import Model, KnownModelName, infer_model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.models.vertexai import VertexAIModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.test import TestModel


def allow_model_requests(allow: bool) -> None:
    models.ALLOW_MODEL_REQUESTS = allow


__all__ = [
    "allow_model_requests",
    "infer_model",
    "Model",
    "KnownModelName",
    "ModelSettings",
    "AnthropicModel",
    "CohereModel",
    "FunctionModel",
    "AgentInfo",
    "GroqModel",
    "MistralModel",
    "GeminiModel",
    "VertexAIModel",
    "OpenAIModel",
    "TestModel",
]
