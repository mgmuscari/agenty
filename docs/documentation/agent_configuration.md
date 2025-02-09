# Agent Configuration

Agents in Agenty can be customized through class attributes and initialization parameters. This guide covers all available configuration options and best practices for configuring agents.

## Basic Configuration

Here's a complete list of configurable attributes with their descriptions:

```python
from typing import Optional, Union, Type
from agenty import Agent
from agenty.agent import EndStrategy
from agenty.models import KnownModelName, Model, ModelSettings
from agenty.types import AgentIO

class CustomAgent(Agent):
    # Required Configuration
    input_schema: Type[AgentIO]  # Defines the expected input type
    output_schema: Type[AgentIO]  # Defines the expected output type

    # Model Configuration
    model: Optional[Model] = None  # The AI model to use
    model_settings: Optional[ModelSettings] = None  # Model-specific settings

    # Behavior Configuration
    name: str = ""  # Optional name for the agent
    system_prompt: str = ""  # System instructions for the agent
    retries: int = 1  # Number of retries for failed runs
    result_retries: Optional[int] = None  # Number of retries for result parsing
    end_strategy: EndStrategy = "early"  # Strategy for ending conversations
```

## Configuration Methods

There are two ways to configure an agent:

1. Class-level configuration:

```python
from agenty.models import OpenAIModel

class ChatAgent(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = OpenAIModel(
        "gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
    system_prompt = "You are a helpful assistant."
```

2. Instance-level configuration:

```python
agent = ChatAgent(
    model="gpt-3.5-turbo",
    system_prompt="You are a specialized coding assistant.",
    retries=2
)
```

## Chat History

Agents include built-in chat history tracking through the ChatHistory class:

```python
from agenty.agent import ChatHistory

agent = ChatAgent(
    chat_history=ChatHistory()  # Custom chat history configuration
)
```

## Template Context

The system prompt supports template variables that are automatically populated from the agent's attributes.
Variables that start with an uppercase letter are automatically included in the template context:

```python
class CustomAgent(Agent[str, str]):
    system_prompt = "You are an AI assistant named {{ ASSISTANT_NAME }}."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ASSISTANT_NAME = "Helper"  # Will be used in the template
```

## Best Practices

### Input/Output Types

Always explicitly define `input_schema` and `output_schema` for type safety:

```python
from pydantic import BaseModel

class UserInput(BaseModel):
    question: str
    context: Optional[str] = None

class AgentResponse(BaseModel):
    answer: str
    confidence: float

class AnalysisAgent(Agent[UserInput, AgentResponse]):
    input_schema = UserInput
    output_schema = AgentResponse
```

---

### Explicit model

Define your model explicitly:

```python
from agenty.models import OpenAIModel

class CustomAgent(Agent):
    model = OpenAIModel(
        "gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
```

---

### Error Handling

Set appropriate retry values based on your use case:

```python
class RobustAgent(Agent):
    retries = 3  # Retry failed runs up to 3 times
    result_retries = 2  # Retry result parsing up to 2 times
```

---

### Model Settings

Use model_settings for fine-tuned control:

```python
from agenty.models import ModelSettings

class PreciseAgent(Agent):
    model_settings = ModelSettings(
        temperature=0.1,  # Lower temperature for more focused outputs
        max_tokens=500
    )
```

See [pydantic-ai model settings](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings) for all available options.
