# Agent Configuration

Agents in Agenty can be customized through class attributes and initialization parameters. This guide covers all available configuration options and best practices for configuring agents.

## Basic Configuration

Here's a complete list of configurable attributes with their descriptions:

```python
from typing import Optional, Union, Type
from pydantic_ai.agent import EndStrategy
from pydantic_ai.models import KnownModelName, Model, ModelSettings
from agenty import Agent
from agenty.types import AgentIO

class CustomAgent(Agent):
    # Required Configuration
    input_schema: Type[AgentIO]  # Defines the expected input type
    output_schema: Type[AgentIO]  # Defines the expected output type
    
    # Model Configuration
    model: Union[KnownModelName, Model] = "gpt-4o"  # The AI model to use
    model_settings: Optional[ModelSettings] = None  # Model-specific settings
    
    # Behavior Configuration
    system_prompt: str = ""  # System instructions for the agent
    retries: int = 1  # Number of retries for failed runs
    result_retries: Optional[int] = None  # Number of retries for result parsing
    end_strategy: EndStrategy = "early"  # Strategy for ending conversations
```

## Configuration Methods

There are two ways to configure an agent:

1. Class-level configuration:
```python
class ChatAgent(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = "gpt-4o"
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

## Memory and Usage Tracking

Agents include built-in memory and usage tracking capabilities that can be configured during initialization:

```python
from agenty.components.memory import AgentMemory
from agenty.components.usage import AgentUsage, AgentUsageLimits

agent = ChatAgent(
    memory=AgentMemory(),  # Custom memory component
    usage=AgentUsage(),  # Usage tracking
    usage_limits=AgentUsageLimits()  # Usage limits configuration
)
```

## Template Context

The system prompt supports template variables that are automatically populated from the agent's attributes. 
Variables that start with an uppercase letter are automatically included in the template context. Agenty's developer
prefers template variables to be `ALL_CAPS` for visibility.:

```python
class CustomAgent(Agent[str, str]):
    system_prompt = "You are an AI assistant named {{ ASSISTANT_NAME }}."
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ASSISTANT_NAME = "Helper"  # Will be used in the template
```

## Best Practices

1. **Input/Output Types**: Always explicitly define `input_schema` and `output_schema` for type safety:
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

2. **Explicit model**: Define your model explicitly:
```python
from pydantic_ai.models.openai import OpenAIModel

class CustomAgent(Agent):
    model = OpenAIModel(
        "gpt-4",
        base_url="https://api.openai.com/v1",
        api_key="your-api-key"
    )
```
See [pydantic-ai models](https://ai.pydantic.dev/api/models/base/) for all available options.


3. **Error Handling**: Set appropriate retry values based on your use case:
```python
class RobustAgent(Agent):
    retries = 3  # Retry failed runs up to 3 times
    result_retries = 2  # Retry result parsing up to 2 times
```

4. **Model Settings**: Use model_settings for fine-tuned control. 
```python
from pydantic_ai.models import ModelSettings

class PreciseAgent(Agent):
    model_settings = ModelSettings(
        temperature=0.1,  # Lower temperature for more focused outputs
        max_tokens=500
    )
```
See [pydantic-ai model settings](https://ai.pydantic.dev/api/settings/#pydantic_ai.settings.ModelSettings) for all available options.

5. **Memory Management**: Set custom memory settings when needed:
```python
agent = Agent(
    memory=AgentMemory(max_messages=10)  # Limit memory to last 10 messages
)
```
