Custom agents can be customized with the following class attributes. The imports have been included below as well for convenience. The settings get passed along to the pydantic-ai agent that powers everything.

```python
from typing import Optional, Union, Type

from agenty import Agent
from agenty.types import AgentIO
from pydantic_ai.agent import EndStrategy
from pydantic_ai.models import KnownModelName, Model, ModelSettings

class CustomAgent(Agent):
    model: Union[KnownModelName, Model] = "gpt-4o"
    system_prompt: str = ""
    model_settings: Optional[ModelSettings]
    input_schema: Type[AgentIO]
    output_schema: Type[AgentIO]
    retries: int
    result_retries: Optional[int]
    end_strategy: EndStrategy
```
