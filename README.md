# Agenty

A Pythonic framework for building AI agents and LLM pipelines, powered by [pydantic-ai](https://github.com/pydantic/pydantic-ai). The framework emphasizes simplicity and maintainability without sacrificing power, making it an ideal choice for both rapid prototyping.

> [!Caution]
> **Initial Development**: Agenty is under active development. Expect frequent breaking changes until we reach a stable release.

Agenty provides a clean, type-safe interface for creating:
- Conversational AI agents with structured inputs and outputs
- LLM pipelines
- Complex agent interactions with minimal boilerplate

## Key Features
- Intuitive Pythonic interfaces that feel natural to use
- Jinja2 templates for prompts and messages for dynamic context
- Automatic conversation history management
- Structured Agent I/O for predictable behavior
- Built on pydantic-ai for type validation

Whether you're building a simple chatbot or a complex multi-agent system, Agenty helps you focus on logic rather than infrastructure.
The framework is currently only officially supported with the OpenAI API (through a proxy such as [LiteLLM](https://docs.litellm.ai/docs/simple_proxy)/[OpenRouter](https://openrouter.ai/docs/quick-start)) although theoretically it supports all the models supported by pydantic-ai.

> [!TIP]
> Looking for a more mature alternative? Check out [atomic-agents](https://github.com/BrainBlend-AI/atomic-agents), which heavily inspired this project.

## Installation

```bash
pip install agenty
```

Or with Poetry:

```bash
poetry add agenty
```

## Getting Started
### Basic Usage

Here's a simple example to get started:
```python
import asyncio
from pydantic_ai.models.openai import OpenAIModel
from agenty import Agent

async def main():
    agent = Agent(
        model=OpenAIModel(
            "gpt-4o",
            api_key="your-api-key"
        ),
        system_prompt="You are a helpful and friendly AI assistant."
    )

    response = await agent.run("Hello, how are you?")
    print(response)

asyncio.run(main())
```
In most cases, to build a custom AI agent, you'll want to create your own class that inherits from `Agent.` The below is functionally equivalent to the above code (and is the recommended way to use this framework)
```python
import asyncio
from agenty import Agent

class Assistant(Agent):
    model = OpenAIModel("gpt-4o", api_key="your-api-key")
    system_prompt = "You are a helpful and friendly AI assistant."

async def main():
    agent = Assistant()
    response = await agent.run("Hello, how are you?")
    print(response)

asyncio.run(main())
```
---

### Templates

Agenty uses [Jinja templates](https://jinja.palletsprojects.com/en/stable/templates/) to create dynamic prompts and messages by automatically populating template variables.


> [!TIP]
> Any attribute of your agent object that starts with a capital letter is automatically added to the template context. Agenty prefers to use template variables in `ALL_CAPS` formatting for better visibility. 
> 
> If you need to modify this default behavior, you can override the `template_context()` method in your custom agent class.

```python
import asyncio
from agenty import Agent
from pydantic_ai.models.openai import OpenAIModel

class Greeter(Agent):
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "You are a greeter. You speak in a {{TONE}} tone. Your response length should be {{RESPONSE_LENGTH}}."
    TONE: str = "friendly"
    VERBOSITY: str = "verbose"

async def main():
    agent = Greeter()
    response = await agent.run("Hello, please greet me!")
    print(response)
    agent.TONE = "angry"
    agent.RESPONSE_LENGTH = "very short"
    response = await agent.run("Hello, please greet me!")
    print(response)
    # Sample Output:
    # Hello there! It's wonderful to see you here. I hope you're having a fantastic day! If there's anything you'd like to talk about or explore, I'm all ears. Welcome! 😊
    # What do you want?!
asyncio.run(main())
```

Templates can be used in both system prompts and user messages, making it easy to create dynamic and evolving interactions. 

---

### Tool Usage
Agenty provides a framework for building custom agents that can leverage functions as tools through a simple decorator pattern.

1. **Define Your Agent:** Create a custom class that inherits from the base Agent class.

2. **Implement Tool Methods**: Add methods to your agent class that will serve as tools. Each method should include a docstring that describes the tool. You can even add parameter descriptions in the docstring and pydantic-ai implements [griffe](https://mkdocstrings.github.io/griffe/) to automatically generate tool parameter descriptions.

3. **Register Tools:** Use the `@tool` decorator to mark methods as tools. The decorator automatically registers these methods, making them available for your agent to use during execution. No additional configuration is needed.

Here's an example of a roulette game agent:
```python
import asyncio
import random

from agenty import Agent, tool
from pydantic_ai.models.openai import OpenAIModel


class RouletteAgent(Agent):
    model = OpenAIModel("gpt-4o", api_key="your-api-key")
    system_prompt = "You're a dice game, you should roll the die and see if the number matches the user's guess."

    def __init__(self, player_name: str, num_sides: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.player_name = player_name
        self.num_sides = num_sides

    @tool
    def get_player_name(self) -> str:
        """Get the player's name."""
        return self.player_name

    @tool
    def roll_die(self) -> int:
        """Roll a n-sided die and return the result."""
        num = random.randint(1, self.num_sides)
        print(f"Rolled a {num}!")
        return num


async def main():
    agent = RouletteAgent(player_name="John", num_sides=6)
    response = await agent.run("I guess the number will be 3!")
    print(response)


asyncio.run(main())
```

You can read more about [function tools](https://ai.pydantic.dev/tools/) by pydantic-ai. (underlying implementation of agenty tools)

---

### Structured Input/Output
Agenty supports structured input and output types through pydantic models (inherit from `agenty.types.BaseIO`). This enables type-safe interactions with your agents. 

Here's an example that extracts user information from text:

```python
import asyncio
from typing import List
from pydantic_ai.models.openai import OpenAIModel
from agenty import Agent
from agenty.types import BaseIO

class User(BaseIO):
    name: str
    age: int
    hobbies: List[str]

class UserExtractor(Agent[str, List[User]]): # Generics are used for static type-checking
    input_schema = str  # Controls input type
    output_schema = List[User]  # Controls output type
    model = OpenAIModel("gpt-4o", api_key="your-api-key")
    system_prompt = "Extract all user information"

async def main():
    agent = UserExtractor()
    story = """At the community center, Emma, 32, was painting a vibrant sunset while Jake, 27, captured it through his camera lens. Nearby, Sophia, 35, a runner and yoga enthusiast, was practicing stretches after her morning jog. Ben, 30, a fitness coach, was doing 
push-ups in the corner, taking a break from his workout. Each of them enjoyed their unique hobbies, creating a lively atmosphere filled with creativity, fitness, and relaxation. They shared stories about their passions, encouraging one another to pursue
what they loved."""
    
    # Static type-checkers correctly detect that input must be str and output must be List[User]
    users = await agent.run(story)
    
    for user in users:
        print(f"Name: {user.name}")
        print(f"Age: {user.age}")
        print(f"Hobbies: {', '.join(user.hobbies)}")
        print()

asyncio.run(main())

# Output:
# Name: Emma
# Age: 32
# Hobbies: painting

# Name: Jake
# Age: 27
# Hobbies: photography

# Name: Sophia
# Age: 35
# Hobbies: running, yoga

# Name: Ben
# Age: 30
# Hobbies: fitness coaching, working out
```

---
## Configuration

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

---
## Requirements

- Python >= 3.12

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Jonathan Chun ([@jonchun](https://github.com/jonchun))
