# Agenty

A Pythonic framework for building AI agents and LLM pipelines, powered by [pydantic-ai](https://github.com/pydantic/pydantic-ai). The framework emphasizes simplicity and maintainability without sacrificing power, making it an ideal choice for both rapid prototyping.

> ⚠️ **Pre-alpha Status**: Agenty is under active development. Expect frequent breaking changes until we reach a stable release.

Agenty provides a clean, type-safe interface for creating:
- Conversational AI agents with structured inputs and outputs
- LLM pipelines
- Complex agent interactions with minimal boilerplate

## Key Features
- Intuitive Pythonic interfaces that feel natural to use
- Jinja2 templates for prompts and messages for dynamic context
- Conversation history management
- Structured Agent I/O for predictable behavior
- Flexible architecture supporting diverse use cases
- Built on pydantic-ai for robust type validation

Whether you're building a simple chatbot or a complex multi-agent system, Agenty helps you focus on logic rather than infrastructure.
The framework is currently only officially supported with the OpenAI API (through a proxy such as [LiteLLM](https://docs.litellm.ai/docs/simple_proxy)/[OpenRouter](https://openrouter.ai/docs/quick-start)) although theoretically it supports all the models supported by pydantic-ai.

> 🔍 Looking for a more mature alternative? Check out [atomic-agents](https://github.com/BrainBlend-AI/atomic-agents), which heavily inspired this project.

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
            "gpt-4",
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
from agenty import Agent

class Assistant(Agent):
    model = OpenAIModel("gpt-4", api_key="your-api-key")
    system_prompt = "You are a helpful and friendly AI assistant."

async def main():
    agent = Assistant()
    response = await agent.run("Hello, how are you?")
    print(response)

asyncio.run(main())
```

### Tool Usage
Agenty provides a framework for building custom agents that can leverage functions as tools through a simple decorator pattern.

1. **Define Your Agent:** Create a custom class that inherits from the base Agent class.

2. **Implement Tool Methods**: Add methods to your agent class that will serve as tools. Each method should include a docstring that describes the tool. You can even add parameter descriptions in the docstring and pydantic-ai implements [griffe](https://mkdocstrings.github.io/griffe/) to automatically generate tool parameter descriptions.

3. **Register Tools:** Use the `@tool` decorator to mark methods as tools. The decorator automatically registers these methods, making them available for your agent to use during execution. No additional configuration is needed.

Here's an example of a roulette game agent:
```python
from agenty import Agent, tool

class RouletteAgent(Agent):
    model = OpenAIModel("gpt-4", api_key="your-api-key")
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
        return random.randint(1, self.num_sides)

async def main():
    agent = RouletteAgent(player_name="John", num_sides=6)
    response = await agent.run("I guess the number will be 3!")
    print(response)

asyncio.run(main())

```
You can read more about [function tools](https://ai.pydantic.dev/tools/) by pydantic-ai. (underlying implementation of agenty tools)

## Key Features

- **Easy Agent Creation**: Create AI agents with just a few lines of code
- **Memory Management**: Built-in conversation memory tracking
- **Custom Tools**: Add custom capabilities to your agents using the `@tool` decorator
- **Usage Tracking**: Monitor and limit API usage
- **Type Safety**: Built with Pydantic for robust type checking
- **Flexible Models**: Support for various LLM providers through pydantic-ai
- **Async Support**: Built for asynchronous operations

## Configuration

Agents can be configured with various options:

```python
agent = Agent(
    model="gpt-4",                    # Model to use
    system_prompt="Your prompt here", # System prompt for the agent
    retries=3,                       # Number of retries for failed requests
    memory=AgentMemory(),           # Custom memory implementation
    usage_limits=AgentUsageLimits() # Usage limits configuration
)
```

## Requirements

- Python >= 3.12

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Jonathan Chun ([@jonchun](https://github.com/jonchun))
