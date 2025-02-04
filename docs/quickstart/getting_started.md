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
