Functions can be used as tools through a simple decorator pattern.

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
    system_prompt = ('You\'re a dice game, you should roll the die '
                    'and see if the number matches the user\'s guess.')

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
