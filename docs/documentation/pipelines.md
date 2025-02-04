You can create pipelines, enabling multiple agents to process data sequentially. This is useful for workflows where one agent's output serves as another's input. You can chain multiple agents and even combine pipelines within a single pipeline.

Here's an example of how to create and use a pipeline:
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

class UserExtractor(Agent[str, List[User]]):
    input_schema = str
    output_schema = List[User]
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Extract all user information"

class UserTitleAgent(Agent[List[User], List[str]]):
    input_schema = List[User]
    output_schema = List[str]
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Generate an appropriate title for each user. e.g. 'Debbie, the receptionist'"

async def main():
    user_extractor = UserExtractor()
    title_agent = UserTitleAgent()
    story = """At the community center, Emma, 32, was painting a vibrant sunset while Jake, 27, captured it through his camera lens. Nearby, Sophia, 35, a runner and yoga enthusiast, was practicing stretches after her morning jog. Ben, 30, a fitness coach, was doing push-ups in the corner, taking a break from his workout. Each of them enjoyed their unique hobbies, creating a lively atmosphere filled with creativity, fitness, and relaxation. They shared stories about their passions, encouraging one another to pursue what they loved."""

    pipeline = user_extractor | title_agent
    res = await pipeline.run(story)
    # res = await (user_extractor | title_agent).run(story) # you can skip the variable assignment if you want
    print(res)
    # Output: ['Emma, the painter', 'Jake, the photographer', 'Sophia, the fitness enthusiast', 'Ben, the fitness coach']

asyncio.run(main())
```

In this example, the `UserExtractor` agent extracts user information from a text input, and the `UserTitleAgent` generates titles for each user. The `|` operator is used to chain these agents into a pipeline.
