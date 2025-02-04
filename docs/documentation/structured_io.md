Agenty supports structured input and output types of type `AgentIO`.
```python
AgentIO = Union[bool, int, float, str, BaseIO, Sequence[AgentIO]]
```

Data validation for structured objects can be done by inheriting from `agenty.types.BaseIO` (which are just Pydantic Models).



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
