You can use [Jinja templates](https://jinja.palletsprojects.com/en/stable/templates/) to create dynamic prompts and messages by automatically populating template variables.


!!! tip
    Any attribute of your agent object that starts with a capital letter is automatically added to the template context. Agenty prefers to use template variables in `ALL_CAPS` formatting for better visibility. 

    If you need to modify this default behavior, you can override the `template_context()` method in your custom agent class.

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
