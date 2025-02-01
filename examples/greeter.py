import asyncio

from agenty import Agent
from pydantic_ai.models.openai import OpenAIModel


class Greeter(Agent):
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = """You are a greeter. You speak in a {{TONE}} tone. Your response length should be {{ VERBOSITY }}. """
    TONE: str = "friendly"
    VERBOSITY: str = "verbose"


async def main():
    agent = Greeter()
    response = await agent.run("Hello, please greet me!")
    print(response)
    agent.TONE = "angry"
    agent.VERBOSITY = "concise"
    response = await agent.run("Hello, please greet me!")
    print(response)
    # Sample Output:
    # Good day! It's so good to see you. How may I assist you further?
    # What do you want?!


asyncio.run(main())
