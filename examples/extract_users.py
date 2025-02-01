from typing import List
import asyncio
import logging
import os

from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console

from agenty import Agent
from agenty.types import BaseIO

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


logging.basicConfig()
logging.getLogger("agenty").setLevel(logging.DEBUG)


class User(BaseIO):
    name: str
    age: int
    hobbies: List[str]


# You can inherit from Agent with Generic type arguments to specify the input and output schemas for your custom Agent
class UserExtractor(Agent[str, List[User]]):
    input_schema = str  # This actually controls input type
    output_schema = List[User]  # This actually controls output type
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Extract all user information"


async def main() -> None:
    console = Console()
    agent = UserExtractor()
    story = """At the community center, Emma, 32, was painting a vibrant sunset while Jake, 27, captured it through his camera lens. Nearby, Sophia, 35, a runner and yoga enthusiast, was practicing stretches after her morning jog. Ben, 30, a fitness coach, was doing push-ups in the corner, taking a break from his workout. Each of them enjoyed their unique hobbies, creating a lively atmosphere filled with creativity, fitness, and relaxation. They shared stories about their passions, encouraging one another to pursue what they loved."""
    console.print(story, style="green")
    console.print("")
    console.print("---------------------------------")
    console.print("")
    # Static type-checkers correctly detect that input must be str and output must be List[User]
    resp = await agent.run(story)
    console.print(f"Type: {type(resp)}")
    console.print("")
    console.print("---------------------------------")
    console.print("")
    for user in resp:
        console.print(f"Name: {user.name}")
        console.print(f"Age: {user.age}")
        console.print(f"Hobbies: {', '.join(user.hobbies)}")
        console.print("")

    # EXAMPLE OUTPUT:
    # At the community center, Emma, 32, was painting a vibrant sunset while Jake, 27, captured it through his camera lens. Nearby, Sophia, 35, a runner and yoga enthusiast, was practicing stretches after her morning jog. Ben, 30, a fitness coach, was doing
    # push-ups in the corner, taking a break from his workout. Each of them enjoyed their unique hobbies, creating a lively atmosphere filled with creativity, fitness, and relaxation. They shared stories about their passions, encouraging one another to pursue
    # what they loved.

    # ---------------------------------

    # Type: <class 'list'>

    # ---------------------------------

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

    asyncio.run(main())
