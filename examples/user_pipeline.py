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


class UserExtractor(Agent[str, List[User]]):
    input_schema = str
    output_schema = List[User]
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Extract all user information"


class UserTitleAgent(Agent[List[User], List[str]]):
    input_schema = List[User]
    output_schema = List[str]
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = (
        "Generate an appropriate title for each user. e.g. 'Debbie, the receptionist'"
    )


async def main() -> None:
    console = Console()
    user_extractor = UserExtractor()
    title_agent: Agent[List[User], List[str]] = UserTitleAgent()
    story = """At the community center, Emma, 32, was painting a vibrant sunset while Jake, 27, captured it through his camera lens. Nearby, Sophia, 35, a runner and yoga enthusiast, was practicing stretches after her morning jog. Ben, 30, a fitness coach, was doing push-ups in the corner, taking a break from his workout. Each of them enjoyed their unique hobbies, creating a lively atmosphere filled with creativity, fitness, and relaxation. They shared stories about their passions, encouraging one another to pursue what they loved."""

    pipeline = user_extractor | title_agent
    res = await pipeline.run(story)
    console.print(res)
    # ['Emma, the painter', 'Jake, the photographer', 'Sophia, the fitness enthusiast', 'Ben, the fitness coach']


if __name__ == "__main__":
    asyncio.run(main())
