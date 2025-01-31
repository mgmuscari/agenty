from types import FrameType
from typing import Optional
import asyncio
import atexit
import os
import random
import readline
import signal

from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console

from agenty import Agent, tool

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")

import logging

logging.basicConfig()
logging.getLogger("agenty").setLevel(logging.DEBUG)


class RouletteAgent(Agent):
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = (
        "You're a dice game, you should roll the die and see if the number "
        "you get back matches the user's guess. If so, tell them they're a winner. "
        "Use the player's name in the response."
    )

    def __init__(
        self,
        player_name: str,
        num_sides: int = 10,
        **kwargs,
    ) -> None:
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


async def main() -> None:
    console = Console()
    agent = RouletteAgent(player_name="John", num_sides=6)
    console.print("Roulette Game | Type /exit or /quit to exit")
    user_prompt = "\033[1;36mUser: \033[0m"  # Use raw ANSI code here because console.input() doesn't work correctly with chat history
    while True:
        user_input = await async_input(user_prompt)
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("[yellow]Exiting chat...[/yellow]")
            break
        resp = await agent.run(user_input)
        console.print(f"[bold blue]Assistant:[/bold blue] {resp}")


####################################################################################################
# Can ignore most of the following code it's just boilierplate to make the example chatbot work


async def async_input(prompt: str) -> str:
    return await asyncio.to_thread(input, prompt)


history_file = ".chatbot_history"
atexit.register(readline.write_history_file, history_file)
try:
    readline.read_history_file(history_file)
except FileNotFoundError:
    pass


def handle_exit_signal(sig: int, frame: Optional[FrameType]) -> None:
    pass


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_exit_signal)  # type: ignore
    asyncio.run(main())
