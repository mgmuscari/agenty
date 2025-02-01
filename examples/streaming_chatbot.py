from types import FrameType
from typing import Optional
import asyncio
import atexit
import os
import readline
import signal

from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from rich.live import Live

from agenty import Agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


# import logging
# logging.basicConfig()
# logging.getLogger("agenty").setLevel(logging.DEBUG)
class ChatAgent(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = (
        "You are a helpful and friendly AI assistant named {{ AGENT_NAME }}."
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.AGENT_NAME: str = "Agenty"


async def main() -> None:
    console = Console()
    agent = ChatAgent()
    console.print("Basic Chatbot | Type /exit or /quit to exit")
    user_prompt = "\033[1;36mUser: \033[0m"  # Use raw ANSI code here because console.input() doesn't work correctly with chat history
    while True:
        user_input = await async_input(user_prompt)
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("[yellow]Exiting chat...[/yellow]")
            break

        with Live("", console=console, vertical_overflow="visible") as live:
            console.print("[bold blue]Assistant:[/bold blue] ")
            async for message in agent.run_stream(user_input):
                live.update(str(message))


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
