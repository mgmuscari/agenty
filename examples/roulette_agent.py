from types import FrameType
from typing import Optional, Tuple
import asyncio
import atexit
import os
import random
import readline
import signal

from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console

from agenty import Agent, tool
from agenty.types import BaseIO

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")

import logging

logging.basicConfig()
logging.getLogger("agenty").setLevel(logging.DEBUG)


class PlayerInfo(BaseIO):
    name: str
    balance: float


class RouletteAgent(Agent):
    input_schema = str
    output_schema = str
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = (
        "You're running a roulette game. Help users place bets and "
        "determine if they've won based on the wheel spin. Use the "
        "player's name in responses to make it more personal."
    )

    def __init__(
        self,
        player_name: str,
        starting_balance: float = 100.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.player_name = player_name
        self.balance = starting_balance
        self.current_bet = 0.0
        self.bet_number = None

    @tool
    def get_player_info(self) -> PlayerInfo:
        """Get the current player's information including name and balance."""
        return PlayerInfo(name=self.player_name, balance=self.balance)

    @tool
    def place_bet(self, amount: float, number: int) -> bool:
        """
        Place a bet on a specific number.

        Args:
            amount: How much money to bet
            number: Which number to bet on (0-36)
        """
        if amount > self.balance:
            return False
        if not 0 <= number <= 36:
            return False

        self.current_bet = amount
        self.bet_number = number
        self.balance -= amount
        return True

    @tool
    def spin_wheel(self) -> Tuple[int, bool]:
        """
        Spin the roulette wheel and determine if the player won.

        Returns:
            A tuple of (winning number, whether player won)
        """
        winning_number = random.randint(0, 36)
        won = False

        if self.current_bet > 0 and self.bet_number is not None:
            if winning_number == self.bet_number:
                self.balance += self.current_bet * 35
                won = True
            self.current_bet = 0
            self.bet_number = None

        return winning_number, won


async def main() -> None:
    console = Console()
    starting_balance = 100.00
    agent = RouletteAgent(
        player_name="John",
        starting_balance=starting_balance,
    )

    console.print("[bold green]Welcome to the Roulette Game![/bold green]")
    console.print("Type /exit or /quit to exit")
    console.print(f"Starting balance: ${starting_balance}")
    user_prompt = "\033[1;36mUser: \033[0m"  # Use raw ANSI code here because console.input() doesn't work correctly with chat history

    while True:
        user_input = await async_input(user_prompt)
        if user_input.lower() in ["/exit", "/quit"]:
            console.print(
                "[yellow]Thanks for playing! Final balance: ${:.2f}[/yellow]".format(
                    agent.balance
                )
            )
            break
        resp = await agent.run(user_input)
        console.print(f"[bold blue]Dealer:[/bold blue] {resp}")


####################################################################################################
# Can ignore most of the following code it's just boilerplate to make the example chatbot work


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
