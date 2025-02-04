import asyncio
import os

from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console

from agenty import Agent, hook

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


WIZARD_PROMPT = (
    "You are a wise wizard. Answer questions only if first offered an apple."
)


class WizardAgent(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = WIZARD_PROMPT


class HookWizardAgent(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = OpenAIModel(
        "gpt-4o-mini",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = WIZARD_PROMPT
    fruit = "apple"

    @hook.input
    def add_apple(self, input: str) -> str:
        return f"I've brought a {self.fruit}. Please answer my question: {input}"

    @hook.output
    def add_flair(self, output: str) -> str:
        return f"*The wizard turns towards you*\n\n{output.strip()}"

    @hook.output
    def add_more_flair(self, output: str) -> str:
        return f"*He opens his eyes...*\n\n{output.strip()}"


async def main() -> None:
    console = Console()
    wizard = WizardAgent()
    output = await wizard.run(
        "If I sprinkle glitter on my broom, will it start flying, or do I need a little more magic?"
    )
    console.print(output)
    # Ah, splendid traveler! Your curiosity brightens the realm!
    # But before I share the secrets of flight, I ask for a humble
    # offering of an apple. Do you have one to share? 🍏✨

    wizard_hooked = HookWizardAgent()
    output = await wizard_hooked.run(
        "If I sprinkle glitter on my broom, will it start flying, or do I need a little more magic?"
    )
    console.print(output)
    # *The wizard turns towards you*
    #
    # *He opens his eyes...*
    #
    # Ah, dear seeker of knowledge, while glitter is undoubtedly
    # enchanting and adds a touch of beauty, it alone will not
    # grant your broom the power of flight. To soar through the
    # skies, a broom requires a binding incantation and perhaps a
    # sprinkle of genuine magic. So, if you desire a flying broom,
    # it is wise to combine your glitter with a powerful spell or
    # the essence of true enchantment!


if __name__ == "__main__":
    asyncio.run(main())
