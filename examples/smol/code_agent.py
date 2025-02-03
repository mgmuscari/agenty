import asyncio
import os
from pydantic_ai.models.openai import OpenAIModel
from smolagents import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
)
from agenty.integrations.smol import SmolCodeAgent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


async def main() -> None:
    code_agent: SmolCodeAgent[str, float] = SmolCodeAgent(
        model=OpenAIModel(
            "gpt-4o-mini",
            base_url=OPENAI_API_URL,
            api_key=OPENAI_API_KEY,
        ),
        smol_tools=[DuckDuckGoSearchTool(), PythonInterpreterTool()],
        smol_verbosity_level=1,
        input_schema=str,
        output_schema=float,
        # The recommended input_schema and output_schema types are str and str.
        # Changing the output schema will attempt to convert the output to the specified type AND automatically add a message to the user prompt requesting the output in the specified type.
        # The output schema type support for the SmolCodeAgent is EXPERIMENTAL and may not work as expected. Use with caution.
    )

    query = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"

    result = await code_agent.run(query)
    print(f"Response: {result}, Type: {type(result)}")


if __name__ == "__main__":
    asyncio.run(main())
