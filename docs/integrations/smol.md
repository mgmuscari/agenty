# smolagents Integration

The [smolagents](https://github.com/huggingface/smolagents) integration allows you to use `CodeAgent` from smolagents as part of your agenty workflow. This integration provides a powerful way to create agents that can execute Python code, perform web searches, and handle complex computational tasks.

## SmolCodeAgent

The `SmolCodeAgent` is a wrapper around smolagents' `CodeAgent` that integrates seamlessly with agenty's pipeline system. It supports various input and output schema types and can be configured with different smol tools.

### Configuration

The `SmolCodeAgent` accepts the following parameters:

- `model`: An instance of a language model (e.g., OpenAIModel)
- `smol_tools`: A list of smol tools to make available to the agent (e.g., PythonInterpreterTool, DuckDuckGoSearchTool)
- `smol_verbosity_level`: Integer controlling the verbosity of smol output (0 for minimal, 1 for detailed)
- `input_schema`: The type for input validation (typically str)
- `output_schema`: The type for output validation and formatting (e.g., str, float, int)

### Example Usage

Here's a basic example of using `SmolCodeAgent` with OpenAI:

```python
import asyncio
from pydantic_ai.models.openai import OpenAIModel
from smolagents import DuckDuckGoSearchTool, PythonInterpreterTool
from agenty.integrations.smol import SmolCodeAgent

async def main() -> None:
    code_agent: SmolCodeAgent[str, float] = SmolCodeAgent(
        model=OpenAIModel(
            "gpt-4",
            api_key="your-api-key",
        ),
        smol_tools=[
            DuckDuckGoSearchTool(),
            PythonInterpreterTool()
        ],
        smol_verbosity_level=1,
        input_schema=str,
        output_schema=float,
    )

    query = ('How many seconds would it take for a leopard at full speed '
            'to run through Pont des Arts?')
    result = await code_agent.run(query)
    print(f"Response: {result}, Type: {type(result)}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Input/Output Schema Types

The `SmolCodeAgent` supports various input and output schema types:

- Input Schema: Typically `str` is recommended for maximum flexibility
- Output Schema: Supports `str`, `float`, and `int`
  - Using `str` is recommended for general use
  - Using `float` or `int` will automatically add type conversion and prompt the model to return the appropriate type
  - Output schema type support is EXPERIMENTAL and may not work as expected for non-string types

### Available Tools

The `SmolCodeAgent` can be configured with various tools from the smolagents library:

- `PythonInterpreterTool`: Enables Python code execution
- `DuckDuckGoSearchTool`: Enables web searches via DuckDuckGo
- Additional tools can be added from the smolagents library
