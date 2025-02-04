# Hooks

Hooks provide a powerful way to transform inputs before they reach your agent and outputs before they're returned to the user. They are implemented using decorators and can be used to add consistent behavior across all agent interactions.

## Overview

Hooks come in two varieties:

- `@hook.input`: Transforms input before it reaches your agent
- `@hook.output`: Transforms output before it's returned to the user

Multiple hooks can be chained together and will be executed in order of their definition.

## Basic Usage

Here's a simple example of using hooks to add prefixes and suffixes:

```python
from agenty import Agent, hook

class MyAgent(Agent[str, str]):
    input_schema = str
    output_schema = str
    
    @hook.input
    def add_prefix(self, input: str) -> str:
        return f"prefix_{input}"
        
    @hook.output 
    def add_suffix(self, output: str) -> str:
        return f"{output}_suffix"
```

When this agent runs:

1. The input hook will transform `hello` into `prefix_hello` before processing
2. The output hook will transform the agent's response by adding `_suffix`

## Multiple Hooks

You can define multiple input and output hooks. They are executed in order of definition:

```python
class AgentWithMultipleHooks(Agent[str, str]):
    @hook.input
    def first_input_hook(self, input: str) -> str:
        return f"first_{input}"
        
    @hook.input
    def second_input_hook(self, input: str) -> str:
        return f"second_{input}"
        
    @hook.output
    def first_output_hook(self, output: str) -> str:
        return f"{output}_first"
        
    @hook.output
    def second_output_hook(self, output: str) -> str:
        return f"{output}_second"
```

For an input of `test`:

**Input processing:**

   - `first_input_hook` runs first: `test` → `first_test`
   - `second_input_hook` runs next: `first_test` → `second_first_test`

**Output processing:**

   - `first_output_hook` runs first: `result` → `result_first`
   - `second_output_hook` runs next: `result_first` → `result_first_second`

## Instance Attributes

Hooks can access instance attributes of your agent class:

```python
class AgentWithState(Agent[str, str]):
    prefix = "default"
    
    @hook.input
    def add_custom_prefix(self, input: str) -> str:
        return f"{self.prefix}_{input}"
```

## Type Safety

Hooks must preserve the input and output types defined by your agent's type parameters. For example, if your agent is defined as `Agent[str, str]`, all hooks must accept and return strings:

```python
class AgentWithHooks(Agent[str, str]):
    @hook.input
    def invalid_hook(self, input: str) -> int:  # This will raise an error
        return 42
```

## Wizard Agent Example

Here's a complete example that demonstrates hooks in action with a wizard agent that requires an offering before answering questions:

**This example shows how hooks can be used to:**

   - Enforce prerequisites (offering an apple)
   - Add consistent formatting (dramatic flair)
   - Transform both inputs and outputs
   - Chain multiple transformations together

```python
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
    
    # Without hooks - wizard refuses to answer without an apple
    wizard = WizardAgent()
    output = await wizard.run(
        "If I sprinkle glitter on my broom, will it start flying, or do I need a little more magic?"
    )
    console.print(output)
    # Output:
    # Ah, splendid traveler! Your curiosity brightens the realm!
    # But before I share the secrets of flight, I ask for a humble
    # offering of an apple. Do you have one to share? 🍏✨

    # With hooks - automatically offers an apple and adds dramatic flair
    wizard_hooked = HookWizardAgent()
    output = await wizard_hooked.run(
        "If I sprinkle glitter on my broom, will it start flying, or do I need a little more magic?"
    )
    console.print(output)
    # Output:
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
```

## Best Practices

1. Keep hooks focused and single-purpose
2. Use descriptive names that indicate the hook's transformation
3. Maintain type consistency between input and output
4. Consider hook order when using multiple transformations
5. Use instance attributes for configurable behavior