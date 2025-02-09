# Hooks

Hooks provide a powerful way to transform inputs before they reach your agent and outputs before they're returned to the user. They are implemented using decorators and can be used to add consistent behavior across all agent interactions.

## Overview

Hooks come in two varieties:

-   `@hook.input`: Transforms input before it reaches your agent
-   `@hook.output`: Transforms output before it's returned to the user

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

For an input of `test`, the processing flow is:

1. Input processing:

    - `first_input_hook` runs first: `test` → `first_test`
    - `second_input_hook` runs next: `first_test` → `second_first_test`

2. Output processing:
    - `first_output_hook` runs first: `result` → `result_first`
    - `second_output_hook` runs next: `result_first` → `result_first_second`

## Instance Attributes

Hooks can access instance attributes of your agent class, allowing for configurable behavior:

```python
class AgentWithState(Agent[str, str]):
    prefix = "default"

    @hook.input
    def add_custom_prefix(self, input: str) -> str:
        return f"{self.prefix}_{input}"
```

For an example of using instance attributes in hooks, see the [Wizard Agent example](https://github.com/jonchun/agenty/blob/main/examples/wizard_agent.py) which uses a configurable `fruit` attribute.

## Type Safety

Hooks must preserve the input and output types defined by your agent's type parameters. For example, if your agent is defined as `Agent[str, float]`, all input hooks must accept and return strings, while output hooks must accept and return floats:

```python
class AgentWithHooks(Agent[str, float]):
    @hook.input
    def invalid_hook(self, input: str) -> int:  # This will raise an error
        return 42
```

Hooks also support structured data types, making it easy to process complex data structures:

```python
class AgentWithHooks(Agent[List[str], str]):
    @hook.input
    def list_hook(self, input: List[str]) -> List[str]:
        new_list = []
        for item in input:
            # do stuff here
            new_list.append(f"hook_{item}")
        return new_list
```

## Wizard Agent Example

For a complete example that demonstrates hooks in action, check out our [Wizard Agent](https://github.com/jonchun/agenty/blob/main/examples/wizard_agent.py). This example shows how to:

-   Enforce prerequisites (requiring an offering)
-   Add consistent formatting (dramatic flair)
-   Transform both inputs and outputs
-   Chain multiple transformations together
-   Use instance attributes for configuration

Here's a key excerpt from the example:

```python

WIZARD_PROMPT = (
    "You are a wise wizard. Answer questions only if first offered an apple."
)

class HookWizardAgent(Agent[str, str]):

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
```

## Best Practices

1. Keep hooks focused and single-purpose
2. Use descriptive names that indicate the hook's transformation
3. Maintain type consistency between input and output
4. Consider hook order when using multiple transformations
5. Use instance attributes for configurable behavior
6. Test hooks thoroughly to ensure they maintain data integrity
7. Document hook behavior, especially when chaining multiple hooks
