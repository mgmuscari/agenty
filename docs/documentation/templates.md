# Templates

## Overview

Templates provide a way to create dynamic prompts and messages by automatically populating variables in your agent's prompts. Using [Jinja templates](https://jinja.palletsprojects.com/en/stable/templates/), you can create flexible and context-aware interactions.

## Basic Usage

Any attribute of your agent that starts with a capital letter is automatically added to the template context:

```python
from agenty import Agent
from agenty.models import OpenAIModel

class SimpleGreeter(Agent):
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "You are a greeter who speaks in a {{TONE}} tone."
    TONE: str = "friendly"

    async def greet(self):
        return await self.run("Hello!")
```

When this agent runs:

1. The template engine detects `{{TONE}}` in the system prompt
2. It replaces it with the value of the `TONE` attribute ("friendly")
3. The final prompt becomes: "You are a greeter who speaks in a friendly tone."

## Dynamic Context

You can modify template variables at runtime to change agent behavior:

```python
class DynamicGreeter(Agent):
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = """
    You are a greeter who:
    - Speaks in a {{TONE}} tone
    - Gives {{LENGTH}} responses
    """
    TONE: str = "friendly"
    LENGTH: str = "medium"

    async def change_personality(self, tone: str, length: str):
        self.TONE = tone
        self.LENGTH = length
```

For a complete working example, see [examples/greeter.py](https://github.com/jonchun/agenty/blob/main/examples/greeter.py).

## Advanced Features

### Custom Template Context

By default, all capitalized attributes are exported as part of the template context.
You can override this behavior via the `template_context()` method:

```python
from datetime import datetime

class CustomContextGreeter(Agent):
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Current time: {{current_time}}"

    def template_context(self):
        context = super().template_context()
        context["current_time"] = datetime.now().strftime("%H:%M:%S")
        return context
```

### Template Inheritance

You can create base templates and extend them:

```python
class BaseAgent(Agent):
    COMMON_RULES = """
    1. Be concise
    2. Be accurate
    3. Be helpful
    """
    system_prompt = """
    Basic Instructions:
    {{COMMON_RULES}}
    """

class SpecializedAgent(BaseAgent):
    system_prompt = """
    {{COMMON_RULES}}
    Additional Instructions:
    4. Use technical language
    5. Provide examples
    """
```

## Best Practices

1. **Clear Variable Names**

    - Use `ALL_CAPS` for template variables
    - Choose descriptive names that indicate purpose
    - Example: `TONE`, `RESPONSE_LENGTH`, `DETAIL_LEVEL`

2. **Maintain Readability**

    - Break long templates into multiple lines using triple quotes
    - Use comments to explain complex template logic
    - Keep template logic simple and maintainable

3. **Variable Management**

    - Define all template variables as class attributes
    - Provide sensible default values
    - Update variables through methods or direct attribute access
    - Consider type hints for better code maintainability

4. **Template Structure**
    - Keep templates focused and single-purpose
    - Use consistent formatting and indentation
    - Consider breaking very long templates into smaller, reusable pieces

## Implementation Details

The template system uses Jinja2's SandboxedEnvironment for secure template rendering. Variables are automatically dedented to normalize whitespace. For implementation details, see [agenty/template.py](https://github.com/jonchun/agenty/blob/main/agenty/template.py).
