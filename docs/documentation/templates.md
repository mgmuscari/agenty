# Templates

## Overview

Templates provide a way to create dynamic prompts and messages by automatically populating variables in your agent's prompts. Using [Jinja templates](https://jinja.palletsprojects.com/en/stable/templates/), you can create flexible and context-aware interactions.

## Basic Usage

Any attribute of your agent that starts with a capital letter is automatically added to the template context:

```python
from agenty import Agent
from pydantic_ai.models.openai import OpenAIModel

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
    - Uses {{STYLE}} language
    """
    TONE: str = "friendly"
    LENGTH: str = "concise"
    STYLE: str = "casual"

    async def change_personality(self, tone: str, length: str, style: str):
        self.TONE = tone
        self.LENGTH = length
        self.STYLE = style
```

## Advanced Features

### Custom Template Context

By default, all capitalized attributes are exported as part of the template context.
You can override this behavior via the `template_context()` method:

```python
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

## Complete Example

Here's a comprehensive example demonstrating templates in action with a weather reporter agent:

```python
import asyncio
from datetime import datetime
from agenty import Agent
from pydantic_ai.models.openai import OpenAIModel

class WeatherReporter(Agent):
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    
    # Template variables
    LOCATION: str = "New York"
    TEMPERATURE_UNIT: str = "Celsius"
    DETAIL_LEVEL: str = "basic"
    TIME_OF_DAY: str = "morning"
    
    system_prompt = """
    You are a weather reporter for {{LOCATION}}.
    Report temperatures in {{TEMPERATURE_UNIT}}.
    Provide {{DETAIL_LEVEL}} weather information.
    
    Current time period: {{TIME_OF_DAY}}
    """
    
    def template_context(self):
        context = super().template_context()
        context["TIME_OF_DAY"] = self._get_time_of_day()
        return context
    
    def _get_time_of_day(self):
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    async def change_location(self, location: str):
        self.LOCATION = location
        return await self.run("What's the weather like?")

async def main():
    reporter = WeatherReporter()
    
    # Basic report
    print(await reporter.run("What's the weather like?"))
    
    # Change location and get new report
    reporter.LOCATION = "Tokyo"
    reporter.TEMPERATURE_UNIT = "Fahrenheit"
    reporter.DETAIL_LEVEL = "detailed"
    
    print(await reporter.run("What's the weather like?"))

if __name__ == "__main__":
    asyncio.run(main())
```

## Best Practices

1. **Clear Variable Names**

    - Prefer `ALL_CAPS` for template variables
    - Choose descriptive names that indicate purpose

2. **Maintain Readability**

    - Break long templates into multiple lines
    - Use comments to explain complex template logic
    - Keep template logic simple and maintainable

3. **Validate Variables**

    - Ensure all required variables are defined
    - Provide default values when appropriate
    - Handle missing variables gracefully

