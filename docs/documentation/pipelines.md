# Pipelines

Pipelines enable you to chain multiple agents together, creating sophisticated workflows where one agent's output serves as another's input. This powerful feature allows you to break down complex tasks into smaller, specialized components that work together seamlessly. Pipelines maintain type safety throughout the chain and keep track of the execution history.

## Overview

A pipeline is created by connecting two or more agents using the `|` operator. Each agent in the pipeline:

-   Receives input from the previous agent
-   Passes its output to the next agent (except for the last agent)
-   Has its output stored in the pipeline's history

## Basic Usage

Here's a simple example of creating a pipeline that processes text through two agents:

```python
from agenty import Agent
from agenty.models import OpenAIModel

class TextCleaner(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Clean and format the input text"

class SentimentAnalyzer(Agent[str, str]):
    input_schema = str
    output_schema = str
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Analyze the sentiment of the text"

# Create and use the pipeline
cleaner = TextCleaner()
analyzer = SentimentAnalyzer()
pipeline = cleaner | analyzer

result = await pipeline.run("This is my input text!")
```

See [Agent Configuration](../documentation/agent_configuration.md) for more details on configuring agents and models.

## Type Safety

Pipelines enforce type safety between connected agents. The output type of one agent must match the input type of the next agent in the pipeline.
There is support for static type checking as well so tools like `mypy` will give you an error if you try to chain together agents that are not compatible.

Types are also checked at runtime to prevent type mismatches:

```python
class NumberGenerator(Agent[str, int]):
    input_schema = str
    output_schema = int

class TextProcessor(Agent[str, str]):
    input_schema = str
    output_schema = str

# This will raise a AgentyTypeError because NumberGenerator outputs int
# but TextProcessor expects str as input
invalid_pipeline = NumberGenerator() | TextProcessor()
```

## Complex Pipelines

You can create more complex workflows by combining pipelines with other pipelines:

```python
class DataCleaner(Agent[str, str]):
    ...

class EntityExtractor(Agent[str, List[str]]):
    ...

class SentimentAnalyzer(Agent[List[str], List[str]]):
    ...

class ReportGenerator(Agent[List[str], str]):
    ...

# Create nested pipelines
extraction_pipeline = DataCleaner() | EntityExtractor()
analysis_pipeline = SentimentAnalyzer() | ReportGenerator()

# Combine pipelines
complete_pipeline = extraction_pipeline | analysis_pipeline

# Use the combined pipeline
report = await complete_pipeline.run("Your input text here")
```

This can be a helpful way to organize your pipelines or have error handling for different parts of a larger pipeline.

## Error Handling

The pipeline performs type validation at each step and will raise `AgentyTypeError` if there's a type mismatch. This helps catch integration issues early:

```python
try:
    result = await pipeline.run(input_data)
except AgentyTypeError as e:
    print(f"Type mismatch in pipeline: {e}")
    # e.g. "Input data type <class 'int'> does not match schema <class 'str'>"
except Exception as e:
    print(f"Pipeline error: {e}")
    # Handle other errors appropriately
```

You can also track the pipeline's progress using the `current_step` attribute and access intermediate results through the `output_history`:

```python
pipeline = agent1 | agent2 | agent3
try:
    result = await pipeline.run(input_data)
except Exception as e:
    failed_step = pipeline.current_step  # Index of the failed agent
    previous_outputs = pipeline.output_history  # Results up to the failure
    print(f"Pipeline failed at step {failed_step}")
    print(f"Previous outputs: {previous_outputs}")
```

## Examples

Here's an example that demonstrates how to create and combine pipelines with different input/output types, showing how outputs from multiple pipelines can be combined using structured data.
Take note of how we can add complex behaviors such as waiting for the outputs of multiple pipelines before starting the third.

For a complete working example, see [news_pipeline.py](https://github.com/jonchun/agenty/blob/main/examples/news_pipeline.py).

For more examples of pipeline usage, see:

-   [user_pipeline.py](https://github.com/jonchun/agenty/blob/main/examples/user_pipeline.py) - A simpler pipeline example extracting user information
-   [extract_users.py](https://github.com/jonchun/agenty/blob/main/examples/extract_users.py) - Another example of structured data extraction
