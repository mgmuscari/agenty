# Pipelines

Pipelines enable you to chain multiple agents together, creating sophisticated workflows where one agent's output serves as another's input. This powerful feature allows you to break down complex tasks into smaller, specialized components that work together seamlessly. Pipelines maintain type safety throughout the chain and keep track of the execution history.

## Overview

A pipeline is created by connecting two or more agents using the `|` operator. Each agent in the pipeline:

- Receives input from the previous agent
- Passes its output to the next agent (except for the last agent)
- Has its output stored in the pipeline's history

## Basic Usage

Here's a simple example of creating a pipeline that processes text through two agents:

```python
from agenty import Agent
from pydantic_ai.models.openai import OpenAIModel

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
This can be a helpful wait to organize your pipelines or have error handling for different parts of a larger pipeline.

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

## Complex Example

Here's an example that demonstrates how to create and combine pipelines with different input/output types, showing how outputs from multiple pipelines can be combined using structured data.
Take note of how we can add complex behaviors such as waiting for the outputs of multiple pipelines before starting the third.

```python
import asyncio
from typing import List
from pydantic_ai.models.openai import OpenAIModel
from agenty import Agent
from agenty.types import BaseIO

class Article(BaseIO):
    title: str
    content: str
    source: str

class NewsAnalysis(BaseIO):
    summaries: List[str]
    main_article_analysis: str

class ArticleExtractor(Agent[str, List[Article]]):
    input_schema = str
    output_schema = List[Article]
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Extract articles from the news feed."

# Pipeline 1: Outputs List[str] of article summaries
class SummaryGenerator(Agent[List[Article], List[str]]):
    input_schema = List[Article]
    output_schema = List[str]
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Generate concise summaries of each article. 25 words or fewer."

# Pipeline 2: Outputs str (detailed analysis of main article)
class MainArticleSelector(Agent[List[Article], Article]):
    input_schema = List[Article]
    output_schema = Article
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Select the most important article"

class DetailedAnalyzer(Agent[Article, str]):
    input_schema = Article
    output_schema = str
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Provide analysis of the article. 25 words or fewer."

# Pipeline 3: Takes NewsAnalysis combining outputs from previous pipelines
class InsightGenerator(Agent[NewsAnalysis, List[str]]):
    input_schema = NewsAnalysis
    output_schema = List[str]
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = (
        "Generate insights from summaries and analysis. 25 words or fewer per insight."
    )

class ReportGenerator(Agent[List[str], str]):
    input_schema = List[str]
    output_schema = str
    model = OpenAIModel("gpt-4o-mini", api_key="your-api-key")
    system_prompt = "Generate comprehensive report from insights. 25 words or fewer."

async def main():
    news_feed = '''
    Breaking: Tech giant launches new AI model
    In a surprising move, the company revealed their latest...
    
    Market Update: Stocks reach record high
    Global markets continued their upward trend as...
    
    Science News: Breakthrough in quantum computing
    Scientists have achieved a new milestone...
    '''
    
    # Extract articles first
    articles = await ArticleExtractor().run(news_feed)

    # Pipeline 1: Generate summaries (List[str])
    summary_pipeline = SummaryGenerator()  # just 1 agent is still a pipeline

    # Pipeline 2: Get detailed analysis of main article (str)
    analysis_pipeline = MainArticleSelector() | DetailedAnalyzer()

    # Wait for Pipelines 1 and 2 to complete in parallel
    summaries, main_analysis = await asyncio.gather(
        summary_pipeline.run(articles), analysis_pipeline.run(articles)
    )
    print("Article Summaries:", summaries)
    # Output: ["Tech giant unveils new AI model...", "Markets hit new high...", "Quantum computing breakthrough..."]

    print("Main Article Analysis:", main_analysis)
    # Output: "The tech company's new AI model represents a significant advancement..."

    # Pipeline 3: Combine outputs using NewsAnalysis model
    combined_analysis = NewsAnalysis(
        summaries=summaries, main_article_analysis=main_analysis
    )
    insight_pipeline = InsightGenerator() | ReportGenerator()
    final_report = await insight_pipeline.run(combined_analysis)
    print("Final Report:", final_report)
    # Output: "Today's Tech & Market Analysis: The technology sector is showing significant momentum..."

asyncio.run(main())
```