import asyncio
import os
from typing import List
from pydantic_ai.models.openai import OpenAIModel
from rich.console import Console
from agenty import Agent
from agenty.types import BaseIO

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://127.0.0.1:4000")


class Article(BaseIO):
    title: str
    content: str
    source: str


class NewsAnalysis(BaseIO):
    summaries: List[str]
    main_article_analysis: str


# Outputs a list of Article objects
class ArticleExtractor(Agent[str, List[Article]]):
    input_schema = str
    output_schema = List[Article]
    model = OpenAIModel(
        "llama3.3-70b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Extract articles from the news feed."


# Pipeline 1: Outputs List[str] (summaries of each article)
class SummaryGenerator(Agent[List[Article], List[str]]):
    input_schema = List[Article]
    output_schema = List[str]
    model = OpenAIModel(
        "llama3.3-70b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Generate concise summaries of each article. 25 words or fewer."


# Pipeline 2: Outputs str (detailed analysis of main article)
class MainArticleSelector(Agent[List[Article], Article]):
    input_schema = List[Article]
    output_schema = Article
    model = OpenAIModel(
        "llama3.3-70b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Select the most important article"


class DetailedAnalyzer(Agent[Article, str]):
    input_schema = Article
    output_schema = str
    model = OpenAIModel(
        "llama3.3-70b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Provide analysis of the article. 25 words or fewer."


# Pipeline 3: Takes NewsAnalysis combining outputs from previous pipelines
class InsightGenerator(Agent[NewsAnalysis, List[str]]):
    input_schema = NewsAnalysis
    output_schema = List[str]
    model = OpenAIModel(
        "llama3.3-70b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = (
        "Generate insights from summaries and analysis. 25 words or fewer per insight."
    )


class ReportGenerator(Agent[List[str], str]):
    input_schema = List[str]
    output_schema = str
    model = OpenAIModel(
        "llama3.3-70b-groq",
        base_url=OPENAI_API_URL,
        api_key=OPENAI_API_KEY,
    )
    system_prompt = "Generate comprehensive report from insights. 25 words or fewer."


async def main() -> None:
    console = Console()
    news_feed = """
    Breaking: Tech giant launches new AI model
    In a surprising move, the company revealed their latest...
    
    Market Update: Stocks reach record high
    Global markets continued their upward trend as...
    
    Science News: Breakthrough in quantum computing
    Scientists have achieved a new milestone...
    """

    articles = await ArticleExtractor().run(news_feed)

    # Pipeline 1: Generate summaries (List[str])
    summary_pipeline = SummaryGenerator()  # just 1 agent is still a pipeline

    # Pipeline 2: Get detailed analysis of main article (str)
    analysis_pipeline = MainArticleSelector() | DetailedAnalyzer()

    # Wait for Pipelines 1 and 2 to complete in parallel
    summaries, main_analysis = await asyncio.gather(
        summary_pipeline.run(articles), analysis_pipeline.run(articles)
    )
    console.print("Article Summaries:", summaries)
    # Article Summaries:
    # ['Tech giant launches AI model', 'Stocks reach record high', 'Breakthrough in quantum computing']

    console.print("Main Article Analysis:", main_analysis)
    # Main Article Analysis: Company announces new AI model, details pending.

    # Pipeline 3: Combine outputs using NewsAnalysis model
    combined_analysis = NewsAnalysis(
        summaries=summaries, main_article_analysis=main_analysis
    )
    insight_pipeline = InsightGenerator() | ReportGenerator()
    final_report = await insight_pipeline.run(combined_analysis)
    console.print("Final Report:", final_report)
    # Final Report: Breakthrough AI launch sparks stock surge.


if __name__ == "__main__":
    asyncio.run(main())
