import pytest

from agenty.components.usage import AgentUsage
from pydantic_ai.usage import Usage


from openai.types import CompletionUsage


def test_agent_usage_set(
    agent_usage: AgentUsage,
    sample1_usage: Usage,
):
    agent_usage["model_a"] = sample1_usage
    assert agent_usage["model_a"] == sample1_usage


def test_agent_usage_add(
    agent_usage: AgentUsage,
    sample1_usage: Usage,
    sample2_usage: Usage,
    sample3_usage: Usage,
):
    # Add usage for a single model
    agent_usage["model_b"] += sample1_usage
    assert agent_usage["model_b"] == sample1_usage

    # Add usage multiple times and check sum
    agent_usage["model_c"] += sample2_usage
    agent_usage["model_c"] += sample3_usage
    assert agent_usage["model_c"] == sample2_usage + sample3_usage


def test_agent_usage_properties(
    agent_usage: AgentUsage,
    sample1_usage: Usage,
    sample2_usage: Usage,
    sample3_usage: Usage,
):
    # Check aggregation across all models
    agent_usage["model_a"] += sample1_usage
    agent_usage["model_b"] += sample2_usage
    agent_usage["model_c"] += sample3_usage

    sum_usage = sample1_usage + sample2_usage + sample3_usage
    assert agent_usage.requests == sum_usage.requests
    assert agent_usage.request_tokens == sum_usage.request_tokens
    assert agent_usage.response_tokens == sum_usage.response_tokens
    assert agent_usage.total_tokens == sum_usage.total_tokens

    # Check iteration
    assert list(agent_usage.keys()) == ["model_a", "model_b", "model_c"]


@pytest.fixture
def empty_usage():
    return Usage()


@pytest.fixture
def sample1_usage():
    return Usage(
        requests=1,
        request_tokens=10,
        response_tokens=5,
        total_tokens=15,
    )


@pytest.fixture
def sample2_usage():
    return Usage(
        requests=3,
        request_tokens=5,
        response_tokens=100,
        total_tokens=105,
    )


@pytest.fixture
def sample3_usage():
    return Usage(
        requests=4,
        request_tokens=15,
        response_tokens=105,
        total_tokens=120,
    )


@pytest.fixture
def sample1_openai_usage():
    return CompletionUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        completion_tokens_details=None,
        prompt_tokens_details=None,
    )


@pytest.fixture
def agent_usage():
    return AgentUsage()
