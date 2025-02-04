"""Tests for usage tracking components."""

import pytest

from agenty.components.usage import AgentUsage, AgentUsageLimits
from pydantic_ai.usage import Usage, UsageLimits


class TestAgentUsage:
    """Tests for AgentUsage class."""

    def test_dictionary_operations(
        self, agent_usage: AgentUsage, sample1_usage: Usage
    ) -> None:
        """Test basic dictionary-like operations."""
        # Test __setitem__ and __getitem__
        agent_usage["model_a"] = sample1_usage
        assert agent_usage["model_a"] == sample1_usage

        # Test auto-creation of new models
        new_usage = agent_usage["new_model"]
        assert isinstance(new_usage, Usage)
        assert new_usage.requests == 0

        # Test __delitem__
        assert "model_a" in agent_usage
        from devtools import debug

        del agent_usage["model_a"]
        assert "model_a" not in agent_usage

        # Test __len__ and iteration
        agent_usage["model_b"] = sample1_usage
        agent_usage["model_c"] = sample1_usage
        assert len(agent_usage) == 3  # new_model, model_b, model_c
        assert set(agent_usage.keys()) == {"new_model", "model_b", "model_c"}

        # Test KeyError on delete
        with pytest.raises(KeyError):
            del agent_usage["nonexistent"]

    def test_usage_addition(
        self, agent_usage: AgentUsage, sample1_usage: Usage, sample2_usage: Usage
    ) -> None:
        """Test usage addition operations."""
        # Test adding to new model
        agent_usage["model_a"] += sample1_usage
        assert agent_usage["model_a"] == sample1_usage

        # Test adding multiple times
        agent_usage["model_a"] += sample2_usage
        expected = Usage(
            requests=(sample1_usage.requests or 0) + (sample2_usage.requests or 0),
            request_tokens=(sample1_usage.request_tokens or 0)
            + (sample2_usage.request_tokens or 0),
            response_tokens=(sample1_usage.response_tokens or 0)
            + (sample2_usage.response_tokens or 0),
            total_tokens=(sample1_usage.total_tokens or 0)
            + (sample2_usage.total_tokens or 0),
        )
        assert agent_usage["model_a"] == expected

        # Test adding to multiple models
        agent_usage["model_b"] += sample1_usage
        agent_usage["model_c"] += sample2_usage
        assert agent_usage["model_b"] == sample1_usage
        assert agent_usage["model_c"] == sample2_usage

    def test_aggregation_properties(
        self, agent_usage: AgentUsage, sample1_usage: Usage, usage_with_none: Usage
    ) -> None:
        """Test usage aggregation across models."""
        # Test normal aggregation
        agent_usage["model_a"] = sample1_usage
        agent_usage["model_b"] = sample1_usage
        assert agent_usage.requests == (sample1_usage.requests or 0) * 2
        assert agent_usage.request_tokens == (sample1_usage.request_tokens or 0) * 2
        assert agent_usage.response_tokens == (sample1_usage.response_tokens or 0) * 2
        assert agent_usage.total_tokens == (sample1_usage.total_tokens or 0) * 2

        # Test handling None values
        agent_usage["model_c"] = usage_with_none
        # None values should be treated as 0 in aggregation
        assert agent_usage.request_tokens == (sample1_usage.request_tokens or 0) * 2
        assert agent_usage.response_tokens == (sample1_usage.response_tokens or 0) * 2
        assert agent_usage.total_tokens == (sample1_usage.total_tokens or 0) * 2


class TestAgentUsageLimits:
    """Tests for AgentUsageLimits class."""

    def test_dictionary_operations(
        self, agent_limits: AgentUsageLimits, mock_limits: UsageLimits
    ) -> None:
        """Test basic dictionary-like operations."""
        # Test setting and getting limits
        agent_limits["model_a"] = mock_limits
        assert agent_limits["model_a"] == mock_limits

        # Test auto-creation
        new_limits = agent_limits["new_model"]
        assert isinstance(new_limits, UsageLimits)

        # Test deletion
        del agent_limits["model_a"]
        assert "model_a" not in agent_limits

        # Test length and iteration
        agent_limits["model_b"] = mock_limits
        agent_limits["model_c"] = mock_limits
        assert len(agent_limits) == 3  # new_model, model_b, model_c
        assert set(agent_limits.keys()) == {"new_model", "model_b", "model_c"}

        # Test KeyError on delete
        with pytest.raises(KeyError):
            del agent_limits["nonexistent"]

    def test_multiple_models(
        self, agent_limits: AgentUsageLimits, mock_limits: UsageLimits
    ) -> None:
        """Test operations with multiple models."""
        # Test setting multiple models
        agent_limits["model_a"] = mock_limits
        agent_limits["model_b"] = mock_limits
        agent_limits["model_c"] = mock_limits

        # Verify all models have correct limits
        assert agent_limits["model_a"] == mock_limits
        assert agent_limits["model_b"] == mock_limits
        assert agent_limits["model_c"] == mock_limits


@pytest.fixture
def agent_usage() -> AgentUsage:
    """AgentUsage instance for testing."""
    return AgentUsage()


@pytest.fixture
def agent_limits() -> AgentUsageLimits:
    """AgentUsageLimits instance for testing."""
    return AgentUsageLimits()


@pytest.fixture
def sample1_usage() -> Usage:
    """Usage instance with typical values."""
    return Usage(
        requests=1,
        request_tokens=10,
        response_tokens=5,
        total_tokens=15,
    )


@pytest.fixture
def sample2_usage() -> Usage:
    """Usage instance with different values for testing addition."""
    return Usage(
        requests=3,
        request_tokens=5,
        response_tokens=100,
        total_tokens=105,
    )


@pytest.fixture
def usage_with_none() -> Usage:
    """Usage instance with None values for testing null handling."""
    return Usage(
        requests=1,
        request_tokens=None,
        response_tokens=None,
        total_tokens=None,
    )


@pytest.fixture
def mock_limits() -> UsageLimits:
    """UsageLimits instance for testing dictionary operations."""
    return UsageLimits()
