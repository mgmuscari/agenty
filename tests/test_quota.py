"""Tests for the quota management system."""

import asyncio
from datetime import datetime, timedelta
import pytest
import pytest_asyncio
import fakeredis.aioredis
from typing import Any, AsyncGenerator, List
from dataclasses import dataclass

from agenty import Agent
from agenty.quota import (
    QuotaManager,
    QuotaConfig,
    QuotaContext,
    QuotaPattern,
    QuotaPatternType,
    ActionType,
)
from agenty.models import FunctionModel, allow_model_requests
@dataclass
class ModelResponse:
    """Mock ModelResponse for testing."""
    data: str
    model_name: str = ""
    parts: List[Any] = None

allow_model_requests(False)

@pytest.fixture
def str_test_model() -> FunctionModel:
    """Create a test function model."""
    async def test_func(messages: list, agent_info: Any) -> ModelResponse:
        # Extract the actual input from the messages
        input_str = messages[-1].parts[-1].content
        return ModelResponse(data=f"success: {input_str}")
    return FunctionModel(function=test_func)

@pytest_asyncio.fixture
async def redis_client() -> AsyncGenerator[fakeredis.aioredis.FakeRedis, None]:
    """Create a fake Redis client for testing."""
    client = fakeredis.aioredis.FakeRedis()
    yield client
    await client.close()

@pytest.fixture
def quota_config() -> QuotaConfig:
    """Create a test quota configuration."""
    return QuotaConfig(
        window_size=60,
        max_tokens=1000,
        max_requests=10,
        circuit_breaker_threshold=3,
        circuit_reset_timeout=1000,
        concurrent_limit=2,
        error_threshold=0.2,
        cost_limit=1.0
    )

@pytest_asyncio.fixture
async def quota_manager(redis_client: fakeredis.aioredis.FakeRedis, quota_config: QuotaConfig) -> QuotaManager:
    """Create a QuotaManager instance with fake Redis."""
    manager = QuotaManager(redis_url="redis://fake", config=quota_config)
    manager._redis = redis_client
    try:
        yield manager
    finally:
        await manager.close()

@pytest.mark.asyncio
async def test_quota_check_basic(quota_manager: QuotaManager):
    """Test basic quota checking functionality."""
    # Should allow requests within limits
    assert await quota_manager.check_quota("test", 100)
    
    # Should deny requests exceeding token limit
    assert not await quota_manager.check_quota("test", 2000)

@pytest.mark.asyncio
async def test_quota_context_tracking(quota_manager: QuotaManager):
    """Test quota context tracking and updates."""
    async with quota_manager.quota_context("test", 100):
        pass
    
    context = await quota_manager._get_quota_context("test")
    assert context.tokens_used == 100
    assert context.request_count == 1
    assert context.error_count == 0

@pytest.mark.asyncio
async def test_quota_error_handling(quota_manager: QuotaManager):
    """Test error handling and circuit breaker functionality."""
    # Simulate errors to trigger circuit breaker
    for _ in range(quota_manager.config.circuit_breaker_threshold):
        with pytest.raises(Exception):
            async with quota_manager.quota_context("test", 100):
                raise Exception("Test error")
    
    # Should be blocked by circuit breaker
    assert not await quota_manager.check_quota("test", 100)

@pytest.mark.asyncio
async def test_quota_pattern_detection(quota_manager: QuotaManager):
    """Test quota usage pattern detection."""
    # Simulate approaching token limit
    async with quota_manager.quota_context("test", int(quota_manager.config.max_tokens * 0.9)):
        pass
    
    context = await quota_manager._get_quota_context("test")
    pattern = await quota_manager._detect_patterns("test", context)
    
    assert pattern is not None
    assert pattern.type == QuotaPatternType.TOKEN_EXHAUSTION
    assert pattern.confidence > 0.8

@pytest.mark.asyncio
async def test_concurrent_operations(quota_manager: QuotaManager):
    """Test concurrent operation handling."""
    async def operation(i: int) -> None:
        async with quota_manager.quota_context(f"test", 100):
            await asyncio.sleep(0.1)  # Simulate work
    
    # Run operations concurrently
    tasks = [operation(i) for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count exceptions that indicate quota limit
    quota_exceeded = sum(1 for r in results if isinstance(r, Exception) and "Quota exceeded" in str(r))
    
    # Verify that some operations were rejected due to concurrent limit
    assert quota_exceeded > 0, "Expected some operations to be rejected due to concurrent limit"
    assert quota_exceeded >= len(tasks) - quota_manager.config.concurrent_limit

@pytest.mark.asyncio
async def test_quota_reset(quota_manager: QuotaManager):
    """Test quota reset functionality."""
    async with quota_manager.quota_context("test", 100):
        pass
    
    await quota_manager.reset_quota("test")
    context = await quota_manager._get_quota_context("test")
    assert context.tokens_used == 0
    assert context.request_count == 0

@pytest.mark.asyncio
async def test_agent_quota_integration(quota_manager: QuotaManager):
    """Test integration between Agent and QuotaManager."""
    # Create a simple test agent that just counts calls
    calls = 0
    async def test_func(messages: list, agent_info: Any) -> str:
        nonlocal calls
        calls += 1
        input_str = messages[-1].parts[-1].content
        return f"call {calls}: {input_str}"

    class QuotaAgent(Agent[str, str]):
        input_schema = str
        output_schema = str
        model = FunctionModel(function=test_func)
        quota_key = "test_agent"
        estimated_tokens_per_request = 100
        max_retries_on_quota = 2
        retry_backoff = 1.1

    agent = QuotaAgent()
    
    # First call should succeed
    await agent.run("test")
    assert calls == 1
    
    # Set quota to exceeded
    context = await quota_manager._get_quota_context("test_agent")
    context.tokens_used = quota_manager.config.max_tokens
    await quota_manager._update_quota_context("test_agent", context)
    
    # Next call should fail with quota exceeded
    with pytest.raises(Exception) as exc_info:
        await agent.run("test")
    assert "Quota exceeded" in str(exc_info.value)
    assert calls == 1  # Call count shouldn't increase

@pytest.mark.asyncio
async def test_pattern_history(quota_manager: QuotaManager):
    """Test pattern history tracking."""
    key = "test_history"
    
    # Simulate multiple patterns
    context = QuotaContext(
        time_window=timedelta(seconds=60),
        tokens_used=900,  # 90% of limit
        request_count=9,  # 90% of limit
        error_count=0,
        pattern_history=[]
    )
    
    await quota_manager._update_quota_context(key, context)
    pattern = await quota_manager._detect_patterns(key, context)
    
    assert pattern is not None
    assert pattern.type in [QuotaPatternType.TOKEN_EXHAUSTION, QuotaPatternType.RATE_LIMIT_APPROACH]
    
    # Update context with pattern
    context.pattern_history.append(pattern.type)
    await quota_manager._update_quota_context(key, context)
    
    # Verify pattern history
    updated_context = await quota_manager._get_quota_context(key)
    assert len(updated_context.pattern_history) == 1
    assert updated_context.pattern_history[0] == pattern.type

@pytest.mark.asyncio
async def test_redis_fallback(quota_config: QuotaConfig):
    """Test local fallback when Redis is unavailable."""
    # Create manager without Redis
    manager = QuotaManager(config=quota_config)
    
    async with manager.quota_context("test", 100):
        pass
    
    # Should track in local state
    assert "test" in manager._local_quotas
    assert manager._local_quotas["test"].tokens_used == 100
