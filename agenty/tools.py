from functools import wraps
import logging
from typing import Callable

from pydantic_ai.tools import RunContext

from agenty.agent import Agent
from agenty.types import AgentInputT, AgentOutputT

logger = logging.getLogger(__name__)


def tool(func: Callable) -> Callable:
    setattr(func, "_is_tool", True)

    @wraps(func)
    def wrapper(ctx: RunContext[Agent[AgentInputT, AgentOutputT]], *args, **kwargs):
        self = ctx.deps
        logger.debug(
            {
                "tool": func.__name__,
                "agent": type(self).__name__,
                "msg": f"Called tool",
                "args": args,
                "kwargs": kwargs,
            }
        )
        return func(self, *args, **kwargs)

    return wrapper
