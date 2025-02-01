from functools import wraps
import logging
from typing import cast, Any, Callable, Concatenate

from pydantic_ai.tools import RunContext, ToolParams

from agenty.agent import Agent
from agenty.types import AgentInputT, AgentOutputT

logger = logging.getLogger(__name__)

# TODO: Do more research...
# https://stackoverflow.com/questions/19314405/how-to-detect-if-decorator-has-been-applied-to-method-or-function


def tool(
    func: Callable[ToolParams, AgentOutputT]
) -> Callable[ToolParams, AgentOutputT]:
    setattr(func, "_is_tool", True)

    @wraps(func)
    def wrapper(
        ctx: RunContext[Agent[AgentInputT, AgentOutputT]], *args, **kwargs
    ) -> Any:
        self = ctx.deps
        _func: Callable[
            Concatenate[Agent[AgentInputT, AgentOutputT], ToolParams],
            AgentOutputT,
        ] = cast(
            Callable[
                Concatenate[Agent[AgentInputT, AgentOutputT], ToolParams],
                AgentOutputT,
            ],
            func,
        )
        result = _func(self, *args, **kwargs)
        logger.debug(
            {
                "tool": func.__name__,
                "result": result,
                "args": args,
                "kwargs": kwargs,
                "agent": type(self).__name__,
            }
        )
        return result

    return wrapper
