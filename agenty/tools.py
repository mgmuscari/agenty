from functools import wraps
import logging
from typing import cast, Any, Callable, Concatenate

from pydantic_ai.tools import RunContext, ToolParams

from agenty.agent import Agent
from agenty.types import AgentInputT, AgentOutputT

__all__ = ["tool"]

logger = logging.getLogger(__name__)


# TODO: Do more research...
# https://stackoverflow.com/questions/19314405/how-to-detect-if-decorator-has-been-applied-to-method-or-function
def tool(
    func: Callable[ToolParams, AgentOutputT]
) -> Callable[ToolParams, AgentOutputT]:
    """Decorator to register a method as an agent tool.

    This decorator marks a method as a tool that can be called by the agent during
    execution. It handles proper type casting and logging of tool invocations.

    Args:
        func: The function to be registered as a tool. Must be a method of an Agent
             subclass with proper type annotations.

    Returns:
        A wrapped function that maintains the original function's type signature
        while providing tool registration and logging functionality.

    Example:
        ```python
        class MyAgent(Agent[str, str]):
            @tool
            def my_tool(self, param: str) -> str:
                return f"Processed {param}"
        ```
    """
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
