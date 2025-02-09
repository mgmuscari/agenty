# Standard library imports
from functools import wraps
import logging
from typing import Any, Callable, Concatenate, cast

# Third-party imports
from pydantic_ai.tools import RunContext, ToolParams
from typing_extensions import ParamSpec

# Local imports
from agenty.agent.base import Agent
from agenty.types import AgentInputT, AgentOutputT

# Type variables
HookParam = ParamSpec("HookParam", default=...)

# All exports
__all__ = ["hook", "tool"]

# Logger setup
logger = logging.getLogger(__name__)


class hook:
    @staticmethod
    def input(
        func: Callable[HookParam, AgentInputT],
    ) -> Callable[HookParam, AgentInputT]:
        """Decorator to register a method as an input hook. It provides a simple way to designate methods that handle input processing.

        Args:
            func: The function to be registered as an input hook.

        Returns:
            The original function, marked as an input hook.

        Example:
            ```python
            class MyAgent(Agent[str, str]):
                @hook.input
                def process_input(self, input: str) -> str:
                    return input.upper()
            ```
        """
        setattr(func, "_is_hook_input", True)
        return func

    @staticmethod
    def output(
        func: Callable[HookParam, AgentOutputT],
    ) -> Callable[HookParam, AgentOutputT]:
        """Decorator to register a method as an output hook. It provides a simple way to designate methods that handle output processing.

        Args:
            func: The function to be registered as an output hook.

        Returns:
            The original function, marked as an output hook.

        Example:
            ```python
            class MyAgent(Agent[str, str]):
                @hook.output
                def process_output(self, output: str) -> str:
                    return output.lower()
            ```
        """
        setattr(func, "_is_hook_output", True)
        return func


# TODO: Do more research...
# https://stackoverflow.com/questions/19314405/how-to-detect-if-decorator-has-been-applied-to-method-or-function
def tool(
    func: Callable[ToolParams, AgentOutputT],
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
        ctx: RunContext[Agent[AgentInputT, AgentOutputT]], *args: Any, **kwargs: Any
    ) -> AgentOutputT:
        self = ctx.deps
        _func = cast(
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

    return cast(Callable[ToolParams, AgentOutputT], wrapper)
