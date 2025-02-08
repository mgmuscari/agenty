import logging
from typing import Any, Callable, List, Type

import pydantic_ai as pai
from pydantic_ai.models.function import FunctionModel

__all__ = ["AgentMeta"]

logger = logging.getLogger(__name__)


def _noop(*args: Any, **kwargs: Any) -> Any:
    pass


class AgentMeta(type):
    """Metaclass for Agent that handles tool registration and agent configuration.

    This metaclass automatically processes tool decorators and configures the underlying
    pydantic-ai agent during class creation.

    Args:
        name (str): The name of the class being created
        bases (tuple[type, ...]): Base classes
        namespace (dict[str, Any]): Class namespace dictionary

    Returns:
        Type: The configured agent class
    """

    def __new__(
        mcls: Type["AgentMeta"],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
    ) -> Any:
        tools: List[Callable[..., Any]] = []
        input_hooks: List[Callable[..., Any]] = []
        output_hooks: List[Callable[..., Any]] = []
        for _, value in namespace.items():
            if hasattr(value, "_is_tool"):
                tools.append(value)
            if hasattr(value, "_is_hook_input"):
                input_hooks.insert(0, value)
            if hasattr(value, "_is_hook_output"):
                output_hooks.insert(0, value)
        cls = super().__new__(mcls, name, bases, namespace)
        setattr(cls, "_input_hooks", input_hooks)
        setattr(cls, "_output_hooks", output_hooks)
        try:
            model = namespace.get("model")
            if model is None:
                model = FunctionModel(_noop)
            pai_agent = pai.Agent(
                model,
                deps_type=cls,
                name=name,
                result_type=getattr(cls, "output_schema", str),
                system_prompt=getattr(cls, "system_prompt", ""),
                model_settings=getattr(cls, "model_settings", None),
                retries=getattr(cls, "retries", 1),
                result_retries=getattr(cls, "result_retries", None),
                end_strategy=getattr(cls, "end_strategy", "early"),
            )
            # Set the pai agent as a private class attribute
            setattr(cls, "_pai_agent", pai_agent)

            # Add tools to the pai agent
            # TODO: Add support for tool decorator with parameters
            tool_decorator = pai_agent.tool(
                retries=None,
                docstring_format="auto",
                require_parameter_descriptions=False,
            )
            for tool in tools:
                tool_decorator(tool)
                logger.debug(
                    {
                        "tool": tool.__name__,
                        "agent": cls.__name__,
                        "msg": "added tool to agent",
                    }
                )
        except Exception:
            pass
        return cls
