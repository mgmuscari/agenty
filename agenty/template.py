from typing import Any
from textwrap import dedent

from jinja2.sandbox import SandboxedEnvironment

__all__ = ["apply_template"]


def apply_template(content: Any, ctx: dict[str, Any] = {}) -> str:
    """Render content as a Jinja2 template with the provided context.

    Args:
        content: Content to render as a template. Will be converted to string.
        ctx: Dictionary of variables available in the template context.
             Defaults to empty dict.

    Returns:
        str: The rendered template with variables substituted and whitespace
             normalized via dedent.

    Example:
        >>> apply_template("Hello {{ name }}!", {"name": "World"})
        'Hello World!'
    """
    return dedent(SandboxedEnvironment().from_string(str(content)).render(**ctx))
