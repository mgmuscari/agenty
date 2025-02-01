from typing import Any
from textwrap import dedent

from jinja2.sandbox import SandboxedEnvironment


def apply_template(content: Any, ctx: dict[str, Any] = {}) -> str:
    """Get message content as a string and render Jinja2 template."""
    return dedent(SandboxedEnvironment().from_string(str(content)).render(**ctx))
