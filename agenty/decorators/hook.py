from typing import Callable

from agenty.types import AgentInputT, AgentOutputT


class hook:
    @staticmethod
    def input(
        func: Callable[[AgentInputT], AgentInputT],
    ) -> Callable[[AgentInputT], AgentInputT]:
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
        func: Callable[[AgentOutputT], AgentOutputT],
    ) -> Callable[[AgentOutputT], AgentOutputT]:
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
