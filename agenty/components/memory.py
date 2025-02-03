from typing import Any, Optional, Literal, Union, Sequence, overload, Iterable
from collections.abc import MutableSequence
import uuid

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    SystemPromptPart,
    TextPart,
)
from agenty.template import apply_template
from agenty.types import AgentIO

__all__ = ["ChatMessage", "AgentMemory", "Role"]

Role = Literal["user", "assistant", "tool", "system", "developer", "function"]
"""Type for message sender roles.

Valid values:
    user: End user sending input to the agent
    assistant: AI model responses
    tool: Tool execution results
    system: System-level instructions/prompts
    developer: Developer annotations/comments
    function: Function call results
"""


class ChatMessage(BaseModel):
    """A message in the conversation history.

    Args:
        role: Message sender's role (user/assistant/tool/etc)
        content: Message content as AgentIO
        turn_id: UUID of the conversation turn
        name: Optional sender name
    """

    role: Role
    content: AgentIO
    turn_id: Optional[str] = None
    name: Optional[str] = None

    def content_str(self, ctx: dict[str, Any] = {}) -> str:
        """Get message content as a string and render Jinja2 template.

        Args:
            ctx: Template context dictionary for variable substitution

        Returns:
            str: Rendered message content
        """
        return apply_template(self.content, ctx)

    def to_openai(self, ctx: dict[str, Any] = {}) -> ChatCompletionMessageParam:
        """Convert message to OpenAI API format.

        Args:
            ctx: Template context dictionary for variable substitution

        Returns:
            ChatCompletionMessageParam: Message formatted for OpenAI API
        """
        content = self.content_str(ctx)
        return {
            "role": self.role,  # type: ignore
            "content": content,
            "name": str(self.name),
        }

    def to_pydantic_ai(self, ctx: dict[str, Any] = {}) -> ModelMessage:
        """Convert message to Pydantic AI format.

        Args:
            ctx: Template context dictionary for variable substitution

        Returns:
            ModelMessage: Message formatted for Pydantic AI

        Raises:
            ValueError: If message role is not supported
        """
        match self.role:
            case "user":
                return ModelRequest(parts=[UserPromptPart(self.content_str(ctx))])
            case "system":
                return ModelRequest(parts=[SystemPromptPart(self.content_str(ctx))])
            case "assistant":
                return ModelResponse(parts=[TextPart(self.content_str(ctx))])
            case _:
                raise ValueError(f"Unsupported role: {self.role}")


class AgentMemory(MutableSequence[ChatMessage]):
    """Manages conversation history for an AI agent.

    Implements MutableSequence for list-like access to message history.
    Handles conversation turns and optional history length limits.

    Args:
        max_messages: Max messages to keep (-1 for unlimited)
        messages: Optional initial messages
    """

    def __init__(
        self, max_messages: int = -1, messages: Optional[Sequence[ChatMessage]] = None
    ) -> None:
        self._messages: list[ChatMessage] = list(messages) if messages else []
        self.max_messages = max_messages
        self.current_turn_id: Optional[str] = None

    def initialize_turn(self) -> None:
        """Start a new conversation turn with a fresh UUID."""
        self.current_turn_id = str(uuid.uuid4())

    def add(
        self,
        role: Role,
        content: Union[AgentIO, str],
        name: Optional[str] = None,
    ) -> None:
        """Add a message to history.

        Args:
            role: Message sender's role
            content: Message content
            name: Optional sender name

        Note:
            Automatically initializes a new turn if none is active. Messages added in the same turn share a turn_id.
        """
        if self.current_turn_id is None:
            self.initialize_turn()

        message = ChatMessage(
            role=role,
            content=content,
            turn_id=self.current_turn_id,
            name=name,
        )
        self.append(message)

    def clear(self) -> None:
        self.current_turn_id = None
        return super().clear()

    def _cull_history(self) -> None:
        """Remove oldest messages if exceeding max_messages.

        This is called automatically when adding new messages if max_messages
        is set to a non-negative value.
        """
        if self.max_messages >= 0:
            while len(self._messages) > self.max_messages:
                self._messages.pop(0)

    def end_turn(self) -> None:
        """End current conversation turn.

        After calling this, the next message added will start a new turn
        with a new turn_id.
        """
        self.current_turn_id = None

    def to_openai(self, ctx: dict[str, Any] = {}) -> list[ChatCompletionMessageParam]:
        """Get history in OpenAI API format.

        Args:
            ctx: Template context dictionary for variable substitution

        Returns:
            list[ChatCompletionMessageParam]: Messages formatted for OpenAI API
        """
        return [msg.to_openai(ctx) for msg in self._messages]

    def to_pydantic_ai(self, ctx: dict[str, Any] = {}) -> list[ModelMessage]:
        """Get history in Pydantic-AI format.

        Args:
            ctx: Template context dictionary for variable substitution

        Returns:
            list[ModelMessage]: Messages formatted for Pydantic-AI
        """
        # TODO: Do we really need to template the entire message history? Maybe just the last message?
        return [msg.to_pydantic_ai(ctx) for msg in self._messages]

    @overload
    def __getitem__(self, index: int) -> ChatMessage: ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[ChatMessage]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ChatMessage, MutableSequence[ChatMessage]]:
        """Get message(s) by index/slice.

        Args:
            index: Integer index or slice

        Returns:
            Single message or sequence of messages
        """
        return self._messages[index]

    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[ChatMessage, Iterable[ChatMessage]],
    ) -> None:
        """Set message(s) at index/slice.

        Args:
            index: Integer index or slice
            value: Message or sequence of messages to set

        Raises:
            TypeError: If value type doesn't match index type
        """
        if isinstance(index, slice):
            if not isinstance(value, Sequence):
                raise TypeError("Can only assign sequence to slice")
            self._messages[index] = value
        else:
            if not isinstance(value, ChatMessage):
                raise TypeError("Can only assign ChatMessage")
            self._messages[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Delete message(s) at index/slice.

        Args:
            index: Integer index or slice
        """
        del self._messages[index]

    def __len__(self) -> int:
        """Get number of messages in history.

        Returns:
            int: Number of messages
        """
        return len(self._messages)

    def insert(self, index: int, value: ChatMessage) -> None:
        """Insert message at index.

        Args:
            index: Position to insert at
            value: Message to insert

        Note:
            May trigger history culling if max_messages is exceeded
        """
        self._messages.insert(index, value)
        self._cull_history()
