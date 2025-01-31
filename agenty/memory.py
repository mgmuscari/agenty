from typing import Optional, Literal, Union, Sequence, overload, Iterable
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
from agenty.types import AgentIO, BaseIO

Role = Literal["user", "assistant", "tool", "system", "developer", "function"]


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

    def to_openai(self) -> ChatCompletionMessageParam:
        """Convert message to OpenAI API format."""
        content = self.content
        if isinstance(content, BaseModel):
            content = content.model_dump_json()
        return {
            "role": self.role,  # type: ignore
            "content": content,
            "name": str(self.name),
        }

    def to_pydantic_ai(self) -> ModelMessage:
        """Convert message to Pydantic AI format."""
        match self.role:
            case "user":
                return ModelRequest(parts=[UserPromptPart(str(self.content))])
            case "system":
                return ModelRequest(parts=[SystemPromptPart(str(self.content))])
            case "assistant":
                return ModelResponse(parts=[TextPart(str(self.content))])
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

    def _cull_history(self) -> None:
        """Remove oldest messages if exceeding max_messages."""
        if self.max_messages >= 0:
            while len(self._messages) > self.max_messages:
                self._messages.pop(0)

    def end_turn(self) -> None:
        """End current conversation turn."""
        self.current_turn_id = None

    @overload
    def __getitem__(self, index: int) -> ChatMessage: ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[ChatMessage]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[ChatMessage, MutableSequence[ChatMessage]]:
        """Get message(s) by index/slice."""
        return self._messages[index]

    def __setitem__(
        self,
        index: Union[int, slice],
        value: Union[ChatMessage, Iterable[ChatMessage]],
    ) -> None:
        """Set message(s) at index/slice."""
        if isinstance(index, slice):
            if not isinstance(value, Sequence):
                raise TypeError("Can only assign sequence to slice")
            self._messages[index] = value
        else:
            if not isinstance(value, ChatMessage):
                raise TypeError("Can only assign ChatMessage")
            self._messages[index] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        """Delete message(s) at index/slice."""
        del self._messages[index]

    def __len__(self) -> int:
        """Get number of messages in history."""
        return len(self._messages)

    def insert(self, index: int, value: ChatMessage) -> None:
        """Insert message at index."""
        self._messages.insert(index, value)
        self._cull_history()

    def to_openai(self) -> list[ChatCompletionMessageParam]:
        """Get history in OpenAI API format."""
        return [msg.to_openai() for msg in self._messages]

    def to_pydantic_ai(self) -> list[ModelMessage]:
        """Get history in Pydantic-AI format."""
        return [msg.to_pydantic_ai() for msg in self._messages]
