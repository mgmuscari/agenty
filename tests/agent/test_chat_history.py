import pytest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    SystemPromptPart,
    TextPart,
)

from agenty.agent.chat_history import ChatHistory, ChatMessage, Role
import agenty.exceptions as exc


def test_chat_message_template() -> None:
    # Test basic template substitution
    user = ChatMessage(role="user", content="My name is {{ NAME }}")
    user_content = user.content_str(ctx={"NAME": "Alice"})
    assert user_content == "My name is Alice"

    # Test missing variable
    user_content = user.content_str(ctx={"NOT_NAME": "Bob"})
    assert user_content != "My name is Bob"

    # Test complex template context
    complex_msg = ChatMessage(
        role="user", content="Hello {{ user.name }}, your score is {{ stats.score }}"
    )
    context = {"user": {"name": "Alice"}, "stats": {"score": 100}}
    assert complex_msg.content_str(ctx=context) == "Hello Alice, your score is 100"


def test_chat_message_to_openai() -> None:
    # Test basic conversion
    msg = ChatMessage(role="user", content="Hello")
    openai_msg = msg.to_openai()
    assert openai_msg.get("role") == "user"
    assert openai_msg.get("content") == "Hello"

    # Test with name
    msg = ChatMessage(role="user", content="Hello", name="Alice")
    openai_msg = msg.to_openai()
    assert openai_msg.get("name") == "Alice"

    # Test with template
    msg = ChatMessage(role="user", content="Hello {{ NAME }}")
    openai_msg = msg.to_openai(ctx={"NAME": "Bob"})
    assert openai_msg.get("content") == "Hello Bob"


def test_chat_to_pai() -> None:
    # Test basic conversions
    user = ChatMessage(role="user", content="{{ TEST_VAR }}")
    system = ChatMessage(role="system", content="{{ TEST_VAR }}")
    assistant = ChatMessage(role="assistant", content="{{ TEST_VAR }}")

    # Test with template context
    ctx = {"TEST_VAR": "Hello"}
    user_model = user.to_pydantic_ai(ctx)
    system_model = system.to_pydantic_ai(ctx)
    assistant_model = assistant.to_pydantic_ai(ctx)

    # Verify types
    assert isinstance(user_model, ModelRequest)
    assert isinstance(system_model, ModelRequest)
    assert isinstance(assistant_model, ModelResponse)

    # Verify part types first
    for part in user_model.parts:
        assert isinstance(part, UserPromptPart)
    for part in system_model.parts:
        assert isinstance(part, SystemPromptPart)
    for part in assistant_model.parts:
        assert isinstance(part, TextPart)

    # Then verify content with proper type checking
    assert isinstance(user_model.parts[0], UserPromptPart)
    assert isinstance(system_model.parts[0], SystemPromptPart)
    assert isinstance(assistant_model.parts[0], TextPart)
    assert user_model.parts[0].content == "Hello"
    assert system_model.parts[0].content == "Hello"
    assert assistant_model.parts[0].content == "Hello"

    # Test invalid role
    with pytest.raises(ValueError):
        ChatMessage(role="developer", content="Test").to_pydantic_ai()


def test_role_validation() -> None:
    # Test valid roles
    valid_roles: list[Role] = [
        "user",
        "assistant",
        "tool",
        "system",
        "developer",
        "function",
    ]
    for role in valid_roles:
        msg = ChatMessage(role=role, content="test")
        assert msg.role == role

    # Test invalid role
    with pytest.raises(ValueError):
        ChatMessage(role="invalid", content="test")  # type: ignore


def test_agent_chat_history_initialization() -> None:
    # Test default initialization
    chat_history = ChatHistory()
    assert len(chat_history) == 0
    assert chat_history.current_turn_id is None
    assert chat_history.max_messages == -1

    # Test with max messages
    chat_history = ChatHistory(max_messages=5)
    assert chat_history.max_messages == 5

    # Test with initial messages
    initial_msgs = [
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi"),
    ]
    chat_history = ChatHistory(messages=initial_msgs)
    assert len(chat_history) == 2
    assert chat_history[0].content == "Hello"
    assert chat_history[1].content == "Hi"


def test_agent_chat_history_turn_management() -> None:
    chat_history = ChatHistory()

    # Test turn initialization
    chat_history.initialize_turn()
    turn_id = chat_history.current_turn_id
    assert turn_id is not None

    # Test messages share turn ID
    chat_history.add("user", "Hello")
    chat_history.add("assistant", "Hi")
    assert chat_history[0].turn_id == turn_id
    assert chat_history[1].turn_id == turn_id

    # Test end turn
    chat_history.end_turn()
    assert chat_history.current_turn_id is None

    # Test new turn gets new ID
    chat_history.add("user", "How are you?")
    new_turn_id = chat_history.current_turn_id
    assert new_turn_id is not None
    assert new_turn_id != turn_id


def test_agent_chat_history_add(
    agent_chat_history: ChatHistory,
    system_message: str,
    user_message: str,
    assistant_message: str,
    user_name: str,
) -> None:
    system = ChatMessage(role="system", content=system_message)
    user = ChatMessage(role="user", content=user_message, name=user_name)
    assistant = ChatMessage(role="assistant", content=assistant_message)

    agent_chat_history.add("system", system_message)
    assert len(agent_chat_history) == 1
    agent_chat_history.add("user", user_message, name=user_name)
    agent_chat_history.add("assistant", assistant_message)
    assert len(agent_chat_history) == 3

    assert agent_chat_history[0].role == system.role
    assert agent_chat_history[0].content == system.content
    assert agent_chat_history[0].name is None
    assert agent_chat_history[0].turn_id is not None

    assert agent_chat_history[1].role == user.role
    assert agent_chat_history[1].content == user.content
    assert agent_chat_history[1].name == user.name
    assert agent_chat_history[1].turn_id == agent_chat_history[0].turn_id

    assert agent_chat_history[2].role == assistant.role
    assert agent_chat_history[2].content == assistant.content
    assert agent_chat_history[2].name == assistant.name
    assert agent_chat_history[2].turn_id == agent_chat_history[0].turn_id


def test_agent_chat_history_max(agent_chat_history_max: ChatHistory) -> None:
    max_messages = agent_chat_history_max.max_messages

    # Fill chat_history to capacity
    for i in range(max_messages):
        agent_chat_history_max.add("developer", str(i))
        assert agent_chat_history_max[i].content == str(i)

    # Test overflow behavior
    agent_chat_history_max.add("developer", "overflow")
    assert len(agent_chat_history_max) == max_messages

    # Verify FIFO behavior
    assert agent_chat_history_max[0].content == "1"  # Oldest message removed
    assert agent_chat_history_max[-1].content == "overflow"  # New message at end

    # Test insert with max messages
    agent_chat_history_max.insert(0, ChatMessage(role="user", content="inserted"))
    assert len(agent_chat_history_max) == max_messages
    # We insert at 0, but since the oldest message is at 0, it is immediately removed
    assert agent_chat_history_max[0].content == "1"
    assert agent_chat_history_max[-1].content == "overflow"


def test_agent_chat_history_setitem() -> None:
    chat_history = ChatHistory()
    msg1 = ChatMessage(role="user", content="Hello")
    msg2 = ChatMessage(role="assistant", content="Hi")
    chat_history.append(msg1)

    # Test single item assignment
    chat_history[0] = msg2
    assert chat_history[0].content == "Hi"

    # Test slice assignment
    chat_history.extend([msg1, msg2])
    chat_history[1:] = [msg2, msg1]
    assert chat_history[1].content == "Hi"
    assert chat_history[2].content == "Hello"

    # # Test invalid assignments
    with pytest.raises(exc.AgentyTypeError):
        chat_history[0] = "invalid"  # type: ignore

    with pytest.raises(exc.AgentyTypeError):
        chat_history[1:] = ["invalid"]  # type: ignore


def test_agent_chat_history_conversion() -> None:
    chat_history = ChatHistory()
    chat_history.add("user", "Hello {{ NAME }}")
    chat_history.add("assistant", "Hi {{ NAME }}")

    # Test OpenAI conversion
    ctx = {"NAME": "Alice"}
    openai_msgs = chat_history.to_openai(ctx)
    assert len(openai_msgs) == 2
    assert openai_msgs[0].get("content") == "Hello Alice"
    assert openai_msgs[1].get("content") == "Hi Alice"

    # Test Pydantic-AI conversion
    pai_msgs = chat_history.to_pydantic_ai(ctx)
    assert len(pai_msgs) == 2
    assert isinstance(pai_msgs[0], ModelRequest)
    assert isinstance(pai_msgs[1], ModelResponse)
    assert isinstance(pai_msgs[0].parts[0], UserPromptPart)
    assert isinstance(pai_msgs[1].parts[0], TextPart)
    assert pai_msgs[0].parts[0].content == "Hello Alice"
    assert pai_msgs[1].parts[0].content == "Hi Alice"


def test_chat_history_clear(
    agent_chat_history: ChatHistory,
    system_message: str,
    user_message: str,
):
    agent_chat_history.add("system", system_message)
    agent_chat_history.add("user", user_message)

    assert len(agent_chat_history) == 2

    agent_chat_history.clear()
    assert len(agent_chat_history) == 0
    assert agent_chat_history.current_turn_id is None


@pytest.fixture
def agent_chat_history():
    return ChatHistory()


@pytest.fixture
def agent_chat_history_max():
    return ChatHistory(max_messages=5)


@pytest.fixture
def user_message() -> str:
    return "Hello I'm a user!"


@pytest.fixture
def user_name() -> str:
    return "agenty"


@pytest.fixture
def system_message() -> str:
    return "This is a system message."


@pytest.fixture
def assistant_message() -> str:
    return "This is a response from an AI."
