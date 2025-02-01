import pytest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    SystemPromptPart,
    TextPart,
)

from agenty.components.memory import AgentMemory, ChatMessage


def test_chat_message_template() -> None:
    user = ChatMessage(role="user", content="My name is {{ NAME }}")
    user_content = user.content_str(ctx={"NAME": "Alice"})
    assert user_content == "My name is Alice"
    user_content = user.content_str(ctx={"NOT_NAME": "Bob"})
    assert user_content != "My name is Bob"


def test_chat_to_pai() -> None:
    user = ChatMessage(role="user", content="{{ TEST_VAR }}")
    system = ChatMessage(role="system", content="{{ TEST_VAR }}")
    assistant = ChatMessage(role="assistant", content="{{ TEST_VAR }}")
    user_model = user.to_pydantic_ai()
    system_model = system.to_pydantic_ai()
    assistant_model = assistant.to_pydantic_ai()
    assert isinstance(user_model, ModelRequest)
    assert isinstance(system_model, ModelRequest)
    assert isinstance(assistant_model, ModelResponse)
    for part in user_model.parts:
        assert isinstance(part, UserPromptPart)
    for part in system_model.parts:
        assert isinstance(part, SystemPromptPart)
    for part in assistant_model.parts:
        assert isinstance(part, TextPart)

    with pytest.raises(ValueError):
        ChatMessage(role="developer", content="Test").to_pydantic_ai()


def test_agent_memory_add(
    agent_memory: AgentMemory,
    system_message: str,
    user_message: str,
    assistant_message: str,
    user_name: str,
) -> None:
    system = ChatMessage(role="system", content=system_message)
    user = ChatMessage(role="user", content=user_message, name=user_name)
    assistant = ChatMessage(role="assistant", content=assistant_message)

    agent_memory.add("system", system_message)
    assert len(agent_memory) == 1
    agent_memory.add("user", user_message, name=user_name)
    agent_memory.add("assistant", assistant_message)
    assert len(agent_memory) == 3

    assert agent_memory[0].role == system.role
    assert agent_memory[0].content == system.content
    assert agent_memory[0].name is None
    assert agent_memory[0].turn_id is not None

    assert agent_memory[1].role == user.role
    assert agent_memory[1].content == user.content
    assert agent_memory[1].name == user.name
    assert agent_memory[1].turn_id == agent_memory[0].turn_id

    assert agent_memory[2].role == assistant.role
    assert agent_memory[2].content == assistant.content
    assert agent_memory[2].name == assistant.name
    assert agent_memory[2].turn_id == agent_memory[0].turn_id


def test_agent_memory_max(
    agent_memory_max: AgentMemory,
):
    max_messages = agent_memory_max.max_messages
    i = 0
    for i in range(max_messages):
        agent_memory_max.add("developer", str(i))
        assert agent_memory_max[i].content == str(i)

    # Add one more message to trigger overflow
    agent_memory_max.add("developer", "overflow")
    with pytest.raises(IndexError):
        agent_memory_max[i + 1]

    # Check that the oldest message was removed
    assert agent_memory_max[0].content == "1"
    assert agent_memory_max[-1].content == "overflow"


def test_memory_clear(
    agent_memory: AgentMemory,
    system_message: str,
    user_message: str,
):
    agent_memory.add("system", system_message)
    agent_memory.add("user", user_message)

    assert len(agent_memory) == 2

    agent_memory.clear()
    assert len(agent_memory) == 0
    assert agent_memory.current_turn_id is None


@pytest.fixture
def agent_memory():
    return AgentMemory()


@pytest.fixture
def agent_memory_max():
    return AgentMemory(max_messages=5)


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


# def test_agent_message_creation():
#     """Test creating and converting agent messages."""
#     # Test with string content
#     msg = ChatMessage(role="user", content="Hello")
#     assert msg.role == "user"
#     assert msg.content == "Hello"

#     openai_msg = msg.to_openai()
#     assert isinstance(openai_msg, Dict)
#     assert openai_msg["role"] == "user"
#     assert openai_msg["content"] == "Hello"

#     # Test with structured content
#     structured_input = AgentInput(input="Test input")
#     msg = ChatMessage(role="user", content=structured_input)
#     assert msg.role == "user"
#     assert isinstance(msg.content, AgentInput)

#     openai_msg = msg.to_openai()
#     assert isinstance(openai_msg, dict)
#     assert openai_msg["role"] == "user"
#     assert isinstance(openai_msg["content"], str)


# def test_memory_initialization():
#     """Test memory initialization and basic properties."""
#     memory = AgentMemory()
#     assert len(memory) == 0
#     assert memory.current_turn_id is None
#     assert isinstance(memory.history, list)

#     memory = AgentMemory(max_messages=5)
#     assert memory.max_messages == 5


# def test_memory_add_messages():
#     """Test adding messages to memory."""
#     memory = AgentMemory()

#     # Add string messages
#     memory.add("user", "Hello")
#     memory.add("assistant", "Hi there!")

#     assert len(memory) == 2
#     assert isinstance(memory.history, list)
#     assert len(memory.history) == 2
#     assert isinstance(memory.history[0], Dict)

#     # Add structured messages
#     memory.add("user", AgentInput(input="Test input"))
#     memory.add("assistant", AgentOutput(output="Test output"))

#     assert len(memory) == 4
#     assert len(memory.history) == 4


# def test_memory_turn_management():
#     """Test conversation turn management."""
#     memory = AgentMemory()

#     # First turn
#     memory.add("user", "Hello")
#     memory.add("assistant", "Hi!")
#     turn_id = memory.current_turn_id
#     assert turn_id is not None

#     memory.end_turn()
#     assert memory.current_turn_id is None

#     # Second turn
#     memory.add("user", "How are you?")
#     memory.add("assistant", "I'm good!")
#     assert memory.current_turn_id is not None
#     assert memory.current_turn_id != turn_id


# def test_memory_max_messages():
#     """Test memory message limit enforcement."""
#     memory = AgentMemory(max_messages=3)

#     memory.add("user", "Message 1")
#     memory.add("assistant", "Response 1")
#     memory.add("user", "Message 2")

#     assert len(memory) == 3
#     assert len(memory.history) == 3

#     # Adding a fourth message should remove the oldest
#     memory.add("assistant", "Response 2")
#     assert len(memory) == 3
#     assert len(memory.history) == 3
#     assert isinstance(memory.history[0], Dict)
#     assert memory.history[0].get("content") == "Response 1"
