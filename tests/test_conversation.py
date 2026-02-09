"""
Tests for the Conversation class.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

from llm_client.providers.types import Role, ToolCall


class TestConversationBasics:
    """Basic conversation functionality tests."""

    def test_create_empty_conversation(self):
        """Test creating an empty conversation."""
        from llm_client import Conversation

        conv = Conversation()

        assert len(conv) == 0
        assert conv.system_message is None

    def test_create_with_system_message(self):
        """Test creating conversation with system message."""
        from llm_client import Conversation

        conv = Conversation(system_message="You are helpful.")

        assert conv.system_message == "You are helpful."
        assert len(conv) == 0  # System message doesn't count

    def test_add_messages(self):
        """Test adding messages to conversation."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_user("Hello")
        conv.add_assistant("Hi there!")
        conv.add_user("How are you?")

        assert len(conv) == 3
        assert conv[0].role == Role.USER
        assert conv[0].content == "Hello"
        assert conv[1].role == Role.ASSISTANT
        assert conv[1].content == "Hi there!"

    def test_add_assistant_with_tools(self):
        """Test adding assistant message with tool calls."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_user("Get weather")

        tool_calls = [ToolCall(id="call_1", name="weather", arguments='{"city": "NYC"}')]
        conv.add_assistant_with_tools(
            content=None,
            tool_calls=tool_calls,
        )

        assert len(conv) == 2
        assert conv[1].tool_calls is not None
        assert len(conv[1].tool_calls) == 1

    def test_add_tool_result(self):
        """Test adding tool result message."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_tool_result(
            tool_call_id="call_1",
            content="Sunny, 72°F",
            name="weather",
        )

        assert len(conv) == 1
        assert conv[0].role == Role.TOOL
        assert conv[0].tool_call_id == "call_1"


class TestConversationAccessors:
    """Test conversation accessor methods."""

    def test_get_last_message(self):
        """Test getting last message."""
        from llm_client import Conversation

        conv = Conversation()

        assert conv.get_last_message() is None

        conv.add_user("First")
        conv.add_assistant("Second")

        assert conv.get_last_message().content == "Second"

    def test_get_last_user_message(self):
        """Test getting last user message."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_user("User 1")
        conv.add_assistant("Response")
        conv.add_user("User 2")

        last_user = conv.get_last_user_message()
        assert last_user.content == "User 2"

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_user("Query")
        conv.add_assistant("Response 1")
        conv.add_user("Follow-up")
        conv.add_assistant("Response 2")

        last_assistant = conv.get_last_assistant_message()
        assert last_assistant.content == "Response 2"

    def test_iteration(self):
        """Test iterating over conversation."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_user("A")
        conv.add_assistant("B")
        conv.add_user("C")

        contents = [msg.content for msg in conv]
        assert contents == ["A", "B", "C"]


class TestConversationGetMessages:
    """Test get_messages with various options."""

    def test_get_messages_includes_system(self):
        """Test that get_messages includes system message by default."""
        from llm_client import Conversation

        conv = Conversation(system_message="Be helpful.")
        conv.add_user("Hello")

        messages = conv.get_messages()

        assert len(messages) == 2
        assert messages[0].role == Role.SYSTEM
        assert messages[0].content == "Be helpful."

    def test_get_messages_dict_format(self):
        """Test get_messages_dict returns proper format."""
        from llm_client import Conversation

        conv = Conversation(system_message="System")
        conv.add_user("User message")

        messages_dict = conv.get_messages_dict()

        assert isinstance(messages_dict, list)
        assert all(isinstance(m, dict) for m in messages_dict)
        assert messages_dict[0]["role"] == "system"
        assert messages_dict[1]["role"] == "user"


class TestConversationSerialization:
    """Test conversation serialization."""

    def test_to_dict_from_dict(self):
        """Test round-trip through dictionary."""
        from llm_client import Conversation

        conv = Conversation(system_message="System prompt")
        conv.add_user("Hello")
        conv.add_assistant("Hi!")

        data = conv.to_dict()
        restored = Conversation.from_dict(data)

        assert restored.system_message == "System prompt"
        assert len(restored) == 2
        assert restored[0].content == "Hello"
        assert restored[1].content == "Hi!"

    def test_to_json_from_json(self):
        """Test round-trip through JSON string."""
        from llm_client import Conversation

        conv = Conversation(system_message="Be concise.")
        conv.add_user("What is 2+2?")
        conv.add_assistant("4")

        json_str = conv.to_json()
        restored = Conversation.from_json(json_str)

        assert restored.system_message == "Be concise."
        assert len(restored) == 2

    def test_save_load_file(self):
        """Test saving and loading from file."""
        from llm_client import Conversation

        with TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "conv.json"

            conv = Conversation(system_message="Test system")
            conv.add_user("Test user")
            conv.add_assistant("Test assistant")

            conv.save(str(filepath))

            assert filepath.exists()

            loaded = Conversation.load(str(filepath))

            assert loaded.system_message == "Test system"
            assert len(loaded) == 2

    def test_preserves_tool_calls_on_serialization(self):
        """Test that tool calls survive serialization."""
        from llm_client import Conversation

        conv = Conversation()
        conv.add_user("Get weather")

        tool_calls = [ToolCall(id="call_abc", name="get_weather", arguments='{"city": "LA"}')]
        conv.add_assistant_with_tools(content=None, tool_calls=tool_calls)
        conv.add_tool_result("call_abc", "Sunny", "get_weather")

        data = conv.to_dict()
        restored = Conversation.from_dict(data)

        assert len(restored) == 3
        assert restored[1].tool_calls is not None
        assert restored[1].tool_calls[0].name == "get_weather"
        assert restored[2].role == Role.TOOL


class TestConversationManagement:
    """Test conversation management methods."""

    def test_clear_preserves_system(self):
        """Test clear keeps system message."""
        from llm_client import Conversation

        conv = Conversation(system_message="Keep me")
        conv.add_user("Remove me")
        conv.add_assistant("Remove me too")

        conv.clear()

        assert conv.system_message == "Keep me"
        assert len(conv) == 0

    def test_fork_creates_copy(self):
        """Test fork creates independent copy."""
        from llm_client import Conversation

        conv = Conversation(system_message="Original")
        conv.add_user("Message")

        forked = conv.fork()

        # Same content
        assert forked.system_message == "Original"
        assert len(forked) == 1

        # But independent - modifying one doesn't affect other
        forked.add_assistant("New message")
        assert len(forked) == 2
        assert len(conv) == 1

    def test_branch_from_index(self):
        """Test branching from specific point."""
        from llm_client import Conversation

        conv = Conversation(system_message="System")
        conv.add_user("M1")
        conv.add_assistant("M2")
        conv.add_user("M3")
        conv.add_assistant("M4")

        branched = conv.branch(from_index=2)

        assert len(branched) == 2  # Only M1 and M2
        assert branched[0].content == "M1"
        assert branched[1].content == "M2"


class TestConversationFormatting:
    """Test conversation formatting."""

    def test_format_history(self):
        """Test format_history produces readable output."""
        from llm_client import Conversation

        conv = Conversation(system_message="You are helpful.")
        conv.add_user("Hello")
        conv.add_assistant("Hi there!")

        formatted = conv.format_history()

        assert "SYSTEM:" in formatted or "system" in formatted.lower()
        assert "USER:" in formatted or "user" in formatted.lower()
        assert "ASSISTANT:" in formatted or "assistant" in formatted.lower()
        assert "Hello" in formatted
        assert "Hi there!" in formatted
