#!/usr/bin/env python3
"""
Example: Conversation Management

Demonstrates:
1. Creating and managing conversations
2. Context window handling with truncation
3. Forking and branching conversations
4. Serialization and persistence
"""
import asyncio
import sys
import tempfile
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    Conversation,
    Message,
    Role,
    OpenAIProvider,
    GPT5Nano,
)


async def main():
    print("=" * 60)
    print("CONVERSATION MANAGEMENT EXAMPLE")
    print("=" * 60)
    
    # === Example 1: Basic Conversation ===
    print("\n" + "=" * 40)
    print("Example 1: Basic Conversation")
    print("=" * 40)
    
    # Create a conversation with system message
    conv = Conversation(
        system_message="You are a helpful coding assistant.",
        max_tokens=4000,
        truncation_strategy="sliding",
    )
    
    # Add messages using convenience methods
    conv.add_user("What is Python?")
    conv.add_assistant("Python is a high-level programming language known for its simplicity and readability.")
    conv.add_user("Show me a hello world example.")
    conv.add_assistant("```python\nprint('Hello, World!')\n```")
    
    print(f"Session ID: {conv.session_id}")
    print(f"Message count: {len(conv)}")
    print("\nConversation history:")
    print(conv.format_history())
    
    # === Example 2: Message Objects ===
    print("\n" + "=" * 40)
    print("Example 2: Working with Messages")
    print("=" * 40)
    
    # Create messages directly
    msg1 = Message.user("How do I read a file?")
    msg2 = Message.assistant("You can use the open() function...")
    
    conv.add_message(msg1)
    conv.add_message(msg2)
    
    # Access messages
    print(f"Last user message: {conv.get_last_user_message().content[:50]}...")
    print(f"Last assistant message: {conv.get_last_assistant_message().content[:50]}...")
    
    # Convert to dict for API calls
    messages_dict = conv.get_messages_dict()
    print(f"\nMessages as dict (first 2):")
    for msg in messages_dict[:2]:
        print(f"  {msg['role']}: {msg['content'][:40]}...")
    
    # === Example 3: Token Counting and Truncation ===
    print("\n" + "=" * 40)
    print("Example 3: Context Window Management")
    print("=" * 40)
    
    # Create a conversation with tight token limit
    small_conv = Conversation(
        system_message="Be brief.",
        max_tokens=500,
        reserve_tokens=100,
        truncation_strategy="sliding",
    )
    
    # Add many messages
    for i in range(10):
        small_conv.add_user(f"Message {i}: " + "x" * 100)
        small_conv.add_assistant(f"Response {i}: " + "y" * 100)
    
    print(f"Total messages: {len(small_conv)}")
    print(f"Token count (GPT5Nano): {small_conv.count_tokens(GPT5Nano)}")
    
    # Get truncated messages for API
    truncated = small_conv.get_messages(model=GPT5Nano)
    print(f"Messages after truncation: {len(truncated)}")
    
    # === Example 4: Forking and Branching ===
    print("\n" + "=" * 40)
    print("Example 4: Forking Conversations")
    print("=" * 40)
    
    original = Conversation(system_message="You are helpful.")
    original.add_user("Tell me about cats.")
    original.add_assistant("Cats are wonderful pets...")
    
    # Fork creates a copy with new session ID
    forked = original.fork()
    forked.add_user("What about dogs?")
    forked.add_assistant("Dogs are loyal companions...")
    
    print(f"Original messages: {len(original)}")
    print(f"Forked messages: {len(forked)}")
    print(f"Same session? {original.session_id == forked.session_id}")
    
    # Branch from a specific point
    branched = original.branch(from_index=1)  # Keep only first message
    branched.add_user("Actually, tell me about birds.")
    
    print(f"Branched messages: {len(branched)}")
    
    # === Example 5: Serialization ===
    print("\n" + "=" * 40)
    print("Example 5: Persistence")
    print("=" * 40)
    
    # Convert to dict/JSON
    conv_dict = original.to_dict()
    print(f"Dict keys: {list(conv_dict.keys())}")
    
    conv_json = original.to_json()
    print(f"JSON length: {len(conv_json)} chars")
    
    # Restore from JSON
    restored = Conversation.from_json(conv_json)
    print(f"Restored messages: {len(restored)}")
    print(f"System message preserved: {restored.system_message is not None}")
    
    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = Path(f.name)
    
    original.save(temp_path)
    loaded = Conversation.load(temp_path)
    print(f"Loaded from file: {len(loaded)} messages")
    
    # Cleanup temp file
    temp_path.unlink()
    
    # === Example 6: Tool Calls in Conversation ===
    print("\n" + "=" * 40)
    print("Example 6: Tool Calls")
    print("=" * 40)
    
    from llm_client import ToolCall
    
    tool_conv = Conversation()
    tool_conv.add_user("What's the weather in Paris?")
    
    # Add assistant response with tool call
    tool_call = ToolCall(
        id="call_123",
        name="get_weather",
        arguments='{"city": "Paris"}'
    )
    tool_conv.add_assistant_with_tools(None, [tool_call])
    
    # Add tool result
    tool_conv.add_tool_result("call_123", "Sunny, 18°C", "get_weather")
    
    print("Conversation with tool calls:")
    print(tool_conv.format_history())
    
    # === Example 7: Using with Provider ===
    print("\n" + "=" * 40)
    print("Example 7: With Provider")
    print("=" * 40)
    
    provider = OpenAIProvider(model="gpt-5-nano")
    
    chat = Conversation(
        system_message="You are a math tutor. Be concise.",
    )
    
    # Build conversation
    chat.add_user("What is 5 + 3?")
    
    # Get response from provider
    result = await provider.complete(chat.get_messages_dict())
    
    if result.ok:
        chat.add_assistant(result.content)
        print(f"Q: {chat.get_last_user_message().content}")
        print(f"A: {result.content}")
    
    # Continue conversation
    chat.add_user("Now multiply that by 2")
    result = await provider.complete(chat.get_messages_dict())
    
    if result.ok:
        chat.add_assistant(result.content)
        print(f"\nQ: {chat.get_last_user_message().content}")
        print(f"A: {result.content}")
    
    print(f"\nFinal conversation length: {len(chat)} messages")
    
    await provider.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())

