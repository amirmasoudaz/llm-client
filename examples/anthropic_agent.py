#!/usr/bin/env python3
"""
Example: Using the Anthropic (Claude) Provider

Demonstrates:
1. Basic completion with AnthropicProvider
2. Streaming responses
3. Tool calling with Claude
4. Using Agent with Anthropic
5. Session persistence (save/load)

Requirements:
    pip install llm-client[anthropic]

    Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    ANTHROPIC_AVAILABLE,
    Agent,
    AnthropicProvider,
    StreamEventType,
    tool,
)

# === Define Tools ===


@tool
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city
        units: Temperature units (celsius or fahrenheit)
    """
    # Simulated weather data
    weather_data = {
        "new york": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "cloudy"},
        "tokyo": {"temp": 28, "condition": "humid"},
        "paris": {"temp": 18, "condition": "rainy"},
        "san francisco": {"temp": 16, "condition": "foggy"},
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        data = weather_data[city_lower]
        temp = data["temp"]
        if units == "fahrenheit":
            temp = int(temp * 9 / 5 + 32)
            unit_symbol = "°F"
        else:
            unit_symbol = "°C"
        return f"Weather in {city}: {data['condition']}, {temp}{unit_symbol}"

    return f"Weather data not available for {city}"


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate (e.g., "2 + 2 * 3")
    """
    try:
        # Simple safe eval for basic math
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating expression: {e}"


async def example_basic_completion():
    """Example: Basic completion with Anthropic."""
    print("\n" + "=" * 50)
    print("Example 1: Basic Completion")
    print("=" * 50)

    provider = AnthropicProvider(
        model="claude-3-5-sonnet-20241022",  # Use a real Claude model name
        max_tokens=1024,
    )

    result = await provider.complete("What are three interesting facts about octopuses? Be brief.")

    if result.ok:
        print(f"\nResponse:\n{result.content}")
        print(f"\nUsage: {result.usage.input_tokens} in, {result.usage.output_tokens} out")
    else:
        print(f"\nError: {result.error}")


async def example_streaming():
    """Example: Streaming response from Claude."""
    print("\n" + "=" * 50)
    print("Example 2: Streaming Response")
    print("=" * 50)

    provider = AnthropicProvider(
        model="claude-3-5-sonnet-20241022",
        max_tokens=512,
    )

    print("\nStreaming response:")
    print("-" * 40)

    async for event in provider.stream("Write a haiku about programming in Python."):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.DONE:
            print("\n" + "-" * 40)
            result = event.data
            if result.usage:
                print(f"Usage: {result.usage.input_tokens} in, {result.usage.output_tokens} out")


async def example_tool_calling():
    """Example: Tool calling with Claude."""
    print("\n" + "=" * 50)
    print("Example 3: Tool Calling")
    print("=" * 50)

    provider = AnthropicProvider(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
    )

    # Create agent with tools
    agent = Agent(
        provider=provider,
        tools=[get_weather, calculate],
        system_message=(
            "You are a helpful assistant with access to weather and calculator tools. Use them when appropriate."
        ),
        max_turns=5,
    )

    print("\nQuery: What's the weather in Tokyo and what's 15 * 7?")
    print("-" * 40)

    result = await agent.run("What's the weather in Tokyo and what's 15 * 7?")

    print(f"\nFinal answer:\n{result.content}")
    print(f"\nTurns: {result.num_turns}")
    print(f"Tool calls: {[tc.name for tc in result.all_tool_calls]}")


async def example_streaming_agent():
    """Example: Streaming agent with tools."""
    print("\n" + "=" * 50)
    print("Example 4: Streaming Agent")
    print("=" * 50)

    provider = AnthropicProvider(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
    )

    agent = Agent(
        provider=provider,
        tools=[get_weather],
        system_message="You are a helpful weather assistant.",
        max_turns=3,
    )

    print("\nStreaming agent response:")
    print("-" * 40)

    async for event in agent.stream("What's the weather like in San Francisco?"):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.META:
            if event.data.get("event") == "tool_result":
                print(f"\n[Tool: {event.data['tool_name']}] {event.data['content']}")
        elif event.type == StreamEventType.DONE:
            print("\n" + "-" * 40)
            agent_result = event.data
            print(f"Status: {agent_result.status}")


async def example_session_persistence():
    """Example: Saving and loading agent sessions."""
    print("\n" + "=" * 50)
    print("Example 5: Session Persistence")
    print("=" * 50)

    provider = AnthropicProvider(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
    )

    # Create and use an agent
    agent = Agent(
        provider=provider,
        tools=[get_weather],
        system_message="You are a helpful weather assistant.",
    )

    # Have a conversation
    print("\nFirst message...")
    result = await agent.run("What's the weather in London?")
    print(f"Response: {result.content[:200]}...")

    # Save the session
    session_path = Path("/tmp/agent_session.json")
    agent.save_session(session_path)
    print(f"\nSession saved to: {session_path}")

    # Load the session with a new agent
    print("\nLoading session into new agent...")
    loaded_agent = Agent.load_session(
        session_path,
        provider=provider,
        tools=[get_weather],
    )

    # Continue the conversation
    print("\nContinuing conversation...")
    result2 = await loaded_agent.run("How about Paris?")
    print(f"Response: {result2.content[:200]}...")

    print(f"\nTotal messages in conversation: {len(loaded_agent.conversation)}")


async def main():
    print("=" * 60)
    print("ANTHROPIC (CLAUDE) PROVIDER EXAMPLES")
    print("=" * 60)

    if not ANTHROPIC_AVAILABLE:
        print("\n⚠️  Anthropic package not installed!")
        print("Install with: pip install llm-client[anthropic]")
        print("\nThis example requires the anthropic package to run.")
        return

    try:
        # Run examples
        await example_basic_completion()
        await example_streaming()
        await example_tool_calling()
        await example_streaming_agent()
        await example_session_persistence()

        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure ANTHROPIC_API_KEY is set correctly.")
        raise


if __name__ == "__main__":
    asyncio.run(main())
