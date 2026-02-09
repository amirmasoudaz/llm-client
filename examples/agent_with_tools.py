#!/usr/bin/env python3
"""
Example: Agent with Tool Calling

Demonstrates:
1. Defining tools with the @tool decorator
2. Creating an agent with multiple tools
3. Running multi-turn conversations with automatic tool execution
4. Streaming agent responses
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    Agent,
    OpenAIProvider,
    StreamEventType,
    sync_tool,
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
async def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for information.

    Args:
        query: Search query string
        max_results: Maximum number of results to return
    """
    # Simulated search results
    return f"""Search results for "{query}":
1. {query} - Wikipedia overview
2. Latest news about {query}
3. {query} official documentation

(Showing {max_results} results)"""


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


# You can also use sync_tool for synchronous functions
@sync_tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def main():
    print("=" * 60)
    print("AGENT WITH TOOLS EXAMPLE")
    print("=" * 60)

    # Create provider
    provider = OpenAIProvider(model="gpt-5-nano")

    # Create agent with tools
    agent = Agent(
        provider=provider,
        tools=[get_weather, search_web, calculate, get_current_time],
        system_message=(
            "You are a helpful assistant with access to tools. "
            "Use tools when appropriate to answer questions accurately."
        ),
        max_turns=5,
    )

    # === Example 1: Single tool call ===
    print("\n" + "=" * 40)
    print("Example 1: Weather Query")
    print("=" * 40)

    result = await agent.run("What's the weather like in Tokyo?")

    print(f"\nFinal answer: {result.content}")
    print(f"Turns taken: {result.num_turns}")
    print(f"Tool calls: {[tc.name for tc in result.all_tool_calls]}")

    # Reset for next example
    agent.reset()

    # === Example 2: Multiple tool calls ===
    print("\n" + "=" * 40)
    print("Example 2: Multi-tool Query")
    print("=" * 40)

    result = await agent.run("Compare the weather in New York and London, and tell me what time it is.")

    print(f"\nFinal answer: {result.content}")
    print(f"Turns taken: {result.num_turns}")
    print(f"Tool calls: {[tc.name for tc in result.all_tool_calls]}")

    # Reset for next example
    agent.reset()

    # === Example 3: Streaming with tools ===
    print("\n" + "=" * 40)
    print("Example 3: Streaming Response")
    print("=" * 40)

    print("\nStreaming response:")
    print("-" * 40)

    async for event in agent.stream("Calculate 15 * 23 + 7"):
        if event.type == StreamEventType.TOKEN:
            print(event.data, end="", flush=True)
        elif event.type == StreamEventType.META:
            if event.data.get("event") == "tool_result":
                print(f"\n[Tool: {event.data['tool_name']}] {event.data['content']}")
        elif event.type == StreamEventType.DONE:
            print("\n" + "-" * 40)
            agent_result = event.data
            print(f"Status: {agent_result.status}")

    # === Example 4: Multi-turn conversation ===
    print("\n" + "=" * 40)
    print("Example 4: Multi-turn Conversation")
    print("=" * 40)

    agent.reset()

    # First message
    result1 = await agent.run("Search for Python programming")
    print("\nUser: Search for Python programming")
    print(f"Assistant: {result1.content}")

    # Follow-up (conversation continues)
    result2 = await agent.run("Now search for JavaScript frameworks")
    print("\nUser: Now search for JavaScript frameworks")
    print(f"Assistant: {result2.content}")

    # Check conversation history
    print(f"\nTotal messages in conversation: {len(agent.conversation)}")

    # === Cleanup ===
    await provider.close()
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
