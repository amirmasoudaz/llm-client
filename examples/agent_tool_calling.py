#!/usr/bin/env python3
"""
Example: Agent Tool Calling
Demonstrates the new BaseAgent class engaging in a tool-calling loop.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import OpenAIClient
from llm_client.agent import BaseAgent

# --- Define Tools ---

async def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get current weather for a location."""
    print(f"  🔧 Tool called: get_weather({location}, {unit})")
    # Mock data
    return {"location": location, "temperature": 22, "unit": unit, "condition": "Sunny"}

async def get_time(location: str) -> dict:
    """Get current time for a location."""
    print(f"  🔧 Tool called: get_time({location})")
    return {"location": location, "time": "14:30"}

# Tool definitions schema (OpenAI format)
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                   "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

# Map names to functions
tool_map = {
    "get_weather": get_weather,
    "get_time": get_time,
}

async def main():
    print("Initializing Client...")
    # Use a cheap model for testing
    client = OpenAIClient(model="gpt-5-nano") 
    
    agent = BaseAgent(
        client=client,
        system_prompt="You are a helpful assistant with access to weather and time tools.",
        tools=tools_schema,
        tool_functions=tool_map
    )
    
    query = "What is the weather and time in Tokyo right now?"
    print(f"\n👤 User: {query}")
    
    print("\n🤖 Agent running...")
    final_answer = await agent.run(query)
    
    print(f"\n💬 Final Answer: {final_answer}")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
