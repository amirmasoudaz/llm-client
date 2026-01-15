#!/usr/bin/env python3
"""
Example: Tool Registry Advanced Usage

Demonstrates:
1. Creating tools with @tool decorator (auto schema inference)
2. Creating tools manually with explicit schema
3. Using tool_from_function for existing functions
4. Managing tools in ToolRegistry
5. Executing tools and handling results
6. Complex parameter types and defaults
"""
import asyncio
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_client import (
    Tool,
    ToolRegistry,
    ToolResult,
    tool,
    sync_tool,
    tool_from_function,
    OpenAIProvider,
)


# === Example Tools with @tool Decorator ===

@tool
async def search_database(
    query: str,
    table: str = "users",
    limit: int = 10,
    include_deleted: bool = False,
) -> str:
    """Search the database for matching records.
    
    Performs a full-text search across the specified table
    and returns matching records.
    
    Args:
        query: Search query string
        table: Database table to search (users, products, orders)
        limit: Maximum number of results to return
        include_deleted: Whether to include soft-deleted records
    
    Returns:
        JSON string containing matching records
    """
    # Simulated database search
    results = [
        {"id": i, "table": table, "match": query, "deleted": False}
        for i in range(min(limit, 3))
    ]
    return json.dumps(results)


@tool
async def send_notification(
    user_id: int,
    message: str,
    channel: str = "email",
    priority: str = "normal",
) -> str:
    """Send a notification to a user.
    
    Args:
        user_id: The ID of the user to notify
        message: The notification message content
        channel: Notification channel (email, sms, push)
        priority: Message priority (low, normal, high, urgent)
    
    Returns:
        Confirmation message with notification ID
    """
    return f"Notification sent to user {user_id} via {channel} (priority: {priority})"


@sync_tool
def get_system_info() -> str:
    """Get current system information.
    
    Returns system status including uptime and resource usage.
    
    Returns:
        JSON string with system information
    """
    import platform
    return json.dumps({
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "processor": platform.processor()[:30] + "...",
    })


# === Manual Tool Creation ===

def create_manual_tool():
    """Create a tool with explicit schema definition."""
    
    async def handler(city: str, units: str = "celsius") -> str:
        temps = {"new york": 20, "london": 15, "tokyo": 25}
        temp = temps.get(city.lower(), 22)
        if units == "fahrenheit":
            temp = int(temp * 9/5 + 32)
        return f"{temp}°{'F' if units == 'fahrenheit' else 'C'}"
    
    return Tool(
        name="get_weather",
        description="Get the current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                    "default": "celsius"
                }
            },
            "required": ["city"]
        },
        handler=handler
    )


async def main():
    print("=" * 60)
    print("TOOL REGISTRY EXAMPLE")
    print("=" * 60)
    
    # === Example 1: Auto Schema Inference ===
    print("\n" + "=" * 40)
    print("Example 1: Auto Schema Inference")
    print("=" * 40)
    
    print("\n@tool decorator automatically infers JSON schema from:")
    print("  - Function signature (parameter types, defaults)")
    print("  - Docstring (description and arg descriptions)")
    
    print(f"\nsearch_database tool schema:")
    print(f"  Name: {search_database.name}")
    print(f"  Description: {search_database.description}")
    print(f"  Parameters: {json.dumps(search_database.parameters, indent=4)}")
    
    # === Example 2: Tool Registry ===
    print("\n" + "=" * 40)
    print("Example 2: Tool Registry")
    print("=" * 40)
    
    # Create registry with tools
    registry = ToolRegistry([
        search_database,
        send_notification,
        get_system_info,
    ])
    
    print(f"\nRegistry created with {len(registry.tools)} tools:")
    for name in registry.names:
        tool_obj = registry.get(name)
        print(f"  - {name}: {tool_obj.description[:50]}...")
    
    # Add more tools dynamically
    weather_tool = create_manual_tool()
    registry.register(weather_tool)
    
    print(f"\nAfter adding weather tool: {len(registry.tools)} tools")
    
    # === Example 3: Executing Tools ===
    print("\n" + "=" * 40)
    print("Example 3: Tool Execution")
    print("=" * 40)
    
    # Execute with JSON arguments (as model would provide)
    print("\nExecuting search_database:")
    result = await registry.execute(
        "search_database",
        '{"query": "john", "table": "users", "limit": 5}'
    )
    print(f"  Success: {result.success}")
    print(f"  Result: {result.content}")
    
    print("\nExecuting get_system_info (no args):")
    result = await registry.execute("get_system_info", "{}")
    print(f"  Success: {result.success}")
    print(f"  Result: {result.content}")
    
    print("\nExecuting get_weather:")
    result = await registry.execute(
        "get_weather",
        '{"city": "Tokyo", "units": "fahrenheit"}'
    )
    print(f"  Success: {result.success}")
    print(f"  Result: {result.content}")
    
    # === Example 4: Error Handling ===
    print("\n" + "=" * 40)
    print("Example 4: Error Handling")
    print("=" * 40)
    
    # Unknown tool
    result = await registry.execute("unknown_tool", "{}")
    print(f"\nUnknown tool:")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    
    # Invalid JSON arguments
    result = await registry.execute("search_database", "not valid json")
    print(f"\nInvalid JSON:")
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error}")
    
    # === Example 5: Tool Result Formatting ===
    print("\n" + "=" * 40)
    print("Example 5: Result Formatting")
    print("=" * 40)
    
    # Success result
    result = await registry.execute("send_notification", json.dumps({
        "user_id": 123,
        "message": "Hello!",
        "channel": "push",
    }))
    
    print(f"\nToolResult properties:")
    print(f"  .content: {result.content}")
    print(f"  .success: {result.success}")
    print(f"  .error: {result.error}")
    print(f"  .to_string(): {result.to_string()[:80]}...")
    
    # Error result for comparison
    error_result = ToolResult.error_result("Something went wrong")
    print(f"\nError result:")
    print(f"  .content: {error_result.content}")
    print(f"  .success: {error_result.success}")
    print(f"  .error: {error_result.error}")
    print(f"  .to_string(): {error_result.to_string()}")
    
    # === Example 6: tool_from_function ===
    print("\n" + "=" * 40)
    print("Example 6: tool_from_function")
    print("=" * 40)
    
    print("\ntool_from_function works with both sync and async functions:")
    
    # Synchronous function
    def calculate_area(length: float, width: float) -> float:
        """Calculate the area of a rectangle.
        
        Args:
            length: Length of the rectangle
            width: Width of the rectangle
        """
        return length * width
    
    # Async function
    async def fetch_url(url: str, timeout: int = 30) -> str:
        """Fetch content from a URL.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
        """
        return f"Content from {url} (timeout: {timeout}s)"
    
    area_tool = tool_from_function(calculate_area)
    fetch_tool = tool_from_function(fetch_url)
    
    print(f"\nSync function -> tool:")
    print(f"  Name: {area_tool.name}")
    print(f"  Description: {area_tool.description}")
    
    print(f"\nAsync function -> tool:")
    print(f"  Name: {fetch_tool.name}")
    print(f"  Description: {fetch_tool.description}")
    
    # Execute both (tools execute with kwargs, not JSON strings)
    result1 = await area_tool.execute(length=5.0, width=3.0)
    print(f"\n  area_tool result: {result1.content}")
    
    result2 = await fetch_tool.execute(url="https://example.com", timeout=10)
    print(f"  fetch_tool result: {result2.content}")
    
    # === Example 8: Integration with Provider ===
    print("\n" + "=" * 40)
    print("Example 8: Integration with Provider")
    print("=" * 40)
    
    # Initialize provider
    provider = OpenAIProvider(model="gpt-5-nano")
    
    query = "Search for a user named 'Alice' in the 'admin' table and send her a 'Hello' email."
    print(f"\nUser Query: {query}")
    print("\nRequesting tool calls from model...")
    
    # Call model with registry tools
    result = await provider.complete(
        query,
        tools=registry.tools,
        tool_choice="auto"
    )
    
    if result.ok and result.has_tool_calls:
        print(f"\nModel requested {len(result.tool_calls)} tool calls:")
        
        for i, tc in enumerate(result.tool_calls):
            print(f"\n[{i+1}] Tool: {tc.name}")
            print(f"    Arguments: {tc.arguments}")
            
            # Execute via registry
            print(f"    Executing...")
            exec_result = await registry.execute(tc.name, tc.arguments)
            
            if exec_result.success:
                print(f"    Result: {exec_result.content}")
            else:
                print(f"    Error: {exec_result.error}")
    else:
        print("\nNo tool calls requested or error occurred.")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Response: {result.content}")
            
    await provider.close()
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
