import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .types import LLMEvent, LLMResult
from .client import OpenAIClient


ToolFunc = Callable[..., Awaitable[Any] | Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: ToolFunc

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def to_openai_tools(self) -> list[dict]:
        return [tool.to_openai() for tool in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    async def execute(self, name: str, arguments: Any) -> Any:
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool {name!r} is not registered.")

        parsed_args = arguments
        if isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = arguments

        if isinstance(parsed_args, dict):
            result = tool.func(**parsed_args)
        else:
            result = tool.func(parsed_args)

        if asyncio.iscoroutine(result):
            return await result
        return result


class AgentRunner:
    def __init__(
        self,
        client: OpenAIClient,
        tools: list[Tool],
        *,
        max_steps: int = 8,
    ) -> None:
        self.client = client
        self.registry = ToolRegistry(tools)
        self.max_steps = max_steps

    async def run(self, messages: list[dict], **kwargs) -> LLMResult:
        tools_param = self.registry.to_openai_tools()
        for _ in range(self.max_steps):
            result = await self.client.invoke(messages=messages, tools=tools_param, **kwargs)
            tool_calls = result.tool_calls or []
            if not tool_calls:
                return result

            assistant_message = {"role": "assistant", "tool_calls": tool_calls}
            if result.output:
                assistant_message["content"] = result.output
            messages.append(assistant_message)

            for call in tool_calls:
                fn = call.get("function") or {}
                name = fn.get("name")
                arguments = fn.get("arguments", "")
                try:
                    output = await self.registry.execute(name, arguments)
                except Exception as exc:
                    output = {"error": str(exc)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": self._tool_output_to_str(output),
                })

        raise RuntimeError("Max agent steps exceeded without producing a final answer.")

    def stream(self, messages: list[dict], **kwargs):
        tools_param = self.registry.to_openai_tools()

        async def generator():
            for step in range(self.max_steps):
                yield LLMEvent("agent_step", {"step": step + 1})
                result_dict = None
                async for event in self.client.stream(messages=messages, tools=tools_param, **kwargs):
                    yield event
                    if event.type == "done":
                        result_dict = event.data.get("result", {})

                tool_calls = (result_dict or {}).get("tool_calls") or []
                if not tool_calls:
                    yield LLMEvent("agent_done", {"result": result_dict})
                    return

                assistant_message = {"role": "assistant", "tool_calls": tool_calls}
                if result_dict and result_dict.get("output"):
                    assistant_message["content"] = result_dict.get("output")
                messages.append(assistant_message)

                for call in tool_calls:
                    fn = call.get("function") or {}
                    name = fn.get("name")
                    arguments = fn.get("arguments", "")
                    yield LLMEvent("tool_start", {"name": name, "id": call.get("id")})
                    try:
                        output = await self.registry.execute(name, arguments)
                        yield LLMEvent("tool_end", {"name": name, "id": call.get("id"), "output": output})
                    except Exception as exc:
                        output = {"error": str(exc)}
                        yield LLMEvent("tool_error", {"name": name, "id": call.get("id"), "error": str(exc)})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": self._tool_output_to_str(output),
                    })

            yield LLMEvent("agent_error", {"error": "Max agent steps exceeded."})

        return generator()

    @staticmethod
    def _tool_output_to_str(output: Any) -> str:
        if isinstance(output, str):
            return output
        return json.dumps(output, ensure_ascii=True, default=str)


__all__ = ["Tool", "ToolRegistry", "AgentRunner"]
