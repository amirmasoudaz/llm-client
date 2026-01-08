import json
from typing import List, Optional, Callable, Any, Union, Dict
from .client import OpenAIClient
from .types import RequestParams
from .streams import StreamResponse

class BaseAgent:
    """
    A simple agent wrapper that manages conversation history and tool execution.
    """
    def __init__(
        self,
        client: OpenAIClient,
        system_prompt: str = "You are a helpful assistant.",
        tools: Optional[List[Dict]] = None,
        tool_functions: Optional[Dict[str, Callable]] = None,
        model: Optional[str] = None,
    ):
        self.client = client
        self.system_prompt = system_prompt
        self.history: List[Dict] = [{"role": "system", "content": system_prompt}]
        self.tools = tools or []
        self.tool_functions = tool_functions or {}
        self.model = model or client.model.model_name

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    async def run(
        self,
        input_text: str,
        stream: bool = False,
        max_steps: int = 10,
        **kwargs
    ) -> Union[str, StreamResponse]:
        """
        Run the agent loop:
        1. Add user message
        2. Call Model
        3. If tool calls -> execute -> add to history -> repeat
        4. If text -> return
        """
        self.add_message("user", input_text)
        
        current_step = 0
        
        while current_step < max_steps:
            current_step += 1
            
            # Prepare params
            params: RequestParams = {
                "messages": self.history,
                "model": self.model,
                **kwargs
            }
            if self.tools:
                params["tools"] = self.tools
                params["tool_choice"] = "auto"
            
            # If streaming is requested, and this is the "final" response generation (no tool calls expected ideally),
            # we can stream. But for Agents, we often need to peek at the first chunk to see if it's a tool call.
            # However, OpenAI stream yields tool_calls chunks too.
            
            # For simplicity in this v1:
            # We will use the client's new "Raw Stream" mode which returns a StreamResponse.
            # But the Agent logic needs to INTERCEPT the stream to checks for tool calls.
            
            if stream:
                # Agent Streaming is complex. We stick to non-streaming for the *thinking* part 
                # if we want to robustly handle tools in a simple loop.
                # OR we stream everything and if we detect tool calls, we consume the stream entirely, run tools, and continue.
                
                params["stream"] = True
                params["stream_mode"] = "raw"
                
                # Get the stream wrapper
                response: StreamResponse = await self.client.get_response(**params)
                
                # We need to spy on the stream. 
                # We will accumulate the content to update history later.
                accumulated_tool_calls = []
                accumulated_content = ""
                
                # We yield the stream response immediately to the user?
                # If we do that, the user sees tool calls. That might be desired.
                # But we need to know IF we should continue the loop.
                # So we can't just return the generator. We have to consume it, OR wrap it in another generator 
                # that handles the loop.
                
                # For this first iteration, let's implement standard non-streaming Agent loop 
                # and maybe a simple "yield events" generator for streaming.
                pass
            
            # Fallback to non-streaming for the loop logic for now unless we do advanced yielding
            if stream:
                raise NotImplementedError("Streaming Agent is WIP. Use stream=False for now.")

            # Standard Tool Loop
            response = await self.client.get_response(**params)
            output = response.get("output")
            
            # Check for tool calls
            tool_calls = response.get("tool_calls")
            
            # Add assistant message to history
            # We need to construct the message correctly for OpenAI
            assistant_msg = {"role": "assistant", "content": output}
            if tool_calls:
                 assistant_msg["tool_calls"] = tool_calls
            
            self.history.append(assistant_msg)
            
            if tool_calls:
                # Execute tools
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]
                    call_id = tool_call["id"]
                    
                    if func_name in self.tool_functions:
                        try:
                            args = json.loads(args_str)
                            result = await self.tool_functions[func_name](**args)
                            result_str = json.dumps(result) if not isinstance(result, str) else result
                        except Exception as e:
                            result_str = f"Error executing tool: {str(e)}"
                    else:
                        result_str = f"Error: Tool {func_name} not found."
                    
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_str
                    })
                # Loop continues to get next response
                continue
            
            else:
                # No tools, this is the final answer
                return output

        return "Max steps reached."
