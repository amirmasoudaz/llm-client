# src/agents/orchestrator/engine.py
"""Dana Orchestrator Engine - Hybrid approach with token-efficient routing."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from llm_client import OpenAIClient, GPT5, GPT5Mini

from src.services.db import DatabaseService
from src.services.jobs import JobService, JobContext
from src.services.events import EventService, EventType, SSEChannel
from src.agents.orchestrator.context import ContextBuilder
from src.agents.orchestrator.tools import ToolRegistry, get_registry, ToolResult
from src.agents.orchestrator.router import route_request, ProcessingMode, RouteDecision
from src.agents.orchestrator.prompts import (
    DANA_SYSTEM_PROMPT,
    REACT_REASONING_PROMPT,
    TOOL_RESULT_SYNTHESIS_PROMPT,
)
from src.schemas.context import OrchestrationContext


class DanaOrchestrator:
    """
    Hybrid Dana orchestrator with token-efficient routing.
    
    Processing Modes:
    1. DIRECT: Single tool call, no reasoning (most efficient)
    2. GUIDED: Predefined tool sequence with minimal synthesis
    3. AGENTIC: Full ReAct with CoT for complex tasks
    
    This hybrid approach optimizes token usage while maintaining
    full agentic capabilities for complex requests.
    """
    
    MAX_TOOL_ITERATIONS = 5
    
    def __init__(
        self,
        db: DatabaseService,
        job_service: JobService,
        event_service: EventService,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.db = db
        self.job_service = job_service
        self.event_service = event_service
        self.tool_registry = tool_registry or get_registry()
        self.context_builder = ContextBuilder(db)
        
        # Tiered LLM clients for token efficiency
        self.llm_fast = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="dana_fast",
        )
        self.llm_smart = OpenAIClient(
            GPT5,
            cache_backend="pg_redis",
            cache_collection="dana_smart",
        )
    
    async def process_stream(
        self,
        thread_id: int,
        message: str,
        document_ids: Optional[List[int]] = None,
        contexts: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a user message with hybrid routing for token efficiency.
        
        Routes to:
        - DIRECT mode: Single tool call (minimal tokens)
        - GUIDED mode: Tool sequence (moderate tokens)
        - AGENTIC mode: Full ReAct (full tokens)
        """
        channel = self.event_service.get_channel(thread_id)
        
        try:
            yield self._format_sse(EventType.RESPONSE_START, {
                "thread_id": thread_id,
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            yield self._format_sse(EventType.PROGRESS_UPDATE, {
                "percent": 5,
                "message": "Loading context...",
            })
            
            # Build lightweight context first (for routing)
            context = await self.context_builder.build(
                thread_id=thread_id,
                current_message=message,
                document_ids=document_ids,
                additional_contexts=contexts,
            )
            
            # Save user message
            await self.db.create_message(
                thread_id=thread_id,
                role="user",
                content={"text": message},
            )
            
            # Route request (zero LLM tokens)
            route = route_request(message, context)
            
            yield self._format_sse(EventType.PROGRESS_UPDATE, {
                "percent": 10,
                "message": f"Processing ({route.mode.value} mode)...",
            })
            
            # Create job
            job_type = f"chat_{route.mode.value}"
            job_id = await self.job_service.create_job(
                student_id=context.user.student_id,
                job_type=job_type,
                thread_id=thread_id,
                target_type="chat_thread",
                target_id=thread_id,
                model="gpt-4o-mini" if route.model_tier == "fast" else "gpt-4o",
            )
            
            # Process based on mode
            response_text = ""
            tool_results = []
            
            if route.mode == ProcessingMode.DIRECT:
                # Direct tool execution - minimal tokens
                async for event in self._process_direct(context, route, job_id):
                    if event["type"] == "token":
                        response_text += event["data"]
                        yield self._format_sse(EventType.RESPONSE_TOKEN, event["data"])
                    elif event["type"] == "tool_end":
                        tool_results.append(event["data"])
                        yield self._format_sse(EventType.TOOL_END, event["data"])
                    else:
                        yield self._format_sse(EventType(event["type"]) if event["type"] in EventType.__members__.values() else event["type"], event["data"])
            
            elif route.mode == ProcessingMode.GUIDED:
                # Guided tool sequence - moderate tokens
                async for event in self._process_guided(context, route, job_id):
                    if event["type"] == "token":
                        response_text += event["data"]
                        yield self._format_sse(EventType.RESPONSE_TOKEN, event["data"])
                    elif event["type"] == "tool_end":
                        tool_results.append(event["data"])
                        yield self._format_sse(EventType.TOOL_END, event["data"])
                    else:
                        yield self._format_sse(EventType.PROGRESS_UPDATE, event["data"]) if event["type"] == "progress" else None
            
            else:
                # Full agentic mode - ReAct with CoT
                async for event in self._process_agentic(context, route, job_id):
                    if event["type"] == "token":
                        response_text += event["data"]
                        yield self._format_sse(EventType.RESPONSE_TOKEN, event["data"])
                    elif event["type"] == "progress":
                        yield self._format_sse(EventType.PROGRESS_UPDATE, event["data"])
                    elif event["type"] == "tool_start":
                        yield self._format_sse(EventType.TOOL_START, event["data"])
                    elif event["type"] == "tool_end":
                        tool_results.append(event["data"])
                        yield self._format_sse(EventType.TOOL_END, event["data"])
                    elif event["type"] == "meta":
                        yield self._format_sse(EventType.META_ACTION, event["data"])
            
            # Save response
            await self.db.create_message(
                thread_id=thread_id,
                role="assistant",
                content={
                    "text": response_text,
                    "tool_results": tool_results,
                    "processing_mode": route.mode.value,
                },
            )
            
            await self.job_service.complete_job(
                job_id=job_id,
                result={"response": response_text, "mode": route.mode.value},
                usage={},
            )
            
            await self.db.save_thread_suggestions(thread_id, [])
            
            yield self._format_sse(EventType.RESPONSE_END, {
                "thread_id": thread_id,
                "job_id": job_id,
                "mode": route.mode.value,
            })
            
        except Exception as e:
            yield self._format_sse(EventType.ERROR, {
                "error": str(e),
                "code": type(e).__name__,
            })
        finally:
            await self.event_service.close_channel(thread_id)
    
    # =========================================================================
    # DIRECT Mode - Single tool, minimal tokens
    # =========================================================================
    
    async def _process_direct(
        self,
        context: OrchestrationContext,
        route: RouteDecision,
        job_id: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Direct tool execution with minimal LLM usage.
        
        Flow:
        1. Execute tool directly
        2. Format result with lightweight synthesis
        """
        tool_name = route.tools[0]
        
        yield {"type": "progress", "data": {"percent": 30, "message": f"Executing {tool_name}..."}}
        
        # Execute tool
        try:
            result = await self.tool_registry.execute(
                name=tool_name,
                arguments={},  # Uses context defaults
                context=context,
            )
            
            yield {
                "type": "tool_end",
                "data": {"name": tool_name, "success": True, "result": result if isinstance(result, dict) else {"data": result}},
            }
            
            yield {"type": "progress", "data": {"percent": 70, "message": "Formatting response..."}}
            
            # Lightweight synthesis (template-based or minimal LLM)
            response = await self._synthesize_direct(tool_name, result, context)
            
            async for token in self._stream_response(response):
                yield {"type": "token", "data": token}
                
        except Exception as e:
            yield {"type": "tool_end", "data": {"name": tool_name, "success": False, "error": str(e)}}
            yield {"type": "token", "data": f"I encountered an issue: {str(e)}. Would you like me to try a different approach?"}
    
    async def _synthesize_direct(
        self,
        tool_name: str,
        result: Any,
        context: OrchestrationContext,
    ) -> str:
        """
        Synthesize response for direct tool execution.
        
        Uses templates where possible, lightweight LLM for complex results.
        """
        # Template-based responses for common tools
        if isinstance(result, dict):
            if tool_name == "email_generate" and result.get("success"):
                email = result.get("content", {})
                return f"""I've drafted an email for Professor {context.professor.full_name}:

**Subject:** {email.get('subject', 'N/A')}

{email.get('greeting', '')}

{email.get('body', '')}

{email.get('closing', '')}
{email.get('signature_name', '')}

Would you like me to review it or make any changes?"""
            
            elif tool_name == "email_review" and result.get("success"):
                score = result.get("score", 0)
                level = result.get("readiness_level", "unknown")
                return f"""**Email Review Complete**

**Overall Score:** {score}/10 ({level})

**Key Findings:**
{self._format_review_dimensions(result.get('dimensions', {}))}

{self._format_suggestions(result.get('suggestions', []))}

Would you like me to help improve any of these areas?"""
            
            elif tool_name == "alignment_evaluate" and result.get("success"):
                score = result.get("score", 0)
                label = result.get("label", "UNKNOWN")
                return f"""**Alignment Evaluation**

**Score:** {score}/10 ({label} alignment)

**Match Analysis:**
{chr(10).join('• ' + r for r in result.get('reasons', [])[:5])}

{self._get_alignment_recommendation(score)}"""
            
            elif tool_name.startswith("get_") and result.get("success"):
                # Context retrieval - summarize key info
                ctx_data = result.get("context", {})
                return self._summarize_context(tool_name, ctx_data)
        
        # Fallback: Lightweight LLM synthesis
        return await self._llm_synthesize_minimal(tool_name, result, context)
    
    def _format_review_dimensions(self, dimensions: Dict[str, Any]) -> str:
        """Format review dimensions as bullet points."""
        if not dimensions:
            return "No detailed dimensions available."
        
        lines = []
        for name, data in dimensions.items():
            if isinstance(data, dict):
                score = data.get("score", "N/A")
                lines.append(f"• **{name.replace('_', ' ').title()}:** {score}/10")
        return "\n".join(lines[:7])  # Top 7 dimensions
    
    def _format_suggestions(self, suggestions: List[str]) -> str:
        """Format suggestions list."""
        if not suggestions:
            return ""
        return "**Suggestions:**\n" + "\n".join(f"• {s}" for s in suggestions[:5])
    
    def _get_alignment_recommendation(self, score: float) -> str:
        """Get recommendation based on alignment score."""
        if score >= 7:
            return "This is a strong match! I recommend proceeding with outreach."
        elif score >= 5:
            return "Moderate alignment. Consider emphasizing your strongest overlapping interests."
        else:
            return "Limited alignment. You may want to explore other professors or highlight transferable skills."
    
    def _summarize_context(self, tool_name: str, ctx: Dict[str, Any]) -> str:
        """Summarize context data concisely."""
        if tool_name == "get_professor_context":
            return f"""**Professor Profile: {ctx.get('full_name', 'Unknown')}**

• **Position:** {ctx.get('occupation', 'Professor')} at {ctx.get('institution_name', 'Unknown')}
• **Department:** {ctx.get('department', 'N/A')}
• **Research Areas:** {', '.join(ctx.get('research_areas', [])[:5])}
• **Email:** {ctx.get('email_address', 'N/A')}"""
        
        elif tool_name == "get_user_context":
            return f"""**Your Profile Summary**

• **Name:** {ctx.get('first_name', '')} {ctx.get('last_name', '')}
• **Research Interests:** {', '.join(ctx.get('research_interests', [])[:5])}
• **Skills:** {', '.join(ctx.get('skills', [])[:5])}"""
        
        return f"Context loaded successfully with {len(ctx)} fields."
    
    async def _llm_synthesize_minimal(
        self,
        tool_name: str,
        result: Any,
        context: OrchestrationContext,
    ) -> str:
        """Minimal LLM call for synthesis when templates don't fit."""
        prompt = f"""Summarize this tool result in 2-3 sentences for the user.
Tool: {tool_name}
Result: {json.dumps(result, default=str)[:1000]}
Be helpful and offer next steps."""
        
        response = await self.llm_fast.get_response(
            messages=[{"role": "user", "content": prompt}],
            response_format="text",
            max_tokens=200,
        )
        return response.get("output", "Task completed. How can I help you next?")
    
    # =========================================================================
    # GUIDED Mode - Tool sequence, moderate tokens
    # =========================================================================
    
    async def _process_guided(
        self,
        context: OrchestrationContext,
        route: RouteDecision,
        job_id: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a predefined tool sequence with minimal reasoning.
        """
        all_results = []
        total_tools = len(route.tools)
        
        for i, tool_name in enumerate(route.tools):
            progress = 20 + int((i / total_tools) * 60)
            yield {"type": "progress", "data": {"percent": progress, "message": f"Step {i+1}/{total_tools}: {tool_name}..."}}
            
            try:
                result = await self.tool_registry.execute(
                    name=tool_name,
                    arguments={},
                    context=context,
                )
                all_results.append({"tool": tool_name, "result": result})
                yield {"type": "tool_end", "data": {"name": tool_name, "success": True}}
                
            except Exception as e:
                all_results.append({"tool": tool_name, "error": str(e)})
                yield {"type": "tool_end", "data": {"name": tool_name, "success": False, "error": str(e)}}
        
        yield {"type": "progress", "data": {"percent": 85, "message": "Synthesizing results..."}}
        
        # Synthesize all results
        response = await self._synthesize_guided(all_results, context)
        async for token in self._stream_response(response):
            yield {"type": "token", "data": token}
    
    async def _synthesize_guided(
        self,
        results: List[Dict[str, Any]],
        context: OrchestrationContext,
    ) -> str:
        """Synthesize multiple tool results into coherent response."""
        # Build concise summary for LLM
        results_summary = []
        for r in results:
            if "error" in r:
                results_summary.append(f"{r['tool']}: ERROR - {r['error']}")
            else:
                results_summary.append(f"{r['tool']}: SUCCESS")
        
        prompt = f"""Summarize these completed tasks for the user in a helpful way.
Tasks completed: {', '.join(results_summary)}
User: {context.user.first_name}
Professor: {context.professor.full_name}
Be concise (3-5 sentences) and offer relevant next steps."""
        
        response = await self.llm_fast.get_response(
            messages=[{"role": "user", "content": prompt}],
            response_format="text",
            max_tokens=300,
        )
        return response.get("output", "All tasks completed. What would you like to do next?")
    
    # =========================================================================
    # AGENTIC Mode - Full ReAct with CoT
    # =========================================================================
    
    async def _process_agentic(
        self,
        context: OrchestrationContext,
        route: RouteDecision,
        job_id: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Full ReAct loop for complex requests."""
        # Select LLM based on tier
        llm = self.llm_smart if route.model_tier == "smart" else self.llm_fast
        
        messages = self._build_initial_messages(context)
        iteration = 0
        
        while iteration < self.MAX_TOOL_ITERATIONS:
            iteration += 1
            
            yield {
                "type": "progress",
                "data": {"percent": 20 + (iteration * 12), "message": f"Reasoning... (step {iteration})"},
            }
            
            response = await llm.get_response(
                messages=messages,
                response_format="text",
                tools=self.tool_registry.get_openai_tools(),
                tool_choice="auto",
            )
            
            output = response.get("output", "")
            tool_calls = self._extract_tool_calls(response)
            
            if not tool_calls:
                async for token in self._stream_response(output):
                    yield {"type": "token", "data": token}
                break
            
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                yield {"type": "tool_start", "data": {"name": tool_name, "arguments": tool_args}}
                
                try:
                    result = await self.tool_registry.execute(
                        name=tool_name,
                        arguments=tool_args,
                        context=context,
                    )
                    
                    yield {
                        "type": "tool_end",
                        "data": {"name": tool_name, "success": True, "result": result if isinstance(result, dict) else {"data": result}},
                    }
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", tool_name),
                        "content": json.dumps(result if isinstance(result, dict) else {"result": result}, default=str)[:2000],
                    })
                    
                except Exception as e:
                    yield {"type": "tool_end", "data": {"name": tool_name, "success": False, "error": str(e)}}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", tool_name),
                        "content": json.dumps({"error": str(e)}),
                    })
        
        if iteration >= self.MAX_TOOL_ITERATIONS:
            final_response = await self._generate_final_response(messages, context)
            async for token in self._stream_response(final_response):
                yield {"type": "token", "data": token}
    
    # Legacy method for backwards compatibility
    async def _react_loop(
        self,
        context: OrchestrationContext,
        job_id: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Legacy ReAct loop - now uses _process_agentic internally."""
        route = RouteDecision(
            mode=ProcessingMode.AGENTIC,
            tools=[],
            confidence=1.0,
            reasoning="Legacy call",
            model_tier="smart",
        )
        async for event in self._process_agentic(context, route, job_id):
            yield event
    
    def _build_initial_messages(self, context: OrchestrationContext) -> List[Dict[str, str]]:
        """Build initial message list for the LLM."""
        messages = [
            {"role": "system", "content": DANA_SYSTEM_PROMPT},
        ]
        
        # Add context as system message
        context_prompt = f"""## Current Context

{context.to_prompt_context()}

## Conversation History
"""
        
        if context.conversation and context.conversation.messages:
            for msg in context.conversation.messages[-10:]:  # Last 10 messages
                context_prompt += f"\n{msg.role.upper()}: {msg.content}"
        
        messages.append({"role": "system", "content": context_prompt})
        
        # Add current user message
        messages.append({"role": "user", "content": context.current_message})
        
        return messages
    
    def _extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        # This is simplified - in production would properly parse OpenAI function calls
        # The response format depends on whether using function calling or responses API
        
        tool_calls = []
        
        # Check for function_call in response
        if "function_call" in response:
            fc = response["function_call"]
            tool_calls.append({
                "id": fc.get("id", "call_0"),
                "name": fc["name"],
                "arguments": json.loads(fc.get("arguments", "{}")),
            })
        
        # Check for tool_calls array
        if "tool_calls" in response:
            for tc in response["tool_calls"]:
                tool_calls.append({
                    "id": tc.get("id", "call_0"),
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"].get("arguments", "{}")),
                })
        
        return tool_calls
    
    async def _stream_response(self, text: str) -> AsyncGenerator[str, None]:
        """Stream a response token by token (simulated for non-streaming)."""
        # In production, this would use actual streaming from the LLM
        # For now, yield chunks
        chunk_size = 20
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Small delay for realistic streaming
    
    async def _generate_final_response(
        self,
        messages: List[Dict[str, str]],
        context: OrchestrationContext,
    ) -> str:
        """Generate final response after tool execution."""
        # Add synthesis prompt
        messages.append({
            "role": "system",
            "content": "Based on the tool results and conversation, provide a helpful response to the user.",
        })
        
        response = await self.llm.get_response(
            messages=messages,
            response_format="text",
        )
        
        return response.get("output", "I apologize, but I encountered an issue generating a response.")
    
    @staticmethod
    def _format_sse(event_type: EventType, data: Any) -> str:
        """Format data as SSE event."""
        event_str = event_type.value if isinstance(event_type, EventType) else event_type
        data_str = json.dumps(data) if not isinstance(data, str) else data
        return f"event: {event_str}\ndata: {data_str}\n\n"
    
    async def process_non_streaming(
        self,
        thread_id: int,
        message: str,
        document_ids: Optional[List[int]] = None,
        contexts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a message without streaming.
        
        Useful for internal calls or when streaming isn't needed.
        """
        context = await self.context_builder.build(
            thread_id=thread_id,
            current_message=message,
            document_ids=document_ids,
            additional_contexts=contexts,
        )
        
        # Save user message
        await self.db.create_message(
            thread_id=thread_id,
            role="user",
            content={"text": message},
        )
        
        # Collect all events
        response_text = ""
        tool_results = []
        
        async for event in self._react_loop(context, job_id=0):
            if event["type"] == "token":
                response_text += event["data"]
            elif event["type"] == "tool_end":
                tool_results.append(event["data"])
        
        # Save assistant message
        await self.db.create_message(
            thread_id=thread_id,
            role="assistant",
            content={
                "text": response_text,
                "tool_results": tool_results,
            },
        )
        
        return {
            "response": response_text,
            "tool_results": tool_results,
        }

