# src/agents/orchestrator/router.py
"""
Intent Router - Routes requests to optimal processing path for token efficiency.

Implements a hybrid approach:
1. DIRECT: Simple, single-tool requests → Direct function call (minimal tokens)
2. GUIDED: Clear multi-step tasks → Structured tool sequence (moderate tokens)
3. AGENTIC: Complex/ambiguous requests → Full ReAct reasoning (full tokens)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.schemas.context import OrchestrationContext


class ProcessingMode(str, Enum):
    """Processing mode for request handling."""
    DIRECT = "direct"      # Single tool, no reasoning needed
    GUIDED = "guided"      # Structured sequence, minimal reasoning
    AGENTIC = "agentic"    # Full ReAct with CoT


@dataclass
class RouteDecision:
    """Routing decision for a request."""
    mode: ProcessingMode
    tools: List[str]           # Tools to use (for DIRECT/GUIDED)
    confidence: float          # Confidence in routing decision
    reasoning: Optional[str]   # Brief reasoning (for debugging)
    model_tier: str            # "fast" (mini) or "smart" (4o)


# Intent patterns for direct routing (regex → tool mapping)
DIRECT_PATTERNS: List[Tuple[str, str, float]] = [
    # Email operations
    (r"(?:write|draft|generate|create)\s+(?:an?\s+)?email", "email_generate", 0.9),
    (r"review\s+(?:my\s+)?(?:the\s+)?email", "email_review", 0.95),
    (r"(?:improve|optimize|fix)\s+(?:my\s+)?(?:the\s+)?email", "email_optimize", 0.9),
    
    # Resume/CV operations
    (r"(?:write|draft|generate|create)\s+(?:my\s+)?(?:a\s+)?(?:cv|resume)", "resume_generate", 0.9),
    (r"review\s+(?:my\s+)?(?:the\s+)?(?:cv|resume)", "resume_review", 0.95),
    (r"(?:improve|optimize|fix)\s+(?:my\s+)?(?:the\s+)?(?:cv|resume)", "resume_optimize", 0.9),
    
    # Alignment
    (r"(?:how\s+)?(?:well\s+)?(?:do\s+i\s+)?(?:align|match|fit)", "alignment_evaluate", 0.85),
    (r"(?:check|evaluate)\s+(?:my\s+)?(?:the\s+)?alignment", "alignment_evaluate", 0.9),
    
    # Context retrieval
    (r"(?:tell\s+me\s+about|what\s+(?:do\s+you\s+)?know\s+about)\s+(?:the\s+)?professor", "get_professor_context", 0.85),
    (r"(?:show|get)\s+(?:my\s+)?profile", "get_user_context", 0.9),
    (r"(?:what|show)\s+(?:is|are)\s+(?:the\s+)?request", "get_request_context", 0.85),
]

# Multi-tool sequences for guided mode
GUIDED_SEQUENCES: Dict[str, List[str]] = {
    "full_email_workflow": ["get_user_context", "get_professor_context", "email_generate", "email_review"],
    "email_with_review": ["email_generate", "email_review"],
    "cv_with_review": ["resume_generate", "resume_review"],
    "full_alignment_check": ["get_user_context", "get_professor_context", "alignment_evaluate"],
}

GUIDED_PATTERNS: List[Tuple[str, str, float]] = [
    (r"(?:write|create)\s+(?:and\s+)?review\s+(?:an?\s+)?email", "email_with_review", 0.9),
    (r"(?:write|create)\s+(?:and\s+)?review\s+(?:my\s+)?(?:cv|resume)", "cv_with_review", 0.9),
    (r"(?:full|complete)\s+alignment\s+(?:check|analysis)", "full_alignment_check", 0.85),
]


class IntentRouter:
    """
    Routes incoming requests to the optimal processing path.
    
    Optimizes for token efficiency by:
    1. Pattern matching for obvious intents (zero LLM tokens)
    2. Lightweight classification for ambiguous cases
    3. Full reasoning only when necessary
    """
    
    def __init__(self):
        # Compile patterns for efficiency
        self._direct_patterns = [
            (re.compile(p, re.IGNORECASE), tool, conf)
            for p, tool, conf in DIRECT_PATTERNS
        ]
        self._guided_patterns = [
            (re.compile(p, re.IGNORECASE), seq, conf)
            for p, seq, conf in GUIDED_PATTERNS
        ]
    
    def route(
        self,
        message: str,
        context: OrchestrationContext,
    ) -> RouteDecision:
        """
        Determine the optimal processing path for a message.
        
        Uses a cascading approach:
        1. Check for exact pattern matches (DIRECT mode)
        2. Check for multi-step patterns (GUIDED mode)
        3. Analyze complexity indicators (AGENTIC mode)
        """
        message_lower = message.lower().strip()
        
        # 1. Check for direct patterns
        direct_match = self._match_direct_pattern(message_lower)
        if direct_match:
            tool, confidence = direct_match
            return RouteDecision(
                mode=ProcessingMode.DIRECT,
                tools=[tool],
                confidence=confidence,
                reasoning=f"Direct pattern match for {tool}",
                model_tier="fast",
            )
        
        # 2. Check for guided sequences
        guided_match = self._match_guided_pattern(message_lower)
        if guided_match:
            sequence_name, confidence = guided_match
            tools = GUIDED_SEQUENCES.get(sequence_name, [])
            return RouteDecision(
                mode=ProcessingMode.GUIDED,
                tools=tools,
                confidence=confidence,
                reasoning=f"Guided sequence: {sequence_name}",
                model_tier="fast",
            )
        
        # 3. Analyze complexity for agentic routing
        complexity = self._analyze_complexity(message_lower, context)
        
        if complexity < 0.3:
            # Low complexity but no pattern match - try lightweight classification
            inferred_tool = self._infer_tool(message_lower)
            if inferred_tool:
                return RouteDecision(
                    mode=ProcessingMode.DIRECT,
                    tools=[inferred_tool],
                    confidence=0.7,
                    reasoning=f"Inferred tool: {inferred_tool}",
                    model_tier="fast",
                )
        
        # 4. Default to agentic mode
        model_tier = "smart" if complexity > 0.6 else "fast"
        return RouteDecision(
            mode=ProcessingMode.AGENTIC,
            tools=[],  # Let ReAct decide
            confidence=0.5,
            reasoning=f"Complex request (complexity={complexity:.2f})",
            model_tier=model_tier,
        )
    
    def _match_direct_pattern(self, message: str) -> Optional[Tuple[str, float]]:
        """Match against direct patterns."""
        for pattern, tool, confidence in self._direct_patterns:
            if pattern.search(message):
                return tool, confidence
        return None
    
    def _match_guided_pattern(self, message: str) -> Optional[Tuple[str, float]]:
        """Match against guided sequence patterns."""
        for pattern, sequence, confidence in self._guided_patterns:
            if pattern.search(message):
                return sequence, confidence
        return None
    
    def _analyze_complexity(
        self,
        message: str,
        context: OrchestrationContext,
    ) -> float:
        """
        Analyze request complexity (0.0 = simple, 1.0 = complex).
        
        Factors:
        - Question complexity (multiple questions, conditionals)
        - Conversation context (ongoing vs new topic)
        - Explicit multi-step indicators
        """
        score = 0.0
        
        # Multiple sentences/questions
        sentences = len(re.findall(r'[.!?]+', message))
        if sentences > 2:
            score += 0.2
        
        # Question words suggesting research
        research_words = ["how", "why", "what if", "compare", "analyze", "explain"]
        if any(w in message for w in research_words):
            score += 0.15
        
        # Conditional language
        conditional_words = ["if", "when", "unless", "depending", "based on"]
        if any(w in message for w in conditional_words):
            score += 0.2
        
        # Multi-step indicators
        multi_step = ["then", "after that", "also", "and then", "first", "next"]
        if any(w in message for w in multi_step):
            score += 0.25
        
        # Long message (likely complex)
        word_count = len(message.split())
        if word_count > 50:
            score += 0.2
        elif word_count > 100:
            score += 0.3
        
        # Conversation context
        if context.conversation and context.conversation.total_messages > 5:
            # Ongoing conversation might need context
            score += 0.1
        
        return min(1.0, score)
    
    def _infer_tool(self, message: str) -> Optional[str]:
        """
        Lightweight tool inference for unmatched but simple requests.
        
        Uses keyword presence without full semantic understanding.
        """
        keywords_to_tools = {
            "email": "email_generate",
            "cv": "resume_generate",
            "resume": "resume_generate",
            "alignment": "alignment_evaluate",
            "match": "alignment_evaluate",
            "professor": "get_professor_context",
            "profile": "get_user_context",
        }
        
        for keyword, tool in keywords_to_tools.items():
            if keyword in message:
                return tool
        
        return None


# Singleton instance
_router = IntentRouter()


def route_request(
    message: str,
    context: OrchestrationContext,
) -> RouteDecision:
    """Route a request using the global router."""
    return _router.route(message, context)





