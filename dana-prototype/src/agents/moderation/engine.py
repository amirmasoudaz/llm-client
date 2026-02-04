# src/agents/moderation/engine.py
"""Moderation Agent - Content moderation and safety checks."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from llm_client import OpenAIClient, GPT5Mini

from src.services.db import DatabaseService


class ModerationCategory(str, Enum):
    """Moderation categories."""
    CLEAN = "clean"
    SPAM = "spam"
    INAPPROPRIATE = "inappropriate"
    HARASSMENT = "harassment"
    ACADEMIC_MISCONDUCT = "academic_misconduct"
    IMPERSONATION = "impersonation"
    POLICY_VIOLATION = "policy_violation"


class ModerationAction(str, Enum):
    """Actions to take based on moderation."""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"
    FLAG = "flag"


@dataclass
class ModerationResult:
    """Result of content moderation."""
    is_safe: bool
    action: ModerationAction
    categories: List[ModerationCategory]
    confidence: float
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ModerationAgent:
    """
    Agent for moderating content for safety and policy compliance.
    
    Checks:
    - Inappropriate or harmful content
    - Academic integrity violations
    - Spam detection
    - Policy compliance
    """
    
    # Blocked patterns (simple keyword matching)
    BLOCKED_PATTERNS = [
        # Academic misconduct
        "write my thesis for me",
        "do my homework",
        "plagiarize",
        "copy someone else",
        "fake credentials",
        "buy a degree",
        
        # Harassment
        "threaten",
        "harass",
        "stalk",
        
        # Spam indicators
        "guaranteed admission",
        "buy your way in",
        "pay for acceptance",
    ]
    
    # Warning patterns
    WARNING_PATTERNS = [
        "urgent reply needed",
        "time sensitive",
        "last chance",
        "don't miss out",
    ]
    
    def __init__(self, db: Optional[DatabaseService] = None):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="moderation",
        )
    
    async def check_content(
        self,
        content: str,
        content_type: str = "message",
        context: Optional[Dict[str, Any]] = None,
    ) -> ModerationResult:
        """
        Check content for policy violations.
        
        Args:
            content: The content to check
            content_type: Type of content (message, email, cv, etc.)
            context: Additional context about the content
        """
        categories = []
        confidence = 1.0
        
        # Quick pattern matching
        content_lower = content.lower()
        
        # Check for blocked patterns
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in content_lower:
                # Determine category
                if any(p in pattern for p in ["thesis", "homework", "plagiarize", "credentials", "degree"]):
                    categories.append(ModerationCategory.ACADEMIC_MISCONDUCT)
                elif any(p in pattern for p in ["threaten", "harass", "stalk"]):
                    categories.append(ModerationCategory.HARASSMENT)
                elif any(p in pattern for p in ["guaranteed", "buy"]):
                    categories.append(ModerationCategory.SPAM)
                else:
                    categories.append(ModerationCategory.POLICY_VIOLATION)
        
        # If blocked patterns found, return immediately
        if categories:
            return ModerationResult(
                is_safe=False,
                action=ModerationAction.BLOCK,
                categories=list(set(categories)),
                confidence=0.95,
                message="This content contains policy violations and cannot be processed.",
            )
        
        # Check for warning patterns
        has_warning = any(p in content_lower for p in self.WARNING_PATTERNS)
        
        # Use LLM for more nuanced checks if content is substantial
        if len(content) > 100:
            llm_result = await self._llm_moderation(content, content_type)
            
            if llm_result["categories"]:
                categories.extend(llm_result["categories"])
                confidence = llm_result["confidence"]
            
            if llm_result["action"] == "block":
                return ModerationResult(
                    is_safe=False,
                    action=ModerationAction.BLOCK,
                    categories=list(set(categories)) or [ModerationCategory.POLICY_VIOLATION],
                    confidence=confidence,
                    message=llm_result.get("message", "This content may violate our policies."),
                    details=llm_result.get("details"),
                )
        
        # Determine action
        if categories:
            return ModerationResult(
                is_safe=False,
                action=ModerationAction.FLAG,
                categories=list(set(categories)),
                confidence=confidence,
                message="This content has been flagged for review.",
            )
        
        if has_warning:
            return ModerationResult(
                is_safe=True,
                action=ModerationAction.WARN,
                categories=[],
                confidence=0.8,
                message="This content contains some patterns that might reduce effectiveness.",
            )
        
        return ModerationResult(
            is_safe=True,
            action=ModerationAction.ALLOW,
            categories=[ModerationCategory.CLEAN],
            confidence=1.0,
        )
    
    async def _llm_moderation(
        self,
        content: str,
        content_type: str,
    ) -> Dict[str, Any]:
        """Use LLM for nuanced content moderation."""
        prompt = f"""Analyze this {content_type} for policy violations in an academic outreach context.

Content:
{content[:2000]}

Check for:
1. Academic misconduct (asking to write papers, fake credentials, etc.)
2. Harassment or threatening language
3. Spam patterns (urgency, too-good-to-be-true promises)
4. Impersonation attempts
5. Professional inappropriateness

Respond with JSON containing:
- "is_safe": boolean
- "action": "allow", "warn", "flag", or "block"
- "categories": list of detected issues
- "confidence": 0.0 to 1.0
- "message": brief explanation if not safe
- "details": any additional context"""

        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                temperature=0,
            )
            
            output = response.get("output", {})
            if isinstance(output, str):
                import json
                output = json.loads(output)
            
            categories = []
            for cat in output.get("categories", []):
                try:
                    categories.append(ModerationCategory(cat))
                except ValueError:
                    pass
            
            return {
                "is_safe": output.get("is_safe", True),
                "action": output.get("action", "allow"),
                "categories": categories,
                "confidence": output.get("confidence", 0.5),
                "message": output.get("message"),
                "details": output.get("details"),
            }
            
        except Exception:
            # If LLM check fails, default to allowing with lower confidence
            return {
                "is_safe": True,
                "action": "allow",
                "categories": [],
                "confidence": 0.5,
            }
    
    async def check_email(
        self,
        subject: str,
        body: str,
        sender_name: str,
        recipient_name: str,
    ) -> ModerationResult:
        """
        Check an email for policy compliance.
        
        Additional checks specific to email content.
        """
        combined = f"Subject: {subject}\n\n{body}"
        
        # Basic moderation
        result = await self.check_content(combined, "email")
        
        if not result.is_safe:
            return result
        
        # Email-specific checks
        issues = []
        
        # Check for impersonation
        if sender_name.lower() != recipient_name.lower():
            # Check if sender claims to be recipient
            if recipient_name.lower() in body.lower() and "i am" in body.lower():
                issues.append("Potential impersonation detected")
        
        # Check for unrealistic claims
        unrealistic_phrases = [
            "guaranteed acceptance",
            "100% success rate",
            "instant admission",
        ]
        body_lower = body.lower()
        for phrase in unrealistic_phrases:
            if phrase in body_lower:
                issues.append(f"Unrealistic claim: {phrase}")
        
        if issues:
            return ModerationResult(
                is_safe=True,
                action=ModerationAction.WARN,
                categories=[ModerationCategory.CLEAN],
                confidence=0.7,
                message="Some content may reduce email effectiveness.",
                details={"issues": issues},
            )
        
        return result
    
    async def check_cv(
        self,
        cv_content: Dict[str, Any],
    ) -> ModerationResult:
        """
        Check CV content for policy compliance.
        
        Focuses on accuracy and authenticity.
        """
        # Convert to string for analysis
        import json
        content = json.dumps(cv_content, default=str)
        
        # Basic moderation
        result = await self.check_content(content, "cv")
        
        if not result.is_safe:
            return result
        
        # CV-specific checks
        issues = []
        
        # Check for suspicious patterns
        education = cv_content.get("education", [])
        for edu in education:
            # Check for impossible graduation dates
            grad_year = edu.get("graduation_year")
            if grad_year and (grad_year < 1950 or grad_year > datetime.now().year + 10):
                issues.append(f"Suspicious graduation year: {grad_year}")
            
            # Check for implausible GPAs
            gpa = edu.get("gpa")
            if gpa and (gpa > 4.0 or gpa < 0):
                issues.append(f"Implausible GPA: {gpa}")
        
        if issues:
            return ModerationResult(
                is_safe=True,
                action=ModerationAction.FLAG,
                categories=[ModerationCategory.CLEAN],
                confidence=0.6,
                message="CV contains some content that may need verification.",
                details={"issues": issues},
            )
        
        return result





