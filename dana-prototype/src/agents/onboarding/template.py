# src/agents/onboarding/template.py
"""Template Agent - Manages email templates for users."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from llm_client import OpenAIClient, GPT5Mini

from src.services.db import DatabaseService


@dataclass
class TemplateIssue:
    """Issue found in a template."""
    template_id: int
    field: str
    issue_type: str
    message: str
    severity: str  # error, warning, suggestion


@dataclass
class TemplateReview:
    """Review result for a template."""
    template_id: int
    is_valid: bool
    issues: List[TemplateIssue]
    score: float
    suggestions: List[str]


@dataclass 
class TemplateStatus:
    """Overall template status."""
    has_templates: bool
    template_count: int
    active_count: int
    issues_count: int
    is_ready: bool
    templates: List[Dict[str, Any]]


class TemplateAgent:
    """
    Agent for managing and reviewing email templates.
    
    Handles:
    - Template listing and status
    - Template validation and review
    - Template suggestions and improvements
    """
    
    # Required template variables
    REQUIRED_VARIABLES = [
        "professor_name",
        "professor_title",
        "user_name",
    ]
    
    # Common template variables
    COMMON_VARIABLES = [
        "professor_name",
        "professor_title",
        "professor_department",
        "professor_institution",
        "user_name",
        "user_email",
        "user_phone",
        "research_interest",
        "paper_title",
    ]
    
    def __init__(self, db: DatabaseService):
        self.db = db
        self.llm = OpenAIClient(
            GPT5Mini,
            cache_backend="pg_redis",
            cache_collection="template_agent",
        )
    
    async def get_status(self, student_id: int) -> TemplateStatus:
        """Get template status for a student."""
        templates = await self.db.client.fundingstudenttemplate.find_many(
            where={"student_id": student_id}
        )
        
        active = [t for t in templates if t.active]
        
        # Count issues across all templates
        total_issues = 0
        template_data = []
        
        for t in templates:
            issues = self._validate_template(t)
            total_issues += len([i for i in issues if i.severity == "error"])
            
            template_data.append({
                "id": int(t.id),
                "subject": t.subject,
                "has_content": bool(t.content),
                "active": t.active,
                "issue_count": len(issues),
            })
        
        return TemplateStatus(
            has_templates=len(templates) > 0,
            template_count=len(templates),
            active_count=len(active),
            issues_count=total_issues,
            is_ready=len(active) > 0 and total_issues == 0,
            templates=template_data,
        )
    
    async def get_templates(self, student_id: int) -> List[Dict[str, Any]]:
        """Get all templates for a student."""
        templates = await self.db.client.fundingstudenttemplate.find_many(
            where={"student_id": student_id}
        )
        
        return [
            {
                "id": int(t.id),
                "subject": t.subject,
                "formatted_subject": t.formatted_subject,
                "content": t.content,
                "formatted_content": t.formatted_content,
                "variables": t.variables,
                "active": t.active,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in templates
        ]
    
    async def review_templates(self, student_id: int) -> List[TemplateReview]:
        """Review all templates for a student."""
        templates = await self.db.client.fundingstudenttemplate.find_many(
            where={"student_id": student_id}
        )
        
        reviews = []
        for t in templates:
            issues = self._validate_template(t)
            
            # Calculate score based on issues
            error_count = len([i for i in issues if i.severity == "error"])
            warning_count = len([i for i in issues if i.severity == "warning"])
            
            score = 10.0
            score -= error_count * 3.0
            score -= warning_count * 1.0
            score = max(0.0, score)
            
            # Generate suggestions
            suggestions = []
            if not t.subject:
                suggestions.append("Add a compelling subject line")
            if t.content and len(t.content) < 200:
                suggestions.append("Consider expanding the email content")
            if t.content and len(t.content) > 1500:
                suggestions.append("Consider making the email more concise")
            
            reviews.append(TemplateReview(
                template_id=int(t.id),
                is_valid=error_count == 0,
                issues=issues,
                score=score,
                suggestions=suggestions,
            ))
        
        return reviews
    
    def _validate_template(self, template: Any) -> List[TemplateIssue]:
        """Validate a single template."""
        issues = []
        template_id = int(template.id)
        
        # Check subject
        if not template.subject:
            issues.append(TemplateIssue(
                template_id=template_id,
                field="subject",
                issue_type="missing",
                message="Template is missing a subject line",
                severity="error",
            ))
        elif len(template.subject) < 10:
            issues.append(TemplateIssue(
                template_id=template_id,
                field="subject",
                issue_type="too_short",
                message="Subject line is too short",
                severity="warning",
            ))
        
        # Check content
        if not template.content:
            issues.append(TemplateIssue(
                template_id=template_id,
                field="content",
                issue_type="missing",
                message="Template is missing content",
                severity="error",
            ))
        else:
            # Check for placeholder variables
            import re
            placeholders = re.findall(r'\{\{(\w+)\}\}', template.content)
            
            # Check for required variables
            for var in self.REQUIRED_VARIABLES:
                if var not in placeholders and f"{{{{{var}}}}}" not in template.content:
                    issues.append(TemplateIssue(
                        template_id=template_id,
                        field="content",
                        issue_type="missing_variable",
                        message=f"Template should include {{{{{var}}}}}",
                        severity="warning",
                    ))
            
            # Check for unknown variables
            for var in placeholders:
                if var not in self.COMMON_VARIABLES:
                    issues.append(TemplateIssue(
                        template_id=template_id,
                        field="content",
                        issue_type="unknown_variable",
                        message=f"Unknown variable: {{{{{var}}}}}",
                        severity="warning",
                    ))
        
        # Check variables field
        if template.variables:
            import json
            vars_data = json.loads(template.variables) if isinstance(template.variables, str) else template.variables
            if not isinstance(vars_data, dict):
                issues.append(TemplateIssue(
                    template_id=template_id,
                    field="variables",
                    issue_type="invalid_format",
                    message="Variables field should be a JSON object",
                    severity="warning",
                ))
        
        return issues
    
    async def suggest_improvements(
        self,
        student_id: int,
        template_id: int,
    ) -> Dict[str, Any]:
        """
        Use AI to suggest improvements for a template.
        """
        template = await self.db.client.fundingstudenttemplate.find_unique(
            where={"id": template_id}
        )
        
        if not template:
            return {"error": "Template not found"}
        
        if template.student_id != student_id:
            return {"error": "Access denied"}
        
        prompt = f"""Review this email template and suggest improvements.

Subject: {template.subject or "(missing)"}

Content:
{template.content or "(missing)"}

Provide specific suggestions to improve:
1. Subject line effectiveness
2. Email structure and flow
3. Tone appropriateness for academic outreach
4. Variable usage ({{{{variable_name}}}})
5. Call to action clarity

Format response as JSON with "suggestions" array and "improved_subject" and "improved_content" fields."""

        try:
            response = await self.llm.get_response(
                messages=[{"role": "user", "content": prompt}],
                response_format="json_object",
                temperature=0.7,
            )
            
            output = response.get("output", {})
            if isinstance(output, str):
                import json
                output = json.loads(output)
            
            return {
                "template_id": template_id,
                "current_subject": template.subject,
                "current_content": template.content,
                "suggestions": output.get("suggestions", []),
                "improved_subject": output.get("improved_subject"),
                "improved_content": output.get("improved_content"),
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def create_default_template(
        self,
        student_id: int,
        template_type: str = "initial_outreach",
    ) -> Dict[str, Any]:
        """Create a default template for a student."""
        default_templates = {
            "initial_outreach": {
                "subject": "Prospective Graduate Student - {{research_interest}}",
                "content": """Dear {{professor_title}} {{professor_name}},

I am writing to express my interest in pursuing graduate studies under your supervision at {{professor_institution}}. I am currently {{user_current_status}} and am particularly drawn to your research on {{research_interest}}.

{{body_paragraph}}

I have attached my CV for your review. I would be grateful for the opportunity to discuss potential research opportunities in your lab.

Thank you for considering my application.

Best regards,
{{user_name}}
{{user_email}}""",
                "variables": {
                    "professor_title": "Dr.",
                    "professor_name": "",
                    "professor_institution": "",
                    "research_interest": "",
                    "user_current_status": "a [degree] student at [institution]",
                    "body_paragraph": "[Describe your relevant background and interest]",
                    "user_name": "",
                    "user_email": "",
                },
            },
            "follow_up": {
                "subject": "Re: Prospective Graduate Student - Following Up",
                "content": """Dear {{professor_title}} {{professor_name}},

I hope this email finds you well. I am following up on my previous email regarding graduate research opportunities in your lab.

I remain very interested in your work on {{research_interest}} and would welcome the chance to discuss how my background might contribute to your research.

Please let me know if you have any availability for a brief conversation.

Best regards,
{{user_name}}""",
                "variables": {
                    "professor_title": "Dr.",
                    "professor_name": "",
                    "research_interest": "",
                    "user_name": "",
                },
            },
        }
        
        template_data = default_templates.get(template_type)
        if not template_data:
            return {"error": f"Unknown template type: {template_type}"}
        
        import json
        
        # Create template
        template = await self.db.client.fundingstudenttemplate.create(
            data={
                "student_id": student_id,
                "funding_template_id": 1,  # Default template ID
                "subject": template_data["subject"],
                "content": template_data["content"],
                "variables": json.dumps(template_data["variables"]),
                "active": True,
            }
        )
        
        return {
            "success": True,
            "template_id": int(template.id),
            "template_type": template_type,
        }
    
    async def get_conversation_response(
        self,
        student_id: int,
        user_message: str,
    ) -> Dict[str, Any]:
        """Handle template-related conversation."""
        status = await self.get_status(student_id)
        message_lower = user_message.lower()
        
        if not status.has_templates:
            if any(word in message_lower for word in ["create", "yes", "make", "new"]):
                result = await self.create_default_template(student_id)
                if result.get("success"):
                    return {
                        "message": "I've created a default email template for you. Would you like me to customize it based on your background?",
                        "action": "template_created",
                        "template_id": result["template_id"],
                    }
            else:
                return {
                    "message": "You don't have any email templates yet. Would you like me to create a default template for professor outreach?",
                    "action": "prompt_create",
                }
        
        if status.issues_count > 0:
            return {
                "message": f"I found {status.issues_count} issues with your templates. Would you like me to review them and suggest improvements?",
                "action": "prompt_review",
                "issues_count": status.issues_count,
            }
        
        if status.is_ready:
            return {
                "message": f"Your templates are ready! You have {status.active_count} active template(s).",
                "action": "ready",
                "template_count": status.template_count,
            }
        
        return {
            "message": "How can I help you with your email templates?",
            "action": "default",
        }





