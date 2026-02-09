# src/agents/onboarding/gmail.py
"""Gmail Onboarding Agent - Guides users through Gmail OAuth setup."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass

from src.services.db import DatabaseService


class OnboardingStep(str, Enum):
    """Gmail onboarding steps."""
    NOT_STARTED = "not_started"
    INSTRUCTIONS = "instructions"
    CLIENT_SECRET = "client_secret"
    AUTH_REQUIRED = "auth_required"
    AUTHORIZING = "authorizing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class GmailStatus:
    """Gmail connection status."""
    step: OnboardingStep
    message: str
    next_action: Optional[str] = None
    auth_url: Optional[str] = None
    is_connected: bool = False
    last_refresh: Optional[datetime] = None
    error: Optional[str] = None


class GmailOnboardingAgent:
    """
    Agent for guiding users through Gmail OAuth setup.
    
    Handles:
    - Status checking
    - Client secret upload and validation
    - OAuth flow initiation
    - Token storage and refresh
    """
    
    INSTRUCTIONS = """To connect your Gmail account for sending emails to professors, you need to:

1. **Create a Google Cloud Project** (if you don't have one)
   - Go to https://console.cloud.google.com
   - Create a new project or select an existing one

2. **Enable the Gmail API**
   - In your project, go to "APIs & Services" > "Library"
   - Search for "Gmail API" and enable it

3. **Create OAuth Credentials**
   - Go to "APIs & Services" > "Credentials"
   - Click "Create Credentials" > "OAuth client ID"
   - Select "Desktop app" as the application type
   - Download the JSON file

4. **Upload the Client Secret**
   - Upload the downloaded JSON file here
   - We'll guide you through authorizing your account

5. **Authorize Access**
   - Click the authorization link we provide
   - Sign in with your Gmail account
   - Grant permission to send emails on your behalf"""
    
    def __init__(self, db: DatabaseService):
        self.db = db
    
    async def get_status(self, student_id: int) -> GmailStatus:
        """Get current Gmail onboarding status for a student."""
        creds = await self.db.client.fundingcredential.find_unique(
            where={"user_id": student_id}
        )
        
        if not creds:
            return GmailStatus(
                step=OnboardingStep.NOT_STARTED,
                message="Gmail not connected. Would you like to set it up?",
                next_action="start_setup",
            )
        
        # Check if we have valid tokens
        token_blob = creds.token_blob
        if not token_blob:
            return GmailStatus(
                step=OnboardingStep.CLIENT_SECRET,
                message="Please upload your Google OAuth client secret file.",
                next_action="upload_client_secret",
            )
        
        # Parse token blob
        import json
        tokens = json.loads(token_blob) if isinstance(token_blob, str) else token_blob
        
        if not tokens.get("access_token"):
            # Have client secret but not authorized
            auth_url = tokens.get("auth_url", "")
            return GmailStatus(
                step=OnboardingStep.AUTH_REQUIRED,
                message="Please authorize access to your Gmail account.",
                next_action="authorize",
                auth_url=auth_url,
            )
        
        # Check if token needs refresh
        # (Simplified - in production would actually check expiry)
        
        return GmailStatus(
            step=OnboardingStep.COMPLETE,
            message="Gmail is connected and ready to send emails!",
            is_connected=True,
            last_refresh=creds.last_refresh,
        )
    
    async def get_instructions(self) -> str:
        """Get Gmail setup instructions."""
        return self.INSTRUCTIONS
    
    async def process_client_secret(
        self,
        student_id: int,
        file_content: bytes,
    ) -> GmailStatus:
        """
        Process uploaded client secret file.
        
        Validates the JSON structure and stores it.
        """
        import json
        
        try:
            # Parse JSON
            secret_data = json.loads(file_content.decode("utf-8"))
            
            # Validate structure
            if "installed" not in secret_data and "web" not in secret_data:
                return GmailStatus(
                    step=OnboardingStep.ERROR,
                    message="Invalid client secret file. Please download a valid OAuth credentials JSON from Google Cloud Console.",
                    error="Missing 'installed' or 'web' key in credentials",
                )
            
            # Extract client config
            config = secret_data.get("installed") or secret_data.get("web")
            
            required_fields = ["client_id", "client_secret", "auth_uri", "token_uri"]
            missing = [f for f in required_fields if f not in config]
            if missing:
                return GmailStatus(
                    step=OnboardingStep.ERROR,
                    message=f"Invalid client secret file. Missing required fields: {', '.join(missing)}",
                    error=f"Missing fields: {missing}",
                )
            
            # Generate authorization URL
            from urllib.parse import urlencode
            
            auth_params = {
                "client_id": config["client_id"],
                "redirect_uri": config.get("redirect_uris", ["urn:ietf:wg:oauth:2.0:oob"])[0],
                "response_type": "code",
                "scope": "https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/gmail.readonly",
                "access_type": "offline",
                "prompt": "consent",
            }
            auth_url = f"{config['auth_uri']}?{urlencode(auth_params)}"
            
            # Store client secret (without tokens yet)
            token_blob = json.dumps({
                "client_id": config["client_id"],
                "client_secret": config["client_secret"],
                "auth_uri": config["auth_uri"],
                "token_uri": config["token_uri"],
                "redirect_uri": auth_params["redirect_uri"],
                "auth_url": auth_url,
            })
            
            # Upsert credentials record
            await self.db.client.fundingcredential.upsert(
                where={"user_id": student_id},
                create={
                    "user_id": student_id,
                    "token_blob": token_blob,
                },
                update={
                    "token_blob": token_blob,
                },
            )
            
            return GmailStatus(
                step=OnboardingStep.AUTH_REQUIRED,
                message="Client secret uploaded successfully! Now please authorize access to your Gmail.",
                next_action="authorize",
                auth_url=auth_url,
            )
            
        except json.JSONDecodeError:
            return GmailStatus(
                step=OnboardingStep.ERROR,
                message="Invalid file format. Please upload a valid JSON file.",
                error="JSON parse error",
            )
        except Exception as e:
            return GmailStatus(
                step=OnboardingStep.ERROR,
                message=f"Error processing client secret: {str(e)}",
                error=str(e),
            )
    
    async def process_auth_code(
        self,
        student_id: int,
        auth_code: str,
    ) -> GmailStatus:
        """
        Process OAuth authorization code.
        
        Exchanges code for access/refresh tokens.
        """
        import json
        import httpx
        
        # Get stored client config
        creds = await self.db.client.fundingcredential.find_unique(
            where={"user_id": student_id}
        )
        
        if not creds or not creds.token_blob:
            return GmailStatus(
                step=OnboardingStep.ERROR,
                message="No client secret found. Please upload client secret first.",
                error="Missing client secret",
            )
        
        config = json.loads(creds.token_blob) if isinstance(creds.token_blob, str) else creds.token_blob
        
        try:
            # Exchange code for tokens
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    config["token_uri"],
                    data={
                        "client_id": config["client_id"],
                        "client_secret": config["client_secret"],
                        "code": auth_code,
                        "grant_type": "authorization_code",
                        "redirect_uri": config["redirect_uri"],
                    },
                )
                
                if response.status_code != 200:
                    error_data = response.json()
                    return GmailStatus(
                        step=OnboardingStep.ERROR,
                        message=f"Failed to authorize: {error_data.get('error_description', 'Unknown error')}",
                        error=str(error_data),
                    )
                
                tokens = response.json()
            
            # Store complete token blob
            config["access_token"] = tokens["access_token"]
            config["refresh_token"] = tokens.get("refresh_token")
            config["expires_in"] = tokens.get("expires_in")
            
            await self.db.client.fundingcredential.update(
                where={"user_id": student_id},
                data={
                    "token_blob": json.dumps(config),
                    "last_refresh": datetime.utcnow(),
                },
            )
            
            return GmailStatus(
                step=OnboardingStep.COMPLETE,
                message="Gmail authorized successfully! You can now send emails to professors.",
                is_connected=True,
            )
            
        except Exception as e:
            return GmailStatus(
                step=OnboardingStep.ERROR,
                message=f"Authorization failed: {str(e)}",
                error=str(e),
            )
    
    async def disconnect(self, student_id: int) -> bool:
        """Disconnect Gmail (revoke and delete credentials)."""
        try:
            await self.db.client.fundingcredential.delete(
                where={"user_id": student_id}
            )
            return True
        except Exception:
            return False
    
    async def get_conversation_response(
        self,
        student_id: int,
        user_message: str,
    ) -> Dict[str, Any]:
        """
        Handle Gmail onboarding conversation.
        
        Returns a response based on current status and user message.
        """
        status = await self.get_status(student_id)
        
        # Keywords that might trigger different actions
        message_lower = user_message.lower()
        
        if status.step == OnboardingStep.NOT_STARTED:
            if any(word in message_lower for word in ["yes", "setup", "connect", "start"]):
                return {
                    "message": self.INSTRUCTIONS,
                    "action": "show_instructions",
                    "next_step": "upload_client_secret",
                }
            else:
                return {
                    "message": "Would you like me to help you connect your Gmail account? This allows Dana to send emails on your behalf.",
                    "action": "prompt_setup",
                }
        
        elif status.step == OnboardingStep.CLIENT_SECRET:
            return {
                "message": "Please upload your Google OAuth client secret JSON file. You can drag and drop the file or click to select it.",
                "action": "request_file_upload",
                "file_type": "application/json",
            }
        
        elif status.step == OnboardingStep.AUTH_REQUIRED:
            return {
                "message": f"Please click the link below to authorize Gmail access:\n\n{status.auth_url}\n\nAfter authorizing, paste the authorization code here.",
                "action": "show_auth_link",
                "auth_url": status.auth_url,
            }
        
        elif status.step == OnboardingStep.COMPLETE:
            return {
                "message": "Your Gmail is already connected! You can send emails to professors.",
                "action": "confirm_connected",
            }
        
        elif status.step == OnboardingStep.ERROR:
            return {
                "message": f"There was an error: {status.error}\n\nWould you like to try again?",
                "action": "show_error",
                "error": status.error,
            }
        
        return {
            "message": "I can help you with Gmail setup. What would you like to do?",
            "action": "default",
        }





